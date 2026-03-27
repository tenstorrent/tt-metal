# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_VALIDATION_CASES_PATH = Path(__file__).with_name("validation_cases.json")
DEFAULT_PERFORMANCE_CASES_PATH = Path(__file__).with_name("performance_cases.json")
DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH = Path(__file__).with_name("reference_audio_manifest.json")


@dataclass
class AudioContent:
    audio_url: str
    raw_audio: str | None = None
    offset: float | None = None
    duration: float | None = None
    row_id: int | None = None
    type: str = "audio"


@dataclass
class TextContent:
    text: str
    type: str = "text"


@dataclass
class Message:
    role: str
    content: str | AudioContent | TextContent | list[str | AudioContent | TextContent]
    recipient: str | None = None


@dataclass
class ChatMLSample:
    messages: list[Message]
    start_index: int | None = None
    misc: dict[str, Any] | None = None
    speaker: str | None = None


def load_cases(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["cases"]
    return payload


def resolve_reference_audio_assets_root(
    reference_audio_manifest_path: str | Path = DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    assets_root: str | Path | None = None,
) -> Path:
    manifest = json.loads(Path(reference_audio_manifest_path).resolve().read_text(encoding="utf-8"))
    configured_root = assets_root or os.environ.get("HIGGS_AUDIO_REFERENCE_ASSETS_DIR")
    if configured_root is None:
        configured_root = Path.home() / ".cache" / "tt-metal" / "higgs_audio_v2_reference_audio"
    return (Path(configured_root).expanduser() / manifest["asset_version"]).resolve()


def load_reference_audio_manifest(
    reference_audio_manifest_path: str | Path = DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    assets_root: str | Path | None = None,
    require_local: bool = False,
) -> dict[str, dict]:
    manifest_path = Path(reference_audio_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_assets_root = resolve_reference_audio_assets_root(manifest_path, assets_root)
    clips = {}
    for clip in manifest["clips"]:
        resolved_audio_path = (resolved_assets_root / clip["audio_path"]).resolve()
        resolved_clip = dict(clip)
        resolved_clip["audio_url"] = str(resolved_audio_path)
        resolved_clip["assets_root"] = str(resolved_assets_root)
        clips[clip["id"]] = resolved_clip
        if require_local and not resolved_audio_path.exists():
            fetch_script = Path(__file__).with_name("fetch_reference_audio_assets.py")
            raise FileNotFoundError(
                f"Missing reference audio asset `{clip['id']}` at `{resolved_audio_path}`. "
                f"Run `python {fetch_script} --reference-audio-manifest {manifest_path}` first."
            )
    return clips


def build_demo_sample(
    transcript: str,
    system_prompt: str,
    ref_audio: str | None = None,
    ref_transcript: str | None = None,
) -> ChatMLSample:
    messages: list[Message] = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    if ref_audio:
        if not ref_transcript:
            raise ValueError("`--ref-transcript` is required when `--ref-audio` is provided.")
        messages.append(Message(role="user", content=ref_transcript))
        messages.append(Message(role="assistant", content=AudioContent(audio_url=ref_audio)))
    messages.append(Message(role="user", content=transcript))
    return ChatMLSample(messages=messages)


def build_case_sample(
    case: dict,
    base_dir: Path | None = None,
    reference_audio_index: dict[str, dict] | None = None,
) -> ChatMLSample:
    if "messages" not in case:
        raise ValueError("Case manifest entries must define `messages`.")
    return ChatMLSample(
        messages=_build_messages(case["messages"], base_dir=base_dir, reference_audio_index=reference_audio_index)
    )


def load_chatml_sample_from_json(
    messages_json_path: str | Path,
    reference_audio_manifest_path: str | Path = DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    reference_audio_assets_root: str | Path | None = None,
) -> ChatMLSample:
    messages_json_path = Path(messages_json_path).resolve()
    payload = json.loads(messages_json_path.read_text(encoding="utf-8"))
    message_dicts = payload["messages"] if isinstance(payload, dict) else payload
    reference_audio_index = None
    if _uses_audio_ref(payload):
        reference_audio_index = load_reference_audio_manifest(
            reference_audio_manifest_path=reference_audio_manifest_path,
            assets_root=reference_audio_assets_root,
            require_local=True,
        )
    return ChatMLSample(
        messages=_build_messages(
            message_dicts,
            base_dir=messages_json_path.parent,
            reference_audio_index=reference_audio_index,
        )
    )


def _uses_audio_ref(payload) -> bool:
    if isinstance(payload, dict):
        if "audio_ref" in payload:
            return True
        return any(_uses_audio_ref(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_uses_audio_ref(value) for value in payload)
    return False


def _build_messages(
    message_dicts: list[dict],
    base_dir: Path | None = None,
    reference_audio_index: dict[str, dict] | None = None,
) -> list[Message]:
    return [
        Message(
            role=message_dict["role"],
            content=_parse_message_content(
                message_dict["content"],
                base_dir=base_dir,
                reference_audio_index=reference_audio_index,
            ),
            recipient=message_dict.get("recipient"),
        )
        for message_dict in message_dicts
    ]


def _resolve_audio_url(audio_url: str, base_dir: Path | None) -> str:
    if not audio_url or audio_url == "placeholder" or base_dir is None:
        return audio_url
    audio_path = Path(audio_url)
    if audio_path.is_absolute():
        return str(audio_path)
    return str((base_dir / audio_path).resolve())


def _resolve_audio_ref(audio_ref: str, reference_audio_index: dict[str, dict] | None) -> str:
    if reference_audio_index is None:
        raise ValueError(f"Audio reference `{audio_ref}` requires a loaded reference-audio manifest.")
    if audio_ref not in reference_audio_index:
        raise KeyError(f"Unknown reference audio `{audio_ref}`.")
    return reference_audio_index[audio_ref]["audio_url"]


def _parse_message_content(
    content_spec,
    base_dir: Path | None = None,
    reference_audio_index: dict[str, dict] | None = None,
):
    if isinstance(content_spec, str):
        return content_spec
    if isinstance(content_spec, list):
        return [
            _parse_message_content(element, base_dir=base_dir, reference_audio_index=reference_audio_index)
            for element in content_spec
        ]
    if not isinstance(content_spec, dict):
        raise TypeError(f"Unsupported message content type: {type(content_spec)!r}")

    content_type = content_spec.get("type", "text")
    if content_type == "text":
        return content_spec["text"]
    if content_type == "audio":
        audio_url = content_spec.get("audio_url", "")
        if content_spec.get("audio_ref"):
            audio_url = _resolve_audio_ref(content_spec["audio_ref"], reference_audio_index)
        return AudioContent(
            audio_url=_resolve_audio_url(audio_url, base_dir),
            raw_audio=content_spec.get("raw_audio"),
            offset=content_spec.get("offset"),
            duration=content_spec.get("duration"),
            row_id=content_spec.get("row_id"),
        )
    raise ValueError(f"Unsupported message content type `{content_type}`.")
