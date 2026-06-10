# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Download VibeVoice demo text and voice assets from the upstream GitHub repo."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Union

from loguru import logger

from models.experimental.vibevoice.common.config import (
    DEFAULT_TXT_PATH,
    DEFAULT_VOICE_PATH,
    GITHUB_DEMO_BRANCH,
    GITHUB_DEMO_REPO,
    RESOURCES_DIR,
)

PathLike = Union[str, Path]

_GITHUB_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
_USER_AGENT = "tt-metal-vibevoice-resource-sync"


def _github_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
    }


def _list_github_dir(repo_path: str, *, repo: str = GITHUB_DEMO_REPO, ref: str = GITHUB_DEMO_BRANCH) -> list[dict]:
    url = _GITHUB_API.format(repo=repo, path=repo_path, ref=ref)
    request = urllib.request.Request(url, headers=_github_headers())
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected directory listing for {repo_path}, got {type(payload)}")
    return payload


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    dest.write_bytes(data)


def _sync_github_tree(
    repo_path: str,
    local_dir: Path,
    *,
    repo: str = GITHUB_DEMO_REPO,
    ref: str = GITHUB_DEMO_BRANCH,
    extensions: Optional[Iterable[str]] = None,
    recursive: bool = False,
) -> list[Path]:
    """Download files under *repo_path* into *local_dir*, preserving basenames."""
    downloaded: list[Path] = []
    for entry in _list_github_dir(repo_path, repo=repo, ref=ref):
        name = entry["name"]
        entry_type = entry["type"]
        if entry_type == "dir":
            if not recursive:
                continue
            sub_repo = f"{repo_path.rstrip('/')}/{name}"
            sub_local = local_dir / name
            downloaded.extend(
                _sync_github_tree(
                    sub_repo,
                    sub_local,
                    repo=repo,
                    ref=ref,
                    extensions=extensions,
                    recursive=recursive,
                )
            )
            continue

        if extensions is not None and not any(name.endswith(ext) for ext in extensions):
            continue

        download_url = entry.get("download_url")
        if not download_url:
            continue

        dest = local_dir / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue

        logger.info(f"Downloading {repo}/{repo_path}/{name} -> {dest}")
        _download_file(download_url, dest)
        downloaded.append(dest)

    return downloaded


def is_demo_resources_ready(
    *,
    resources_dir: Optional[PathLike] = None,
    require_default_text: bool = True,
    require_default_voice: bool = True,
) -> bool:
    """Return True when the minimum demo text/voice assets are present locally."""
    root = Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    voices_dir = root / "voices"
    text_dir = root / "text"

    if require_default_voice:
        default_voice = voices_dir / DEFAULT_VOICE_PATH.name
        if not default_voice.is_file():
            return False

    if require_default_text:
        default_text = text_dir / DEFAULT_TXT_PATH.name
        if not default_text.is_file():
            return False

    return voices_dir.is_dir() and text_dir.is_dir()


def download_demo_resources(
    resources_dir: Optional[PathLike] = None,
    *,
    repo: str = GITHUB_DEMO_REPO,
    ref: str = GITHUB_DEMO_BRANCH,
    include_streaming_voices: bool = False,
) -> Path:
    """Download demo voices and text examples from vibevoice-community/VibeVoice."""
    root = Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    voices_dir = root / "voices"
    text_dir = root / "text"
    voices_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Syncing demo text examples from {repo}@{ref} -> {text_dir}")
    _sync_github_tree(
        "demo/text_examples",
        text_dir,
        repo=repo,
        ref=ref,
        extensions=(".txt",),
        recursive=False,
    )

    logger.info(f"Syncing demo voice WAVs from {repo}@{ref} -> {voices_dir}")
    _sync_github_tree(
        "demo/voices",
        voices_dir,
        repo=repo,
        ref=ref,
        extensions=(".wav",),
        recursive=False,
    )

    if include_streaming_voices:
        logger.info(f"Syncing streaming_model voice presets -> {voices_dir / 'streaming_model'}")
        _sync_github_tree(
            "demo/voices/streaming_model",
            voices_dir / "streaming_model",
            repo=repo,
            ref=ref,
            extensions=(".pt",),
            recursive=False,
        )

    return root.resolve()


def ensure_demo_resources(
    resources_dir: Optional[PathLike] = None,
    *,
    download: bool = True,
    include_streaming_voices: bool = False,
    repo: str = GITHUB_DEMO_REPO,
    ref: str = GITHUB_DEMO_BRANCH,
) -> Path:
    """Return local resources dir, downloading demo assets from GitHub when missing."""
    root = Path(resources_dir) if resources_dir is not None else RESOURCES_DIR

    if is_demo_resources_ready(resources_dir=root):
        return root.resolve()

    if not download:
        raise FileNotFoundError(
            f"VibeVoice demo resources not found under {root}. "
            "Run ensure_demo_resources(download=True) or download manually from "
            f"https://github.com/{repo}/tree/{ref}/demo"
        )

    try:
        download_demo_resources(
            root,
            repo=repo,
            ref=ref,
            include_streaming_voices=include_streaming_voices,
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to download VibeVoice demo resources from https://github.com/{repo}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to download VibeVoice demo resources: {exc}") from exc

    if not is_demo_resources_ready(resources_dir=root):
        raise RuntimeError(
            f"Download completed but demo resources are still incomplete under {root}. "
            f"Expected {DEFAULT_TXT_PATH.name} and {DEFAULT_VOICE_PATH.name}."
        )

    return root.resolve()


_SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def normalize_script(script: str) -> str:
    """Convert plain text or ``Speaker N:`` lines into processor-ready script format.

    Upstream demo files such as ``1p_vibevoice.txt`` are plain paragraphs without
    speaker prefixes; ``VibeVoiceProcessor._parse_script`` requires ``Speaker N:`` lines.
    """
    script = script.strip().replace("\u2019", "'")
    if not script:
        raise ValueError("Empty script")

    out: list[str] = []
    current_speaker = 1
    for line in script.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _SPEAKER_LINE.match(line)
        if match:
            speaker_id = int(match.group(1))
            text = match.group(2).strip()
            if text:
                out.append(f"Speaker {speaker_id}: {text}")
        else:
            out.append(f"Speaker {current_speaker}: {line}")

    if not out:
        raise ValueError("No valid content in script")
    return "\n".join(out)


def load_script(text_path: Optional[PathLike] = None) -> str:
    """Load and normalize a speaker script (default: upstream 1p_vibevoice.txt)."""
    path = Path(text_path) if text_path is not None else DEFAULT_TXT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Script not found: {path}. Call ensure_demo_resources() first.")
    with open(path, encoding="utf-8") as handle:
        return normalize_script(handle.read())
