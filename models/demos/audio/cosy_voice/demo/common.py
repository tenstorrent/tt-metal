from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from huggingface_hub import snapshot_download

SUPPORTED_MODES = ("sft", "zero_shot", "cross_lingual", "instruct")
MODE_TO_MODEL_SUBDIR = {
    "sft": "CosyVoice-300M-SFT",
    "zero_shot": "CosyVoice-300M",
    "cross_lingual": "CosyVoice-300M",
    "instruct": "CosyVoice-300M-Instruct",
}
MODE_TO_MODEL_ID = {
    "sft": "FunAudioLLM/CosyVoice-300M-SFT",
    "zero_shot": "FunAudioLLM/CosyVoice-300M",
    "cross_lingual": "FunAudioLLM/CosyVoice-300M",
    "instruct": "FunAudioLLM/CosyVoice-300M-Instruct",
}
DEFAULT_REFERENCE_REPO_ENV = "COSYVOICE_REFERENCE_REPO"
DEFAULT_MODEL_ROOT_ENV = "COSYVOICE_MODEL_ROOT"
DEFAULT_DOWNLOAD_ROOT_ENV = "COSYVOICE_DOWNLOAD_ROOT"
DEFAULT_REFERENCE_REPO_CANDIDATES = (
    "/vol/stor/reference_repos/CosyVoice",
    "/vol/stor/CosyVoice",
)


@dataclass(frozen=True)
class CosyVoiceCase:
    name: str
    mode: str
    text: str
    language: str | None = None
    speaker_id: str | None = None
    prompt_text: str | None = None
    prompt_audio: str | None = None
    instruction: str | None = None
    model: str | None = None
    target_tokens_per_second: float | None = None
    target_rtf: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CosyVoiceCase":
        case = cls(**raw)
        case.validate()
        return case

    def validate(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported CosyVoice mode: {self.mode}")
        if not self.text:
            raise ValueError(f"Case {self.name} must provide non-empty text")
        if self.mode in {"sft", "instruct"} and not self.speaker_id:
            raise ValueError(f"Case {self.name} requires speaker_id for mode={self.mode}")
        if self.mode == "zero_shot" and (not self.prompt_text or not self.prompt_audio):
            raise ValueError(f"Case {self.name} requires prompt_text and prompt_audio for zero_shot mode")
        if self.mode == "cross_lingual" and not self.prompt_audio:
            raise ValueError(f"Case {self.name} requires prompt_audio for cross_lingual mode")
        if self.mode == "instruct" and not self.instruction:
            raise ValueError(f"Case {self.name} requires instruction for instruct mode")


def load_cases(manifest_path: str | os.PathLike[str]) -> list[CosyVoiceCase]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return [CosyVoiceCase.from_dict(item) for item in json.load(handle)]


def resolve_reference_repo(reference_repo: str | None = None) -> Path:
    candidates: list[str] = []
    if reference_repo:
        candidates.append(reference_repo)
    env_repo = os.environ.get(DEFAULT_REFERENCE_REPO_ENV)
    if env_repo:
        candidates.append(env_repo)
    candidates.extend(DEFAULT_REFERENCE_REPO_CANDIDATES)
    for candidate in candidates:
        repo_path = Path(candidate).expanduser().resolve()
        if (repo_path / "cosyvoice").exists():
            return repo_path
    raise FileNotFoundError(
        "Could not resolve the CosyVoice reference repo. Pass --reference-repo or set COSYVOICE_REFERENCE_REPO."
    )


def configure_reference_imports(reference_repo: str | os.PathLike[str]) -> None:
    repo = str(Path(reference_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    matcha = Path(repo) / "third_party" / "Matcha-TTS"
    if matcha.exists():
        matcha_str = str(matcha.resolve())
        if matcha_str not in sys.path:
            sys.path.insert(0, matcha_str)


def resolve_model_dir(mode: str, model_root: str | None = None, explicit_model: str | None = None) -> Path:
    model_name = explicit_model or MODE_TO_MODEL_SUBDIR[mode]
    model_root = model_root or os.environ.get(DEFAULT_MODEL_ROOT_ENV)
    if model_root:
        root = Path(model_root).expanduser().resolve()
        candidate = root / model_name
        if candidate.exists():
            return candidate
        if Path(model_name).exists():
            return Path(model_name).resolve()
    if explicit_model and Path(explicit_model).exists():
        return Path(explicit_model).resolve()

    download_root = os.environ.get(DEFAULT_DOWNLOAD_ROOT_ENV)
    if download_root is None:
        download_root = str(Path.home() / ".cache" / "cosy_voice")
    local_dir = Path(download_root).expanduser().resolve() / MODE_TO_MODEL_SUBDIR[mode]
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(explicit_model or MODE_TO_MODEL_ID[mode], local_dir=str(local_dir))).resolve()


def resolve_prompt_audio(reference_repo: str | os.PathLike[str], prompt_audio: str | None) -> str | None:
    if prompt_audio is None:
        return None
    candidate = Path(prompt_audio).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    relative = Path(reference_repo) / prompt_audio
    if relative.exists():
        return str(relative.resolve())
    raise FileNotFoundError(f"Could not resolve prompt audio path: {prompt_audio}")


def save_audio(output_path: str | os.PathLike[str], waveform: torch.Tensor, sample_rate: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(output_path), waveform.cpu(), sample_rate)


def write_json(output_path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def runtime_summary() -> dict[str, str]:
    summary = {"python": sys.executable}
    try:
        import ttnn  # noqa: PLC0415

        summary["ttnn"] = ttnn.__file__
    except Exception as exc:  # pragma: no cover - used for runtime reporting
        summary["ttnn_error"] = repr(exc)
    return summary
