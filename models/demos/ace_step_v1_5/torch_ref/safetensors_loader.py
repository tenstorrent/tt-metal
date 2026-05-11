from __future__ import annotations

"""
Torch reference safetensors loader.

ACE-Step provides weights as `.safetensors` (HF snapshots). For parity tests and
torch_ref modules we want:
- host tensors (CPU) so call sites can decide device placement
- dtype preservation when torch is available (especially BF16)
- optional prefix filtering + stripping, matching the HF-style `decoder.` namespace
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class SafetensorsStateDict:
    """
    Minimal safetensors-backed state_dict wrapper.

    `tensors` contains CPU host tensors/arrays keyed by string.
    """

    tensors: Dict[str, Any]

    def __contains__(self, k: str) -> bool:  # pragma: no cover
        return k in self.tensors

    def __getitem__(self, k: str) -> Any:
        return self.tensors[k]

    def keys(self) -> Iterable[str]:  # pragma: no cover
        return self.tensors.keys()


def load_safetensors_state_dict(path: str, *, prefix: Optional[str] = None) -> SafetensorsStateDict:
    """
    Load a `.safetensors` file as host tensors.

    Args:
        path: Path to `.safetensors`.
        prefix: If provided, only keys that start with `prefix` are kept and the
            prefix is stripped from returned keys.

    Returns:
        SafetensorsStateDict(tensors=...)
    """
    raw: Dict[str, Any]
    try:
        # Prefer torch loader when available to preserve dtype (e.g. BF16).
        from safetensors.torch import load_file as torch_load_file  # type: ignore

        raw = {k: v.detach().cpu() for k, v in torch_load_file(path, device="cpu").items()}
    except Exception:
        # Torch-free fallback: numpy loader.
        from safetensors.numpy import load_file  # type: ignore

        raw = load_file(path)

    if prefix is None:
        return SafetensorsStateDict(tensors=raw)

    keep: Dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            keep[k[len(prefix) :]] = v
    return SafetensorsStateDict(tensors=keep)
