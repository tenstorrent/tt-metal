from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class SafetensorsStateDict:
    """
    Minimal safetensors-backed state_dict wrapper.

    Tensors are exposed as host tensors, preferring CPU ``torch.Tensor`` when
    available (preserves dtype like BF16). Callers are expected to transfer
    weights to device exactly once during model construction.
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
    Load a `.safetensors` file.

    Args:
        path: Path to `.safetensors`.
        prefix: If provided, only keys that start with `prefix` are kept, and
            the prefix is stripped from returned keys.
    """
    try:
        pass

        from safetensors.torch import load_file as torch_load_file  # type: ignore

        # Prefer torch loader when available to preserve dtype (e.g. BF16).
        raw: Dict[str, Any] = {k: v.detach().cpu() for k, v in torch_load_file(path, device="cpu").items()}
    except Exception:
        # Torch-free fallback: load as numpy arrays. Note that BF16 support may vary by environment.
        from safetensors.numpy import load_file  # type: ignore

        raw = load_file(path)
    if prefix is None:
        return SafetensorsStateDict(tensors=raw)

    keep: Dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            keep[k[len(prefix) :]] = v
    return SafetensorsStateDict(tensors=keep)
