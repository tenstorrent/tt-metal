from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class SafetensorsStateDict:
    """
    Minimal safetensors-backed state_dict wrapper.

    All tensors are exposed as numpy arrays on host. Callers are expected to
    transfer weights to device exactly once during model construction.
    """

    tensors: Dict[str, np.ndarray]

    def __contains__(self, k: str) -> bool:  # pragma: no cover
        return k in self.tensors

    def __getitem__(self, k: str) -> np.ndarray:
        return self.tensors[k]

    def keys(self) -> Iterable[str]:  # pragma: no cover
        return self.tensors.keys()


def load_safetensors_state_dict(path: str, *, prefix: Optional[str] = None) -> SafetensorsStateDict:
    """
    Load a `.safetensors` file without torch.

    Args:
        path: Path to `.safetensors`.
        prefix: If provided, only keys that start with `prefix` are kept, and
            the prefix is stripped from returned keys.
    """
    # Prefer torch-free loading, but numpy backend doesn't support BF16 in many environments.
    raw: Dict[str, np.ndarray]
    try:
        from safetensors.numpy import load_file  # type: ignore

        raw = load_file(path)
    except Exception:
        # Fallback: use torch loader to handle BF16 safely, then convert to numpy float32 on host.
        import torch
        from safetensors.torch import load_file as torch_load_file  # type: ignore

        tdict = torch_load_file(path, device="cpu")
        raw = {k: v.detach().to(torch.float32).cpu().numpy() for k, v in tdict.items()}
    if prefix is None:
        return SafetensorsStateDict(tensors=raw)

    keep: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            keep[k[len(prefix) :]] = v
    return SafetensorsStateDict(tensors=keep)
