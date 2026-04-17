"""Load a subset of the Fast3R safetensors checkpoint into the reference model.

Only encoder + decoder keys are needed for the initial port; DPT head weights
are left for a later iteration.
"""
from __future__ import annotations

import os
from typing import Dict

import torch
from safetensors import safe_open

from .model import Fast3R


def _default_path() -> str:
    return os.environ.get(
        "FAST3R_WEIGHTS",
        "/home/ttuser/.cache/huggingface/hub/models--jedyang97--Fast3R_ViT_Large_512/"
        "snapshots/a2c770b768ceb3a53c36c4f7a3619db0413dc3a1/model.safetensors",
    )


def _take(f, prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}


def load_fast3r(weights_path: str | None = None, device: str = "cpu") -> Fast3R:
    path = weights_path or _default_path()
    model = Fast3R().eval().to(device)
    with safe_open(path, framework="pt", device=device) as f:
        enc_sd = _take(f, "encoder.")
        dec_sd = _take(f, "decoder.")
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    assert not unexpected, f"encoder unexpected keys: {unexpected[:5]}"
    missing, unexpected = model.decoder.load_state_dict(dec_sd, strict=False)
    assert not unexpected, f"decoder unexpected keys: {unexpected[:5]}"
    return model
