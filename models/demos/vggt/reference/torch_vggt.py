"""Torch reference loader for facebook/VGGT-1B.

Delegates to the upstream code at /home/ttuser/experiments/vggt/vggt_ref
and loads pretrained weights from the local HuggingFace cache. We only
need forward+eval for PCC validation; training utilities are not wired.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

_VGGT_REF = Path("/home/ttuser/experiments/vggt/vggt_ref")
if str(_VGGT_REF) not in sys.path:
    sys.path.insert(0, str(_VGGT_REF))

_WEIGHTS = Path(
    "/home/ttuser/.cache/huggingface/hub/"
    "models--facebook--VGGT-1B/snapshots/"
    "860abec7937da0a4c03c41d3c269c366e82abdf9/model.safetensors"
)


def load_state_dict() -> dict[str, torch.Tensor]:
    return load_file(str(_WEIGHTS))


def load_vggt(eval_mode: bool = True):
    from vggt.models.vggt import VGGT  # type: ignore

    model = VGGT()
    sd = load_state_dict()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict: missing={missing[:3]} unexpected={unexpected[:3]}")
    if eval_mode:
        model.eval()
    return model
