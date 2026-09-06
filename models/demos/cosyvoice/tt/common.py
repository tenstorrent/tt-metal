# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the CosyVoice TTNN bring-up: goldens, PCC, weight folding."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")


# --------------------------------------------------------------------------
# goldens
# --------------------------------------------------------------------------
def load_golden(name: str, golden_dir: str | None = None) -> dict[str, np.ndarray]:
    """Load tests/golden/<name>.npz, resolving the dedup alias map.

    gen_golden.py stores byte-identical arrays once (step N's input KV cache is
    step N-1's output verbatim), so a plain np.load would report missing keys.
    """
    d = golden_dir or GOLDEN_DIR
    path = os.path.join(d, name if name.endswith(".npz") else f"{name}.npz")
    with np.load(path) as z:
        data = {k: z[k] for k in z.files if k != "__aliases__"}
        if "__aliases__" in z.files:
            for alias, target in json.loads(bytes(z["__aliases__"]).decode()).items():
                data[alias] = data[target]
    return data


def golden_manifest(golden_dir: str | None = None) -> dict:
    d = golden_dir or GOLDEN_DIR
    with open(os.path.join(d, "manifest.json")) as fh:
        return json.load(fh)


def as_torch(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Goldens store large float arrays as fp16; widen before comparing."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(dtype) if t.is_floating_point() else t


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over flattened tensors, computed in float64.

    float64 matters: at PCC >= 0.9999 the quantity being reported is the *fifth*
    significant figure, which float32 accumulation over 10^5 elements does not
    reliably carry.
    """
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()).clamp_min(1e-30)
    return float((a * b).sum() / denom)


def report(name: str, got: torch.Tensor, want: torch.Tensor, gate: float = 0.99) -> float:
    """Print a one-line PCC report and return the value. Never asserts -- callers
    assert, so a failing test shows the number alongside the failure."""
    p = pcc(got, want)
    md = float((got.detach().flatten().double() - want.detach().flatten().double()).abs().max())
    status = "PASS" if p >= gate else "FAIL"
    print(f"  {status}  {name:<34} PCC {p:.10f}  max|d| {md:.3e}  gate {gate}")
    return p


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def fold_weight_norm(weight_v: torch.Tensor, weight_g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Collapse torch's weight_norm parameterisation w = g * v/||v|| into one tensor.

    Every conv in HiFT is weight_norm-wrapped. Folding at load time removes a
    per-inference normalisation the device would otherwise carry for no benefit --
    the norm is constant once the weights are frozen.
    """
    norm_dims = [d for d in range(weight_v.dim()) if d != dim]
    norm = weight_v.norm(2, dim=norm_dims, keepdim=True)
    return weight_g * weight_v / norm.clamp_min(1e-12)


def fold_conv_bn(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
    eps: float = 1e-5,
):
    """Standard BN fold. Kept here because the flow encoder's estimator uses it."""
    scale = bn_w / torch.sqrt(bn_var + eps)
    w = conv_w * scale.reshape([-1] + [1] * (conv_w.dim() - 1))
    b0 = conv_b if conv_b is not None else torch.zeros_like(bn_mean)
    b = (b0 - bn_mean) * scale + bn_b
    return w, b
