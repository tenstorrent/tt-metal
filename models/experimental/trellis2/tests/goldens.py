# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Golden files from the CPU fp32 TRELLIS.2 reference (dump_goldens.py in the project's scripts).

TRELLIS2_GOLDEN_DIR holds cond.pt and ss.pt for one image and seed; TRELLIS2_WEIGHTS holds TRELLIS.2-4B/ckpts.
Tests that need them skip when the variables are unset.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

SS_CKPT = "TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"


def golden_dir() -> str | None:
    return os.environ.get("TRELLIS2_GOLDEN_DIR")


def weights_dir() -> str | None:
    return os.environ.get("TRELLIS2_WEIGHTS")


def missing_reason() -> str | None:
    if not golden_dir() or not weights_dir():
        return "set TRELLIS2_GOLDEN_DIR and TRELLIS2_WEIGHTS"
    return None


@lru_cache(maxsize=None)
def structure_calls() -> list[dict]:
    return torch.load(f"{golden_dir()}/ss.pt", weights_only=False)["calls"]


@lru_cache(maxsize=None)
def structure_steps() -> list[dict]:
    """Per sampler step: t, t_prev, x_t, x_prev, x_0 (x_t of step 0 is the noise)."""
    return torch.load(f"{golden_dir()}/ss.pt", weights_only=False)["steps"]


@lru_cache(maxsize=None)
def conditions() -> dict[str, torch.Tensor]:
    return torch.load(f"{golden_dir()}/cond.pt", weights_only=False)


def call_inputs(i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """x_t, t, cond (cond or neg_cond by the recorded CFG branch), golden output, branch."""
    c = structure_calls()[i]
    cond = conditions()["cond" if c["branch"] == "pos" else "neg_cond"]
    return c["args"][0], c["args"][1], cond, c["out"], c["branch"]


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()
