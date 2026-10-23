# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the t0 device tests: goldens, weights, PCC, padded uploads and downloads."""

from __future__ import annotations

from functools import lru_cache

import pytest
import torch

from models.experimental.t0_forecast.common import GOLDEN_DIR
from models.experimental.t0_forecast.tt import host
from models.experimental.t0_forecast.tt.common import pad_to_tile, upload
from models.experimental.t0_forecast.tt.weights import T0Weights


@lru_cache(maxsize=None)
def golden(name: str) -> dict:
    path = GOLDEN_DIR / name
    if not path.is_file():
        pytest.fail(f"golden {path} missing; run tests/golden/make_goldens.py")
    return torch.load(path, weights_only=False)


@lru_cache(maxsize=1)
def weights() -> T0Weights:
    return T0Weights.load()


def pcc(expected: torch.Tensor, got: torch.Tensor) -> float:
    a = expected.double().flatten()
    b = got.double().flatten()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300))


def module_inputs(x: torch.Tensor, group_ids, variate_type, mask) -> host.T0HostInputs:
    """``T0HostInputs`` for running layers on hidden states ``x`` ``[R, S, d]`` with the module goldens' fields."""
    cfg = weights().config
    time_mask, group_mask, real, gs = host.structured_masks(group_ids, variate_type, mask)
    r, s, _ = x.shape
    return host.T0HostInputs(
        rows=r,
        num_patches=s,
        context_patches=s,
        readout_start=0,
        readout_count=s,
        group_size=gs,
        tokens=torch.zeros(r, s, 3 * cfg.patch_size),
        patch_type=torch.zeros(r, s, dtype=torch.int64),
        time_mask=time_mask,
        group_mask=group_mask,
        real_tokens=real,
        rope=host.rope_tables(s, cfg),
        loc=torch.zeros(r, s),
        scale=torch.ones(r, s),
        horizon=s * cfg.patch_size,
        target_rows=tuple(range(r)),
    )


def clean_fields(rows: int, seq: int, group_size: int = 1):
    """Patched group ids / variate types / masks of clean target rows, ``group_size`` rows per series."""
    ps = weights().config.patch_size
    groups = (torch.arange(rows) // group_size)[:, None, None].expand(rows, seq, ps).clone()
    zeros = torch.zeros(rows, seq, ps, dtype=torch.int64)
    return groups, zeros, zeros.clone()


def upload_rows(device, x: torch.Tensor):
    """``[R, S, d]`` -> flat device ``(1, 1, R * S_pad, d)`` with zero pad slots."""
    r, s, d = x.shape
    sp = pad_to_tile(s)
    padded = torch.zeros(r, sp, d)
    padded[:, :s] = x
    return upload(device, padded.reshape(1, 1, r * sp, d))


def download_rows(t, rows: int, seq: int) -> torch.Tensor:
    """Flat device ``(1, 1, R * S_pad, d)`` -> ``[R, S, d]`` fp32."""
    import ttnn

    out = ttnn.to_torch(t).float()
    return out.reshape(rows, pad_to_tile(seq), -1)[:, :seq]
