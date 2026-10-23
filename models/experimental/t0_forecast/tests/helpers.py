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
    if float(a.norm()) == 0.0 or float(b.norm()) == 0.0:
        return 0.0  # a constant carries no correlation (a fully collapsed spread)
    return float((a @ b) / (a.norm() * b.norm()))


def quantile_metrics(expected: torch.Tensor, got: torch.Tensor) -> dict:
    """Gate metrics on read-out quantiles ``[rows, n, ps, Q]`` (asinh space):

    ``pcc_normalized``  PCC over everything (dominated by each row's level)
    ``pcc_spread``      PCC of ``h - mean_q(h)``: the quantile spread around each step's centre
    ``pcc_row_p01``, ``pcc_row_min``  per-row PCC of ``h``, 1st percentile and minimum over rows
    """
    e, g = expected.double(), got.double()
    centre = lambda t: t - t.mean(dim=-1, keepdim=True)
    rows = torch.tensor([pcc(e[r], g[r]) for r in range(e.shape[0])], dtype=torch.float64)
    rows = torch.nan_to_num(rows, nan=-1.0)
    return {
        "pcc_normalized": pcc(e, g),
        "pcc_spread": pcc(centre(e), centre(g)),
        "pcc_row_p01": float(torch.quantile(rows, 0.01)),
        "pcc_row_min": float(rows.min()),
    }


# Gate floors per case class (proposed at the G1 re-raise; the orchestrator sets the final values). Shared by the
# PCC suite (tests/pcc/test_model.py) and the bench records (``floor_ok``).
FLOORS = {
    "accuracy": {
        "pcc_normalized": 0.999,
        "pcc_spread": 0.998,
        "pcc_row_p01": 0.995,
        "pcc_row_min": 0.98,
        "pcc_final": 0.99,
    },
    "performance": {
        "pcc_normalized": 0.995,
        "pcc_spread": 0.99,
        "pcc_row_p01": 0.97,
        "pcc_row_min": 0.90,
        "pcc_final": 0.98,
    },
    # Branch-coverage inputs (e2e: the smooth generator plus counts and gaps; paths). Per-row PCC is reported, not
    # gated: it is ill-conditioned on rows whose quantiles span ~1 % of their level (e2e row 9: std 0.013 in asinh
    # space, so a 0.002 absolute error reads 0.979).
    "coverage": {"pcc_normalized": 0.999, "pcc_spread": 0.998, "pcc_final": 0.99},
    "single": {"pcc_normalized": 0.999, "pcc_spread": 0.999, "pcc_final": 0.99},
    "single_long": {"pcc_normalized": 0.9975, "pcc_spread": 0.998, "pcc_final": 0.99},
}


def meets(metrics: dict, floors: dict) -> bool:
    return all(metrics[k] >= v for k, v in floors.items() if k in metrics)


def collapsed(t: torch.Tensor) -> torch.Tensor:
    """Every quantile replaced by the mean over the quantile axis (the bf16-softplus failure, at its worst)."""
    return t.mean(dim=-1, keepdim=True).expand_as(t).contiguous()


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
