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


def _centre(t: torch.Tensor) -> torch.Tensor:
    return t - t.mean(dim=-1, keepdim=True)


def quantile_metrics(expected: torch.Tensor, got: torch.Tensor) -> dict:
    """Metrics on read-out quantiles ``[rows, n, ps, Q]`` (asinh space), the golden first:

    ``pcc_normalized``  PCC over everything (dominated by each row's level)
    ``pcc_spread``      PCC of ``h - mean_q(h)``: the quantile spread around each step's centre
    ``pcc_row_p01``, ``pcc_row_min``  per-row PCC of ``h`` (1st percentile, minimum); ``pcc_row_min_row`` is the row
    ``*_active``        the same over the active rows: golden row std >= 5 % of the golden's global std (the set comes
                        from the golden only; ``rows_excluded`` is fixed per golden)
    ``spread_ratio_inactive_p01``, ``pcc_spread_inactive``  the same band checks over the inactive rows (reported)
    ``spread_ratio_p01`` / ``_p50`` / ``_p99`` / ``_min``  per active row, ``|h_dev - mean_q| / |h_ref - mean_q|`` (band amplitude; 0 for a
                        collapsed output)
    """
    e, g = expected.double(), got.double()
    n = e.shape[0]
    rows = torch.nan_to_num(torch.tensor([pcc(e[r], g[r]) for r in range(n)], dtype=torch.float64), nan=-1.0)
    flat = e.reshape(n, -1)
    active = flat.std(dim=1) >= 0.05 * flat.std()
    ce, cg = _centre(e).reshape(n, -1), _centre(g).reshape(n, -1)
    ratio_all = cg.norm(dim=1) / ce.norm(dim=1).clamp(min=1e-30)
    ratio, ratio_inactive = ratio_all[active], ratio_all[~active]
    rows_active = rows[active]
    q = lambda t, p: float(torch.quantile(t, p)) if t.numel() else float("nan")
    return {
        "pcc_normalized": pcc(e, g),
        "pcc_spread": pcc(_centre(e), _centre(g)),
        "pcc_row_p01": q(rows, 0.01),
        "pcc_row_min": float(rows.min()),
        "pcc_row_min_row": int(rows.argmin()),
        "pcc_row_p01_active": q(rows_active, 0.01),
        "pcc_row_min_active": float(rows_active.min()) if rows_active.numel() else float("nan"),
        "rows_excluded": int((~active).sum()),
        "spread_ratio_p01": q(ratio, 0.01),
        "spread_ratio_p50": q(ratio, 0.5),
        "spread_ratio_p99": q(ratio, 0.99),
        "spread_ratio_min": float(ratio.min()) if ratio.numel() else float("nan"),
        # D4 item 2 (reported, no floor yet): the inactive rows' band, which the floors above never see.
        "spread_ratio_inactive_p01": q(ratio_inactive, 0.01),
        "pcc_spread_inactive": pcc(ce[~active], cg[~active]) if ratio_inactive.numel() else float("nan"),
    }


def final_metrics(expected: torch.Tensor, got: torch.Tensor) -> dict:
    """Data-space quantiles ``[..., Q]``: ``pcc_final`` and ``pcc_final_spread`` (PCC of the quantiles minus their
    mean over the quantile axis; ``pcc_final`` alone scores a collapsed forecast near 1)."""
    return {"pcc_final": pcc(expected, got), "pcc_final_spread": pcc(_centre(expected.double()), _centre(got.double()))}


# Gate floors per case class, set by the orchestrator in D3 and D4 (/home/andy/avan/fleet/t0-gamma/DECISIONS.md). They
# change only through a decision. Shared by the PCC suite (tests/pcc/test_model.py), the bench records (``floor_ok``) and
# tests/perf/check_gates.py, which also applies the drift margins. A number is a lower bound; ``(lo, hi)`` a range.
_BAND_ACCURACY = {  # D4: the bench shapes at accuracy
    "pcc_final_spread": 0.998,
    "spread_ratio_p01": 0.95,
    "spread_ratio_p50": (0.97, 1.03),
    "spread_ratio_p99": (0.0, 1.25),
    "spread_ratio_min": 0.80,
}
_BAND_PERFORMANCE = {
    "pcc_final_spread": 0.99,
    "spread_ratio_p01": 0.92,
    "spread_ratio_p50": (0.95, 1.05),
    "spread_ratio_p99": (0.0, 1.75),
    "spread_ratio_min": 0.70,
}
_BAND_SINGLE = {"pcc_final_spread": 0.998, "spread_ratio_p99": (0.80, 1.25)}  # one row: p01 = p99 = the row
_BAND_COVERAGE = {  # e2e and the GIFT-settings cases, both precisions; spread_ratio_min reported only
    "pcc_final_spread": 0.998,
    "spread_ratio_p01": 0.90,
    "spread_ratio_p50": (0.95, 1.05),
    "spread_ratio_p99": (0.0, 1.5),
}
FLOORS = {
    "accuracy": {
        "pcc_normalized": 0.999,
        "pcc_spread": 0.998,
        "pcc_row_p01": 0.995,
        "pcc_row_min": 0.99,
        "pcc_final": 0.99,
        **_BAND_ACCURACY,
    },
    "performance": {
        "pcc_normalized": 0.995,
        "pcc_spread": 0.99,
        "pcc_row_p01": 0.97,
        "pcc_row_min": 0.93,
        "pcc_final": 0.98,
        **_BAND_PERFORMANCE,
    },
    # Branch-coverage inputs (e2e: the smooth generator plus counts and gaps; paths). Per-row PCC is reported every
    # round, not gated: it is ill-conditioned on rows whose quantiles span ~1 % of their level (e2e row 9: std 0.013
    # in asinh space, so a 0.002 absolute error reads 0.979).
    "coverage": {"pcc_normalized": 0.999, "pcc_spread": 0.998, "pcc_final": 0.99, **_BAND_COVERAGE},
    "coverage_performance": {"pcc_normalized": 0.995, "pcc_spread": 0.99, "pcc_final": 0.98, **_BAND_COVERAGE},
    "single": {"pcc_normalized": 0.999, "pcc_spread": 0.999, "pcc_final": 0.99, **_BAND_SINGLE},
    "single_long": {"pcc_normalized": 0.9975, "pcc_spread": 0.998, "pcc_final": 0.9975, **_BAND_SINGLE},
    "single_performance": {"pcc_normalized": 0.985, "pcc_final": 0.985, **_BAND_SINGLE},
    "single_long_performance": {"pcc_normalized": 0.97, "pcc_final": 0.97, **_BAND_SINGLE},
}
# Drift margins on the target (paper-accuracy); twice these on every other bench record. D3: against round 0; D4: the
# band metrics against the round-2 bench (job 614), since round 0 did not record them. Upper-bounded metrics (p99)
# drift upwards.
DRIFT = {"pcc_normalized": 0.0005, "pcc_spread": 0.002, "pcc_row_p01": 0.002, "pcc_row_min": 0.005, "pcc_final": 0.0005}
DRIFT_BAND = {"pcc_final_spread": 0.002, "spread_ratio_p01": 0.02, "spread_ratio_p99": 0.02}


def below(value: float, floor) -> bool:
    """``value`` misses ``floor`` (a lower bound, or an inclusive ``(lo, hi)`` range); NaN always misses."""
    if value != value:
        return True
    if isinstance(floor, tuple):
        return not floor[0] <= value <= floor[1]
    return value < floor


def misses(metrics: dict, floors: dict) -> list[str]:
    """The floors ``metrics`` misses, as ``"<metric> <value> outside <floor>"``; metrics absent are skipped."""
    return [f"{k} {metrics[k]:.6f} outside {f}" for k, f in floors.items() if k in metrics and below(metrics[k], f)]


def meets(metrics: dict, floors: dict) -> bool:
    return not misses(metrics, floors)


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
