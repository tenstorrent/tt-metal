# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Gate metrics of a forecast against the reference goldens (CONTRACT v1.1 §3; copied from TiRex-2 651e8c4af93
tests/pcc/metrics.py and adapted to FlowState's head).

The head (normalized space) is the reference's own CPM combine of the pre-denorm decoder output,
``combined_normalized`` ``(B, 9, H', 1)``, taken as ``[rows, 9, H']`` (the quantile axis before torch.quantile's
sort). Data space: ``quantile_outputs`` ``(B, 9, H', 1)`` and ``prediction_outputs`` ``(B, H', 1)``.
    pcc_normalized  PCC over every head value
    pcc_spread      PCC of the head minus its mean over the 9 quantiles (the band shape)
    pcc_row_p01 / pcc_row_min, and the _active variants over the active rows
    active_rows / excluded_rows   active: golden head std >= ACTIVE_FRAC x the golden's global std (golden only)
    spread_ratio_p01 / p50 / p99 / min / max   per active row, |spread(h)| / |spread(h_ref)|
    pcc_row_min_var, flat_rows, flat_max_err   rows whose golden head std < FLAT_STD (normalized units)
    inactive_max_err   worst absolute error over the inactive rows
    abs_err_max        worst absolute error over every element: a gross single-element check (v1.16)
    pcc_final, pcc_final_spread   data-space quantiles, and minus their quantile mean
    pcc_mean           the weighted-mean point forecast
``collapsed(head)`` replaces every quantile by the quantile mean: the failure these metrics must catch.
"""

from __future__ import annotations

import torch

Q = 9
FLAT_STD = 0.05
ACTIVE_FRAC = 0.05


def head(t: torch.Tensor) -> torch.Tensor:
    """(B, 9, H, 1) -> [rows, 9, H]."""
    return t[..., 0] if t.dim() == 4 else t


def pearson(a, b) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a * b).sum()) / den if den > 0 else 0.0


def spread(h):
    return h - h.mean(-2, keepdim=True)


def collapsed(h):
    return h.mean(-2, keepdim=True).expand_as(h).contiguous()


def row_pccs(exp, got):
    e = exp.double().reshape(exp.shape[0], -1)
    g = got.double().reshape(got.shape[0], -1)
    e, g = e - e.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    den = e.norm(dim=1) * g.norm(dim=1)
    return torch.where(den > 0, (e * g).sum(1) / den.clamp_min(1e-300), torch.zeros_like(den))


def active_rows(g_head):
    flat = g_head.double().reshape(g_head.shape[0], -1)
    return flat.std(1) >= ACTIVE_FRAC * flat.std()


def spread_ratios(exp, got):
    e = spread(exp).double().reshape(exp.shape[0], -1).norm(dim=1)
    g = spread(got).double().reshape(got.shape[0], -1).norm(dim=1)
    return g / e.clamp_min(1e-300)


def _q(t, p):
    return float(torch.quantile(t, p)) if t.numel() else float("nan")


def gate_metrics(h, g_h, q, g_q, mean=None, g_mean=None) -> dict:
    h, g_h, q, g_q = head(h), head(g_h), head(q), head(g_q)
    rows = row_pccs(g_h, h)
    std = g_h.double().reshape(g_h.shape[0], -1).std(1)
    flat = std < FLAT_STD
    act = active_rows(g_h)
    ratio = spread_ratios(g_h[act], h[act])
    err = (h.double() - g_h.double()).abs().reshape(g_h.shape[0], -1).amax(1)
    m = {
        "pcc_normalized": pearson(g_h, h),
        "pcc_spread": pearson(spread(g_h), spread(h)),
        "pcc_row_p01": _q(rows, 0.01),
        "pcc_row_min": float(rows.min()),
        "pcc_row_p01_active": _q(rows[act], 0.01),
        "pcc_row_min_active": float(rows[act].min()) if bool(act.any()) else float("nan"),
        "active_rows": int(act.sum()),
        "excluded_rows": int((~act).sum()),
        "spread_ratio_p01": _q(ratio, 0.01),
        "spread_ratio_p99": _q(ratio, 0.99),
        "spread_ratio_p50": _q(ratio, 0.5),
        "spread_ratio_min": float(ratio.min()) if ratio.numel() else float("nan"),
        "spread_ratio_max": float(ratio.max()) if ratio.numel() else float("nan"),
        "pcc_row_min_var": float(rows[~flat].min()) if bool((~flat).any()) else 1.0,
        "flat_rows": int(flat.sum()),
        "flat_max_err": float(err[flat].max()) if bool(flat.any()) else 0.0,
        "inactive_max_err": float(err[~act].max()) if bool((~act).any()) else 0.0,
        "abs_err_max": float(err.max()),
        "pcc_final": pearson(g_q, q),
        "pcc_final_spread": pearson(spread(g_q), spread(q)),
    }
    if mean is not None:
        m["pcc_mean"] = pearson(g_mean, mean)
    return m


def fmt(m: dict) -> str:
    return "  ".join(f"{k} {v}" if isinstance(v, int) else f"{k} {v:.6f}" for k, v in m.items())
