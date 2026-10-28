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
    bias_row_p50 / bias_row_max   the row |mean error| (normalized), median and max over every row; no n_active
                       gate (v1.24): a level shift. bias_row_argmax names the row.
    bias_rows          each active row's signed mean error (None on inactive rows): check_gates gates
                       bias_row_drift_max = max_i |e_i - e0_i| against the fixed record (v1.26, D3 item 3)
    bias_quantiles     each quantile's signed mean error over the active rows (normalized; v1.32, D6): check_gates
                       gates bias_q_drift_max = max_q |m_q - m0_q| against the fixed record (v1.31, D5 item 2)
    inactive_bias_rows each inactive row's signed mean error (None on active rows): check_gates gates
                       inactive_bias_drift_max with the v1.27 interval (v1.33, D6 item 3)
    bias_rq            each active row's per-quantile signed mean error over steps (None on inactive rows):
                       check_gates reports bias_rq_drift_max = max_{i,q} |e_iq - e0_iq|, ungated (v1.33)
    spread_ratio_rows  each active row's spread ratio (None on inactive rows): spread_ratio_row_drift_max
    pcc_step           PCC of the step differences along the horizon, per quantile: a one-step lag (v1.24)
    row_info           per row metric, the row index, that row's golden std and band RMS (v1.25); for
                       abs_err_max also the quantile, the step and the golden band half-width there
``collapsed(head)`` replaces every quantile by the quantile mean: the failure these metrics must catch.
"""

from __future__ import annotations

import math

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
    if not math.isfinite(den):  # R1-0b: a NaN / inf output is NaN here, not 0, so the checker fails it closed
        return float("nan")
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
    out = torch.where(den > 0, (e * g).sum(1) / den.clamp_min(1e-300), torch.zeros_like(den))
    return torch.where(den.isfinite(), out, torch.full_like(den, float("nan")))  # R1-0b: NaN stays NaN


def active_rows(g_head):
    flat = g_head.double().reshape(g_head.shape[0], -1)
    return flat.std(1) >= ACTIVE_FRAC * flat.std()


def spread_ratios(exp, got):
    e = spread(exp).double().reshape(exp.shape[0], -1).norm(dim=1)
    g = spread(got).double().reshape(got.shape[0], -1).norm(dim=1)
    return g / e.clamp_min(1e-300)


def _q(t, p):
    return float(torch.quantile(t, p)) if t.numel() else float("nan")


def step_diff(h):
    h = h.double()
    return h[..., 1:] - h[..., :-1]


def _row(g_h, i, extra=None):
    g = g_h[i].double()
    out = {
        "row": int(i),
        "row_std": float(g.std()),
        "band_rms": float(spread(g_h[i : i + 1]).double().pow(2).mean().sqrt()),
    }
    return {**out, **(extra or {})}


# every key gate_metrics writes (check_gates fails a gated one with neither floor nor drift, R1-0b)
GATE_KEYS = {
    "pcc_normalized",
    "pcc_spread",
    "pcc_row_p01",
    "pcc_row_min",
    "pcc_row_p01_active",
    "pcc_row_min_active",
    "active_rows",
    "excluded_rows",
    "spread_ratio_p01",
    "spread_ratio_p99",
    "spread_ratio_p50",
    "spread_ratio_min",
    "spread_ratio_max",
    "pcc_row_min_var",
    "flat_rows",
    "flat_max_err",
    "inactive_max_err",
    "inactive_spread_ratio_min",
    "abs_err_max",
    "bias_row_p50",
    "bias_row_max",
    "bias_row_argmax",
    "bias_rows",
    "bias_quantiles",
    "inactive_bias_rows",
    "bias_rq",
    "spread_ratio_rows",
    "pcc_step",
    "row_info",
    "pcc_final",
    "pcc_final_spread",
    "pcc_mean",
}


def gate_metrics(h, g_h, q, g_q, mean=None, g_mean=None) -> dict:
    h, g_h, q, g_q = head(h), head(g_h), head(q), head(g_q)
    rows = row_pccs(g_h, h)
    std = g_h.double().reshape(g_h.shape[0], -1).std(1)
    flat = std < FLAT_STD
    act = active_rows(g_h)
    ratio = spread_ratios(g_h[act], h[act])
    err = (h.double() - g_h.double()).abs().reshape(g_h.shape[0], -1).amax(1)
    diff = h.double() - g_h.double()
    signed = diff.reshape(g_h.shape[0], -1).mean(1)
    bias = signed.abs()
    ratio_all = spread_ratios(g_h, h)
    aidx = act.nonzero().flatten()
    flat_e = diff.abs().reshape(-1)
    r_e, q_e, t_e = (int(v) for v in torch.unravel_index(flat_e.argmax(), diff.shape))
    band_hw = float((g_h[r_e, :, t_e].max() - g_h[r_e, :, t_e].min()) / 2)
    info = {
        "bias_row_max": _row(g_h, int(bias.argmax())),
        "abs_err_max": _row(g_h, r_e, {"quantile": q_e, "step": t_e, "band_half_width": band_hw}),
    }
    if aidx.numel():
        rr = rows[aidx]
        info["pcc_row_min_active"] = _row(g_h, int(aidx[rr.argmin()]))
        info["spread_ratio_min"] = _row(g_h, int(aidx[ratio.argmin()]))
        info["spread_ratio_max"] = _row(g_h, int(aidx[ratio.argmax()]))
    info["pcc_row_min"] = _row(g_h, int(rows.argmin()))  # D5 item 3: name the row behind every row-min metric
    if bool((~act).any()):
        iidx = (~act).nonzero().flatten()
        info["inactive_max_err"] = _row(g_h, int(iidx[err[iidx].argmax()]))
        info["inactive_bias_rows"] = _row(g_h, int(iidx[bias[iidx].argmax()]))  # D6 item 3: the worst inactive row
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
        # R1-0c: band of the inactive rows against the golden's (an inactive collapse is 0; n/a without them)
        "inactive_spread_ratio_min": float(ratio_all[~act].min()) if bool((~act).any()) else 1.0,
        "abs_err_max": float(err.max()),
        "bias_row_p50": float(bias.median()),
        "bias_row_max": float(bias.max()),
        "bias_row_argmax": int(bias.argmax()),
        "bias_rows": [round(float(b), 7) if a else None for b, a in zip(signed, act)],
        # v1.32: active rows only, like bias_rows (inactive rows have inactive_bias_rows); None without any
        "bias_quantiles": (
            [round(float(v), 7) for v in diff[act].mean((0, 2))] if bool(act.any()) else [None] * diff.shape[1]
        ),
        "inactive_bias_rows": [None if a else round(float(b), 7) for b, a in zip(signed, act)],
        "bias_rq": [[round(float(v), 7) for v in e] if a else None for e, a in zip(diff.mean(2), act)],
        "spread_ratio_rows": [round(float(r), 7) if a else None for r, a in zip(ratio_all, act)],
        "pcc_step": pearson(step_diff(g_h), step_diff(h)),
        "row_info": info,
        "pcc_final": pearson(g_q, q),
        "pcc_final_spread": pearson(spread(g_q), spread(q)),
    }
    if mean is not None:
        m["pcc_mean"] = pearson(g_mean, mean)
    return m


def fmt(m: dict) -> str:
    def one(k, v):
        if isinstance(v, (list, dict)):
            return f"{k} [{len(v)}]"
        return f"{k} {v}" if isinstance(v, int) else f"{k} {v:.6f}"

    return "  ".join(one(k, v) for k, v in m.items())
