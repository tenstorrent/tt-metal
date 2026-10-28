# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Copied from TiRex-2 651e8c4af93 tests/pcc/metrics.py (CONTRACT §3); Q, P = 9, 32 and Tafsut post-processing.

Gate metrics of a forecast against the upstream goldens (CONTRACT section 3, PLAN section 9).

All head metrics use the head output at the read-out tokens in normalized space, ``[rows, n, 288]`` with 288 =
9 quantiles x 32 steps (quantile-major):

    pcc_normalized  PCC over every value
    pcc_spread      PCC of the head output minus its mean over the quantile axis (the band shape; blind to
                    per-row offsets, which dominate pcc_normalized on trending inputs)
    pcc_row_p01     1st percentile of the per-row PCC
    pcc_row_min     minimum per-row PCC
    pcc_row_min_var minimum per-row PCC over rows whose upstream head output varies (std >= FLAT_STD); a flat
                    row (e.g. an intermittent series forecast as a constant band, std ~0.003) has a per-row PCC
                    set by rounding noise, so it is checked by ``flat_max_err`` instead
    flat_max_err    worst absolute error (normalized units) over the flat rows (0 when there are none)
    inactive_max_err
                    worst absolute error (normalized units) over the inactive rows (the golden-defined complement
                    of the active set; D4 condition b: ``flat_max_err`` misses inactive rows that are not flat)
    abs_err_max     worst absolute error (normalized units) over every element: a gross single-element check, not a
                    band check (PLAN section 9 v1.16)
    pcc_row_p01_active, pcc_row_min_active
                    per-row PCC over the active rows only: rows whose golden head std is >= ACTIVE_FRAC of the
                    golden's global std (PLAN section 9 v1.2); the active set depends on the golden only
    active_rows, excluded_rows
    spread_ratio_p01, spread_ratio_p99
                    per active row, |h - mean_q h| / |h_ref - mean_q h_ref| (band amplitude ratio; PLAN v1.3)
    spread_ratio_p50
                    the median ratio: a uniform band scaling (0.88x, 1.12x) moves it, row noise does not
    spread_ratio_min, spread_ratio_max
                    the extremes of that ratio (floor-only, no drift; D3 condition 1b)
    bias_row_p50    median over rows of |mean(h - h_ref)| (normalized units): a level shift of the whole forecast
                    (PLAN v1.22 ``level_shift``) moves it one for one, element noise averages out
    pcc_step        PCC of the step-to-step differences along the horizon (within each 32-step patch): a forecast
                    delayed by a step (PLAN v1.22 ``lag1``) keeps its values but not its increments
    pcc_final       PCC of the single-pass [targets, 9, H] forecast after inverse scaling and band calibration
    pcc_final_spread  PCC of that forecast minus its mean over the quantile axis (data space)
    median_err      worst |median error| / scale over target rows

``collapsed(head)`` replaces every quantile by the mean over quantiles: the failure mode these metrics must
catch (t0-gamma's ``pcc_normalized`` scored 0.99965 on it).
"""

from __future__ import annotations

import torch

from models.experimental.tafsut_forecast.reference import dataflow as df

Q, P = 9, 32
MIN_P50, MIN_P01 = 8, 100  # PLAN v1.14: p50 at >= 8 active rows, p01/p99 at >= 100
FLAT_STD = 0.05  # normalized units: 5 % of the series' own scale
ACTIVE_FRAC = 0.05  # active rows: golden head std >= 5 % of the golden's global std


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a * b).sum()) / den if den > 0 else 0.0


def spread(head: torch.Tensor) -> torch.Tensor:
    h = head.reshape(*head.shape[:-1], Q, P)
    return (h - h.mean(-2, keepdim=True)).reshape(head.shape)


def step_diff(head: torch.Tensor) -> torch.Tensor:
    h = head.double().reshape(*head.shape[:-1], Q, P)
    return h[..., 1:] - h[..., :-1]


def collapsed(head: torch.Tensor) -> torch.Tensor:
    h = head.reshape(*head.shape[:-1], Q, P)
    return h.mean(-2, keepdim=True).expand_as(h).reshape(head.shape)


def row_pccs(exp: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    e = exp.double().reshape(exp.shape[0], -1)
    g = got.double().reshape(got.shape[0], -1)
    e, g = e - e.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    den = e.norm(dim=1) * g.norm(dim=1)
    return torch.where(den > 0, (e * g).sum(1) / den.clamp_min(1e-300), torch.zeros_like(den))


def forecast_from_head(head: torch.Tensor, prep, horizon: int) -> torch.Tensor:
    """``[rows, 9, H]`` data-space quantiles (fallback, sinh, rescale, slice)."""
    return df.postprocess(head.float(), prep, horizon).transpose(1, 2)


def active_rows(g_head: torch.Tensor) -> torch.Tensor:
    """Rows whose golden head output varies: std >= ACTIVE_FRAC x the golden's global std (golden only)."""
    flat = g_head.double().reshape(g_head.shape[0], -1)
    return flat.std(1) >= ACTIVE_FRAC * flat.std()


def spread_ratios(exp: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    e = spread(exp).double().reshape(exp.shape[0], -1).norm(dim=1)
    g = spread(got).double().reshape(got.shape[0], -1).norm(dim=1)
    return g / e.clamp_min(1e-300)


def gate_metrics(head, g_head, q, g_q, scale_targets) -> dict:
    rows = row_pccs(g_head, head)
    std = g_head.double().reshape(g_head.shape[0], -1).std(1)
    flat = std < FLAT_STD
    act = active_rows(g_head)
    ratio = spread_ratios(g_head[act], head[act])
    n_act = int(act.sum())
    na = lambda ok, f: float(f()) if ok else None  # None prints as "not applicable (n_active=N < M)" (v1.14)
    err = (head.double() - g_head.double()).abs().reshape(g_head.shape[0], -1).amax(1)
    med_err = (q[:, Q // 2] - g_q[:, Q // 2]).abs().amax(-1) / scale_targets.reshape(-1)
    q_spread = lambda t: t - t.mean(1, keepdim=True)  # [targets, 9, H]: minus the quantile mean
    return {
        "pcc_normalized": pearson(g_head, head),
        "pcc_spread": pearson(spread(g_head), spread(head)),
        "pcc_row_p01": float(torch.quantile(rows, 0.01)),
        "pcc_row_min": float(rows.min()),
        "pcc_row_p01_active": na(n_act >= MIN_P01, lambda: torch.quantile(rows[act], 0.01)),
        "pcc_row_min_active": na(n_act >= 1, lambda: rows[act].min()),
        "active_rows": int(act.sum()),
        "excluded_rows": int((~act).sum()),
        "spread_ratio_p01": na(n_act >= MIN_P01, lambda: torch.quantile(ratio, 0.01)),
        "spread_ratio_p99": na(n_act >= MIN_P01, lambda: torch.quantile(ratio, 0.99)),
        "spread_ratio_p50": na(n_act >= MIN_P50, lambda: torch.quantile(ratio, 0.5)),
        "spread_ratio_min": na(n_act >= 1, lambda: ratio.min()),
        "spread_ratio_max": na(n_act >= 1, lambda: ratio.max()),
        "pcc_row_min_var": float(rows[~flat].min()) if bool((~flat).any()) else 1.0,
        "flat_rows": int(flat.sum()),
        "flat_max_err": float(err[flat].max()) if bool(flat.any()) else 0.0,
        "inactive_max_err": float(err[~act].max()) if bool((~act).any()) else 0.0,
        "abs_err_max": float(err.max()),  # gross single-element check, not a band check (v1.16)
        "bias_row_p50": float((head.double() - g_head.double()).reshape(g_head.shape[0], -1).mean(1).abs().median()),
        "pcc_step": pearson(step_diff(g_head), step_diff(head)),
        "pcc_final": pearson(g_q, q),
        "pcc_final_spread": pearson(q_spread(g_q), q_spread(q)),
        "median_err": float(med_err.max()),
    }


def collapsed_metrics(g_head, g_q, prep, horizon) -> dict:
    """The gate metrics of the golden head output with every quantile collapsed to the quantile mean."""
    c = collapsed(g_head)
    return gate_metrics(c, g_head, forecast_from_head(c, prep, horizon), g_q, prep.scale)


def fmt(m: dict) -> str:
    def one(k, v):
        if v is None:
            return f"{k} n/a"
        if isinstance(v, int):
            return f"{k} {v}"
        return f"{k} {v:.3e}" if k in ("median_err", "flat_max_err") else f"{k} {v:.6f}"

    return "  ".join(one(k, v) for k, v in m.items())
