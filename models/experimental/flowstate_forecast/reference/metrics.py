# Template: copy into <model dir>/tests/helpers.py (or a metrics module) and adapt the axes.
"""Gate metrics for quantile forecasts. `q` is [rows, horizon, n_quantiles]; `median` is the median's index.

Normalized space = the model's own scaled space (before de-normalizing); final = data space.
"""

from __future__ import annotations

import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = (a.norm() * b.norm()).item()
    return float((a @ b).item() / den) if den > 0 else float(torch.allclose(a, b))


def spread(q: torch.Tensor, median: int) -> torch.Tensor:
    """The band around the median: q - median, per quantile."""
    return q - q[..., median : median + 1]


def row_pccs(exp: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    return torch.tensor([pcc(e, g) for e, g in zip(exp, got)])


def active_rows(exp: torch.Tensor, median: int, rel: float = 1e-3) -> torch.Tensor:
    """Rows whose reference band is not flat (flat intermittent series make per-row PCC meaningless)."""
    width = spread(exp, median).abs().amax(dim=(-1, -2))
    return width > rel * width.max().clamp_min(1e-12)


def spread_ratio(exp: torch.Tensor, got: torch.Tensor, median: int) -> tuple[float, float]:
    """p01 and p99 over active rows of (device band width / reference band width). 1.0 is exact."""
    act = active_rows(exp, median)
    we = spread(exp, median).abs().mean(dim=(-1, -2))[act]
    wg = spread(got, median).abs().mean(dim=(-1, -2))[act]
    r = (wg / we).double()
    return float(r.quantile(0.01)), float(r.quantile(0.99))


def gate_metrics(exp_norm, got_norm, exp_final, got_final, median: int) -> dict[str, float]:
    rows = row_pccs(exp_norm, got_norm)
    act = active_rows(exp_norm, median)
    lo, hi = spread_ratio(exp_norm, got_norm, median)
    return {
        "pcc_normalized": pcc(exp_norm, got_norm),
        "pcc_spread": pcc(spread(exp_norm, median), spread(got_norm, median)),
        "pcc_row_p01": float(rows.double().quantile(0.01)),
        "pcc_row_min": float(rows.min()),
        "pcc_row_p01_active": float(rows[act].double().quantile(0.01)) if act.any() else float("nan"),
        "pcc_row_min_active": float(rows[act].min()) if act.any() else float("nan"),
        "spread_ratio_p01": lo,
        "spread_ratio_p99": hi,
        "pcc_final": pcc(exp_final, got_final),
        "pcc_final_spread": pcc(spread(exp_final, median), spread(got_final, median)),
    }


def collapsed(q: torch.Tensor) -> torch.Tensor:
    """Every quantile replaced by the mean over quantiles: the failure plain PCC misses (scored 0.9996)."""
    return q.mean(dim=-1, keepdim=True).expand_as(q).contiguous()
