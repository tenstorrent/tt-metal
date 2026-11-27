# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MoeScorer``: a MoE Block's long-context accuracy against its golden, every row scored (``block.BlockScorer``).

Per row: the MoE output's error (``windowed_errors.row_sums``) and the routing against the golden's top-k
(``routing_error.routing_rows``: recall, the golden weight on missed experts, missed experts ranked in the golden's
selection). The MoE carries no state across chunks, so its error is judged per window, absolutely, not by growth.

Every window measure (``measure``) has its metric name; the same measure over the whole run is reported as
``run_<name>``, since a metric name lives in one metric tier only. Metric tiers (``block.METRIC_TIERS``):

* 0, headline: ``run_moe_nrmse`` and ``run_recall_mean`` over the run;
* 1, tails: every ``window``-row window aligned to row 0 (a shorter run is one partial window) with all of
  ``measure``, and the run's other ``run_*`` measures (absolute error, per-row NRMSE median / p99 / max, dropped
  weight mean / p99, misses and the fraction in the top half of the golden's selection);
* 2, diagnostics: the run's output gain (``run_moe_gain``) and histogram of missed experts' golden ranks
  (``run_miss_rank_histogram``), and the windows' trends (``<metric>_trend``, ``windowed_errors.trend``): the
  absolute error against the reference's own growth, the NRMSE, row NRMSE median, recall and dropped weight.

Per chunk (``score_chunk``): the Block's contracts only (finite output; no row routed to one expert twice), no
measurement. The bars are per window (``run`` bars): the window values do not hold on a chunk's smaller sample.

Device outputs (``BlockOutputs``): ``output`` ``[rows, hidden]``; ``extras["expert_ids"]`` ``[rows, k]`` (the
golden's expert numbering) and ``extras["expert_weights"]`` ``[rows, k]``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from models.demos.common.prefill.long_context_accuracy.block import BlockOutputs, ChunkScore, MetricTiers
from models.demos.common.prefill.long_context_accuracy.moe.routing_error import routing_rows
from models.demos.common.prefill.long_context_accuracy.windowed_errors import (
    SUM_YY,
    Window,
    WindowedErrors,
    gain,
    golden_floor,
    nrmse,
    quantile,
    rms_err,
    rms_ref,
    row_nrmse,
    row_sums,
    trend,
)

HEADLINE = ("moe_nrmse", "recall_mean")  # (as run_<name> in metric tier 0)
# Window metrics whose trend over the windows is a diagnostic, with the way each gets worse.
TRENDS = (
    ("moe_rms_err", "max"),
    ("moe_rms_ref", "max"),
    ("moe_nrmse", "max"),
    ("moe_row_nrmse_median", "max"),
    ("recall_mean", "min"),
    ("dropped_weight_mean", "max"),
)


def measure(floor: float) -> Callable[[Window], dict]:
    """A window's measures (``WindowedErrors.windows`` / ``whole``) with the per-row NRMSE floored at ``floor``: the
    output's absolute and relative error (``moe_nrmse``, ``moe_rms_err``, ``moe_rms_ref``), the median / p99 / max
    per-row NRMSE, and the routing: mean recall of the golden's top-k, the golden weight on missed experts (mean,
    p99), the misses and the fraction of them in the top half of the golden's selection (0 without a miss)."""

    def of(w: Window) -> dict:
        sums, recall, rank = w["sums"], w["recall"], w["misses"]["rank"]
        nr = row_nrmse(sums, floor)
        return {
            "moe_nrmse": nrmse(sums),
            "moe_rms_err": rms_err(sums),
            "moe_rms_ref": rms_ref(sums),
            "moe_row_nrmse_median": float(nr.median()),
            "moe_row_nrmse_p99": quantile(nr, 0.99),
            "moe_row_nrmse_max": float(nr.max()),
            "recall_mean": float(recall.mean()),
            "dropped_weight_mean": float(w["dropped"].mean()),
            "dropped_weight_p99": quantile(w["dropped"], 0.99),
            "misses": int(rank.numel()),
            # (no miss: none in the top half, not undefined)
            "miss_top_half_fraction": (
                float((2 * rank < w["misses"]["selected"]).float().mean()) if rank.numel() else 0.0
            ),
        }

    return of


class MoeScorer:
    """A MoE Block's ``BlockScorer`` against golden products ``output`` (the MoE output), ``expert_ids`` (the
    golden's top-k, best first) and ``expert_weights``; tier-1 windows of ``window`` rows."""

    def __init__(self, *, output: str, expert_ids: str, expert_weights: str, window: int) -> None:
        self.output, self.expert_ids, self.expert_weights, self.window = output, expert_ids, expert_weights, window
        self.golden_tensors = (output, expert_ids, expert_weights)
        self.series = WindowedErrors()
        self._top_k = 0

    def score_chunk(self, lo: int, outputs: BlockOutputs, rows: Mapping[str, torch.Tensor]) -> ChunkScore:
        want_ids = rows[self.expert_ids].long()
        order = -torch.arange(want_ids.shape[1], dtype=torch.float32).expand(want_ids.shape)  # (best first)
        r = routing_rows(
            outputs.extras["expert_ids"].long(),
            outputs.extras["expert_weights"],
            want_ids,
            rows[self.expert_weights],
            order,
        )
        sums = row_sums(rows[self.output], outputs.output)
        self.series.append(lo, sums=sums, recall=r["recall"], dropped=r["dropped"])
        self.series.add_events("misses", **{**r["misses"], "row": r["misses"]["row"] + lo})
        self._top_k = want_ids.shape[1]
        violations = []
        if not torch.isfinite(outputs.output).all():
            violations.append("non-finite MoE output")
        if not (torch.sort(outputs.extras["expert_ids"], dim=1).values.diff(dim=1) > 0).all():
            violations.append("a row routed to one expert twice")
        return ChunkScore(violations=violations)

    def score_run(self, carried_state: Mapping[str, torch.Tensor]) -> MetricTiers:
        """The tiers (module docstring); ``carried_state`` is unused (no state)."""
        series = self.series
        sums = series.column("sums")
        of = measure(golden_floor(sums[:, SUM_YY].sqrt()))
        windows = series.windows(self.window, of)
        whole = series.whole(of)
        span = {k: whole.pop(k) for k in ("start", "end", "rows")}
        run = {f"run_{k}": v for k, v in whole.items()}
        headline = {f"run_{k}": run.pop(f"run_{k}") for k in HEADLINE}
        ranks = series.whole(lambda w: {"ranks": w["misses"]["rank"]})["ranks"]
        diagnostics = {
            "run_moe_gain": gain(sums),
            "run_miss_rank_histogram": torch.bincount(ranks, minlength=self._top_k).tolist(),
            **{f"{metric}_trend": trend(windows, metric, worse) for metric, worse in TRENDS},
        }
        return {0: [{**span, **headline}], 1: [{**span, **run}, *windows], 2: [{**span, **diagnostics}]}
