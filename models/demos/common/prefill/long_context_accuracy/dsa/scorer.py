# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``DsaScorer``: a DSA (indexer top-k + sparse MLA) layer's device results against its FP64 golden, chunk by chunk,
then over the run in metric tiers (``block.BlockScorer``).

What it reads. Per row, from the golden (``GoldenNames``): the layer input, its FP64 keys (latent, rope part, index
key) and its FP64 indexer selection (best first, -1 past the causal keys) with its scores; on the golden's sampled rows
only, the FP64 attention output. From the device (``block.BlockOutputs``): ``output`` the attention output ``[n,
hidden]`` and ``extras["topk"]`` its selection ``[n, topk]`` (int64, -1 for no key); after the run, ``carried_state``
``kv_cache`` ``[rows, lora + rope]`` and ``index_cache`` ``[rows, index_dim]``.

Which rows are scored. Per ``WINDOW`` window (aligned to row 0), ``ROWS_PER_WINDOW`` rows evenly spread past top-k
plus the golden's sampled rows there (``scored_rows``): on each, the FP64 attention recompute
(``fp64_attention.score_rows``): the device output's error, the selection-only arm, the near-tie tail baseline and
the largest missed-key weight. The recompute must match the golden's stored attention on its sampled rows
(``RECOMPUTE_NRMSE``) before any of it is trusted. Every row: top-k recall, miss ranks, future keys. After the run,
the caches against the golden's FP64 keys.

Metric tiers (``score_run``; a whole-run metric whose name a window also uses carries the prefix ``run_``):

* 0 headline, over the run's scored rows: ``run_attn_nrmse`` (device attention output) and ``run_sel_nrmse``
  (selection arm);
* 1 tails, per window: attention ``attn_nrmse``, ``attn_rms_err``, ``attn_rms_ref``, ``attn_nrmse_median``; the
  selection arm's per-row ``sel_nrmse_median`` and ``sel_nrmse_p90``; ``recall_mean``, ``future_keys`` and the
  window's ``scored_rows``;
* 2 diagnostics, per window: ``attn_pcc``, the selection arm's pooled ``sel_*`` errors, ``tail_nrmse_median``,
  ``missed_key_max_median`` / ``_max`` (per-head outlier weight of one missed key), ``miss_top_half_fraction`` and the
  caches' errors (``kv_latent_*``, ``kv_rope_*``, ``index_key_*``); over the run: ``attn_gain`` (least-squares gain of
  the device output on the reference), ``run_scored_rows``, ``recompute_nrmse`` and ``nrmse_floor`` (the per-row
  NRMSE floor, ``windowed_errors.golden_floor`` of the reference row norms).

Per chunk (``check_chunk``): the output is finite and no row selects a key after itself (causality); DSA has no
per-chunk bars (its bars are per window).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar
from models.demos.common.prefill.long_context_accuracy.block import BlockOutputs, MetricTiers
from models.demos.common.prefill.long_context_accuracy.dsa.fp64_attention import Dims, RowArms, normed_input, score_rows
from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden
from models.demos.common.prefill.long_context_accuracy.topk_selection import missed_keys, topk_hits, topk_recall
from models.demos.common.prefill.long_context_accuracy.windowed_errors import (
    SUM_YY,
    Window,
    WindowedErrors,
    errors,
    gain,
    golden_floor,
    nrmse,
    quantile,
    row_nrmse,
    row_sums,
)

WINDOW = 16384  # rows per scoring window, aligned to row 0 (m06n.3 resolved the early-context rise at 16K)
ROWS_PER_WINDOW = 256  # scored rows per full window ("a few hundred per window": every-row-scoring note)
RECOMPUTE_NRMSE = 1e-6  # the FP64 recompute vs the golden's stored attention (measured ~3e-8, m06n.3 review)
CACHE_NAMES = ("kv_latent", "kv_rope", "index_key")

# Per-window metrics of metric tier 1; every other per-window metric is tier 2.
WINDOW_TIER_1 = frozenset(
    {
        "scored_rows",
        "attn_nrmse",
        "attn_rms_err",
        "attn_rms_ref",
        "attn_nrmse_median",
        "sel_nrmse_median",
        "sel_nrmse_p90",
        "recall_mean",
        "future_keys",
    }
)
_SPAN = ("start", "end", "rows")


@dataclass(frozen=True)
class GoldenNames:
    """The golden's product names of the layer: per row its ``input`` (before ``input_layernorm``), its keys
    ``latent``, ``kpe``, ``index_key``, its ``selection`` and ``selection_scores``; on sampled rows ``sampled_rows``
    and the attention ``sampled_output``."""

    input: str
    latent: str
    kpe: str
    index_key: str
    selection: str
    selection_scores: str
    sampled_rows: str
    sampled_output: str

    @property
    def per_row(self) -> tuple[str, ...]:
        return (self.input, self.latent, self.kpe, self.index_key, self.selection, self.selection_scores)


def future_keys(selection: torch.Tensor, start: int) -> torch.Tensor:
    """Per row ``start + i`` of ``selection``, how many selected keys lie after the row (causality: must be 0)."""
    rows = torch.arange(start, start + selection.shape[0]).unsqueeze(1)
    return (selection > rows).sum(1)


def scored_rows(rows: int, sampled: torch.Tensor, topk: int, per_window: int = ROWS_PER_WINDOW) -> torch.Tensor:
    """The rows of ``[0, rows)`` that are scored: per ``WINDOW`` window, ``per_window`` rows evenly spread over its
    rows at or past ``topk`` (``per_window`` scaled down for a partial window), plus the golden's sampled rows there,
    sorted and distinct. Rows below ``topk`` select every causal key on both sides (no selection error), so they are
    not scored."""
    picked = [sampled[(sampled >= topk) & (sampled < rows)].long()]
    for start in range(0, rows, WINDOW):
        lo, hi = max(start, topk), min(start + WINDOW, rows)
        if hi <= lo:
            continue
        n = max(1, round(per_window * (hi - lo) / WINDOW))
        picked.append(torch.linspace(lo, hi - 1, min(n, hi - lo)).round().long())
    return torch.cat(picked).unique()


def measure(window: Window, floor: float, topk: int) -> dict:
    """One window's metrics of the scored rows (event ``scored``, ``RowArms`` fields) and the row columns
    ``recall``, ``misses_top_half``, ``misses_all``, ``future``. Every window holds scored rows (each spans rows past
    top-k), so no gated metric is left out by the device's data."""
    s = window["scored"]
    if not s["row"].numel():  # unreachable by construction (scored_rows); a silent omission would skip gated bars
        raise ValueError(f"window [{window.start}, {window.end}) has no scored row")
    dev_rows = row_nrmse(s["device"], floor)
    sel_rows = row_nrmse(s["sel"], floor)
    past = slice(max(window.start, topk) - window.start, None)
    misses = window["misses_all"][past].sum()
    return {
        "scored_rows": int(s["row"].numel()),
        **errors(s["device"], "attn_"),
        "attn_nrmse_median": float(dev_rows.median()),
        **{f"sel_{k}": v for k, v in errors(s["sel"]).items()},
        "sel_nrmse_median": float(sel_rows.median()),
        "sel_nrmse_p90": quantile(sel_rows, 0.9),
        "tail_nrmse_median": float(row_nrmse(s["tail"], floor).median()),
        "missed_key_max_median": float(s["missed_key_max"].median()),
        "missed_key_max_max": float(s["missed_key_max"].max()),
        "recall_mean": float(window["recall"][past].mean()) if window.end > topk else None,
        "miss_top_half_fraction": float(window["misses_top_half"][past].sum() / misses) if misses else 0.0,
        "future_keys": int(window["future"].sum()),
    }


class DsaScorer:
    """One device run of rows ``[0, rows)`` of a DSA layer scored against its golden, holding per row only the recall
    columns and the golden's FP64 keys, and the scored rows' arms. Reads the golden's sampled rows below ``rows``
    when built. ``weights``: the layer's FP64 ``fp64_attention.WEIGHTS``."""

    def __init__(
        self, golden: Fp64Golden, rows: int, names: GoldenNames, weights: Mapping[str, torch.Tensor], dims: Dims
    ):
        if rows > golden.rows:
            raise ValueError(f"golden {golden.golden_id} has {golden.rows} rows, fewer than {rows}")
        self.rows, self.names, self.w, self.dims = rows, names, weights, dims
        self.golden_tensors = names.per_row
        sampled = golden.read_sampled(rows)
        self.sampled_rows, self.sampled_output = sampled[names.sampled_rows], sampled[names.sampled_output]
        self.scored = scored_rows(rows, self.sampled_rows, dims.topk)
        index_dim = golden.manifest["products"][names.index_key]["shape"][0]
        self.latent = torch.empty(rows, dims.lora, dtype=torch.float64)
        self.kpe = torch.empty(rows, dims.rope, dtype=torch.float64)
        self.index_key = torch.empty(rows, index_dim, dtype=torch.float64)
        self.series = WindowedErrors()
        self.arms: list[RowArms] = []
        self.recompute: list[torch.Tensor] = []

    def device_input(self, rows: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The device input of the golden ``rows``: the FP64 ``input_layernorm`` of the layer input, rounded once to
        bf16."""
        return normed_input(rows[self.names.input], self.w, self.dims).to(torch.bfloat16)

    def score_chunk(self, lo: int, outputs: BlockOutputs, rows: Mapping[str, torch.Tensor]) -> None:
        names, out, got = self.names, outputs.output, outputs.extras["topk"]
        n = rows[names.input].shape[0]
        hi = lo + n
        if out.shape[0] != n or got.shape[0] != n:
            raise ValueError(f"device rows {out.shape[0]} / {got.shape[0]} differ from the golden's {n} at row {lo}")
        self.latent[lo:hi], self.kpe[lo:hi] = rows[names.latent], rows[names.kpe]
        self.index_key[lo:hi] = rows[names.index_key]
        want, scores = rows[names.selection].long(), rows[names.selection_scores].float()
        hits = topk_hits(got, want)  # one membership per chunk: recall, misses and the scored rows' missed keys
        miss = missed_keys(hits, want, scores)
        self.series.append(
            lo,
            recall=topk_recall(hits, want),
            misses_all=torch.bincount(miss["row"], minlength=n),
            misses_top_half=torch.bincount(miss["row"][2 * miss["rank"] < miss["selected"]], minlength=n),
            future=future_keys(got, lo),
        )
        inside = self.scored[(self.scored >= lo) & (self.scored < hi)]
        if not inside.numel():
            return
        local = inside - lo
        normed = normed_input(rows[names.input], self.w, self.dims)  # (the chunk's, as its device input)
        arms, ref = score_rows(
            inside,
            normed[local],
            out[local],
            got[local],
            want[local],
            ~hits[local] & (want[local] >= 0),
            self.latent,
            self.kpe,
            self.w,
            self.dims,
        )
        self.arms.append(arms)
        at = torch.isin(self.sampled_rows, inside)
        if at.any():
            self.recompute.append(row_sums(self.sampled_output[at], ref[torch.isin(inside, self.sampled_rows)]))

    def check_chunk(
        self, lo: int, outputs: BlockOutputs, rows: Mapping[str, torch.Tensor], bars: Sequence[AcceptanceBar]
    ) -> list[str]:
        if bars:
            raise ValueError(f"DSA has no per-chunk bars (its bars are per {WINDOW}-row window), got {list(bars)}")
        violations = []
        if not torch.isfinite(outputs.output).all():
            violations.append("non-finite device attention output")
        future = future_keys(outputs.extras["topk"], lo)
        if future.any():
            first = lo + int(future.nonzero()[0])
            violations.append(f"{int(future.sum())} selected keys lie after their row (first on row {first})")
        return violations

    def score_run(self, carried_state: Mapping[str, torch.Tensor], golden: Fp64Golden) -> MetricTiers:
        """Raises when the series does not cover every row, or when the FP64 recompute is unchecked (no sampled row
        scored) or off the golden's stored attention by more than ``RECOMPUTE_NRMSE``."""
        if self.series.end != self.rows:
            raise ValueError(f"scored rows [0, {self.series.end}) of {self.rows}")
        if not self.recompute:
            raise AssertionError("no golden sampled row was scored: the FP64 recompute is unchecked")
        recompute_nrmse = nrmse(torch.cat(self.recompute))
        if not recompute_nrmse <= RECOMPUTE_NRMSE:
            raise AssertionError(
                f"FP64 attention recompute differs from the golden's stored attention: NRMSE {recompute_nrmse:.3g} > "
                f"{RECOMPUTE_NRMSE}"
            )
        fields = ("rows", "device", "sel", "tail", "missed_key_max")
        arms = {f: torch.cat([getattr(a, f) for a in self.arms]) for f in fields}
        floor = golden_floor(arms["device"][:, SUM_YY].sqrt())  # every arm's sums hold the same reference row norms
        self.series.add_events("scored", row=arms["rows"], **{f: arms[f] for f in fields[1:]})
        lora, kv, index = self.dims.lora, carried_state["kv_cache"].double(), carried_state["index_cache"].double()
        caches = WindowedErrors()
        caches.append(
            0,
            kv_latent=row_sums(self.latent, kv[:, :lora]),
            kv_rope=row_sums(self.kpe, kv[:, lora:]),
            index_key=row_sums(self.index_key, index),
        )
        windows = self.series.windows(WINDOW, lambda win: measure(win, floor, self.dims.topk))
        cache_windows = caches.windows(
            WINDOW, lambda win: {k: v for n in CACHE_NAMES for k, v in errors(win[n], f"{n}_").items()}
        )
        tiers: MetricTiers = {0: [], 1: [], 2: []}
        for window, cache in zip(windows, cache_windows):
            span = {k: window[k] for k in _SPAN}
            metrics = {k: v for k, v in {**window, **cache}.items() if k not in _SPAN}
            tiers[1].append({**span, **{k: v for k, v in metrics.items() if k in WINDOW_TIER_1}})
            tiers[2].append({**span, **{k: v for k, v in metrics.items() if k not in WINDOW_TIER_1}})
        tiers[0].append(
            self.series.whole(
                lambda win: {"run_attn_nrmse": nrmse(arms["device"]), "run_sel_nrmse": nrmse(arms["sel"])}
            )
        )
        tiers[2].append(
            self.series.whole(
                lambda win: {
                    "attn_gain": gain(arms["device"]),
                    "run_scored_rows": int(self.scored.numel()),
                    "recompute_nrmse": recompute_nrmse,
                    "nrmse_floor": floor,
                }
            )
        )
        return tiers
