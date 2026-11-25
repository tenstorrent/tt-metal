# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 DSA layer 3 against the FP64 chain golden over long context: the FP64 attention the device is scored
against, which rows are scored, and the per-window measures and bars of ``test_kimi_k3_1_dsa_long_context.py``.

CPU only; imports no device code. The golden (``k31_chain_golden.K31_CHAIN_1M``) holds, for every row, layer 3's
input ``dsa_input_3`` (the AttnRes read), its FP64 keys (``latent``, ``kpe``, ``index_key``) and its FP64 indexer
top-2048 (``topk_indices_3``, best first), but the FP64 attention output ``attn_output_3`` only on 512 sampled rows
(attention is quadratic in the chain). The test therefore recomputes the FP64 gated attention on the rows it scores,
from the golden's keys and selection (``score_rows``), and checks that recompute against the stored output on the sampled
rows before trusting it.

Per scored row, three FP64 attentions over the golden's FP64 keys, each against the reference (FP64 attention over
the golden's top-k):

* the device output itself: its absolute and relative error (``windowed.errors``), gated;
* ``sel``, the selection-only arm: FP64 attention over the device's own top-k. It isolates what the device's
  selection costs, the part of the device error that grows with context (m06n.3); its per-row NRMSE is gated by
  its window median;
* ``tail``, the near-tie baseline: FP64 attention over the golden's top-k without its ``m`` lowest-ranked keys, ``m``
  the device's miss count on that row, i.e. what dropping the cut's tail costs. ``sel`` near ``tail`` reads as misses
  at the cut; ``sel`` far above it as misses of keys that carry weight. Diagnostic.

Also per scored row, the largest attention weight (FP64 softmax over the golden's selection) of one key the device
missed, per head, maximised over heads (``accuracy.dsa.selection_masses``): a per-head outlier diagnostic, not a bar
(the user's provisional default, epic m06n). Per row of the run: top-k recall and the share of misses ranked in the
top half of the golden's selection, diagnostics; any selected future key, an exact causality contract, gated at 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.accuracy.dsa import selection_masses
from models.demos.deepseek_v3_d_p.tests.accuracy.topk import missed_keys, topk_recall
from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import (
    SUM_YY,
    Bar,
    RowSeries,
    Window,
    errors,
    nrmse,
    quantile,
    row_nrmse,
    row_sums,
)

C = KimiK31Config
LAYER = 3  # 0-based: the first DSA layer, the one the FP64 chain golden covers
EPS = 1e-5  # rms_norm_eps of K3.1 (the golden's model.eps)
WINDOW = 16384  # rows per scoring window, aligned to row 0 (m06n.3 resolved the early-context rise at 16K)
ROWS_PER_WINDOW = 256  # scored rows per full window ("a few hundred per window": every-row-scoring note)
FLOOR_FRACTION = 0.01  # per-row NRMSE denominator floor: 1% of the median reference row norm (as the KDA golden)
RECOMPUTE_NRMSE = 1e-6  # the FP64 recompute vs the golden's stored attn_output_3 (measured ~3e-8, m06n.3 review)
ATTEND_BATCH = 8  # query rows per batched FP64 attention (bounds the [rows, k, lora] gather)

# The layer's FP64 weights the attention needs (checkpoint names under ``layers.3.``).
WEIGHTS = (
    "input_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
)


@dataclass(frozen=True)
class Dims:
    """The attention's dimensions: K3.1's by default, smaller ones in the host tests."""

    heads: int = C.NUM_ATTENTION_HEADS
    nope: int = C.QK_NOPE_HEAD_DIM
    rope: int = C.QK_ROPE_HEAD_DIM
    v: int = C.V_HEAD_DIM
    lora: int = C.KV_LORA_RANK
    topk: int = C.INDEX_TOPK
    eps: float = EPS


def load_weights(checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """``WEIGHTS`` of layer 3 from a K3.1 checkpoint in FP64, and the digest of the tensor set the chain read for
    the layer (``Golden.require(weights_sha256=...)`` checks the golden was built from these weights)."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.k31_chain_golden import layer_weights

    tensors, digest = layer_weights(checkpoint, LAYER)
    return {n: tensors[n].double() for n in WEIGHTS}, digest


def rms64(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def normed_input(read: torch.Tensor, w: Mapping[str, torch.Tensor], dims: Dims = Dims()) -> torch.Tensor:
    """``input_layernorm`` of the AttnRes read ``dsa_input_3`` in FP64, as the chain computes it. The device is fed
    this rounded once to bf16; the reference attends from it unrounded."""
    return rms64(read.double(), w["input_layernorm.weight"], dims.eps)


def queries(x: torch.Tensor, w: Mapping[str, torch.Tensor], dims: Dims = Dims()) -> dict[str, torch.Tensor]:
    """FP64 per row of the normed input ``x``: the absorbed query on the latent (``q_nope W_k``, ``[n, heads,
    lora]``), the rope query (``[n, heads, rope]``) and the output gate (``[n, heads * v]``)."""
    x = x.double()
    qr = rms64(x @ w["self_attn.q_a_proj.weight"].T, w["self_attn.q_a_layernorm.weight"], dims.eps)
    q = (qr @ w["self_attn.q_b_proj.weight"].T).reshape(-1, dims.heads, dims.nope + dims.rope)
    kvb = w["self_attn.kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)
    return {
        "q_latent": torch.einsum("nhd,hdl->nhl", q[..., : dims.nope], kvb[:, : dims.nope]),
        "q_pe": q[..., dims.nope :],
        "gate": torch.sigmoid(x @ w["self_attn.g_proj.weight"].T),
    }


def _scores(q: Mapping[str, torch.Tensor], lo: int, latent, kpe, selection: torch.Tensor, dims: Dims):
    """Pre-softmax scores ``[b, heads, k]`` of query rows ``lo + [0, b)`` over their selected keys (-inf where an entry
    selects nothing), and the gathered latent ``[b, k, lora]``."""
    keys = selection.clamp_min(0)
    lat, pe = latent[keys].double(), kpe[keys].double()
    b = selection.shape[0]
    scores = torch.bmm(q["q_latent"][lo : lo + b], lat.transpose(1, 2))
    scores += torch.bmm(q["q_pe"][lo : lo + b], pe.transpose(1, 2))
    scores *= (dims.nope + dims.rope) ** -0.5
    return scores.masked_fill_((selection < 0).unsqueeze(1), -math.inf), lat


def _heads(p: torch.Tensor, lat: torch.Tensor, w: Mapping[str, torch.Tensor], dims: Dims) -> torch.Tensor:
    """Per-head value outputs ``[b, heads * v]`` of softmax weights ``p`` over the gathered latent ``lat``."""
    kvb = w["self_attn.kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)[:, dims.nope :]
    mixed = torch.bmm(p, lat)  # [b, heads, lora]
    return torch.einsum("bhl,hvl->bhv", mixed, kvb).reshape(p.shape[0], -1)


def _output(q: Mapping[str, torch.Tensor], heads: torch.Tensor, w: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Gated ``o_proj`` of per-head outputs ``[n, heads * v]``: the layer's attention output ``[n, hidden]``."""
    return (heads * q["gate"]) @ w["self_attn.o_proj.weight"].T


def device_selection(topk: torch.Tensor) -> torch.Tensor:
    """The device's top-k ``[rows, k]`` as int64 with its uint32 no-key sentinel (0xFFFFFFFF) as -1."""
    topk = topk.long()
    return torch.where((topk < 0) | (topk >= 2**32 - 1), -1, topk)


def future_keys(selection: torch.Tensor, start: int) -> torch.Tensor:
    """Per row ``start + i`` of ``selection``, how many selected keys lie after the row (causality: must be 0)."""
    rows = torch.arange(start, start + selection.shape[0]).unsqueeze(1)
    return (selection > rows).sum(1)


def tail_dropped(want: torch.Tensor, misses: torch.Tensor) -> torch.Tensor:
    """``want`` (best first, -1 past the valid keys) without each row's ``misses`` lowest-ranked valid keys."""
    valid = (want >= 0).sum(1, keepdim=True)
    rank = torch.arange(want.shape[1]).unsqueeze(0)
    return want.masked_fill(rank >= valid - misses.unsqueeze(1), -1)


def scored_rows(rows: int, sampled: torch.Tensor, dims: Dims = Dims(), per_window: int = ROWS_PER_WINDOW):
    """The rows of ``[0, rows)`` the test scores: per ``WINDOW`` window, ``per_window`` rows evenly spread over its
    rows at or past top-k (``per_window`` scaled down for a partial window), plus the golden's sampled rows there,
    sorted and distinct. Rows below top-k select every causal key on both sides (no selection error), so they are
    not scored."""
    picked = [sampled[(sampled >= dims.topk) & (sampled < rows)].long()]
    for start in range(0, rows, WINDOW):
        lo, hi = max(start, dims.topk), min(start + WINDOW, rows)
        if hi <= lo:
            continue
        n = max(1, round(per_window * (hi - lo) / WINDOW))
        picked.append(torch.linspace(lo, hi - 1, min(n, hi - lo)).round().long())
    return torch.cat(picked).unique()


@dataclass
class RowArms:
    """Per scored row: the row sums (``windowed.row_sums``) of the device output, the selection arm and the tail
    baseline against the reference, and the largest missed-key weight."""

    rows: torch.Tensor
    device: torch.Tensor
    sel: torch.Tensor
    tail: torch.Tensor
    missed_key_max: torch.Tensor


def score_rows(
    rows: torch.Tensor,
    normed: torch.Tensor,
    device_out: torch.Tensor,
    device_sel: torch.Tensor,
    want_sel: torch.Tensor,
    latent: torch.Tensor,
    kpe: torch.Tensor,
    w: Mapping[str, torch.Tensor],
    dims: Dims = Dims(),
) -> tuple[RowArms, torch.Tensor]:
    """The arms of the scored ``rows`` (absolute), given per row their FP64 normed input, the device's output and
    selection and the golden's selection; ``latent`` / ``kpe`` hold the golden's keys of rows ``[0, > max(rows)]``.
    Also returns the FP64 reference output ``[rows, hidden]`` (for the recompute check)."""
    q = queries(normed, w, dims)
    ref_heads, sel_heads, tail_heads, missed_key_max = [], [], [], []
    threads = torch.get_num_threads()
    for lo in range(0, rows.shape[0], ATTEND_BATCH):
        want, got = want_sel[lo : lo + ATTEND_BATCH], device_sel[lo : lo + ATTEND_BATCH]
        scores, lat = _scores(q, lo, latent, kpe, want, dims)
        p_ref = scores.softmax(-1)
        sel_scores, sel_lat = _scores(q, lo, latent, kpe, got, dims)
        p_sel = sel_scores.softmax(-1)
        torch.set_num_threads(1)  # per-row bookkeeping on [heads, k]: thread fan-out costs ~80x on a loaded host
        try:
            masses = [
                selection_masses(
                    p_ref[i][:, want[i] >= 0], p_sel[i][:, got[i] >= 0], want[i][want[i] >= 0], got[i][got[i] >= 0]
                )
                for i in range(want.shape[0])
            ]
        finally:
            torch.set_num_threads(threads)
        missed = torch.tensor([m["missed"] for m in masses])
        # the tail baseline drops the reference selection's `missed` lowest-ranked keys: a mask on the same scores
        tail_scores = scores.masked_fill((tail_dropped(want, missed) < 0).unsqueeze(1), -math.inf)
        ref_heads.append(_heads(p_ref, lat, w, dims))
        sel_heads.append(_heads(p_sel, sel_lat, w, dims))
        tail_heads.append(_heads(tail_scores.softmax(-1), lat, w, dims))
        missed_key_max += [m["missed_key_max"] for m in masses]
    ref = _output(q, torch.cat(ref_heads), w)
    arms = RowArms(
        rows=rows,
        device=row_sums(ref, device_out.double()),
        sel=row_sums(ref, _output(q, torch.cat(sel_heads), w)),
        tail=row_sums(ref, _output(q, torch.cat(tail_heads), w)),
        missed_key_max=torch.tensor(missed_key_max, dtype=torch.float64),
    )
    return arms, ref


def floor_of(reference_sums: torch.Tensor) -> float:
    """Per-row NRMSE floor: ``FLOOR_FRACTION`` of the median reference row norm."""
    return FLOOR_FRACTION * float(reference_sums[:, SUM_YY].sqrt().median())


def measure(window: Window, floor: float, dims: Dims = Dims()) -> dict:
    """One window's metrics. Row columns: ``recall``, ``misses_top_half``, ``misses_all``, ``future``; events
    ``scored`` (``RowArms`` fields). Gated: ``attn_nrmse`` / ``attn_rms_err`` (device output, pooled over the
    window's scored rows), ``sel_nrmse_median``, ``future_keys``. The rest are diagnostics. Every window holds scored
    rows (each spans rows past top-k), so no gated metric is left out by the device's data."""
    s = window["scored"]
    if not s["row"].numel():  # unreachable by construction (scored_rows); a silent omission would skip gated bars
        raise ValueError(f"window [{window.start}, {window.end}) has no scored row")
    dev_rows = row_nrmse(s["device"], floor)
    sel_rows = row_nrmse(s["sel"], floor)
    past = slice(max(window.start, dims.topk) - window.start, None)
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
        "recall_mean": float(window["recall"][past].mean()) if window.end > dims.topk else None,
        "miss_top_half_fraction": float(window["misses_top_half"][past].sum() / misses) if misses else 0.0,
        "future_keys": int(window["future"].sum()),
    }


def bars(rung: Mapping[str, float]) -> list[Bar]:
    """The gated bars of one rung: ``rung`` maps ``sel_nrmse_median``, ``attn_nrmse`` and ``attn_rms_err`` to their
    per-window maxima; future keys must be 0 at every rung. A non-finite value (a bar not yet set) raises: ``check``
    would pass it, since no comparison with NaN fails."""
    unset = [metric for metric, value in rung.items() if not math.isfinite(value)]
    if unset:
        raise ValueError(f"no bar value for {unset}")
    return [Bar(metric, "max", value) for metric, value in rung.items()] + [Bar("future_keys", "max", 0)]


CACHE_NAMES = ("kv_latent", "kv_rope", "index_key")


class RunScorer:
    """One device run of rows ``[0, rows)`` scored against the chain golden, chunk by chunk, holding per row only
    the recall columns, the golden's FP64 keys and the scored rows' arms.

    ``chunk_input(lo, hi)`` reads the golden's rows and returns the FP64 normed input (round it to bf16 for the
    device); ``add(lo, out, got)`` scores the device's output ``[n, hidden]`` and top-k ``[n, k]`` of those rows;
    ``finish(kvpe, index)``, with the device's caches ``[rows, lora + rope]`` and ``[rows, index_dim]``, returns
    the per-window metrics and the run's summary."""

    def __init__(self, golden, rows: int, weights: Mapping[str, torch.Tensor], dims: Dims = Dims()):
        from models.demos.deepseek_v3_d_p.tests.kimi_k3.k31_chain_golden import read_sampled

        if rows > golden.rows:
            raise ValueError(f"golden {golden.golden_id} has {golden.rows} rows, fewer than {rows}")
        self.golden, self.rows, self.w, self.dims = golden, rows, weights, dims
        self.sampled = read_sampled(golden, rows)
        self.scored = scored_rows(rows, self.sampled["sampled_rows"], dims)
        index_dim = golden.manifest["products"]["index_key"]["shape"][0]
        self.latent = torch.empty(rows, dims.lora, dtype=torch.float64)
        self.kpe = torch.empty(rows, dims.rope, dtype=torch.float64)
        self.index_key = torch.empty(rows, index_dim, dtype=torch.float64)
        self.series = RowSeries()
        self.arms: list[RowArms] = []
        self.recompute: list[torch.Tensor] = []
        self._chunk: dict | None = None

    def chunk_input(self, lo: int, hi: int) -> torch.Tensor:
        from models.demos.deepseek_v3_d_p.tests.kimi_k3.k31_chain_golden import read_rows

        names = ["dsa_input_3", "latent", "kpe", "index_key", "topk_indices_3", "topk_scores_3"]
        p = read_rows(self.golden, names, lo, hi)
        self.latent[lo:hi], self.kpe[lo:hi], self.index_key[lo:hi] = p["latent"], p["kpe"], p["index_key"]
        normed = normed_input(p["dsa_input_3"], self.w, self.dims)
        self._chunk = {
            "lo": lo,
            "hi": hi,
            "normed": normed,
            "want": p["topk_indices_3"].long(),
            "scores": p["topk_scores_3"].float(),
        }
        return normed

    def add(self, lo: int, out: torch.Tensor, got: torch.Tensor) -> None:
        c = self._chunk
        if c is None or c["lo"] != lo or out.shape[0] != c["hi"] - lo or got.shape[0] != c["hi"] - lo:
            raise ValueError(f"add({lo}, {out.shape[0]} rows) does not match the last chunk_input")
        if not torch.isfinite(out).all():
            raise AssertionError(f"non-finite device attention output in rows [{lo}, {c['hi']})")
        n, want = c["hi"] - lo, c["want"]
        miss = missed_keys(got, want, c["scores"])
        self.series.append(
            lo,
            recall=topk_recall(got, want),
            misses_all=torch.bincount(miss["row"], minlength=n),
            misses_top_half=torch.bincount(miss["row"][2 * miss["rank"] < miss["selected"]], minlength=n),
            future=future_keys(got, lo),
        )
        inside = self.scored[(self.scored >= lo) & (self.scored < c["hi"])]
        if inside.numel():
            local = inside - lo
            arms, ref = score_rows(
                inside,
                c["normed"][local],
                out[local],
                got[local],
                want[local],
                self.latent,
                self.kpe,
                self.w,
                self.dims,
            )
            self.arms.append(arms)
            at = torch.isin(self.sampled["sampled_rows"], inside)
            if at.any():
                self.recompute.append(
                    row_sums(self.sampled["attn_output_3"][at], ref[torch.isin(inside, self.sampled["sampled_rows"])])
                )
        self._chunk = None

    def finish(self, kvpe: torch.Tensor, index: torch.Tensor) -> tuple[list[dict], dict]:
        """Per-window metrics (``measure`` and the caches' errors) and the summary (``scored_rows``,
        ``recompute_nrmse``: the FP64 recompute against the golden's stored ``attn_output_3`` on its sampled rows,
        ``nrmse_floor``). Raises when the series does not cover every row or the recompute is unchecked or off."""
        if self.series.end != self.rows:
            raise ValueError(f"scored rows [0, {self.series.end}) of {self.rows}")
        if not self.recompute:
            raise AssertionError("no golden sampled row was scored: the FP64 recompute is unchecked")
        recompute_nrmse = nrmse(torch.cat(self.recompute))
        if not recompute_nrmse <= RECOMPUTE_NRMSE:
            raise AssertionError(
                f"FP64 attention recompute differs from the golden's attn_output_3: NRMSE {recompute_nrmse:.3g} > "
                f"{RECOMPUTE_NRMSE}"
            )
        fields = ("rows", "device", "sel", "tail", "missed_key_max")
        arms = {f: torch.cat([getattr(a, f) for a in self.arms]) for f in fields}
        floor = floor_of(arms["device"])  # every arm's sums hold the same reference row norms
        self.series.add_events("scored", row=arms.pop("rows"), **arms)
        lora, kvpe, index = self.dims.lora, kvpe.double(), index.double()
        caches = RowSeries()
        caches.append(
            0,
            kv_latent=row_sums(self.latent, kvpe[:, :lora]),
            kv_rope=row_sums(self.kpe, kvpe[:, lora:]),
            index_key=row_sums(self.index_key, index),
        )
        windows = self.series.windows(WINDOW, lambda win: measure(win, floor, self.dims))
        cache_windows = caches.windows(
            WINDOW, lambda win: {k: v for n in CACHE_NAMES for k, v in errors(win[n], f"{n}_").items()}
        )
        summary = {"scored_rows": int(self.scored.numel()), "recompute_nrmse": recompute_nrmse, "nrmse_floor": floor}
        return [{**a, **b} for a, b in zip(windows, cache_windows)], summary
