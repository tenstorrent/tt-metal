# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 MoE layers teacher-fed from a vLLM trace: host input, host scoring, error vs context position.

The three steps of ``teacher_feed.py`` for one MoE layer (latent MoE, 896 routed experts, top-16, two shared experts):

1. ``input``: the MoE's input, ``post_attention_layernorm`` of the layer's pre-FFN AttnRes read (``mlp_res``) over
   the sealed block snapshots and the live in-block sum including the layer's own attention output, rebuilt from the
   trace by the tracer's replay rule (or read from a captured stream with ``--input-stream``). Every segment is
   checked by routing sampled rows with the layer's gate in fp32 against the trace's ``expert_ids`` (the tracer
   recomputes the gate from the same input, ``routing_capture: gate_recompute``), and fails closed below
   ``INPUT_CHECK_RECALL``.
2. The device test (``test_moe_teacher_feed.py``) runs the production MoE (the MoE block, K3.1's default, from the
   stage's weight cache) on every 2560-token chunk of the prefix and saves each chunk's MoE output and the device's
   routing (expert ids, weights).
3. ``score``: per window of context positions, the MoE output's per-row NRMSE against ``decoder_output_layer_L``
   (the tracer's FFN output alone), and the routing three ways - device, trace and an fp32 host router on the same
   input - as recall of each other's top-16 sets, the rank of every expert the device misses in the trace's
   selection (ordered by the fp32 router's choice score, the order the selection uses), and the trace routing
   weight the device drops. ``host`` recomputes the whole MoE in fp32 on sampled rows, so the MoE output's three
   distances (device / trace / fp32) are reported side by side.

    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed input --trace T --layer L --out IN
    TT_MOE_TF_INPUT=IN TT_MOE_TF_OUT=OUT scripts/run_safe_pytest.sh ...test_moe_teacher_feed.py::test_...
    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed score --trace T --out OUT
    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed host --trace T --out OUT
    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed same OUT OUT_REVERSED

An attribution run swaps the routed experts' weight format (bfloat4_b in production) for bfloat8_b: ``routed-cache``
writes a layer's bfp8 routed experts into a private cache (on a mock mesh, no device), and the device test reads them
with ``TT_MOE_TF_ROUTED_BFP8``:

    TT_METAL_MOCK_CLUSTER_DESC_PATH=.../blackhole_galaxy.yaml python -m ...moe_teacher_feed routed-cache --layer L \\
        --cache PRIVATE
    TT_MOE_TF_ROUTED_BFP8=PRIVATE TT_MOE_TF_INPUT=IN TT_MOE_TF_OUT=OUT8 scripts/run_safe_pytest.sh ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_attn_res
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import TokenErrorAccumulator, golden_floor
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import (
    EPS,
    FFN,
    Trace,
    build_input,
    check_bars,
    chunk_path,
    complete_run,
    load_input,
    missed_keys,
    replay_live_and_sealed,
    replay_missing,
    rms_norm,
    topk_recall,
)

TOP_K = KimiK3Config.NUM_EXPERTS_PER_TOKEN  # 16
NUM_EXPERTS = KimiK3Config.NUM_ROUTED_EXPERTS  # 896
CHECK_ROWS = 64  # rows per segment the input check routes
# The rebuilt input, routed in fp32, must recall the trace's top-16 sets to this mean: measured 0.993-0.996 per layer
# (layers 1/49/85/92, 10K and 128K, every segment >= 0.988), against 0.04-0.34 for the decoder_input alias. Not 1:
# the trace's routing is the tracer's own gate recompute, whose numerics differ slightly from an fp32 router.
INPUT_CHECK_RECALL = 0.98

# Stability bars, every one held per window of context positions: PROPOSED, pending the owner's decision. Measured
# on 9 layers (1/13/25/37/49/61/73/85/92) at 128K and 4 (1/49/85/92) at 10K, 4x4-sp1, against the vLLM trace
# (window medians):
# * MoE output per-row NRMSE against the trace <= 0.20 (measured 0.068-0.185 at 128K, the worst layer 73; the layer
#   sets the level). On sampled rows the device reads the same distance against an fp32 host recompute of the
#   reference MoE, and the trace reads 0.004-0.007 against it: the device's MoE is 14-25x further from fp32 than the
#   GPU's. The bar gates stability, not that precision gap. And no window more than 15% above the first: measured
#   <= 1.114 at 128K (layers 85 / 92), but 1.152 at 10K layer 92 in 2K windows, which FAILS. The MoE carries no state
#   across chunks (a reverse-order run is bit-identical), so a trend over position is the inputs'; whether to gate
#   growth or the reverse-order identity is the owner's decision.
# * routing: mean recall of the trace's top-16 >= 0.98 (measured >= 0.991); misses are near-ties: <= 1% of the
#   trace experts the device misses rank in the top half of the trace's selection (measured <= 0.06%; most at rank
#   15); trace routing weight the device drops, mean <= 0.01 (measured 0.002-0.006).
BARS = {
    "moe_nrmse_median": ("max", 0.20),
    "recall_vs_trace_mean": ("min", 0.98),
    "miss_top_half_fraction": ("max", 0.01),
    "dropped_trace_weight_mean": ("max", 0.01),
}
NRMSE_GROWTH = 1.15


def layer_weights(checkpoint: Path, layer: int) -> dict[str, torch.Tensor]:
    """The host weights the input and the router need: the pre-FFN AttnRes query, post_attention_layernorm, the gate
    and its selection bias."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import checkpoint_prefix, load_checkpoint_tensors

    root = f"{checkpoint_prefix(checkpoint)}layers.{layer}."
    names = [
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
        "post_attention_layernorm.weight",
        "block_sparse_moe.gate.weight",
        "block_sparse_moe.gate.e_score_correction_bias",
    ]
    tensors = load_checkpoint_tensors(checkpoint, [root + n for n in names])
    return {n: tensors[root + n] for n in names}


def route(
    x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, top_k: int = TOP_K
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The model reference's router (`modeling_kimi_moe.KimiMoEGate`: sigmoid scores, selection by score + bias,
    top-k, weights the selected scores renormalized, x ROUTE_SCALE) in fp32, the selection sorted best first:
    (ids ``[n, k]`` int64, weights ``[n, k]``, choice scores ``[n, experts]``, the order the selection ranks by)."""
    scores = (x.float() @ gate.float().T).sigmoid()
    choice = scores + bias.float()
    ids = choice.topk(top_k, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * KimiK3Config.ROUTE_SCALE
    return ids, weights, choice


def check_input(normed: torch.Tensor, trace: Trace, start: int, w: dict, layer: int) -> dict[str, float]:
    """The trace's top-16 sets recalled by the fp32 router on ``CHECK_ROWS`` sampled rows of ``normed``: mean recall
    and the fraction of rows whose set is identical."""
    rows = torch.linspace(0, normed.shape[0] - 1, min(CHECK_ROWS, normed.shape[0])).long()
    ids, _, _ = route(
        normed[rows], w["block_sparse_moe.gate.weight"], w["block_sparse_moe.gate.e_score_correction_bias"]
    )
    want = torch.cat([trace.rows(f"expert_ids_layer_{layer}", start + r, start + r + 1) for r in rows.tolist()])
    recall = topk_recall(ids, want.long())
    return {"router_recall": float(recall.mean()), "router_set_equal": float((recall == 1).float().mean())}


def build_moe_input(
    *,
    trace: Trace,
    checkpoint: Path,
    layer: int,
    count: int,
    out: Path,
    segment: int,
    input_stream: str | None = None,
    input_normed: bool = False,
) -> dict:
    """Write (or resume) layer ``layer``'s teacher-fed MoE input ``post_attention_layernorm(mlp_res read)``: replayed,
    or from ``input_stream``, a captured stream holding the read's output (``input_normed``: the norm's output, the
    gate's input, as is)."""
    assert KimiK3Config.NUM_DENSE_LAYERS <= layer < KimiK3Config.NUM_LAYERS, f"layer {layer} has no MoE"
    if input_stream is None:
        missing = replay_missing(trace, layer, FFN)
        if missing:
            raise ValueError(
                f"{trace.path.name} cannot rebuild layer {layer}'s MoE input: no attn_output/decoder_output for "
                f"{len(missing)} layers ({missing[:8]}...); pass --input-stream with a captured pre-FFN read"
            )
    elif not trace.has(input_stream):
        raise ValueError(f"{trace.path.name} has no stream {input_stream}")
    w = layer_weights(checkpoint, layer)

    def make(start: int, end: int) -> tuple[torch.Tensor, dict[str, float]]:
        if input_stream is None:
            live, sealed = replay_live_and_sealed(trace, start, end, layer, FFN)
            read = hf_attn_res(live, sealed, w["mlp_res_norm.weight"], w["mlp_res_proj.weight"], EPS)
        else:
            read = trace.rows(input_stream, start, end)
        normed = read if input_normed else rms_norm(read, w["post_attention_layernorm.weight"])
        return normed, check_input(normed, trace, start, w, layer)

    expected = {
        "layer": layer,
        "trace": str(trace.path),
        "trace_commit": trace.metadata.get("trace_commit"),
        "input": f"stream {input_stream}" if input_stream else "attn_res replay to the FFN site (bf16, tracer order)",
        "norm": "captured" if input_normed else "KimiRMSNorm bf16 (normalize fp32, round, scale)",
        "checkpoint": str(checkpoint),
    }
    return build_input(
        out=out,
        expected=expected,
        count=count,
        segment=segment,
        make=make,
        check_bars={"router_recall": INPUT_CHECK_RECALL},
    )


ROUTED_CACHE_MANIFEST = "routed-cache.json"


def build_routed_cache(*, checkpoint: Path, layer: int, cache: Path, dtype_name: str) -> dict:
    """Write layer ``layer``'s routed experts at ``dtype_name`` (e.g. ``BFLOAT8_B``) into a private cache ``cache`` for
    the 4x4-sp1 stage, with the production writer (``TtRoutedExpert.build_ttnn_cache``: the same expert placement,
    W^T orientation and tensorbin names as the stage's shared cache, only the dtype differs), on a MOCK 4x4 mesh
    (``$TT_METAL_MOCK_CLUSTER_DESC_PATH``: no device is opened). The device test reads it with
    ``TT_MOE_TF_ROUTED_BFP8``. Refuses the production cache roots and any directory holding a layer marker (a stage
    cache): the shared caches are read-only. Records the layer in ``routed-cache.json``."""
    import ttnn
    from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import provenance
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import _DEFAULT_CACHE, load_routed_expert_weights
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

    assert os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"), "routed-cache writes on a mock mesh only"
    dtype = getattr(ttnn.DataType, dtype_name)
    cache = cache.resolve()
    adapter = k31_stage.ADAPTER
    for root in (os.environ.get(adapter.ttnn_cache_env) or adapter.ttnn_cache_default, _DEFAULT_CACHE):
        root = Path(root).resolve()
        assert cache != root and root not in cache.parents, f"{cache} is inside the shared cache {root}"
    cache.mkdir(parents=True, exist_ok=True)
    assert not any(cache.glob(".layer_*.complete")), f"{cache} holds a stage cache's layer markers"
    shape = k31_stage.MESH_4X4
    sp_axis = k31_stage.STAGE_LAYOUTS[shape].sp_axis
    began = time.perf_counter()
    weights = load_routed_expert_weights(checkpoint, layer, NUM_EXPERTS)
    loaded = time.perf_counter()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    try:
        TtRoutedExpert.build_ttnn_cache(
            weights,
            NUM_EXPERTS // (shape[0] * shape[1]),
            mesh,
            dtype,
            cache,
            f"layer_{layer}.routed_expert",
            sp_axis=sp_axis,
        )
    finally:
        ttnn.close_mesh_device(mesh)
    files = sorted(cache.glob(f"layer_{layer}.routed_expert.local_*_dtype_{dtype_name}_layout_TILE.tensorbin"))
    entry = {
        "layer": layer,
        "dtype": dtype_name,
        "mesh": list(shape),
        "sp_axis": sp_axis,
        "checkpoint": str(checkpoint),
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "load_seconds": round(loaded - began, 1),
        "write_seconds": round(time.perf_counter() - loaded, 1),
        **provenance(),
    }
    record = cache / ROUTED_CACHE_MANIFEST
    layers = json.loads(record.read_text()) if record.is_file() else {}
    layers[str(layer)] = entry
    record.write_text(json.dumps(layers, indent=1))
    return entry


def owner_positions(
    sp_row: int, tp_col: int, *, rows_per_sp_row: int, pad_rows: int, row_blocks: int, tp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the MoE block chip (``sp_row``, ``tp_col``)'s owner rows (its routing rows, in order) sit in the chunk:
    (index into the chip's owner rows, chunk row) for every real row it owns. SP row r holds chunk rows
    [r * rows_per_sp_row, (r + 1) * rows_per_sp_row) plus ``pad_rows`` block padding rows at its end; in each of the
    ``row_blocks`` blocks of that SP row, TP column p owns the p-th of ``tp`` equal pieces
    (``test_moe_block_padding._owner_rows``)."""
    sub = (rows_per_sp_row + pad_rows) // row_blocks
    piece = sub // tp
    pos = torch.cat([j * sub + tp_col * piece + torch.arange(piece) for j in range(row_blocks)])
    owned = torch.arange(pos.numel())
    keep = pos < rows_per_sp_row
    return owned[keep], sp_row * rows_per_sp_row + pos[keep]


# --- scoring -------------------------------------------------------------------------------------------------


def routing_window(got, got_w, want, want_w, choice, num_experts: int = NUM_EXPERTS) -> dict[str, torch.Tensor]:
    """Per row, the device routing (``got`` ids / ``got_w`` weights) against the trace's (``want`` / ``want_w``):
    ``recall`` of the trace's set, the trace weight on the experts the device misses (``dropped``), the largest
    weight difference on the experts both pick (``weight_err``), and every missed expert (``missed_keys``: its rank
    in the trace's selection ordered by ``choice``, the fp32 router's choice score of each trace expert)."""
    recall = topk_recall(got, want)
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    missed = ~torch.isin(rows + want, rows + got)
    dropped = (want_w.float() * missed).sum(dim=1)
    both = torch.zeros(want.shape[0], num_experts).scatter_(1, want, want_w.float())
    dev = torch.zeros(want.shape[0], num_experts).scatter_(1, got, got_w.float())
    shared = (both != 0) & (dev != 0)
    weight_err = ((both - dev).abs() * shared).amax(dim=1)
    return {"recall": recall, "dropped": dropped, "weight_err": weight_err, "misses": missed_keys(got, want, choice)}


def _q(t: torch.Tensor, q: float) -> float:
    return float(t.float().quantile(q)) if t.numel() else float("nan")


def score(*, trace: Trace, out: Path, window: int, gate: torch.Tensor, bias: torch.Tensor) -> dict:
    """Score a complete device run in ``out`` against the trace per context position; writes metrics.json,
    windows.tsv and rows.pt. Routes every row of the run's input with the fp32 router too (``gate`` / ``bias``: the
    layer's gate weight and selection bias), for the trace's own router noise and the device's distance to fp32,
    beside the device's distance to the trace."""
    manifest = complete_run(out)
    count, chunk, layer = manifest["count"], manifest["chunk"], manifest["layer"]
    source = Path(manifest["input"])
    gate = gate.float()
    num_chunks = -(-count // chunk)
    moe = TokenErrorAccumulator()
    per_row = {k: [] for k in ("recall", "dropped", "weight_err", "recall_trace_vs_fp32", "recall_device_vs_fp32")}
    misses = []
    for c in range(num_chunks):
        payload = torch.load(chunk_path(out, c), weights_only=True)
        lo, hi = payload["start"], payload["end"]
        moe.update(trace.rows(f"decoder_output_layer_{layer}", lo, hi), payload["moe_out"])
        want = trace.rows(f"expert_ids_layer_{layer}", lo, hi).long()
        want_w = trace.rows(f"expert_weights_layer_{layer}", lo, hi)
        got, got_w = payload["expert_ids"].long(), payload["expert_weights"]
        host_ids, _, choice = route(load_input(source, lo, hi), gate, bias)
        r = routing_window(got, got_w, want, want_w, choice.gather(1, want))
        for k in ("recall", "dropped", "weight_err"):
            per_row[k].append(r[k])
        per_row["recall_trace_vs_fp32"].append(topk_recall(host_ids, want))
        per_row["recall_device_vs_fp32"].append(topk_recall(got, host_ids))
        r["misses"]["row"] = r["misses"]["row"] + lo
        misses.append(r["misses"])
        print(f"{time.strftime('%H:%M:%S')} scored chunk {c + 1}/{num_chunks} rows [{lo}, {hi})", flush=True)
    floor = golden_floor(moe.expected_norms())
    row_nrmse = moe.token_nrmse(floor)
    per_row = {k: torch.cat(v) for k, v in per_row.items()}
    misses = {k: torch.cat([m[k] for m in misses]) for k in misses[0]}
    windows = []
    for lo in range(0, count, window):
        hi = min(lo + window, count)
        inside = (misses["row"] >= lo) & (misses["row"] < hi)
        rank, selected = misses["rank"][inside], misses["selected"][inside]
        recall = per_row["recall"][lo:hi]
        windows.append(
            {
                "start": lo,
                "end": hi,
                "moe_nrmse_median": float(row_nrmse[lo:hi].median()),
                "moe_nrmse_p99": _q(row_nrmse[lo:hi], 0.99),
                "moe_nrmse_max": float(row_nrmse[lo:hi].max()),
                "recall_vs_trace_mean": float(recall.mean()),
                "set_equal_vs_trace": float((recall == 1).float().mean()),
                "recall_trace_vs_fp32_mean": float(per_row["recall_trace_vs_fp32"][lo:hi].mean()),
                "recall_device_vs_fp32_mean": float(per_row["recall_device_vs_fp32"][lo:hi].mean()),
                "misses": int(rank.numel()),
                "miss_rank_median": _q(rank, 0.5),
                "miss_rank_min": int(rank.min()) if rank.numel() else -1,
                "miss_top_half_fraction": float((2 * rank < selected).float().mean()) if rank.numel() else float("nan"),
                "dropped_trace_weight_mean": float(per_row["dropped"][lo:hi].mean()),
                "dropped_trace_weight_p99": _q(per_row["dropped"][lo:hi], 0.99),
                "weight_err_median": float(per_row["weight_err"][lo:hi].median()),
                "weight_err_p99": _q(per_row["weight_err"][lo:hi], 0.99),
            }
        )
    rank = misses["rank"]
    result = {
        "layer": layer,
        "count": count,
        "window": window,
        "trace": str(trace.path),
        "run": manifest,
        "moe_nrmse_floor": floor,
        "overall": {
            "moe_nrmse_median": float(row_nrmse.median()),
            "recall_vs_trace_mean": float(per_row["recall"].mean()),
            "set_equal_vs_trace": float((per_row["recall"] == 1).float().mean()),
            "recall_trace_vs_fp32_mean": float(per_row["recall_trace_vs_fp32"].mean()),
            "recall_device_vs_fp32_mean": float(per_row["recall_device_vs_fp32"].mean()),
            "misses": int(rank.numel()),
            "miss_top_half_fraction": float((2 * rank < misses["selected"]).float().mean()) if rank.numel() else 0.0,
            "miss_rank_histogram": torch.bincount(rank, minlength=TOP_K).tolist(),
            "dropped_trace_weight_mean": float(per_row["dropped"].mean()),
            "weight_err_median": float(per_row["weight_err"].median()),
        },
        "windows": windows,
    }
    result["bar_failures"] = check_bars(windows, BARS, ("moe_nrmse_median", NRMSE_GROWTH))
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    columns = list(windows[0])
    (out / "windows.tsv").write_text(
        "\t".join(columns) + "\n" + "".join("\t".join(f"{w[c]:.6g}" for c in columns) + "\n" for w in windows)
    )
    torch.save({"moe_nrmse": row_nrmse.float(), **{k: v.float() for k, v in per_row.items()}}, out / "rows.pt")
    return result


# --- fp32 host recompute on sampled rows ------------------------------------------------------------------------


# The device MoE block's weight formats by tensor family: the routed experts at MoEDtypes.routed_weights (bfloat4_b);
# the shared expert and the latent down projection are the MoE head's weights (MoEDtypes.shared_weights, bfloat8_b);
# the latent up projection is moe_tail's W_up (prepare_moe_tail_weights' default, bfloat8_b). Host emulation only.
DEVICE_WEIGHT_FORMATS = {
    "routed": "BFLOAT4_B",
    "shared": "BFLOAT8_B",
    "latent_down": "BFLOAT8_B",
    "latent_up": "BFLOAT8_B",
}
WEIGHT_FAMILIES = {
    "shared": ("shared_experts.gate_proj.weight", "shared_experts.up_proj.weight", "shared_experts.down_proj.weight"),
    "latent_down": ("routed_expert_down_proj.weight",),
    "latent_up": ("routed_expert_up_proj.weight",),
}


def block_float(w: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """A Linear weight ``w`` [out, in] as the device holds it: W^T [in, out] in ttnn's block-float TILE format
    (``dtype_name``, e.g. ``BFLOAT4_B``: 16 elements of a face row share an exponent), round-tripped by ttnn on the
    host, back as [out, in] fp32. Run with ``$TT_METAL_MOCK_CLUSTER_DESC_PATH``: the conversion otherwise opens the
    real cluster to learn its arch."""
    import ttnn

    t = ttnn.from_torch(w.T.contiguous().float(), dtype=getattr(ttnn.DataType, dtype_name), layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().T.contiguous()


class HostMoE:
    """The model reference's MoE layer (`modeling_kimi_moe.KimiSparseMoeBlock`, latent: down-project to 3584, the
    top-16 SiTU experts and their weighted sum there, ``routed_expert_norm``, up-project, plus the shared experts on
    the full hidden) in fp32 for a few rows. ``w``: the non-routed weights by their checkpoint names under
    ``block_sparse_moe.``; ``expert(e)``: routed expert e's (w1 gate, w3 up, w2 down), read when a row picks it.
    ``formats``: weight families (``routed`` or a ``WEIGHT_FAMILIES`` key) held in a device block-float format
    (``block_float``) instead of the checkpoint's values - the math stays fp32, so the distance to the plain fp32
    MoE is that weight format's error alone."""

    def __init__(self, w: dict[str, torch.Tensor], expert, top_k: int = TOP_K, formats: dict[str, str] | None = None):
        self.raw, self.formats = w, dict(formats or {})
        assert set(self.formats) <= {"routed", *WEIGHT_FAMILIES}, f"unknown weight families {self.formats}"
        held = {n: f for k, f in self.formats.items() for n in WEIGHT_FAMILIES.get(k, ())}
        self.w = {n: block_float(t, held[n]) if n in held else t.float() for n, t in w.items()}
        self.load_expert, self.top_k = expert, top_k
        self.experts: dict[int, tuple[torch.Tensor, ...]] = {}

    NON_ROUTED = (
        "gate.weight",
        "gate.e_score_correction_bias",
        "routed_expert_down_proj.weight",
        "routed_expert_up_proj.weight",
        "routed_expert_norm.weight",
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    )

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, layer: int) -> "HostMoE":
        from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import checkpoint_prefix, load_checkpoint_tensors

        root = f"{checkpoint_prefix(checkpoint)}layers.{layer}.block_sparse_moe."
        t = load_checkpoint_tensors(checkpoint, [root + n for n in cls.NON_ROUTED])

        def expert(e: int) -> tuple[torch.Tensor, ...]:
            names = [f"{root}experts.{e}.{m}.weight" for m in ("w1", "w3", "w2")]
            loaded = load_checkpoint_tensors(checkpoint, names)
            return tuple(loaded[n] for n in names)

        return cls({n: t[root + n] for n in cls.NON_ROUTED}, expert)

    @staticmethod
    def situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """`modeling_kimi_moe.SituAndMul` with K3's beta / linear beta, in fp32."""
        beta, linear = KimiK3Config.ACTIVATION_SITU_BETA, KimiK3Config.ACTIVATION_SITU_LINEAR_BETA
        return beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * (linear * torch.tanh(up / linear))

    def expert(self, e: int) -> tuple[torch.Tensor, ...]:
        if e not in self.experts:
            fmt = self.formats.get("routed")
            self.experts[e] = tuple(block_float(t, fmt) if fmt else t.float() for t in self.load_expert(e))
        return self.experts[e]

    def with_formats(self, formats: dict[str, str]) -> "HostMoE":
        """This MoE (its checkpoint weights, sharing the expert loader) with ``formats``."""
        return HostMoE(self.raw, self.load_expert, self.top_k, formats)

    def __call__(self, x: torch.Tensor, ids: torch.Tensor | None = None, weights: torch.Tensor | None = None):
        """MoE output fp32 ``[n, hidden]`` for rows ``x``, routed by the fp32 router, or by ``ids`` / ``weights``
        when given (e.g. the trace's routing)."""
        x = x.float()
        if ids is None:
            ids, weights, _ = route(x, self.w["gate.weight"], self.w["gate.e_score_correction_bias"], self.top_k)
        latent = x @ self.w["routed_expert_down_proj.weight"].T
        y = torch.zeros_like(latent)
        for row in range(x.shape[0]):
            for e, wt in zip(ids[row].tolist(), weights[row].float().tolist()):
                w1, w3, w2 = self.expert(e)
                h = self.situ(latent[row] @ w1.T, latent[row] @ w3.T)
                y[row] += wt * (h @ w2.T)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS) * self.w["routed_expert_norm.weight"]
        y = y @ self.w["routed_expert_up_proj.weight"].T
        shared = self.situ(
            x @ self.w["shared_experts.gate_proj.weight"].T, x @ self.w["shared_experts.up_proj.weight"].T
        )
        return y + shared @ self.w["shared_experts.down_proj.weight"].T


def nrmse_rows(got: torch.Tensor, want: torch.Tensor, floor: float) -> torch.Tensor:
    """Per-row ||got - want|| / max(||want||, floor) (the scorer's per-row NRMSE, against ``want``)."""
    return (got.float() - want.float()).norm(dim=-1) / want.float().norm(dim=-1).clamp_min(floor)


def host_check(*, trace: Trace, out: Path, checkpoint: Path, rows: int) -> dict:
    """fp32 host MoE on ``rows`` rows spread over the scored run in ``out``: per-row NRMSE (floored as the scorer
    floors) of device vs trace, device vs fp32, trace vs fp32 (fp32 routed by itself) and trace vs fp32 routed by
    the trace's own top-16 (the experts' numerics without routing differences). Writes host.json."""
    metrics = json.loads((out / "metrics.json").read_text())
    manifest, layer, floor = metrics["run"], metrics["layer"], metrics["moe_nrmse_floor"]
    source, chunk = Path(manifest["input"]), manifest["chunk"]
    picks = torch.linspace(0, manifest["count"] - 1, rows).long().tolist()
    host = HostMoE.from_checkpoint(checkpoint, layer)
    x = torch.cat([load_input(source, r, r + 1) for r in picks])
    want = torch.cat([trace.rows(f"decoder_output_layer_{layer}", r, r + 1) for r in picks])
    ids = torch.cat([trace.rows(f"expert_ids_layer_{layer}", r, r + 1) for r in picks]).long()
    ws = torch.cat([trace.rows(f"expert_weights_layer_{layer}", r, r + 1) for r in picks])
    dev = torch.stack([torch.load(chunk_path(out, r // chunk), weights_only=True)["moe_out"][r % chunk] for r in picks])
    began = time.perf_counter()
    fp32 = host(x)
    fp32_trace_routing = host(x, ids, ws)
    distances = {
        "device_vs_trace": nrmse_rows(dev, want, floor),
        "device_vs_fp32": nrmse_rows(dev, fp32, floor),
        "trace_vs_fp32": nrmse_rows(want, fp32, floor),
        "trace_vs_fp32_trace_routing": nrmse_rows(want, fp32_trace_routing, floor),
        "device_vs_fp32_trace_routing": nrmse_rows(dev, fp32_trace_routing, floor),
    }
    result = {
        "layer": layer,
        "run_code_revision": manifest.get("code_revision"),
        "rows": picks,
        "seconds": round(time.perf_counter() - began, 1),
        "experts_read": len(host.experts),
        **{k: {"median": float(v.median()), "p90": _q(v, 0.9), "max": float(v.max())} for k, v in distances.items()},
    }
    (out / "host.json").write_text(json.dumps(result, indent=1))
    return result


# Host emulations of the device's weight formats, one family at a time and all together (DEVICE_WEIGHT_FORMATS, and
# the same with bfloat8_b routed experts: the attribution run's formats).
EMULATIONS = {
    "routed_bfp4": {"routed": "BFLOAT4_B"},
    "routed_bfp8": {"routed": "BFLOAT8_B"},
    "shared_bfp8": {"shared": "BFLOAT8_B"},
    "latent_down_bfp8": {"latent_down": "BFLOAT8_B"},
    "latent_up_bfp8": {"latent_up": "BFLOAT8_B"},
    "device_weights": DEVICE_WEIGHT_FORMATS,
    "device_weights_routed_bfp8": {**DEVICE_WEIGHT_FORMATS, "routed": "BFLOAT8_B"},
}


def emulate(*, out: Path, checkpoint: Path, rows: int, names: list[str]) -> dict:
    """On ``rows`` rows of the scored device run in ``out`` (host_check's rows), the fp32 MoE with each ``EMULATIONS``
    entry's weights in its device format, every one routed by the fp32 router's selection on the unchanged gate (the
    weight formats alone, no routing difference): per-row NRMSE of each emulation vs fp32 and vs the device run, and
    of the device vs fp32 (floored as the scorer floors). Distances are measured, never combined. Writes
    emulate.json."""
    metrics = json.loads((out / "metrics.json").read_text())
    manifest, layer, floor = metrics["run"], metrics["layer"], metrics["moe_nrmse_floor"]
    source, chunk = Path(manifest["input"]), manifest["chunk"]
    picks = torch.linspace(0, manifest["count"] - 1, rows).long().tolist()
    base = HostMoE.from_checkpoint(checkpoint, layer)
    raw: dict[int, tuple[torch.Tensor, ...]] = {}  # (each expert read once for every emulation)
    load = base.load_expert
    base.load_expert = lambda e: raw[e] if e in raw else raw.setdefault(e, load(e))
    x = torch.cat([load_input(source, r, r + 1) for r in picks])
    dev = torch.stack([torch.load(chunk_path(out, r // chunk), weights_only=True)["moe_out"][r % chunk] for r in picks])
    ids, weights, _ = route(x, base.w["gate.weight"], base.w["gate.e_score_correction_bias"], base.top_k)
    began = time.perf_counter()
    fp32 = base(x, ids, weights)
    stat = lambda v: {"median": float(v.median()), "p90": _q(v, 0.9), "max": float(v.max())}  # noqa: E731
    result = {
        "layer": layer,
        "run_code_revision": manifest.get("code_revision"),
        "run_routed_weights": manifest.get("routed_weights", "BFLOAT4_B"),
        "rows": picks,
        "device_vs_fp32": stat(nrmse_rows(dev, fp32, floor)),
        "emulations": {},
    }
    base.experts.clear()
    for name in names:
        moe = base.with_formats(EMULATIONS[name])
        y = moe(x, ids, weights)
        result["emulations"][name] = {
            "formats": EMULATIONS[name],
            "vs_fp32": stat(nrmse_rows(y, fp32, floor)),
            "device_vs_emulation": stat(nrmse_rows(dev, y, floor)),
        }
        del moe, y
    result["seconds"] = round(time.perf_counter() - began, 1)
    result["experts_read"] = len(raw)
    (out / "emulate.json").write_text(json.dumps(result, indent=1))
    return result


def same_rows(a: Path, b: Path) -> dict:
    """Whether two complete device runs of one input (e.g. forward and reverse chunk order) hold bit-identical MoE
    outputs and routing in every chunk: the MoE has no state across chunks."""
    ma, mb = complete_run(a), complete_run(b)
    assert (ma["input"], ma["count"], ma["chunk"]) == (mb["input"], mb["count"], mb["chunk"]), "not runs of one input"
    differ = []
    for c in range(-(-ma["count"] // ma["chunk"])):
        pa, pb = (torch.load(chunk_path(d, c), weights_only=True) for d in (a, b))
        if not all(torch.equal(pa[k], pb[k]) for k in ("moe_out", "expert_ids", "expert_weights")):
            differ.append(c)
    return {"chunks": -(-ma["count"] // ma["chunk"]), "differ": differ, "orders": [ma.get("order"), mb.get("order")]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    checkpoint = Path(os.environ.get("KIMI_K31_HF_MODEL", ""))
    build = sub.add_parser("input", help="rebuild the teacher-fed MoE input from the trace")
    build.add_argument("--trace", type=Path, required=True)
    build.add_argument("--layer", type=int, required=True)
    build.add_argument("--checkpoint", type=Path, default=checkpoint)
    build.add_argument("--count", type=int, help="rows to build (default: the trace's prompt)")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--segment", type=int, default=4096)
    build.add_argument("--input-stream", help="a captured stream holding the layer's pre-FFN AttnRes read output")
    build.add_argument(
        "--input-normed", action="store_true", help="--input-stream holds post_attention_layernorm's output already"
    )
    rate = sub.add_parser("score", help="score a device run against the trace")
    rate.add_argument("--trace", type=Path, required=True)
    rate.add_argument("--out", type=Path, required=True)
    rate.add_argument("--window", type=int, default=4096)
    rate.add_argument("--checkpoint", type=Path, default=checkpoint)
    hostp = sub.add_parser("host", help="fp32 host MoE on sampled rows of a scored run")
    hostp.add_argument("--trace", type=Path, required=True)
    hostp.add_argument("--out", type=Path, required=True)
    hostp.add_argument("--rows", type=int, default=32)
    hostp.add_argument("--checkpoint", type=Path, default=checkpoint)
    samep = sub.add_parser("same", help="are two device runs of one input bit-identical chunk by chunk")
    samep.add_argument("runs", type=Path, nargs=2)
    emup = sub.add_parser("emulate", help="fp32 host MoE with the device's weight formats, one family at a time")
    emup.add_argument("--out", type=Path, required=True)
    emup.add_argument("--rows", type=int, default=32)
    emup.add_argument("--only", nargs="*", default=list(EMULATIONS), choices=list(EMULATIONS))
    emup.add_argument("--checkpoint", type=Path, default=checkpoint)
    cachep = sub.add_parser("routed-cache", help="a layer's routed experts at another dtype into a private cache")
    cachep.add_argument("--layer", type=int, required=True)
    cachep.add_argument("--cache", type=Path, required=True)
    cachep.add_argument("--dtype", default="BFLOAT8_B", help="ttnn.DataType name")
    cachep.add_argument("--checkpoint", type=Path, default=checkpoint)
    args = parser.parse_args()
    if args.command == "emulate":
        print(json.dumps(emulate(out=args.out, checkpoint=args.checkpoint, rows=args.rows, names=args.only), indent=1))
        return
    if args.command == "routed-cache":
        print(
            json.dumps(
                build_routed_cache(
                    checkpoint=args.checkpoint, layer=args.layer, cache=args.cache, dtype_name=args.dtype
                )
            )
        )
        return
    if args.command == "same":
        result = same_rows(*args.runs)
        print(json.dumps(result))
        raise SystemExit(1 if result["differ"] else 0)
    trace = Trace(args.trace)
    if args.command == "input":
        build_moe_input(
            trace=trace,
            checkpoint=args.checkpoint,
            layer=args.layer,
            count=args.count or trace.prompt_len,
            out=args.out,
            segment=args.segment,
            input_stream=args.input_stream,
            input_normed=args.input_normed,
        )
    elif args.command == "score":
        w = layer_weights(args.checkpoint, complete_run(args.out)["layer"])
        gate, bias = w["block_sparse_moe.gate.weight"], w["block_sparse_moe.gate.e_score_correction_bias"]
        result = score(trace=trace, out=args.out, window=args.window, gate=gate, bias=bias)
        print(json.dumps(result["overall"], indent=1))
        print("bars: " + ("PASS" if not result["bar_failures"] else "FAIL\n  " + "\n  ".join(result["bar_failures"])))
        raise SystemExit(1 if result["bar_failures"] else 0)
    else:
        print(json.dumps(host_check(trace=trace, out=args.out, checkpoint=args.checkpoint, rows=args.rows), indent=1))


if __name__ == "__main__":
    main()
