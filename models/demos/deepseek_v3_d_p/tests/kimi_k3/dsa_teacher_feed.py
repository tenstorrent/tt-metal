# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 DSA layer 92 teacher-fed from a vLLM trace: host input, host scoring, error vs context position.

Three steps, and only the middle one opens a device (``test_dsa_teacher_feed.py``), so CPU work never holds it:

1. ``input``: the layer's attention input, rebuilt from the trace segment by segment and stored with a
   ``manifest.json`` (resumable, with ``tests/kda/long_context_golden.py``'s manifest helpers). Layer 92 reads
   AttnRes, not ``decoder_output_layer_91`` (the trace's ``decoder_input`` alias): its input is ``input_layernorm``
   of the AttnRes read over the eight sealed block snapshots and the live sum of layers 84-91. A trace that records
   every layer's ``attn_output`` and FFN output (``decoder_output``, the tracer's FFN output alone) rebuilds it by
   the tracer's replay rule (``debug_trace_io.get_prefix_sum``) in vLLM's bf16 order; a trace that records the read's
   output itself names that stream with ``--input-stream``. Every segment is checked against the layer's own
   captured projections (indexer key, KV latent) recomputed from the rebuilt input, and fails closed below
   ``INPUT_CHECK_PCC``.
2. The device test runs the production DSA module (``ttMLA``, layer 92, the stage's weight cache) chunk by chunk
   over the whole prefix, filling its own kvpe and index-key caches, and saves each chunk's attention output and
   top-k indices and, at the end, both caches.
3. ``score``: every saved row against the trace, per context position, as ``metrics.json`` and a per-window table.

    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_teacher_feed input --trace T --count N --out IN
    TT_DSA_TF_INPUT=IN TT_DSA_TF_OUT=OUT scripts/run_safe_pytest.sh ...test_dsa_teacher_feed.py::test_...
    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_teacher_feed score --trace T --out OUT
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_attn_res
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import TokenErrorAccumulator, golden_floor, pcc
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import (
    EPS,
    Trace,
    build_input,
    check_bars,
    chunk_path,
    complete_run,
    missed_keys,
    replay_live_and_sealed,
    replay_missing,
    rms_norm,
    topk_recall,
)

LAYER = KimiK3Config.NUM_LAYERS - 1  # 92: the one DSA layer whose input and outputs every K3.1 trace records
# The rebuilt input must reproduce the layer's captured projections to this PCC; measured 0.999992 (KV latent) and
# 0.999996 (indexer key) at 10K and 128K, against <= 0.16 for the decoder_output_layer_91 alias.
INPUT_CHECK_PCC = 0.9999
CHECK_ROWS = 64  # rows per segment the input check samples

# Stability bars, every one held per window of context positions (proposed, pending the owner's decision; measured on
# layer 92 at 10K and 128K, 4x4-sp1, against the vLLM trace):
# * attention output per-row NRMSE against the trace, window median <= 0.06 (measured 0.044-0.048). On sampled rows
#   the device reads 0.053 against an fp32 host recompute from the trace's own KV and top-k, and the trace itself
#   0.032 against it; which part of the device's distance is its own is left to an FP64 golden. And no window more
#   than 15% above the first (measured +3.4% at most): growth with context is what the run looks for.
# * kvpe latent / rope and index key PCC >= 0.999 / 0.9999 / 0.999 (measured 0.99958 / 0.99999 / 0.99994, flat).
# * top-k recall, window mean over rows >= top-k, >= 0.85 (measured 0.955-0.987; 0.85 is the 128K floor of the
#   depth ladder at HiFi2) and recall 1 below top-k; misses are near-ties: <= 1% of missed keys ranked in the top
#   half of the trace's own selection by its score.
BARS = {
    "attn_nrmse_median": ("max", 0.06),
    "kv_latent_pcc": ("min", 0.999),
    "kv_rope_pcc": ("min", 0.9999),
    "index_key_pcc": ("min", 0.999),
    "recall_mean": ("min", 0.85),
    "recall_min_below_topk": ("min", 1.0),
    "miss_top_half_fraction": ("max", 0.01),
}
NRMSE_GROWTH = 1.15


def layer_weights(checkpoint: Path, layer: int = LAYER) -> dict[str, torch.Tensor]:
    """The host weights the input and its check need: AttnRes query, input_layernorm, KV-A and indexer key."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import checkpoint_prefix, load_checkpoint_tensors

    root = f"{checkpoint_prefix(checkpoint)}layers.{layer}."
    names = [
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "input_layernorm.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias",
    ]
    tensors = load_checkpoint_tensors(checkpoint, [root + n for n in names])
    return {n: tensors[root + n] for n in names}


def check_input(normed: torch.Tensor, trace: Trace, start: int, w: dict, layer: int = LAYER) -> dict[str, float]:
    """PCC of the layer's indexer key and KV latent recomputed (fp32) from ``normed`` against the trace's."""
    rows = torch.linspace(0, normed.shape[0] - 1, min(CHECK_ROWS, normed.shape[0])).long()
    x = normed[rows].float()
    kv = x @ w["self_attn.kv_a_proj_with_mqa.weight"].float().T
    latent = rms_norm(kv[:, : KimiK3Config.KV_LORA_RANK], w["self_attn.kv_a_layernorm.weight"]).float()
    key = torch.nn.functional.layer_norm(
        x @ w["self_attn.indexer.wk.weight"].float().T,
        (KimiK31Config.INDEX_HEAD_DIM,),
        w["self_attn.indexer.k_norm.weight"].float(),
        w["self_attn.indexer.k_norm.bias"].float(),
        EPS,
    )
    want_kv = torch.cat(
        [trace.rows(f"kv_post_transform_layer_{layer}", start + r, start + r + 1) for r in rows.tolist()]
    )
    want_key = torch.cat([trace.rows(f"indexer_k_layer_{layer}", start + r, start + r + 1) for r in rows.tolist()])
    return {"kv_latent_pcc": pcc(want_kv[:, : KimiK3Config.KV_LORA_RANK], latent), "index_key_pcc": pcc(want_key, key)}


def build_dsa_input(
    *, trace: Trace, checkpoint: Path, count: int, out: Path, segment: int, input_stream: str | None = None
) -> dict:
    """Write (or resume) the teacher-fed input ``normed = input_layernorm(read)`` for rows ``[0, count)``."""
    if input_stream is None:
        missing = replay_missing(trace, LAYER)
        if missing:
            raise ValueError(
                f"{trace.path.name} cannot rebuild layer {LAYER}'s attention input: no attn_output/decoder_output "
                f"for {len(missing)} of {LAYER} layers ({missing[:8]}...); pass --input-stream with a captured read"
            )
    elif not trace.has(input_stream):
        raise ValueError(f"{trace.path.name} has no stream {input_stream}")
    w = layer_weights(checkpoint)

    def make(start: int, end: int) -> tuple[torch.Tensor, dict[str, float]]:
        if input_stream is None:
            live, sealed = replay_live_and_sealed(trace, start, end, LAYER)
            read = hf_attn_res(
                live, sealed, w["self_attention_res_norm.weight"], w["self_attention_res_proj.weight"], EPS
            )
        else:
            read = trace.rows(input_stream, start, end)
        normed = rms_norm(read, w["input_layernorm.weight"])
        return normed, check_input(normed, trace, start, w)

    expected = {
        "layer": LAYER,
        "trace": str(trace.path),
        "trace_commit": trace.metadata.get("trace_commit"),
        "input": f"stream {input_stream}" if input_stream else "attn_res replay (bf16, tracer order)",
        "norm": "KimiRMSNorm bf16 (normalize fp32, round, scale)",
        "checkpoint": str(checkpoint),
    }
    bars = {"kv_latent_pcc": INPUT_CHECK_PCC, "index_key_pcc": INPUT_CHECK_PCC}
    return build_input(out=out, expected=expected, count=count, segment=segment, make=make, check_bars=bars)


# --- scoring -------------------------------------------------------------------------------------------------


def score(*, trace: Trace, out: Path, window: int) -> dict:
    """Score a complete device run in ``out`` against the trace per context position; writes metrics.json,
    windows.tsv and rows.pt.

    Streams chunk by chunk (the 1M attention output is 15 GB per side). Per row: attention output NRMSE (floored at
    1% of the median trace row norm, as the KDA golden), top-k recall, and each missed key's rank and spread share.
    Per ``window`` rows: summaries of those, and the kvpe latent / rope and index key PCC.
    """
    manifest = complete_run(out)
    count, chunk, layer = manifest["count"], manifest["chunk"], manifest["layer"]
    num_chunks = -(-count // chunk)
    attn = TokenErrorAccumulator()
    recall, misses = [], []
    for c in range(num_chunks):
        payload = torch.load(chunk_path(out, c), weights_only=True)
        lo, hi = payload["start"], payload["end"]
        attn.update(trace.rows(f"attn_output_layer_{layer}", lo, hi), payload["attn_out"])
        want = trace.rows(f"dsa_topk_indices_layer_{layer}", lo, hi).long()
        got = payload["topk"].long()
        recall.append(topk_recall(got, want))
        chunk_misses = missed_keys(got, want, trace.rows(f"dsa_topk_scores_layer_{layer}", lo, hi).float())
        chunk_misses["row"] = chunk_misses["row"] + lo
        misses.append(chunk_misses)
        print(f"{time.strftime('%H:%M:%S')} scored chunk {c + 1}/{num_chunks} rows [{lo}, {hi})", flush=True)
    floor = golden_floor(attn.expected_norms())
    row_nrmse = attn.token_nrmse(floor)
    recall = torch.cat(recall)
    misses = {k: torch.cat([m[k] for m in misses]) for k in misses[0]}
    caches = torch.load(out / "caches.pt", weights_only=True)
    kv_ref = trace.rows(f"kv_post_transform_layer_{layer}", 0, count)
    key_ref = trace.rows(f"indexer_k_layer_{layer}", 0, count)
    lora = KimiK3Config.KV_LORA_RANK
    topk = KimiK31Config.INDEX_TOPK
    nan = float("nan")
    windows = []
    for lo in range(0, count, window):
        hi = min(lo + window, count)
        inside = (misses["row"] >= lo) & (misses["row"] < hi)
        rank, selected, share = (misses[k][inside] for k in ("rank", "selected", "spread_share"))
        gated = recall[max(lo, topk) : hi]  # rows < top-k select every causal key on both sides
        windows.append(
            {
                "start": lo,
                "end": hi,
                "attn_nrmse_median": float(row_nrmse[lo:hi].median()),
                "attn_nrmse_p99": float(row_nrmse[lo:hi].quantile(0.99)),
                "attn_nrmse_max": float(row_nrmse[lo:hi].max()),
                "kv_latent_pcc": pcc(kv_ref[lo:hi, :lora], caches["kvpe"][lo:hi, :lora]),
                "kv_rope_pcc": pcc(kv_ref[lo:hi, lora:], caches["kvpe"][lo:hi, lora:]),
                "index_key_pcc": pcc(key_ref[lo:hi], caches["index"][lo:hi]),
                "recall_mean": float(gated.mean()) if gated.numel() else nan,
                "recall_p05": float(gated.quantile(0.05)) if gated.numel() else nan,
                "recall_min_below_topk": float(recall[lo : min(hi, topk)].min()) if lo < topk else nan,
                "misses": int(rank.numel()),
                "miss_rank_p10": float(rank.float().quantile(0.1)) if rank.numel() else nan,
                "miss_rank_median": float(rank.float().median()) if rank.numel() else nan,
                "miss_rank_min": int(rank.min()) if rank.numel() else -1,
                "miss_top_half_fraction": float((2 * rank < selected).float().mean()) if rank.numel() else nan,
                "miss_spread_share_median": float(share.median()) if share.numel() else nan,
                "miss_spread_share_p90": float(share.quantile(0.9)) if share.numel() else nan,
            }
        )
    result = {
        "layer": layer,
        "count": count,
        "window": window,
        "trace": str(trace.path),
        "run": manifest,
        "attn_nrmse_floor": floor,
        "overall": {
            "attn_nrmse_median": float(row_nrmse.median()),
            "kv_latent_pcc": pcc(kv_ref[:, :lora], caches["kvpe"][:, :lora]),
            "kv_rope_pcc": pcc(kv_ref[:, lora:], caches["kvpe"][:, lora:]),
            "index_key_pcc": pcc(key_ref, caches["index"]),
            "recall_mean": float(recall[topk:].mean()) if count > topk else nan,
            "misses": int(misses["rank"].numel()),
            "miss_top_half_fraction": (
                float((2 * misses["rank"] < misses["selected"]).float().mean()) if misses["rank"].numel() else nan
            ),
        },
        "windows": windows,
    }
    result["bar_failures"] = check_bars(windows, BARS, ("attn_nrmse_median", NRMSE_GROWTH))
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    columns = list(windows[0])
    (out / "windows.tsv").write_text(
        "\t".join(columns) + "\n" + "".join("\t".join(f"{w[c]:.6g}" for c in columns) + "\n" for w in windows)
    )
    torch.save({"attn_nrmse": row_nrmse.float(), "recall": recall.float()}, out / "rows.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("input", help="rebuild the teacher-fed input from the trace")
    build.add_argument("--trace", type=Path, required=True)
    build.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("KIMI_K31_HF_MODEL", "")))
    build.add_argument("--count", type=int, help="rows to build (default: the trace's prompt)")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--segment", type=int, default=4096)
    build.add_argument("--input-stream", help="a captured stream holding the layer's AttnRes read output")
    rate = sub.add_parser("score", help="score a device run against the trace")
    rate.add_argument("--trace", type=Path, required=True)
    rate.add_argument("--out", type=Path, required=True)
    rate.add_argument("--window", type=int, default=4096)
    args = parser.parse_args()
    trace = Trace(args.trace)
    if args.command == "input":
        build_dsa_input(
            trace=trace,
            checkpoint=args.checkpoint,
            count=args.count or trace.prompt_len,
            out=args.out,
            segment=args.segment,
            input_stream=args.input_stream,
        )
    else:
        result = score(trace=trace, out=args.out, window=args.window)
        print(json.dumps(result["overall"], indent=1))
        print("bars: " + ("PASS" if not result["bar_failures"] else "FAIL\n  " + "\n  ".join(result["bar_failures"])))
        raise SystemExit(1 if result["bar_failures"] else 0)


if __name__ == "__main__":
    main()
