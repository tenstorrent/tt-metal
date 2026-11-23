# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 DSA layer teacher-fed from a vLLM trace (layer 92) or the FP64 chain (layer 3): host input, host scoring,
error vs context position.

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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.accuracy.topk import missed_keys, topk_recall
from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import (
    SUM_YY,
    Bar,
    GrowthBar,
    RowSeries,
    check,
    errors,
    quantile,
    row_nrmse,
    row_sums,
)
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import golden_floor, pcc
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import (
    EPS,
    Trace,
    build_input,
    chain_identity,
    checkpoint_tensors,
    chunk_path,
    complete_run,
    replay_live_and_sealed,
    replay_missing,
    rms_norm,
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
NRMSE_GROWTH = 1.15
BARS = [
    Bar("attn_nrmse_median", "max", 0.06),
    Bar("kv_latent_pcc", "min", 0.999),
    Bar("kv_rope_pcc", "min", 0.9999),
    Bar("index_key_pcc", "min", 0.999),
    Bar("recall_mean", "min", 0.85),
    Bar("recall_min_below_topk", "min", 1.0),
    Bar("miss_top_half_fraction", "max", 0.01),
    GrowthBar("attn_nrmse_median", NRMSE_GROWTH),
]


def layer_weights(checkpoint: Path, layer: int = LAYER) -> dict[str, torch.Tensor]:
    """The host weights the input and its check need: AttnRes query, input_layernorm, KV-A and indexer key."""
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
    return checkpoint_tensors(checkpoint, layer, names)


def check_rows(n: int) -> torch.Tensor:
    return torch.linspace(0, n - 1, min(CHECK_ROWS, n)).long()


def recomputed_keys(x: torch.Tensor, w: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer's KV latent and indexer key recomputed in fp32 from input rows ``x`` (the input check's)."""
    x = x.float()
    kv = x @ w["self_attn.kv_a_proj_with_mqa.weight"].float().T
    lora = w["self_attn.kv_a_layernorm.weight"].numel()
    latent = rms_norm(kv[:, :lora], w["self_attn.kv_a_layernorm.weight"]).float()
    key = torch.nn.functional.layer_norm(
        x @ w["self_attn.indexer.wk.weight"].float().T,
        (w["self_attn.indexer.k_norm.weight"].numel(),),
        w["self_attn.indexer.k_norm.weight"].float(),
        w["self_attn.indexer.k_norm.bias"].float(),
        EPS,
    )
    return latent, key


def check_input(normed: torch.Tensor, trace: Trace, start: int, w: dict, layer: int = LAYER) -> dict[str, float]:
    """PCC of the layer's indexer key and KV latent recomputed (fp32) from ``normed`` against the trace's."""
    rows = check_rows(normed.shape[0])
    latent, key = recomputed_keys(normed[rows], w)
    want_kv = torch.cat(
        [trace.rows(f"kv_post_transform_layer_{layer}", start + r, start + r + 1) for r in rows.tolist()]
    )
    want_key = torch.cat([trace.rows(f"indexer_k_layer_{layer}", start + r, start + r + 1) for r in rows.tolist()])
    return {"kv_latent_pcc": pcc(want_kv[:, : KimiK3Config.KV_LORA_RANK], latent), "index_key_pcc": pcc(want_key, key)}


def build_dsa_input(
    *,
    trace: Trace,
    checkpoint: Path,
    count: int,
    out: Path,
    segment: int,
    input_stream: str | None = None,
    layer: int = LAYER,
) -> dict:
    """Write (or resume) the teacher-fed input ``normed = input_layernorm(read)`` of DSA layer ``layer`` (92, the one
    a K3.1 trace records the outputs of, by default) for rows ``[0, count)``."""
    if input_stream is None:
        missing = replay_missing(trace, layer)
        if missing:
            raise ValueError(
                f"{trace.path.name} cannot rebuild layer {layer}'s attention input: no attn_output/decoder_output "
                f"for {len(missing)} of {layer} layers ({missing[:8]}...); pass --input-stream with a captured read"
            )
    elif not trace.has(input_stream):
        raise ValueError(f"{trace.path.name} has no stream {input_stream}")
    w = layer_weights(checkpoint, layer)

    def make(start: int, end: int) -> tuple[torch.Tensor, dict[str, float]]:
        if input_stream is None:
            live, sealed = replay_live_and_sealed(trace, start, end, layer)
            read = hf_attn_res(
                live, sealed, w["self_attention_res_norm.weight"], w["self_attention_res_proj.weight"], EPS
            )
        else:
            read = trace.rows(input_stream, start, end)
        normed = rms_norm(read, w["input_layernorm.weight"])
        return normed, check_input(normed, trace, start, w, layer)

    expected = {
        "layer": layer,
        "trace": str(trace.path),
        "trace_commit": trace.metadata.get("trace_commit"),
        "input": f"stream {input_stream}" if input_stream else "attn_res replay (bf16, tracer order)",
        "norm": "KimiRMSNorm bf16 (normalize fp32, round, scale)",
        "checkpoint": str(checkpoint),
    }
    bars = {"kv_latent_pcc": INPUT_CHECK_PCC, "index_key_pcc": INPUT_CHECK_PCC}
    return build_input(out=out, expected=expected, count=count, segment=segment, make=make, check_bars=bars)


def build_dsa_input_from_chain(*, chain: Path, checkpoint: Path, count: int, out: Path, segment: int) -> dict:
    """Write (or resume) DSA layer 3's teacher-fed input from an FP64 chain run: ``input_layernorm`` of the chain's
    ``dsa_input_3`` (the AttnRes read, FP64 stored fp32) in FP64, rounded once to bf16, the device's input format.
    Every segment is checked by recomputing (fp32) the KV latent and index key of sampled rows from that bf16 input
    against the chain's FP64 keys, and fails closed below ``INPUT_CHECK_PCC``."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import DSA_LAYER, load_products

    w = layer_weights(checkpoint, DSA_LAYER)
    norm = w["input_layernorm.weight"].double()

    def make(start: int, end: int) -> tuple[torch.Tensor, dict[str, float]]:
        p = load_products(chain, ["dsa_input_3", "latent", "index_key"], start, end)
        read = p["dsa_input_3"].double()
        normed = (read * torch.rsqrt(read.pow(2).mean(-1, keepdim=True) + EPS) * norm).to(torch.bfloat16)
        rows = check_rows(normed.shape[0])
        latent, key = recomputed_keys(normed[rows], w)
        return normed, {
            "kv_latent_pcc": pcc(p["latent"][rows], latent),
            "index_key_pcc": pcc(p["index_key"][rows], key),
        }

    expected = {
        "layer": DSA_LAYER,
        **chain_identity(chain),
        "input": "fp64_chain dsa_input_3 (AttnRes read, FP64, stored fp32)",
        "norm": "input_layernorm in FP64, one bf16 rounding",
    }
    bars = {"kv_latent_pcc": INPUT_CHECK_PCC, "index_key_pcc": INPUT_CHECK_PCC}
    return build_input(out=out, expected=expected, count=count, segment=segment, make=make, check_bars=bars)


# --- scoring -------------------------------------------------------------------------------------------------


def score(*, trace: Trace, out: Path, window: int) -> dict:
    """Score a complete device run in ``out`` per context position against its reference; returns the metrics with
    the per-row values under ``rows`` (``write_score`` stores them in the run directory).

    The reference is the run's input's source. From a trace: the trace's attention output (every row), top-k and its
    scores, kv_post_transform and indexer_k. From an FP64 chain (``fp64_chain.py``): the chain's FP64 attention output
    on its sampled rows only, its top-2048 and FP64 scores, and its FP64 keys; the GPU trace's top-k recall of the
    chain's selection is reported beside the device's (``recall_gpu_vs_reference``). Streams chunk by chunk (the 1M
    attention output is 15 GB per side). Per row: attention output NRMSE (floored at 1% of the median reference row
    norm, as the KDA golden), top-k recall, and each missed key's rank and spread share in the reference's scores.
    Per ``window`` rows: summaries of those, the attention output's and the kvpe latent / rope and index key's absolute
    and relative error (``windowed.errors``; their PCC is the bars' ``*_pcc``). A quantity a window does not define
    (no scored attention row, no row past top-k, rows past top-k for ``recall_min_below_topk``) is left out of it; a
    window without a miss has no miss in the top half.
    """
    manifest = complete_run(out)
    count, chunk, layer = manifest["count"], manifest["chunk"], manifest["layer"]
    chain = json.loads((Path(manifest["input"]) / MANIFEST).read_text()).get("chain")
    num_chunks = -(-count // chunk)
    attn, attn_rows = [], []
    recall, gpu_recall, misses = [], [], []
    if chain is not None:
        from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import load_products, load_sampled, open_chain

        chain_path = open_chain(chain).path
        sampled = load_sampled(chain_path)
        keep = sampled["sampled_rows"] < count
        sampled = {k: v[keep] for k, v in sampled.items()}
    for c in range(num_chunks):
        payload = torch.load(chunk_path(out, c), weights_only=True)
        lo, hi = payload["start"], payload["end"]
        got = payload["topk"].long()
        if chain is None:
            attn.append(row_sums(trace.rows(f"attn_output_layer_{layer}", lo, hi), payload["attn_out"]))
            attn_rows.append(torch.arange(lo, hi))
            want = trace.rows(f"dsa_topk_indices_layer_{layer}", lo, hi).long()
            want_scores = trace.rows(f"dsa_topk_scores_layer_{layer}", lo, hi).float()
        else:
            inside = (sampled["sampled_rows"] >= lo) & (sampled["sampled_rows"] < hi)
            if inside.any():
                rows = sampled["sampled_rows"][inside]
                attn.append(row_sums(sampled["attn_output_3"][inside].float(), payload["attn_out"][rows - lo]))
                attn_rows.append(rows)
            p = load_products(chain_path, ["topk_indices_3", "topk_scores_3"], lo, hi)
            want, want_scores = p["topk_indices_3"].long(), p["topk_scores_3"].float()
            gpu_recall.append(topk_recall(trace.rows(f"dsa_topk_indices_layer_{layer}", lo, hi).long(), want))
        recall.append(topk_recall(got, want))
        chunk_misses = missed_keys(got, want, want_scores)
        chunk_misses["row"] = chunk_misses["row"] + lo
        misses.append(chunk_misses)
        print(f"{time.strftime('%H:%M:%S')} scored chunk {c + 1}/{num_chunks} rows [{lo}, {hi})", flush=True)
    attn = torch.cat(attn)
    floor = golden_floor(attn[:, SUM_YY].sqrt())
    caches = torch.load(out / "caches.pt", weights_only=True)
    if chain is None:
        lora = KimiK3Config.KV_LORA_RANK
        kv_ref = trace.rows(f"kv_post_transform_layer_{layer}", 0, count)
        key_ref = trace.rows(f"indexer_k_layer_{layer}", 0, count)
    else:
        k = load_products(chain_path, ["latent", "kpe", "index_key"], 0, count)
        lora, kv_ref, key_ref = k["latent"].shape[1], torch.cat([k["latent"], k["kpe"]], -1), k["index_key"]
    topk = want.shape[1]  # the reference's selection size (2048)
    series = RowSeries()  # per row: recall and the caches' row sums; the scored attention rows and misses as events
    series.append(
        0,
        recall=torch.cat(recall)[:count],
        kv_latent=row_sums(kv_ref[:, :lora], caches["kvpe"][:count, :lora]),
        kv_rope=row_sums(kv_ref[:, lora:], caches["kvpe"][:count, lora:]),
        index_key=row_sums(key_ref, caches["index"][:count]),
        **({"gpu_recall": torch.cat(gpu_recall)[:count]} if gpu_recall else {}),
    )
    series.add_events("attn", row=torch.cat(attn_rows), sums=attn, nrmse=row_nrmse(attn, floor))
    for m in misses:
        series.add_events("misses", **m)

    def common(w) -> dict:
        """What windows and the whole run report alike; a quantity the window does not define is None."""
        nr, rank = w["attn"]["nrmse"], w["misses"]["rank"]
        gated = slice(max(w.start, topk) - w.start, None)  # rows < top-k select every causal key on both sides
        return {
            "attn_rows": int(nr.numel()),
            "attn_nrmse_median": float(nr.median()) if nr.numel() else None,
            **(errors(w["attn"]["sums"], "attn_") if nr.numel() else {}),
            **{k: v for name in ("kv_latent", "kv_rope", "index_key") for k, v in errors(w[name], f"{name}_").items()},
            "recall_mean": float(w["recall"][gated].mean()) if w.end > topk else None,
            "recall_gpu_vs_reference_mean": (
                float(w["gpu_recall"][gated].mean()) if w.end > topk and "gpu_recall" in w else None
            ),
            "misses": int(rank.numel()),
            # no miss: no miss in the top half, not undefined
            "miss_top_half_fraction": (
                float((2 * rank < w["misses"]["selected"]).float().mean()) if rank.numel() else 0.0
            ),
        }

    def per_window(w) -> dict:
        nr, rank, share = w["attn"]["nrmse"], w["misses"]["rank"], w["misses"]["spread_share"]
        gated = w["recall"][max(w.start, topk) - w.start :]
        return {
            **common(w),
            "attn_nrmse_p99": quantile(nr, 0.99) if nr.numel() else None,
            "attn_nrmse_max": float(nr.max()) if nr.numel() else None,
            "recall_p05": quantile(gated, 0.05) if gated.numel() else None,
            "recall_min_below_topk": float(w["recall"][: topk - w.start].min()) if w.start < topk else None,
            "miss_rank_p10": quantile(rank, 0.1) if rank.numel() else None,
            "miss_rank_median": quantile(rank, 0.5) if rank.numel() else None,
            "miss_rank_min": int(rank.min()) if rank.numel() else None,
            "miss_spread_share_median": quantile(share, 0.5) if rank.numel() else None,
            "miss_spread_share_p90": quantile(share, 0.9) if rank.numel() else None,
        }

    windows = series.windows(window, per_window)
    result = {
        "layer": layer,
        "count": count,
        "window": window,
        "trace": str(trace.path),
        "reference": chain or "trace",
        "run": manifest,
        "attn_nrmse_floor": floor,
        "overall": series.whole(common),
        "windows": windows,
    }
    result["bar_failures"] = check(windows, BARS)
    result["rows"] = {
        "attn_rows": torch.cat(attn_rows),
        "attn_nrmse": row_nrmse(attn, floor).float(),
        "recall": series.column("recall").float(),
    }
    return result


def write_score(out: Path, result: dict) -> None:
    """``score``'s result into the run directory: metrics.json, windows.tsv (the columns every window has) and the
    per-row values (rows.pt)."""
    (out / "metrics.json").write_text(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    columns = [c for c in result["windows"][0] if all(c in w for w in result["windows"])]
    (out / "windows.tsv").write_text(
        "\t".join(columns) + "\n" + "".join("\t".join(f"{w[c]:.6g}" for c in columns) + "\n" for w in result["windows"])
    )
    torch.save(result["rows"], out / "rows.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("input", help="rebuild the teacher-fed input from the trace, or take it from a chain")
    build.add_argument("--trace", type=Path, help="rebuild from this trace")
    build.add_argument(
        "--chain", type=Path, help="take layer 3's input from this fp64_chain run (and score against it)"
    )
    build.add_argument("--layer", type=int, default=LAYER, help="DSA layer (0-based) to rebuild from the trace")
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
    if args.command == "input" and args.chain:
        count = args.count or json.loads((args.chain / MANIFEST).read_text())["segments"][-1]["end"]
        build_dsa_input_from_chain(
            chain=args.chain, checkpoint=args.checkpoint, count=count, out=args.out, segment=args.segment
        )
        return
    trace = Trace(args.trace)
    if args.command == "input":
        build_dsa_input(
            trace=trace,
            checkpoint=args.checkpoint,
            count=args.count or trace.prompt_len,
            out=args.out,
            segment=args.segment,
            input_stream=args.input_stream,
            layer=args.layer,
        )
    else:
        result = score(trace=trace, out=args.out, window=args.window)
        write_score(args.out, result)
        print(json.dumps(result["overall"], indent=1))
        print("bars: " + ("PASS" if not result["bar_failures"] else "FAIL\n  " + "\n  ".join(result["bar_failures"])))
        raise SystemExit(1 if result["bar_failures"] else 0)


if __name__ == "__main__":
    main()
