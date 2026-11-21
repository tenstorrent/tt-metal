# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 DSA layer 92 teacher-fed from a vLLM trace: host input, host scoring, error vs context position.

Three steps, and only the middle one opens a device (``test_dsa_teacher_feed.py``), so CPU work never holds it:

1. ``input``: the layer's attention input, rebuilt from the trace segment by segment and stored with a
   ``manifest.json`` (resumable, as ``tests/kda/long_context_golden.py``). Layer 92 reads AttnRes, not
   ``decoder_output_layer_91`` (the trace's ``decoder_input`` alias): its input is ``input_layernorm`` of the
   AttnRes read over the eight sealed block snapshots and the live sum of layers 84-91. A trace that records every
   layer's ``attn_output`` and FFN output (``decoder_output``, the tracer's FFN output alone) rebuilds it by the
   tracer's replay rule (``debug_trace_io.get_prefix_sum``) in vLLM's bf16 order; a trace that records the read's
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
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_attn_res
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config

LAYER = KimiK3Config.NUM_LAYERS - 1  # 92: the one DSA layer whose input and outputs every K3.1 trace records
MANIFEST = "manifest.json"
EPS = KimiK3Config.RMS_NORM_EPS
# The rebuilt input must reproduce the layer's captured projections to this PCC; measured 0.999992 (KV latent) and
# 0.999996 (indexer key) at 10K and 128K, against <= 0.16 for the decoder_output_layer_91 alias.
INPUT_CHECK_PCC = 0.9999
CHECK_ROWS = 64  # rows per segment the input check samples

# Stability bars, every one held per window of context positions (proposed, pending
# the owner's decision; measured on layer 92 at 10K and 128K, 4x4-sp1, against the vLLM trace):
# * attention output per-row NRMSE, window median <= 0.06 (measured 0.044-0.048; an fp32 host recompute from the
#   trace's own KV and top-k reads 0.032 against the trace, so the trace's own noise is two thirds of it), and no
#   window more than 15% above the first (measured +3.4% at most): growth with context is what the run looks for.
# * kvpe latent / rope and index key PCC >= 0.999 / 0.9999 / 0.999 (measured 0.99958 / 0.99999 / 0.99994, flat).
# * top-k recall, window mean over rows >= top-k, >= 0.85 (measured 0.955-0.987; 0.85 is the 128K floor of the
#   depth ladder at HiFi2) and recall 1 below top-k; misses near-ties: <= 1% of missed keys from the trace's top half
#   (measured 0).
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


def check_bars(windows: list[dict]) -> list[str]:
    """The windows that break a bar (NaN, a metric a window does not define, passes), and NRMSE growth."""
    failures = []
    for name, (kind, bar) in BARS.items():
        for w in windows:
            v = w[name]
            if v == v and (v < bar if kind == "min" else v > bar):
                failures.append(f"{name} {v:.6f} {'<' if kind == 'min' else '>'} {bar} at [{w['start']}, {w['end']})")
    first = windows[0]["attn_nrmse_median"]
    worst = max(windows, key=lambda w: w["attn_nrmse_median"])
    if worst["attn_nrmse_median"] > NRMSE_GROWTH * first:
        failures.append(
            f"attn NRMSE grows {worst['attn_nrmse_median'] / first:.3f}x (> {NRMSE_GROWTH}) "
            f"by [{worst['start']}, {worst['end']})"
        )
    return failures


class Trace:
    """A ``chunked_group_a_v1`` vLLM trace: each stream a directory of row shards listed in ``index.json``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.streams = json.loads((self.path / "index.json").read_text())["tensor_streams"]
        self.metadata = json.loads((self.path / "metadata.json").read_text())

    @property
    def prompt_len(self) -> int:
        return int(self.metadata["n_prompt_tokens"])

    def has(self, key: str) -> bool:
        return key in self.streams

    def rows(self, key: str, start: int, end: int) -> torch.Tensor:
        """Rows ``[start, end)`` of one stream in its stored dtype."""
        parts = []
        for chunk in self.streams[key]["chunks"]:
            lo, hi = chunk["row_start"], chunk["row_end"]
            if hi <= start or lo >= end:
                continue
            with safe_open(str(self.path / chunk["path"]), framework="pt") as handle:
                parts.append(handle.get_slice(key)[max(start, lo) - lo : min(end, hi) - lo])
        if not parts:
            raise ValueError(f"{self.path.name}: no {key} rows in [{start}, {end})")
        return torch.cat(parts)


def replay_missing(trace: Trace, layer: int = LAYER) -> list[int]:
    """The layers below ``layer`` whose attn_output or FFN output the trace lacks: the replay needs all of them."""
    return [
        l for l in range(layer) if not (trace.has(f"attn_output_layer_{l}") and trace.has(f"decoder_output_layer_{l}"))
    ]


def replay_live_and_sealed(trace: Trace, start: int, end: int, layer: int = LAYER) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnRes state entering ``layer``'s attention for rows ``[start, end)``: (live sum ``[n, d]``, sealed
    snapshots ``[n, S, d]``), by the tracer's replay rule in bf16 (vLLM's order, which reproduces its snapshots bit
    for bit): P = emb; per layer L: P += ffn_{L-1} (L > 0); at a block start seal P and restart P = attn_L, else
    P += attn_L; finally P += ffn_{layer-1}."""
    block = KimiK3Config.ATTN_RES_BLOCK_SIZE
    keys = ["decoder_input_layer_0"]
    for l in range(layer):
        keys += [f"attn_output_layer_{l}", f"decoder_output_layer_{l}"]
    with ThreadPoolExecutor(max_workers=16) as pool:
        streams = dict(zip(keys, pool.map(lambda k: trace.rows(k, start, end), keys)))
    live = streams["decoder_input_layer_0"]
    sealed = []
    for l in range(layer):
        if l > 0:
            live = live + streams[f"decoder_output_layer_{l - 1}"]
        if l % block == 0:
            sealed.append(live)
            live = streams[f"attn_output_layer_{l}"]
        else:
            live = live + streams[f"attn_output_layer_{l}"]
    live = live + streams[f"decoder_output_layer_{layer - 1}"]
    return live, torch.stack(sealed, dim=1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """vLLM's RMSNorm: normalize and scale in fp32, one bf16 rounding."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()).to(torch.bfloat16)


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


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


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
    return {"kv_latent_pcc": pcc(latent, want_kv[:, : KimiK3Config.KV_LORA_RANK]), "index_key_pcc": pcc(key, want_key)}


def provenance() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {"code_revision": revision, "torch": torch.__version__, "host": platform.node()}


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


def segment_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"segment_{index:06d}.pt"


def build_input(
    *, trace: Trace, checkpoint: Path, count: int, out: Path, segment: int, input_stream: str | None = None
) -> dict:
    """Write (or resume) the teacher-fed input ``normed = input_layernorm(read)`` for rows ``[0, count)``."""
    out.mkdir(parents=True, exist_ok=True)
    if input_stream is None:
        missing = replay_missing(trace)
        if missing:
            raise ValueError(
                f"{trace.path.name} cannot rebuild layer {LAYER}'s attention input: no attn_output/decoder_output "
                f"for {len(missing)} of {LAYER} layers ({missing[:8]}...); pass --input-stream with a captured read"
            )
    elif not trace.has(input_stream):
        raise ValueError(f"{trace.path.name} has no stream {input_stream}")
    expected = {
        "layer": LAYER,
        "trace": str(trace.path),
        "trace_commit": trace.metadata.get("trace_commit"),
        "input": f"stream {input_stream}" if input_stream else "attn_res replay (bf16, tracer order)",
        "checkpoint": str(checkpoint),
        "count": count,
        "segment": segment,
    }
    manifest_path = out / MANIFEST
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        mismatched = sorted(k for k, v in expected.items() if manifest.get(k) != v)
        if mismatched:
            raise ValueError(f"{out} holds a different input; mismatched fields: {mismatched}")
    else:
        manifest = {**expected, "sessions": [], "segments": []}
    segments = -(-count // segment)
    done = len(manifest["segments"])
    if done < segments:
        manifest["sessions"].append({"first_segment": done, **provenance()})
        write_json(manifest_path, manifest)
    w = layer_weights(checkpoint)
    for index in range(done, segments):
        start, end = index * segment, min((index + 1) * segment, count)
        began = time.perf_counter()
        if input_stream is None:
            live, sealed = replay_live_and_sealed(trace, start, end)
            read = hf_attn_res(
                live, sealed, w["self_attention_res_norm.weight"], w["self_attention_res_proj.weight"], EPS
            )
        else:
            read = trace.rows(input_stream, start, end)
        normed = rms_norm(read, w["input_layernorm.weight"])
        check = check_input(normed, trace, start, w)
        if min(check.values()) < INPUT_CHECK_PCC:
            raise ValueError(f"rows [{start}, {end}): rebuilt input fails its check {check} (< {INPUT_CHECK_PCC})")
        torch.save({"start": start, "end": end, "normed": normed}, segment_path(out, index))
        seconds = time.perf_counter() - began
        manifest["segments"].append({"start": start, "end": end, "seconds": round(seconds, 2), **check})
        write_json(manifest_path, manifest)
        print(
            f"{time.strftime('%H:%M:%S')} segment {index + 1}/{segments} rows [{start}, {end}) {seconds:.1f} s "
            + " ".join(f"{k}={v:.6f}" for k, v in check.items()),
            flush=True,
        )
    return manifest


def load_input(run_dir: Path, start: int, end: int) -> torch.Tensor:
    """Teacher-fed input rows ``[start, end)`` as bf16 ``[n, 7168]``; the input must hold them."""
    manifest = json.loads((run_dir / MANIFEST).read_text())
    segment = manifest["segment"]
    available = manifest["segments"][-1]["end"] if manifest["segments"] else 0
    if not 0 <= start < end <= available:
        raise ValueError(f"rows [{start}, {end}) are not inside the {available} input rows of {run_dir}")
    rows = []
    for index in range(start // segment, -(-end // segment)):
        payload = torch.load(segment_path(run_dir, index), weights_only=True)
        rows.append(payload["normed"][max(start, payload["start"]) - payload["start"] : end - payload["start"]])
    return torch.cat(rows)


# --- scoring -------------------------------------------------------------------------------------------------


def chunk_path(out: Path, index: int) -> Path:
    return out / f"chunk_{index:06d}.pt"


def topk_recall(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per row |got ∩ want| / |want| over ``want``'s valid (>= 0) keys; ``got``'s negatives match nothing."""
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    valid = want >= 0
    return (torch.isin(rows + want, rows + got.masked_fill(got < 0, -(1 << 31))) & valid).sum(1) / valid.sum(1)


def miss_margins(got: torch.Tensor, want: torch.Tensor, scores: torch.Tensor) -> list[torch.Tensor]:
    """Per row, the trace keys the device missed as (trace score - cut) / (top - cut), the cut being the trace's
    lowest selected score: 0 is a miss at the cut (a near-tie), 1 a miss of the row's best key. Rows with fewer
    than two valid trace keys have no spread and yield no margins."""
    out = []
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    missed = ~torch.isin(rows + want, rows + got.masked_fill(got < 0, -(1 << 31))) & (want >= 0)
    valid = want >= 0
    for r in range(want.shape[0]):
        s = scores[r][valid[r]]
        if s.numel() < 2 or not missed[r].any():
            out.append(torch.empty(0))
            continue
        cut, top = s.min(), s.max()
        out.append(((scores[r][missed[r]] - cut) / (top - cut).clamp_min(1e-30)).float())
    return out


class Pooled:
    """Pearson correlation pooled over every pair of rows added, from float64 sums, so a window can span chunks."""

    def __init__(self):
        self.sums = torch.zeros(6, dtype=torch.float64)  # n, sx, sy, sxx, syy, sxy

    def add(self, want: torch.Tensor, got: torch.Tensor) -> None:
        x, y = want.double().flatten(), got.double().flatten()
        self.sums += torch.stack(
            [torch.tensor(float(x.numel()), dtype=torch.float64), x.sum(), y.sum(), x @ x, y @ y, x @ y]
        )

    def value(self) -> float:
        n, sx, sy, sxx, syy, sxy = self.sums.tolist()
        return (sxy - sx * sy / n) / ((sxx - sx * sx / n) * (syy - sy * sy / n)) ** 0.5


def score(*, trace: Trace, out: Path, window: int) -> dict:
    """Score a device run in ``out`` against the trace per context position; writes metrics.json and windows.tsv.

    Streams chunk by chunk (the 1M attention output is 15 GB per side). Per row: attention output NRMSE (floored at
    1% of the median trace row norm, as the KDA golden), top-k recall and missed-key score margins. Per ``window``
    rows: pooled PCC of the attention output, kvpe latent / rope and index key, and summaries of the per-row values.
    """
    from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import golden_floor

    manifest = json.loads((out / MANIFEST).read_text())
    count, chunk, layer = manifest["count"], manifest["chunk"], manifest["layer"]
    num_chunks = -(-count // chunk)
    num_windows = -(-count // window)
    attn_pcc = [Pooled() for _ in range(num_windows)]
    ref_norm, err_norm, recall, margins = [], [], [], []
    for c in range(num_chunks):
        payload = torch.load(chunk_path(out, c), weights_only=True)
        lo, hi = payload["start"], payload["end"]
        got_attn = payload["attn_out"].double()
        want_attn = trace.rows(f"attn_output_layer_{layer}", lo, hi).double()
        ref_norm.append(want_attn.norm(dim=-1))
        err_norm.append((got_attn - want_attn).norm(dim=-1))
        for w in range(lo // window, -(-hi // window)):
            a, b = max(lo, w * window) - lo, min(hi, (w + 1) * window) - lo
            attn_pcc[w].add(want_attn[a:b], got_attn[a:b])
        want = trace.rows(f"dsa_topk_indices_layer_{layer}", lo, hi).long()
        got = payload["topk"].long()
        recall.append(topk_recall(got, want))
        margins += miss_margins(got, want, trace.rows(f"dsa_topk_scores_layer_{layer}", lo, hi).float())
        print(f"{time.strftime('%H:%M:%S')} scored chunk {c + 1}/{num_chunks} rows [{lo}, {hi})", flush=True)
    ref_norm, err_norm, recall = torch.cat(ref_norm), torch.cat(err_norm), torch.cat(recall)
    floor = golden_floor(ref_norm)
    row_nrmse = err_norm / ref_norm.clamp_min(floor)
    caches = torch.load(out / "caches.pt", weights_only=True)
    kv_ref = trace.rows(f"kv_post_transform_layer_{layer}", 0, count).float()
    key_ref = trace.rows(f"indexer_k_layer_{layer}", 0, count).float()
    lora = KimiK3Config.KV_LORA_RANK
    topk = KimiK31Config.INDEX_TOPK
    windows = []
    for w, lo in enumerate(range(0, count, window)):
        hi = min(lo + window, count)
        m = torch.cat(margins[lo:hi])
        gated = recall[max(lo, topk) : hi]  # rows < top-k select every causal key on both sides
        nan = float("nan")
        windows.append(
            {
                "start": lo,
                "end": hi,
                "attn_pcc": attn_pcc[w].value(),
                "attn_nrmse_median": float(row_nrmse[lo:hi].median()),
                "attn_nrmse_p99": float(row_nrmse[lo:hi].quantile(0.99)),
                "attn_nrmse_max": float(row_nrmse[lo:hi].max()),
                "kv_latent_pcc": pcc(kv_ref[lo:hi, :lora], caches["kvpe"][lo:hi, :lora]),
                "kv_rope_pcc": pcc(kv_ref[lo:hi, lora:], caches["kvpe"][lo:hi, lora:]),
                "index_key_pcc": pcc(key_ref[lo:hi], caches["index"][lo:hi]),
                "recall_mean": float(gated.mean()) if gated.numel() else nan,
                "recall_p05": float(gated.quantile(0.05)) if gated.numel() else nan,
                "recall_min_below_topk": float(recall[lo : min(hi, topk)].min()) if lo < topk else nan,
                "miss_margin_median": float(m.median()) if m.numel() else nan,
                "miss_margin_p90": float(m.quantile(0.9)) if m.numel() else nan,
                "miss_top_half_fraction": float((m > 0.5).float().mean()) if m.numel() else nan,
            }
        )
    whole = Pooled()
    whole.sums = sum((p.sums for p in attn_pcc), torch.zeros(6, dtype=torch.float64))
    result = {
        "layer": layer,
        "count": count,
        "window": window,
        "trace": str(trace.path),
        "run": manifest,
        "attn_nrmse_floor": floor,
        "overall": {
            "attn_pcc": whole.value(),
            "attn_nrmse_median": float(row_nrmse.median()),
            "kv_latent_pcc": pcc(kv_ref[:, :lora], caches["kvpe"][:, :lora]),
            "kv_rope_pcc": pcc(kv_ref[:, lora:], caches["kvpe"][:, lora:]),
            "index_key_pcc": pcc(key_ref, caches["index"]),
            "recall_mean": float(recall[topk:].mean()) if count > topk else float("nan"),
        },
        "windows": windows,
    }
    result["bar_failures"] = check_bars(windows)
    write_json(out / "metrics.json", result)
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
        build_input(
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
