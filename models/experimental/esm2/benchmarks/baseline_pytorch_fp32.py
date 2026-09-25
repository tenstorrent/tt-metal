# SPDX-License-Identifier: MIT
"""PyTorch FP32 CPU baseline on the TT host (BENCHMARK.md protocol, stage 0).

Measures wall-clock forward latency and host memory for the full 33-layer
model on fixed suite shapes, per-layer encoder latencies on the max-length
case, output checksums, and a whole-pipeline NRMSE cross-check against
transformers.EsmForMaskedLM FP32 on one short case (offline insurance for the
loader + twin before any TT work). Results are preserved as a baseline
artifact JSON under benchmarks/artifacts/.

Run from the model root:
    python benchmarks/baseline_pytorch_fp32.py --weights /weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from backend import create_backend  # noqa: E402

# Fixed suite shapes (name, batch, padded_len, seed) — envelope from CONTRACT.md.
# Seeds are explicit literals so the artifact is reproducible across runs.
SHAPES = [
    ("single-short", 1, 130, 1101),
    ("single-max", 1, 1026, 1102),
    ("batch2-max", 2, 1026, 1103),
]
CROSSCHECK_SEED = 7  # historical seed for the oracle cross-check case

# Frozen masking rule from the evaluator: residue positions 5, 21, 37, ...
# are replaced by mask_token 32 (residue index +1 for the cls token).


def make_case(cfg, batch: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = rng.integers(4, 24, size=(batch, length), dtype=np.int64)
    ids[:, 0] = 0  # <cls>
    ids[:, -1] = 2  # <eos>
    am = np.ones((batch, length), dtype=np.int64)
    if batch > 1:  # ragged batch: second sequence shorter
        short = length // 2 + 37
        ids[1, short:] = cfg.pad_token_id
        am[1, short:] = 0
    residues = np.arange(1, length - 1)  # residue idx (0-based) + cls offset
    masked = residues[(residues % 16) == 5]  # positions 5, 21, 37... (1-based)
    for b in range(batch):
        m = masked[am[b, masked] == 1]
        ids[b, m] = cfg.mask_token_id
    return ids, am


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sha_fp32(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def per_layer_ms(model, run_once) -> list:
    """Instrument each encoder layer with pre/post hooks for one pass."""
    n = len(model.layers)
    times, starts = [0.0] * n, [0.0] * n
    handles = []
    for i, layer in enumerate(model.layers):
        def pre_hook(module, args, _i=i):
            starts[_i] = time.perf_counter()

        def post_hook(module, args, output, _i=i):
            times[_i] = time.perf_counter() - starts[_i]

        handles.append(layer.register_forward_pre_hook(pre_hook))
        handles.append(layer.register_forward_hook(post_hook))
    try:
        run_once()
    finally:
        for h in handles:
            h.remove()
    return [round(t * 1000.0, 2) for t in times]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/artifacts/baseline_pytorch_fp32.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))  # polite on shared host
    t0 = time.perf_counter()
    backend = create_backend(args.weights, os.path.join(args.weights, "config.json"), "cpu")
    load_s = time.perf_counter() - t0
    cfg = backend.config

    results = {
        "protocol": "pytorch_fp32_cpu_baseline_v1",
        "host": {"machine": platform.node(), "python": platform.python_version(),
                 "torch": torch.__version__, "numpy": np.__version__,
                 "threads": torch.get_num_threads(), "cpu_count": os.cpu_count()},
        "model": {"layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
                  "heads": cfg.num_attention_heads, "ffn": cfg.intermediate_size},
        "weight_load_seconds": round(load_s, 3),
        "cases": [],
        "notes": [
            "Latency = single embed() call (already-masked ids -> logits+hidden),",
            "CPU FP32, threads=cpu//2 (shared host), warmup=1, repeats=3 mean.",
        ],
    }

    for name, b, l, seed in SHAPES:
        ids, am = make_case(cfg, b, l, seed=seed)
        with torch.no_grad():
            backend.embed(ids, am)  # warmup
        times = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            backend.embed(ids, am)
            times.append(time.perf_counter() - t)
        tokens = int(am.sum())
        results["cases"].append({
            "name": name, "batch": b, "padded_len": l, "seed": seed,
            "real_tokens": tokens,
            "latency_s_mean": round(float(np.mean(times)), 4),
            "latency_s_min": round(float(np.min(times)), 4),
            "tokens_per_s": round(tokens / float(np.mean(times)), 1),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        })
        print(f"{name}: {np.mean(times)*1000:.1f} ms mean, tokens/s "
              f"{tokens/np.mean(times):.1f}, rss {peak_rss_mb():.0f} MB")

    # Per-layer encoder latency on the max-length case (one instrumented pass).
    ids, am = make_case(cfg, 1, 1026, seed=1102)
    backend.embed(ids, am)  # warm
    t_total = time.perf_counter()
    layer_ms = per_layer_ms(backend._model, lambda: backend.embed(ids, am))
    total_ms = (time.perf_counter() - t_total) * 1000.0
    arr = np.array(layer_ms)
    results["per_layer"] = {
        "case": "single-max", "batch": 1, "padded_len": 1026, "seed": 1102,
        "layer_ms": layer_ms,
        "layer_ms_mean": round(float(arr.mean()), 2),
        "layer_ms_min": round(float(arr.min()), 2),
        "layer_ms_max": round(float(arr.max()), 2),
        "layers_sum_ms": round(float(arr.sum()), 2),
        "whole_pass_ms": round(total_ms, 2),
        "non_layer_ms": round(total_ms - float(arr.sum()), 2),
    }
    print(f"per-layer: mean {arr.mean():.1f} ms, sum {arr.sum():.0f} ms of "
          f"{total_ms:.0f} ms whole pass")

    # Whole-pipeline cross-check vs the FP32 oracle implementation (short case)
    # plus fixed-case output checksums for run-to-run drift detection.
    try:
        from tests.util import nrmse

        ids, am = make_case(cfg, 1, 130, seed=CROSSCHECK_SEED)
        got = backend.embed(ids, am)
        results["output_checksums_fp32"] = {
            "case": "single-short", "seed": CROSSCHECK_SEED,
            "logits_shape": list(got["logits"].shape),
            "hidden_shape": list(got["hidden"].shape),
            "logits_sha256": sha_fp32(got["logits"]),
            "hidden_sha256": sha_fp32(got["hidden"]),
            "logits_sum": float(got["logits"].sum()),
            "hidden_sum": float(got["hidden"].sum()),
        }
        from transformers import EsmForMaskedLM

        ref = EsmForMaskedLM.from_pretrained(args.weights).eval()
        with torch.no_grad():
            out = ref(torch.from_numpy(ids), attention_mask=torch.from_numpy(am))
        logits_nrmse = nrmse(out.logits, torch.from_numpy(got["logits"]))
        hidden_nrmse = nrmse(ref.esm(torch.from_numpy(ids), attention_mask=torch.from_numpy(am)).last_hidden_state,
                             torch.from_numpy(got["hidden"]))
        results["oracle_crosscheck_fp32"] = {
            "case": "single-short", "logits_nrmse": logits_nrmse,
            "final_hidden_nrmse": hidden_nrmse,
            "logits_argmax_match": bool(
                (out.logits.argmax(-1).numpy() == got["logits"].argmax(-1)).all()
            ),
        }
        print(f"oracle cross-check: logits NRMSE {logits_nrmse:.2e}, "
              f"hidden NRMSE {hidden_nrmse:.2e}")
    except Exception as e:  # pragma: no cover - record, don't crash baseline
        results["oracle_crosscheck_fp32"] = {"error": repr(e)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
