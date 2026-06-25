# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
On-device accuracy evaluation for pplx-embed-v1-4B at ISL=512 with all
performance flags ON (BFP4 weights, BFP8 activations, LoFi math, L1/big-grid,
bidirectional SDPA, KV-cache-fill skip — i.e. the exact perf serving config).

Runs the STS-B benchmark on the TT device and reports the Spearman correlation
between model cosine similarities and human scores, so it can be compared
directly against the CPU fp32 reference from ``eval_accuracy.py``.

Two pooling paths (both run the same perf-optimized forward):
  --pool fast    Fixed-ISL traced path (the low-latency serving path). Every text
                 is padded to ISL and mean-pooled on-device over the full ISL.
  --pool masked  Same traced forward, but mean-pool over the *real* tokens only
                 (masked reduce on device). Isolates the pooling effect from the
                 (unmasked) SDPA padding effect.
  --pool masked-attn  Real-token pool + SDPA padding mask (near-reference for any
                 input length; recommended for short / variable-length inputs).

Distribute across chips with --num-devices (e.g. 32) to verify accuracy + per-chip
latency hold under DP; texts are sharded across chips and gathered in order.

Examples:
  # Single device, fast path
  python models/demos/blackhole/pplx_embed_4b/demo/eval_accuracy_tt.py

  # All 32 chips (DP), fast path
  python models/demos/blackhole/pplx_embed_4b/demo/eval_accuracy_tt.py --num-devices 32

  # Masked real-token pooling + SDPA mask (recommended accuracy path)
  python models/demos/blackhole/pplx_embed_4b/demo/eval_accuracy_tt.py --pool masked-attn
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time

import numpy as np


def _encode_worker(
    chip_id: int,
    texts: list[str],
    max_length: int,
    pool: str,
    bucket: bool,
    cores_per_worker: int,
    result_dict: dict,
) -> None:
    """Build the perf model on one chip, encode a shard of texts, return embeddings."""
    os.environ["TT_VISIBLE_DEVICES"] = str(chip_id)

    total_cores = os.cpu_count() or 64
    start_core = (chip_id * cores_per_worker) % total_cores
    affinity = set(range(start_core, min(start_core + cores_per_worker, total_cores)))
    try:
        os.sched_setaffinity(0, affinity)
    except (OSError, AttributeError):
        pass

    try:
        import ttnn
        from models.demos.blackhole.pplx_embed_4b.demo._common import apply_workload_env, build_single_device_model
        from models.demos.blackhole.pplx_embed_4b.demo.live_demo import (
            BucketedEncoder,
            TracedEncoder,
            _extract_final_norm,
            bucket_lengths,
        )
        from models.tt_transformers.tt.common import get_padded_prefill_len

        apply_workload_env(1, max_length)

        device = ttnn.open_device(
            device_id=0,
            l1_small_size=32768,
            trace_region_size=200_000_000,
            num_command_queues=1,
        )

        generator, model_args, kv_caches, page_table = build_single_device_model(
            device, batch_size=1, seq_len=max_length
        )
        model = generator.model[0]
        tokenizer = model_args.tokenizer
        norm_weight, eps = _extract_final_norm(model)
        dim = model_args.dim

        use_mask = pool in ("masked-attn", "mask-fastpool")
        base_pool = "masked" if pool in ("masked", "masked-attn") else "fast"
        if bucket:
            encoder = BucketedEncoder(
                generator,
                model,
                kv_caches[0],
                page_table,
                tokenizer,
                norm_weight,
                eps,
                bucket_lengths(max_length),
                device,
                pool=base_pool,
                use_mask=use_mask,
            )
        else:
            fixed_isl = get_padded_prefill_len(max_length)
            encoder = TracedEncoder(
                generator,
                model,
                kv_caches[0],
                page_table,
                tokenizer,
                norm_weight,
                eps,
                fixed_isl,
                device,
                pool=base_pool,
                use_mask=use_mask,
            )

        embs = np.zeros((len(texts), dim), dtype=np.float32)
        latencies: list[float] = []
        for i, text in enumerate(texts):
            t0 = time.perf_counter()
            emb = encoder.encode(text, normalize=False)
            latencies.append((time.perf_counter() - t0) * 1000)
            if emb is not None:
                embs[i] = emb.numpy()

        # Record results BEFORE closing the device: close_device can SIGABRT in
        # the profiler teardown (DeviceProfiler::dumpDeviceResults), which would
        # kill the process and lose an otherwise-complete result.
        steady = latencies[1:] if len(latencies) > 1 else latencies  # drop compile/warm iter
        result_dict[chip_id] = {
            "ok": True,
            "embs": embs,
            "lat_mean": statistics.mean(steady),
            "lat_min": min(steady),
            "lat_median": statistics.median(steady),
            "n": len(texts),
        }

        # Hard-exit instead of ttnn.close_device(): with many processes tearing
        # down simultaneously, the C++ device teardown (device-profiler marker
        # dump, profiler.cpp) can SIGABRT across all workers at once, hanging the
        # parent on join and corrupting board state. Results are already recorded
        # in the manager dict above; let the OS reclaim the device on exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        import traceback

        result_dict[chip_id] = {"ok": False, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _encode_all(texts: list[str], max_length: int, pool: str, num_devices: int, bucket: bool):
    """Shard texts across ``num_devices`` chips, return (embeddings_in_order, lat_stats)."""
    total_cores = os.cpu_count() or 64
    cores_per_worker = max(1, total_cores // max(1, num_devices))

    # Contiguous shards preserve global ordering when concatenated by chip id.
    shards = [list(c) for c in np.array_split(np.array(texts, dtype=object), num_devices)]

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_dict = manager.dict()

    procs = []
    for chip_id in range(num_devices):
        p = ctx.Process(
            target=_encode_worker,
            args=(chip_id, list(shards[chip_id]), max_length, pool, bucket, cores_per_worker, result_dict),
            name=f"chip-{chip_id}",
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=1800)
        if p.is_alive():
            print(f"  WARNING: {p.name} hung — terminating")
            p.terminate()
            p.join(timeout=10)

    all_embs = []
    lat_means, lat_mins, lat_medians = [], [], []
    for chip_id in range(num_devices):
        r = result_dict.get(chip_id)
        if not r or not r.get("ok"):
            err = (r or {}).get("error", "no result")
            raise RuntimeError(f"chip {chip_id} failed: {err[:500]}")
        all_embs.append(r["embs"])
        lat_means.append(r["lat_mean"])
        lat_mins.append(r["lat_min"])
        lat_medians.append(r["lat_median"])

    embeddings = np.concatenate(all_embs, axis=0)
    lat_stats = {
        "per_chip_mean": statistics.mean(lat_means),
        "per_chip_median": statistics.mean(lat_medians),
        "best_chip_min": min(lat_mins),
        "slowest_chip_median": max(lat_medians),
    }
    return embeddings, lat_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="On-device pplx-embed-v1-4B STS-B accuracy")
    parser.add_argument("--num-devices", type=int, default=1, help="Chips to shard across (1=single device)")
    parser.add_argument("--max-length", type=int, default=512, help="ISL / max tokens per text (default 512)")
    parser.add_argument(
        "--pool",
        choices=["fast", "masked", "masked-attn", "mask-fastpool"],
        default="fast",
        help="fast=device mean over ISL; masked=real-token pool; masked-attn=real-token pool + SDPA padding mask",
    )
    parser.add_argument(
        "--no-bucket",
        action="store_true",
        help="Disable sequence-length bucketing (pad everything to --max-length). By default each "
        "text is routed to the smallest padded-length bucket (128/256/512...) that fits.",
    )
    args = parser.parse_args()
    bucket = not args.no_bucket

    from scipy.stats import spearmanr

    from models.demos.blackhole.pplx_embed_4b.demo.eval_accuracy import load_stsb

    print(f"Loading STS-B test set...")
    s1, s2, gold = load_stsb()
    all_texts = s1 + s2
    print(
        f"  {len(s1)} pairs, {len(all_texts)} texts | ISL={args.max_length} | "
        f"pool={args.pool} | num_devices={args.num_devices}"
    )

    t0 = time.perf_counter()
    embeddings, lat = _encode_all(all_texts, args.max_length, args.pool, args.num_devices, bucket)
    wall = time.perf_counter() - t0

    embs1 = embeddings[: len(s1)]
    embs2 = embeddings[len(s1) :]
    # Cosine similarity per pair.
    a = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-12)
    b = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-12)
    cos = (a * b).sum(axis=1)
    corr, pval = spearmanr(cos, np.array(gold))

    print()
    print("=" * 60)
    print(f"  pplx-embed-v1-4B  TT-device STS-B accuracy")
    print("=" * 60)
    print(f"  ISL:                    {args.max_length}")
    print(f"  Pooling:                {args.pool}")
    print(f"  Bucketing:              {'on' if bucket else 'off'}")
    print(f"  Chips:                  {args.num_devices}")
    print(f"  STS-B Spearman:         {corr:.4f}  (p={pval:.2e})")
    print("-" * 60)
    print(f"  Per-chip mean latency:  {lat['per_chip_mean']:.1f}ms")
    print(f"  Per-chip median:        {lat['per_chip_median']:.1f}ms")
    print(f"  Best chip (min):        {lat['best_chip_min']:.1f}ms")
    print(f"  Slowest chip (median):  {lat['slowest_chip_median']:.1f}ms")
    print(f"  Wall time:              {wall:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
