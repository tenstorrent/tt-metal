#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone TttGenerationWorker speed benchmark.

Single rank, single process. Opens a ``[1, --devices]`` mesh (every chip becomes
its own ``[1, 1]`` submesh, i.e. ``data_parallel == --devices``), builds one
:class:`TttGenerationWorker`, feeds it real BoolQ prompts through the same
tokeniser / chat template as ``bench_ttt.py``, and times ``worker.generate()``
over ``--iters`` iterations.

No ttml, no MPI, no GRPO, no weight sync. This measures pure generation cost
on the ttt side. Compare against the ``gen_time_s`` column of the full
benchmark (which also incurs the RPC round-trip and per-step weight push).

Typical usage (2+2 parity with bench_ttt.py: global batch 32, 2 chips, 16/DP)::

    tt-train/sources/examples/grpo_rollout_benchmark/bench_ttt_worker.py \\
        --devices 2 --batch-size 32 --max-new-tokens 256 --iters 5

Defaults use dummy weights (fast boot) and disable early stop so every
iteration decodes exactly ``--max-new-tokens`` tokens per user (deterministic
timing). Pass ``--real-weights`` to load HF Llama-3.2-1B-Instruct weights on
the worker itself (no MPI push needed); pair with ``--stop-at-eos`` if you
want completions to terminate naturally.
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from typing import Any, List

import ttnn

# Must pin BEFORE any device open (same reason as bench_ttt.py). Even in this
# single-rank script, the fabric config is process-wide and TttGenerationWorker
# stack expects FABRIC_2D.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import benchmark_common as bc  # noqa: E402
from utils.ttt_remote.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad  # noqa: E402
from utils.ttt_remote.ttt_generation_worker import TttGenerationWorker  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone TttGenerationWorker speed benchmark on real BoolQ prompts")
    p.add_argument(
        "--devices",
        type=int,
        required=True,
        help="Number of chips; opens a [1, N] mesh so every chip becomes its own [1, 1] submesh (data_parallel = N).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Global generation batch size per iteration; must be a positive multiple of --devices.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256, help="Decode tokens per user (default: 256)")
    p.add_argument("--iters", type=int, default=5, help="Timed iterations after warmup (default: 5)")
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed warmup iterations (first call captures the decode trace). Default: 1.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Model max sequence length; must exceed max prompt + --max-new-tokens (default: 2048).",
    )
    p.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="Enable EOS-based early stopping (default: off, so every iter decodes exactly --max-new-tokens).",
    )
    p.add_argument(
        "--real-weights",
        action="store_true",
        help="Load real HF Llama-3.2-1B-Instruct weights on the worker instead of dummy weights (slower boot).",
    )
    p.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="Skip the first N prompts in the (seeded-shuffled) BoolQ split before drawing (default: 0).",
    )
    args = p.parse_args()

    if args.devices <= 0:
        p.error(f"--devices must be positive (got {args.devices})")
    if args.batch_size <= 0:
        p.error(f"--batch-size must be positive (got {args.batch_size})")
    if args.batch_size % args.devices:
        p.error(
            f"--batch-size ({args.batch_size}) must be a multiple of --devices ({args.devices}); "
            f"per-submesh batch = batch_size / devices."
        )
    if args.iters <= 0:
        p.error(f"--iters must be positive (got {args.iters})")
    if args.warmup < 0:
        p.error(f"--warmup must be non-negative (got {args.warmup})")
    if args.max_new_tokens <= 0:
        p.error(f"--max-new-tokens must be positive (got {args.max_new_tokens})")

    return args


def _load_boolq_prompt_ids(tokenizer: Any) -> List[List[int]]:
    """Load the same BoolQ split bench_ttt.py trains on, tokenised to id lists."""
    dataset = bc.build_boolq_dataset(tokenizer)
    return [tokenizer.encode(row["prompt"]) for row in dataset]


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("GRPO_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    args = _parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(bc.MODEL_ID)
    prompt_pool = _load_boolq_prompt_ids(tokenizer)
    total_iters = args.warmup + args.iters
    required = args.prompt_offset + total_iters * args.batch_size
    if required > len(prompt_pool):
        raise RuntimeError(
            f"Requested {required} BoolQ prompts (offset {args.prompt_offset} + "
            f"{total_iters} iters * batch {args.batch_size}) but only {len(prompt_pool)} available."
        )

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    logging.info(
        "opening mesh [1, %d] (global batch=%d, per-submesh batch=%d, max_new_tokens=%d)",
        args.devices,
        args.batch_size,
        args.batch_size // args.devices,
        args.max_new_tokens,
    )
    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, args.devices),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    worker: Any = None
    try:
        stop_token_ids, pad_token_id = llama_stop_and_pad(bc.MODEL_ID)
        worker = TttGenerationWorker(
            mesh_device=parent_mesh,
            model_source=bc.MODEL_ID,
            max_batch_size=args.batch_size // args.devices,
            max_seq_len=args.max_seq_len,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            temperature=bc.TEMPERATURE,
            top_k=0,
            top_p=1.0,
            seed=None,
            dummy_weights=not args.real_weights,
        )

        iter_totals: List[float] = []
        iter_tokps: List[float] = []
        iter_prompt_lens: List[int] = []
        iter_completion_lens: List[float] = []

        for it in range(total_iters):
            start = args.prompt_offset + it * args.batch_size
            batch = [list(prompt_pool[start + i]) for i in range(args.batch_size)]
            batch_max_prompt_len = max(len(p) for p in batch)

            t0 = time.perf_counter()
            completions = worker.generate(
                batch,
                max_new_tokens=args.max_new_tokens,
                stop_at_eos=args.stop_at_eos,
            )
            elapsed = time.perf_counter() - t0

            total_completion_tokens = sum(len(c) for c in completions)
            mean_completion_len = total_completion_tokens / max(len(completions), 1)
            tok_per_s = total_completion_tokens / elapsed if elapsed > 0 else 0.0

            tag = f"warmup {it + 1}/{args.warmup}" if it < args.warmup else f"iter {it - args.warmup + 1}/{args.iters}"
            print(
                f"[bench_ttt_worker] {tag}: {elapsed:.2f}s | "
                f"batch={args.batch_size} | max_prompt_len={batch_max_prompt_len} | "
                f"completion_toks={total_completion_tokens} (mean {mean_completion_len:.1f}/user) | "
                f"{tok_per_s:.1f} tok/s",
                flush=True,
            )

            if it >= args.warmup:
                iter_totals.append(elapsed)
                iter_tokps.append(tok_per_s)
                iter_prompt_lens.append(batch_max_prompt_len)
                iter_completion_lens.append(mean_completion_len)

        if iter_totals:
            print(
                "[bench_ttt_worker] SUMMARY: "
                f"devices={args.devices} batch={args.batch_size} "
                f"(per-submesh {args.batch_size // args.devices}) "
                f"max_new_tokens={args.max_new_tokens} stop_at_eos={args.stop_at_eos} "
                f"real_weights={args.real_weights} | "
                f"median gen={statistics.median(iter_totals):.2f}s "
                f"median tok/s={statistics.median(iter_tokps):.1f} "
                f"median completion_len={statistics.median(iter_completion_lens):.1f} "
                f"median max_prompt_len={statistics.median(iter_prompt_lens):.0f} "
                f"over {len(iter_totals)} timed iters (excl. {args.warmup} warmup)",
                flush=True,
            )
    finally:
        worker = None
        import gc

        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


if __name__ == "__main__":
    main()
