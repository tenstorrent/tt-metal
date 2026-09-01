# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-model prefill + decode at the shipped batch, not batch 1.

Why this exists. Every other full-model test on this port pins the batch:
``full_model_perf.py``, ``full_model_perf_warm.py``, ``full_model_long_prefill.py``,
``full_model_trace_lifecycle.py`` and ``full_model_qualitative.py`` all build the
generator with ``batch=1``, and ``full_model_mixed_slots.py`` with ``batch=2``. Yet
the shipped serving batch is 32, the decoder policies (``decode_storage_cores``,
the L1 width-sharded residual, the advisor placement) are tuned at 32, and every
per-layer number in ``doc/`` is measured at 32. So the one shape the model
actually runs in had no full-model measurement at all.

The vLLM path does reach batch 32 but cannot answer this question: it disables
chunked prefill for ``model_type=qwen3_5`` and therefore issues one prefill per
scheduler step, paying the full-slot-width prefill 32 times over. Calling
``prefill_forward`` directly with all rows active pays it once, which is what a
serving stack ought to cost and what this measures.

Reports prefill (cold and warm) and traced token-out decode, so TSU at the
shipped batch is directly comparable to the batch-1 number from
``full_model_perf_warm.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-iters", type=int, default=3)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import LINEAR_PREFILL_CHUNK_SIZE

    result = {
        "label": args.label,
        "batch": args.batch,
        "prefill_iters": args.prefill_iters,
        "linear_prefill_chunk_size": LINEAR_PREFILL_CHUNK_SIZE,
    }
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=args.max_context,
            batch=args.batch,
        )
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt.read_text().strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        all_ids = generator.tokenizer.encode(rendered, add_special_tokens=False)
        if args.prompt_tokens > len(all_ids):
            raise SystemExit(f"prompt has only {len(all_ids)} tokens, need {args.prompt_tokens}")
        token_ids = all_ids[: args.prompt_tokens]
        # Every slot carries the same real prompt, so all rows are active and the
        # prefill is not being cheated by zero-length rows.
        tokens = torch.tensor([token_ids] * args.batch, dtype=torch.long)
        prompt_lens = [len(token_ids)] * args.batch

        ttft_ms = []
        logits = None
        for i in range(args.prefill_iters):
            generator.reset()
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            logits = generator.prefill_forward(
                tokens,
                page_table=generator._page_table,
                kv_cache=generator.kv_cache,
                prompt_lens=prompt_lens,
            )
            ttnn.synchronize_device(mesh)
            elapsed = 1000 * (time.perf_counter() - started)
            ttft_ms.append(elapsed)
            print(
                f"  B={args.batch} S={len(token_ids)} iter {i}: {elapsed:.2f} ms" + ("  (cold)" if i == 0 else ""),
                flush=True,
            )

        warm = ttft_ms[1:]
        result.update(
            {
                "prompt_tokens": len(token_ids),
                "ttft_ms_all": ttft_ms,
                "ttft_ms_cold": ttft_ms[0],
                "ttft_ms_warm_median": statistics.median(warm) if warm else None,
                "cold_overhead_ms": (ttft_ms[0] - statistics.median(warm)) if warm else None,
            }
        )

        first_token = int(torch.argmax(logits[0, 0]).item())
        capture_started = time.perf_counter()
        generator._capture_token_out_trace(first_token, len(token_ids))
        generator._seed_token_out_trace(first_token, len(token_ids))
        ttnn.synchronize_device(mesh)
        capture_seconds = time.perf_counter() - capture_started

        started = time.perf_counter()
        for _ in range(args.decode_tokens):
            generator.token_out_decode_step(readback=False)
        ttnn.synchronize_device(mesh)
        decode_seconds = time.perf_counter() - started

        # TSU is per user: every replay advances all `batch` rows one token.
        result.update(
            {
                "measured_decode_replays": args.decode_tokens,
                "trace_capture_seconds": capture_seconds,
                "token_out_seconds": decode_seconds,
                "ms_per_token": 1000 * decode_seconds / args.decode_tokens,
                "token_out_t_s_u": args.decode_tokens / decode_seconds,
                "token_out_total_t_s": args.batch * args.decode_tokens / decode_seconds,
            }
        )
    finally:
        if generator is not None:
            generator.reset()
        ttnn.close_mesh_device(mesh)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
