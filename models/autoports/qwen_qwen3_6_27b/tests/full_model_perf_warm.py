# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cold vs warm B1 TTFT, plus the canonical split-trace token-out number.

Why this exists. Every TTFT figure recorded on this port is a single observation
of a COLD prefill:

  * ``tests/full_model_perf.py`` times the first ``prefill_forward`` in the
    process. ``generator.reset()`` precedes it, but that does not populate
    tt-metal's program cache, so the measurement includes kernel compilation and
    program build for every distinct prefill op shape across all 64 layers.
  * the stage 09 and 10 vLLM benchmarks each ran ONE request at concurrency 1.
    Their TTFT P50 equals their P99 (4139/4139 and 3784/3784), which is the
    signature of n=1, not of a stable distribution. Their ITL P50/P99 differ
    (55.840/56.850), because ITL genuinely has many samples.

Consequences: the ~500 ms TTFT spread stage 08 used to rank precision
candidates, and the 8.6% TTFT gain stage 10 claims over stage 09, are both
differences between single cold observations -- and a precision change alters
which matmul program configs get compiled, so it moves compile cost directly.
Warm TTFT, what the second and later requests actually pay, is unmeasured.

This measures both from one weight load: iteration 0 is the cold number and is
directly comparable to the recorded ones; iterations 1..N-1 are warm. The
token-out section replicates tests/full_model_perf.py exactly so that number
stays comparable to the stage-cited value.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-iters", type=int, default=6)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    result = {"label": args.label, "prefill_iters": args.prefill_iters}
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=512,
            batch=1,
        )
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt.read_text().strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = generator.tokenizer.encode(rendered, add_special_tokens=False)
        token_ids = token_ids[: args.prompt_tokens]
        tokens = torch.tensor([token_ids], dtype=torch.long)

        # ---- cold + warm prefill -------------------------------------------
        # Identical call each iteration; reset() restores cache and releases
        # traces so every iteration prefills the same prompt from a clean slot.
        # Only the program cache persists across iterations, which is exactly
        # the variable under test.
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
                prompt_lens=[len(token_ids)],
            )
            ttnn.synchronize_device(mesh)
            elapsed = 1000 * (time.perf_counter() - started)
            ttft_ms.append(elapsed)
            print(f"  prefill iter {i}: {elapsed:.2f} ms" + ("  (cold)" if i == 0 else ""), flush=True)

        warm = ttft_ms[1:]
        result.update(
            {
                "prompt_tokens": len(token_ids),
                "ttft_ms_all": ttft_ms,
                "ttft_ms_cold": ttft_ms[0],
                "ttft_ms_warm_median": statistics.median(warm) if warm else None,
                "ttft_ms_warm_min": min(warm) if warm else None,
                "ttft_ms_warm_max": max(warm) if warm else None,
                # how much of the recorded cold number was one-time program build
                "cold_overhead_ms": (ttft_ms[0] - statistics.median(warm)) if warm else None,
            }
        )

        # ---- canonical split-trace token-out (mirrors full_model_perf.py) ---
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

        result.update(
            {
                "measured_decode_replays": args.decode_tokens,
                "trace_capture_seconds": capture_seconds,
                "token_out_seconds": decode_seconds,
                "token_out_t_s_u": args.decode_tokens / decode_seconds,
                "final_sampled_token": generator._read_sampled_token(),
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
