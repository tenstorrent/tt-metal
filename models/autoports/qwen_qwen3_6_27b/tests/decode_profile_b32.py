# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-op device profile of the *shipped* decode step: all 64 layers, batch 32.

Why this exists.  ``full_model_perf.py`` is the only harness that emits the
``FULL_MODEL_DECODE`` signposts the tracy op report keys off, and it pins
``batch=1`` and takes ``--num-layers`` to shrink the stack -- so every profile in
``doc/`` so far describes a model that is not the one being served.
``full_model_perf_batch.py`` runs the right shape but emits no signposts, so its
ops cannot be separated from construction, prefill and trace capture.

This runs the served shape (batch 32, all 64 layers, TP4) and brackets exactly
``--decode-tokens`` traced replays with signposts, so the report attributes the
real decode step per op.  No server, no vLLM.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--timed-tokens", type=int, default=32, help="timing loop before the profiled replays")
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help=(
            "Capture the decode trace against freshly reset caches instead of a prefilled one. "
            "The decode graph -- programs, shapes, core grids -- is identical either way; only the "
            "values differ. Under the device profiler this matters: a 128-token batch-32 prefill "
            "emits far more markers than the profiler DRAM buffer holds, and the post-processor then "
            "fails with 'Device data missing: Op N not present in cpp_device_perf_report.csv'."
        ),
    )
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=args.max_context,
            batch=args.batch,
            num_layers=args.num_layers,
        )
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt.read_text().strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        all_ids = generator.tokenizer.encode(rendered, add_special_tokens=False)
        token_ids = all_ids[: args.prompt_tokens]
        tokens = torch.tensor([token_ids] * args.batch, dtype=torch.long)

        generator.reset()
        if args.skip_prefill:
            first_token = int(all_ids[0])
        else:
            logits = generator.prefill_forward(
                tokens,
                page_table=generator._page_table,
                kv_cache=generator.kv_cache,
                prompt_lens=[len(token_ids)] * args.batch,
            )
            first_token = int(torch.argmax(logits[0, 0]).item())
        generator._capture_token_out_trace(first_token, len(token_ids))
        generator._seed_token_out_trace(first_token, len(token_ids))
        ttnn.synchronize_device(mesh)

        # Wall-clock the steady state first, so the profiled replays can be
        # checked against a number the profiler did not perturb.
        started = time.perf_counter()
        for _ in range(args.timed_tokens):
            generator.token_out_decode_step(readback=False)
        ttnn.synchronize_device(mesh)
        timed_seconds = time.perf_counter() - started

        # Construction, prefill, compile and capture emit far more markers than
        # the device profiler buffers retain.  Flush before the measured window.
        ttnn.ReadDeviceProfiler(mesh)
        signpost("FULL_MODEL_DECODE", f"batch {args.batch} traced decode replays")
        for _ in range(args.decode_tokens):
            generator.token_out_decode_step(readback=False)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("FULL_MODEL_DECODE_END")

        result = {
            "batch": args.batch,
            "skip_prefill": args.skip_prefill,
            "num_layers": args.num_layers if args.num_layers is not None else len(generator.model.layers),
            "prompt_tokens": len(token_ids),
            "timed_tokens": args.timed_tokens,
            "timed_seconds": timed_seconds,
            "ms_per_token": 1000 * timed_seconds / args.timed_tokens,
            "t_s_u": args.timed_tokens / timed_seconds,
            "profiled_replays": args.decode_tokens,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if generator is not None:
            generator.reset()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
