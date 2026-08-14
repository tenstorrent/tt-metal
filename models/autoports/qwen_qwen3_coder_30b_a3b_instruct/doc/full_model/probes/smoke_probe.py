# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reduced full-model probe: the whole wrapper with ``--layers`` real layers.

The debugging loop the full-model skill asks for. Every shape, memory config,
cache/page-table layout, terminal norm/LM head and sampling call is the real
one; only the layer count is cut, so a wrapper, trace, cache, page-table,
LM-head or sampler bug reproduces here in a couple of minutes instead of the
ten-plus a 48-layer load costs.

    python .../probes/smoke_probe.py --layers 2 --tokens 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--prompt", type=str, default="Write a Python function that reverses a string.")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        t0 = time.perf_counter()
        gen = build_generator(
            MODEL_DIR,
            mesh,
            override_num_layers=args.layers,
            max_context_len=args.context,
            max_batch_size=args.batch,
        )
        print(f"[smoke] build_generator({args.layers} layers): {time.perf_counter() - t0:.1f}s")

        messages = [{"role": "user", "content": args.prompt}]
        rendered = gen.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = gen.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        print(f"[smoke] prompt tokens: {len(prompt_ids)}")

        t0 = time.perf_counter()
        out = gen.generate(prompt_ids, args.tokens, enable_trace=True, sampling_mode="device")
        dt = time.perf_counter() - t0
        print(f"[smoke] device-sampling generate: {dt:.2f}s for {args.tokens} tokens")
        print(f"[smoke] tokens: {out}")
        print(f"[smoke] text: {gen.tokenizer.decode(out)!r}")
        print(f"[smoke] trace_stats: {gen.trace_stats}")

        # Steady-state host work: replay only.
        before = dict(gen.trace_stats)
        gen.decode_forward(
            None, None, page_table=None, kv_cache=gen._kv_cache, sampling_mode="device", enable_trace=True
        )
        after = dict(gen.trace_stats)
        moved = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        print(f"[smoke] one steady-state replay moved: {moved}")

        # Host-sampling compatibility mode must agree with the device path.
        gen.reset()
        host_out = gen.generate(prompt_ids, min(4, args.tokens), sampling_mode="host")
        print(f"[smoke] host-compat tokens: {host_out}  (device first 4: {out[:4]})")

        print(f"[smoke] fallback audit: {gen.model.runtime_fallback_audit()}")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
