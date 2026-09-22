# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synchronized two-layer prefill diagnostic; timings are not HTTP benchmark results."""

import argparse
import json
import time
from collections import defaultdict

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--variant", choices=["sdpa", "flatten"], default="sdpa")
    args = parser.parse_args()
    policy_key = "batched_prefill_sdpa" if args.variant == "sdpa" else "flatten_prefill_batch"
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    originals = []
    try:
        gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=[0, 3])
        gen.batched_prefill = True
        cache = gen._ensure_cache(args.batch, ((args.length + 31) // 32) * 32)
        tokens = (torch.arange(args.length) % 256 + 100).repeat(args.batch, 1)
        tokens += torch.arange(args.batch)[:, None] * 13
        phases, counts = defaultdict(float), defaultdict(int)

        def wrap(owner, name, label):
            original = getattr(owner, name)
            originals.append((owner, name, original))

            def measured(*a, **kw):
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                result = original(*a, **kw)
                ttnn.synchronize_device(mesh)
                phases[label] += time.perf_counter() - begin
                counts[label] += 1
                return result

            setattr(owner, name, measured)

        for variant in (False, True):
            for layer in gen.model.layers:
                layer.policy[policy_key] = variant
            for repeat in range(2):
                gen.reset()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                output = gen.prefill_forward(
                    tokens, page_table=gen.page_table, kv_cache=cache, prompt_lens=[args.length] * args.batch
                )
                ttnn.synchronize_device(mesh)
                print(
                    "UNINSTRUMENTED",
                    json.dumps(
                        dict(variant=args.variant, enabled=variant, repeat=repeat, seconds=time.perf_counter() - begin)
                    ),
                    flush=True,
                )
                del output
        for layer in gen.model.layers:
            for name in ("_delta", "_full_prefill", "_finish", "_linear"):
                wrap(layer, name, layer.kind + "." + name)
        wrap(ttnn.transformer, "chunk_gated_delta_rule", "subset.native_gated_delta")
        wrap(ttnn.transformer, "chunked_scaled_dot_product_attention", "subset.native_sdpa")
        wrap(ttnn.experimental.kda, "qkv_causal_conv1d_silu", "subset.native_conv")
        for variant in (False, True):
            for layer in gen.model.layers:
                layer.policy[policy_key] = variant
            gen.reset()
            phases.clear()
            counts.clear()
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            output = gen.prefill_forward(
                tokens, page_table=gen.page_table, kv_cache=cache, prompt_lens=[args.length] * args.batch
            )
            ttnn.synchronize_device(mesh)
            print(
                "PREFILL_PHASES",
                json.dumps(
                    dict(
                        batch=args.batch,
                        length=args.length,
                        variant=args.variant,
                        enabled=variant,
                        instrumented_seconds=time.perf_counter() - begin,
                        seconds=dict(phases),
                        counts=dict(counts),
                    )
                ),
                flush=True,
            )
            del output
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
