# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Time canonical Falcon3 split traces at the vLLM physical decode shape."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import build_generator


def _timed(mesh, fn, iterations: int) -> float:
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    ttnn.synchronize_device(mesh)
    return time.perf_counter() - start


def collect(args: argparse.Namespace) -> dict:
    mesh = None
    generator = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=512_000_000)
        generator = build_generator(
            args.model_dir,
            mesh,
            max_batch_size=32,
            max_context_len=32768,
            weight_cache_path=args.weight_cache_path,
        )
        generator.set_sampling_params(active_batch=1)
        kv_cache = generator.model.allocate_kv_cache(paged=True, num_blocks=4128)

        # Match vLLM's fixed [max_num_seqs, max_model_len / block_size]
        # table. Only the live request row is mapped; inactive rows stay -1.
        page_table = torch.full((32, 1024), -1, dtype=torch.int32)
        mapped_pages = generator._sdpa_rounded_page_count(args.position + args.iterations + 1)
        page_table[0, :mapped_pages] = torch.arange(mapped_pages, dtype=torch.int32)
        host_inputs = generator._prepare_decode_host_inputs(
            torch.tensor([args.token]), torch.tensor([args.position]), page_table
        )
        generator._refresh_trace_state(host_inputs, kv_cache, active_batch=1)

        model_s = _timed(
            mesh,
            lambda: ttnn.execute_trace(mesh, generator._trace_model_id, cq_id=0, blocking=False),
            args.iterations,
        )
        sampling_s = _timed(
            mesh,
            lambda: ttnn.execute_trace(mesh, generator._trace_sampling_id, cq_id=0, blocking=False),
            args.iterations,
        )
        pair_s = _timed(mesh, generator._replay_split_sampling, args.iterations)

        def replay_and_read() -> None:
            generator._sampled_to_torch(generator._replay_split_sampling())

        caller_s = _timed(mesh, replay_and_read, args.iterations)
        token, position, rotary, traced_page_table = generator._trace_inputs
        result = {
            "contract": {
                "max_batch_size": 32,
                "active_batch": 1,
                "max_context_len": 32768,
                "page_block_size": generator.model.page_block_size,
                "external_cache_blocks": 4128,
                "iterations": args.iterations,
            },
            "physical_shapes": {
                "token": list(token.shape),
                "position": list(position.shape),
                "rotary_position": list(rotary.shape),
                "page_table": list(traced_page_table.shape),
                "sampled_token": list(generator._trace_sampled.shape),
                "decoder_residual": [1, 1, generator.model.max_batch_size, generator.model.hidden_size],
                "kv_cache_per_layer": [list(tensor.shape) for tensor in kv_cache[0]],
            },
            "timings": {
                "model_trace_ms_per_token": model_s * 1000 / args.iterations,
                "sampling_trace_ms_per_token": sampling_s * 1000 / args.iterations,
                "pair_ms_per_token": pair_s * 1000 / args.iterations,
                "pair_t_s_u": args.iterations / pair_s,
                "caller_visible_ms_per_token": caller_s * 1000 / args.iterations,
                "caller_visible_t_s_u": args.iterations / caller_s,
            },
            "trace_stats": dict(generator.trace_stats),
            "precision_summary": generator.model.precision_summary(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-cache-path", required=True)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--position", type=int, default=128)
    parser.add_argument("--token", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1 or args.position + args.iterations >= 32768:
        parser.error("iterations must be positive and remain within the 32768-token context")
    collect(args)


if __name__ == "__main__":
    main()
