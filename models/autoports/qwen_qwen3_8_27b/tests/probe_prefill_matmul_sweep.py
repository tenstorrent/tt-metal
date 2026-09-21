# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-process prefill geometry sweep, preserving weights and arithmetic fidelity."""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy

VARIANTS = {
    "baseline": {},
    "flat_projections": dict(flatten_prefill_batch=True),
    "batched_cache_fill": dict(batched_prefill_cache_fill=True),
    "packed_prefill_conv": dict(experimental_packed_prefill_conv=True),
    "split_mlp": dict(prefill_split_mlp=True),
    "links1": dict(num_links=1),
    "links4": dict(num_links=4),
    "row_parallel_norm": dict(prefill_row_parallel_norm=True),
    "chunk2048": dict(batched_prefill_chunk_size=2048),
    "chunk8192": dict(batched_prefill_chunk_size=8192),
    "chunk8192_inner": dict(batched_prefill_chunk_size=8192, chunk_size=8192),
    "compact_head": dict(prefill_compact_head=True),
    "batched_head": dict(prefill_batched_head=True),
    "skip_intermediate_head": dict(skip_intermediate_prefill_head=True),
    "sharded_norm": dict(prefill_sharded_residual=True),
    "sharded_norm_rect": dict(prefill_sharded_residual=True, prefill_norm_rectangular=True),
    "sharded_norm_fp32": dict(prefill_sharded_residual=True, prefill_norm_stats_dtype="float32"),
    "sharded_norm_rect_fp32": dict(
        prefill_sharded_residual=True, prefill_norm_rectangular=True, prefill_norm_stats_dtype="float32"
    ),
    "sharded_replicated_norm": dict(prefill_sharded_residual=True, prefill_replicated_norm=True),
    "mmrs": dict(output_scheme="mmrs", fused_n=8, fused_grid=[10, 8], flatten_prefill_batch=True),
    "mmrswide": dict(output_scheme="mmrs", fused_n=8, fused_grid=[11, 8], flatten_prefill_batch=True),
    "m8n8": dict(minimal_m=8, minimal_n=8),
    "m4n16k4": dict(minimal_m=4, minimal_n=16, minimal_k=4),
    "m8n16k4": dict(minimal_m=8, minimal_n=16, minimal_k=4),
    "m16n8k4": dict(minimal_m=16, minimal_n=8, minimal_k=4),
    "subblock2x2": dict(minimal_subblock_h=2, minimal_subblock_w=2),
    "subblock4x1": dict(minimal_subblock_h=4, minimal_subblock_w=1),
    "m8n16": dict(minimal_m=8, minimal_n=16),
    "m16n8": dict(minimal_m=16, minimal_n=8),
    "m16n16": dict(minimal_m=16, minimal_n=16),
    "auto": dict(minimal_prefill=False, minimal_prefill_roles=[]),
    "2d8x8": dict(minimal_prefill=False, minimal_prefill_roles=[], prefill_2d=True, prefill_grid=[8, 8]),
    "2d10x8": dict(minimal_prefill=False, minimal_prefill_roles=[], prefill_2d=True, prefill_grid=[10, 8]),
    "2dblocked": dict(
        minimal_prefill=False,
        minimal_prefill_roles=[],
        prefill_2d=True,
        prefill_grid=[8, 8],
        prefill_out_block_h=4,
        prefill_out_block_w=4,
    ),
    "fusedmlp": dict(minimal_mlp=True),
    "2dblocked1": dict(
        minimal_prefill=False,
        minimal_prefill_roles=[],
        prefill_2d=True,
        prefill_grid=[8, 8],
        prefill_out_block_h=4,
        prefill_out_block_w=1,
    ),
    "sdpa_q256k128": dict(prefill_sdpa_q=256, prefill_sdpa_k=128),
    "sdpa_q128k256": dict(prefill_sdpa_q=128, prefill_sdpa_k=256),
    "sdpa_q256k256": dict(prefill_sdpa_q=256, prefill_sdpa_k=256),
    "sdpa_q128k512": dict(prefill_sdpa_q=128, prefill_sdpa_k=512),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--load-fused-mlp", action="store_true")
    parser.add_argument("--load-sharded-prefill", action="store_true")
    parser.add_argument(
        "--base-best", action="store_true", help="Start every arm with norm-preserving layout and heads"
    )
    parser.add_argument("--check-state", action="store_true", help="Compare all cache/state elements outside timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    report = dict(batch=args.batch, length=args.length, full=args.full, base_best=args.base_best, rows=[])
    try:

        def policy(precision, layer):
            return {
                **decoder_policy(precision, layer),
                "minimal_mlp": args.load_fused_mlp,
                "prefill_sharded_residual": args.load_sharded_prefill or args.base_best,
            }

        with patch("models.autoports.qwen_qwen3_8_27b.tt.model.decoder_policy", side_effect=policy):
            gen = build_generator(
                "models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=None if args.full else [0, 3]
            )
        for layer in gen.model.layers:
            layer.policy["minimal_mlp"] = False
        gen.batched_prefill = True
        cache = gen._ensure_cache(args.batch, ((args.length + 31) // 32) * 32)
        policies = [dict(layer.policy) for layer in gen.model.layers]
        tokens = (torch.arange(args.length) % 256 + 100).repeat(args.batch, 1)
        tokens += torch.arange(args.batch)[:, None] * 13
        reference = None
        reference_state = None
        for name in args.variants.split(","):
            variant = {
                **(
                    dict(
                        prefill_sharded_residual=True,
                        prefill_replicated_norm=True,
                        prefill_batched_head=True,
                        skip_intermediate_prefill_head=True,
                    )
                    if args.base_best
                    else {}
                ),
                **VARIANTS[name],
            }
            gen.model.prefill_sharded_residual = variant.get("prefill_sharded_residual", False)
            gen.model.prefill_compact_head = variant.get("prefill_compact_head", False)
            gen.model.prefill_batched_head = variant.get("prefill_batched_head", False)
            gen.skip_intermediate_prefill_head = variant.get("skip_intermediate_prefill_head", False)
            gen.batched_prefill_chunk_size = variant.get("batched_prefill_chunk_size", 4096)
            for layer, baseline in zip(gen.model.layers, policies):
                layer.policy = {**baseline, **variant}
                layer.CHUNK_SIZE = layer.policy["chunk_size"]
            for repeat in range(3):
                gen.reset()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                outputs = gen.prefill_forward(
                    tokens, page_table=gen.page_table, kv_cache=cache, prompt_lens=[args.length] * args.batch
                )
                ttnn.synchronize_device(mesh)
                seconds = time.perf_counter() - begin
                actual = torch.cat([gen._host_logits(output) for output in outputs], dim=2)
                if reference is None:
                    reference = actual.clone()
                pcc = torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1].item()
                row = dict(
                    variant=name,
                    repeat=repeat,
                    seconds=seconds,
                    pcc=pcc,
                    exact=torch.equal(actual, reference),
                    top1_equal=torch.equal(actual.argmax(-1), reference.argmax(-1)),
                )
                if args.check_state:
                    state = {
                        f"{i}.{field}": ttnn.to_torch(
                            tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
                        ).clone()
                        for i, layer_state in enumerate(cache.layers)
                        for field in ("key", "value", "conv", "recurrent")
                        if (tensor := getattr(layer_state, field, None)) is not None
                    }
                    if reference_state is None:
                        reference_state = state
                    row["state_exact"] = {
                        name: torch.equal(tensor, reference_state[name]) for name, tensor in state.items()
                    }
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("PREFILL_MATMUL_SWEEP", json.dumps(row), flush=True)
                del outputs, actual
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
