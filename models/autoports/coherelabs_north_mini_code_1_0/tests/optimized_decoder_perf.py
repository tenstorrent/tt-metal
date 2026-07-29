# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed optimized-decoder latency, candidate-sweep, and Tracy harness."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.functional_decoder_perf import _decode, _prefill
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _synthetic_state,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import (
    MODEL_ID,
    OptimizationConfig,
    OptimizedDecoder,
)

DTYPES = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
}
FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
}


def _policy(args):
    cfg = OptimizationConfig()
    overrides = {}
    for field, value in (
        ("attention_weight_dtype", args.attention_dtype),
        ("dense_gate_up_dtype", args.gate_up_dtype),
        ("dense_down_dtype", args.down_dtype),
        ("expert_gate_up_dtype", args.expert_gate_up_dtype),
        ("expert_down_dtype", args.expert_down_dtype),
        ("dense_expert_gate_up_dtype", args.dense_expert_gate_up_dtype),
        ("dense_expert_down_dtype", args.dense_expert_down_dtype),
        ("kv_cache_dtype", args.kv_dtype),
    ):
        if value is not None:
            overrides[field] = DTYPES[value]
    for field, value in (
        ("attention_fidelity", args.attention_fidelity),
        ("dense_gate_up_fidelity", args.gate_up_fidelity),
        ("dense_down_fidelity", args.down_fidelity),
        ("expert_gate_up_fidelity", args.expert_gate_up_fidelity),
        ("expert_down_fidelity", args.expert_down_fidelity),
    ):
        if value is not None:
            overrides[field] = FIDELITIES[value]
    for field in (
        "decode_qkv_cores",
        "decode_o_cores",
        "decode_dense_gate_up_cores",
        "decode_dense_down_cores",
        "decode_qkv_in0_block_w",
        "decode_o_in0_block_w",
        "decode_dense_gate_up_in0_block_w",
        "decode_dense_down_in0_block_w",
        "sparse_gate_up_in0_block_w",
        "sparse_down_in0_block_w",
        "moe_chunk_size",
        "serving_decode_qkv_cores",
        "serving_decode_o_cores",
        "serving_decode_dense_gate_up_cores",
        "serving_decode_dense_down_cores",
        "serving_decode_qkv_in0_block_w",
        "serving_decode_o_in0_block_w",
        "serving_decode_dense_gate_up_in0_block_w",
        "serving_decode_dense_down_in0_block_w",
        "decode_residual_cores",
        "serving_decode_residual_cores",
        "dense_expert_chunk_size",
        "dense_expert_cores",
        "dense_expert_gate_up_in0_block_w",
        "dense_expert_gate_up_per_core_m",
        "dense_expert_gate_up_per_core_n",
        "dense_expert_gate_up_subblock_h",
        "dense_expert_gate_up_subblock_w",
        "dense_expert_down_in0_block_w",
        "dense_expert_down_per_core_m",
        "dense_expert_down_per_core_n",
        "dense_expert_down_subblock_h",
        "dense_expert_down_subblock_w",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if args.split_dense_gate_up:
        overrides["packed_dense_gate_up"] = False
    if args.packed_dense_gate_up:
        overrides["packed_dense_gate_up"] = True
    if args.separate_kv_update:
        overrides["fused_kv_update"] = False
    if args.sparse_intermediate_dram:
        overrides["sparse_intermediate_dram"] = True
    if args.serving_fused_kv_update:
        overrides["serving_fused_kv_update"] = True
    if args.default_sdpa:
        overrides["explicit_sdpa_program"] = False
    if args.direct_o_input:
        overrides["direct_o_input"] = True
    if args.direct_down_input:
        overrides["direct_down_input"] = True
    if args.packed_dense_experts:
        overrides["packed_dense_experts"] = True
    if args.sparse_gate_grid_x is not None or args.sparse_gate_grid_y is not None:
        overrides["sparse_gate_up_grid"] = (
            args.sparse_gate_grid_x or cfg.sparse_gate_up_grid[0],
            args.sparse_gate_grid_y or cfg.sparse_gate_up_grid[1],
        )
    if args.sparse_down_grid_x is not None or args.sparse_down_grid_y is not None:
        overrides["sparse_down_grid"] = (
            args.sparse_down_grid_x or cfg.sparse_down_grid[0],
            args.sparse_down_grid_y or cfg.sparse_down_grid[1],
        )
    return replace(cfg, **overrides)


def _render_policy(policy):
    result = {}
    for key, value in asdict(policy).items():
        if value in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b):
            result[key] = str(value)
        elif value in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0, choices=(0, 1, 4))
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--candidate", default="default")
    for name in (
        "attention-dtype",
        "gate-up-dtype",
        "down-dtype",
        "expert-gate-up-dtype",
        "expert-down-dtype",
        "dense-expert-gate-up-dtype",
        "dense-expert-down-dtype",
        "kv-dtype",
    ):
        parser.add_argument(f"--{name}", choices=tuple(DTYPES))
    for name in (
        "attention-fidelity",
        "gate-up-fidelity",
        "down-fidelity",
        "expert-gate-up-fidelity",
        "expert-down-fidelity",
    ):
        parser.add_argument(f"--{name}", choices=tuple(FIDELITIES))
    for name in (
        "decode-qkv-cores",
        "decode-o-cores",
        "decode-dense-gate-up-cores",
        "decode-dense-down-cores",
        "decode-qkv-in0-block-w",
        "decode-o-in0-block-w",
        "decode-dense-gate-up-in0-block-w",
        "decode-dense-down-in0-block-w",
        "sparse-gate-up-in0-block-w",
        "sparse-down-in0-block-w",
        "moe-chunk-size",
        "serving-decode-qkv-cores",
        "serving-decode-o-cores",
        "serving-decode-dense-gate-up-cores",
        "serving-decode-dense-down-cores",
        "serving-decode-qkv-in0-block-w",
        "serving-decode-o-in0-block-w",
        "serving-decode-dense-gate-up-in0-block-w",
        "serving-decode-dense-down-in0-block-w",
        "decode-residual-cores",
        "serving-decode-residual-cores",
        "dense-expert-chunk-size",
        "dense-expert-cores",
        "dense-expert-gate-up-in0-block-w",
        "dense-expert-gate-up-per-core-m",
        "dense-expert-gate-up-per-core-n",
        "dense-expert-gate-up-subblock-h",
        "dense-expert-gate-up-subblock-w",
        "dense-expert-down-in0-block-w",
        "dense-expert-down-per-core-m",
        "dense-expert-down-per-core-n",
        "dense-expert-down-subblock-h",
        "dense-expert-down-subblock-w",
    ):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--split-dense-gate-up", action="store_true")
    parser.add_argument("--packed-dense-gate-up", action="store_true")
    parser.add_argument("--separate-kv-update", action="store_true")
    parser.add_argument("--sparse-intermediate-dram", action="store_true")
    parser.add_argument("--serving-fused-kv-update", action="store_true")
    parser.add_argument("--default-sdpa", action="store_true")
    parser.add_argument("--direct-o-input", action="store_true")
    parser.add_argument("--direct-down-input", action="store_true")
    parser.add_argument("--packed-dense-experts", action="store_true")
    parser.add_argument("--sparse-gate-grid-x", type=int)
    parser.add_argument("--sparse-gate-grid-y", type=int)
    parser.add_argument("--sparse-down-grid-x", type=int)
    parser.add_argument("--sparse-down-grid-y", type=int)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION, local_files_only=True)
    policy = _policy(args)
    state = _synthetic_state(config, args.layer, sparse_weights=args.layer != 0)
    max_cache_len = args.sequence if args.mode == "prefill" else 32
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=max_cache_len,
            optimization_config=policy,
        )
        if args.mode == "prefill":
            result = _prefill(
                decoder,
                mesh_device,
                config,
                sequence=args.sequence,
                warmups=args.warmups,
                iterations=args.iterations,
            )
        else:
            result = _decode(
                decoder,
                mesh_device,
                config,
                warmups=args.warmups,
                iterations=args.iterations,
            )
    finally:
        ttnn.close_mesh_device(mesh_device)
    result.update(
        {
            "candidate": args.candidate,
            "mode": args.mode,
            "batch": args.batch,
            "sequence": args.sequence,
            "layer": args.layer,
            "model_revision": REAL_REVISION,
            "policy": _render_policy(decoder.optimization_config),
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
