# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed optimized-decoder latency and Tracy-signpost harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.functional_decoder_perf import _decode, _prefill
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _synthetic_state,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import MODEL_ID
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0, choices=(0, 1, 4))
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dense-gate-up-in0-block-w", type=int, default=8)
    parser.add_argument("--dense-down-in0-block-w", type=int, default=12)
    parser.add_argument("--sparse-weight-dtype", choices=("auto", "bfp8", "bfp4"), default="auto")
    parser.add_argument("--sparse-cores", type=int, default=11)
    parser.add_argument("--sparse-gate-up-in0-block-w", type=int, default=32)
    parser.add_argument("--sparse-down-in0-block-w", type=int, default=24)
    parser.add_argument("--attention-weight-dtype", choices=("bfp8", "bfp4"), default="bfp8")
    parser.add_argument(
        "--attention-decode-variant",
        choices=("auto", "advisor_1d", "dram_sharded"),
        default="auto",
    )
    parser.add_argument("--sparse-compute-fidelity", choices=("lofi", "hifi2"), default="lofi")
    parser.add_argument("--attention-compute-fidelity", choices=("lofi", "hifi2"), default="lofi")
    parser.add_argument("--sparse-gate-up-out-subblock-w", type=int, default=1)
    parser.add_argument("--sparse-down-out-subblock-w", type=int, default=1)
    parser.add_argument("--prefill-program-variant", choices=("auto", "default", "expert_2d"), default="auto")
    parser.add_argument("--kv-cache-dtype", choices=("bf16", "bfp8", "bfp4"), default="bf16")
    parser.add_argument("--sdpa-decode-variant", choices=("default", "k32", "k64"), default="default")
    parser.add_argument(
        "--dense-decode-variant",
        choices=(
            "auto",
            "packed_interleaved",
            "advisor_dram_sharded",
            "advisor_dram_sharded_bfp4_gate_up",
            "advisor_dram_sharded_bfp4_all",
            "separate_dram_sharded_bfp4",
        ),
        default="auto",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION)
    state = _synthetic_state(config, args.layer)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=args.sequence if args.mode == "prefill" else 32,
            dense_decode_variant=args.dense_decode_variant,
            dense_gate_up_in0_block_w=args.dense_gate_up_in0_block_w,
            dense_down_in0_block_w=args.dense_down_in0_block_w,
            sparse_weight_dtype=args.sparse_weight_dtype,
            sparse_cores=args.sparse_cores,
            sparse_gate_up_in0_block_w=args.sparse_gate_up_in0_block_w,
            sparse_down_in0_block_w=args.sparse_down_in0_block_w,
            attention_weight_dtype=args.attention_weight_dtype,
            attention_decode_variant=args.attention_decode_variant,
            sparse_compute_fidelity=args.sparse_compute_fidelity,
            attention_compute_fidelity=args.attention_compute_fidelity,
            sparse_gate_up_out_subblock_w=args.sparse_gate_up_out_subblock_w,
            sparse_down_out_subblock_w=args.sparse_down_out_subblock_w,
            prefill_program_variant=args.prefill_program_variant,
            kv_cache_dtype=args.kv_cache_dtype,
            sdpa_decode_variant=args.sdpa_decode_variant,
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
            result = _decode(decoder, mesh_device, config, warmups=args.warmups, iterations=args.iterations)
    finally:
        ttnn.close_mesh_device(mesh_device)
    result.update(
        {
            "implementation": "OptimizedDecoder",
            "mode": args.mode,
            "batch": args.batch,
            "sequence": args.sequence,
            "layer": args.layer,
            "model_revision": REAL_REVISION,
            "dense_decode_variant": args.dense_decode_variant,
            "dense_gate_up_in0_block_w": args.dense_gate_up_in0_block_w,
            "dense_down_in0_block_w": args.dense_down_in0_block_w,
            "sparse_weight_dtype": args.sparse_weight_dtype,
            "sparse_cores": args.sparse_cores,
            "sparse_gate_up_in0_block_w": args.sparse_gate_up_in0_block_w,
            "sparse_down_in0_block_w": args.sparse_down_in0_block_w,
            "attention_weight_dtype": args.attention_weight_dtype,
            "attention_decode_variant": args.attention_decode_variant,
            "sparse_compute_fidelity": args.sparse_compute_fidelity,
            "attention_compute_fidelity": args.attention_compute_fidelity,
            "sparse_gate_up_out_subblock_w": args.sparse_gate_up_out_subblock_w,
            "sparse_down_out_subblock_w": args.sparse_down_out_subblock_w,
            "prefill_program_variant": args.prefill_program_variant,
            "kv_cache_dtype": args.kv_cache_dtype,
            "sdpa_decode_variant": args.sdpa_decode_variant,
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
