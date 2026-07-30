# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-32 decode measurement/capture harness for advisor-challenger.

The constructor arguments below are deliberately explicit: they are the
shipped policy exercised by the optimized-decoder perf test, not an inspection
of class constructor defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import ttnn

from models.autoports.google_gemma_4_12b.tests import test_functional_decoder as functional
from models.autoports.google_gemma_4_12b.tests import test_optimized_decoder as optimized


BATCH = 32
SEQ_LEN = 128
REPEATS = 3
POLICY = {
    "activation_dtype": "bf16",
    "attention_dtype_by_kind": {"sliding_attention": "bfp8", "full_attention": "bf16"},
    "attention_qkv_dtype_by_kind": {"sliding_attention": "bfp8", "full_attention": "bf16"},
    "attention_o_dtype": "bf16",
    "shared_mlp_gate_up_dtype": "bfp8",
    "shared_mlp_down_dtype": "bfp8",
    "shared_mlp_decode_gate_up_dtype": "bfp8",
    "shared_mlp_decode_down_dtype": "bfp8",
    "kv_cache_dtype": "bf16",
    "fuse_mlp_gelu": True,
    "decode_norm_sharded": True,
    "attention_decode_o_interleaved": False,
}


def _signpost(name: str) -> None:
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


def _build(
    layer_kind: str,
    device,
    *,
    attention_o_interleaved: bool = False,
    attention_head_norm_sharded: bool = False,
):
    text_config = functional._hf_text_config()
    layer_idx = functional._find_layer_idx(text_config, layer_kind)
    hf_layer = functional._synthetic_hf_layer(text_config, layer_idx)
    OptimizedDecoder = optimized._load_optimized_decoder_class()
    candidate_kwargs = (
        {"attention_decode_head_norm_sharded": True} if attention_head_norm_sharded else {}
    )
    decoder = OptimizedDecoder.from_state_dict(
        hf_layer.state_dict(),
        hf_config=text_config,
        layer_idx=layer_idx,
        mesh_device=device,
        dtype=ttnn.bfloat16,
        attention_dtype=ttnn.bfloat8_b if layer_kind == "sliding_attention" else ttnn.bfloat16,
        attention_qkv_dtype=ttnn.bfloat8_b if layer_kind == "sliding_attention" else ttnn.bfloat16,
        attention_o_dtype=ttnn.bfloat16,
        shared_mlp_dtype=ttnn.bfloat8_b,
        shared_mlp_down_dtype=ttnn.bfloat8_b,
        shared_mlp_decode_dtype=ttnn.bfloat8_b,
        shared_mlp_decode_down_dtype=ttnn.bfloat8_b,
        kv_cache_dtype=ttnn.bfloat16,
        fuse_mlp_gelu=True,
        decode_norm_sharded=True,
        attention_decode_o_interleaved=attention_o_interleaved,
        **candidate_kwargs,
    )

    blocks_per_user = math.ceil((SEQ_LEN + 1) / functional.BLOCK_SIZE) + 1
    max_num_blocks = BATCH * blocks_per_user
    page_table_cpu = torch.arange(max_num_blocks, dtype=torch.int32).reshape(BATCH, blocks_per_user)
    page_table = functional._to_tt(
        page_table_cpu, device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )
    kv_cache = decoder.create_paged_kv_cache(
        block_size=functional.BLOCK_SIZE, max_num_blocks=max_num_blocks
    )
    _, rope2 = functional._rope_tables(text_config, device, SEQ_LEN + 64, layer_idx)

    torch.manual_seed(20260730 + layer_idx)
    hidden = torch.randn(1, 1, BATCH, text_config.hidden_size, dtype=torch.bfloat16)
    hidden_tt = functional._to_tt(hidden, device)
    positions = torch.full((1, BATCH), SEQ_LEN, dtype=torch.uint32)
    cache_positions = torch.full((BATCH,), SEQ_LEN, dtype=torch.int32)
    position_idx = functional._to_tt(
        positions, device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    position_idx_cache = functional._to_tt(
        cache_positions, device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )
    decoder._last_prefill_seq_len = SEQ_LEN
    kwargs = {
        "rope_mats": rope2,
        "page_table": page_table,
        "kv_cache": kv_cache,
        "position_idx": position_idx,
        "position_idx_cache": position_idx_cache,
    }
    return decoder, hidden_tt, kwargs


def measure(
    layer_kind: str,
    device,
    repeats: int,
    *,
    attention_o_interleaved: bool = False,
    attention_head_norm_sharded: bool = False,
):
    decoder, hidden, kwargs = _build(
        layer_kind,
        device,
        attention_o_interleaved=attention_o_interleaved,
        attention_head_norm_sharded=attention_head_norm_sharded,
    )
    decoder.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = decoder.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)

    repeats_ms = []
    for repeat in range(repeats):
        _signpost(f"PERF_DECODE_B32_R{repeat}")
        start = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)
        repeats_ms.append((time.perf_counter() - start) * 1000.0)
        _signpost(f"PERF_DECODE_B32_R{repeat}_END")
    host_output = functional._from_tt(output, device)
    ttnn.release_trace(device, trace_id)
    return {
        "layer_kind": layer_kind,
        "decode_batch": BATCH,
        "sequence_length": SEQ_LEN,
        "repeats_ms": repeats_ms,
        "best_ms": min(repeats_ms),
        "output_checksum": float(host_output.float().sum()),
        "output_l2": float(torch.linalg.vector_norm(host_output.float())),
        "optimization_summary": decoder.optimization_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-kind", choices=("sliding_attention", "full_attention"), required=True)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-o-interleaved", action="store_true")
    parser.add_argument("--attention-head-norm-sharded", action="store_true")
    args = parser.parse_args()
    device = ttnn.open_device(device_id=0, trace_region_size=32 * 1024 * 1024)
    try:
        result = measure(
            args.layer_kind,
            device,
            args.repeats,
            attention_o_interleaved=args.attention_o_interleaved,
            attention_head_norm_sharded=args.attention_head_norm_sharded,
        )
    finally:
        ttnn.close_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
