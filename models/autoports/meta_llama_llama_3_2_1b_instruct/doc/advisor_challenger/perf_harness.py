# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 incumbent/candidate harness for the advisor-challenger stage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.autoports.meta_llama_llama_3_2_1b_instruct.tests.test_functional_decoder import (
    DecodeRotaryHelper,
    LAYER_IDX,
    _make_page_table,
    _synthetic_layer_state_dict,
    _to_tt_decode,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
    try:
        cfg = LlamaConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=16,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=64,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            attention_bias=False,
            mlp_bias=False,
        )
        state = _synthetic_layer_state_dict(cfg)
        max_seq_len = 256
        page_block_size = 64
        _, page_table = _make_page_table(
            mesh, batch=args.batch, max_seq_len=max_seq_len, block_size=page_block_size, seed=3202
        )
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=LAYER_IDX,
            mesh_device=mesh,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
            max_batch_size=args.batch,
        )
        torch.manual_seed(3202)
        # Decode uses [seq=1, batch, hidden]; _to_tt_decode adds the leading
        # decoder dimension to form the physical [1, 1, batch, hidden] tensor.
        host_hidden = torch.randn(1, args.batch, cfg.hidden_size, dtype=torch.bfloat16) * 0.15
        hidden = _to_tt_decode(host_hidden, decoder, mesh)
        host_pos = torch.full((args.batch,), 128, dtype=torch.int32)
        current_pos = ttnn.from_torch(
            host_pos,
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        rotary = DecodeRotaryHelper(LlamaRotaryEmbedding(cfg), 256, cfg.head_dim, mesh).get_rot_mats(host_pos)

        eager = decoder.decode_forward(hidden, current_pos=current_pos, rot_mats=rotary, page_table=page_table)
        ttnn.synchronize_device(mesh)
        eager_host = ttnn.to_torch(eager)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        replay = decoder.decode_forward(hidden, current_pos=current_pos, rot_mats=rotary, page_table=page_table)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)

        replay_host = ttnn.to_torch(replay)
        pcc_passed, pcc_message = comp_pcc(eager_host, replay_host, 0.995)
        repeats_ms = []
        for repeat in range(args.repeats):
            if args.profile and repeat == 0:
                from tracy import signpost

                signpost("CHALLENGER_DECODE")
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            repeats_ms.append((time.perf_counter() - start) * 1000.0)
            if args.profile and repeat == 0:
                signpost("CHALLENGER_DECODE_END")
        ttnn.release_trace(mesh, trace_id)

        payload = {
            "batch": args.batch,
            "repeats_ms": repeats_ms,
            "best_ms": min(repeats_ms),
            "spread_ms": max(repeats_ms) - min(repeats_ms),
            "oracle": "synthetic batch-32 eager-vs-traced-replay PCC >= 0.995",
            "oracle_passed": bool(pcc_passed),
            "oracle_detail": pcc_message,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
