# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 traced-decode harness for the advisor-challenger stage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    PAGE_BLOCK_SIZE,
    PCC_THRESHOLD,
    _assert_pcc,
    _decode_rot_mats,
    _hf_rotary,
    _page_table,
    _real_state_dict,
    _reference_decode,
    _reference_layer,
    _rope_setup,
    _synthetic_state_dict,
    _tt_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)
from models.common.auto_compose import to_torch_auto_compose

try:
    from tracy import signpost
except ImportError:

    def signpost(header: str) -> None:
        del header


def _enum_name(value) -> str:
    return str(value).split(".")[-1]


def _config() -> LlamaConfig:
    # The checkpoint is intentionally not needed for the synthetic full-shape
    # performance oracle. These are the published Llama 3.1 8B dimensions.
    return LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        max_position_embeddings=131072,
        rope_theta=500000.0,
    )


def run(*, repeats: int, real_weights: bool, decode_norm_core_count: int) -> dict:
    batch = 32
    context = 0
    max_seq_len = 128
    max_num_blocks = batch * 2
    state = _real_state_dict() if real_weights else _synthetic_state_dict()
    cfg = _config()
    policy = OptimizedDecoderPolicy()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=104857600)
    try:
        decoder_kwargs = {}
        if decode_norm_core_count != 32:
            raise RuntimeError(
                "Non-incumbent norm-core variants were measured with the temporary "
                "challenger knob and are preserved in norm16_run.json/norm64_run.json; "
                "the losing knob is intentionally absent from the shipped decoder."
            )
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            page_block_size=PAGE_BLOCK_SIZE,
            max_num_blocks=max_num_blocks,
            policy=policy,
            **decoder_kwargs,
        )
        _, page_table = _page_table(mesh, batch=batch, max_num_blocks=max_num_blocks)
        rotary = _hf_rotary(cfg)
        rope = _rope_setup(mesh, cfg, rotary, max_seq_len + 1, batch)
        current_pos_host = torch.full((batch,), context, dtype=torch.int32)
        current_pos = ttnn.from_torch(
            current_pos_host,
            device=mesh,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        rot_mats = _decode_rot_mats(rope, current_pos_host.to(torch.long))
        torch.manual_seed(123)
        hidden = torch.randn(batch, 1, cfg.hidden_size, dtype=torch.bfloat16) * 0.05
        decode_hidden = hidden.transpose(0, 1).unsqueeze(0)
        tt_hidden = ttnn.to_memory_config(
            _tt_tensor(mesh, decode_hidden), decoder.decode_residual_memcfg
        )

        eager = decoder.decode_forward(
            tt_hidden, current_pos=current_pos, rot_mats=rot_mats, page_table=page_table
        )
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = decoder.decode_forward(
            tt_hidden, current_pos=current_pos, rot_mats=rot_mats, page_table=page_table
        )
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)

        samples = []
        for index in range(repeats):
            if index == 0:
                signpost(header="PERF_DECODE")
            started = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - started) * 1000.0)
            if index == 0:
                signpost(header="PERF_DECODE_END")

        traced_host = to_torch_auto_compose(traced)[:, 0, :batch, :].reshape(
            batch, 1, cfg.hidden_size
        )
        eager_host = to_torch_auto_compose(eager)[:, 0, :batch, :].reshape(
            batch, 1, cfg.hidden_size
        )
        eager_trace_pcc = _assert_pcc(
            "batch32_eager_vs_trace", eager_host, traced_host, threshold=0.9999
        )
        reference = _reference_decode(
            _reference_layer(cfg, state), rotary, DynamicCache(), hidden, context
        )
        reference_pcc = _assert_pcc(
            "batch32_reference", reference, traced_host, threshold=PCC_THRESHOLD
        )
        ttnn.release_trace(mesh, trace_id)
        effective = decoder.policy
        return {
            "decode_batch": batch,
            "context": context,
            "decode_norm_core_count": decode_norm_core_count,
            "weights": "real" if real_weights else "synthetic",
            "repeats_ms": samples,
            "best_ms": min(samples),
            "spread_ms": max(samples) - min(samples),
            "oracle": f"HF LlamaDecoderLayer PCC >= {PCC_THRESHOLD}",
            "oracle_passed": reference_pcc >= PCC_THRESHOLD,
            "reference_pcc": reference_pcc,
            "eager_trace_pcc": eager_trace_pcc,
            "effective_policy": {
                "name": effective.name,
                "activation_dtype": _enum_name(effective.activation_dtype),
                "attention_weight_dtype": _enum_name(effective.attention_weight_dtype),
                "mlp_gate_up_dtype": _enum_name(effective.mlp_gate_up_dtype),
                "mlp_down_dtype": _enum_name(effective.mlp_down_dtype),
                "kv_cache_dtype": _enum_name(effective.kv_cache_dtype),
                "mlp_mul_dtype": _enum_name(effective.mlp_mul_dtype),
                "mlp_math_fidelity": _enum_name(effective.mlp_math_fidelity),
            },
        }
    finally:
        ttnn.close_mesh_device(mesh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--decode-norm-core-count", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    result = run(
        repeats=args.repeats,
        real_weights=args.real_weights,
        decode_norm_core_count=args.decode_norm_core_count,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
