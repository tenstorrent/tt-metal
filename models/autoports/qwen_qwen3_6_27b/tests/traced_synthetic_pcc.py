# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Nonzero synthetic traced-decode regression for both Qwen3.5 layer kinds.

The trace captures token 0 and replays tokens 1 and 2 after copying new hidden
states and positions into the original device buffers.  The HF oracle advances
the same cache/state one token at a time.  This makes stale trace inputs,
positions, KV cache, convolution state, recurrent state, and batch-row aliasing
observable while keeping the reference memory bounded.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tracy import signpost
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER as FULL_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _hf_layer as full_hf_layer
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER as LINEAR_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _hf_layer as linear_hf_layer
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, FunctionalDecoder, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder
from models.common.utility_functions import comp_pcc


def _host_tensor(value, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(value, layout=layout, dtype=dtype)


def _copy_host(value, destination, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    source = _host_tensor(value, layout=layout, dtype=dtype)
    ttnn.copy_host_to_device_tensor(source, destination, cq_id=0)


def _reference_steps(kind, layer, config, tokens):
    cache = DynamicCache(config=config)
    outputs = []
    rotary = Qwen3_5TextRotaryEmbedding(config) if kind == "full" else None
    for position, token in enumerate(tokens):
        if kind == "full":
            positions = torch.full((token.shape[0], 1), position, dtype=torch.long)
            position_ids = positions.unsqueeze(0).expand(3, -1, -1)
            position_embeddings = rotary(token, position_ids)
        else:
            positions = None
            position_embeddings = (None, None)
        outputs.append(
            layer(
                token,
                position_embeddings=position_embeddings,
                position_ids=positions,
                attention_mask=None,
                past_key_values=cache,
            )
        )
    return outputs


@torch.no_grad()
def run(kind, batch, optimized=False, candidate="default", steps=3):
    # Make any Python/host fallback in the measured TTNN path a hard failure.
    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FALLBACK_AUDIT", f"throw_exception_on_fallback={ttnn.CONFIG.throw_exception_on_fallback}")
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    config._attn_implementation = "eager"
    if kind == "full":
        layer_idx = FULL_LAYER
        state = full_state(config)
        hf_layer = full_hf_layer(config, state)
    else:
        layer_idx = LINEAR_LAYER
        state = linear_state(config)
        hf_layer = linear_hf_layer(config, state)

    # A row offset makes batch-row aliasing visible even when the random stream
    # happens to produce highly correlated activations.
    row_offset = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1) * 0.01
    tokens = [
        ((torch.randn(batch, 1, config.hidden_size) * 0.2) + row_offset + step * 0.03).bfloat16()
        for step in range(steps)
    ]
    references = _reference_steps(kind, hf_layer, config, tokens)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace_id = None
    try:
        decoder_cls = OptimizedDecoder if optimized else FunctionalDecoder
        decoder = decoder_cls.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh,
            batch=batch,
            max_context=64,
            page_size=64,
            **({"candidate": candidate} if optimized else {}),
        )
        hidden_device = _to_device(tokens[0].reshape(1, 1, batch, -1), mesh_device=mesh)
        page_values = torch.arange(batch, dtype=torch.int32).reshape(batch, 1).flip(0)
        page_table = _to_device(
            page_values,
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        positions_device = _to_device(
            torch.zeros(batch, dtype=torch.uint32),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )

        def decode():
            return decoder.decode_forward(
                hidden_states=hidden_device,
                page_table=page_table,
                current_positions=positions_device,
            )

        # Compile every op before capture.  Snapshot/restore the mutable cache
        # so capture still represents oracle step 0 (essential for the
        # non-idempotent linear-attention recurrence).
        cache_names = ("key", "value") if kind == "full" else ("conv", "recurrent")
        initial_cache = {name: ttnn.to_torch(ttnn.get_device_tensors(decoder.caches[name])[0]) for name in cache_names}
        decode()
        ttnn.synchronize_device(mesh)
        for name in cache_names:
            # Optimized full attention intentionally uses a compressed KV
            # cache, and an optimized conv state may be re-laid row-major, so
            # restore using the destination's actual device dtype and layout
            # rather than assuming tiled bfloat16.
            _copy_host(
                initial_cache[name],
                decoder.caches[name],
                layout=decoder.caches[name].layout,
                dtype=decoder.caches[name].dtype,
            )
        ttnn.synchronize_device(mesh)

        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = decode()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)

        actual_steps = []
        replay_times = []
        pcc_messages = []
        signpost("PERF_DECODE")
        for step in range(1, steps):
            _copy_host(tokens[step].reshape(1, 1, batch, -1), hidden_device)
            _copy_host(
                torch.full((batch,), step, dtype=torch.uint32),
                positions_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            started = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            replay_times.append((time.perf_counter() - started) * 1000)
            actual = ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0])
            actual_steps.append(actual.reshape(batch, 1, config.hidden_size))
            passed, message = comp_pcc(references[step].float(), actual_steps[-1].float(), 0.995)
            print(
                f"{kind.upper()}_TRACED_SYNTHETIC_PCC",
                f"batch={batch}",
                f"step={step}",
                message,
            )
            pcc_messages.append(str(message))
            assert passed, message
        signpost("PERF_DECODE_END")
        print(
            f"{kind.upper()}_TRACED_SYNTHETIC_LATENCY",
            f"path={'optimized' if optimized else 'functional'}",
            f"candidate={candidate if optimized else 'functional'}",
            f"batch={batch}",
            f"median_ms={torch.tensor(replay_times).median().item():.6f}",
            f"min_ms={min(replay_times):.6f}",
        )

        assert not torch.equal(actual_steps[0], actual_steps[1]), "trace output ignored new input"
        if batch == 32:
            row_delta = (actual_steps[-1][1:] - actual_steps[-1][:-1]).abs().max(dim=-1).values
            assert torch.all(row_delta > 0), "one or more batch rows aliased"
        policy = decoder.policy if optimized else None
        return {
            "command_result": "pass",
            "kind": kind,
            "batch": batch,
            "path": "optimized" if optimized else "functional",
            "candidate": candidate if optimized else "functional",
            "steps": steps,
            "pcc": pcc_messages,
            "median_ms": float(torch.tensor(replay_times).median().item()),
            "min_ms": float(min(replay_times)),
            "watcher_enabled": os.environ.get("TT_METAL_WATCHER") == "1",
            "policy": (
                None
                if policy is None
                else {
                    "attention_weight_dtype": str(policy.attention_weight_dtype),
                    "mlp_gate_up_dtype": str(policy.mlp_gate_up_dtype),
                    "mlp_down_dtype": str(policy.mlp_down_dtype),
                    "cache_dtype": str(policy.cache_dtype),
                    "sdpa_fidelity": str(policy.attention_fidelity),
                    "qkv_fidelity": str(policy.qkv_fidelity or policy.attention_fidelity),
                    "o_fidelity": str(policy.o_fidelity or policy.attention_fidelity),
                    "qkv_in0_block_w": policy.qkv_decode_in0_block_w or policy.decode_in0_block_w,
                    "o_in0_block_w": policy.o_decode_in0_block_w,
                    "gate_in0_block_w": policy.mlp_gate_decode_in0_block_w or policy.decode_in0_block_w,
                    "up_in0_block_w": policy.mlp_up_decode_in0_block_w or policy.decode_in0_block_w,
                    "down_in0_block_w": policy.mlp_down_in0_block_w,
                    "linear_packed_decode": policy.linear_packed_decode,
                    "linear_outer_product": policy.linear_outer_product,
                    "linear_recurrent_program": policy.linear_recurrent_program,
                    "linear_recurrent_fidelity": str(policy.linear_recurrent_fidelity),
                    "linear_recurrent_state_dtype": str(policy.linear_recurrent_state_dtype),
                    "linear_input_weight_dtype": str(policy.linear_input_weight_dtype),
                    "linear_input_fidelity": str(policy.linear_input_fidelity),
                    "linear_packed_in0_block_w": policy.linear_packed_in0_block_w,
                    "linear_output_weight_dtype": str(policy.linear_output_weight_dtype),
                    "linear_output_fidelity": str(policy.linear_output_fidelity),
                    "linear_out_in0_block_w": policy.linear_out_in0_block_w,
                }
            ),
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("full", "linear"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), required=True)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="default")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")
    result = run(args.kind, args.batch, args.optimized, args.candidate, args.steps)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
