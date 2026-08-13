# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused real-shape gated-delta decode smoke used during bring-up."""

import argparse
import time

import torch
from tracy import signpost
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, FunctionalDecoder, _to_device


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run(batch, mode, sequence, full_layer, iterations, decode_position):
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    hidden = config.hidden_size
    key_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    key_dim = config.linear_key_head_dim
    value_dim = config.linear_value_head_dim
    key_width = key_heads * key_dim
    value_width = value_heads * value_dim
    conv_width = 2 * key_width + value_width
    kernel = config.linear_conv_kernel_dim
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def to_tt(tensor, *, dtype=ttnn.bfloat16):
        return _to_device(tensor, mesh_device=mesh, dtype=dtype)

    try:
        weights = {
            "in_qkv": to_tt(torch.zeros(hidden, conv_width, dtype=torch.bfloat16)),
            "in_z": to_tt(torch.zeros(hidden, value_width, dtype=torch.bfloat16)),
            "in_b": to_tt(torch.zeros(hidden, value_heads, dtype=torch.bfloat16)),
            "in_a": to_tt(torch.zeros(hidden, value_heads, dtype=torch.bfloat16)),
            "conv": to_tt(torch.zeros(1, 1, conv_width, kernel, dtype=torch.bfloat16)),
            "dt_bias": to_tt(
                torch.ones(1, 1, 1, value_heads, dtype=torch.float32),
                dtype=ttnn.float32,
            ),
            "a": to_tt(
                -torch.ones(1, 1, 1, value_heads, dtype=torch.float32),
                dtype=ttnn.float32,
            ),
            "gated_norm": to_tt(torch.ones(value_dim, dtype=torch.bfloat16)),
            "out_proj": to_tt(torch.zeros(value_width, hidden, dtype=torch.bfloat16)),
            "linear_identity": to_tt(torch.eye(value_dim, dtype=torch.bfloat16).reshape(1, 1, value_dim, value_dim)),
        }
        if full_layer:
            weights.update(
                {
                    "input_norm": to_tt(torch.ones(hidden, dtype=torch.bfloat16)),
                    "post_attention_norm": to_tt(torch.ones(hidden, dtype=torch.bfloat16)),
                    "mlp_gate": to_tt(torch.zeros(hidden, config.intermediate_size, dtype=torch.bfloat16)),
                    "mlp_up": to_tt(torch.zeros(hidden, config.intermediate_size, dtype=torch.bfloat16)),
                    "mlp_down": to_tt(torch.zeros(config.intermediate_size, hidden, dtype=torch.bfloat16)),
                }
            )
        caches = {
            "conv": to_tt(torch.zeros(1, batch, conv_width, kernel, dtype=torch.bfloat16)),
            "recurrent": to_tt(
                torch.zeros(batch, value_heads, value_dim, value_dim, dtype=torch.float32),
                dtype=ttnn.float32,
            ),
        }
        decoder = FunctionalDecoder(
            hf_config=config,
            layer_idx=0,
            mesh_device=mesh,
            batch=batch,
            max_context=64,
            page_size=64,
            weights=weights,
            caches=caches,
            rope={},
        )
        hidden_shape = (1, 1, batch, hidden) if mode == "decode" else (1, batch, sequence, hidden)
        torch.manual_seed(20260729)
        host_hidden = (
            torch.randn(hidden_shape, dtype=torch.float32).mul_(0.05).bfloat16()
            if full_layer
            else torch.zeros(hidden_shape, dtype=torch.bfloat16)
        )
        hidden_states = to_tt(host_hidden)
        page_table = to_tt(
            torch.zeros(batch, 1, dtype=torch.int32),
            dtype=ttnn.int32,
        )
        current_positions = to_tt(
            torch.full((batch,), decode_position, dtype=torch.uint32),
            dtype=ttnn.uint32,
        )

        if mode == "prefill":

            def prefill():
                if full_layer:
                    return decoder.prefill_forward(
                        hidden_states=hidden_states,
                        page_table=page_table,
                        current_positions=current_positions,
                    )
                return decoder._linear_attention_prefill(hidden_states)

            prefill()
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            started = time.perf_counter()
            output = prefill()
            ttnn.synchronize_device(mesh)
            elapsed_ms = (time.perf_counter() - started) * 1000
            signpost("PERF_PREFILL_END")
            output_torch = ttnn.to_torch(ttnn.get_device_tensors(output)[0])
            assert tuple(output_torch.shape) == hidden_shape
            assert torch.equal(output_torch, host_hidden)
            print("LINEAR_PREFILL_SMOKE_OK", tuple(output_torch.shape), f"warmed_ms={elapsed_ms:.6f}")
            return

        def decode():
            if full_layer:
                return decoder.decode_forward(
                    hidden_states=hidden_states,
                    page_table=page_table,
                    current_positions=current_positions,
                )
            return decoder._linear_attention_decode(hidden_states)

        output = decode()
        ttnn.synchronize_device(mesh)
        output_torch = ttnn.to_torch(ttnn.get_device_tensors(output)[0])
        assert tuple(output_torch.shape) == (1, 1, batch, hidden)
        expected = host_hidden if full_layer else torch.zeros_like(host_hidden)
        eager_pcc = pcc(output_torch, expected) if full_layer else 1.0
        assert torch.equal(output_torch, expected)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = decode()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        try:
            replay_times = []
            signpost("PERF_DECODE")
            for _ in range(iterations):
                started = time.perf_counter()
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                replay_times.append((time.perf_counter() - started) * 1000)
            signpost("PERF_DECODE_END")
            replay = ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0])
            assert torch.equal(replay, output_torch)
            replay_pcc = pcc(replay, expected) if full_layer else 1.0
            assert replay_pcc >= 0.995
        finally:
            ttnn.release_trace(mesh, trace_id)
        print(
            "LINEAR_DECODE_TRACED_SMOKE_OK",
            tuple(output_torch.shape),
            f"eager_pcc={eager_pcc:.9f}",
            f"replay_pcc={replay_pcc:.9f}",
            f"batch={batch}",
            f"iterations={iterations}",
            f"median_ms={torch.tensor(replay_times).median().item():.6f}",
            f"min_ms={min(replay_times):.6f}",
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--sequence", type=int, default=4)
    parser.add_argument("--full-layer", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--decode-position", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.decode_position < 262144:
        parser.error("--decode-position must be in [0, 262144)")
    run(
        args.batch,
        args.mode,
        args.sequence,
        args.full_layer,
        args.iterations,
        args.decode_position,
    )
