# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused real-shape full-attention decode smoke used during bring-up."""

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


def run(batch, mode, sequence, full_layer, iterations, max_context, decode_position, permute_pages):
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    page_size = 64
    head_dim = config.head_dim
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def to_tt(tensor, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return _to_device(tensor, mesh_device=mesh, layout=layout, dtype=dtype)

    try:
        weights = {
            "q_proj": to_tt(
                torch.zeros(
                    config.hidden_size,
                    2 * config.num_attention_heads * head_dim,
                    dtype=torch.bfloat16,
                )
            ),
            "k_proj": to_tt(
                torch.zeros(
                    config.hidden_size,
                    config.num_key_value_heads * head_dim,
                    dtype=torch.bfloat16,
                )
            ),
            "v_proj": to_tt(
                torch.zeros(
                    config.hidden_size,
                    config.num_key_value_heads * head_dim,
                    dtype=torch.bfloat16,
                )
            ),
            "o_proj": to_tt(
                torch.zeros(
                    config.num_attention_heads * head_dim,
                    config.hidden_size,
                    dtype=torch.bfloat16,
                )
            ),
            "q_norm": to_tt(torch.ones(head_dim, dtype=torch.bfloat16)),
            "k_norm": to_tt(torch.ones(head_dim, dtype=torch.bfloat16)),
        }
        if full_layer:
            weights.update(
                {
                    "input_norm": to_tt(torch.ones(config.hidden_size, dtype=torch.bfloat16)),
                    "post_attention_norm": to_tt(torch.ones(config.hidden_size, dtype=torch.bfloat16)),
                    "mlp_gate": to_tt(
                        torch.zeros(
                            config.hidden_size,
                            config.intermediate_size,
                            dtype=torch.bfloat16,
                        )
                    ),
                    "mlp_up": to_tt(
                        torch.zeros(
                            config.hidden_size,
                            config.intermediate_size,
                            dtype=torch.bfloat16,
                        )
                    ),
                    "mlp_down": to_tt(
                        torch.zeros(
                            config.intermediate_size,
                            config.hidden_size,
                            dtype=torch.bfloat16,
                        )
                    ),
                }
            )
        pages_per_batch = (max_context + page_size - 1) // page_size
        caches = {
            name: to_tt(
                torch.zeros(
                    batch * pages_per_batch,
                    config.num_key_value_heads,
                    page_size,
                    head_dim,
                    dtype=torch.bfloat16,
                )
            )
            for name in ("key", "value")
        }
        caches["batch_indices"] = to_tt(
            torch.arange(batch, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        positions = torch.arange(max_context, dtype=torch.float32).unsqueeze(1)
        inv_freq = 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
        frequencies = positions * inv_freq.unsqueeze(0)
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        rope = {
            "cos": to_tt(embedding.cos().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT),
            "sin": to_tt(embedding.sin().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT),
        }
        grid = mesh.compute_with_storage_grid_size()
        grid_x = min(batch, grid.x)
        while batch % grid_x or batch // grid_x > grid.y:
            grid_x -= 1
        cores = ttnn.CoreGrid(y=batch // grid_x, x=grid_x)
        memory_config = ttnn.create_sharded_memory_config(
            shape=(32, head_dim),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        decoder = FunctionalDecoder(
            hf_config=config,
            layer_idx=3,
            mesh_device=mesh,
            batch=batch,
            max_context=max_context,
            page_size=page_size,
            weights=weights,
            caches=caches,
            rope=rope,
            decode_attention_memory_config=memory_config,
        )
        hidden_shape = (
            (1, 1, batch, config.hidden_size) if mode == "decode" else (1, batch, sequence, config.hidden_size)
        )
        torch.manual_seed(20260729)
        host_hidden = (
            torch.randn(hidden_shape, dtype=torch.float32).mul_(0.05).bfloat16()
            if full_layer
            else torch.zeros(hidden_shape, dtype=torch.bfloat16)
        )
        hidden_states = to_tt(host_hidden)
        page_values = torch.arange(batch * pages_per_batch, dtype=torch.int32).reshape(batch, pages_per_batch)
        if permute_pages:
            page_values = page_values.flip(-1)
        page_table = to_tt(
            page_values,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        position_values = (
            torch.full((batch,), decode_position, dtype=torch.uint32)
            if mode == "decode"
            else torch.arange(sequence, dtype=torch.int64).to(torch.uint32).repeat(batch, 1)
        )
        current_positions = to_tt(
            position_values,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )

        def decode():
            if full_layer:
                return decoder.decode_forward(
                    hidden_states=hidden_states,
                    page_table=page_table,
                    current_positions=current_positions,
                )
            return decoder._full_attention_decode(hidden_states, page_table, current_positions)

        if mode == "prefill":

            def prefill():
                if full_layer:
                    return decoder.prefill_forward(
                        hidden_states=hidden_states,
                        page_table=page_table,
                        current_positions=current_positions,
                    )
                return decoder._full_attention_prefill(hidden_states, page_table, current_positions)

            # Very-long runs are capacity probes.  A second live full output
            # would distort the DRAM question the probe is intended to answer.
            if sequence <= 65536:
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
            print("FULL_PREFILL_SMOKE_OK", tuple(output_torch.shape), f"warmed_ms={elapsed_ms:.6f}")
            return

        output = decode()
        ttnn.synchronize_device(mesh)
        output_torch = ttnn.to_torch(ttnn.get_device_tensors(output)[0])
        print("FULL_DECODE_OUTPUT", tuple(output.shape), tuple(output_torch.shape))
        assert tuple(output_torch.shape) == (1, 1, batch, config.hidden_size)
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
            assert tuple(replay.shape) == (1, 1, batch, config.hidden_size)
            assert torch.equal(replay, output_torch)
            replay_pcc = pcc(replay, expected) if full_layer else 1.0
            assert replay_pcc >= 0.995
        finally:
            ttnn.release_trace(mesh, trace_id)
        print(
            "FULL_DECODE_TRACED_SMOKE_OK",
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
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--full-layer", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-context", type=int, default=64)
    parser.add_argument("--decode-position", type=int, default=0)
    parser.add_argument("--permute-pages", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.decode_position < args.max_context:
        parser.error("--decode-position must be in [0, --max-context)")
    run(
        args.batch,
        args.mode,
        args.sequence,
        args.full_layer,
        args.iterations,
        args.max_context,
        args.decode_position,
        args.permute_pages,
    )
