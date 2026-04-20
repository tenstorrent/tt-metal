# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.gated_attention_gated_deltanet.tt.fused_chunked_delta_rule_placeholder import (
    fused_chunked_delta_rule_ttnn,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def assert_with_pcc(torch_output, ttnn_output, pcc_threshold=0.99):
    if isinstance(ttnn_output, torch.Tensor):
        ttnn_tensor = ttnn_output.to(torch.float32)
    else:
        ttnn_tensor = ttnn.to_torch(ttnn_output).to(torch.float32)

    torch_flat = torch_output.to(torch.float32).flatten()
    ttnn_flat = ttnn_tensor.flatten()

    if torch_flat.shape != ttnn_flat.shape:
        raise ValueError(f"Shape mismatch: torch {torch_flat.shape} vs pipeline {ttnn_flat.shape}")

    if torch_flat.std() < 1e-10 and ttnn_flat.std() < 1e-10:
        return 1.0

    mean_torch = torch_flat.mean()
    mean_ttnn = ttnn_flat.mean()
    diff_torch = torch_flat - mean_torch
    diff_ttnn = ttnn_flat - mean_ttnn
    pcc = (diff_torch * diff_ttnn).sum() / (
        torch.sqrt((diff_torch**2).sum()) * torch.sqrt((diff_ttnn**2).sum()) + 1e-12
    )
    pcc_value = pcc.item()

    if pcc_value < pcc_threshold:
        max_diff = (torch_flat - ttnn_flat).abs().max().item()
        mean_diff = (torch_flat - ttnn_flat).abs().mean().item()
        raise AssertionError(
            f"PCC {pcc_value:.6f} < {pcc_threshold}. " f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
        )

    return pcc_value


def _pad_last_dim(tensor, padded_width):
    current_width = tensor.shape[-1]
    if current_width == padded_width:
        return tensor

    padded_shape = (*tensor.shape[:-1], padded_width - current_width)
    padding = torch.zeros(padded_shape, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=-1)


def pack_fused_chunked_delta_rule_inputs(q, k, v, beta, g, padded_width):
    q_padded = _pad_last_dim(q, padded_width)
    k_padded = _pad_last_dim(k, padded_width)
    v_padded = _pad_last_dim(v, padded_width)
    beta_padded = _pad_last_dim(beta.unsqueeze(-1), padded_width)
    g_padded = _pad_last_dim(g.unsqueeze(-1), padded_width)

    # Stack inputs along sequence so the packed tensor keeps a tile-friendly width.
    return torch.cat([q_padded, k_padded, v_padded, beta_padded, g_padded], dim=1)


def create_fused_chunked_delta_rule_pipeline_model(head_k_dim, head_v_dim, chunk_size, device):
    padded_width = head_v_dim

    def run(device_packed_input_tensor):
        assert device_packed_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects device tensor input"

        packed_input = device_packed_input_tensor
        if packed_input.layout != ttnn.TILE_LAYOUT:
            packed_input = ttnn.to_layout(packed_input, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        batch_size, packed_seq_len, num_heads, packed_width = packed_input.shape
        assert packed_width == padded_width, f"Expected packed width {padded_width}, got {packed_width}"
        assert packed_seq_len % 5 == 0, f"Expected packed sequence to be divisible by 5, got {packed_seq_len}"
        seq_len = packed_seq_len // 5

        q = ttnn.slice(
            packed_input,
            [0, 0, 0, 0],
            [batch_size, seq_len, num_heads, packed_width],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.slice(
            packed_input,
            [0, seq_len, 0, 0],
            [batch_size, 2 * seq_len, num_heads, packed_width],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.slice(
            packed_input,
            [0, 2 * seq_len, 0, 0],
            [batch_size, 3 * seq_len, num_heads, packed_width],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta = ttnn.slice(
            packed_input,
            [0, 3 * seq_len, 0, 0],
            [batch_size, 4 * seq_len, num_heads, packed_width],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        g = ttnn.slice(
            packed_input,
            [0, 4 * seq_len, 0, 0],
            [batch_size, 5 * seq_len, num_heads, packed_width],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q = ttnn.slice(
            q, [0, 0, 0, 0], [batch_size, seq_len, num_heads, head_k_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k = ttnn.slice(
            k, [0, 0, 0, 0], [batch_size, seq_len, num_heads, head_k_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        beta = ttnn.slice(
            beta, [0, 0, 0, 0], [batch_size, seq_len, num_heads, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g = ttnn.slice(g, [0, 0, 0, 0], [batch_size, seq_len, num_heads, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        beta = ttnn.reshape(beta, [batch_size, seq_len, num_heads], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.reshape(g, [batch_size, seq_len, num_heads], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output, state = fused_chunked_delta_rule_ttnn(q, k, v, beta, g, chunk_size=chunk_size, device=device)

        if output.layout != ttnn.ROW_MAJOR_LAYOUT:
            output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        if state.layout != ttnn.ROW_MAJOR_LAYOUT:
            state = ttnn.to_layout(state, ttnn.ROW_MAJOR_LAYOUT)

        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        state = ttnn.to_memory_config(state, ttnn.DRAM_MEMORY_CONFIG)
        return output, state

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 10000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [8])
@pytest.mark.parametrize(
    "seq_len, chunk_size, batch_size, num_heads, head_k_dim, head_v_dim",
    [(1, 64, 2, 4, 128, 256)],
)
def test_fused_chunked_delta_rule_e2e_pipeline(
    device,
    num_iterations,
    seq_len,
    chunk_size,
    batch_size,
    num_heads,
    head_k_dim,
    head_v_dim,
):
    torch.manual_seed(0)

    q = torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim, dtype=torch.float32)
    beta = torch.rand(batch_size, seq_len, num_heads, dtype=torch.float32)
    g = -torch.rand(batch_size, seq_len, num_heads, dtype=torch.float32) * 2

    logger.info("Packing fused delta-rule inputs for pipeline execution")
    packed_input = pack_fused_chunked_delta_rule_inputs(q, k, v, beta, g, padded_width=head_v_dim)
    ttnn_input_tensor = ttnn.from_torch(
        packed_input,
        device=None,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    pipeline_model = create_fused_chunked_delta_rule_pipeline_model(
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        chunk_size=chunk_size,
        device=device,
    )
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=False, num_command_queues=1, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        l1_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    logger.info("Compiling single-CQ pipeline")
    compile_start = time.time()
    pipeline.compile(ttnn_input_tensor)
    compile_time = time.time() - compile_start
    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running {num_iterations} pipeline iterations")
    run_start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    average_inference_time = (time.time() - run_start) / num_iterations
    pipeline.cleanup()

    pipeline_output, pipeline_state = outputs[-1]
    logger.info(f"Compile time: {compile_time:.4f}s")
    logger.info(f"Average pipeline iteration time: {average_inference_time * 1000.0:.2f} ms")

    q_ttnn = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_ttnn = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_ttnn = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    beta_ttnn = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    reference_output, reference_state = fused_chunked_delta_rule_ttnn(
        q_ttnn,
        k_ttnn,
        v_ttnn,
        beta_ttnn,
        g_ttnn,
        chunk_size=chunk_size,
        device=device,
    )

    output_pcc = assert_with_pcc(ttnn.to_torch(reference_output), pipeline_output, pcc_threshold=0.99)
    state_pcc = assert_with_pcc(ttnn.to_torch(reference_state), pipeline_state, pcc_threshold=0.99)

    logger.info(f"Pipeline output PCC={output_pcc:.6f}")
    logger.info(f"Pipeline state PCC={state_pcc:.6f}")
