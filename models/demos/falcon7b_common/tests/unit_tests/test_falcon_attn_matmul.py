# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc, tt2torch_tensor
import torch


def run_falcon_attn_matmul_test(
    falcon_op,
    transpose_hw,
    batch,
    seq_len,
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    device,
):
    torch.manual_seed(1234)

    pcc = 0.99

    if falcon_op == ttnn.experimental.attn_matmul:
        q_len = 1
        kv_heads = 1
        q_heads = 71
        a_shape = [q_len, q_heads, batch, K]
        b_shape = [batch, kv_heads, K, seq_len]
        expected_output_shape = [1, q_heads, batch, seq_len]

        B = torch.randn(b_shape) - 0.95
        b_t = ttnn.Tensor(B, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, in1_mem_config)

    elif falcon_op == ttnn.experimental.attn_matmul_from_cache:
        q_len = 1
        kv_heads = 1
        q_heads = 71
        max_seq_len = 2048

        if transpose_hw:
            # Pre-attention matmul
            a_shape = [q_len, q_heads, batch, K]
            b_shape = [batch, kv_heads, max_seq_len, K]
            kv_cache = torch.randn(b_shape) - 0.95
            B = kv_cache[:, :, :seq_len, :].transpose(-1, -2)
            expected_output_shape = [1, q_heads, batch, seq_len]
        else:
            # Post-attention matmul
            a_shape = [q_len, q_heads, batch, seq_len]
            b_shape = [batch, kv_heads, max_seq_len, K]
            kv_cache = torch.randn(b_shape) - 0.95
            B = kv_cache[:, :, :seq_len, :]
            expected_output_shape = [1, q_heads, batch, K]

        b_t = ttnn.Tensor(kv_cache, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, in1_mem_config)

    else:
        raise NotImplementedError(f"falcon matmul op is undefined!")

    A = torch.randn(a_shape)

    a_t = ttnn.Tensor(A, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    compute_grid_size = device.compute_with_storage_grid_size()

    if falcon_op == ttnn.experimental.attn_matmul:
        out = falcon_op(
            a_t,
            b_t,
            compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
            memory_config=out_mem_config,
            dtype=out_dtype,
        )

    elif falcon_op == ttnn.experimental.attn_matmul_from_cache:
        out = ttnn.experimental.attn_matmul_from_cache(
            a_t,
            b_t,
            num_tokens=seq_len,
            transpose_hw=transpose_hw,
            compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
            memory_config=out_mem_config,
            dtype=out_dtype,
        )

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.get_dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.get_dtype() == in1_dtype
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    assert out.get_dtype() == out_dtype
    logger.debug(f"in0 ({a_shape}): {a_t.memory_config().buffer_type} and {a_t.get_dtype()}")
    logger.debug(f"in1 ({b_shape}): {b_t.memory_config().buffer_type} and {b_t.get_dtype()}")
    logger.debug(f"out ({expected_output_shape}): {out.memory_config().buffer_type} and {out.get_dtype()}")

    assert out.shape.with_tile_padding() == expected_output_shape
    pyt_got_back_rm = tt2torch_tensor(out)

    ref_bmm = torch.matmul(A.transpose(0, 2), B).transpose(0, 2)

    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, pcc)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


# TODO: We could parametrize these separately for comprehensive testing
@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, out_mem_config",
    (
        (
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ),
    ids=["DRAM"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttnn.bfloat16,),
    ids=["out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttnn.bfloat16,),
    ids=["in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttnn.bfloat16,),
    ids=["in0_BFLOAT16"],
)
@pytest.mark.parametrize(
    "falcon_op, transpose_hw",
    (
        (ttnn.experimental.attn_matmul, None),
        (ttnn.experimental.attn_matmul_from_cache, True),
        (ttnn.experimental.attn_matmul_from_cache, False),
    ),
    ids=["attn_matmul", "pre_attn_matmul_from_cache", "post_attn_matmul_from_cache"],
)
@pytest.mark.parametrize(
    "batch, seq_len, K",
    ((32, 128, 64), (64, 2048, 64), (32, 64, 128), (64, 64, 2048)),
)
def test_falcon_matmul(
    falcon_op,
    transpose_hw,
    batch,
    seq_len,
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    request,
    device,
):
    run_falcon_attn_matmul_test(
        falcon_op,
        transpose_hw,
        batch,
        seq_len,
        K,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        device,
    )
