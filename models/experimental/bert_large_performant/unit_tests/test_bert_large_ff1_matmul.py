# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger


import numpy as np

import ttnn
from models.utility_functions import (
    comp_pcc,
)
from models.demos.metal_BERT_large_11.tt import custom_matmuls
import torch
import pytest


def run_bert_large_ff1_matmul_test(
    device,
    dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
    fused_activation,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 12:
        pytest.skip(f"Grid size {compute_grid_size} is not supported")

    if (
        dtype == ttnn.bfloat16
        and out_mem_config.buffer_type == ttnn.BufferType.L1
        and (in0_mem_config.buffer_type == ttnn.BufferType.L1 or in1_mem_config.buffer_type == ttnn.BufferType.L1)
    ):
        pytest.skip("Skipping test since these tensors won't fit on device!")

    torch.manual_seed(1234)

    a_shape = [9, 1, 384, 1024]
    b_shape = [1, 1, 1024, 4096]
    bias_shape = [1, 1, 1, 4096]
    bias_pad_shape = [1, 1, 32, 4096]
    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95
    BIAS = torch.randint(-20, 20, bias_shape, dtype=torch.float)

    a_t = (
        ttnn.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device, in0_mem_config)
    )
    b_t = (
        ttnn.Tensor(
            B.flatten().tolist(),
            b_shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device, in1_mem_config)
    )

    if bias_mem_config is not None:
        bias_t = (
            ttnn.Tensor(
                BIAS.flatten().tolist(),
                bias_shape,
                dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .pad(bias_pad_shape, [0, 0, 0, 0], 0)
            .to(ttnn.TILE_LAYOUT)
            .to(device, bias_mem_config)
        )
    else:
        bias_t = None

    t2 = custom_matmuls.bert_large_ff1_matmul(
        a_t,
        b_t,
        bias=bias_t,
        fused_activation=fused_activation,
        output_mem_config=out_mem_config,
    )
    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    if bias_mem_config is not None:
        assert bias_t.memory_config().buffer_type == bias_mem_config.buffer_type
    assert t2.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_type}")
    logger.debug(f"in1 is on: {b_t.memory_config().buffer_type}")
    if bias_mem_config is not None:
        logger.debug(f"bias is on: {bias_t.memory_config().buffer_type}")
    logger.debug(f"out is on: {t2.memory_config().buffer_type}")

    assert t2.shape.with_tile_padding() == [9, 1, 384, 4096]
    tt_host_rm = t2.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    pyt_got_back_rm = tt_host_rm.to_torch()

    ref_bmm = torch.matmul(A, B)
    if bias_mem_config is not None:
        ref_bmm = ref_bmm + BIAS
    if fused_activation is not None:
        if fused_activation[0] == ttnn.UnaryOpType.GELU:
            ref_bmm = torch.nn.functional.gelu(ref_bmm)
        else:
            assert False, "Unknown activation"
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "activation",
    ((ttnn.UnaryOpType.GELU, True), None),
    ids=["gelu_activation", "no_activation"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "bias_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        None,
    ),
    ids=["bias_DRAM", "bias_L1", "bias_None"],
)
@pytest.mark.parametrize(
    "in1_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in1_DRAM", "in1_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
def test_bert_large_ff1_matmul_test(
    device,
    dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
    activation,
    request,
):
    run_bert_large_ff1_matmul_test(
        device,
        dtype,
        in0_mem_config,
        in1_mem_config,
        bias_mem_config,
        out_mem_config,
        activation,
    )


def test_bert_large_ff1_matmul_with_program_cache(device, use_program_cache):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_ff1_matmul_test(
            device,
            dtype,
            mem_config,
            mem_config,
            mem_config,
            mem_config,
            fused_activation=None,
        )
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_ff1_matmul_test(
            device,
            dtype,
            mem_config,
            mem_config,
            mem_config,
            mem_config,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        )
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
