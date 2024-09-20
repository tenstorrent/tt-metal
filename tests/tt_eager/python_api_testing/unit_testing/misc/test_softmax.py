# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import math

import ttnn

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero
from models.utility_functions import is_grayskull, is_blackhole, skip_for_blackhole


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_softmax(device, inplace, dtype):
    if (is_grayskull() or is_blackhole()) and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull and Blackhole. For Blackhole see #12349")

    torch.manual_seed(0)
    sm_op = ttnn.softmax_in_place if inplace else ttnn.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        if not is_grayskull():
            if dtype == ttnn.float32:
                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                )
            else:
                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                )

        tt_output_tensor_on_device = sm_op(
            tt_input_tensor, compute_kernel_config=compute_kernel_config if not is_grayskull() else None
        )
        tt_output_tensor = ttnn.to_layout(tt_output_tensor_on_device, ttnn.ROW_MAJOR_LAYOUT)
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        logger.info(output)
        assert allclose, f"FAILED: {output}"


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("inplace", [True, False])
def test_softmax_with_program_cache(device, use_program_cache, inplace):
    torch.manual_seed(0)
    sm_op = ttnn.softmax_in_place if inplace else ttnn.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output_tensor_on_device = sm_op(tt_input_tensor)
        tt_output_tensor = ttnn.to_layout(tt_output_tensor_on_device, ttnn.ROW_MAJOR_LAYOUT)
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize(
    "in_dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["bfloat16", "bfloat8_b"],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_softmax_mix_precision(device, inplace, in_dtype):
    torch.manual_seed(0)
    sm_op = ttnn.softmax_in_place if inplace else ttnn.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = ttnn.from_torch(input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output_tensor_on_device = sm_op(tt_input_tensor)
        tt_output_tensor = ttnn.to_layout(tt_output_tensor_on_device, ttnn.ROW_MAJOR_LAYOUT)
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize(
    "seq_len",
    [64, 384],
    ids=["64", "384"],
)
@pytest.mark.parametrize(
    "causal_mask",
    [True, False],
    ids=["causal", "no-causal"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=["float32", "bfloat16", "bfloat8_b"],
)
def test_scale_mask_softmax_inplace(device, in_dtype, in0_mem_config, causal_mask, seq_len):
    if is_grayskull() and in_dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(0)
    fuse_head = 2

    grid_size = (12, 8)
    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, 1, num_cores_r * fuse_head * seq_len, seq_len)
    M = input_shape[2]
    K = input_shape[3] * batch

    hidden_dim = 1024
    num_heads = 16
    scale = 1 / math.sqrt(hidden_dim // num_heads)

    mask_dtype = ttnn.float32 if in_dtype == ttnn.float32 else ttnn.bfloat16

    if causal_mask == False:
        attention_mask = torch.rand(batch, 1, 32, seq_len)
        mask = torch.rand_like(attention_mask) < 0.2
        attention_mask[mask] = float("-inf")
        attention_mask_t = ttnn.from_torch(attention_mask, dtype=mask_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        attention_mask = torch.rand(batch, 1, seq_len, seq_len)
        mask = torch.rand_like(attention_mask) < 0.2
        attention_mask[mask] = float("-inf")
        attention_mask = pad_weight(attention_mask)
        attention_mask_t = ttnn.from_torch(attention_mask, dtype=mask_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = ttnn.from_torch(
        input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )

    if not is_grayskull():
        if in_dtype == ttnn.float32:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            )
        else:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            )

    tt_output = ttnn.scale_mask_softmax_in_place(
        in1_t,
        scale,
        attention_mask_t,
        is_causal_mask=causal_mask,
        compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
    )

    tt_output_tensor = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    if causal_mask == False:
        attention_mask = attention_mask.reshape(batch, 1, 32, seq_len)[:, :, 0, :]
    else:
        attention_mask = attention_mask.repeat(1, 1, num_cores_r * fuse_head, 1)

    for i in range(batch):
        golden_output_tensor = input_tensor[i] * scale + attention_mask[i]
        golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

        allclose, output = comp_pcc(
            tt_output_tensor[i],
            golden_output_tensor,
        )
        logger.info(output)
        assert allclose, f"FAILED: {output}"


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["bfloat16", "bfloat8_b"],
)
def test_scale_mask_softmax(device, in_dtype, in0_mem_config):
    torch.manual_seed(0)
    fuse_head = 2

    grid_size = (12, 8)
    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, 1, num_cores_r * fuse_head * 384, 384)

    hidden_dim = 1024
    num_heads = 16
    scale = 1 / math.sqrt(hidden_dim // num_heads)
    attention_mask = torch.rand(batch, 1, 32, 384)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = ttnn.from_torch(
        input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )

    tt_output = ttnn.scale_mask_softmax(in1_t, scale, attention_mask_t)

    tt_output_tensor = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    attention_mask = attention_mask.reshape(batch, 1, 32, 384)

    attention_mask_ref = attention_mask[:, :, 0, :]

    for i in range(batch):
        golden_output_tensor = input_tensor[i] * scale + attention_mask_ref[i]
        golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

        allclose, output = comp_pcc(
            tt_output_tensor[i],
            golden_output_tensor,
        )
        logger.info(output)
        assert allclose, f"FAILED: {output}"
