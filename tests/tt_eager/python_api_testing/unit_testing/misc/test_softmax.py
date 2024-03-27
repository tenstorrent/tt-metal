# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import math

import tt_lib as ttl
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero
from models.utility_functions import is_grayskull


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_softmax(device, inplace, dtype):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(0)
    sm_op = ttl.operations.primary.softmax_in_place if inplace else ttl.tensor.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = ttl.tensor.Tensor(input_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device)

        if not is_grayskull():
            if dtype == ttl.tensor.DataType.FLOAT32:
                compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
                    math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                )
            else:
                compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
                    math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                )

        tt_output_tensor_on_device = sm_op(
            tt_input_tensor, compute_kernel_config=compute_kernel_config if not is_grayskull() else None
        )
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        logger.info(output)
        assert allclose, f"FAILED: {output}"


@pytest.mark.parametrize("inplace", [True, False])
def test_softmax_with_program_cache(device, use_program_cache, inplace):
    torch.manual_seed(0)
    sm_op = ttl.operations.primary.softmax_in_place if inplace else ttl.tensor.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = (
            ttl.tensor.Tensor(input_tensor, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
        )
        tt_output_tensor_on_device = sm_op(tt_input_tensor)
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"


@pytest.mark.parametrize(
    "cb_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_softmax_mix_precision(device, inplace, in_dtype, cb_dtype):
    torch.manual_seed(0)
    sm_op = ttl.operations.primary.softmax_in_place if inplace else ttl.tensor.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = ttl.tensor.Tensor(input_tensor, in_dtype).to(ttl.tensor.Layout.TILE).to(device)
        tt_output_tensor_on_device = sm_op(tt_input_tensor)
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"


@pytest.mark.parametrize(
    "seq_len",
    [64, 384],
    ids=["64", "384"],
)
@pytest.mark.parametrize(
    "casual_mask",
    [True, False],
    ids=["causal", "no-causal"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["FLOAT32", "BFLOAT16", "BFLOAT8_B"],
)
def test_scale_mask_softmax_inplace(device, in_dtype, in0_mem_config, casual_mask, seq_len):
    if is_grayskull() and in_dtype == ttl.tensor.DataType.FLOAT32:
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
    # scale = 1.0
    scale = 1 / math.sqrt(hidden_dim // num_heads)

    if casual_mask == False:
        attention_mask = torch.rand(batch, 1, 32, seq_len)
        mask = torch.rand_like(attention_mask) < 0.2
        attention_mask[mask] = float("-inf")
        attention_mask32 = tilize_to_list(pad_weight(attention_mask))
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask32,
            [batch, 1, 32, seq_len],
            # ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.FLOAT32 if in_dtype == ttl.tensor.DataType.FLOAT32 else ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            device,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        )
    else:
        # attention_mask = torch.zeros(batch, 1, seq_len, seq_len)
        attention_mask = torch.rand(batch, 1, seq_len, seq_len)
        mask = torch.rand_like(attention_mask) < 0.2
        attention_mask[mask] = float("-inf")
        attention_mask32 = tilize_to_list(pad_weight(attention_mask))
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask32,
            [batch, 1, seq_len, seq_len],
            # ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.FLOAT32 if in_dtype == ttl.tensor.DataType.FLOAT32 else ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            device,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = torch2tt_tensor(input_tensor, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)

    if not is_grayskull():
        if in_dtype == ttl.tensor.DataType.FLOAT32:
            compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            )
        else:
            compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            )

    tt_output = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
        in1_t,
        scale,
        attention_mask_t,
        is_causal_mask=casual_mask,
        compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
    )

    tt_output_tensor = tt_output.cpu().to_torch().float()
    tt_output_tensor = torch.Tensor(tt_output_tensor).reshape(input_shape)
    tt_output_tensor = untilize(tt_output_tensor)

    if casual_mask == False:
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


@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
def test_scale_mask_softmax(device, in_dtype, in0_mem_config):
    torch.manual_seed(0)
    fuse_head = 2

    grid_size = (12, 8)
    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, 1, num_cores_r * fuse_head * 384, 384)
    M = input_shape[2]
    K = input_shape[3] * batch

    hidden_dim = 1024
    num_heads = 16
    scale = 1 / math.sqrt(hidden_dim // num_heads)
    attention_mask = torch.rand(batch, 1, 32, 384)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask32 = tilize_to_list(pad_weight(attention_mask))
    attention_mask_t = ttl.tensor.Tensor(
        attention_mask32,
        [batch, 1, 32, 384],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = torch2tt_tensor(input_tensor, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)

    tt_output = ttl.tensor.scale_mask_softmax(in1_t, scale, attention_mask_t)

    tt_output_tensor = tt_output.cpu().to_torch().float()
    tt_output_tensor = torch.Tensor(tt_output_tensor).reshape(input_shape)
    tt_output_tensor = untilize(tt_output_tensor)

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
