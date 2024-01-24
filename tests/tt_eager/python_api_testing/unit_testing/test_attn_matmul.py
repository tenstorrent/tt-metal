# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import skip_for_wormhole_b0


def generate_input_shapes():
    batch_size = 32
    kv_heads = 1
    q_len = 1
    q_heads = 71
    seq_len = 128
    K = 64
    yield [q_len, q_heads, batch_size, K], [batch_size, kv_heads, K, seq_len]

    batch_size = 64
    kv_heads = 1
    q_len = 1
    q_heads = 10
    seq_len = 32
    K = 96
    yield [q_len, q_heads, batch_size, K], [batch_size, kv_heads, K, seq_len]


@pytest.mark.parametrize("in0_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("in1_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("out_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_attn_matmul(in0_dtype, in1_dtype, out_dtype, device):
    torch.manual_seed(0)

    for input_shape_a, input_shape_b in generate_input_shapes():
        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = ttl.tensor.Tensor(input_tensor_a, in0_dtype).to(ttl.tensor.Layout.TILE).to(device)
        tt_input_tensor_b = ttl.tensor.Tensor(input_tensor_b, in1_dtype).to(ttl.tensor.Layout.TILE).to(device)

        compute_grid_size = device.compute_with_storage_grid_size()

        tt_output_tensor_on_device = ttl.operations.primary.transformers.attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=ttl.tensor.CoreCoord(compute_grid_size.x, compute_grid_size.y),
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=out_dtype,
        )
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"


@pytest.mark.parametrize("in0_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("in1_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("out_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_attn_matmul_with_program_cache(in0_dtype, in1_dtype, out_dtype, device, use_program_cache):
    torch.manual_seed(0)

    for input_shape_a, input_shape_b in generate_input_shapes():
        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = ttl.tensor.Tensor(input_tensor_a, in0_dtype).to(ttl.tensor.Layout.TILE).to(device)
        tt_input_tensor_b = ttl.tensor.Tensor(input_tensor_b, in1_dtype).to(ttl.tensor.Layout.TILE).to(device)

        compute_grid_size = device.compute_with_storage_grid_size()

        tt_output_tensor_on_device = ttl.operations.primary.transformers.attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=ttl.tensor.CoreCoord(compute_grid_size.x, compute_grid_size.y),
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=out_dtype,
        )
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"
