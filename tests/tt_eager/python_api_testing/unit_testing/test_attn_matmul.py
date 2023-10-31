# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax, comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0

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

@skip_for_wormhole_b0
def test_attn_matmul(device):

    torch.manual_seed(0)


    for input_shape_a, input_shape_b in generate_input_shapes():

        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = (
            ttl.tensor.Tensor(input_tensor_a, ttl.tensor.DataType.BFLOAT16)
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        tt_input_tensor_b = (
            ttl.tensor.Tensor(input_tensor_b, ttl.tensor.DataType.BFLOAT16)
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

        tt_output_tensor_on_device = ttl.operations.primary.transformers.attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=ttl.tensor.CoreCoord(12, 9),
            output_mem_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
            output_dtype=ttl.tensor.DataType.BFLOAT16,
        )
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"

@skip_for_wormhole_b0
def test_attn_matmul_with_program_cache(device, use_program_cache):

    torch.manual_seed(0)

    for input_shape_a, input_shape_b in generate_input_shapes():

        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = (
            ttl.tensor.Tensor(input_tensor_a, ttl.tensor.DataType.BFLOAT16)
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        tt_input_tensor_b = (
            ttl.tensor.Tensor(input_tensor_b, ttl.tensor.DataType.BFLOAT16)
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

        tt_output_tensor_on_device = ttl.operations.primary.transformers.attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=ttl.tensor.CoreCoord(12, 9),
            output_mem_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
            output_dtype=ttl.tensor.DataType.BFLOAT16,
        )
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"
