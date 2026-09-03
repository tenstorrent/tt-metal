# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_tanh(input_shapes, device):
    # tt tan supports input range [-1.45, 1.45]
    in_data, input_tensor = data_gen_with_range(input_shapes, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)

    tt_output_tensor_on_device = ttnn.tanh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.95)
    assert status


@pytest.fixture
def isolate_program_cache(device):
    """Ensure the test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def test_bw_tanh_preallocated_output_buffer_type_not_cached(device, isolate_program_cache):
    """The preallocated input_grad's buffer type must take part in the program-cache key.

    The writer kernel bakes the output buffer's TensorAccessorArgs -- IsDram included -- into its
    compile-time args, and the host wrapper takes output_memory_config from the INPUT
    (unary_backward.cpp, output_mem_config.value_or(input.memory_config())), so operation_attributes
    say DRAM even when the real buffer is L1. validate compares only memory_layout, INTERLEAVED both
    ways, so the L1 input_grad is accepted. When the two calls collided, the second ran a DRAM
    accessor against an L1 address and the L1 tensor kept stale data.
    """
    torch.manual_seed(0)
    shape = torch.Size([1, 1, 32, 32])

    in_data, input_tensor = data_gen_with_range(shape, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(shape, -1e4, 1e4, device)

    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    # Call 1: no preallocated output -> created interleaved DRAM, writer compiled for DRAM.
    dram_result = ttnn.tanh_bw(grad_tensor, input_tensor)
    entries_after_dram = device.num_program_cache_entries()

    # Call 2: identical inputs, preallocated output in L1. Same operation_attributes, so this must
    # miss the cache on the buffer type alone.
    l1_input_grad = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.tanh_bw(grad_tensor, input_tensor, input_grad=l1_input_grad)

    assert compare_pcc(dram_result, golden_tensor, 0.95)
    assert compare_pcc([l1_input_grad], golden_tensor, 0.95)

    assert device.num_program_cache_entries() > entries_after_dram, (
        "L1 preallocated output reused the DRAM program: the output buffer type is missing from "
        "the program-cache key"
    )


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_tanh_with_output(input_shapes, device):
    # tt tan supports input range [-1.45, 1.45]
    in_data, input_tensor = data_gen_with_range(input_shapes, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)
    input_grad = None

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.tanh_bw(
        grad_tensor,
        input_tensor,
        input_grad=input_grad,
        queue_id=cq_id,
    )

    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.95)
    assert status
