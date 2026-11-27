# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros debug: drive the dtypes Phase 0 refuses through the same program, past validate().

The kernel writes bytes; only the page size differs per dtype. These cases check that claim on the
two production shapes outside Phase 0 SUPPORTED (p4 bfloat16, p1 bfloat8_b) over a poisoned range,
and give the profiler a handle on p1, the performance focus.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros.g6_zeros_program_descriptor import get_program

P1 = (32831, 1, 32, 256)
P4 = (1, 4, 256, 128)


def run_kernel(shape, dtype, device):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.generic_op([output], get_program(output, device))


def poison_range(shape, dtype, device):
    poison = torch.full(tuple(shape), 1.0, dtype=torch.bfloat16)
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(poison, dtype=dtype, layout=ttnn.TILE_LAYOUT), dirty)
    assert int(torch.count_nonzero(ttnn.to_torch(dirty))) == dirty.logical_volume(), "the poison did not land"
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    return address


def assert_all_positive_zero(output):
    bits = ttnn.to_torch(output).to(torch.float32).contiguous().view(torch.int32)
    nonzero = int(torch.count_nonzero(bits))
    assert nonzero == 0, f"{nonzero} of {bits.numel()} elements are not +0.0"


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(P4, ttnn.bfloat16, id="p4-bfloat16"),
        pytest.param(P1, ttnn.bfloat8_b, id="p1-bfloat8_b"),
    ],
)
def test_dirty_destination_outside_phase0(shape, dtype, device):
    ttnn.deallocate(run_kernel(shape, dtype, device))
    address = poison_range(shape, dtype, device)
    output = run_kernel(shape, dtype, device)
    assert output.buffer_address() == address
    assert_all_positive_zero(output)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(P1, ttnn.bfloat8_b, id="p1-bfloat8_b"),
        pytest.param((16, 12, 128, 128), ttnn.float32, id="p2-float32"),
        pytest.param((8, 12, 128, 128), ttnn.float32, id="p3-float32"),
        pytest.param(P4, ttnn.bfloat16, id="p4-bfloat16"),
    ],
)
def test_perf_shape(shape, dtype, device):
    """No readback: a profiling handle, one dispatch per shape."""
    ttnn.deallocate(run_kernel(shape, dtype, device))
