# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input

EXPERTS, NON_ZERO = 128, 32


def _routing_weights(dtype):
    """[1, EXPERTS] float64 weights: NON_ZERO of them non-zero, over the whole range of ``dtype``, and +0.0 for the
    others. Also returns the torch dtype that ``ttnn.from_torch`` takes for ``dtype``."""
    values = full_range_input((1, EXPERTS), dtype)
    wide = values.to(torch.float64)  # holds every value of these dtypes exactly
    candidates = wide[0].nonzero().flatten()  # -0.0 counts as zero here, so no kept weight is -0.0
    # The first candidates are +-inf and NaN, or the integer extremes; the others spread up to the last one.
    spread = torch.linspace(3, len(candidates) - 1, NON_ZERO - 3).round().long()
    kept = torch.cat([candidates[:3], candidates[spread]])
    weights = torch.zeros_like(wide)
    weights[0, kept] = wide[0, kept]
    return weights, values.dtype


def _check(output, expected, dtype, memory_config):
    assert output.dtype == dtype
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT
    assert output.memory_config() == memory_config
    assert_exact(expected, ttnn.to_torch(output))


@pytest.mark.parametrize("output_buffer_type", docstring_interleaved_buffer_type_params(ttnn.moe_routing_remap))
@pytest.mark.parametrize("input_buffer_type", docstring_interleaved_buffer_type_params(ttnn.moe_routing_remap))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.moe_routing_remap))
def test_moe_routing_remap_docstring_supported_inputs(dtype, layout, input_buffer_type, output_buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.moe_routing_remap docstring lists, for the input
    and the output, on a single device. There, expert_parallel_size is 1 along either cluster_axis, and the output
    equals the input. Runs the default memory configuration, a given one, and a preallocated output. The non-zero
    weights cover the whole range of the dtype, and are compared exactly."""
    check_readable(dtype, layout, input_buffer_type, output_buffer_type)
    input_memory_config = interleaved_memory_config(input_buffer_type)
    output_memory_config = interleaved_memory_config(output_buffer_type)

    expected, torch_dtype = _routing_weights(dtype)
    routing_weights = ttnn.from_torch(
        expected.to(torch_dtype), dtype=dtype, layout=layout, device=device, memory_config=input_memory_config
    )

    # By default the output takes the memory configuration of the input.
    output = ttnn.moe_routing_remap(
        routing_weights, non_zero_weight_size=NON_ZERO, expert_parallel_size=1, cluster_axis=0
    )
    _check(output, expected, dtype, input_memory_config)

    # A given memory_config sets the output's. The single device is also the only one along cluster_axis 1.
    output = ttnn.moe_routing_remap(
        routing_weights,
        non_zero_weight_size=NON_ZERO,
        expert_parallel_size=1,
        cluster_axis=1,
        memory_config=output_memory_config,
    )
    _check(output, expected, dtype, output_memory_config)

    # A preallocated output starts with values that differ from the expected output everywhere.
    preallocated = ttnn.from_torch(
        torch.where(expected == 1, 2, 1).to(torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=output_memory_config,
    )
    output = ttnn.moe_routing_remap(
        routing_weights,
        non_zero_weight_size=NON_ZERO,
        expert_parallel_size=1,
        cluster_axis=0,
        memory_config=output_memory_config,
        optional_output_tensor=preallocated,
    )
    _check(preallocated, expected, dtype, output_memory_config)
    _check(output, expected, dtype, output_memory_config)
