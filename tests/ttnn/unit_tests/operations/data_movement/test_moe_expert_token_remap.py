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

# On a single device every expert is local. 32 experts give topk rows of 64 or 128 bytes, as a topk_tensor in DRAM
# needs.
BATCH, SEQ, EXPERTS, K = 4, 8, 32, 4

_TOPK_CAPTION = "Supported dtypes and layouts for `topk_tensor`"
_INDEX_CAPTION = "Supported dtypes and layouts for `expert_mapping_tensor` and `expert_metadata_tensor`"


def _selected_experts():
    """[1, BATCH, SEQ, K] expert indices: K different experts for each token."""
    torch.manual_seed(0)
    return torch.stack([torch.randperm(EXPERTS)[:K] for _ in range(BATCH * SEQ)]).reshape(1, BATCH, SEQ, K)


def _expected(topk, selected, reduction_size):
    """The weight of each selected expert and 0 for the others, and for each group of reduction_size tokens a 1 for
    each expert that a token of the group selects. In float64, which holds every topk value exactly."""
    chosen = torch.zeros((1, BATCH, SEQ, EXPERTS), dtype=torch.bool)
    chosen.scatter_(-1, selected.long(), True)
    weights = torch.where(chosen, topk.to(torch.float64), torch.zeros((), dtype=torch.float64))
    flags = chosen.reshape(BATCH * SEQ // reduction_size, reduction_size, EXPERTS).any(dim=1)
    return weights, flags.reshape(1, 1, -1, EXPERTS).to(torch.float64)


def _check_outputs(outputs, topk_tensor, selected, reduction_size, dtype, memory_config):
    weights, flags = outputs
    expected_weights, expected_flags = _expected(ttnn.to_torch(topk_tensor), selected, reduction_size)
    assert weights.dtype == dtype
    assert flags.dtype == ttnn.uint16
    for output in (weights, flags):
        assert output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert output.memory_config() == memory_config
    assert_exact(expected_weights, ttnn.to_torch(weights))
    assert_exact(expected_flags, ttnn.to_torch(flags))


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.moe_expert_token_remap))
@pytest.mark.parametrize(
    "index_dtype, index_layout", docstring_dtype_layout_params(ttnn.moe_expert_token_remap, caption=_INDEX_CAPTION)
)
@pytest.mark.parametrize(
    "dtype, layout", docstring_dtype_layout_params(ttnn.moe_expert_token_remap, caption=_TOPK_CAPTION)
)
def test_moe_expert_token_remap_docstring_supported_inputs(
    dtype, layout, index_dtype, index_layout, buffer_type, device
):
    """Runs every dtype, layout and interleaved buffer type the ttnn.moe_expert_token_remap docstring lists, on a
    single device, where every expert is local. Runs the default memory configuration and reduction_size, a given
    memory configuration and reduction_size, and preallocated outputs. The topk values cover the whole range of the
    dtype, and are compared exactly."""
    check_readable(dtype, layout, index_dtype, index_layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)

    def to_device(x, tensor_dtype, tensor_layout):
        return ttnn.from_torch(x, dtype=tensor_dtype, layout=tensor_layout, device=device, memory_config=memory_config)

    selected = _selected_experts()
    topk = to_device(full_range_input((1, BATCH, SEQ, EXPERTS), dtype), dtype, layout)
    mapping = to_device(torch.ones((1, 1, EXPERTS, 1), dtype=torch.int32), index_dtype, index_layout)
    metadata = to_device(selected.to(torch.int32), index_dtype, index_layout)

    # The default reduction_size is 16.
    outputs = ttnn.moe_expert_token_remap(topk, mapping, metadata)
    _check_outputs(outputs, topk, selected, 16, dtype, ttnn.DRAM_MEMORY_CONFIG)

    outputs = ttnn.moe_expert_token_remap(topk, mapping, metadata, memory_config=memory_config, reduction_size=4)
    _check_outputs(outputs, topk, selected, 4, dtype, memory_config)

    # Preallocated outputs start with other values, which the operation overwrites.
    preallocated = (
        to_device(full_range_input((1, BATCH, SEQ, EXPERTS), dtype, seed=1), dtype, layout),
        to_device(
            full_range_input((1, 1, BATCH * SEQ // 4, EXPERTS), ttnn.uint16, seed=1), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
        ),
    )
    ttnn.moe_expert_token_remap(
        topk,
        mapping,
        metadata,
        optional_output_mapping_tensor=preallocated[0],
        optional_output_reduced_tensor=preallocated[1],
        reduction_size=4,
    )
    _check_outputs(preallocated, topk, selected, 4, dtype, memory_config)
