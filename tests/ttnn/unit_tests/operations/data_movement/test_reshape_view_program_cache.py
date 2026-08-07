# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Program-cache behaviour for the tiled ttnn.reshape path (ReshapeViewDeviceOperation).

ReshapeViewDeviceOperation carried a custom compute_program_hash to exclude
operation_attributes_t::recreate_mapping_tensor -- a control flag for the op-owned
mapping tensor that the tiled factory never acted on. The attribute is gone, so the
framework default hash applies and it now also keys padded_output_shape (the old hash
keyed only logical_output_shape).

These tests pin the cache-entry counts across that change.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_reshape(device, shape, new_shape, tile_mode=None):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        if tile_mode is None:
            out = ttnn.reshape(tt_input, new_shape)
        else:
            out = ttnn.reshape(tt_input, new_shape, reshape_tile_mode=tile_mode)
    assert_with_pcc(torch_input.reshape(new_shape), ttnn.to_torch(out), 0.999)


def test_reshape_cache_reuse_same_config(device, isolate_program_cache):
    for _ in range(3):
        run_reshape(device, [1, 1, 64, 128], [1, 1, 128, 64])
    assert device.cache_entries_counter.total == 1


def test_reshape_cache_reuse_across_tile_mode(device, isolate_program_cache):
    """
    reshape_tile_mode is not part of the key and never was: the tiled factory does not
    act on it, so CACHE and RECREATE must land on the same program.
    """
    run_reshape(device, [1, 1, 64, 128], [1, 1, 128, 64], tile_mode=ttnn.TileReshapeMapMode.CACHE)
    run_reshape(device, [1, 1, 64, 128], [1, 1, 128, 64], tile_mode=ttnn.TileReshapeMapMode.RECREATE)
    assert device.cache_entries_counter.total == 1


def test_reshape_cache_miss_on_output_shape(device, isolate_program_cache):
    run_reshape(device, [1, 1, 64, 128], [1, 1, 128, 64])
    run_reshape(device, [1, 1, 64, 128], [1, 1, 32, 256])
    assert device.cache_entries_counter.total == 2


def test_reshape_cache_miss_on_input_shape(device, isolate_program_cache):
    run_reshape(device, [1, 1, 64, 128], [1, 1, 128, 64])
    run_reshape(device, [1, 1, 128, 128], [1, 1, 256, 64])
    assert device.cache_entries_counter.total == 2


def test_reshape_cache_hit_reapplies_buffers(device, isolate_program_cache):
    """
    Freeze test. Three dispatches of DIFFERENT data in DIFFERENT buffers must share one
    program and each must return its own data. A frozen buffer address returns the first
    dispatch's result and fails PCC here.
    """
    with device.cache_entries_counter.measure():
        for i in range(3):
            torch_input = torch.randn([1, 1, 64, 128], dtype=torch.bfloat16) * (i + 1)
            tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
            tt_output = ttnn.reshape(tt_input, [1, 1, 128, 64])
            assert_with_pcc(torch_input.reshape([1, 1, 128, 64]), ttnn.to_torch(tt_output), 0.999)
    assert device.cache_entries_counter.total == 1
