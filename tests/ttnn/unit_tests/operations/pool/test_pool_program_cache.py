# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Program-cache behaviour for ttnn.max_pool2d (Pool2D device op).

Pool2D carried a custom compute_program_hash whose only job was to EXCLUDE
operation_attributes_t::memory_used -- a live L1 allocator reading captured by the
host op and threaded into the device op's attributes so a TT_FATAL in the factory
could cross-check CB sizes. Being live allocator state it varies with unrelated
allocations, so keying on it would miss the cache on nearly every call.

memory_used is now gone from the attributes and the default hash applies. Counts are
asserted as growth / no-growth because run_max_pool2d also dispatches halo; the
absolute number of entries is not the point, whether a repeat adds any is.
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@pytest.fixture
def tensor_map():
    return {}


pytestmark = pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)


def run_pool(device, tensor_map, shape, kernel=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1)):
    run_max_pool2d(
        list(shape),
        list(kernel),
        list(padding),
        list(stride),
        list(dilation),
        device,
        tensor_map,
        ttnn.bfloat16,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=False,
        nightly_skips=False,
    )


def test_pool_cache_reuse_same_config(device, tensor_map, isolate_program_cache):
    """A repeat of an identical config must add no cache entries."""
    with device.cache_entries_counter.measure():
        run_pool(device, tensor_map, [1, 16, 25, 23])
    after_first = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        run_pool(device, tensor_map, [1, 16, 25, 23])
        run_pool(device, tensor_map, [1, 16, 25, 23])
    assert device.cache_entries_counter.total == after_first


def test_pool_cache_reuse_under_l1_churn(device, tensor_map, isolate_program_cache):
    """
    The regression memory_used would cause if it were keyed: unrelated L1 allocations
    move the allocator total, so a key containing it would rebuild every dispatch.
    No growth here proves the key carries no live allocator state.
    """
    with device.cache_entries_counter.measure():
        run_pool(device, tensor_map, [1, 16, 25, 23])
    after_first = device.cache_entries_counter.total

    held = []
    with device.cache_entries_counter.measure():
        for i in range(3):
            held.append(
                ttnn.from_torch(
                    torch.randn([1, 1, 32, 32 * (i + 1)]),
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            )
            run_pool(device, tensor_map, [1, 16, 25, 23])
    assert device.cache_entries_counter.total == after_first


def test_pool_cache_miss_on_kernel_size(device, tensor_map, isolate_program_cache):
    """sliding_window_config is keyed."""
    with device.cache_entries_counter.measure():
        run_pool(device, tensor_map, [1, 16, 25, 23], kernel=(2, 2), stride=(2, 2))
    after_first = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        run_pool(device, tensor_map, [1, 16, 25, 23], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
    assert device.cache_entries_counter.total > after_first


def test_pool_cache_miss_on_input_shape(device, tensor_map, isolate_program_cache):
    with device.cache_entries_counter.measure():
        run_pool(device, tensor_map, [1, 16, 25, 23])
    after_first = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        run_pool(device, tensor_map, [1, 16, 50, 46])
    assert device.cache_entries_counter.total > after_first


def test_pool_cache_hit_reapplies_buffers(device, isolate_program_cache):
    """
    Freeze test with data the shared harness cannot dedupe: run_max_pool2d reuses torch
    tensors via tensor_map, so a fresh map per iteration is required for this to be able to
    fail at all. Three dispatches of DIFFERENT data in DIFFERENT buffers must share one
    program, and run_max_pool2d PCC-checks each against its own input.
    """
    with device.cache_entries_counter.measure():
        run_pool(device, {}, [1, 16, 25, 23])
    after_first = device.cache_entries_counter.total

    with device.cache_entries_counter.measure():
        for _ in range(3):
            run_pool(device, {}, [1, 16, 25, 23])
    assert device.cache_entries_counter.total == after_first
