# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Program-cache behaviour for ttnn.slice.

SliceDeviceOperation carried a custom compute_program_hash that hand-listed all 8
SliceParams members plus derived state (factory index, output spec, per-tensor
shape/layout/dtype). SliceParams has no attribute_names, so the framework default
hash keys the same 8 members, derives the rest, and additionally keys end_tensor
and preallocated_output -- which the custom hash ignored.

These tests pin the cache-entry counts on every axis so the removal is proven,
not asserted: they were written against the custom hash and must give identical
counts without it.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_slice(device, shape, begins, ends, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    slices = tuple(slice(begins[i], ends[i]) for i in range(len(begins)))
    torch_output = torch_input[slices]

    tt_input = ttnn.from_torch(
        torch_input, layout=layout, dtype=ttnn.bfloat16, device=device, memory_config=memory_config
    )
    with device.cache_entries_counter.measure():
        tt_output = ttnn.slice(tt_input, begins, ends, memory_config=memory_config)

    assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.999)


def run_slice_tensor_args(device, shape, begins, ends, dim, layout=ttnn.TILE_LAYOUT):
    """Device-resident start/end tensors: takes the use_tensor_args factory.
    Host-side start/end tensors are read on host and fall back to the scalar path."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    slices = tuple(slice(begins[i], ends[i]) for i in range(len(begins)))
    torch_output = torch_input[slices]

    tt_input = ttnn.from_torch(torch_input, layout=layout, dtype=ttnn.bfloat16, device=device)
    tt_begins = ttnn.from_torch(torch.tensor(begins), device=device)
    tt_ends = ttnn.from_torch(torch.tensor(ends), device=device)
    num_devices = shape[dim] // (ends[dim] - begins[dim])
    with device.cache_entries_counter.measure():
        tt_output = ttnn.slice(tt_input, tt_begins, tt_ends, slice_dim=dim, num_devices=num_devices)

    assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.999)


def test_slice_cache_reuse_same_config(device, isolate_program_cache):
    """Identical calls share one program."""
    for _ in range(3):
        run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 64, 64])
    assert device.cache_entries_counter.total == 1


def test_slice_cache_reuse_fresh_tensors(device, isolate_program_cache):
    """Fresh allocations each call: addresses ride bindings, they are not in the key."""
    for _ in range(3):
        run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 64, 64])
        dummy = ttnn.from_torch(
            torch.randn([1, 1, 32, 32]), layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(dummy)
    assert device.cache_entries_counter.total == 1


def test_slice_cache_miss_on_slice_bounds(device, isolate_program_cache):
    """slice_end is keyed: a different output extent is a different program."""
    run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 64, 64])
    run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 96, 64])
    assert device.cache_entries_counter.total == 2


def test_slice_cache_miss_on_input_shape(device, isolate_program_cache):
    """Input logical shape is keyed via TensorSpec."""
    run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 64, 64])
    run_slice(device, [1, 1, 256, 128], [0, 0, 0, 0], [1, 1, 64, 64])
    assert device.cache_entries_counter.total == 2


def test_slice_cache_miss_on_layout(device, isolate_program_cache):
    """Layout selects the program factory."""
    run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 64, 64], layout=ttnn.TILE_LAYOUT)
    run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 64, 64], layout=ttnn.ROW_MAJOR_LAYOUT)
    assert device.cache_entries_counter.total == 2


def test_slice_cache_scalar_vs_tensor_args_distinct(device, isolate_program_cache):
    """use_tensor_args picks a different factory, so the two paths must not collide."""
    run_slice(device, [1, 16, 128, 128], [0, 0, 0, 0], [1, 16, 64, 128])
    run_slice_tensor_args(device, [1, 16, 128, 128], [0, 0, 0, 0], [1, 16, 64, 128], dim=2)
    assert device.cache_entries_counter.total == 2


def test_slice_cache_tensor_args_reuse(device, isolate_program_cache):
    """Same tensor-args config twice: the start/end tensor VALUES are not keyed."""
    for _ in range(2):
        run_slice_tensor_args(device, [1, 16, 128, 128], [0, 0, 0, 0], [1, 16, 64, 128], dim=2)
    assert device.cache_entries_counter.total == 1


def test_slice_cache_miss_on_sub_core_grid(device, isolate_program_cache):
    """sub_core_grids is a keyed attribute."""
    run_slice(device, [1, 1, 128, 128], [0, 0, 0, 0], [1, 1, 64, 64])
    torch_input = torch.randn([1, 1, 128, 128], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    with device.cache_entries_counter.measure():
        ttnn.slice(tt_input, [0, 0, 0, 0], [1, 1, 64, 64], sub_core_grids=grid)
    assert device.cache_entries_counter.total == 2


def test_slice_cache_hit_reapplies_buffers(device, isolate_program_cache):
    """
    Freeze test. Three dispatches of DIFFERENT data in DIFFERENT buffers must share one
    program and each must return its own data. A frozen buffer address or a frozen
    per-dispatch scalar returns the first dispatch's result and fails PCC here.
    """
    with device.cache_entries_counter.measure():
        for i in range(3):
            torch_input = torch.randn([1, 1, 128, 128], dtype=torch.bfloat16) * (i + 1)
            tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
            tt_output = ttnn.slice(tt_input, [0, 0, 0, 0], [1, 1, 64, 64])
            assert_with_pcc(torch_input[:, :, 0:64, 0:64], ttnn.to_torch(tt_output), 0.999)
    assert device.cache_entries_counter.total == 1


def test_slice_cache_preallocated_output_adds_entry(device, isolate_program_cache):
    """
    preallocated_output rides tensor_args, so the default hash keys it and a slice writing
    into a caller-provided output no longer shares the plain slice's program. The custom
    hash ignored it and reused that program, which was legitimate -- the output buffer rides
    a binding -- so this costs exactly one extra cache entry on this axis. Pinned so the
    cost stays visible.
    """
    torch_input = torch.randn([1, 3, 640, 640], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    with device.cache_entries_counter.measure():
        ttnn.slice(tt_input, starts=(0, 0, 0, 0), ends=(1, 3, 320, 320), steps=(1, 1, 1, 1))
    assert device.cache_entries_counter.total == 1

    tt_out = ttnn.from_torch(
        torch.zeros([1, 3, 320, 320], dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with device.cache_entries_counter.measure():
        ttnn.slice(tt_input, starts=(0, 0, 0, 0), ends=(1, 3, 320, 320), steps=(1, 1, 1, 1), output_tensor=tt_out)
    assert device.cache_entries_counter.total == 3
    assert_with_pcc(torch_input[:, :, 0:320, 0:320], ttnn.to_torch(tt_out), 0.999)
