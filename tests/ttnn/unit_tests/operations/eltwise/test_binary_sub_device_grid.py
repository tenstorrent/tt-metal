# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Sub-device placement of binary ops whose output config is inferred from a sharded input."""

import time

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch


def _sub_device_delay_program(cores, iterations):
    runtime_args = ttnn.RuntimeArgs()
    for core_range in cores.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                runtime_args[x][y] = [iterations]
    kernel = ttnn.KernelDescriptor(
        kernel_source="tests/tt_metal/tt_metal/test_kernels/misc/sub_device/delay.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[],
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])


def _inputs(shape):
    numel = shape[2] * shape[3]
    torch_a = (torch.arange(numel, dtype=torch.float32).reshape(shape) % 7 + 1.0).bfloat16()
    torch_b = (torch.arange(numel, dtype=torch.float32).reshape(shape) % 5 + 2.0).bfloat16()
    return torch_a, torch_b


def _reference(op_name, torch_a, torch_b):
    if op_name == "add":
        return torch_a + torch_b
    if op_name == "multiply":
        return torch_a * torch_b
    return (torch_a > torch_b).bfloat16()


BINARY_OPS = [
    (ttnn.add, "add"),
    (ttnn.multiply, "multiply"),
    (ttnn.gt, "gt"),
]


@pytest.mark.parametrize("op_fn, op_name", BINARY_OPS)
@skip_for_slow_dispatch()
def test_binary_inferred_sharded_output_runs_on_its_sub_device(device, op_fn, op_name):
    """With a sharded `a`, DRAM `b` and no memory_config, the inferred output lives on the second sub-device,
    so the binary op must run there: a read that waits only for that sub-device must see its output."""
    sd0_cores = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))})
    sd1_cores = ttnn.CoreRangeSet({ttnn.CoreRange((4, 0), (5, 3))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([sd0_cores]), ttnn.SubDevice([sd1_cores])], 3200)
    device.load_sub_device_manager(manager)
    try:
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange((4, 0), (4, 3))})
        a_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        shape = (1, 1, 128, 128)
        torch_a, torch_b = _inputs(shape)
        expected = _reference(op_name, torch_a, torch_b)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_config
        )
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Compile and cache the op first, so the timed call below only dispatches.
        warm = op_fn(tt_a, tt_b)
        assert torch.equal(ttnn.to_torch(warm), expected)
        output_config = warm.memory_config()
        ttnn.deallocate(warm)
        # Leave a known value where the output will be allocated, so an unwritten output is detectable.
        placeholder = ttnn.from_torch(
            torch.full(shape, -7.0).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=output_config,
        )
        ttnn.deallocate(placeholder)

        dummy = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # Compile the delay kernel first, so the calibration below times only its run.
        ttnn.generic_op([dummy, dummy], _sub_device_delay_program(sd0_cores, 1))
        ttnn.synchronize_device(device)
        # Calibrate the delay kernel to roughly one second on this device.
        start = time.monotonic()
        ttnn.generic_op([dummy, dummy], _sub_device_delay_program(sd0_cores, 5_000_000))
        ttnn.synchronize_device(device)
        iterations = int(5_000_000 / max(time.monotonic() - start, 1e-3))

        ttnn.generic_op([dummy, dummy], _sub_device_delay_program(sd0_cores, iterations))
        device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
        try:
            out = op_fn(tt_a, tt_b)
            early = ttnn.to_torch(out)
        finally:
            device.reset_sub_device_stall_group()
            ttnn.synchronize_device(device)
        assert out.memory_config() == output_config
        assert torch.equal(early, expected), f"{int((early == -7).sum())} elements were read before it wrote them"
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


@pytest.mark.parametrize("op_fn, op_name", BINARY_OPS)
@skip_for_slow_dispatch()
def test_binary_inferred_sharded_output_explicit_configs(device, op_fn, op_name):
    """Controls: an explicit memory_config and an explicit sub_device_id both select the second sub-device,
    so a read that waits only for it sees the output."""
    sd0_cores = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))})
    sd1_cores = ttnn.CoreRangeSet({ttnn.CoreRange((4, 0), (5, 3))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([sd0_cores]), ttnn.SubDevice([sd1_cores])], 3200)
    device.load_sub_device_manager(manager)
    try:
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange((4, 0), (4, 3))})
        a_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        shape = (1, 1, 128, 128)
        torch_a, torch_b = _inputs(shape)
        expected = _reference(op_name, torch_a, torch_b)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_config
        )
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        out_explicit = op_fn(tt_a, tt_b, memory_config=a_config)
        ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(1)])
        assert out_explicit.memory_config() == a_config
        assert torch.equal(ttnn.to_torch(out_explicit), expected)

        out_sd = op_fn(tt_a, tt_b, sub_device_id=ttnn.SubDeviceId(1))
        ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(1)])
        assert torch.equal(ttnn.to_torch(out_sd), expected)
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


@pytest.mark.parametrize("op_fn, op_name", BINARY_OPS)
@skip_for_slow_dispatch()
def test_binary_inferred_sharded_output_single_sub_device(device, op_fn, op_name):
    """Control: with one sub-device the inferred output config selects it under either rule, so the values
    and the placement are unchanged."""
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 7))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([all_cores])], 3200)
    device.load_sub_device_manager(manager)
    try:
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 0))})
        a_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard_grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
        )
        shape = (1, 1, 128, 256)
        torch_a, torch_b = _inputs(shape)
        expected = _reference(op_name, torch_a, torch_b)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_config
        )
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        out = op_fn(tt_a, tt_b)
        ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0)])
        assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        assert torch.equal(ttnn.to_torch(out), expected)
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)
