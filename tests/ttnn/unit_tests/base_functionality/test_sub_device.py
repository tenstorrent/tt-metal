# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch


def run_sub_devices(device):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_devices_1 = [sub_device_1, sub_device_2]
    sub_devices_2 = [sub_device_2]
    sub_device_manager1 = device.create_sub_device_manager(sub_devices_1, 3200)
    sub_device_manager2 = device.create_sub_device_manager(sub_devices_2, 3200)
    device.load_sub_device_manager(sub_device_manager1)
    ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(1)])
    ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    ttnn.synchronize_device(device)
    device.reset_sub_device_stall_group()
    ttnn.synchronize_device(device)
    device.load_sub_device_manager(sub_device_manager2)
    ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0)])
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager1)
    device.remove_sub_device_manager(sub_device_manager2)


def run_sub_devices_program(device):
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    if is_mesh_device:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        num_devices = device.get_num_devices()
    else:
        inputs_mesh_mapper = None
        output_mesh_composer = None
        num_devices = 1
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_devices = [sub_device_1, sub_device_2]
    sub_device_manager = device.create_sub_device_manager(sub_devices, 3200)
    device.load_sub_device_manager(sub_device_manager)

    x = torch.randn(num_devices, 1, 64, 64, dtype=torch.bfloat16)
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
    )
    device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
    xt_host = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )

    ttnn.copy_host_to_device_tensor(xt_host, xt)

    grid_size = device.compute_with_storage_grid_size()
    shard_size = [32, 64]
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    yt = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=ttnn.bfloat16
    )
    y = ttnn.to_torch(yt, device=device, mesh_composer=output_mesh_composer)

    eq = torch.equal(x, y)
    assert eq
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    y = ttnn.to_torch(yt.cpu(), mesh_composer=output_mesh_composer)

    eq = torch.equal(x, y)
    assert eq

    yt2 = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=ttnn.bfloat16
    )
    event = ttnn.record_event(device, 0, [ttnn.SubDeviceId(1)])
    ttnn.wait_for_event(0, event)
    y2 = ttnn.to_torch(yt2, device=device, mesh_composer=output_mesh_composer)

    eq = torch.equal(x, y2)
    assert eq

    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


@skip_for_slow_dispatch()
def test_sub_devices(device):
    run_sub_devices(device)


@skip_for_slow_dispatch()
def test_sub_devices_mesh(mesh_device):
    run_sub_devices(mesh_device)


@skip_for_slow_dispatch()
def test_sub_device_program(device):
    run_sub_devices_program(device)


@skip_for_slow_dispatch()
def test_sub_device_program_mesh(mesh_device):
    run_sub_devices_program(mesh_device)


# Watches the runtime args and semaphore that fast dispatch wrote into its kernel-config slot, and reports the first
# change. Its runtime args open the slot and cover every word another program's misplaced semaphore or circular buffer
# config would land on, so such a write changes data this kernel reads, never its code.
_WATCHER_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_args = get_compile_time_arg_val(0);
    constexpr uint32_t arg_base = get_compile_time_arg_val(1);
    constexpr uint32_t semaphore_value = get_compile_time_arg_val(2);
    constexpr uint32_t report_addr = get_compile_time_arg_val(3);
    constexpr uint32_t polls = get_compile_time_arg_val(4);
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    volatile tt_l1_ptr uint32_t* args = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(0));
    volatile tt_l1_ptr uint32_t* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    report[0] = 0;
    for (uint32_t poll = 0; poll < polls && report[0] == 0; ++poll) {
        for (uint32_t i = 0; i < num_args; ++i) {
            if (args[i] != arg_base + i) {
                report[0] = 1;
                report[1] = i;
                report[2] = args[i];
                break;
            }
        }
        if (report[0] == 0 && *semaphore != semaphore_value) {
            report[0] = 2;
            report[2] = *semaphore;
        }
    }
}
"""
_NUM_ARGS, _ARG_BASE, _WATCHER_SEMAPHORE, _SPILLED_SEMAPHORE, _POLLS, _REPORT_WORDS = (
    128,
    0xA1A10000,
    0x5EED0001,
    0xBAD05E30,
    50_000,
    8,
)


def _cores(*ranges):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in ranges]
    )


def _watcher(column, rows, report_address):
    runtime_args = ttnn.RuntimeArgs()
    for y in range(rows):
        runtime_args[column][y] = [_ARG_BASE + i for i in range(_NUM_ARGS)]
    cores = _cores((column, 0, column, rows - 1))
    kernel = ttnn.KernelDescriptor(
        kernel_source=_WATCHER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[_NUM_ARGS, _ARG_BASE, _WATCHER_SEMAPHORE, report_address, _POLLS],
        runtime_args=runtime_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    semaphores = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=_WATCHER_SEMAPHORE)]
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=semaphores)


def _spiller(spilled, cores, variant=0):
    """A kernel on core (0, 0) only, and a semaphore or circular buffer on `cores`."""
    kernel = ttnn.KernelDescriptor(
        kernel_source='#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n',
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_cores((0, 0, 0, 0)),
        compile_time_args=[variant],
        config=ttnn.WriterConfigDescriptor(),
    )
    if spilled == "semaphore":
        semaphores = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=_SPILLED_SEMAPHORE)]
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=semaphores)
    circular_buffer = ttnn.CBDescriptor(
        total_size=192,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=64)],
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], cbs=[circular_buffer])


@skip_for_slow_dispatch()
@pytest.mark.parametrize("spilled", ["semaphore", "circular_buffer"])
def test_a_program_cannot_spill_its_objects_into_a_program_running_on_another_sub_device(device, spilled, expect_error):
    """Sub-device 1 (the last column) runs a program; a program on sub-device 0's core (0, 0) also places a semaphore
    or circular buffer on the last column. Fast dispatch wrote that object's initial value or config into sub-device 1's
    kernel-config ring, over the running program's runtime args, 20 times out of 20 right after a manager load. Metal
    refuses the second program before anything of it is written."""
    grid = device.compute_with_storage_grid_size()
    last, rows = grid.x - 1, grid.y
    sub_device_0, sub_device_1 = _cores((0, 0, last - 1, rows - 1)), _cores((last, 0, last, rows - 1))
    spill = _cores((0, 0, 0, 0), (last, 0, last, rows - 1))
    report = ttnn.from_torch(
        torch.zeros(rows, _REPORT_WORDS, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(sub_device_1, [1, _REPORT_WORDS], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    out = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, device=device)
    io = [report, out]  # generic_op takes an input and an output; neither kernel touches them through it
    watcher = _watcher(last, rows, report.buffer_address())
    # With no manager loaded both run, one after the other on one sub-device; the run below only dispatches them.
    ttnn.generic_op(io, watcher)
    ttnn.generic_op(io, _spiller(spilled, spill))
    ttnn.synchronize_device(device)

    manager = device.create_sub_device_manager([ttnn.SubDevice([sub_device_0]), ttnn.SubDevice([sub_device_1])], 0)
    device.load_sub_device_manager(manager)  # fresh kernel-config rings: both programs take their first slot
    try:
        ttnn.generic_op(io, watcher)
        refusal = None
        try:
            ttnn.generic_op(io, _spiller(spilled, spill))
        except RuntimeError as error:
            refusal = str(error)
        ttnn.synchronize_device(device)
        seen = (ttnn.to_torch(report).to(torch.int64) & 0xFFFFFFFF).tolist()
        changed = [(y, row[:3]) for y, row in enumerate(seen) if row[0] != 0]
        assert (
            not changed
        ), f"sub-device 1's program saw its kernel config overwritten (row, [kind, index, value]): {changed}"
        assert refusal is not None and "in sub-device 1, outside sub-device 0 where its kernels run" in refusal, refusal
        for _ in range(2):  # built under the manager this time, and nothing of a refusal is cached
            with expect_error(RuntimeError, "outside sub-device 0 where its kernels run"):
                ttnn.generic_op(io, _spiller(spilled, spill, variant=1))
        # Cores of the kernels' sub-device that run no kernel are fine.
        ttnn.generic_op(io, _spiller(spilled, _cores((0, 0, 1, 1))))
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)
