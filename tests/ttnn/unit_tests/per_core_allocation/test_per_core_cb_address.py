# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A circular buffer on a per-core-allocated tensor is programmed at its own core's shard.

A per-core tensor's shard sits at a different address on each core, and Buffer::address() is only
the first core's. A CB has one address, which used to be taken from Buffer::address(), so a CB on
any other core landed on the first core's address. A kernel's get_write_ptr then disagreed with the
raw per-core address the host and other cores used for the same shard (how tt-blaze's router demux
lost its pages).

Every test skews one core's per-core allocator first, so the two cores' shards sit at different
addresses. Skewing the first core puts the second core's shard above Buffer::address(); skewing the
second puts it below.
"""

import pytest
import torch

import ttnn
from conftest import requires_hybrid_allocator

PAGE = 1024
SHARD = 2 * PAGE
LAST_PAGE = SHARD - PAGE
CB_INDEX = 0
FIRST, SECOND, WRITER = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)

SKEWS = pytest.mark.parametrize("skewed", ["first", "second"], ids=["second_above_first", "second_below_first"])


def _one(core):
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _cores(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])


def _sharded_tensor(mesh, cores, shard_bytes, *, per_core, data=None):
    """A HEIGHT_SHARDED uint8 L1 tensor with one ``shard_bytes`` shard on each of ``cores``."""
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_cores(cores), [1, shard_bytes], ttnn.ShardOrientation.ROW_MAJOR),
    )
    if per_core:
        mem_config.experimental_set_per_core_allocation(True)
    if data is None:
        data = torch.zeros(len(cores), shard_bytes, dtype=torch.uint8)
    return ttnn.from_torch(
        data,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _addr(tensor, core):
    (coord,) = tensor.device_coords()
    return int(tensor.experimental_per_core_buffer_address(coord, core))


def _skewed_tensor(mesh, skewed, skew_bytes=2 * SHARD):
    """A per-core tensor over FIRST and SECOND whose two shards sit at different addresses.

    A per-core reservation of ``skew_bytes`` on one core only pushes the next allocation lower there.
    Returns (tensor, skew); keep both alive.
    """
    skew = _sharded_tensor(mesh, [FIRST if skewed == "first" else SECOND], skew_bytes, per_core=True)
    tensor = _sharded_tensor(mesh, [FIRST, SECOND], SHARD, per_core=True)
    assert _addr(tensor, FIRST) != _addr(tensor, SECOND), "the skew did not separate the shards"
    return tensor, skew


def _cb(tensor, core, offset):
    """One single-core CB on ``core`` over ``tensor``, a page long, ``offset`` into the shard."""
    descriptor = ttnn.cb_descriptor_from_sharded_tensor(
        CB_INDEX, tensor, address_offset=offset, total_size=PAGE, core_ranges=_one(core)
    )
    descriptor.format_descriptors = [ttnn.CBFormatDescriptor(CB_INDEX, ttnn.uint8, PAGE)]
    return descriptor


def _run(mesh, io_tensors, program):
    mesh_program = ttnn.MeshProgramDescriptor()
    coord = ttnn.MeshCoordinate(0, 0)
    mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
    ttnn.generic_op(io_tensors, mesh_program)
    ttnn.synchronize_device(mesh)


# Writes (get_write_ptr(cb) == expected, expected, get_write_ptr(cb)) to this core's output shard.
_PROBE_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    const uint32_t expected = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    const uint32_t cb_addr = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
    out[0] = cb_addr == expected ? 1 : 0;
    out[1] = expected;
    out[2] = cb_addr;
}
"""


def _probe_program(tensor, out, offset):
    """One CB per core over ``tensor`` and a kernel on each comparing its CB with the raw address."""
    runtime_args = ttnn.RuntimeArgs()
    for core in (FIRST, SECOND):
        runtime_args[core.x][core.y] = [_addr(tensor, core) + offset, out.buffer_address()]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_PROBE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_cores([FIRST, SECOND]),
        compile_time_args=[CB_INDEX],
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(
        kernels=[kernel], semaphores=[], cbs=[_cb(tensor, FIRST, offset), _cb(tensor, SECOND, offset)]
    )


def _probe_results(out):
    rows = ttnn.to_torch(out).contiguous().view(torch.int32).reshape(2, -1)
    return {core: [int(v) for v in rows[i, :3]] for i, core in enumerate((FIRST, SECOND))}


def _assert_probe_matched(results):
    for core, (match, expected, cb_addr) in results.items():
        assert match == 1, f"core {core}: CB write pointer {cb_addr:#x} != per-core address {expected:#x}"


@requires_hybrid_allocator
@SKEWS
@pytest.mark.parametrize("offset", [0, LAST_PAGE], ids=["start_of_shard", "last_page_of_shard"])
def test_cb_address_is_its_own_cores_shard(per_core_mesh_device, skewed, offset):
    """get_cb_address of a single-core CB is that core's shard plus the offset, on either core.

    The last-page case also builds a descriptor that ends exactly at the end of its shard, on a core
    away from Buffer::address(), which the address_offset + total_size check must accept.
    """
    tensor, _skew = _skewed_tensor(per_core_mesh_device, skewed)
    for core in (FIRST, SECOND):
        assert ttnn.get_cb_address(_cb(tensor, core, offset)) == _addr(tensor, core) + offset, f"core {core}"


@requires_hybrid_allocator
@SKEWS
def test_cb_write_pointer_matches_the_raw_per_core_address(per_core_mesh_device, skewed):
    """On hardware, each core's CB write pointer is its own shard (the address raw accesses use)."""
    tensor, _skew = _skewed_tensor(per_core_mesh_device, skewed)
    out = _sharded_tensor(per_core_mesh_device, [FIRST, SECOND], 16, per_core=False)

    _run(per_core_mesh_device, [tensor, out], _probe_program(tensor, out, LAST_PAGE))

    _assert_probe_matched(_probe_results(out))


# Sends page i of a local buffer to receiver i at a raw L1 address, then bumps the receiver's semaphore.
_WRITER_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_receivers = get_arg_val<uint32_t>(1);
    const uint32_t page_bytes = get_arg_val<uint32_t>(2);
    const uint32_t semaphore_addr = get_semaphore(0);

    for (uint32_t i = 0; i < num_receivers; ++i) {
        const uint32_t noc_x = get_arg_val<uint32_t>(3 + 3 * i);
        const uint32_t noc_y = get_arg_val<uint32_t>(4 + 3 * i);
        const uint32_t dst_addr = get_arg_val<uint32_t>(5 + 3 * i);
        noc_async_write(src_addr + i * page_bytes, get_noc_addr(noc_x, noc_y, dst_addr), page_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(noc_x, noc_y, semaphore_addr), 1);
    }
    noc_async_atomic_barrier();
}
"""

# Waits for the writer, then copies a page from its CB's start to this core's output shard.
_RECEIVER_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_bytes = get_arg_val<uint32_t>(1);

    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
    noc_async_read(get_noc_addr(get_write_ptr(cb)), out_addr, page_bytes);
    noc_async_read_barrier();
}
"""


@requires_hybrid_allocator
@SKEWS
def test_pages_written_by_raw_address_are_read_through_the_cb(per_core_mesh_device, skewed):
    """Another core writes pages into each receiver's CB by raw per-core address; the receiver reads
    them through the CB. This is the tt-blaze router demux pattern that lost its pages."""
    mesh = per_core_mesh_device
    tensor, _skew = _skewed_tensor(mesh, skewed)
    receivers = (FIRST, SECOND)
    pages = torch.stack([torch.full((PAGE,), 0x11 * (i + 1), dtype=torch.uint8) for i in range(len(receivers))])
    source = _sharded_tensor(mesh, [WRITER], len(receivers) * PAGE, per_core=False, data=pages.reshape(1, -1))
    out = _sharded_tensor(mesh, list(receivers), PAGE, per_core=False)

    to_receivers = [source.buffer_address(), len(receivers), PAGE]
    for core in receivers:
        noc = mesh.worker_core_from_logical_core(core)
        to_receivers += [noc.x, noc.y, _addr(tensor, core) + LAST_PAGE]
    writer_args = ttnn.RuntimeArgs()
    writer_args[WRITER.x][WRITER.y] = to_receivers
    receiver_args = ttnn.RuntimeArgs()
    for core in receivers:
        receiver_args[core.x][core.y] = [out.buffer_address(), PAGE]

    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_one(WRITER),
        compile_time_args=[],
        runtime_args=writer_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    receiver = ttnn.KernelDescriptor(
        kernel_source=_RECEIVER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_cores(receivers),
        compile_time_args=[CB_INDEX],
        runtime_args=receiver_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(
        kernels=[writer, receiver],
        semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=_cores([*receivers, WRITER]), initial_value=0)],
        cbs=[_cb(tensor, core, LAST_PAGE) for core in receivers],
    )

    _run(mesh, [tensor, source, out], program)

    result = ttnn.to_torch(out).reshape(len(receivers), PAGE)
    for i, core in enumerate(receivers):
        assert torch.equal(result[i], pages[i]), f"core {core} read {result[i, :8].tolist()} through its CB"


@requires_hybrid_allocator
def test_cached_program_follows_a_new_per_core_tensor(per_core_mesh_device):
    """A cached program re-run on a per-core tensor at other addresses re-points each CB at its own core.

    The second run is a program cache hit, so the CBs are re-pointed (UpdateDynamicCircularBufferAddress)
    rather than rebuilt, and the cached dispatch commands must pick up the new per-core addresses.
    """
    mesh = per_core_mesh_device
    first, _first_skew = _skewed_tensor(mesh, "first")
    # FIRST already holds more than SECOND, so SECOND needs a bigger skew to end up below it.
    second, _second_skew = _skewed_tensor(mesh, "second", skew_bytes=4 * SHARD)
    for core in (FIRST, SECOND):
        assert _addr(first, core) != _addr(second, core), f"core {core}: both tensors share a shard address"
    out = _sharded_tensor(mesh, [FIRST, SECOND], 16, per_core=False)

    _run(mesh, [first, out], _probe_program(first, out, LAST_PAGE))
    _assert_probe_matched(_probe_results(out))
    entries = mesh.num_program_cache_entries()

    _run(mesh, [second, out], _probe_program(second, out, LAST_PAGE))
    assert mesh.num_program_cache_entries() == entries, "the second run was not a program cache hit"
    _assert_probe_matched(_probe_results(out))


@requires_hybrid_allocator
def test_cb_spanning_cores_at_different_addresses_is_rejected(per_core_mesh_device, expect_error):
    """One CB over both cores has no single right address, so asking for it or building it fails."""
    tensor, _skew = _skewed_tensor(per_core_mesh_device, "first")
    spanning = ttnn.cb_descriptor_from_sharded_tensor(
        CB_INDEX, tensor, total_size=PAGE, core_ranges=_cores([FIRST, SECOND])
    )
    spanning.format_descriptors = [ttnn.CBFormatDescriptor(CB_INDEX, ttnn.uint8, PAGE)]

    with expect_error(RuntimeError, "a circular buffer has one address"):
        ttnn.get_cb_address(spanning)

    out = _sharded_tensor(per_core_mesh_device, [FIRST, SECOND], 16, per_core=False)
    program = _probe_program(tensor, out, 0)
    program.cbs = [spanning]
    with expect_error(RuntimeError, "a circular buffer has one address"):
        _run(per_core_mesh_device, [tensor, out], program)
