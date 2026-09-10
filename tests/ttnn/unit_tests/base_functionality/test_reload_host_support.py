# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side support for the runtime binary reload: raw L1 access by mesh coordinate, configuring a
program without launching it, and the reload fields on ProgramDescriptor."""

import os

import pytest
import torch

import ttnn

MARKER = 0xC0FFEE01
CORE = ttnn.CoreCoord(0, 0)
CORE_SET = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
WORDS = 32  # one [1, 32] uint32 tile per core


def _l1_words(mesh_device):
    """A tiny L1 tensor sharded on CORE and replicated across the mesh: scratch this test owns,
    at the same address on every device."""
    shard = ttnn.ShardSpec(CORE_SET, (1, WORDS), ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    return ttnn.from_torch(
        torch.zeros(1, WORDS, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile([1, WORDS]),
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _marker_program(target):
    """One data-movement kernel on CORE that writes MARKER to ``target``'s L1 address."""
    rt = ttnn.RuntimeArgs()
    rt[CORE.x][CORE.y] = [target.buffer_address(), MARKER]
    kernel = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/base_functionality/kernels/write_marker.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=CORE_SET,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])


def _words(mesh_device, addr, n, coord=None):
    got = (
        mesh_device.read_core_l1(CORE, addr, 4 * n, coord)
        if coord is not None
        else mesh_device.read_core_l1(CORE, addr, 4 * n)
    )
    return list(got)[:n]


# ---------------------------------------------------------------- raw L1, by coordinate


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True)
def test_write_and_read_core_l1_round_trip_per_coordinate(mesh_device):
    """Each coordinate's L1 is its own: a word written to device (0,1) is not on device (0,0),
    and an omitted coordinate means the first device."""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("needs a 1x2 mesh")
    scratch = _l1_words(mesh_device)
    addr = scratch.buffer_address()
    coords = [ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)]
    for i, coord in enumerate(coords):
        mesh_device.write_core_l1(CORE, addr, [0x1000 + i, 0x2000 + i, 0x3000 + i], coord)
    for i, coord in enumerate(coords):
        assert _words(mesh_device, addr, 3, coord) == [0x1000 + i, 0x2000 + i, 0x3000 + i]
    assert _words(mesh_device, addr, 3) == [0x1000, 0x2000, 0x3000], "no coordinate = the first device"
    ttnn.deallocate(scratch)


# ---------------------------------------------------------------- configure without launch


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1")], indirect=True)
def test_configure_only_installs_the_program_and_never_runs_it(mesh_device):
    """With set_configure_only, a dispatch lands the kernel-config block on the core -- binary,
    runtime args, launch message -- and the kernel does not execute. The same descriptor,
    dispatched normally afterwards, runs."""
    if not os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        pytest.skip("configure-only is implemented on the slow-dispatch mesh command queue")
    scratch, other = _l1_words(mesh_device), _l1_words(mesh_device)
    addr = scratch.buffer_address()
    mesh_device.write_core_l1(CORE, addr, [0])
    program = _marker_program(scratch)

    ttnn.set_configure_only(mesh_device, True)
    try:
        ttnn.generic_op([other, scratch], program)
    finally:
        ttnn.set_configure_only(mesh_device, False)
    assert _words(mesh_device, addr, 1) == [0], "configure-only must not run the kernel"
    cfg = mesh_device.read_kernel_config(CORE)
    assert set(cfg) >= {"kernel_config_base", "kernel_text_offset", "kernel_text_size", "enables", "rta_offset"}
    assert max(cfg["kernel_text_size"]) > 0, "but the configured program's binary is on the core"
    assert cfg["enables"][0] != 0

    ttnn.generic_op([other, scratch], program)
    # Raw L1 reads do not wait for the asynchronously launched kernel to finish.
    ttnn.synchronize_device(mesh_device)
    assert _words(mesh_device, addr, 1) == [MARKER], "the same descriptor, launched for real, runs"
    ttnn.deallocate(scratch)
    ttnn.deallocate(other)


# ---------------------------------------------------------------- descriptor reload fields


def test_program_descriptor_reload_fields_default_off_and_round_trip():
    pd = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
    assert pd.reload_table_addr is None, "a program that does not reload has no table, not a table at 0"
    assert pd.reload_core_ranges.num_cores() == 0
    pd.reload_table_addr = 0x1000
    pd.reload_core_ranges = CORE_SET
    assert pd.reload_table_addr == 0x1000
    assert pd.reload_core_ranges.num_cores() == 1


def test_merging_descriptors_keeps_one_reload_table_or_refuses(expect_error):
    def pd(addr):
        d = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
        if addr is not None:
            d.reload_table_addr = addr
            d.reload_core_ranges = CORE_SET
        return d

    assert (
        ttnn.merge_program_descriptors([pd(None), pd(0x1000)]).reload_table_addr == 0x1000
    ), "a table joins one without"
    assert ttnn.merge_program_descriptors([pd(0x1000), pd(0x1000)]).reload_table_addr == 0x1000, "the same table merges"
    assert ttnn.merge_program_descriptors([pd(None), pd(None)]).reload_table_addr is None
    with expect_error(RuntimeError, "different reload_table_addr"):
        ttnn.merge_program_descriptors([pd(0x1000), pd(0x2000)])


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1")], indirect=True)
def test_reload_table_address_is_part_of_the_program_cache_key(mesh_device):
    """Two descriptors that differ only in reload_table_addr are two cached programs: the address
    is baked into the cached launch message, so a program first run without a table must not keep
    running without one. Stock firmware ignores the field, so setting it here is inert."""
    mesh_device.enable_program_cache()
    scratch, other = _l1_words(mesh_device), _l1_words(mesh_device)
    base = mesh_device.num_program_cache_entries()

    plain = _marker_program(scratch)
    ttnn.generic_op([other, scratch], plain)
    assert mesh_device.num_program_cache_entries() == base + 1
    ttnn.generic_op([other, scratch], plain)
    assert mesh_device.num_program_cache_entries() == base + 1, "same descriptor: a cache hit"

    tabled = _marker_program(scratch)
    tabled.reload_table_addr = 0x1000
    ttnn.generic_op([other, scratch], tabled)
    assert mesh_device.num_program_cache_entries() == base + 2, "a reload table address is a different program"
    ttnn.deallocate(scratch)
    ttnn.deallocate(other)
