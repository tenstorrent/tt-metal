# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""block_table: each distinct per-core block sent once in common args, one pointer per core.

A table family must decode exactly like the same family with per-core blocks, and move the same data."""

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import (
    KERNEL_DIR,
    core_set,
    make_cb,
    run_family_case,
    run_wrapper_case,
)

CT_WORDS = 11  # multicast compile-time header; chain families add one word
FIXED_GROUPS = [
    ([(0, 0), (2, 0), (3, 0), (0, 1)], [(0, 0)]),
    ([(3, 2), (4, 2)], [(2, 2)]),
    ([(1, 3)], [(1, 3)]),
]
ROTATING_GROUPS = [
    ([(0, 0), (2, 0), (3, 0), (0, 1)], [(0, 0), (4, 0)]),
    ([(1, 2)], [(1, 2), (3, 2)]),
    ([(0, 3), (2, 3)], [(0, 3), (2, 3)]),
]


def _grid(cols, rows):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])


def _family(device, config, groups):
    family = ttnn.McastFamily(device, config)
    for receivers, senders in groups:
        family.add_group(core_set(receivers), [ttnn.CoreCoord(*c) for c in senders])
    family.prepare_arguments()
    return family


# name -> (cores the kernel runs on, builder(device, McastConfig) -> helper)
TOPOLOGIES = {
    "row_uniform": (
        _grid(4, 3),
        lambda d, c: ttnn.Mcast1D(d, _grid(4, 3), ttnn.Mcast1DShape.PerRow, ttnn.Mcast1DFixedSenderConfig(), c),
    ),
    "row_diagonal": (
        _grid(4, 3),
        lambda d, c: ttnn.Mcast1D(
            d,
            _grid(4, 3),
            ttnn.Mcast1DShape.PerRow,
            ttnn.Mcast1DFixedSenderConfig(
                starting_sender_index=3, sender_placement=ttnn.Mcast1DSenderPlacement.Diagonal
            ),
            c,
        ),
    ),
    "column_diagonal": (
        _grid(5, 4),
        lambda d, c: ttnn.Mcast1D(
            d,
            _grid(5, 4),
            ttnn.Mcast1DShape.PerColumn,
            ttnn.Mcast1DFixedSenderConfig(
                starting_sender_index=1, sender_placement=ttnn.Mcast1DSenderPlacement.Diagonal
            ),
            c,
        ),
    ),
    "line_of_one": (
        _grid(3, 1),
        lambda d, c: ttnn.Mcast1D(d, _grid(3, 1), ttnn.Mcast1DShape.PerColumn, ttnn.Mcast1DFixedSenderConfig(), c),
    ),
    "rotating_line": (
        _grid(4, 2),
        lambda d, c: ttnn.Mcast1D(d, _grid(4, 2), ttnn.Mcast1DShape.PerRow, ttnn.Mcast1DRotatingSenderConfig(), c),
    ),
    "rectangle": (
        _grid(3, 3),
        lambda d, c: ttnn.Mcast2D(d, _grid(3, 3), ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(1, 1)), c),
    ),
    "irregular": (_grid(5, 4), lambda d, c: _family(d, c, FIXED_GROUPS)),
    "irregular_rotating": (_grid(5, 4), lambda d, c: _family(d, c, ROTATING_GROUPS)),
}


def _config(noc, block_table, chain=False):
    return ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        block_table=block_table,
        irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast if chain else ttnn.TransferMode.Multicast,
    )


def _attach_both(device, cores, build, noc, chain=False):
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/mcast_block_table_equivalence.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1 if noc == 0 else ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        ),
    )
    descriptor = ttnn.ProgramDescriptor()
    build(device, _config(noc, False, chain)).attach(descriptor, "p", [kernel])
    build(device, _config(noc, True, chain)).attach(descriptor, "t", [kernel])
    return descriptor, kernel


def _cores(cores):
    return [
        (x, y) for r in cores.ranges() for y in range(r.start.y, r.end.y + 1) for x in range(r.start.x, r.end.x + 1)
    ]


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("name", list(TOPOLOGIES) + ["chain"])
def test_table_holds_every_block(device, name, noc):
    """Host side: following a core's pointer into the table yields exactly its per-core block."""
    if name == "chain":
        cores, build, chain = _grid(5, 4), (lambda d, c: _family(d, c, FIXED_GROUPS)), True
    else:
        (cores, build), chain = TOPOLOGIES[name], False
    _, kernel = _attach_both(device, cores, build, noc, chain)
    named = dict(kernel.named_compile_time_args)
    p_rt, t_rt, t_ct = named["p_rt_offset"], named["t_rt_offset"], named["t_ct_offset"]
    ct = list(kernel.compile_time_args)
    assert (ct[t_ct + 5] >> 5) & 1 == 1, "table family sets the BLOCK_TABLE flag"
    table_base = ct[t_ct + CT_WORDS + int(chain)]
    common = list(kernel.common_runtime_args)
    words = t_rt - p_rt
    blocks = set()
    for x, y in _cores(cores):
        args = list(kernel.runtime_args[x][y])
        assert len(args) == t_rt + 1, f"core ({x},{y}): a table family adds exactly one word"
        block = args[p_rt:t_rt]
        pointer = args[t_rt]
        assert common[table_base + pointer : table_base + pointer + words] == block, f"core ({x},{y})"
        blocks.add(tuple(block))
    assert len(common) - table_base == words * len(blocks), "each distinct block appears once"


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("name", list(TOPOLOGIES))
def test_device_reads_match(device, name, noc):
    """Device side: every word and accessor reads the same through the table as through per-core blocks."""
    cores, build = TOPOLOGIES[name]
    size = device.compute_with_storage_grid_size()
    coords = _cores(cores)
    if any(x >= size.x or y >= size.y for x, y in coords):
        pytest.skip("requires a larger worker grid")
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(coords), 8]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(coords):
        rt[x][y] = [output.buffer_address(), i]
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/mcast_block_table_equivalence.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=list(ttnn.TensorAccessorArgs(output).get_compile_time_args()),
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1 if noc == 0 else ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        ),
    )
    descriptor = ttnn.ProgramDescriptor(cbs=[make_cb(0, cores, pages=1, page_bytes=32, dtype=ttnn.uint32)])
    build(device, _config(noc, False)).attach(descriptor, "p", [kernel])
    build(device, _config(noc, True)).attach(descriptor, "t", [kernel])
    descriptor.kernels = [kernel]
    result = ttnn.to_torch(ttnn.generic_op([output], descriptor)).to(torch.int64)
    for i, (x, y) in enumerate(coords):
        assert result[i, 0] == 0, f"core ({x},{y}): {int(result[i, 0])} mismatches (pointer {int(result[i, 2])})"
        assert result[i, 1] > 0


@pytest.mark.parametrize("noc", [0, 1])
def test_fixed_families_move_data(device, noc):
    run_family_case(device, FIXED_GROUPS, noc=noc, block_table=True)


@pytest.mark.parametrize("noc", [0, 1])
def test_rotating_families_move_data(device, noc):
    run_family_case(device, ROTATING_GROUPS, noc=noc, block_table=True)


@pytest.mark.parametrize("kind", ["row", "column", "rectangle"])
@pytest.mark.parametrize("rotating", [False, True])
def test_wrappers_move_data(device, kind, rotating):
    run_wrapper_case(
        device, width=4, senders=[0, 2] if rotating else [1], rotating=rotating, kind=kind, block_table=True
    )
