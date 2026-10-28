# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""block_table: each distinct per-core block sent once in common args, one pointer per core.

A table channel must decode exactly like the same channel with per-core blocks, and move the same data."""

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import (
    KERNEL_DIR,
    inspect_mcast_ct,
    make_cb,
    make_mcast,
    run_mcast_groups_case,
    run_positional_mcast_case,
)

FIXED_GROUPS = [
    ([(0, 0), (2, 0), (3, 0), (0, 1)], [(0, 0)]),
    ([(3, 2), (4, 2)], [(2, 2)]),
    ([(1, 3)], [(1, 3)]),
]
# Equal consecutive row-major groups, one of them irregular (an L): representable as one Mcast channel.
IRREGULAR_GROUPS = [
    ([(0, 0), (1, 0), (0, 1)], [(0, 0)]),
    ([(1, 1), (2, 1), (3, 1)], [(2, 1)]),
]

ROTATING_GROUPS = [
    ([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (2, 0)]),
    ([(1, 2), (2, 2)], [(1, 2), (3, 2)]),
    ([(0, 3), (1, 3)], [(0, 3), (1, 3)]),
]


def _grid(cols, rows, x0=0, y0=0):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + cols - 1, y0 + rows - 1))])


def _staggered(index):
    return ttnn.McastFixedSenderConfig(sender_index=index, placement=ttnn.McastSenderPlacement.Staggered)


# name -> (cores the kernel runs on, builder(device, McastConfig) -> Mcast)
TOPOLOGIES = {
    "rows_uniform": (_grid(4, 3), lambda d, c: ttnn.Mcast(d, c, _grid(4, 3), 4, ttnn.McastFixedSenderConfig())),
    "rows_staggered": (_grid(4, 3), lambda d, c: ttnn.Mcast(d, c, _grid(4, 3), 4, _staggered(3))),
    "columns_staggered": (
        _grid(5, 4),
        lambda d, c: ttnn.Mcast(d, c, _grid(5, 4), 4, _staggered(1), ttnn.McastCoreOrder.ColumnMajor),
    ),
    "line_of_one": (
        _grid(3, 1),
        lambda d, c: ttnn.Mcast(d, c, _grid(3, 1), 1, ttnn.McastFixedSenderConfig(), ttnn.McastCoreOrder.ColumnMajor),
    ),
    "rectangle": (
        _grid(3, 3),
        lambda d, c: ttnn.Mcast(d, c, _grid(3, 3), 9, ttnn.McastFixedSenderConfig(sender_index=4)),
    ),
    "rotating_rows": (_grid(4, 2), lambda d, c: ttnn.Mcast(d, c, _grid(4, 2), 4, ttnn.McastRotatingSenderConfig())),
    "outside_senders": (
        _grid(5, 2),
        lambda d, c: ttnn.Mcast(
            d,
            c,
            _grid(3, 2),
            3,
            ttnn.McastExplicitFixedSenderConfig([ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 1)]),
        ),
    ),
    "irregular": (_grid(4, 2), lambda d, c: make_mcast(d, IRREGULAR_GROUPS, c)),
    "rotating_sender_grid": (
        _grid(5, 2),
        lambda d, c: ttnn.Mcast(
            d,
            c,
            _grid(3, 2),
            3,
            ttnn.McastRotatingSenderConfig(_grid(2, 2, x0=3), sender_order=ttnn.McastCoreOrder.RowMajor),
        ),
    ),
}


def _config(noc, block_table, chain=False):
    return ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        block_table=block_table,
        irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast if chain else ttnn.TransferMode.Multicast,
    )


def _kernel(cores, noc, **kwargs):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/mcast_block_table_equivalence.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1 if noc == 0 else ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        ),
        **kwargs,
    )


def _attach_both(device, descriptor, kernel, build, noc, chain=False):
    positional = build(device, _config(noc, False, chain))
    positional.attach(descriptor, "p", [kernel], 0)
    build(device, _config(noc, True, chain)).attach(descriptor, "t", [kernel], positional.next_semaphore_id())


def _coords(cores):
    return [
        (x, y) for r in cores.ranges() for y in range(r.start.y, r.end.y + 1) for x in range(r.start.x, r.end.x + 1)
    ]


def _fits(device, cores):
    size = device.compute_with_storage_grid_size()
    return all(x < size.x and y < size.y for x, y in _coords(cores))


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("name", list(TOPOLOGIES) + ["chain"])
def test_table_holds_every_block(device, name, noc):
    """Host side: following a core's pointer into the table yields exactly its per-core block."""
    if name == "chain":
        cores, build, chain = _grid(4, 2), (lambda d, c: make_mcast(d, IRREGULAR_GROUPS, c)), True
    else:
        (cores, build), chain = TOPOLOGIES[name], False
    if not _fits(device, cores):
        pytest.skip("requires a larger worker grid")
    kernel = _kernel(cores, noc)
    _attach_both(device, ttnn.ProgramDescriptor(), kernel, build, noc, chain)
    named = dict(kernel.named_compile_time_args)
    p_rt, t_rt = named["p_rt_offset"], named["t_rt_offset"]
    table_base = inspect_mcast_ct(kernel, "t")["table"]
    assert inspect_mcast_ct(kernel, "p")["table"] is None
    common = list(kernel.common_runtime_args)
    words = t_rt - p_rt
    if words == 0:
        # Nothing per core (e.g. every core is the lone sender of its line): the table is dropped, no words added.
        assert table_base is None and not common
        assert all(len(list(kernel.runtime_args[x][y])) == t_rt for x, y in _coords(cores))
        return
    assert table_base is not None, "table channel sets the block-table control bit"
    blocks = set()
    for x, y in _coords(cores):
        args = list(kernel.runtime_args[x][y])
        assert len(args) == t_rt + 1, f"core ({x},{y}): a table channel adds exactly one word"
        block, pointer = args[p_rt:t_rt], args[t_rt]
        assert common[table_base + pointer : table_base + pointer + words] == block, f"core ({x},{y})"
        blocks.add(tuple(block))
    assert len(common) - table_base == words * len(blocks), "each distinct block appears once"


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("name", list(TOPOLOGIES))
def test_device_reads_match(device, name, noc):
    """Device side: every word and accessor reads the same through the table as through per-core blocks."""
    cores, build = TOPOLOGIES[name]
    if not _fits(device, cores):
        pytest.skip("requires a larger worker grid")
    coords = _coords(cores)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(coords), 8]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    unused_input = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 8]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )  # generic_op takes inputs then the output
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(coords):
        rt[x][y] = [output.buffer_address(), i]
    kernel = _kernel(
        cores, noc, compile_time_args=list(ttnn.TensorAccessorArgs(output).get_compile_time_args()), runtime_args=rt
    )
    descriptor = ttnn.ProgramDescriptor(cbs=[make_cb(0, cores, pages=1, page_bytes=32, dtype=ttnn.uint32)])
    _attach_both(device, descriptor, kernel, build, noc)
    descriptor.kernels = [kernel]
    named = dict(kernel.named_compile_time_args)
    words = named["t_rt_offset"] - named["p_rt_offset"]  # the positional block's size, as the host laid it out
    result = ttnn.to_torch(ttnn.generic_op([unused_input, output], descriptor)).to(torch.int64)
    for i, (x, y) in enumerate(coords):
        assert result[i, 0] == 0, f"core ({x},{y}): {int(result[i, 0])} mismatches (pointer {int(result[i, 2])})"
        assert result[i, 1] == words, f"core ({x},{y}): device decoded {int(result[i, 1])} words, host laid out {words}"


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_fixed_groups_move_data(device, noc, counter):
    run_mcast_groups_case(device, FIXED_GROUPS, noc=noc, counter=counter, block_table=True)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_rotating_groups_move_data(device, noc, counter):
    run_mcast_groups_case(device, ROTATING_GROUPS, noc=noc, counter=counter, block_table=True)


@pytest.mark.parametrize("kind", ["row", "column", "rectangle"])
@pytest.mark.parametrize("rotating", [False, True])
def test_positional_channels_move_data(device, kind, rotating):
    run_positional_mcast_case(
        device, width=4, senders=[0, 2] if rotating else [1], rotating=rotating, kind=kind, block_table=True
    )
