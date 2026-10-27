# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# all_gather functional matrix on a BH (1,8) box. Every run is bit-exact on every device.
#
# Each case runs under several levers (all_gather_test_utils.LEVERS), which reach the algorithms through public
# arguments only: auto on a line and on a ring, a one-column sub_core_grid (multicast at any size), and a
# neighbor-exchange fabric (unicast at any size). The host gtest AllGatherPlan.DeviceTestAlgorithms pins the
# algorithm, worker count and link count of every (case, lever) here, so keep the two tables in step. Also here:
# FABRIC_2D on (2,4), the program cache, and the watcher subset.
#
# Watcher subset: unicast and multicast on a line and a ring, run with the watcher and its NOC sanitizer (on by default);
# the ethernet part stays off because the fabric ERISC kernels overflow their config buffer with it:
#   TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 pytest <this file> -k watcher
# Without TT_METAL_WATCHER those tests skip (they repeat matrix cases).

import os

import pytest
import ttnn

from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.all_gather_test_utils import (
    BFP8,
    COLUMN,
    F2D,
    FP32,
    L1,
    LEVERS,
    LINE,
    NE,
    RING,
    RM,
    Case,
    check,
    device_input,
    dram_width_sharded,
    fabric_params,
    gather,
    grid,
    l1_height_sharded,
    l1_width_sharded,
    nd_sharded,
    run_once,
    with_lever,
)

# Lever sets. Submeshes of the (1,8) loop are lines. Composite-by-shape cases run auto only: no lever changes them.
ALL = ("line", "ring", "line_mc", "ring_mc", "ne")
LINE_ONLY = ("line", "line_mc", "ne")
COMPOSITE = ("line", "ring")
# One case per layout class also runs these.
EXTRA = ("ne_n1", "ne_1link", "line_mc_1link", "ring_mc_1link")

# id -> (case, levers). Shapes are OUTPUT shapes.
CASES = {
    # tile, DRAM interleaved: stripe of 10 tiles (relay + cadence), small, wide rows, narrow rows
    "tile_stripe10": (Case([1, 1, 1024, 2560], 3), ALL + EXTRA),
    "tile_small": (Case([1, 1, 64, 2560], 3), ALL),
    "tile_wide": (Case([1, 1, 256, 16384], 3), ALL),
    "tile_narrow": (Case([1, 1, 2048, 1280], 3), ALL),
    # dims -2 / -3 / -4
    "tile_dim2": (Case([1, 1, 2048, 256], 2), ALL),
    "tile_dim1": (Case([2, 8, 64, 128], 1), ALL),
    "tile_dim0": (Case([8, 1, 128, 1024], 0), ALL),
    # odd element count per device (11 tiles x 7 rows): slices start mid-row
    "mid_row_odd": (Case([1, 1, 224, 2816], 3), ALL),
    # one tile per device: most worker slices are empty
    "one_tile_per_device": (Case([1, 1, 32, 256], 3), ALL + EXTRA),
    # other dtypes; bfp8 with one tile per stripe row (S = 1)
    "bfp8_tile": (Case([1, 1, 512, 4096], 3, dtype=BFP8), ALL + EXTRA),
    "bfp8_one_tile_row": (Case([1, 1, 256, 256], 3, dtype=BFP8), ALL),
    "fp32_tile": (Case([1, 1, 128, 1024], 3, dtype=FP32), ALL),
    # row-major: 8 KB pages; a chunk over one packet; 20000 B rows (> payload, % 64 = 32); 48 B rows; 384 KB pages
    "rm_8k_dim0": (Case([8, 1, 256, 4096], 0, layout=RM), ALL + EXTRA),
    "rm_chunk_over_packet": (Case([1, 1, 64, 4096], 2, layout=RM), ALL),
    "rm_20000B": (Case([1, 1, 512, 10000], 2, layout=RM), ALL),
    "rm_48B_rows": (Case([1, 1, 4096, 24], 2, layout=RM), ALL),
    "rm_big_page": (Case([1, 1, 8, 196608], 2, layout=RM), ALL),
    # 256 B per device per row: about 30 short packets per slot
    "rm_rowseg256": (Case([1, 1, 2048, 1024], 3, layout=RM), ALL),
    # L1 interleaved
    "l1_tile": (Case([1, 1, 256, 2048], 3, mem_in=L1, mem_out=L1), ALL + EXTRA),
    "rm_concat_l1": (Case([1, 1, 2048, 64], 3, layout=RM, mem_in=L1, mem_out=L1), ALL),
    "dram_to_l1": (Case([1, 1, 256, 256], 3, mem_out=L1), ALL),
    # sharded: s2s / s2i / i2s; height / width / ND
    "nd_dram": (Case([1, 1, 256, 256], 3, mem_in=nd_sharded(ttnn.BufferType.DRAM, [1, 1, 32, 32])), ALL + EXTRA),
    "nd_dram_s2s": (
        Case(
            [1, 1, 256, 256],
            3,
            mem_in=nd_sharded(ttnn.BufferType.DRAM, [1, 1, 32, 32]),
            mem_out=nd_sharded(ttnn.BufferType.DRAM, [1, 1, 32, 32]),
        ),
        ALL,
    ),
    "nd_l1_s2s": (
        Case(
            [1, 1, 128, 512],
            3,
            mem_in=nd_sharded(ttnn.BufferType.L1, [1, 1, 32, 64]),
            mem_out=nd_sharded(ttnn.BufferType.L1, [1, 1, 32, 64]),
        ),
        ALL,
    ),
    "hs_l1_to_dram": (Case([1, 1, 256, 256], 3, mem_in=l1_height_sharded(32, 32, 8)), ALL),
    "dram_to_l1_ws": (Case([1, 1, 64, 2048], 3, mem_out=l1_width_sharded(64, 256, 8)), ALL),
    # 128 B input rows into 64 B output pages (input pieces)
    "rm_split_ws": (
        Case(
            [1, 1, 2048, 64], 2, layout=RM, mem_in=l1_width_sharded(256, 64, 1), mem_out=l1_width_sharded(2048, 32, 2)
        ),
        ALL,
    ),
    "rm_one_page_shard": (
        Case([1, 1, 256, 512], 2, layout=RM, mem_in=l1_width_sharded(32, 64, 8), mem_out=l1_width_sharded(256, 64, 8)),
        ALL,
    ),
    # 32 B rows, L1 (16 B read alignment) -> DRAM (64 B)
    "rm32B_l1_to_dram": (
        Case(
            [8, 1, 20480, 16],
            0,
            layout=RM,
            mem_in=l1_width_sharded(20480, 16, 1),
            mem_out=dram_width_sharded(163840, 16, 1),
        ),
        ALL,
    ),
    # 16 B L1 rows into 128 B DRAM rows: unicast would read its own output at 16 B, so it is illegal and every lever
    # runs multicast, except neighbor exchange (no multicast either): composite on 2 devices only (below)
    "rm16B_l1_to_dram": (Case([1, 1, 64, 64], 3, layout=RM, mem_in=L1), ("line", "ring", "line_mc", "ring_mc")),
    "rm16B_l1_to_dram_big": (Case([1, 1, 65536, 64], 3, layout=RM, mem_in=L1), ("line", "ring")),
    # composite by shape, with a sub_core_grid: a tile slice padded on the gather dim; 48 B RM rows DRAM -> L1 (the
    # 64 B aligned input page does not fit the 48 B output slot)
    "padded_tile": (Case([1, 1, 32, 40], 3, grid=grid(3, 3)), COMPOSITE),
    "rm48_dram_to_l1": (Case([1, 1, 4096, 24], 2, layout=RM, mem_out=L1, grid=grid(3, 3)), COMPOSITE),
    # sub_core_grid with only two NOC rows: the groups of both links double up on them
    "two_row_grid": (Case([1, 1, 1024, 2560], 3, grid=grid(7, 1)), ("line", "ring")),
    # other device counts (submeshes are lines): odd N, line ends, N = 2
    "n5": (Case([1, 1, 256, 640], 3, mesh=(1, 5)), LINE_ONLY),
    "n3": (Case([1, 1, 64, 384], 3, mesh=(1, 3)), LINE_ONLY),
    "n2": (Case([1, 1, 64, 256], 3, mesh=(1, 2)), LINE_ONLY),
    "n2_rm16B_composite": (Case([1, 1, 64, 16], 3, layout=RM, mem_in=L1, mesh=(1, 2)), ("ne",)),
    # one tile per device on 4 devices: one element, empty halves
    "n4_one_tile": (Case([1, 1, 32, 128], 3, mesh=(1, 4)), LINE_ONLY),
    # 26 MB: deep pipelines, many cadence signals, relays under load
    "big_wide": (Case([1, 1, 800, 16384], 3), ALL),
    "big_narrow": (Case([1, 1, 13312, 1024], 3), ALL),
}

# Also run with a freshly allocated output (everything else writes a persistent output).
FRESH_TOO = ("tile_stripe10", "rm_20000B", "padded_tile", "nd_dram")


def _matrix_params():
    params = []
    for cid, (_, levers) in CASES.items():
        for lever in levers:
            fabric = fabric_params(LEVERS[lever][0])
            params.append(pytest.param(fabric, cid, lever, True, id=f"{cid}-{lever}"))
            if cid in FRESH_TOO:
                params.append(pytest.param(fabric, cid, lever, False, id=f"{cid}-{lever}-fresh"))
    # Group the runs by fabric config, so consecutive tests share it.
    order = {LINE: 0, RING: 1, NE: 2}
    return sorted(params, key=lambda p: order[p.values[0]["fabric_config"]])


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params, case_id, lever, persistent", _matrix_params(), indirect=["device_params"])
def test_all_gather_matrix(mesh_device, device_params, case_id, lever, persistent):
    case, _ = CASES[case_id]
    run_once(mesh_device, with_lever(case, lever), f"{case_id} {lever}", persistent)


# FABRIC_2D on the box's system mesh shape (2,4): a 1D gather along the 4-device rows (cluster_axis 1), and the
# true 2D gather over all 8 devices, which only multicast serves.
CASES_2D = {
    "axis1": Case([1, 1, 1024, 8192], 3, mesh=(2, 4), cluster_axis=1),
    "axis1_small": Case([1, 1, 32, 1024], 3, mesh=(2, 4), cluster_axis=1),
    "true_2d": Case([1, 1, 256, 256], 3, mesh=(2, 4)),
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_params(F2D)], indirect=True, ids=["fabric_2d"])
@pytest.mark.parametrize("case_id", list(CASES_2D))
@pytest.mark.parametrize("grid_id", ["auto", "mc"])
def test_all_gather_matrix_2d(mesh_device, device_params, case_id, grid_id):
    case = CASES_2D[case_id]
    if grid_id == "mc":
        case = Case(case.shape, case.dim, mesh=case.mesh, cluster_axis=case.cluster_axis, grid=COLUMN)
    run_once(mesh_device, case, f"{case_id} {grid_id}")


# At N = 8 neither native algorithm is legal here, and the composite fallback's all_broadcast needs multi-hop routes,
# which NEIGHBOR_EXCHANGE lacks: the plan must refuse before launch, not hang.
@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_params(NE)], indirect=True, ids=["fabric_ne"])
def test_all_gather_ne_composite_n8_refused(mesh_device, device_params, expect_error):
    case = Case([1, 1, 64, 64], 3, layout=RM, mem_in=L1)
    tt_in, _ = device_input(mesh_device, case, seed=0)
    with expect_error(RuntimeError, "neighbor-exchange fabric"):
        gather(tt_in, case)
        ttnn.synchronize_device(mesh_device)


# Program cache: equal calls on fresh tensors share one entry; a new shape or a sub_core_grid that changes the
# algorithm (unicast -> multicast) or the worker count adds one.
@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_params(NE)], indirect=True, ids=["ne"])
def test_all_gather_program_cache(mesh_device, device_params):
    case = Case([1, 1, 256, 2560], 3)
    other = Case([1, 1, 128, 1280], 3)
    inputs = [device_input(mesh_device, case, seed=k) for k in range(3)]
    other_in = device_input(mesh_device, other, seed=7)

    def step(c, tt_in, expected, label):
        before = mesh_device.num_program_cache_entries()
        out = gather(tt_in, c)
        ttnn.synchronize_device(mesh_device)
        added = mesh_device.num_program_cache_entries() - before
        check(out, expected, label)
        return added

    added = [step(case, *inputs[k], f"call {k}") for k in range(3)]
    assert added == [1, 0, 0], f"three equal calls added {added} cache entries, expected [1, 0, 0]"
    assert step(other, *other_in, "new shape") == 1, "a new shape must add one cache entry"
    one_worker = with_lever(case, "ne_n1")
    assert step(one_worker, *inputs[0], "one worker") == 1, "a new worker count must add one cache entry"
    assert step(one_worker, *inputs[1], "one worker, again") == 0


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_params(LINE)], indirect=True, ids=["line"])
def test_all_gather_program_cache_algorithm(mesh_device, device_params):
    case = Case([1, 1, 4096, 2560], 3)  # 20 MB: auto picks unicast on a line
    tt_in, expected = device_input(mesh_device, case, seed=0)
    for c, label, want in ((case, "auto", 1), (with_lever(case, "line_mc"), "multicast", 1), (case, "auto again", 0)):
        before = mesh_device.num_program_cache_entries()
        out = gather(tt_in, c)
        ttnn.synchronize_device(mesh_device)
        check(out, expected, label)
        added = mesh_device.num_program_cache_entries() - before
        assert added == want, f"{label}: added {added} cache entries, expected {want}"


# Watcher subset (see the header): unicast (line, ring auto on tile_stripe10) and multicast (one-column grid), each
# also on a sharded row-major split.
@skip_for_wormhole_b0()
@pytest.mark.skipif(not os.environ.get("TT_METAL_WATCHER"), reason="run with TT_METAL_WATCHER=1")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params, lever",
    [(fabric_params(LEVERS[lever][0]), lever) for lever in ("line", "ring", "line_mc", "ring_mc")],
    indirect=["device_params"],
    ids=["line", "ring", "line_mc", "ring_mc"],
)
def test_all_gather_watcher_subset(mesh_device, device_params, lever):
    for cid in ("tile_stripe10", "rm_split_ws"):
        run_once(mesh_device, with_lever(CASES[cid][0], lever), f"watcher {cid} {lever}")
