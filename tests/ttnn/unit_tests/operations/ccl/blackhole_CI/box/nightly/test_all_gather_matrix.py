# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# all_gather functional matrix on a BH (1,8) box. Every run is bit-exact on every device.
#
# Each case runs under every algorithm column in ALGOS (auto plus the algorithms forced through
# TTNN_ALL_GATHER_FORCE) on a line and on a ring; uc_n1 / uc_n3 / composite run on one case per layout class
# (n differs only in the schedule, which the host gtests check). A forced algorithm the plan finds illegal must
# fail with the plan's reason: those (case, algorithm) pairs are listed in the case's `illegal`, and any other
# pair must run. Also here: FABRIC_2D on (2,4), NEIGHBOR_EXCHANGE, the program cache, and the watcher subset.
#
# Watcher subset: one case per algorithm x topology, run with the watcher and its NOC sanitizer (on by default);
# the ethernet part stays off because the fabric ERISC kernels overflow their config buffer with it:
#   TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 pytest <this file> -k watcher
# Without TT_METAL_WATCHER those tests skip (they repeat matrix cases).

import os

import pytest
import ttnn

from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.all_gather_test_utils import (
    BFP8,
    F2D,
    FP32,
    L1,
    LINE,
    NE,
    RING,
    RM,
    Case,
    device_input,
    dram_width_sharded,
    fabric_params,
    gather,
    check,
    grid,
    l1_height_sharded,
    l1_width_sharded,
    nd_sharded,
    run_once,
    set_force,
    submesh,
)

ALGOS = ("auto", "uc_n2", "mc")
EXTRA = ("uc_n1", "uc_n3", "composite")
# A forced native algorithm the shape rules out fails with the plan's reason (or, for gather-dim padding, which is
# checked before the plan, with that reason).
ILLEGAL = "is illegal here|forces a native algorithm"
TOPOLOGIES = {"line": LINE, "ring": RING}

# id -> (case, extra algorithms, algorithms the plan must refuse, topologies). Shapes are OUTPUT shapes.
_BOTH = ("line", "ring")
_LINE_ONLY = ("line",)  # submeshes of the (1,8) loop are lines
CASES = {
    # tile, DRAM interleaved: stripe of 10 tiles (relay + cadence), small, wide rows, narrow rows
    "tile_stripe10": (Case([1, 1, 1024, 2560], 3), EXTRA, (), _BOTH),
    "tile_small": (Case([1, 1, 64, 2560], 3), (), (), _BOTH),
    "tile_wide": (Case([1, 1, 256, 16384], 3), (), (), _BOTH),
    "tile_narrow": (Case([1, 1, 2048, 1280], 3), (), (), _BOTH),
    # dims -2 / -3 / -4
    "tile_dim2": (Case([1, 1, 2048, 256], 2), (), (), _BOTH),
    "tile_dim1": (Case([2, 8, 64, 128], 1), (), (), _BOTH),
    "tile_dim0": (Case([8, 1, 128, 1024], 0), (), (), _BOTH),
    # odd element count per device (11 tiles x 7 rows): slices start mid-row
    "mid_row_odd": (Case([1, 1, 224, 2816], 3), (), (), _BOTH),
    # one tile per device: most worker slices are empty
    "one_tile_per_device": (Case([1, 1, 32, 256], 3), (), (), _BOTH),
    "one_link": (Case([1, 1, 1024, 2560], 3, num_links=1), (), (), _BOTH),
    # other dtypes; bfp8 with one tile per stripe row (S = 1)
    "bfp8_tile": (Case([1, 1, 512, 4096], 3, dtype=BFP8), EXTRA, (), _BOTH),
    "bfp8_one_tile_row": (Case([1, 1, 256, 256], 3, dtype=BFP8), (), (), _BOTH),
    "fp32_tile": (Case([1, 1, 128, 1024], 3, dtype=FP32), (), (), _BOTH),
    # row-major: 8 KB pages; a chunk over one packet; 20000 B rows (> payload, % 64 = 32); 48 B rows; 384 KB pages
    "rm_8k_dim0": (Case([8, 1, 256, 4096], 0, layout=RM), EXTRA, (), _BOTH),
    "rm_chunk_over_packet": (Case([1, 1, 64, 4096], 2, layout=RM), (), (), _BOTH),
    "rm_20000B": (Case([1, 1, 512, 10000], 2, layout=RM), (), (), _BOTH),
    "rm_48B_rows": (Case([1, 1, 4096, 24], 2, layout=RM), (), (), _BOTH),
    "rm_big_page": (Case([1, 1, 8, 196608], 2, layout=RM), (), (), _BOTH),
    # L1 interleaved
    "l1_tile": (Case([1, 1, 256, 2048], 3, mem_in=L1, mem_out=L1), EXTRA, (), _BOTH),
    "rm_concat_l1": (Case([1, 1, 2048, 64], 3, layout=RM, mem_in=L1, mem_out=L1), (), (), _BOTH),
    "dram_to_l1": (Case([1, 1, 256, 256], 3, mem_out=L1), (), (), _BOTH),
    # sharded: s2s / s2i / i2s; height / width / ND
    "nd_dram": (
        Case([1, 1, 256, 256], 3, mem_in=nd_sharded(ttnn.BufferType.DRAM, [1, 1, 32, 32])),
        ("uc_n1", "uc_n3"),
        (),
        _BOTH,
    ),
    "nd_dram_s2s": (
        Case(
            [1, 1, 256, 256],
            3,
            mem_in=nd_sharded(ttnn.BufferType.DRAM, [1, 1, 32, 32]),
            mem_out=nd_sharded(ttnn.BufferType.DRAM, [1, 1, 32, 32]),
        ),
        (),
        (),
        _BOTH,
    ),
    "nd_l1_s2s": (
        Case(
            [1, 1, 128, 512],
            3,
            mem_in=nd_sharded(ttnn.BufferType.L1, [1, 1, 32, 64]),
            mem_out=nd_sharded(ttnn.BufferType.L1, [1, 1, 32, 64]),
        ),
        (),
        (),
        _BOTH,
    ),
    "hs_l1_to_dram": (Case([1, 1, 256, 256], 3, mem_in=l1_height_sharded(32, 32, 8)), (), (), _BOTH),
    "dram_to_l1_ws": (Case([1, 1, 64, 2048], 3, mem_out=l1_width_sharded(64, 256, 8)), (), (), _BOTH),
    # 128 B input rows into 64 B output pages (input pieces)
    "rm_split_ws": (
        Case(
            [1, 1, 2048, 64], 2, layout=RM, mem_in=l1_width_sharded(256, 64, 1), mem_out=l1_width_sharded(2048, 32, 2)
        ),
        (),
        (),
        _BOTH,
    ),
    "rm_one_page_shard": (
        Case([1, 1, 256, 512], 2, layout=RM, mem_in=l1_width_sharded(32, 64, 8), mem_out=l1_width_sharded(256, 64, 8)),
        (),
        (),
        _BOTH,
    ),
    # F1: 32 B rows, L1 (16 B read alignment) -> DRAM (64 B)
    "f1_l1_to_dram": (
        Case(
            [8, 1, 20480, 16],
            0,
            layout=RM,
            mem_in=l1_width_sharded(20480, 16, 1),
            mem_out=dram_width_sharded(163840, 16, 1),
        ),
        (),
        (),
        _BOTH,
    ),
    # 16 B L1 rows into 128 B DRAM rows: unicast reads its own output, so it is illegal; small and 8 MB
    "rm16B_l1_to_dram": (Case([1, 1, 64, 64], 3, layout=RM, mem_in=L1), (), ("uc_n2",), _BOTH),
    "rm16B_l1_to_dram_big": (Case([1, 1, 65536, 64], 3, layout=RM, mem_in=L1), (), ("uc_n2",), _BOTH),
    # composite by shape, with a sub_core_grid: a tile slice padded on the gather dim; 48 B RM rows DRAM -> L1 (the
    # 64 B aligned input page does not fit the 48 B output slot)
    "padded_tile": (Case([1, 1, 32, 40], 3, grid=grid(3, 3)), (), ("uc_n2", "mc"), _BOTH),
    "rm48_dram_to_l1": (Case([1, 1, 4096, 24], 2, layout=RM, mem_out=L1, grid=grid(3, 3)), (), ("uc_n2", "mc"), _BOTH),
    # sub_core_grid with only two NOC rows
    "two_row_grid": (Case([1, 1, 1024, 2560], 3, grid=grid(7, 1)), ("uc_n3",), (), _BOTH),
    # other device counts (submeshes are lines): odd N, line ends, N = 2
    "n5": (Case([1, 1, 256, 640], 3, mesh=(1, 5)), (), (), _LINE_ONLY),
    "n3": (Case([1, 1, 64, 384], 3, mesh=(1, 3)), (), (), _LINE_ONLY),
    "n2": (Case([1, 1, 64, 256], 3, mesh=(1, 2)), (), (), _LINE_ONLY),
    # one tile per device on 4 devices: one element, empty halves
    "n4_one_tile": (Case([1, 1, 32, 128], 3, mesh=(1, 4)), (), (), _BOTH),
    # 26 MB: deep pipelines, many cadence signals, relays under load
    "big_wide": (Case([1, 1, 800, 16384], 3), (), (), _BOTH),
    "big_narrow": (Case([1, 1, 13312, 1024], 3), (), (), _BOTH),
}

# Also run with a freshly allocated output (everything else writes a persistent output).
FRESH_TOO = ("tile_stripe10", "rm_20000B", "padded_tile", "nd_dram")


def _matrix_params():
    params = []
    for cid, (_, extra, _, topos) in CASES.items():
        for topo in topos:
            for algo in ALGOS + extra:
                params.append(pytest.param(fabric_params(TOPOLOGIES[topo]), cid, algo, True, id=f"{cid}-{topo}-{algo}"))
                if cid in FRESH_TOO:
                    params.append(
                        pytest.param(fabric_params(TOPOLOGIES[topo]), cid, algo, False, id=f"{cid}-{topo}-{algo}-fresh")
                    )
    return params


def _run(mesh_device, monkeypatch, expect_error, case, illegal, algo, persistent, label):
    set_force(monkeypatch, algo)
    if algo in illegal:
        mesh = submesh(mesh_device, case.mesh)
        tt_in, _ = device_input(mesh, case, seed=0)
        with expect_error(RuntimeError, ILLEGAL):
            gather(tt_in, case)
        return
    run_once(mesh_device, case, label, persistent)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params, case_id, algo, persistent", _matrix_params(), indirect=["device_params"])
def test_all_gather_matrix(mesh_device, device_params, case_id, algo, persistent, monkeypatch, expect_error):
    case, _, illegal, _ = CASES[case_id]
    _run(mesh_device, monkeypatch, expect_error, case, illegal, algo, persistent, f"{case_id} {algo}")


# FABRIC_2D on the box's system mesh shape (2,4): a 1D gather along the 4-device rows (cluster_axis 1), and the
# true 2D gather over all 8 devices.
CASES_2D = {
    "axis1": (Case([1, 1, 1024, 8192], 3, mesh=(2, 4), cluster_axis=1), ()),
    "axis1_small": (Case([1, 1, 32, 1024], 3, mesh=(2, 4), cluster_axis=1), ()),
    "true_2d": (Case([1, 1, 256, 256], 3, mesh=(2, 4)), ("uc_n2",)),  # unicast needs a 1D axis
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_params(F2D)], indirect=True, ids=["fabric_2d"])
@pytest.mark.parametrize("case_id", list(CASES_2D))
@pytest.mark.parametrize("algo", ALGOS)
def test_all_gather_matrix_2d(mesh_device, device_params, case_id, algo, monkeypatch, expect_error):
    case, illegal = CASES_2D[case_id]
    _run(mesh_device, monkeypatch, expect_error, case, illegal, algo, True, f"{case_id} {algo}")


# FABRIC_1D_NEIGHBOR_EXCHANGE: one hop only, so multicast is illegal and unicast relays hop by hop. N = 2 with
# 16 B L1 rows into DRAM rows (unicast illegal too) goes composite, whose all_broadcast is one hop there.
CASES_NE = {
    "tile_stripe10": (Case([1, 1, 1024, 2560], 3), ("mc",)),
    "rm_20000B": (Case([1, 1, 512, 10000], 2, layout=RM), ("mc",)),
    "big": (Case([1, 1, 4096, 3328], 3), ("mc",)),
    "n2_rm16B_composite": (Case([1, 1, 64, 16], 3, layout=RM, mem_in=L1, mesh=(1, 2)), ("uc_n2", "mc")),
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric_params(NE)], indirect=True, ids=["fabric_ne"])
@pytest.mark.parametrize("case_id", list(CASES_NE))
@pytest.mark.parametrize("algo", ALGOS)
def test_all_gather_matrix_ne(mesh_device, device_params, case_id, algo, monkeypatch, expect_error):
    case, illegal = CASES_NE[case_id]
    _run(mesh_device, monkeypatch, expect_error, case, illegal, algo, True, f"ne {case_id} {algo}")


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


# Program cache: equal calls on fresh tensors share one entry; a new shape or a changed override adds one (the
# override is part of the hashed params).
@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [fabric_params(LINE), fabric_params(RING)], indirect=True, ids=["line", "ring"]
)
def test_all_gather_program_cache(mesh_device, device_params, monkeypatch):
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

    set_force(monkeypatch, "uc_n2")
    added = [step(case, *inputs[k], f"call {k}") for k in range(3)]
    assert added == [1, 0, 0], f"three equal calls added {added} cache entries, expected [1, 0, 0]"
    assert step(other, *other_in, "new shape") == 1, "a new shape must add one cache entry"
    set_force(monkeypatch, "mc")
    assert step(case, *inputs[0], "changed force") == 1, "a changed override must add one cache entry"
    assert step(case, *inputs[1], "changed force, again") == 0


# Watcher subset (see the header): unicast and multicast on a line and a ring.
@skip_for_wormhole_b0()
@pytest.mark.skipif(not os.environ.get("TT_METAL_WATCHER"), reason="run with TT_METAL_WATCHER=1")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [fabric_params(LINE), fabric_params(RING)], indirect=True, ids=["line", "ring"]
)
@pytest.mark.parametrize("algo", ["uc_n2", "mc"])
def test_all_gather_watcher_subset(mesh_device, device_params, algo, monkeypatch):
    set_force(monkeypatch, algo)
    for cid in ("tile_stripe10", "rm_split_ws"):
        run_once(mesh_device, CASES[cid][0], f"watcher {cid} {algo}")
