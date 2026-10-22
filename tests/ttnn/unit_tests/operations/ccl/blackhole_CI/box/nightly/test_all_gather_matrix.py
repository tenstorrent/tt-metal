# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# all_gather functional matrix on a BH (1,8) box: every case compares bit-exactly on every device.
#
# Shapes are OUTPUT (gathered) shapes. Each case runs one "instance", which picks the algorithm through the
# fabric and the size alone (there is no algorithm override):
#   U-NE         FABRIC_1D_NEIGHBOR_EXCHANGE: unicast at any size, line
#   U-ring/line  1D ring/line: unicast. BH picks it on a ring below N x 96 KB per link, and above that on stripe
#                rows of >= 32 KB (ring) / >= 64 KB (line)
#   M-ring/line  1D ring/line: multicast (BH: narrow stripe rows above that size, or a small line)
#   2D           FABRIC_2D on (2,4): a 1D gather along the rows (HybridMesh headers), or true 2D
#   auto-*       whatever the plan picks (composite or a forced choice)
# Busiest-link bytes = gathered bytes / effective links (2 here, fewer when there are fewer elements); a stripe row
# is what one device contributes to one output row. The host test G7 (test_all_gather_plan.cpp) pins each
# instance's algorithm (4x clear of the size threshold, 2x of the stripe one), so a recalibrated threshold that
# flips one fails there; resize the case here when it does.

import pytest
import torch
import ttnn
from loguru import logger

from ttnn import ConcatMeshToTensor, ShardTensorToMesh
from models.common.utility_functions import skip_for_wormhole_b0
from tests.nightly.t3000.ccl.test_all_gather import (
    _dram_nd_sharded,
    _dram_width_sharded,
    _l1_width_sharded,
    create_fabric_router_config,
)

_PACKET = 15168
_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG
_TILE = ttnn.TILE_LAYOUT
_RM = ttnn.ROW_MAJOR_LAYOUT
_BF16 = ttnn.bfloat16
_BFP8 = ttnn.bfloat8_b


def _fabric(config, trace_region_size=None):
    params = {"fabric_config": config, "fabric_router_config": create_fabric_router_config(_PACKET)}
    if trace_region_size is not None:
        params["trace_region_size"] = trace_region_size
    return params


_NE = _fabric(ttnn.FabricConfig.FABRIC_1D_NEIGHBOR_EXCHANGE)
_RING = _fabric(ttnn.FabricConfig.FABRIC_1D_RING)
_LINE = _fabric(ttnn.FabricConfig.FABRIC_1D)
_F2D = _fabric(ttnn.FabricConfig.FABRIC_2D)
_INSTANCE_PARAMS = {
    "U-NE": _NE,
    "auto-NE": _NE,
    "U-ring": _RING,
    "U-line": _LINE,
    "M-ring": _RING,
    "M-line": _LINE,
    "auto-line": _LINE,
    "2D": _F2D,
}


def _height_sharded_l1(shard_height, shard_width, num_cores):
    grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _grid(x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})


def _case(shape, dim, layout, dtype, mem_in, mem_out, mesh=(1, 8), cluster_axis=None, tile=None, grid=None):
    return dict(
        shape=shape,
        dim=dim,
        layout=layout,
        dtype=dtype,
        mem_in=mem_in,
        mem_out=mem_out,
        mesh=mesh,
        cluster_axis=cluster_axis,
        tile=tile,
        grid=grid,
    )


# id -> (instance, case). The prefix is the plan's row (IMPL_PLAN §6b). Shapes match G7's table one for one; the
# few extra rows are marked and are U-NE or composite, whose algorithm depends on legality only, not on size.
CASES = {
    # D1 baseline: stripe 10, relay + data_valid cadence
    "D1-une": ("U-NE", _case([1, 1, 1024, 2560], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D1-u_ring": ("U-ring", _case([1, 1, 256, 8192], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D1-u_line": ("U-line", _case([1, 1, 256, 16384], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D1-m_ring": ("M-ring", _case([1, 1, 2048, 1280], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D1-m_line": ("M-line", _case([1, 1, 64, 2560], 3, _TILE, _BF16, _DRAM, _DRAM)),
    # D2 bfp8, one tile per stripe-row (S = 1): cross_row
    "D2-une": ("U-NE", _case([1, 1, 256, 256], 3, _TILE, _BFP8, _DRAM, _DRAM)),
    "D2-m_line": ("M-line", _case([1, 1, 256, 256], 3, _TILE, _BFP8, _DRAM, _DRAM)),
    # D3 RM sub-page and whole-page cuts: 8 KB rows; old known_bad chunk_over_packet; 20000 B rows (> P, % 64 = 32)
    "D3-8k_une": ("U-NE", _case([8, 1, 256, 4096], 0, _RM, _BF16, _DRAM, _DRAM)),
    "D3-8k_m_line": ("M-line", _case([8, 1, 4, 4096], 0, _RM, _BF16, _DRAM, _DRAM)),
    "D3-chunk_over_packet_une": ("U-NE", _case([1, 1, 64, 4096], 2, _RM, _BF16, _DRAM, _DRAM)),
    "D3-chunk_over_packet_m_line": ("M-line", _case([1, 1, 32, 4096], 2, _RM, _BF16, _DRAM, _DRAM)),
    "D3-20000B_une": ("U-NE", _case([1, 1, 512, 10000], 2, _RM, _BF16, _DRAM, _DRAM)),
    "D3-20000B_m_line": ("M-line", _case([1, 1, 16, 10000], 2, _RM, _BF16, _DRAM, _DRAM)),
    # D4 48 B DRAM rows: whole-page, slot pitch == bank pitch so runs merge
    "D4-une": ("U-NE", _case([1, 1, 4096, 24], 2, _RM, _BF16, _DRAM, _DRAM)),
    "D4-m_line": ("M-line", _case([1, 1, 4096, 24], 2, _RM, _BF16, _DRAM, _DRAM)),
    # D5 concat, E = 1 (16 B L1 rows into 128 B rows; 16 B DRAM rows would be composite)
    "D5-une": ("U-NE", _case([1, 1, 2048, 64], 3, _RM, _BF16, _L1, _L1)),
    "D5-m_line": ("M-line", _case([1, 1, 2048, 64], 3, _RM, _BF16, _L1, _L1)),
    # D6 split: 128 B input rows into 64 B output pages, so input pieces
    "D6-split_une": (
        "U-NE",
        _case([1, 1, 2048, 64], 2, _RM, _BF16, _l1_width_sharded(256, 64, 1), _l1_width_sharded(2048, 32, 2)),
    ),
    # D7 source stride != destination stride
    "D7-hs_l1_to_dram_une": ("U-NE", _case([1, 1, 256, 256], 3, _TILE, _BF16, _height_sharded_l1(32, 32, 8), _DRAM)),
    "D7-hs_l1_to_dram_m_line": (
        "M-line",
        _case([1, 1, 256, 256], 3, _TILE, _BF16, _height_sharded_l1(32, 32, 8), _DRAM),
    ),
    "D7-dram_to_l1_une": ("U-NE", _case([1, 1, 256, 256], 3, _TILE, _BF16, _DRAM, _L1)),
    "D7-dram_to_l1_m_line": ("M-line", _case([1, 1, 256, 256], 3, _TILE, _BF16, _DRAM, _L1)),
    # D8 ND-sharded DRAM in and out; (extra) a one-page-wide shard
    "D8-nd_dram_une": (
        "U-NE",
        _case([1, 1, 256, 256], 3, _TILE, _BF16, _dram_nd_sharded([1, 1, 32, 32]), _dram_nd_sharded([1, 1, 32, 32])),
    ),
    "D8-nd_dram_m_line": (
        "M-line",
        _case([1, 1, 256, 256], 3, _TILE, _BF16, _dram_nd_sharded([1, 1, 32, 32]), _dram_nd_sharded([1, 1, 32, 32])),
    ),
    "D8-one_page_shard_une": (
        "U-NE",
        _case([1, 1, 256, 512], 2, _RM, _BF16, _l1_width_sharded(32, 64, 8), _l1_width_sharded(256, 64, 8)),
    ),
    # D9 F1 repro (old known_bad mismatched_alignment): 32 B rows, L1 (16 B align) -> DRAM (64 B align).
    # Unicast is whole-page (A 64, slot 64 = bank pitch, 32 B reads); multicast is sub-page (A 16).
    "D9-f1_u_ring": (
        "U-ring",
        _case([8, 1, 20480, 16], 0, _RM, _BF16, _l1_width_sharded(20480, 16, 1), _dram_width_sharded(163840, 16, 1)),
    ),
    "D9-f1_m_line": (
        "M-line",
        _case([8, 1, 1024, 16], 0, _RM, _BF16, _l1_width_sharded(1024, 16, 1), _dram_width_sharded(8192, 16, 1)),
    ),
    # D9b 16 B L1 rows into 128 B DRAM rows: unicast is illegal, so multicast at any size (the big one would
    # prefer unicast by size)
    "D9b-small_auto_line": ("auto-line", _case([1, 1, 64, 64], 3, _RM, _BF16, _L1, _DRAM)),
    "D9b-big_auto_line": ("auto-line", _case([1, 1, 65536, 64], 3, _RM, _BF16, _L1, _DRAM)),
    # D11 odd N, line endpoints, N = 2.
    "D11-n5_une": ("U-NE", _case([1, 1, 256, 640], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(1, 5))),
    "D11-n5_m_line": ("M-line", _case([1, 1, 64, 640], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(1, 5))),
    "D11-n3_une": ("U-NE", _case([1, 1, 64, 384], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(1, 3))),
    "D11-n2_une": ("U-NE", _case([1, 1, 64, 256], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(1, 2))),
    # D12 one tile per device on a 4-device ring: one element, one link, empty forward half of the antipode stripe.
    # Only a ring if the (1,4) submesh wraps; otherwise the plan sees a line (and picks multicast, as on the BH box).
    "D12-u_ring4": ("U-ring", _case([1, 1, 32, 128], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(1, 4))),
    # One 384 KB row-major page per device on the 8-ring, sub-page.
    "D12-u_ring8_big_page": ("U-ring", _case([1, 1, 8, 196608], 2, _RM, _BF16, _DRAM, _DRAM)),
    # D14 composite (F3): gather-dim padded tile; 48 B RM rows DRAM -> L1 (slot pitch 64 > L1 bank pitch 48).
    # Persistent output and sub_core_grid both reach the composite path. (The plan's bfp8 16x32 tile case is
    # test_all_gather_tiny_tile_composite: composite's concat only knows 32x32 tiles, so it must refuse.)
    "D14-padded_auto_line": ("auto-line", _case([1, 1, 32, 40], 3, _TILE, _BF16, _DRAM, _DRAM, grid=_grid(3, 3))),
    "D14-rm48_dram_to_l1_auto_line": (
        "auto-line",
        _case([1, 1, 4096, 24], 2, _RM, _BF16, _DRAM, _L1, grid=_grid(3, 3)),
    ),
    # D15 sub_core_grid with only 2 NOC rows: unicast's 4 cores double up on rows
    "D15-une": ("U-NE", _case([1, 1, 1024, 2560], 3, _TILE, _BF16, _DRAM, _DRAM, grid=_grid(7, 1))),
    "D15-m_line": ("M-line", _case([1, 1, 64, 2560], 3, _TILE, _BF16, _DRAM, _DRAM, grid=_grid(7, 1))),
    # D16 FABRIC_2D on the box's system mesh shape (2,4) (other views of a 2D fabric can hang): a 1D gather along
    # the 4-device rows (HybridMesh headers), and the true 2D gather.
    "D16-axis_u": ("2D", _case([1, 1, 1024, 8192], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(2, 4), cluster_axis=1)),
    "D16-axis_m": ("2D", _case([1, 1, 32, 1024], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(2, 4), cluster_axis=1)),
    "D16-true_2d_m": ("2D", _case([1, 1, 256, 256], 3, _TILE, _BF16, _DRAM, _DRAM, mesh=(2, 4))),
    # D17 F2: D9b's shape where unicast is forced (NE) and illegal -> composite. (extra) N = 2, where all_broadcast
    # is one hop; N = 8 is test_all_gather_f2_neighbor_exchange_n8.
    "D17-n2_auto_ne": ("auto-NE", _case([1, 1, 64, 16], 3, _RM, _BF16, _L1, _DRAM, mesh=(1, 2))),
    # D18 26-27 MB: deep pipeline, trid wrap, relay gates under load
    "D18-une": ("U-NE", _case([1, 1, 4096, 3328], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D18-u_ring": ("U-ring", _case([1, 1, 800, 16384], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D18-u_line": ("U-line", _case([1, 1, 800, 16384], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D18-m_ring": ("M-ring", _case([1, 1, 13312, 1024], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "D18-m_line": ("M-line", _case([1, 1, 13312, 1024], 3, _TILE, _BF16, _DRAM, _DRAM)),
}


def _devices(mesh_device, mesh):
    if tuple(mesh_device.shape) == tuple(mesh):
        return mesh_device
    return mesh_device.create_submesh(ttnn.MeshShape(*mesh))


def _host_input(mesh, c, seed):
    torch.manual_seed(seed)
    full = torch.randn(c["shape"], dtype=torch.bfloat16)
    kwargs = {"tile": ttnn.Tile(c["tile"])} if c["tile"] is not None else {}
    host = ttnn.from_torch(full, layout=c["layout"], dtype=c["dtype"], mesh_mapper=_mapper(mesh, c), **kwargs)
    return host, full


def _mapper(mesh, c):
    if c["cluster_axis"] is None:
        return ShardTensorToMesh(mesh, dim=c["dim"])
    dims = (None, c["dim"]) if c["cluster_axis"] == 1 else (c["dim"], None)
    return ttnn.ShardTensor2dMesh(mesh, dims=dims, mesh_shape=tuple(mesh.shape))


def _input(mesh, c, seed):
    host, full = _host_input(mesh, c, seed)
    tt_in = ttnn.to_device(host, mesh, memory_config=c["mem_in"])
    if c["dtype"] == _BF16:
        return tt_in, full
    # Lossy dtype: the gather must reproduce exactly what the devices hold.
    assert c["cluster_axis"] is None
    return tt_in, ttnn.to_torch(tt_in, mesh_composer=ConcatMeshToTensor(mesh, dim=c["dim"]))


def _output(mesh, c):
    # A sentinel, so a region the gather never writes cannot pass.
    kwargs = {"tile": ttnn.Tile(c["tile"])} if c["tile"] is not None else {}
    return ttnn.from_torch(
        torch.full(c["shape"], -3.0, dtype=torch.bfloat16),
        device=mesh,
        layout=c["layout"],
        dtype=c["dtype"],
        memory_config=c["mem_out"],
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        **kwargs,
    )


def _check(tt_out, expected, label):
    for d, dev_out in enumerate(ttnn.get_device_tensors(tt_out)):
        got = ttnn.to_torch(dev_out)
        ok = got.shape == expected.shape and torch.equal(got, expected)
        if not ok and got.shape == expected.shape:
            bad = (got != expected).nonzero()
            logger.error(f"{label}: device {d}: {bad.shape[0]} mismatches, first at {bad[0].tolist()}")
        assert ok, f"{label}: device {d} mismatch"


def _gather(tt_in, c, out):
    return ttnn.all_gather(
        tt_in,
        dim=c["dim"],
        cluster_axis=c["cluster_axis"],
        memory_config=c["mem_out"],
        output_tensor=out,
        sub_core_grids=c["grid"],
    )


# Every case writes into a persistent output; these also run with a freshly allocated one.
_FRESH_TOO = {"D1-une", "D1-m_line", "D3-20000B_une", "D14-padded_auto_line", "D16-true_2d_m"}


def _params(two_d):
    params = []
    for cid, (inst, c) in CASES.items():
        if (c["mesh"] == (2, 4)) != two_d:
            continue
        params.append(pytest.param(_INSTANCE_PARAMS[inst], cid, True, id=f"{cid}-persistent"))
        if cid in _FRESH_TOO:
            params.append(pytest.param(_INSTANCE_PARAMS[inst], cid, False, id=f"{cid}-fresh"))
    return params


def _run_case(mesh_device, case_id, persistent):
    c = CASES[case_id][1]
    mesh = _devices(mesh_device, c["mesh"])
    tt_in, expected = _input(mesh, c, seed=0)
    out = _output(mesh, c) if persistent else None
    tt_out = _gather(tt_in, c, out)
    ttnn.synchronize_device(mesh)
    _check(tt_out, expected, case_id)
    if out is not None:
        _check(out, expected, f"{case_id} (persistent output)")


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params, case_id, persistent", _params(two_d=False), indirect=["device_params"])
def test_all_gather_matrix(mesh_device, device_params, case_id, persistent):
    _run_case(mesh_device, case_id, persistent)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params, case_id, persistent", _params(two_d=True), indirect=["device_params"])
def test_all_gather_matrix_2d(mesh_device, device_params, case_id, persistent):
    _run_case(mesh_device, case_id, persistent)


# D17 at N = 8: the composite fallback's all_broadcast needs multi-hop multicast, which NE does not provide, so the
# op must fail with that reason (not hang, and not a factory FATAL).
@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [_NE], indirect=True, ids=["fabric_ne"])
def test_all_gather_f2_neighbor_exchange_n8(mesh_device, device_params, expect_error):
    c = _case([1, 1, 64, 64], 3, _RM, _BF16, _L1, _DRAM)
    tt_in, _ = _input(mesh_device, c, seed=0)
    with expect_error(RuntimeError, "NEIGHBOR_EXCHANGE"):
        _gather(tt_in, c, None)
        ttnn.synchronize_device(mesh_device)


# D14 (plan): bfp8 16x32 tiles DRAM -> L1 cannot go native (slot pitch 576 > L1 bank pitch 544), and the composite
# path's concat only supports 32x32 tiles, so the op must refuse instead of returning scrambled data.
@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [_LINE], indirect=True, ids=["fabric_line"])
def test_all_gather_tiny_tile_composite(mesh_device, device_params, expect_error):
    c = _case([1, 1, 64, 256], 3, _TILE, _BFP8, _DRAM, _L1, tile=(16, 32))
    tt_in, _ = _input(mesh_device, c, seed=0)
    with expect_error(RuntimeError, "32x32 tiles"):
        _gather(tt_in, c, None)
        ttnn.synchronize_device(mesh_device)


# D13 (B1/B2): 50 all_gathers captured back to back in one trace, each with its own input and persistent output,
# replayed twice with fresh inputs; every gather of every replay is checked. Unicast on the even ring, multicast on
# the line.
_STRESS_RUNS = 50
_STRESS = {
    "u_ring": (ttnn.FabricConfig.FABRIC_1D_RING, _case([1, 1, 256, 8192], 3, _TILE, _BF16, _DRAM, _DRAM)),
    "m_line": (ttnn.FabricConfig.FABRIC_1D, _case([1, 1, 64, 2560], 3, _TILE, _BF16, _DRAM, _DRAM)),
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params, stress_id",
    [pytest.param(_fabric(cfg, 8_000_000), sid, id=sid) for sid, (cfg, _) in _STRESS.items()],
    indirect=["device_params"],
)
def test_all_gather_traced_stress(mesh_device, device_params, stress_id):
    c = _STRESS[stress_id][1]
    ins = [_input(mesh_device, c, seed=k)[0] for k in range(_STRESS_RUNS)]
    outs = [_output(mesh_device, c) for _ in range(_STRESS_RUNS)]

    _gather(ins[0], c, outs[0])  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for k in range(_STRESS_RUNS):
        _gather(ins[k], c, outs[k])
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    for replay in range(2):
        expected = []
        for k in range(_STRESS_RUNS):
            host_in, full = _host_input(mesh_device, c, seed=1000 * (replay + 1) + k)
            ttnn.copy_host_to_device_tensor(host_in, ins[k])
            expected.append(full)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        for k in range(_STRESS_RUNS):
            _check(outs[k], expected[k], f"{stress_id} replay {replay} run {k}")
    ttnn.release_trace(mesh_device, trace_id)
