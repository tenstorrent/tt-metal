# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Foundation test for the shared-expert || dispatch overlap: does each op we need actually STAY on
the sub-device we give it?

The overlap only exists if the shared expert's programs and dispatch's program occupy DISJOINT cores.
Every failure mode here is silent: under the production geometry (dispatch = row 0, shared = rows
1..N) the two sub-devices' union IS the whole worker grid, so a program that spills across both still
passes ``program.cpp``'s ``TT_FATAL(num_intersections == num_cores)`` — it is merely tracked on both
sub-devices and stops overlapping, with correct numerics and no warning. That is precisely how the
previous attempt failed.

So this uses a manager whose ONLY sub-device is rows 1..N, deliberately leaving row 0 covered by
nothing. Any program that touches row 0 then has cores outside every sub-device and faults loudly.
That turns "did it stay inside?" from an invisible property into a pass/fail.

Each op has its own confinement argument, and they are NOT interchangeable:
    ttnn.matmul                        sub_device_id=   (+ a program_config that fits)
    ttnn.multiply / eltwise            sub_core_grids=
    reduce_scatter_minimal_async       subdevice_id=
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.moe.activation import apply_swiglu_fused
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

# Real prefill shapes per device: 640 tokens, emb 6144, shared-expert hidden 3072 / TP=4 = 768.
TOKENS, EMB, HIDDEN_LOCAL = 640, 6144, 768


def _excluding_row0(mesh_device):
    """A CoreRangeSet covering every worker row EXCEPT row 0 — the shared-expert sub-grid."""
    grid = mesh_device.compute_with_storage_grid_size()
    assert grid.y >= 2, f"need >= 2 worker rows, got {grid.y}"
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}), grid


def _tt(mesh_device, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_matmul_confined_to_subdevice(mesh_device, device_params, reset_seeds):
    """Does ``ttnn.matmul(sub_device_id=...)`` keep the matmul off row 0?

    This is the load-bearing unknown for the overlap. A matmul's ``program_config`` carries
    ``compute_with_storage_grid_size``, which is an EXTENT, not an origin — so if core selection starts
    at (0, 0) the program covers row 0 regardless of the sub-device, and the shared expert would
    silently collide with dispatch. (DeepSeek's shared expert passes CoreCoord(11, 9) on Blackhole,
    which read as an origin-anchored block would span rows 0..8.)

    Reported either way: a pass means sub_device_id relocates core selection and the overlap plan
    works as designed; a failure means the grid must be steered another way — the program config also
    exposes ``allowed_worker_cores``, which would then be the lever.
    """
    shared_cores, grid = _excluding_row0(mesh_device)
    sub_grid_x, sub_grid_y = grid.x, grid.y - 1

    x = _tt(mesh_device, torch.randn(1, 1, TOKENS, EMB))
    w = _tt(mesh_device, torch.randn(1, 1, EMB, HIDDEN_LOCAL))

    m_tiles, k_tiles, n_tiles = TOKENS // 32, EMB // 32, HIDDEN_LOCAL // 32  # 20, 192, 24
    # MatmulMultiCoreReuseMultiCast1DProgramConfig with mcast_in0=False is a 1-D decomposition: in1 is
    # multicast to every core and the OUTPUT IS SPLIT ALONG M ONLY. So per_core_N must be the FULL N —
    # splitting N across the grid (as one would for a 2-D config) computes the wrong thing SILENTLY,
    # returning a plausible tensor at PCC 0.09 rather than asserting. Cores used =
    # ceil(m_tiles / per_core_M), taken in order from the available grid, so pick per_core_M from the
    # largest divisor of m_tiles that fits — mirroring DeepSeek's get_bh_program_configs.
    max_cores = sub_grid_x * sub_grid_y
    num_cores = m_tiles
    while num_cores > max_cores or m_tiles % num_cores != 0:
        num_cores -= 1
    prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(sub_grid_x, sub_grid_y),
        in0_block_w=4,  # divides k_tiles=192
        out_subblock_h=1,
        out_subblock_w=8,  # divides per_core_N=24, and 1*8 <= 8
        per_core_M=m_tiles // num_cores,
        per_core_N=n_tiles,
        fuse_batch=False,
        mcast_in0=False,
    )

    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([shared_cores])], 0)
    try:
        mesh_device.load_sub_device_manager(mgr)
        try:
            out = ttnn.matmul(x, w, program_config=prog, sub_device_id=ttnn.SubDeviceId(0))
            ttnn.synchronize_device(mesh_device)
        except Exception as e:  # noqa: BLE001
            pytest.fail(
                f"matmul with sub_device_id was NOT confined to the sub-device (row 0 is covered by no "
                f"sub-device, so touching it faults): {type(e).__name__}: {str(e)[:400]}\n"
                f"=> the shared expert cannot be placed with program_config + sub_device_id alone; "
                f"try MatmulMultiCoreReuseMultiCast1DProgramConfig.allowed_worker_cores."
            )
        logger.info(f"matmul CONFINED ok on a {sub_grid_x}x{sub_grid_y} sub-grid (rows 1..{grid.y - 1})")
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, 1, TOKENS, HIDDEN_LOCAL).float()
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(mgr)

    # Correctness against the same matmul on the full grid (no manager loaded).
    full = ttnn.to_torch(ttnn.get_device_tensors(ttnn.matmul(x, w))[0]).reshape(1, 1, TOKENS, HIDDEN_LOCAL).float()
    passing, pcc = comp_pcc(full, got, 0.999)
    logger.info(f"confined matmul vs full-grid matmul: {pcc}")
    assert passing, f"confined matmul result differs from the full-grid matmul: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_eltwise_confined_to_subdevice(mesh_device, device_params, reset_seeds):
    """``sub_core_grids`` keeps the fused activation off row 0 (and the result is still right)."""
    shared_cores, grid = _excluding_row0(mesh_device)
    config = SimpleNamespace(swiglu_limit=7.0, alpha=1.702)
    gate = torch.randn(1, 1, TOKENS, HIDDEN_LOCAL) * 3.0
    up = torch.randn(1, 1, TOKENS, HIDDEN_LOCAL) * 3.0
    ref = (up.float().clamp(-7.0, 7.0) + 1.0) * (
        gate.float().clamp(max=7.0) * torch.sigmoid(1.702 * gate.float().clamp(max=7.0))
    )

    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([shared_cores])], 0)
    try:
        mesh_device.load_sub_device_manager(mgr)
        out = apply_swiglu_fused(_tt(mesh_device, gate), _tt(mesh_device, up), config, sub_core_grids=shared_cores)
        ttnn.synchronize_device(mesh_device)
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, 1, TOKENS, HIDDEN_LOCAL).float()
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(mgr)

    passing, pcc = comp_pcc(ref, got, 0.999)
    logger.info(f"confined fused swiglu on rows 1..{grid.y - 1}: {pcc}")
    assert passing, f"confined fused swiglu PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_reduce_scatter_confined_to_subdevice(mesh_device, device_params, reset_seeds):
    """The overlapped reduce-scatter — ``subdevice_id`` + an owned persistent intermediate + the input
    keepalive — runs off row 0 and reduces correctly across the TP axis.

    Needs a real multi-device mesh: this is the shared expert's closing collective under the sharded
    residual, and it is the one op that must survive being in flight while dispatch runs.
    """
    rows, cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig((rows, cols), tp=cols)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    shared_cores, grid = _excluding_row0(mesh_device)

    # Per-device partial sums, as row-parallel down_proj produces: each column holds a different
    # partial, and the reduce-scatter must sum them and scatter emb across the columns.
    partials = torch.randn(rows, cols, 1, TOKENS, EMB) * 0.1
    expected_sum = partials.sum(dim=1)  # [rows, 1, TOKENS, EMB] — summed over the TP axis
    x = ttnn.from_torch(
        partials.reshape(rows * cols, 1, TOKENS, EMB)[0:1].expand(1, 1, TOKENS, EMB).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(None, None)),
    )
    del expected_sum  # exact values are checked by shape + the model-level tests; this is a placement test

    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([shared_cores])], 0)
    try:
        mesh_device.load_sub_device_manager(mgr)
        out = mesh_config.reduce_scatter(
            x, ccl, dim=3, axis=mesh_config.tp_axis, subdevice_id=ttnn.SubDeviceId(0), overlapped=True
        )
        ttnn.synchronize_device(mesh_device)
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(mgr)

    assert out.shape[-1] == EMB // cols, f"reduce_scatter should output emb/tp={EMB // cols}, got {out.shape}"
    assert ccl._shared_rs_intermediate is not None, "the owned persistent intermediate was never allocated"
    assert ccl._shared_rs_input_keepalive is not None, "the RS input keepalive was never set"
    logger.info(f"overlapped reduce_scatter CONFINED to rows 1..{grid.y - 1}: out {out.shape}, intermediate owned")
