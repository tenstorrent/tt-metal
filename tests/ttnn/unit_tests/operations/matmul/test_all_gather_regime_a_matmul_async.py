# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 2 tests for the fused all-gather + regime_a_matmul op (REGIME_A_AGMM_EXECUTION_PLAN.md):
#  - Offline host-plan tests (device-free) over D=1/2/4/8: global-K-block ownership/coverage, core
#    reservation fit + collision, and L1 sizing.
#  - D=1 correctness: the op is behaviorally identical to regime_a_matmul.

import gc

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


def _plan(**kw):
    # Device-free host-plan preview (registered on the C++ experimental submodule).
    return ttnn._ttnn.operations.experimental.all_gather_regime_a_matmul_plan(**kw)


# ---------------------------------------------------------------- offline plan tests (no device)
@pytest.mark.parametrize("D", [1, 2, 4, 8])
@pytest.mark.parametrize("topology", ["ring", "linear"])
def test_plan_kblock_coverage_and_fit(D, topology):
    # K=6144 -> Kt=192; kb=2 -> global_k_blocks=96, divisible by all D in {1,2,4,8}.
    p = _plan(M=32, K=6144, N=3072, D=D, Ns=1, Pk=3, Sm=1, kb=2, nsb=6,
              topology=topology, num_links=2, num_workers_per_link=6)
    assert p["valid"], p["errors"]
    assert p["global_k_blocks"] == 96
    # Every global K-block owned by exactly one device (explicit ids, full coverage, no dupes/gaps).
    seen = [b for dev in p["devices"] for b in dev["local_k_blocks"]]
    assert sorted(seen) == list(range(p["global_k_blocks"]))
    assert len(seen) == len(set(seen))
    assert len(p["devices"]) == D
    assert p["core_fit"] and not p["core_collision"]
    assert p["l1_fit"]
    # D=1 reserves no fabric cores; D>1 reserves mux + workers.
    if D == 1:
        assert p["reserved_fabric_cores"] == 0
    else:
        exp_mux = 2 * (2 if topology == "ring" else 1)  # num_links * (ring?2:1)
        assert p["mux_cores"] == exp_mux
        assert p["fabric_worker_cores"] == 2 * 6
        assert p["reserved_fabric_cores"] == exp_mux + 12


def test_plan_invalid_kt_not_divisible_by_D():
    p = _plan(M=32, K=6144, N=3072, D=5, kb=2)  # Kt=192 not divisible by 5
    assert not p["valid"]
    assert any("divisible by D" in e for e in p["errors"])


def test_plan_invalid_kt_not_divisible_by_kb():
    p = _plan(M=32, K=6144, N=3072, D=2, kb=5)  # Kt=192 not divisible by 5
    assert not p["valid"]
    assert any("kb" in e for e in p["errors"])


def test_plan_core_oversubscription_detected():
    # 8*Ns*Pk*Sm = 8*4*6*2 = 384 compute cores > usable 120.
    p = _plan(M=32, K=6144, N=3072, D=4, Ns=4, Pk=6, Sm=2, kb=2, num_links=2, num_workers_per_link=6)
    assert not p["core_fit"] or p["core_collision"]
    assert not p["valid"]


def test_plan_l1_oversubscription_detected():
    # Force a large transport footprint: big Mt (M=8192 -> Mt=256) * kb * C * slots.
    p = _plan(M=8192, K=6144, N=3072, D=2, Ns=1, Pk=1, Sm=1, kb=8, nsb=0,
              C=8, transport_slots=4, num_links=2, num_workers_per_link=6)
    assert not p["l1_fit"]
    assert not p["valid"]


# ---------------------------------------------------------------- D=1 correctness (device)
@pytest.mark.skipif(not is_blackhole(), reason="regime_a_matmul is Blackhole-only")
def test_d1_matches_regime_a(device):
    M, K, N = 32, 6144, 3072
    torch.manual_seed(0)
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = (t0.float() @ t1.float())[0, 0]
    a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    cfg = ttnn.RegimeAMatmulConfig(k_slices=3, n_slices=1, m_slices=1, k_block_tiles=4, n_subblock_tiles=6)

    # D=1 (K_local == K_global) => delegates to regime_a_matmul.
    out = ttnn.experimental.all_gather_regime_a_matmul_async(a, b, config=cfg)
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert_with_pcc(ref, got.float(), 0.999)

    # Byte-for-byte identical to calling regime_a_matmul directly.
    out2 = ttnn.experimental.regime_a_matmul(a, b, config=cfg)
    got2 = ttnn.to_torch(ttnn.from_device(out2))[0, 0]
    assert torch.equal(got, got2), "D=1 fused op must equal regime_a_matmul exactly"


# ---------------------------------------------------------------- multi-device full-gather correctness (device)
# Task 3 (#17 D=2, #18 D=4/8): fuse the fabric all-gather of the in0 K-shards with regime_a_matmul in ONE
# program. Fabric only initializes on the full galaxy torus (a small mesh can't complete the ethernet router
# handshake) and the conftest fixture's STRICT_INIT fabric config + the watcher both overflow the ACTIVE_ETH
# kernel-config buffer on this build — so we open the full mesh with a BARE fabric config, no watcher, and carve
# a (1,D) submesh. D>2 uses a ring so the injector's forward hops 1..D-1 reach every other device.
def _run_full_gather(D, mesh_shape, fabric_config, topology):
    M, K, N = 32, 6144, 3072  # K == K_global; each device owns K_local = K/D
    torch.manual_seed(0)
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)  # global in0 [M, K_global], sharded on K
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)  # in1 [K_global, N], replicated
    ref = (t0.float() @ t1.float())[0, 0]

    md = a = b = out = sems = None
    ttnn.set_fabric_config(fabric_config)
    full = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
    try:
        md = full.create_submesh(ttnn.MeshShape(1, D))

        grid = md.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        md.load_sub_device_manager(md.create_sub_device_manager([ttnn.SubDevice([ccl_crs])], 0))
        md.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
        # D+1 GlobalSemaphores: [0] gather_ready (monotonic prefix; injector -> compute fan-out), [1+e]
        # shard_landed[e] (device e marks its shard present everywhere). GlobalSemaphores (not program-local
        # sems) so they can be reset between launches -> the streaming barrier stays correct on cache replay.
        sems = [ttnn.create_global_semaphore(md, ccl_crs, 0) for _ in range(D + 1)]

        a = ttnn.from_torch(
            t0,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.create_mesh_mapper(
                md,
                ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(1, D)),
            ),
        )
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, md)
        b = ttnn.from_torch(
            t1,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            dtype=ttnn.bfloat16,
            memory_config=wcfg,
            mesh_mapper=ttnn.create_mesh_mapper(
                md,
                ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, D)),
            ),
        )
        cfg = ttnn.RegimeAMatmulConfig(k_slices=3, n_slices=1, m_slices=1, k_block_tiles=4, n_subblock_tiles=6)

        # Two invocations: iteration 0 is a fresh program (cache miss), iteration 1 replays the cached program
        # through override_runtime_arguments (exercises the custom compute_program_hash + arg relocation).
        out = None
        for _it in range(2):
            for s in sems:
                ttnn.reset_global_semaphore_value(s, 0)  # fresh barrier state each launch (cache replay safe)
            out = ttnn.experimental.all_gather_regime_a_matmul_async(
                a,
                b,
                config=cfg,
                cluster_axis=1,
                topology=topology,
                num_links=1,
                num_workers_per_link=1,
                multi_device_global_semaphore=sems,
            )
            ttnn.synchronize_device(md)
            # Output [M, N] is computed (replicated) on every device — each must match the full matmul.
            per_dev = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
            for d in range(D):
                assert_with_pcc(ref, per_dev[d].float().reshape(ref.shape), 0.999)
    finally:
        # Drop device buffers (even on assertion failure) so close_mesh_device fully releases all devices and
        # the next parametrization can re-set the fabric config without "devices still open".
        a = b = out = md = sems = None
        gc.collect()
        ttnn.close_mesh_device(full)
        gc.collect()
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(not is_blackhole(), reason="regime_a_matmul is Blackhole-only")
def test_d2_full_gather_correctness():
    # D=2 on a line: device 0 -> forward, device 1 -> backward (both one hop).
    _run_full_gather(2, (8, 4), ttnn.FabricConfig.FABRIC_1D, ttnn.Topology.Linear)


@pytest.mark.skipif(not is_blackhole(), reason="regime_a_matmul is Blackhole-only")
@pytest.mark.parametrize(
    "D, mesh_shape",
    [(4, (8, 4)), (8, (4, 8))],  # (1,8) submesh needs axis-1 extent 8 -> open the mesh as (4,8)
)
def test_dn_full_gather_correctness_ring(D, mesh_shape):
    # D>2: ring so the injector's forward hops 1..D-1 reach every other device.
    _run_full_gather(D, mesh_shape, ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring)
