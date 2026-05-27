# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fused wan_fused_distributed_rmsnorm device op.

Two cases:
  * TP=1 (1x1 submesh) — the AG is degenerate and the compute kernel pushes
    stats directly into stats_gathered_cb (is_tp_1==1); writer just drains
    output. Isolates the merged pre+post compute kernel.
  * TP=2 (1x2 submesh) — exercises the writer-resident fabric forwarder:
    each worker core fabric-multicasts its stats tile to the matching core
    on the neighbor chip, then waits on a GlobalSemaphore.

The existing TP=4 benchmark still routes through the composite path
(use_device_op=False) so production behavior is unchanged.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.tensor import bf16_tensor, from_torch, to_torch
from ....utils.test import line_params, ring_params


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Reference RMSNorm: out = x * 1/sqrt(mean(x**2) + eps) [* weight]."""
    x = x.to(torch.float32)
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(mean_sq + eps)
    if weight is not None:
        out = out * weight.to(torch.float32).reshape(1, 1, 1, -1)
    return out.to(torch.bfloat16)


def _torch_rmsnorm_then_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reference: rmsnorm(x) [* weight], then per-head RoPE.

    x: [B, 1, N, H]; weight: [H] (broadcast); cos/sin: [B, num_heads_dim, N, head_dim]
    where num_heads_dim is 1 (broadcast) or num_heads (per-head).
    Returns [B, num_heads, N, head_dim].
    """
    rms = _torch_rmsnorm(x, weight, eps).to(torch.float32)  # [B, 1, N, H]
    B, _, N, H = rms.shape
    # [B, 1, N, H] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
    heads = rms.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    rotated = torch.stack([-heads[..., 1::2], heads[..., 0::2]], dim=-1).flatten(-2)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    out = heads * cos + rotated * sin  # broadcasts over num_heads_dim when cos/sin dim 1 is 1
    return out.to(torch.bfloat16)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "H", "has_weight"),
    [
        # Small validation shapes:
        pytest.param(32, 256, False, id="N32_H256_noweight"),
        pytest.param(32, 256, True, id="N32_H256_weight"),
        pytest.param(64, 512, True, id="N64_H512_weight"),
        # Production-scale shapes at TP=1 (equivalent to Wan2.2 SP=32 / TP=4 single TP slice).
        # N=9472 currently exceeds the default 5s dispatch timeout in run_safe_pytest.sh;
        # run with TT_METAL_OPERATION_TIMEOUT_SECONDS=60 or direct pytest invocation to pass.
        pytest.param(2368, 1280, True, id="N2368_H1280_weight"),
        pytest.param(9472, 1280, True, id="N9472_H1280_weight", marks=pytest.mark.slow),
    ],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp1(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H: int,
    has_weight: bool,
) -> None:
    """Run the new fused op on a 1x1 submesh; compare against torch RMSNorm.

    Skips RoPE for this first validation — RoPE adds a lot of moving parts
    and we want to isolate the L1-residency + absolute-indexing claim first.
    """
    # 1x1 submesh: just one device. ring_size==1, no AG needed.
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))

    # CCL manager not strictly needed for ring_size==1, but the API takes a
    # semaphore vector; we supply an empty one because the device op short-
    # circuits AG.
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(0)  # cluster_axis=0 unused at ring_size=1

    EPS = 1e-6
    NUM_HEADS_PER_DEVICE = 1  # No head-split for this first test; keep output 4D-flat.
    # NOTE: num_heads_per_device=1 means output shape == input shape.

    torch.manual_seed(0)
    x_torch = torch.randn((1, 1, N, H), dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_torch, device=submesh)

    if has_weight:
        weight_torch = torch.randn(H, dtype=torch.bfloat16)
        # Weight as TILE-layout [1, H] for the existing post-allgather contract.
        tt_weight = from_torch(weight_torch.reshape(1, H), device=submesh, dtype=ttnn.bfloat16)
    else:
        weight_torch = None
        tt_weight = None

    logger.info(f"Running fused op (use_device_op=True) on 1x1 submesh, N={N}, H={H}, has_weight={has_weight}")
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        0,  # cluster_axis (unused at ring_size=1)
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=EPS,
        num_heads_per_device=NUM_HEADS_PER_DEVICE,
        weight=tt_weight,
        use_device_op=True,
    )

    tt_out_torch = to_torch(out)
    ref = _torch_rmsnorm(x_torch, weight_torch, EPS)
    assert_quality(ref, tt_out_torch, pcc=0.999)


# ---------------------------------------------------------------------------
# TP=2 ring test — exercises the writer-resident fabric forwarder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "H_per_device", "has_weight"),
    [
        # H_per_device must be a multiple of TILE_WIDTH (32) AND of
        # num_heads_per_device * head_dim. We keep num_heads_per_device=1.
        pytest.param(32, 128, False, id="N32_H128pd_noweight"),
        pytest.param(32, 128, True, id="N32_H128pd_weight"),
        pytest.param(64, 256, True, id="N64_H256pd_weight"),
    ],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp2(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H_per_device: int,
    has_weight: bool,
) -> None:
    """Run the new fused op on a 1x2 submesh; compare against torch RMSNorm.

    The input is sharded across the cluster_axis: each chip sees H_per_device
    columns, and the full hidden dim H_full = H_per_device * 2 is the reduce
    axis. The torch reference computes RMSNorm on the full unsharded tensor;
    after the fabric AG the per-chip output should match the corresponding
    slice of the reference.
    """
    H_FULL = H_per_device * 2  # noqa: N806 — TP=2

    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 2))
    cluster_axis = 1  # the axis with size 2

    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)

    EPS = 1e-6
    NUM_HEADS_PER_DEVICE = 1

    torch.manual_seed(0)
    x_full = torch.randn((1, 1, N, H_FULL), dtype=torch.bfloat16)
    # Shard the H_FULL hidden dim across the TP cluster axis.
    tt_x = bf16_tensor(x_full, device=submesh, mesh_axis=cluster_axis, shard_dim=-1)

    if has_weight:
        # Weight is broadcast along N and TP-sharded along H to match the per-chip H_per_device.
        weight_full = torch.randn(H_FULL, dtype=torch.bfloat16)
        tt_weight = bf16_tensor(
            weight_full.reshape(1, H_FULL),
            device=submesh,
            mesh_axis=cluster_axis,
            shard_dim=-1,
        )
    else:
        weight_full = None
        tt_weight = None

    logger.info(
        f"Running fused TP=2 op on 1x2 submesh, N={N}, H_per_device={H_per_device}, "
        f"H_full={H_FULL}, has_weight={has_weight}"
    )
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        cluster_axis,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=EPS,
        num_heads_per_device=NUM_HEADS_PER_DEVICE,
        weight=tt_weight,
        use_device_op=True,
    )

    # Output is sharded along the cluster_axis on dim -1 (head_dim is what
    # remains after the head-split — for num_heads_per_device=1 this == H_per_device).
    # Concatenate device shards back to the unsharded reference.
    out_full = to_torch(out, mesh_axes=[None, None, None, cluster_axis])
    ref = _torch_rmsnorm(x_full, weight_full, EPS)
    # Reshape ref to match out shape [1, num_heads_per_device(=1), N, H_full]
    ref = ref.reshape(1, 1, N, H_FULL)
    assert_quality(ref, out_full, pcc=0.999)


# ---------------------------------------------------------------------------
# TP=4 ring test — same writer-resident forwarder, ring topology
# ---------------------------------------------------------------------------


def _make_ring_submesh_tp4(parent_mesh: ttnn.MeshDevice) -> ttnn.MeshDevice:
    """Carve a 2x2 submesh (4-cycle) and present it as 1x4 so wraparound links exist.

    Same trick as the production benchmark — the physical 2x2 cycle gives us a
    real ring; the 1x4 reshape makes axis 1 the TP axis.
    """
    submesh = parent_mesh.create_submesh(ttnn.MeshShape(2, 2))
    submesh.reshape(ttnn.MeshShape(1, 4))
    return submesh


def _make_line_submesh_tp4(parent_mesh: ttnn.MeshDevice) -> ttnn.MeshDevice:
    """Carve a 1x4 submesh directly (no reshape) — straight line, no wraparound."""
    return parent_mesh.create_submesh(ttnn.MeshShape(1, 4))


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "topology"),
    [
        ((2, 4), {**line_params, "trace_region_size": 90112}, ttnn.Topology.Linear),
        ((2, 4), {**ring_params, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["mesh_device", "device_params"],
    ids=["bh_2x4_line", "bh_2x4_ring"],
)
@pytest.mark.parametrize(
    ("N", "H_per_device", "has_weight"),
    [
        pytest.param(32, 128, False, id="N32_H128pd_noweight"),
        pytest.param(32, 128, True, id="N32_H128pd_weight"),
        pytest.param(64, 256, True, id="N64_H256pd_weight"),
    ],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp4(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H_per_device: int,
    has_weight: bool,
    topology: ttnn.Topology,
) -> None:
    """Run the new fused op on a 1x4 submesh; compare against torch RMSNorm.

    This is the production topology (TP=4). Input is sharded H_per_device
    columns per chip across the cluster_axis. With the current single-AG-core
    restriction each chip runs on 1 worker — correctness should still hold.
    """
    H_FULL = H_per_device * 4  # noqa: N806 — TP=4

    if topology == ttnn.Topology.Ring:
        submesh = _make_ring_submesh_tp4(mesh_device)
    else:
        # Linear: prefer a true 1x4 line from one row of the parent (no reshape).
        submesh = _make_line_submesh_tp4(mesh_device)
    cluster_axis = 1

    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)

    EPS = 1e-6
    NUM_HEADS_PER_DEVICE = 1

    torch.manual_seed(0)
    x_full = torch.randn((1, 1, N, H_FULL), dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_full, device=submesh, mesh_axis=cluster_axis, shard_dim=-1)

    if has_weight:
        weight_full = torch.randn(H_FULL, dtype=torch.bfloat16)
        tt_weight = bf16_tensor(
            weight_full.reshape(1, H_FULL),
            device=submesh,
            mesh_axis=cluster_axis,
            shard_dim=-1,
        )
    else:
        weight_full = None
        tt_weight = None

    logger.info(
        f"Running fused TP=4 op on 1x4 submesh, topology={topology}, N={N}, "
        f"H_per_device={H_per_device}, H_full={H_FULL}, has_weight={has_weight}"
    )
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        cluster_axis,
        submesh,
        ag_sem,
        topology=topology,
        epsilon=EPS,
        num_heads_per_device=NUM_HEADS_PER_DEVICE,
        weight=tt_weight,
        use_device_op=True,
    )

    out_full = to_torch(out, mesh_axes=[None, None, None, cluster_axis])
    ref = _torch_rmsnorm(x_full, weight_full, EPS)
    ref = ref.reshape(1, 1, N, H_FULL)
    assert_quality(ref, out_full, pcc=0.999)


# ---------------------------------------------------------------------------
# TP=4 LINE perf comparison — composite (use_device_op=False) vs fused (=True)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "H_per_device"),
    [
        pytest.param(32, 128, id="N32_H128pd"),
        pytest.param(64, 256, id="N64_H256pd"),
        pytest.param(128, 512, id="N128_H512pd"),
        pytest.param(256, 1280, id="N256_H1280pd_prodsize"),
    ],
)
def test_wan_fused_distributed_rmsnorm_perf_tp4_line(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H_per_device: int,
) -> None:
    """Compare composite vs fused device op latency at TP=4 LINE on natural 1x4
    submesh. Both paths run with traced execution to amortize dispatch cost; we
    report avg us/iter for each.

    Note: the production topology is TP=4 RING (2x2 reshape submesh). Our fused
    device op doesn't support that reshape yet (see TP milestone memory).
    This test uses LINE on a natural 1x4 row of the parent — apples-to-apples
    comparison at the same shape and topology.
    """
    import time

    H_FULL = H_per_device * 4  # noqa: N806
    NUM_ITERS = 50

    submesh = _make_line_submesh_tp4(mesh_device)
    cluster_axis = 1
    topology = ttnn.Topology.Linear

    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)

    EPS = 1e-6
    NUM_HEADS_PER_DEVICE = 1

    torch.manual_seed(0)
    x_full = torch.randn((1, 1, N, H_FULL), dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_full, device=submesh, mesh_axis=cluster_axis, shard_dim=-1)
    weight_full = torch.randn(H_FULL, dtype=torch.bfloat16)
    tt_weight = bf16_tensor(
        weight_full.reshape(1, H_FULL),
        device=submesh,
        mesh_axis=cluster_axis,
        shard_dim=-1,
    )

    def _run(use_device_op: bool) -> ttnn.Tensor:
        return ttnn.experimental.wan_fused_distributed_rmsnorm(
            tt_x,
            cluster_axis,
            submesh,
            ag_sem,
            topology=topology,
            epsilon=EPS,
            num_heads_per_device=NUM_HEADS_PER_DEVICE,
            weight=tt_weight,
            use_device_op=use_device_op,
        )

    def _trace_and_time(use_device_op: bool) -> float:
        # Warmup + compile
        _run(use_device_op)
        ttnn.synchronize_device(submesh)
        # Capture trace
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        _run(use_device_op)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)
        # Time
        t0 = time.perf_counter()
        for _ in range(NUM_ITERS):
            ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(submesh)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        ttnn.release_trace(submesh, trace_id)
        return elapsed_us / NUM_ITERS

    logger.info(f"=== TP=4 LINE perf, N={N}, H_per_device={H_per_device}, H_full={H_FULL} ===")
    composite_us = _trace_and_time(use_device_op=False)
    logger.info(f"[composite]  {composite_us:.2f} us/iter")
    fused_us = _trace_and_time(use_device_op=True)
    logger.info(f"[fused-op]   {fused_us:.2f} us/iter")
    speedup = composite_us / fused_us
    logger.info(f"speedup:     {speedup:.2f}x ({composite_us:.2f} → {fused_us:.2f} us)")


# ---------------------------------------------------------------------------
# TP=8 ring on natively-opened 1x8 mesh
# ---------------------------------------------------------------------------
#
# Avoids the 2x2-reshape submesh fabric issue by opening the parent mesh
# directly as 1x8. With FABRIC_1D_RING and Ring topology, all 8 chips form a
# physical ring whose order matches the mesh-coord cycle, so multi-hop mcast
# routes correctly. Both correctness vs torch and a fused-vs-composite perf
# comparison live here.


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((1, 8), {**ring_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_1x8_ring"],
)
@pytest.mark.parametrize(
    ("N", "H_per_device", "has_weight"),
    [
        pytest.param(32, 64, False, id="N32_H64pd_noweight"),
        pytest.param(32, 64, True, id="N32_H64pd_weight"),
        pytest.param(64, 128, True, id="N64_H128pd_weight"),
        pytest.param(128, 256, True, id="N128_H256pd_weight"),
    ],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp8_ring_native(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H_per_device: int,
    has_weight: bool,
) -> None:
    """TP=8 ring AG on a natively-opened 1x8 mesh (no submesh reshape).

    The 8 BH chips form a physical Hamiltonian cycle, and `(1, 8)` opens them
    in that cycle order, so mesh-coord adjacency == fabric adjacency. Per-row
    multi-hop mcast should route correctly here (unlike the 2x2-reshape case).
    """
    H_FULL = H_per_device * 8  # noqa: N806

    # Use the parent mesh directly — already 1x8.
    submesh = mesh_device
    cluster_axis = 1

    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Ring)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)

    EPS = 1e-6
    NUM_HEADS_PER_DEVICE = 1

    torch.manual_seed(0)
    x_full = torch.randn((1, 1, N, H_FULL), dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_full, device=submesh, mesh_axis=cluster_axis, shard_dim=-1)

    if has_weight:
        weight_full = torch.randn(H_FULL, dtype=torch.bfloat16)
        tt_weight = bf16_tensor(
            weight_full.reshape(1, H_FULL),
            device=submesh,
            mesh_axis=cluster_axis,
            shard_dim=-1,
        )
    else:
        weight_full = None
        tt_weight = None

    logger.info(
        f"Running fused TP=8 op on native 1x8 ring mesh, N={N}, H_per_device={H_per_device}, "
        f"H_full={H_FULL}, has_weight={has_weight}"
    )
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        cluster_axis,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Ring,
        epsilon=EPS,
        num_heads_per_device=NUM_HEADS_PER_DEVICE,
        weight=tt_weight,
        use_device_op=True,
    )

    out_full = to_torch(out, mesh_axes=[None, None, None, cluster_axis])
    ref = _torch_rmsnorm(x_full, weight_full, EPS)
    ref = ref.reshape(1, 1, N, H_FULL)
    assert_quality(ref, out_full, pcc=0.999)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((1, 8), {**ring_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_1x8_ring"],
)
@pytest.mark.parametrize(
    ("N", "H_per_device"),
    [
        pytest.param(32, 64, id="N32_H64pd"),
        pytest.param(64, 128, id="N64_H128pd"),
        pytest.param(128, 256, id="N128_H256pd"),
        pytest.param(256, 640, id="N256_H640pd_prodlike"),
        pytest.param(512, 64, id="N512_H64pd_multichunk_small"),
        pytest.param(1024, 64, id="N1024_H64pd_multichunk_med"),
        pytest.param(2048, 64, id="N2048_H64pd_multichunk_big"),
    ],
)
def test_wan_fused_distributed_rmsnorm_perf_tp8_ring_native(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H_per_device: int,
) -> None:
    """Composite vs fused perf comparison at TP=8 RING on native 1x8 mesh."""
    import time

    H_FULL = H_per_device * 8  # noqa: N806
    NUM_ITERS = 50

    submesh = mesh_device
    cluster_axis = 1
    topology = ttnn.Topology.Ring

    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)

    EPS = 1e-6
    NUM_HEADS_PER_DEVICE = 1

    torch.manual_seed(0)
    x_full = torch.randn((1, 1, N, H_FULL), dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_full, device=submesh, mesh_axis=cluster_axis, shard_dim=-1)
    weight_full = torch.randn(H_FULL, dtype=torch.bfloat16)
    tt_weight = bf16_tensor(
        weight_full.reshape(1, H_FULL),
        device=submesh,
        mesh_axis=cluster_axis,
        shard_dim=-1,
    )

    def _run(use_device_op: bool) -> ttnn.Tensor:
        return ttnn.experimental.wan_fused_distributed_rmsnorm(
            tt_x,
            cluster_axis,
            submesh,
            ag_sem,
            topology=topology,
            epsilon=EPS,
            num_heads_per_device=NUM_HEADS_PER_DEVICE,
            weight=tt_weight,
            use_device_op=use_device_op,
        )

    def _trace_and_time(use_device_op: bool) -> float:
        _run(use_device_op)
        ttnn.synchronize_device(submesh)
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        _run(use_device_op)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)
        t0 = time.perf_counter()
        for _ in range(NUM_ITERS):
            ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(submesh)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        ttnn.release_trace(submesh, trace_id)
        return elapsed_us / NUM_ITERS

    logger.info(f"=== TP=8 RING perf, N={N}, H_per_device={H_per_device}, H_full={H_FULL} ===")
    composite_us = _trace_and_time(use_device_op=False)
    logger.info(f"[composite]  {composite_us:.2f} us/iter")
    fused_us = _trace_and_time(use_device_op=True)
    logger.info(f"[fused-op]   {fused_us:.2f} us/iter")
    speedup = composite_us / fused_us
    logger.info(f"speedup:     {speedup:.2f}x ({composite_us:.2f} → {fused_us:.2f} us)")


# ---------------------------------------------------------------------------
# TP=1 RoPE tests — validates broadcast (per_head_rope=False) and per-head
# (per_head_rope=True) cos/sin handling.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "H", "num_heads", "per_head_rope"),
    [
        pytest.param(32, 256, 1, False, id="N32_H256_1head_bcast"),
        pytest.param(32, 512, 4, False, id="N32_H512_4heads_bcast"),
        pytest.param(32, 512, 4, True, id="N32_H512_4heads_perhead"),
        pytest.param(64, 1024, 8, True, id="N64_H1024_8heads_perhead"),
    ],
)
def test_wan_fused_distributed_rmsnorm_tp1_rope(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H: int,
    num_heads: int,
    per_head_rope: bool,
) -> None:
    """TP=1 RoPE: per_head_rope=True passes cos/sin with shape [1, num_heads, N, head_dim]
    so each head sees its own rotation; per_head_rope=False uses [1, 1, N, head_dim] broadcast."""
    head_dim = H // num_heads
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ccl = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)
    ag_sem = ccl.get_ag_ping_pong_semaphore(0)

    EPS = 1e-6
    torch.manual_seed(0)
    x_torch = torch.randn((1, 1, N, H), dtype=torch.bfloat16)
    weight_torch = torch.randn(H, dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_torch, device=submesh)
    tt_w = from_torch(weight_torch.reshape(1, H), device=submesh, dtype=ttnn.bfloat16)

    num_heads_dim = num_heads if per_head_rope else 1
    cos_raw = torch.randn(1, num_heads_dim, N, head_dim // 2)
    sin_raw = torch.randn(1, num_heads_dim, N, head_dim // 2)
    cos_full, sin_full = stack_cos_sin(cos_raw, sin_raw)
    tt_cos = from_torch(cos_full, device=submesh, dtype=ttnn.float32)
    tt_sin = from_torch(sin_full, device=submesh, dtype=ttnn.float32)
    tt_trans = bf16_tensor(get_rot_transformation_mat(), device=submesh)

    logger.info(f"TP=1 RoPE: N={N} H={H} num_heads={num_heads} head_dim={head_dim} per_head_rope={per_head_rope}")
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        0,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=EPS,
        num_heads_per_device=num_heads,
        weight=tt_w,
        transformation_mat=tt_trans,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
        use_device_op=True,
    )

    out_torch = to_torch(out)  # [1, num_heads, N, head_dim]
    ref = _torch_rmsnorm_then_rope(x_torch, weight_torch, EPS, cos_full, sin_full, num_heads, head_dim)
    assert_quality(ref, out_torch, pcc=0.999)
