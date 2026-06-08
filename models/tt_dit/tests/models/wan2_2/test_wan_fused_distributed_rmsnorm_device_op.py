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
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch, to_torch
from ....utils.test import line_params, ring_params


def _torch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference RMSNorm: out = x * 1/sqrt(mean(x**2) + eps) [* weight] [+ bias]."""
    x = x.to(torch.float32)
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(mean_sq + eps)
    if weight is not None:
        out = out * weight.to(torch.float32).reshape(1, 1, 1, -1)
    if bias is not None:
        out = out + bias.to(torch.float32).reshape(1, 1, 1, -1)
    return out.to(torch.bfloat16)


def _torch_per_head_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Per-head RMSNorm: reduce over head_dim only.

    x: [B, 1, N, H] where H = num_heads * head_dim. Returns same shape.
    """
    B, _, N, H = x.shape
    assert H == num_heads * head_dim
    x_heads = x.to(torch.float32).reshape(B, 1, N, num_heads, head_dim)
    mean_sq = (x_heads * x_heads).mean(dim=-1, keepdim=True)  # [B,1,N,num_heads,1]
    out = x_heads * torch.rsqrt(mean_sq + eps)
    out = out.reshape(B, 1, N, H)
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


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "H"),
    [
        pytest.param(32, 256, id="N32_H256"),
        pytest.param(64, 1280, id="N64_H1280"),
    ],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp1_bias(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H: int,
) -> None:
    """TP=1 with optional bias: y = rmsnorm(x) * weight + bias."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(0)

    EPS = 1e-6
    torch.manual_seed(0)
    x_torch = torch.randn((1, 1, N, H), dtype=torch.bfloat16)
    weight_torch = torch.randn(H, dtype=torch.bfloat16)
    bias_torch = torch.randn(H, dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_torch, device=submesh)
    tt_weight = from_torch(weight_torch.reshape(1, H), device=submesh, dtype=ttnn.bfloat16)
    tt_bias = from_torch(bias_torch.reshape(1, H), device=submesh, dtype=ttnn.bfloat16)

    logger.info(f"TP=1 with bias: N={N} H={H}")
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        0,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=EPS,
        num_heads_per_device=1,
        weight=tt_weight,
        bias=tt_bias,
        use_device_op=True,
    )

    out_torch = to_torch(out)
    ref = _torch_rmsnorm(x_torch, weight_torch, EPS, bias=bias_torch)
    assert_quality(ref, out_torch, pcc=0.999)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "H", "has_bias"),
    [
        pytest.param(32, 256, False, id="N32_H256_noBias"),
        pytest.param(32, 256, True, id="N32_H256_Bias"),
        pytest.param(64, 1280, True, id="N64_H1280_Bias"),
    ],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp1_per_token_weight(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H: int,
    has_bias: bool,
) -> None:
    """TP=1 with per-token weight (and optional per-token bias) — shape [N, H]."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(0)

    EPS = 1e-6
    torch.manual_seed(0)
    x_torch = torch.randn((1, 1, N, H), dtype=torch.bfloat16)
    weight_torch = torch.randn(N, H, dtype=torch.bfloat16)
    bias_torch = torch.randn(N, H, dtype=torch.bfloat16) if has_bias else None
    tt_x = bf16_tensor(x_torch, device=submesh)
    tt_weight = from_torch(weight_torch, device=submesh, dtype=ttnn.bfloat16)
    tt_bias = from_torch(bias_torch, device=submesh, dtype=ttnn.bfloat16) if has_bias else None

    logger.info(f"Per-token weight: N={N} H={H} has_bias={has_bias}")
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        0,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=EPS,
        num_heads_per_device=1,
        weight=tt_weight,
        bias=tt_bias,
        use_device_op=True,
    )

    out_torch = to_torch(out)
    x_f = x_torch.to(torch.float32)
    mean_sq = (x_f * x_f).mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(mean_sq + EPS)
    # Per-token weight/bias broadcast across heads/channels but per row.
    w_f = weight_torch.to(torch.float32).reshape(1, 1, N, H)
    ref = x_f * rms * w_f
    if has_bias:
        ref = ref + bias_torch.to(torch.float32).reshape(1, 1, N, H)
    ref = ref.to(torch.bfloat16)
    assert_quality(ref, out_torch, pcc=0.999)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp1_fp32_input(
    mesh_device: ttnn.MeshDevice,
) -> None:
    """TP=1 with FP32 input. Validates the host-side fp32_dest_acc_en + unpack-
    to-dest-fp32 fix so the unpacker doesn't silently downcast through SrcA to
    TF32. Output cast to BFLOAT16 to compare against an fp32 torch reference.
    """
    N, H = 32, 256
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(0)

    EPS = 1e-6
    torch.manual_seed(0)
    x_torch = torch.randn((1, 1, N, H), dtype=torch.float32)
    weight_torch = torch.randn(H, dtype=torch.bfloat16)
    tt_x = from_torch(x_torch, device=submesh, dtype=ttnn.float32)
    tt_weight = from_torch(weight_torch.reshape(1, H), device=submesh, dtype=ttnn.bfloat16)

    logger.info(f"FP32 input: N={N} H={H}")
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        0,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=EPS,
        num_heads_per_device=1,
        weight=tt_weight,
        dtype=ttnn.bfloat16,
        use_device_op=True,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )

    out_torch = to_torch(out)
    ref = _torch_rmsnorm(x_torch, weight_torch, EPS)
    assert_quality(ref, out_torch, pcc=0.999)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "num_heads", "head_dim", "has_weight"),
    [
        pytest.param(32, 4, 128, False, id="N32_4heads_d128_noweight"),
        pytest.param(32, 4, 128, True, id="N32_4heads_d128_weight"),
        pytest.param(64, 8, 128, True, id="N64_8heads_d128_weight"),
    ],
)
def test_wan_fused_distributed_rmsnorm_device_op_tp1_per_head_norm(
    mesh_device: ttnn.MeshDevice,
    N: int,
    num_heads: int,
    head_dim: int,
    has_weight: bool,
) -> None:
    """Per-head normalization (FLUX.2): reduce over head_dim per head, not full row."""
    H = num_heads * head_dim
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(0)

    EPS = 1e-6
    torch.manual_seed(0)
    x_torch = torch.randn((1, 1, N, H), dtype=torch.bfloat16)
    tt_x = bf16_tensor(x_torch, device=submesh)
    if has_weight:
        weight_torch = torch.randn(H, dtype=torch.bfloat16)
        tt_weight = from_torch(weight_torch.reshape(1, H), device=submesh, dtype=ttnn.bfloat16)
    else:
        weight_torch = None
        tt_weight = None

    logger.info(f"Per-head norm: N={N} num_heads={num_heads} head_dim={head_dim} has_weight={has_weight}")
    out = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        0,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=EPS,
        num_heads_per_device=num_heads,
        per_head_norm=True,
        weight=tt_weight,
        use_device_op=True,
    )

    out_torch = to_torch(out)  # [1, num_heads, N, head_dim]
    # Compare in [1, num_heads, N, head_dim] form. Reference is [1, 1, N, H]
    # interleaved by head; reshape via [1, N, num_heads, head_dim] then permute.
    ref_flat = _torch_per_head_rmsnorm(x_torch, weight_torch, EPS, num_heads, head_dim)
    ref = ref_flat.reshape(1, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    assert_quality(ref, out_torch, pcc=0.999)


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
        # bh_2x4_ring removed: known pre-existing hang on multi-hop fabric mcast
        # via the zig-zag 2x4→1x4 reshape path. See memory
        # fused_distributed_rmsnorm_tp4_ring_blocker.md.
    ],
    indirect=["mesh_device", "device_params"],
    ids=["bh_2x4_line"],
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
# TP=4 LINE correctness with the model-like call pattern: multi-head + RoPE.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("N", "H_per_device", "num_heads_per_device"),
    [
        pytest.param(128, 1280, 10, id="N128_H1280pd_10heads"),
        pytest.param(2368, 1280, 10, id="N2368_H1280pd_10heads"),
        pytest.param(12400, 1280, 10, id="N12400_H1280pd_10heads"),
    ],
)
def test_wan_fused_distributed_rmsnorm_tp4_line_multihead_rope(
    mesh_device: ttnn.MeshDevice,
    N: int,
    H_per_device: int,
    num_heads_per_device: int,
) -> None:
    """Mirror the production model call: TP=4 LINE + weight + broadcast RoPE +
    num_heads_per_device > 1. This is the MUX path with multi-head writer
    output indexing — not previously correctness-tested."""
    H_FULL = H_per_device * 4  # noqa: N806
    head_dim = H_per_device // num_heads_per_device

    # Use the full 2x4 parent mesh with cluster_axis=1 (TP). Input gets
    # sp-sharded along axis 0 too — matches the model's setup.
    submesh = mesh_device
    cluster_axis = 1
    sp_axis = 0
    sp_factor = 2
    topology = ttnn.Topology.Linear
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)
    ag_sem = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)

    EPS = 1e-6
    torch.manual_seed(0)
    # Use a torch matmul to source x so values have realistic magnitudes
    # (sqrt(K)~70 stdev for K=5120) — the model's matmul-output distribution
    # is what triggers the divergence; plain randn doesn't.
    DIM = 5120
    matmul_a = torch.randn((1, 1, N * sp_factor, DIM), dtype=torch.bfloat16)
    matmul_b = torch.randn((DIM, H_FULL), dtype=torch.bfloat16)
    x_full = (matmul_a.to(torch.float32) @ matmul_b.to(torch.float32)).to(torch.bfloat16)
    weight_full = torch.randn(H_FULL, dtype=torch.bfloat16)
    cos_raw = torch.randn(1, 1, N * sp_factor, head_dim // 2)
    sin_raw = torch.randn(1, 1, N * sp_factor, head_dim // 2)
    cos_full, sin_full = stack_cos_sin(cos_raw, sin_raw)

    tt_x = bf16_tensor_2dshard(x_full, device=submesh, shard_mapping={sp_axis: 2, cluster_axis: 3})
    tt_weight = bf16_tensor(weight_full.reshape(1, H_FULL), device=submesh, mesh_axis=cluster_axis, shard_dim=-1)
    tt_cos = from_torch(cos_full, device=submesh, dtype=ttnn.float32, mesh_axes=[None, None, sp_axis, None])
    tt_sin = from_torch(sin_full, device=submesh, dtype=ttnn.float32, mesh_axes=[None, None, sp_axis, None])
    tt_trans = bf16_tensor(get_rot_transformation_mat(), device=submesh)

    persistent_output_buffer = ttnn.experimental.wan_fused_distributed_rmsnorm_create_stats_buffer(
        tt_x, cluster_axis, submesh, num_heads_per_device=num_heads_per_device
    )

    logger.info(
        f"TP=4 LINE multihead+RoPE: N={N} H_per_device={H_per_device} " f"num_heads_per_device={num_heads_per_device}"
    )

    # Run BOTH composite and fused on the same input to check bit-exactness.
    out_fused = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        cluster_axis,
        submesh,
        ag_sem,
        topology=topology,
        epsilon=EPS,
        num_heads_per_device=num_heads_per_device,
        weight=tt_weight,
        transformation_mat=tt_trans,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
        persistent_output_buffer=persistent_output_buffer,
        use_device_op=True,
    )

    out_composite = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        cluster_axis,
        submesh,
        ccl_manager.get_ag_ping_pong_semaphore(cluster_axis),
        topology=topology,
        epsilon=EPS,
        num_heads_per_device=num_heads_per_device,
        weight=tt_weight,
        transformation_mat=tt_trans,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
        use_device_op=False,
    )

    # Per-chip comparison (chip 0).
    c_t = ttnn.to_torch(ttnn.get_device_tensors(out_composite)[0]).to(torch.float32).flatten()
    f_t = ttnn.to_torch(ttnn.get_device_tensors(out_fused)[0]).to(torch.float32).flatten()
    diff = (c_t - f_t).abs()
    max_d = float(diff.max())
    cov = torch.cov(torch.stack([c_t, f_t]))
    std_c = float(cov[0, 0].sqrt())
    std_f = float(cov[1, 1].sqrt())
    pcc_cf = float(cov[0, 1]) / (std_c * std_f) if std_c * std_f > 0 else float("nan")
    logger.info(
        f"COMPOSITE_VS_FUSED chip0: PCC={pcc_cf * 100:.4f}% max|diff|={max_d:.4e} "
        f"c_range=[{float(c_t.min()):.3f},{float(c_t.max()):.3f}] "
        f"f_range=[{float(f_t.min()):.3f},{float(f_t.max()):.3f}]"
    )

    # Output shape per chip: [1, num_heads_per_device, N, head_dim].
    out_full = to_torch(out_fused, mesh_axes=[None, cluster_axis, sp_axis, None])
    ref_flat = _torch_rmsnorm_then_rope(
        x_full, weight_full, EPS, cos_full, sin_full, num_heads_per_device * 4, head_dim
    )
    assert_quality(ref_flat, out_full, pcc=0.999)


# ---------------------------------------------------------------------------
# Composite-vs-fused bit-exactness bisection. Toggles weight / RoPE to
# localize which sub-phase introduces the precision divergence that the
# wan transformer block test surfaces.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((2, 4), {**line_params, "trace_region_size": 90112})],
    indirect=True,
    ids=["bh_2x4_line"],
)
@pytest.mark.parametrize(
    ("use_weight", "use_rope"),
    [
        pytest.param(False, False, id="rms_only"),
        pytest.param(True, False, id="rms_weight"),
        pytest.param(True, True, id="rms_weight_rope"),
    ],
)
@pytest.mark.parametrize(
    "N",
    [32, 4096, 8192, 12400],
    ids=["N32_1chunk", "N4096_1chunk", "N8192_2chunks", "N12400_prod_2chunks"],
)
def test_wan_fused_distributed_rmsnorm_tp4_composite_vs_fused_bisect(
    mesh_device: ttnn.MeshDevice,
    use_weight: bool,
    use_rope: bool,
    N: int,
) -> None:
    """Compare composite vs fused per-chip output at the Wan production shape,
    with matmul-derived input (realistic value magnitudes that trigger the
    divergence). Logs PCC + max|diff|; intentionally does NOT assert so all
    variants always run. N parametrized to isolate chunk-count effects."""
    H_per_device = 1280
    num_heads_per_device = 10
    H_FULL = H_per_device * 4
    head_dim = H_per_device // num_heads_per_device
    sp_factor = 2

    submesh = mesh_device
    cluster_axis = 1
    sp_axis = 0
    topology = ttnn.Topology.Linear
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=topology)

    EPS = 1e-6
    torch.manual_seed(0)
    DIM = 5120
    matmul_a = torch.randn((1, 1, N * sp_factor, DIM), dtype=torch.bfloat16)
    matmul_b = torch.randn((DIM, H_FULL), dtype=torch.bfloat16)
    x_full = (matmul_a.to(torch.float32) @ matmul_b.to(torch.float32)).to(torch.bfloat16)
    tt_x = bf16_tensor_2dshard(x_full, device=submesh, shard_mapping={sp_axis: 2, cluster_axis: 3})

    tt_weight = None
    if use_weight:
        weight_full = torch.randn(H_FULL, dtype=torch.bfloat16)
        tt_weight = bf16_tensor(weight_full.reshape(1, H_FULL), device=submesh, mesh_axis=cluster_axis, shard_dim=-1)

    tt_cos = tt_sin = tt_trans = None
    if use_rope:
        cos_raw = torch.randn(1, 1, N * sp_factor, head_dim // 2)
        sin_raw = torch.randn(1, 1, N * sp_factor, head_dim // 2)
        cos_full, sin_full = stack_cos_sin(cos_raw, sin_raw)
        tt_cos = from_torch(cos_full, device=submesh, dtype=ttnn.float32, mesh_axes=[None, None, sp_axis, None])
        tt_sin = from_torch(sin_full, device=submesh, dtype=ttnn.float32, mesh_axes=[None, None, sp_axis, None])
        tt_trans = bf16_tensor(get_rot_transformation_mat(), device=submesh)

    persistent_output_buffer = ttnn.experimental.wan_fused_distributed_rmsnorm_create_stats_buffer(
        tt_x, cluster_axis, submesh, num_heads_per_device=num_heads_per_device
    )

    common_kwargs = dict(
        topology=topology,
        epsilon=EPS,
        num_heads_per_device=num_heads_per_device,
        weight=tt_weight,
        transformation_mat=tt_trans,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
    )

    out_fused = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        cluster_axis,
        submesh,
        ccl_manager.get_ag_ping_pong_semaphore(cluster_axis),
        persistent_output_buffer=persistent_output_buffer,
        use_device_op=True,
        **common_kwargs,
    )

    out_composite = ttnn.experimental.wan_fused_distributed_rmsnorm(
        tt_x,
        cluster_axis,
        submesh,
        ccl_manager.get_ag_ping_pong_semaphore(cluster_axis),
        use_device_op=False,
        **common_kwargs,
    )

    c_t_full = ttnn.to_torch(ttnn.get_device_tensors(out_composite)[0]).to(torch.float32)
    f_t_full = ttnn.to_torch(ttnn.get_device_tensors(out_fused)[0]).to(torch.float32)
    # Output shape: [1, num_heads_per_device, N, head_dim].
    n_dim = c_t_full.shape[2]

    # Extract effective 1/rms from output ÷ input (per-row scalar across heads).
    if not use_weight and not use_rope:
        # chip 0 input was x_full[:, :, :N, 0:H_per_device]
        x_chip0 = x_full[0, 0, :N, 0:H_per_device].to(torch.float32)  # [N, H_per_device]
        # Output is [1, nh, N, head_dim] → reshape head dim back to H_per_device
        c_out_flat = c_t_full.squeeze(0).permute(1, 0, 2).reshape(N, -1).to(torch.float32)
        f_out_flat = f_t_full.squeeze(0).permute(1, 0, 2).reshape(N, -1).to(torch.float32)
        # Per-row: 1/rms = output / input (mask zero inputs)
        mask = x_chip0.abs() > 1e-6
        c_invrms = torch.where(mask, c_out_flat / x_chip0, torch.zeros_like(c_out_flat))
        f_invrms = torch.where(mask, f_out_flat / x_chip0, torch.zeros_like(f_out_flat))
        # Average over the row (should be ~constant per row = 1/rms)
        c_invrms_per_row = c_invrms.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        f_invrms_per_row = f_invrms.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Check rows 0, 96 (chunk 0 r=0 and chunk 1 r=0 for worker 0)
        for check_row in [0, 32, 64, 96, 128, 160]:
            if check_row < N:
                logger.info(
                    f"INVRMS row={check_row}: "
                    f"composite={float(c_invrms_per_row[check_row]):.8f} "
                    f"fused={float(f_invrms_per_row[check_row]):.8f} "
                    f"diff={float((c_invrms_per_row[check_row] - f_invrms_per_row[check_row]).abs()):.2e}"
                )

        # Direct value comparison for first 8 cols of row 0 and (when present) row 96:
        logger.info(f"INPUT row 0 cols 0..7: {x_chip0[0, :8].tolist()}")
        logger.info(f"COMPOSITE OUT row 0 cols 0..7: {c_out_flat[0, :8].tolist()}")
        logger.info(f"FUSED OUT row 0 cols 0..7: {f_out_flat[0, :8].tolist()}")
        if N > 96:
            logger.info(f"INPUT row 96 cols 0..7: {x_chip0[96, :8].tolist()}")
            logger.info(f"COMPOSITE OUT row 96 cols 0..7: {c_out_flat[96, :8].tolist()}")
            logger.info(f"FUSED OUT row 96 cols 0..7: {f_out_flat[96, :8].tolist()}")

    diff_full = (c_t_full - f_t_full).abs()
    # Per-row max diff (collapse heads + head_dim):
    per_row_max = diff_full.amax(dim=(0, 1, 3))  # [N]
    # Group rows by tile (32 rows = 1 tile) and chunk (chunk_size_rows tiles = 1 chunk):
    chunk_size_rows = 3
    n_tile_rows = (n_dim + 31) // 32
    # Truncate to last full tile boundary for clean grouping
    n_full_tiles = n_dim // 32
    tile_grouped = per_row_max[: n_full_tiles * 32].reshape(n_full_tiles, 32).amax(dim=1)
    # Show first and last few tile rows + by chunk position (chunk_offset = tile_row % chunk_size_rows):
    chunk0_pos = tile_grouped[::chunk_size_rows]  # tile rows at chunk position 0
    chunk1_pos = tile_grouped[1::chunk_size_rows]  # chunk position 1
    chunk2_pos = tile_grouped[2::chunk_size_rows]  # chunk position 2

    def _mean_max(t):  # empty slices occur when N spans < chunk_size_rows tiles
        return (float(t.mean()), float(t.max())) if t.numel() else (float("nan"), float("nan"))

    c0_mean, c0_max = _mean_max(chunk0_pos)
    c1_mean, c1_max = _mean_max(chunk1_pos)
    c2_mean, c2_max = _mean_max(chunk2_pos)
    logger.info(
        f"BISECT N={N} per-tile-row max diff: "
        f"first_18={[f'{x:.3f}' for x in tile_grouped[:18].tolist()]} "
        f"chunk_pos0 mean={c0_mean:.4f} max={c0_max:.4f} "
        f"chunk_pos1 mean={c1_mean:.4f} max={c1_max:.4f} "
        f"chunk_pos2 mean={c2_mean:.4f} max={c2_max:.4f}"
    )
    c_t = c_t_full.flatten()
    f_t = f_t_full.flatten()
    diff = (c_t - f_t).abs()
    max_d = float(diff.max())
    cov = torch.cov(torch.stack([c_t, f_t]))
    std_c = float(cov[0, 0].sqrt())
    std_f = float(cov[1, 1].sqrt())
    pcc_cf = float(cov[0, 1]) / (std_c * std_f) if std_c * std_f > 0 else float("nan")
    n_diff = int((diff > 1e-3).sum())

    logger.info(
        f"BISECT N={N} use_weight={use_weight} use_rope={use_rope} | "
        f"PCC={pcc_cf * 100:.6f}% max|diff|={max_d:.4e} n_diff>1e-3={n_diff}/{c_t.numel()} "
        f"c_range=[{float(c_t.min()):.3f},{float(c_t.max()):.3f}] "
        f"f_range=[{float(f_t.min()):.3f},{float(f_t.max()):.3f}]"
    )

    # Pre+AG bit-exactness check at THIS N: compare composite's gathered stats
    # tensor (col 0 = sum-of-squares) against fused's persistent_output_buffer
    # (packed-page format: 32 fp32 col-0 values per device per chunk).
    # Only meaningful when the fused op used the multi-worker MUX path (which
    # populates the persistent buffer); single-worker shapes leave it None.
    if not use_weight and not use_rope and persistent_output_buffer is not None:
        composite_stats = ttnn.experimental.wan_fused_rmsnorm_pre_allgather(tt_x, dtype=ttnn.float32)
        ring_size = 4
        composite_stats_gathered = ccl_manager.all_gather_persistent_buffer(
            composite_stats, dim=3, mesh_axis=cluster_axis
        )
        cs_full = ttnn.to_torch(ttnn.get_device_tensors(composite_stats_gathered)[0]).to(torch.float32)
        # Shape after AG: [1, 1, N_padded, 32*ring_size]. Col 0 of each
        # ring_size-th band is the per-row sum from each device.
        cs = cs_full.squeeze(0).squeeze(0)  # [N_padded, 32*ring_size]
        # Composite col-0 stats: indices [0, 32, 64, 96] across ring_size devices.
        composite_col0 = cs[:, ::32]  # [N_padded, ring_size]

        # Fused persistent_output_buffer: packed-page format. Layout per chip:
        #   ring_size pages stacked × num_chunks_per_device chunks
        # Each page = chunk_size_rows × 32 fp32 (the writer-extracted col-0).
        # Total = ring_size × num_chunks_per_device × chunk_size_rows × 32 fp32.
        pob_t = ttnn.to_torch(ttnn.get_device_tensors(persistent_output_buffer)[0]).to(torch.float32)
        ring = 4
        # Each fp32 in the 32-block is one input row's stat. Page (d, c) holds
        # chunk_size_rows tiles × 32 rows/tile = chunk_size_rows*32 input rows.
        # pob layout: [pages_total, page_floats] = [ring*num_chunks, chunk_rows*32].
        pob_flat = pob_t.reshape(-1)
        rows_per_device = pob_flat.numel() // ring
        fused_per_device = pob_flat.reshape(ring, rows_per_device)  # [ring, rows_per_dev]
        n_compare = min(composite_col0.shape[0], rows_per_device)

        # [DEBUG] OWN-device pob slot now contains writer's packed_local content
        # (from the debug noc_async_write added in writer_mux.cpp). Compare
        # chunk 0 r=0 vs chunk 1 r=0's packed_local from CHIP 0.
        # packed_local layout: chunk_size_rows × 128 bytes (32 fp32 col-0).
        chunk_size_rows = 3
        packed_local_chip0 = fused_per_device[0]  # device 0 = chip 0's own data
        # This packed_local debug dump assumes the legacy chunk_size_rows=3
        # packing with at least 3 chunks; skip it when the shape doesn't fit.
        num_chunks_per_dev = packed_local_chip0.numel() // (chunk_size_rows * 32)
        if packed_local_chip0.numel() % (chunk_size_rows * 32) == 0 and num_chunks_per_dev >= 3:
            # Reshape to [num_chunks, chunk_size_rows, 32]
            packed_reshape = packed_local_chip0.reshape(num_chunks_per_dev, chunk_size_rows, 32)
            # For each chunk, row 0's first 4 fp32 values are stats for first 4 input rows in tile
            logger.info(f"PACKED_LOCAL chip0 chunk 0 r=0 first 4 stats: {packed_reshape[0, 0, :4].tolist()}")
            logger.info(f"PACKED_LOCAL chip0 chunk 1 r=0 first 4 stats: {packed_reshape[1, 0, :4].tolist()}")
            logger.info(f"PACKED_LOCAL chip0 chunk 2 r=0 first 4 stats: {packed_reshape[2, 0, :4].tolist()}")
            # Diff between chunk 0 r=0 and chunk 1 r=0
            diff_packed = (packed_reshape[0, 0] - packed_reshape[1, 0]).abs()
            logger.info(
                f"PACKED_LOCAL chunk 0 vs chunk 1 r=0 diff: "
                f"max={float(diff_packed.max()):.3e} mean={float(diff_packed.mean()):.3e} "
                f"identical_count={int((diff_packed == 0).sum())}/32"
            )

            # ALL devices' packed_local. Note: chip 0's OWN page is now OVERWRITTEN
            # by [DEBUG2]'s dump of stats_gathered_cb's first 3 tiles col-0 values.
            # Remote devices' pages still have their original packed_local from fabric.
            all_packed = pob_flat.reshape(ring, num_chunks_per_dev, chunk_size_rows, 32)

            # [DEBUG2 interpretation]: own slot (d=0) for each chunk now holds
            # stats_gathered_cb tiles 0,1,2 col-0 (first row stat per dev 0,1,2).
            # That means own_slot[chunk_idx][0..2] = stats_gathered[d=0,1,2] r=0 row 0.
            for chunk_idx in [0, 1, 2]:
                gathered_d0 = all_packed[0, chunk_idx, 0, 0]  # stats_gathered_cb tile 0 = dev 0 stat
                gathered_d1 = all_packed[0, chunk_idx, 1, 0]  # stats_gathered_cb tile 1 = dev 1 stat
                gathered_d2 = all_packed[0, chunk_idx, 2, 0]  # stats_gathered_cb tile 2 = dev 2 stat
                # Remote chip 1's packed_local (still fabric-correct, since chip 1's debug write
                # was to ITS own slot in ITS own pob, which is page 130+chunk_idx; chip 0 reads it via fabric)
                remote_d1 = all_packed[1, chunk_idx, 0, 0]
                remote_d2 = all_packed[2, chunk_idx, 0, 0]
                remote_d3 = all_packed[3, chunk_idx, 0, 0]
                logger.info(
                    f"PER-CHUNK chunk_idx={chunk_idx} r=0: "
                    f"stats_gathered[d0,d1,d2]={[gathered_d0.item(), gathered_d1.item(), gathered_d2.item()]} "
                    f"packed_remote[d1,d2,d3]={[remote_d1.item(), remote_d2.item(), remote_d3.item()]}"
                )
        # Composite is [N, ring], fused_per_device is [ring, rows_per_dev]. Transpose.
        fused_per_row = fused_per_device[:, :n_compare].transpose(0, 1)  # [n_compare, ring]
        diff_preag = (composite_col0[:n_compare] - fused_per_row).abs()
        max_preag = float(diff_preag.max())
        nonzero_preag = int((diff_preag > 0).sum())
        # Look at REMOTE-device diffs only (own slot in fused is uninit).
        # Skip device 0 (chip 0's own page); compare devices 1..3:
        diff_remote = (composite_col0[:n_compare, 1:] - fused_per_row[:, 1:]).abs()
        max_remote = float(diff_remote.max())
        # Per-row max diff (across remote devices)
        per_row_diff = diff_remote.amax(dim=1)  # [n_compare]
        bad_rows = (per_row_diff > 0).nonzero(as_tuple=True)[0]
        # Check chunk 1's first row (input row 96 = chunk 1 r=0 for worker 0):
        check_rows = [0, 32, 64, 96, 128, 160]  # row 0, then chunk 1 r=0 at row 96
        diffs_at_rows = [float(per_row_diff[r]) if r < n_compare else -1.0 for r in check_rows]
        logger.info(
            f"PRE+AG CHECK N={N}: max|remote_diff|={max_remote:.4e} "
            f"bad_rows_count={bad_rows.numel()} "
            f"diffs at rows {check_rows}={diffs_at_rows}"
        )


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
    # Required for multi-worker shapes (returns None when a single worker is used).
    persistent_output_buffer = ttnn.experimental.wan_fused_distributed_rmsnorm_create_stats_buffer(
        tt_x, cluster_axis, submesh, num_heads_per_device=NUM_HEADS_PER_DEVICE
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
        persistent_output_buffer=persistent_output_buffer,
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
        # Repro of the tp2_a_selfattn compute shape (feat 1024, 16 heads, head_dim
        # 64, 64 rows) in isolation at TP=1 (no AG/ring) — isolates compute vs AG.
        pytest.param(64, 1024, 16, True, id="N64_H1024_16heads_perhead"),
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
