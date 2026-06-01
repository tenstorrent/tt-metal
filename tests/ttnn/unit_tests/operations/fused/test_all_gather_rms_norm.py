# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test harness for the generic fused ``ttnn.all_gather_rms_norm`` op.

``all_gather_rms_norm`` fuses, into a single multi-device op:
    per-device partial stats (E[x^2] over the local shard of the reduction dim)
      -> cross-device all-gather of the stats over ``cluster_axis``
      -> post-normalization ``x / sqrt(E[x^2] + eps) * gamma + beta``
with optional weight (gamma), optional bias (beta) and optional fused residual add.

It is the single-op replacement for the
``rms_norm_pre_all_gather`` -> ``all_gather_async`` -> ``rms_norm_post_all_gather``
sequence (see ``distributed_norm_test_utils.compute_ttnn_distributed_norm``).

NOTE: the device kernels (LLKs) for this op are currently STUBS, so running the op on device
will not produce correct results (and may hang on the unimplemented fabric/semaphore handshake).
This module is therefore skipped by default; set ``ALL_GATHER_RMS_NORM_RUN=1`` to actually dispatch
it once the kernels are implemented. The harness (golden, sharding, semaphore setup, PCC compare)
is complete and ready for that point.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# The op's compute/dataflow kernels are stubs (LLK math is TODO). Skip on-device dispatch by default
# so CI does not hang; flip this env var once the kernels are implemented.
pytestmark = pytest.mark.skipif(
    not os.environ.get("ALL_GATHER_RMS_NORM_RUN"),
    reason=(
        "all_gather_rms_norm device kernels are stubs (LLK math TODO). "
        "Set ALL_GATHER_RMS_NORM_RUN=1 to run on device once they are implemented."
    ),
)


def torch_rms_norm(x, weight, bias, eps):
    """Reference RMSNorm over the last dim, matching the op's math."""
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.type_as(x)


def _make_ccl_semaphore(mesh_device):
    """Create a single global semaphore over the full compute grid for the all-gather handshake."""
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_global_semaphore(mesh_device, ccl_crs, 0)


def run_all_gather_rms_norm(
    mesh_device,
    *,
    batch_size,
    seq_len,
    hidden_dim,
    cluster_axis,
    eps,
    has_weight,
    has_bias,
    has_residual,
    dtype=ttnn.bfloat16,
    pcc=0.99,
    seed=1234,
):
    """End-to-end harness: build sharded inputs, run the fused op, compare to a torch golden."""
    num_devices = tuple(mesh_device.shape)[cluster_axis]
    assert hidden_dim % (32 * num_devices) == 0, (
        f"hidden_dim ({hidden_dim}) must be divisible by 32 * num_devices ({32 * num_devices}); "
        "the reduction dim is sharded across the cluster axis."
    )

    torch.manual_seed(seed)
    input_shape = (batch_size, 1, seq_len, hidden_dim)

    torch_input = (torch.randn(input_shape) * 4 - 1).to(torch.bfloat16)
    torch_residual = (torch.randn(input_shape) * 2).to(torch.bfloat16) if has_residual else None
    torch_weight = (torch.rand(hidden_dim) * 2 - 1).to(torch.bfloat16) if has_weight else None
    torch_bias = (torch.rand(hidden_dim) * 2 - 1).to(torch.bfloat16) if has_bias else None

    # Golden (residual is added before the norm, matching the fused FUSE_PRE_ADD path).
    golden_input = torch_input.float() + (torch_residual.float() if has_residual else 0.0)
    torch_output = torch_rms_norm(golden_input, torch_weight, torch_bias, eps)

    # Shard input/residual/gamma/beta on the reduction (last) dim across the cluster axis, so each device
    # holds its local hidden_dim/num_devices slice. The fused op gathers only the stats; the output stays
    # sharded on the last dim, so we reassemble with ConcatMeshToTensor(dim=-1).
    def to_dev(t):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )

    ttnn_input = to_dev(torch_input)
    ttnn_residual = to_dev(torch_residual) if has_residual else None
    ttnn_weight = to_dev(torch_weight.reshape(1, 1, 1, hidden_dim)) if has_weight else None
    ttnn_bias = to_dev(torch_bias.reshape(1, 1, 1, hidden_dim)) if has_bias else None

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    global_semaphore = _make_ccl_semaphore(mesh_device)
    ttnn.synchronize_device(mesh_device)

    ttnn_output = ttnn.all_gather_rms_norm(
        ttnn_input,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        global_semaphore=global_semaphore,
        weight=ttnn_weight,
        bias=ttnn_bias,
        epsilon=eps,
        residual_input_tensor=ttnn_residual,
        topology=ttnn.Topology.Linear,
        num_links=1,
        compute_kernel_config=compute_kernel_config,
    )

    # Output stays sharded on the last dim; concat the per-device slices back to the full hidden_dim.
    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    passing, pcc_msg = assert_with_pcc(torch_output, ttnn_output_torch.to(torch_output.dtype), pcc=pcc)
    logger.info(f"all_gather_rms_norm PCC: {pcc_msg}")
    return passing


# Single-device (1x1) path validates the fused compute math (pre-reduce -> rsqrt -> normalize -> gamma/beta)
# with no fabric. This is the only path that computes correctly until the ring_size > 1 stats all-gather
# (fabric) is implemented in the writer/compute kernels. Residual fusion (FUSE_PRE_ADD) is also not wired
# up yet, so it is not parametrized here.
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [1024, 4096], ids=["seq1k", "seq4k"])
# Full DiT width: the compute kernel fuses normalize -> gamma -> beta per block, so x_normed/gamma_out are
# block-sized and the fused single-kernel pre+post fits L1 even at Wt=128.
@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize(
    "has_weight, has_bias",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
    ids=["plain", "gamma", "gamma_beta"],
)
def test_all_gather_rms_norm_single_device(mesh_device, seq_len, hidden_dim, has_weight, has_bias):
    passing = run_all_gather_rms_norm(
        mesh_device,
        batch_size=1,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        cluster_axis=1,
        eps=1e-6,
        has_weight=has_weight,
        has_bias=has_bias,
        has_residual=False,
    )
    assert passing, "all_gather_rms_norm output did not match the torch RMSNorm golden"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_all_gather_rms_norm_program_cache(mesh_device):
    """Run twice to exercise the descriptor cache-hit (Buffer-binding) fast path."""
    for _ in range(2):
        passing = run_all_gather_rms_norm(
            mesh_device,
            batch_size=1,
            seq_len=1024,
            hidden_dim=2048,
            cluster_axis=1,
            eps=1e-6,
            has_weight=True,
            has_bias=False,
            has_residual=False,
        )
        assert passing


# Multi-device path (cluster_axis=1, reduction dim sharded across the ring). Exercises the fabric stats
# all-gather (ring_size > 1) in the writer/compute kernels. Fabric must be enabled via device_params.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
# Use the full 1x8 ring: smaller line submeshes (1x2/1x4) don't train fabric on this host.
@pytest.mark.parametrize("mesh_device", [(1, 8)], ids=["1x8"], indirect=True)
@pytest.mark.parametrize("hidden_dim", [2048])
@pytest.mark.parametrize(
    "has_weight, has_bias",
    [(False, False), (True, False), (True, True)],
    ids=["plain", "gamma", "gamma_beta"],
)
def test_all_gather_rms_norm_multi_device(mesh_device, hidden_dim, has_weight, has_bias):
    if mesh_device.get_num_devices() < tuple(mesh_device.shape)[1]:
        pytest.skip("not enough devices for this mesh")
    passing = run_all_gather_rms_norm(
        mesh_device,
        batch_size=1,
        seq_len=1024,
        hidden_dim=hidden_dim,
        cluster_axis=1,
        eps=1e-6,
        has_weight=has_weight,
        has_bias=has_bias,
        has_residual=False,
    )
    assert passing, "all_gather_rms_norm output did not match the torch RMSNorm golden"
