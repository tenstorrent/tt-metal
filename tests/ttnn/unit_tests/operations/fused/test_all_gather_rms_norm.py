# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test harness for the generic fused ``ttnn.all_gather_rms_norm`` op.

``all_gather_rms_norm`` fuses, into a single (multi-device) op:
    per-device partial stats (E[x^2] over the local shard of the reduction dim)
      -> cross-device all-gather of the stats over ``cluster_axis``
      -> post-normalization ``x / sqrt(E[x^2] + eps) * gamma + beta``
with optional weight (gamma) and optional bias (beta).

It is the single-op replacement for the
``rms_norm_pre_all_gather`` -> ``all_gather_async`` -> ``rms_norm_post_all_gather`` sequence.

Coverage:
  * ``test_*_single_device``  : the fused compute math (ring_size == 1, no fabric), many shapes.
  * ``test_*_multi_device``   : the fabric stats all-gather through the mux (ring_size > 1) on a 1x8 ring,
                                many shapes including edge cases (odd NCHt -> zero-row worker cores,
                                multi-chunk gathers, small/large widths).
  * ``test_*_accuracy``       : tight accuracy (high PCC + output-magnitude ratio + allclose) on the
                                scale-sensitive gamma_beta case. The magnitude-ratio check guards against
                                the "PCC is scale-invariant" trap (a uniformly mis-scaled output still
                                scores high PCC but is numerically wrong).
  * ``test_*_program_cache``  : descriptor cache-hit (Buffer-binding) fast path, single- and multi-device.

Set ``ALL_GATHER_RMS_NORM_RUN=1`` to dispatch on device (skipped by default so CI without the op built
does not error).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose

pytestmark = pytest.mark.skipif(
    not os.environ.get("ALL_GATHER_RMS_NORM_RUN"),
    reason="Set ALL_GATHER_RMS_NORM_RUN=1 to dispatch all_gather_rms_norm on device.",
)

FEATURE_COMBOS = [(False, False), (True, False), (True, True)]
FEATURE_IDS = ["plain", "gamma", "gamma_beta"]


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
    has_residual=False,
    dtype=ttnn.bfloat16,
    pcc=0.99,
    mag_tol=0.05,
    allclose_rtol=None,
    allclose_atol=None,
    seed=1234,
):
    """End-to-end harness: build sharded inputs, run the fused op, compare to a torch golden.

    Always checks PCC (>= ``pcc``) and the output-magnitude ratio (within ``mag_tol`` of 1.0 -- catches
    uniform-scale errors that PCC misses). When ``allclose_rtol``/``allclose_atol`` are given, also asserts
    an element-wise allclose. Returns the (pcc_passed, mag_ratio)."""
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
    actual = ttnn_output_torch.to(torch_output.dtype)

    # Functional check: PCC (raises on failure).
    pcc_passed, pcc_msg = assert_with_pcc(torch_output, actual, pcc=pcc)

    # Accuracy diagnostics: max abs error + overall output-magnitude ratio.
    max_abs_err = (actual.float() - torch_output.float()).abs().max().item()
    golden_mag = torch_output.float().abs().mean().clamp_min(1e-9)
    mag_ratio = (actual.float().abs().mean() / golden_mag).item()
    logger.info(
        f"all_gather_rms_norm[{batch_size}x{seq_len}x{hidden_dim} dev={num_devices} "
        f"w={int(has_weight)} b={int(has_bias)} r={int(has_residual)}] "
        f"PCC={pcc_msg} | max_abs_err={max_abs_err:.4f} | mag_ratio={mag_ratio:.4f}"
    )

    # Magnitude guard: a uniformly mis-scaled output passes PCC but is wrong. Always enforced.
    assert (
        abs(mag_ratio - 1.0) <= mag_tol
    ), f"output magnitude off (possible scale bug): ratio={mag_ratio:.4f}, tol={mag_tol}"

    # Optional stricter element-wise accuracy assertion.
    if allclose_rtol is not None:
        ac_passed, ac_msg = comp_allclose(torch_output, actual, rtol=allclose_rtol, atol=allclose_atol)
        assert ac_passed, f"allclose failed (rtol={allclose_rtol}, atol={allclose_atol}): {ac_msg}"

    return pcc_passed, mag_ratio


# ---------------------------------------------------------------------------------------------------------
# Single device (1x1): the fused compute math (pre-reduce -> rsqrt -> normalize -> gamma/beta), no fabric.
# Sweep widths {1024, 2048, 4096} x sequence lengths giving NCHt in {1, 3, 32, 128} (incl. the L1-tight
# full DiT width Wt=128 at hidden=4096) x {plain, gamma, gamma_beta}.
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 96, 1024, 4096], ids=["seq32", "seq96", "seq1k", "seq4k"])
@pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096], ids=["h1024", "h2048", "h4096"])
@pytest.mark.parametrize("has_weight, has_bias", FEATURE_COMBOS, ids=FEATURE_IDS)
def test_all_gather_rms_norm_single_device(mesh_device, seq_len, hidden_dim, has_weight, has_bias):
    run_all_gather_rms_norm(
        mesh_device,
        batch_size=1,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        cluster_axis=1,
        eps=1e-6,
        has_weight=has_weight,
        has_bias=has_bias,
    )


# ---------------------------------------------------------------------------------------------------------
# Multi device (1x8 ring): the fabric stats all-gather through the mux. Curated (seq_len, hidden_dim) pairs:
#   (96, 1024)   small: NCHt=3 -> 3 worker cores, 1 row each.
#   (288, 2048)  odd NCHt=9 -> 8 worker cores incl. ZERO-row cores (exercises mux-connect/teardown when a
#                core has no rows) + a gather chunk remainder.
#   (1024, 1024) / (1024, 2048) / (1024, 4096)  baseline widths (local Wt = 4 / 8 / 16).
#   (4096, 2048) large NCHt=128 -> 16 rows/core -> MULTI-chunk batched gather (gather_chunk=8).
# All widths divisible by 32*8=256. 1x8 only: smaller line submeshes don't train fabric on this host.
# ---------------------------------------------------------------------------------------------------------
MULTI_SHAPES = [
    (96, 1024),
    (288, 2048),
    (1024, 1024),
    (1024, 2048),
    (1024, 4096),
    (4096, 2048),
]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], ids=["1x8"], indirect=True)
@pytest.mark.parametrize("seq_len, hidden_dim", MULTI_SHAPES, ids=[f"seq{s}_h{h}" for s, h in MULTI_SHAPES])
@pytest.mark.parametrize("has_weight, has_bias", FEATURE_COMBOS, ids=FEATURE_IDS)
def test_all_gather_rms_norm_multi_device(mesh_device, seq_len, hidden_dim, has_weight, has_bias):
    if mesh_device.get_num_devices() < tuple(mesh_device.shape)[1]:
        pytest.skip("not enough devices for this mesh")
    run_all_gather_rms_norm(
        mesh_device,
        batch_size=1,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        cluster_axis=1,
        eps=1e-6,
        has_weight=has_weight,
        has_bias=has_bias,
    )


# ---------------------------------------------------------------------------------------------------------
# Accuracy: tight PCC + element-wise allclose + magnitude ratio on the scale-sensitive gamma_beta case.
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len, hidden_dim", [(1024, 2048), (1024, 4096)], ids=["seq1k_h2048", "seq1k_h4096"])
def test_all_gather_rms_norm_single_device_accuracy(mesh_device, seq_len, hidden_dim):
    run_all_gather_rms_norm(
        mesh_device,
        batch_size=1,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        cluster_axis=1,
        eps=1e-6,
        has_weight=True,
        has_bias=True,
        pcc=0.999,
        mag_tol=0.02,
        allclose_rtol=0.1,
        allclose_atol=0.1,
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], ids=["1x8"], indirect=True)
@pytest.mark.parametrize("seq_len, hidden_dim", [(1024, 2048), (1024, 4096)], ids=["seq1k_h2048", "seq1k_h4096"])
def test_all_gather_rms_norm_multi_device_accuracy(mesh_device, seq_len, hidden_dim):
    if mesh_device.get_num_devices() < tuple(mesh_device.shape)[1]:
        pytest.skip("not enough devices for this mesh")
    run_all_gather_rms_norm(
        mesh_device,
        batch_size=1,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        cluster_axis=1,
        eps=1e-6,
        has_weight=True,
        has_bias=True,
        pcc=0.999,
        mag_tol=0.02,
        allclose_rtol=0.1,
        allclose_atol=0.1,
    )


# ---------------------------------------------------------------------------------------------------------
# Program cache: run twice to exercise the descriptor cache-hit (Buffer-binding) fast path, on both the
# single-device and multi-device (mux) paths.
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_all_gather_rms_norm_program_cache(mesh_device):
    for _ in range(2):
        run_all_gather_rms_norm(
            mesh_device,
            batch_size=1,
            seq_len=1024,
            hidden_dim=2048,
            cluster_axis=1,
            eps=1e-6,
            has_weight=True,
            has_bias=False,
        )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], ids=["1x8"], indirect=True)
def test_all_gather_rms_norm_multi_device_program_cache(mesh_device):
    if mesh_device.get_num_devices() < tuple(mesh_device.shape)[1]:
        pytest.skip("not enough devices for this mesh")
    for _ in range(2):
        run_all_gather_rms_norm(
            mesh_device,
            batch_size=1,
            seq_len=1024,
            hidden_dim=2048,
            cluster_axis=1,
            eps=1e-6,
            has_weight=True,
            has_bias=True,
        )
