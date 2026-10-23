# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for distributed_rms_norm (immutable spec — do not modify).

distributed_rms_norm is a multi-device op: every device (r, c) holds a (..., M, W_local) slice of each row's
hidden dim; the G devices along `cluster_axis` together hold the full row W = G * W_local, and device p's output
is its slice of torch.nn.functional.rms_norm(concat_p x[p], (W,), concat_p gamma[p], eps).

The single-chip `device` fixture cannot run a Fabric op, so this module uses the golden suite's fabric mesh
fixture (`mesh_device`, opened once per mesh shape with FABRIC_2D_TORUS_XY and the production router payload;
a test may pin `mesh_shape` to open the box's snake ring as 1xN / Nx1). No device is opened manually here.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

# Fabric mesh fixtures (session cache + per-test mesh) are registered once in this directory's conftest.py.
from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_gamma, create_ttnn_input_tensor
from eval.golden_tests.distributed_rms_norm.topology import ring_mock_meshes, system_mesh_shape, topology_feasible
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# single tile, multi-tile, non-square decode (Kimi TP4), multi-batch rank 4, perf-focus prefill, rank 3, rank 2
SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (1, 1, 32, 1792),
    (2, 4, 64, 256),
    (1, 1, 640, 1792),
    (1, 128, 512),
    (64, 256),
]

LINEAR_AXES = [a for a in (0, 1) if topology_feasible(ttnn.Topology.Linear, a) is None]
RING_AXES = [a for a in (0, 1) if topology_feasible(ttnn.Topology.Ring, a) is None]


# --- reference ---------------------------------------------------------------


def reference(x_stacked, gamma_stacked, cluster_axis, eps):
    """x_stacked (R, C, *shape); gamma_stacked (R, C, W_local) or None. Pure torch, fp32."""
    x = x_stacked.to(torch.float32)
    rows, cols = x.shape[0], x.shape[1]
    w_local = x.shape[-1]
    out = torch.empty_like(x)
    for other in range(cols if cluster_axis == 0 else rows):
        members = [(r, other) for r in range(rows)] if cluster_axis == 0 else [(other, c) for c in range(cols)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        weight = None
        if gamma_stacked is not None:
            weight = torch.cat([gamma_stacked[r, c].to(torch.float32).reshape(-1) for r, c in members])
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=weight, eps=eps)
        for p, (r, c) in enumerate(members):
            out[r, c] = normed[..., p * w_local : (p + 1) * w_local]
    return out


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return 1.0 if den == 0 else float((a * b).sum() / den)


def run_case(
    mesh,
    shape,
    dtype,
    cluster_axis,
    *,
    use_gamma,
    gamma_dtype=ttnn.bfloat16,
    eps=1e-6,
    topology=ttnn.Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
):
    torch.manual_seed(42)
    rows, cols = tuple(mesh.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    gamma = None
    tg = None
    if use_gamma:
        gamma = torch.randn((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tg = create_ttnn_gamma(gamma, mesh, dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT)
    tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    kwargs = dict(cluster_axis=cluster_axis, gamma=tg, epsilon=eps, topology=topology)
    if num_links is not None:
        kwargs["num_links"] = num_links
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config
    out = distributed_rms_norm(tx, **kwargs)

    assert list(out.shape) == list(shape)
    assert out.dtype == dtype
    assert out.layout == ttnn.TILE_LAYOUT

    expected = reference(x, None if gamma is None else gamma.reshape(rows, cols, shape[-1]), cluster_axis, eps)
    device_tensors = ttnn.get_device_tensors(out)
    assert len(device_tensors) == rows * cols
    for idx, t in enumerate(device_tensors):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(t).to(torch.float32).reshape(shape)
        got = pcc(actual, expected[r, c])
        assert not torch.isnan(actual).any(), f"device ({r},{c}) produced NaN"
        assert got >= PCC[dtype], f"device ({r},{c}): pcc {got:.6f} < {PCC[dtype]}"
    return out


# --- correctness over shapes / dtypes / gamma, Linear on the system mesh -------


@pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")
@pytest.mark.parametrize("cluster_axis", LINEAR_AXES)
@pytest.mark.parametrize("use_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_distributed_rms_norm(shape, dtype, use_gamma, cluster_axis, mesh_device):
    run_case(mesh_device, shape, dtype, cluster_axis, use_gamma=use_gamma)


@pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")
@pytest.mark.parametrize("gamma_dtype", [ttnn.float32, ttnn.bfloat16], ids=["g_fp32", "g_bf16"])
def test_mixed_gamma_dtype_and_epsilon(gamma_dtype, mesh_device):
    """bf16 activations with an fp32 (or bf16) weight, custom epsilon, explicit maxed-out compute config."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    run_case(
        mesh_device,
        (1, 1, 64, 512),
        ttnn.bfloat16,
        LINEAR_AXES[-1],
        use_gamma=True,
        gamma_dtype=gamma_dtype,
        eps=1e-5,
        compute_kernel_config=cfg,
    )


@pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")
@pytest.mark.parametrize("num_links", [1, 2])
def test_num_links(num_links, mesh_device):
    run_case(mesh_device, (1, 1, 640, 1792), ttnn.bfloat16, LINEAR_AXES[-1], use_gamma=True, num_links=num_links)


# --- Ring: real torus axes, else the box's snake ring opened as 1xN / Nx1 ------

_RING_CASES = [pytest.param(None, a, id=f"system-axis{a}") for a in RING_AXES]
_RING_CASES += [
    pytest.param(shape_, axis, id=f"mock{shape_[0]}x{shape_[1]}-axis{axis}") for shape_, axis in ring_mock_meshes()
]


@pytest.mark.skipif(not _RING_CASES, reason="no ring this cluster can close")
@pytest.mark.parametrize("mesh_shape,cluster_axis", _RING_CASES)
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 1792), (1, 1, 640, 1792), (1, 1, 64, 256)], ids=lambda s: "x".join(map(str, s))
)
def test_ring(shape, mesh_shape, cluster_axis, mesh_device):
    run_case(mesh_device, shape, ttnn.bfloat16, cluster_axis, use_gamma=True, topology=ttnn.Topology.Ring)


# --- determinism / back-to-back calls -----------------------------------------


@pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")
def test_back_to_back_calls_bit_identical(mesh_device):
    """Two calls with no host sync in between must give bit-identical results (ready fence, re-armed counters)."""
    torch.manual_seed(42)
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 128, 1024)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    a = distributed_rms_norm(tx, cluster_axis=LINEAR_AXES[-1])
    b = distributed_rms_norm(tx, cluster_axis=LINEAR_AXES[-1])
    for ta, tb in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)):
        assert torch.equal(ttnn.to_torch(ta), ttnn.to_torch(tb))


# --- caller errors ---------------------------------------------------------------


@pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")
def test_bad_gamma_shape_raises(mesh_device, expect_error):
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 32, 256)
    x = torch.randn((rows, cols, *shape)).to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    bad = torch.randn((rows, cols, 1, 1, 1, 128)).to(torch.bfloat16)  # W_local mismatch
    tg = create_ttnn_input_tensor(bad, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), r"(?i)gamma"):
        distributed_rms_norm(tx, cluster_axis=LINEAR_AXES[-1], gamma=tg)


@pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")
def test_too_many_links_raises(mesh_device, expect_error):
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, 1, 1, 32, 256)).to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), r"(?i)link"):
        distributed_rms_norm(tx, cluster_axis=LINEAR_AXES[-1], num_links=64)


@pytest.mark.parametrize("cluster_axis", [a for a in (0, 1) if system_mesh_shape()[a] == 1] or [None])
def test_size_one_axis_raises(cluster_axis, mesh_device, expect_error):
    if cluster_axis is None:
        pytest.skip("both mesh axes have size >= 2")
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, 1, 1, 32, 256)).to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), r"(?i)cluster_axis"):
        distributed_rms_norm(tx, cluster_axis=cluster_axis)
