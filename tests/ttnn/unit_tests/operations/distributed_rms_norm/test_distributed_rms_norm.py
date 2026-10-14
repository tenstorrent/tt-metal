# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for distributed_rms_norm (Phase 0). IMMUTABLE: the implementer must not edit this file.

Multi-device CCL op: every device (r, c) of the mesh holds its own x[r, c] of per-device shape (..., M, W_local).
The G devices along `cluster_axis` together hold the full hidden dim W = G * W_local; the op must equal
torch.nn.functional.rms_norm over the concatenated row, each device keeping its own slice.

The mesh is opened through the root `mesh_device` fixture (indirect `device_params` carrying the fabric config the
golden suite uses: FABRIC_2D_TORUS_XY + the production router payload). The `device` fixture cannot be used here —
a fabric-enabled mesh is required.

Shapes pin the Phase 0 regime-selection cases of op_design.md:
  (1, 1, 32, 64)      single tile row, S_w = 1 (one worker)
  (1, 1, 32, 1792)    decode: S_w > 1, one combine tile
  (1, 1, 640, 1792)   PERF FOCUS: rows x W-chunks
  (2, 4, 128, 512)    multi-batch (rank 4)
  (1, 2048, 1280)     rank 3, multi-row row groups
  (32, 4096)          rank 2, wide: G * S_w > 32 -> several combine tiles
"""

import pytest
import torch
import ttnn

from ttnn.operations.distributed_rms_norm import distributed_rms_norm, default_compute_kernel_config

# --- mesh / fabric setup -------------------------------------------------------------------------------------------


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    try:
        is_bh = "blackhole" in ttnn.get_arch_name()
    except Exception:  # noqa: BLE001
        is_bh = True
    cfg.max_packet_payload_size_bytes = (14 if is_bh else 7) * 1024 + 64
    return cfg


SYSTEM_MESH = _system_mesh_shape()
FABRIC_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "fabric_router_config": _router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
}
_MESH = pytest.mark.parametrize("mesh_device", [SYSTEM_MESH], indirect=True)
_FABRIC = pytest.mark.parametrize("device_params", [FABRIC_PARAMS], indirect=True)

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


# --- reference & helpers -------------------------------------------------------------------------------------------


def reference(x_stacked, gamma_stacked, cluster_axis, eps):
    """x_stacked: (R, C, *shape); gamma_stacked: (R, C, 1, 1, 1, W_local) or None. fp32 torch reference."""
    x = x_stacked.to(torch.float32)
    g = None if gamma_stacked is None else gamma_stacked.to(torch.float32)
    rows, cols = x.shape[0], x.shape[1]
    w_local = x.shape[-1]
    out = torch.empty_like(x)
    for other in range(cols if cluster_axis == 0 else rows):
        members = [(r, other) for r in range(rows)] if cluster_axis == 0 else [(other, c) for c in range(cols)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        weight = None if g is None else torch.cat([g[r, c].reshape(-1) for r, c in members])
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=weight, eps=eps)
        for p, (r, c) in enumerate(members):
            out[r, c] = normed[..., p * w_local : (p + 1) * w_local]
    return out


def to_mesh(stacked, mesh_device, *, dtype, layout):
    rows, cols = stacked.shape[0], stacked.shape[1]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


def pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return 1.0 if den == 0 else float((a * b).sum() / den)


def per_device(out):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]


def skip_if_axis_trivial(mesh_device, cluster_axis):
    if tuple(mesh_device.shape)[cluster_axis] < 2:
        pytest.skip(f"cluster_axis={cluster_axis} has size 1 on mesh {tuple(mesh_device.shape)}")


def make_inputs(mesh_device, shape, dtype, gamma_dtype):
    torch.manual_seed(42)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    g = None
    if gamma_dtype is not None:
        g = torch.randn((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
    return x, g


def run_and_check(mesh_device, shape, *, dtype, gamma_dtype, cluster_axis, eps=1e-6, **op_kwargs):
    skip_if_axis_trivial(mesh_device, cluster_axis)
    x, g = make_inputs(mesh_device, shape, dtype, gamma_dtype)
    expected = reference(x, g, cluster_axis, eps)
    tx = to_mesh(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = None if g is None else to_mesh(g, mesh_device, dtype=gamma_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = distributed_rms_norm(tx, cluster_axis=cluster_axis, gamma=tg, epsilon=eps, **op_kwargs)

    assert list(out.shape) == list(shape)
    assert out.dtype == dtype
    assert out.layout == ttnn.TILE_LAYOUT
    cols = tuple(mesh_device.shape)[1]
    outs = per_device(out)
    assert len(outs) == x.shape[0] * x.shape[1]
    for idx, actual in enumerate(outs):
        r, c = divmod(idx, cols)
        got = actual.to(torch.float32)
        assert not torch.isnan(got).any() and not torch.isinf(got).any(), f"device ({r},{c}) has nan/inf"
        p = pcc(got, expected[r, c])
        assert p >= PCC[dtype], f"device ({r},{c}) pcc={p:.6f} < {PCC[dtype]} (shape={shape}, axis={cluster_axis})"
    return out


SHAPES = [
    (1, 1, 32, 64),
    (1, 1, 32, 1792),
    (1, 1, 640, 1792),
    (2, 4, 128, 512),
    (1, 2048, 1280),
    (32, 4096),
]


# --- correctness ---------------------------------------------------------------------------------------------------


@_MESH
@_FABRIC
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("cluster_axis", [1, 0])
def test_bf16_gamma(mesh_device, shape, cluster_axis):
    run_and_check(mesh_device, shape, dtype=ttnn.bfloat16, gamma_dtype=ttnn.bfloat16, cluster_axis=cluster_axis)


@_MESH
@_FABRIC
@pytest.mark.parametrize("shape", [(1, 1, 32, 1792), (1, 1, 640, 1792)], ids=lambda s: "x".join(map(str, s)))
def test_bf16_no_gamma(mesh_device, shape):
    run_and_check(mesh_device, shape, dtype=ttnn.bfloat16, gamma_dtype=None, cluster_axis=1)


@_MESH
@_FABRIC
@pytest.mark.parametrize("shape", [(1, 1, 32, 1792), (2, 4, 128, 512)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("gamma_dtype", [ttnn.float32, None], ids=["gamma_fp32", "no_gamma"])
def test_fp32(mesh_device, shape, gamma_dtype):
    run_and_check(mesh_device, shape, dtype=ttnn.float32, gamma_dtype=gamma_dtype, cluster_axis=1)


@_MESH
@_FABRIC
def test_bf16_x_fp32_gamma_eps_axis0(mesh_device):
    """Mixed dtypes (bf16 activations, fp32 weights) along a mesh column, custom epsilon."""
    run_and_check(
        mesh_device, (1, 1, 640, 1536), dtype=ttnn.bfloat16, gamma_dtype=ttnn.float32, cluster_axis=0, eps=1e-5
    )


@_MESH
@_FABRIC
@pytest.mark.parametrize("num_links", [1, 2, None], ids=["L1", "L2", "Lauto"])
def test_num_links(mesh_device, num_links):
    run_and_check(
        mesh_device,
        (1, 1, 640, 1792),
        dtype=ttnn.bfloat16,
        gamma_dtype=ttnn.bfloat16,
        cluster_axis=1,
        num_links=num_links,
    )


@_MESH
@_FABRIC
def test_explicit_compute_config_and_memory_config(mesh_device):
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    out = run_and_check(
        mesh_device,
        (1, 1, 32, 1280),
        dtype=ttnn.bfloat16,
        gamma_dtype=ttnn.bfloat16,
        cluster_axis=1,
        compute_kernel_config=cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


@_MESH
@_FABRIC
def test_repeat_is_bit_identical(mesh_device):
    """Back-to-back calls (program cache, ready fence, re-armed counters) give bit-identical results."""
    skip_if_axis_trivial(mesh_device, 1)
    x, g = make_inputs(mesh_device, (1, 1, 640, 1792), ttnn.bfloat16, ttnn.bfloat16)
    tx = to_mesh(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = to_mesh(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    first = per_device(distributed_rms_norm(tx, cluster_axis=1, gamma=tg))
    for _ in range(3):
        again = per_device(distributed_rms_norm(tx, cluster_axis=1, gamma=tg))
        for a, b in zip(first, again):
            assert torch.equal(a, b)


def test_default_compute_kernel_config():
    cfg = default_compute_kernel_config()
    assert cfg.fp32_dest_acc_en is True
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi4
    assert default_compute_kernel_config() is not cfg  # factory, not a shared constant


# --- validation ----------------------------------------------------------------------------------------------------


@_MESH
@_FABRIC
def test_caller_errors(mesh_device, expect_error):
    skip_if_axis_trivial(mesh_device, 1)
    x, g = make_inputs(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.bfloat16)
    tx = to_mesh(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # gamma last dim != W_local
    _, g_bad = make_inputs(mesh_device, (1, 1, 32, 128), ttnn.bfloat16, ttnn.bfloat16)
    tg_bad = to_mesh(g_bad, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):  # message text is the implementer's
        distributed_rms_norm(tx, cluster_axis=1, gamma=tg_bad)

    # num_links beyond what any hop can provide
    with expect_error((ValueError, RuntimeError), ""):  # message text is the implementer's
        distributed_rms_norm(tx, cluster_axis=1, num_links=64)

    # rank < 2
    flat = ttnn.from_torch(
        torch.randn(32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with expect_error((ValueError, RuntimeError), ""):  # message text is the implementer's
        distributed_rms_norm(flat, cluster_axis=1)


@_MESH
@_FABRIC
def test_fp32_without_fp32_acc_is_refused(mesh_device, expect_error):
    skip_if_axis_trivial(mesh_device, 1)
    x, _ = make_inputs(mesh_device, (1, 1, 32, 64), ttnn.float32, None)
    tx = to_mesh(x, mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    with expect_error(NotImplementedError, ""):  # a SupportRefusal (ExcludedCell / UnsupportedAxisValue)
        distributed_rms_norm(tx, cluster_axis=1, compute_kernel_config=cfg)
