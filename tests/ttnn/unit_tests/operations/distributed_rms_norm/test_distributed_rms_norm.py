# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for distributed_rms_norm (immutable spec — the implementer does not edit this file).

distributed_rms_norm is a multi-device (CCL) op: every device (r, c) of a mesh holds x[r, c] of shape
(..., M, W_local); the G devices of a group along `cluster_axis` together hold the full hidden dim
W = G * W_local, and device p returns its slice of torch.nn.functional.rms_norm(concat_p x[p], (W,), gamma, eps).

A single-device `device` fixture cannot run it, so this module uses the golden suite's fabric mesh fixture
(`mesh_device`: the system mesh — or a pinned `mesh_shape` such as the 1xN ring mock — opened with
FABRIC_2D_TORUS_XY and the production router payload, cached across tests). Every test is parametrized on
`mesh_shape` (None = the system mesh), which that fixture reads.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401 (fixtures)
from eval.golden_tests.distributed_rms_norm.topology import ring_mock_meshes, system_mesh_shape, topology_feasible
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


# ---------------------------------------------------------------------------------------------------------------------
# Host helpers (pure torch reference; no ttnn op in the reference)
# ---------------------------------------------------------------------------------------------------------------------


def reference(x_stacked, gamma_stacked, cluster_axis, eps):
    """x_stacked (R, C, ..., M, W_local), gamma_stacked (R, C, W_local) or None -> stacked expected output (fp32)."""
    x = x_stacked.to(torch.float32)
    rows, cols = x.shape[0], x.shape[1]
    w_local = x.shape[-1]
    out = torch.empty_like(x)
    others = range(cols) if cluster_axis == 0 else range(rows)
    for o in others:
        members = [(r, o) for r in range(rows)] if cluster_axis == 0 else [(o, c) for c in range(cols)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        weight = None
        if gamma_stacked is not None:
            weight = torch.cat([gamma_stacked[r, c].to(torch.float32).reshape(-1) for r, c in members])
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=weight, eps=eps)
        for p, (r, c) in enumerate(members):
            out[r, c] = normed[..., p * w_local : (p + 1) * w_local]
    return out


def to_mesh(stacked, mesh, *, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Stacked (R, C, *shape) torch tensor -> mesh tensor with device (r, c) holding stacked[r, c]."""
    rows, cols = stacked.shape[0], stacked.shape[1]
    assert (rows, cols) == tuple(mesh.shape)
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )


def make_gamma(gamma_stacked, mesh, *, dtype, layout):
    """Logical stacked gamma (R, C, W_local) -> TILE (1,1,1,W_local) or ROW_MAJOR sticks (1,1,ceil(W/32),32)."""
    rows, cols, w_local = gamma_stacked.shape
    g = gamma_stacked.reshape(rows, cols, 1, 1, 1, w_local)
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        padded = -(-w_local // 32) * 32
        if padded != w_local:
            g = torch.nn.functional.pad(g, (0, padded - w_local))
        g = g.reshape(rows, cols, 1, 1, padded // 32, 32)
    return to_mesh(g, mesh, dtype=dtype, layout=layout)


def quantize_like_device(stacked, dtype):
    """bfloat8_b: the values the device actually holds, so the reference measures the op, not the format."""
    if dtype != ttnn.bfloat8_b:
        return stacked
    flat = stacked.reshape(-1, stacked.shape[-1])
    t = ttnn.from_torch(flat, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).reshape(stacked.shape).to(torch.bfloat16)


def pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return 1.0 if den == 0 else float((a * b).sum() / den)


def device_outputs(out, mesh):
    rows, cols = tuple(mesh.shape)
    devs = ttnn.get_device_tensors(out)
    assert len(devs) == rows * cols
    return [[ttnn.to_torch(devs[r * cols + c]) for c in range(cols)] for r in range(rows)]


def check(out, expected, mesh, *, shape, dtype):
    """Every device is scored on its own; the worst device decides."""
    assert list(out.shape) == list(shape), f"shape {list(out.shape)} != {list(shape)}"
    assert out.dtype == dtype
    assert out.layout == ttnn.TILE_LAYOUT
    got = device_outputs(out, mesh)
    worst = None
    for r, row in enumerate(got):
        for c, actual in enumerate(row):
            assert (
                not torch.isnan(actual.float()).any() and not torch.isinf(actual.float()).any()
            ), f"inf/nan on ({r},{c})"
            v = pcc(actual, expected[r, c])
            if worst is None or v < worst[0]:
                worst = (v, (r, c))
    assert worst[0] >= PCC[dtype], f"PCC {worst[0]:.6f} < {PCC[dtype]} on device {worst[1]}"


def config(fp32_dest_acc_en=True):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


def run_case(
    mesh,
    shape,
    *,
    cluster_axis,
    dtype=ttnn.bfloat16,
    gamma_dtype=ttnn.bfloat16,
    gamma_layout=ttnn.TILE_LAYOUT,
    has_gamma=True,
    topology=ttnn.Topology.Linear,
    num_links=None,
    eps=1e-6,
    fp32_dest_acc_en=True,
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    out_memory=None,
    seed=42,
):
    torch.manual_seed(seed)
    rows, cols = tuple(mesh.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    x = quantize_like_device(x, dtype)
    gamma = (
        torch.randn((rows, cols, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype]) if has_gamma else None
    )
    expected = reference(x, gamma, cluster_axis, eps)

    kwargs = {}
    mem = ttnn.DRAM_MEMORY_CONFIG
    if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        mem = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=mesh)
    tx = to_mesh(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem)
    if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        kwargs["memory_config"] = tx.memory_config()
    if out_memory is not None:
        kwargs["memory_config"] = out_memory(shape) if callable(out_memory) else out_memory
    tg = make_gamma(gamma, mesh, dtype=gamma_dtype, layout=gamma_layout) if has_gamma else None

    out = distributed_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        gamma=tg,
        epsilon=eps,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=config(fp32_dest_acc_en),
        **kwargs,
    )
    check(out, expected, mesh, shape=shape, dtype=dtype)
    return tx, tg, out


def _usable_links(mesh, cluster_axis):
    rows, cols = tuple(mesh.shape)
    node = lambda r, c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    counts = []
    for o in range(cols if cluster_axis == 0 else rows):
        coords = [(r, o) for r in range(rows)] if cluster_axis == 0 else [(o, c) for c in range(cols)]
        for a, b in zip(coords, coords[1:]):
            counts.append(len(ttnn.get_forwarding_link_indices(node(*a), node(*b))))
    return min(counts)


# ---------------------------------------------------------------------------------------------------------------------
# Collection-time cluster capability
# ---------------------------------------------------------------------------------------------------------------------

_SYSTEM = system_mesh_shape()
LINEAR_AXES = [a for a in (0, 1) if topology_feasible(ttnn.Topology.Linear, a, _SYSTEM) is None]
RING_AXES = [a for a in (0, 1) if topology_feasible(ttnn.Topology.Ring, a, _SYSTEM) is None]
# (mesh_shape, cluster_axis) pairs that close a ring: real torus axes on the system mesh, else the snake-ring mock.
RING_MESHES = [(None, a) for a in RING_AXES] or [tuple(m) for m in ring_mock_meshes()]

needs_group = pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis with >= 2 devices")
needs_ring = pytest.mark.skipif(not RING_MESHES, reason="this cluster cannot close a ring")

SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 32, 1792),  # decode, DeepSeek-V3 / Kimi 7168 / TP4 (W-split regime, 1 wave)
    (1, 1, 640, 1792),  # PERF FOCUS (W-split regime, 2 waves)
    (1, 1, 2048, 1280),  # prefill (row-parallel regime)
    (2, 4, 128, 512),  # multi-batch, 4D
    (1, 32, 1792),  # 3D
    (128, 256),  # 2D, non-square
    (1, 1, 17, 50),  # both non-aligned
    (1, 1, 64, 1000),  # W_local non-aligned
    (1, 1, 100, 512),  # H non-aligned
    (4, 1, 512, 512),  # batched, row-parallel
]


def _id(shape):
    return "x".join(map(str, shape))


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize("cluster_axis", LINEAR_AXES or [1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=_id)
def test_linear(mesh_shape, cluster_axis, dtype, shape, mesh_device):
    run_case(mesh_device, shape, cluster_axis=cluster_axis, dtype=dtype, gamma_dtype=ttnn.bfloat16)


@needs_ring
@pytest.mark.parametrize("mesh_shape,cluster_axis", RING_MESHES or [(None, 1)])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("shape", [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280), (1, 1, 64, 128)], ids=_id)
def test_ring(mesh_shape, cluster_axis, num_links, shape, mesh_device):
    if num_links > _usable_links(mesh_device, cluster_axis):
        pytest.skip(f"only {_usable_links(mesh_device, cluster_axis)} usable link(s) per hop")
    run_case(mesh_device, shape, cluster_axis=cluster_axis, topology=ttnn.Topology.Ring, num_links=num_links)


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize(
    "gamma_dtype,gamma_layout,has_gamma",
    [
        (ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        (ttnn.float32, ttnn.TILE_LAYOUT, True),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
        (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, True),
        (None, None, False),
    ],
    ids=["tile-bf16", "tile-fp32", "sticks-bf16", "sticks-fp32", "no_gamma"],
)
@pytest.mark.parametrize("shape", [(1, 1, 640, 1792), (1, 1, 64, 1000), (1, 32, 1792)], ids=_id)
def test_gamma_formats(mesh_shape, gamma_dtype, gamma_layout, has_gamma, shape, mesh_device):
    run_case(
        mesh_device,
        shape,
        cluster_axis=LINEAR_AXES[0],
        gamma_dtype=gamma_dtype or ttnn.bfloat16,
        gamma_layout=gamma_layout or ttnn.TILE_LAYOUT,
        has_gamma=has_gamma,
    )


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize("shape", [(1, 1, 32, 1792), (1, 1, 640, 1792), (2, 4, 128, 512), (1, 1, 100, 512)], ids=_id)
def test_bfloat8_b(mesh_shape, shape, mesh_device):
    run_case(mesh_device, shape, cluster_axis=LINEAR_AXES[0], dtype=ttnn.bfloat8_b)


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8b"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 1792), (1, 1, 640, 1792), (1, 1, 2048, 1280)], ids=_id)
def test_fp32_dest_off(mesh_shape, dtype, shape, mesh_device):
    run_case(mesh_device, shape, cluster_axis=LINEAR_AXES[0], dtype=dtype, fp32_dest_acc_en=False)


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize(
    "memory_layout",
    [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED],
    ids=["height", "width"],
)
@pytest.mark.parametrize("shape", [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 64, 1000), (2, 4, 128, 512)], ids=_id)
def test_sharded(mesh_shape, memory_layout, shape, mesh_device):
    _, _, out = run_case(mesh_device, shape, cluster_axis=LINEAR_AXES[0], memory_layout=memory_layout)
    assert out.memory_config().memory_layout == memory_layout


def _width_out(grid_x, grid_y, x0=0):
    """WIDTH_SHARDED L1 output over a grid_x x grid_y rectangle at (x0, 0) (ragged last shard allowed)."""

    def make(shape):
        rows = 1
        for d in shape[:-1]:
            rows *= d
        n = grid_x * grid_y
        shard_w = -(-shape[-1] // (n * 32)) * 32
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x0 + grid_x - 1, grid_y - 1))})
        return ttnn.create_sharded_memory_config(
            shape=(-(-rows // 32) * 32, shard_w),
            core_grid=crs,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    return make


# Refinement 2: output placement independent of the input placement (writer TensorAccessor path).
@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize(
    "in_layout,out_memory",
    [
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.TensorMemoryLayout.INTERLEAVED, _width_out(8, 3)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, _width_out(4, 2, 1)),
    ],
    ids=["ws_to_dram", "hs_to_dram", "dram_to_ws", "ws_to_ws_regrid"],
)
@pytest.mark.parametrize("shape", [(1, 1, 32, 1792), (1, 1, 64, 1000)], ids=_id)
def test_output_placement(mesh_shape, in_layout, out_memory, shape, mesh_device):
    _, _, out = run_case(
        mesh_device, shape, cluster_axis=LINEAR_AXES[0], memory_layout=in_layout, out_memory=out_memory
    )
    want = out_memory(shape) if callable(out_memory) else out_memory
    assert out.memory_config().memory_layout == want.memory_layout


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize("cluster_axis", LINEAR_AXES or [1])
def test_custom_epsilon_and_axis(mesh_shape, cluster_axis, mesh_device):
    run_case(mesh_device, (1, 1, 128, 1024), cluster_axis=cluster_axis, eps=1e-5, gamma_dtype=ttnn.float32)


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize("shape", [(1, 1, 640, 1792), (1, 1, 32, 1792)], ids=_id)
def test_repeat_deterministic(mesh_shape, shape, mesh_device):
    """Back-to-back calls (cached program, persistent semaphores re-armed by the op) are bit-identical, and a call
    on new tensors after them is still correct."""
    axis = LINEAR_AXES[0]
    tx, tg, first = run_case(mesh_device, shape, cluster_axis=axis)
    ref = device_outputs(first, mesh_device)
    for _ in range(3):
        again = distributed_rms_norm(tx, cluster_axis=axis, gamma=tg, epsilon=1e-6, compute_kernel_config=config())
        for row_a, row_b in zip(ref, device_outputs(again, mesh_device)):
            for a, b in zip(row_a, row_b):
                assert torch.equal(a, b), "repeated call is not bit-identical"
    run_case(mesh_device, shape, cluster_axis=axis, seed=7)


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
def test_multi_block(mesh_shape, mesh_device):
    """More rows than one resident block holds on the fused grid (fp32, wide prefill)."""
    run_case(mesh_device, (1, 1, 4096, 1024), cluster_axis=LINEAR_AXES[0], dtype=ttnn.float32, gamma_dtype=ttnn.float32)


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
def test_validation(mesh_shape, mesh_device, expect_error):
    axis = LINEAR_AXES[0]
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 64, 128)
    x = torch.randn((rows, cols, *shape))
    tx = to_mesh(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # gamma whose shape does not match its layout's format for this W_local
    bad_tile = to_mesh(
        torch.randn((rows, cols, 1, 1, 1, 64)), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=axis, gamma=bad_tile)
    bad_sticks = to_mesh(
        torch.randn((rows, cols, 1, 1, 1, 128)), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # (1,1,1,128) row-major instead of the (1,1,4,32) sticks
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=axis, gamma=bad_sticks)

    # num_links above what the cluster discovered
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=axis, num_links=_usable_links(mesh_device, axis) + 1)

    # a cluster_axis of size 1, where the mesh has one
    for a in (0, 1):
        if tuple(mesh_device.shape)[a] == 1:
            with expect_error((ValueError, RuntimeError), ""):
                distributed_rms_norm(tx, cluster_axis=a)

    # Ring along an axis this cluster cannot close (no wrap link / non-wrapping fabric config)
    for a in LINEAR_AXES:
        if (
            topology_feasible(ttnn.Topology.Ring, a, tuple(mesh_device.shape)) is not None
            and tuple(mesh_device.shape)[a] >= 3
        ):
            with expect_error((ValueError, RuntimeError), ""):
                distributed_rms_norm(tx, cluster_axis=a, topology=ttnn.Topology.Ring)

    # float32 input with 16-bit DEST is a support refusal (EXCLUSION), not a silent downgrade
    tx32 = to_mesh(x, mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    with expect_error(NotImplementedError, ""):
        distributed_rms_norm(tx32, cluster_axis=axis, compute_kernel_config=config(fp32_dest_acc_en=False))
