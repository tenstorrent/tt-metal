# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for distributed_rms_norm (the immutable spec; see op_design.md).

A multi-device CCL op: it needs a fabric mesh, not the single-chip `device` fixture. Every mesh comes from the root
conftest `mesh_device` fixture (module-scoped via this dir's conftest), opened the way the golden suite opens it
(FABRIC_2D_TORUS_XY + the production router payload through `device_params`): the system mesh by default, and the
ring tests parametrize it to the place's mesh shape (a real torus axis, or the box's snake ring opened as 1xN / Nx1).
No test opens a device itself — a mesh opened inside a test body breaks under the harness's inline precompile pass
(every upload in the real pass read back as zeros).

Reference: torch.nn.functional.rms_norm over each group's concatenated hidden dim
(eval/golden_tests/distributed_rms_norm/helpers.pytorch_distributed_rms_norm). Tolerances: the golden suite's
(helpers.TOLERANCES via check_output; the WORST device decides).

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm.py
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import (
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from eval.golden_tests.distributed_rms_norm.topology import (
    FABRIC_CONFIG,
    ring_mock_meshes,
    system_mesh_shape,
    topology_feasible,
    wrap_axes,
)
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.distributed_rms_norm import default_compute_kernel_config, distributed_rms_norm

SEED = 42
_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


# ----------------------------------------------------------------------------------------------------------------------
# Mesh handling (module-scoped, one fabric init per mesh shape)
# ----------------------------------------------------------------------------------------------------------------------


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [{"fabric_config": FABRIC_CONFIG, "fabric_router_config": _router_config()}]
on_fabric = pytest.mark.parametrize("device_params", FABRIC, indirect=True)


class _Meshes:
    """`meshes.get(shape)` -> the fixture's mesh, which must already have that shape (parametrize `mesh_device`)."""

    def __init__(self, mesh):
        self.mesh = mesh

    def get(self, mesh_shape):
        assert tuple(self.mesh.shape) == tuple(mesh_shape), f"mesh {tuple(self.mesh.shape)} vs wanted {mesh_shape}"
        return self.mesh


@pytest.fixture
def meshes(mesh_device):
    return _Meshes(mesh_device)


SYSTEM = system_mesh_shape()
on_system = pytest.mark.parametrize("mesh_device", [SYSTEM], indirect=True, ids=lambda m: f"{m[0]}x{m[1]}")
# cluster axes of the system mesh that form a group (size >= 2); Linear always runs on them
SYSTEM_AXES = [a for a in (0, 1) if SYSTEM[a] >= 2]
# where a ring of G >= 3 can run: a real torus axis of the system mesh, else the box's snake ring (1xN / Nx1)
RING_PLACES = [(SYSTEM, a) for a in (0, 1) if topology_feasible(ttnn.Topology.Ring, a, SYSTEM) is None]
RING_PLACES += [(shape, axis) for shape, axis in ring_mock_meshes()]


def _group_hops(mesh_shape, axis, ring):
    G = mesh_shape[axis]
    hops = []
    for other in range(mesh_shape[1 - axis]):
        coords = [(i, other) if axis == 0 else (other, i) for i in range(G)]
        for i in range(G if ring else G - 1):
            a, b = coords[i], coords[(i + 1) % G]
            hops += [(a, b), (b, a)]
    return hops


def _links_available(mesh, axis, ring):
    node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))
    return min(
        len(ttnn.get_forwarding_link_indices(node(a), node(b))) for a, b in _group_hops(tuple(mesh.shape), axis, ring)
    )


def _inputs(mesh, shape, dtype, gamma_dtype, cluster_axis, epsilon, seed=SEED):
    x = random_stacked(mesh, shape, _TORCH[dtype], seed=seed)
    gamma = None
    if gamma_dtype is not None:
        gamma = random_stacked(mesh, (1, 1, 1, shape[-1]), _TORCH[gamma_dtype], seed=seed + 1)
    expected = pytorch_distributed_rms_norm(x, gamma, cluster_axis, epsilon)
    tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = None if gamma is None else create_ttnn_gamma(gamma, mesh, dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT)
    return tx, tg, expected


def _run(
    mesh,
    shape,
    *,
    cluster_axis,
    dtype=ttnn.bfloat16,
    gamma_dtype=ttnn.bfloat16,
    topology=ttnn.Topology.Linear,
    num_links=1,
    epsilon=1e-6,
    seed=SEED,
    **kwargs,
):
    tx, tg, expected = _inputs(mesh, shape, dtype, gamma_dtype, cluster_axis, epsilon, seed)
    out = distributed_rms_norm(
        tx, cluster_axis=cluster_axis, gamma=tg, epsilon=epsilon, topology=topology, num_links=num_links, **kwargs
    )
    check_output(out, expected, shape=shape, dtype=dtype, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT)
    return out


def _device_outputs(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


# ----------------------------------------------------------------------------------------------------------------------
# System mesh: shapes (single tile, multi-tile, non-square, multi-batch, rank 2 / 3 / 4) over every group axis, Linear
# ----------------------------------------------------------------------------------------------------------------------

SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 32, 1792),  # decode (one tile-row): R3-X W-split
    (1, 1, 640, 1792),  # perf focus: R3-X row-groups
    (1, 1, 2048, 1280),  # prefill: R1-X
    (2, 4, 128, 512),  # multi-batch rank 4
    (1, 2048, 1280),  # rank 3
    (32, 2048),  # rank 2
    (1, 1, 32, 8192),  # wide local hidden
]


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
@pytest.mark.parametrize("cluster_axis", SYSTEM_AXES, ids=lambda a: f"axis{a}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_shapes_linear(meshes, shape, cluster_axis):
    _run(meshes.get(SYSTEM), shape, cluster_axis=cluster_axis)


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
@pytest.mark.parametrize(
    "dtype,gamma_dtype",
    [
        (ttnn.float32, ttnn.float32),
        (ttnn.float32, None),
        (ttnn.bfloat16, ttnn.float32),  # mixed: bf16 activations, fp32 weights
        (ttnn.bfloat16, None),
        (ttnn.float32, ttnn.bfloat16),
    ],
    ids=["fp32-g_fp32", "fp32-nog", "bf16-g_fp32", "bf16-nog", "fp32-g_bf16"],
)
@pytest.mark.parametrize("shape", [(1, 1, 128, 1024), (1, 1, 32, 1792)], ids=lambda s: "x".join(map(str, s)))
def test_dtypes_and_gamma(meshes, shape, dtype, gamma_dtype):
    _run(meshes.get(SYSTEM), shape, cluster_axis=SYSTEM_AXES[-1], dtype=dtype, gamma_dtype=gamma_dtype)


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
def test_epsilon_and_explicit_config(meshes):
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    _run(meshes.get(SYSTEM), (1, 1, 64, 1024), cluster_axis=SYSTEM_AXES[0], epsilon=1e-5, compute_kernel_config=cfg)


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
def test_large_input_scale_small_epsilon(meshes):
    """Rows whose mean square is far from 1 (x ~ 30·N(0,1)): the divisor and ε placement must be exact."""
    mesh = meshes.get(SYSTEM)
    shape, axis, eps = (1, 1, 64, 512), SYSTEM_AXES[-1], 1e-6
    x = random_stacked(mesh, shape, torch.float32, seed=SEED, scale=30.0)
    gamma = random_stacked(mesh, (1, 1, 1, shape[-1]), torch.float32, seed=SEED + 1)
    expected = pytorch_distributed_rms_norm(x, gamma, axis, eps)
    tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(gamma, mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    out = distributed_rms_norm(tx, cluster_axis=axis, gamma=tg, epsilon=eps)
    check_output(
        out, expected, shape=shape, dtype=ttnn.float32, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT
    )


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["out_dram", "out_l1"])
def test_output_memory_config(meshes, memory_config):
    out = _run(meshes.get(SYSTEM), (1, 1, 128, 512), cluster_axis=SYSTEM_AXES[0], memory_config=memory_config)
    assert out.memory_config().buffer_type == memory_config.buffer_type
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
def test_default_compute_kernel_config_is_fresh_maxed_out():
    a, b = default_compute_kernel_config(), default_compute_kernel_config()
    assert a is not b
    assert a.fp32_dest_acc_en is True
    assert a.math_fidelity == ttnn.MathFidelity.HiFi4


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
@pytest.mark.parametrize("shape", [(1, 1, 640, 1792), (1, 1, 32, 1792)], ids=lambda s: "x".join(map(str, s)))
def test_repeat_bit_identical_and_program_cache(meshes, shape):
    """Back-to-back calls with no host sync: bit-identical repeats (fixed sum order, counters re-armed), and a cached
    program run on new tensors (only runtime args change) still produces the right answer."""
    mesh = meshes.get(SYSTEM)
    axis = SYSTEM_AXES[-1]
    tx, tg, expected = _inputs(mesh, shape, ttnn.bfloat16, ttnn.bfloat16, axis, 1e-6)
    outs = [distributed_rms_norm(tx, cluster_axis=axis, gamma=tg) for _ in range(3)]
    ref = _device_outputs(outs[0])
    for o in outs[1:]:
        for a, b in zip(ref, _device_outputs(o)):
            assert torch.equal(a, b), "repeated calls must be bit-identical"
    check_output(
        outs[0], expected, shape=shape, dtype=ttnn.bfloat16, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT
    )
    # new tensors, same program (cache hit): different data must give its own correct result
    _run(mesh, shape, cluster_axis=axis, seed=SEED + 7)


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
def test_every_device_identical_inv_across_group(meshes):
    """All devices of a group hold the same rows of the same full row: with x identical on every device and no gamma,
    every device's output must be bit-identical (same Σ in the same device order everywhere)."""
    mesh = meshes.get(SYSTEM)
    axis = SYSTEM_AXES[-1]
    shape = (1, 1, 64, 256)
    rows, cols = tuple(mesh.shape)
    torch.manual_seed(SEED)
    one = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x = one.expand(rows, cols, *shape).contiguous()
    tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = distributed_rms_norm(tx, cluster_axis=axis)
    check_output(
        out,
        pytorch_distributed_rms_norm(x, None, axis, 1e-6),
        shape=shape,
        dtype=ttnn.bfloat16,
        fp32_dest_acc_en=True,
        expected_layout=ttnn.TILE_LAYOUT,
    )
    devs = _device_outputs(out)
    for d in devs[1:]:
        assert torch.equal(devs[0], d)


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
@pytest.mark.parametrize(
    "shape", [(1, 1, 256, 512), (1, 1, 8192, 256)], ids=["r3_two_rows_per_group", "r1_several_rows_per_core"]
)
def test_waves_forced(meshes, monkeypatch, shape):
    """Regime WAVES (op_design.md -> Regimes): BLOCK_ROWS_CAP = 1 forces one tile-row per block, so every core runs
    several blocks and the group runs several exchange rounds in one call."""
    import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "BLOCK_ROWS_CAP", 1)
    mesh = meshes.get(SYSTEM)
    _run(mesh, shape, cluster_axis=SYSTEM_AXES[-1])
    _run(mesh, shape, cluster_axis=SYSTEM_AXES[-1], seed=SEED + 3)  # second call: counters were re-armed


# ----------------------------------------------------------------------------------------------------------------------
# Validation: caller errors raise ValueError / RuntimeError; out-of-contract cells raise a support refusal
# ----------------------------------------------------------------------------------------------------------------------


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
def test_caller_errors(meshes, expect_error):
    mesh = meshes.get(SYSTEM)
    axis = SYSTEM_AXES[0]
    shape = (1, 1, 32, 256)
    tx, _, _ = _inputs(mesh, shape, ttnn.bfloat16, None, axis, 1e-6)

    # rank < 2
    t1 = ttnn.from_torch(
        torch.randn(256, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(t1, cluster_axis=axis)

    # gamma shape does not match its layout's format for this W_local
    bad_gamma = create_ttnn_input_tensor(
        random_stacked(mesh, (1, 1, 1, shape[-1] + 32), torch.bfloat16, seed=1),
        mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=axis, gamma=bad_gamma)

    # num_links above what the fabric discovered, and non-positive
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=axis, num_links=_links_available(mesh, axis, False) + 1)
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=axis, num_links=0)

    # cluster_axis outside {0, 1}
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=2)


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
def test_ring_without_wrap_link_raises(meshes, expect_error):
    no_wrap = [a for a in SYSTEM_AXES if SYSTEM[a] >= 3 and a not in wrap_axes()]
    if not no_wrap:
        pytest.skip("no system-mesh axis of size >= 3 without a wrap link")
    mesh = meshes.get(SYSTEM)
    tx, _, _ = _inputs(mesh, (1, 1, 32, 256), ttnn.bfloat16, None, no_wrap[0], 1e-6)
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=no_wrap[0], topology=ttnn.Topology.Ring)


@on_fabric
@on_system
@pytest.mark.skipif(not SYSTEM_AXES, reason="system mesh has no axis of size >= 2")
def test_unsupported_cells_refused(meshes, expect_error):
    mesh = meshes.get(SYSTEM)
    axis = SYSTEM_AXES[0]
    # (w_non_aligned and ROW_MAJOR gamma sticks are supported since Refinement 2.)
    # fp32 input with 16-bit DEST: refused (an axis value outside SUPPORTED, and an EXCLUSION)
    tx32, _, _ = _inputs(mesh, (1, 1, 32, 256), ttnn.float32, None, axis, 1e-6)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    with expect_error((UnsupportedAxisValue, ExcludedCell), ""):
        distributed_rms_norm(tx32, cluster_axis=axis, compute_kernel_config=cfg)


# ----------------------------------------------------------------------------------------------------------------------
# Rings (the production topology): a real torus axis, or the box's snake ring. Ring and Linear, 1 and 2 links, over the
# perf-focus shapes (feature_spec.LOOSE_CASES). Last in the file: they may reopen the mesh as 1xN / Nx1.
# ----------------------------------------------------------------------------------------------------------------------

RING_SHAPES = [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280)]
_RING_CASES = [
    pytest.param(
        place[0],
        place[1],
        topology,
        num_links,
        shape,
        id=f"{place[0][0]}x{place[0][1]}-axis{place[1]}-{topology.name}-L{num_links}-{'x'.join(map(str, shape))}",
    )
    for place in RING_PLACES
    for topology in (ttnn.Topology.Ring, ttnn.Topology.Linear)
    for num_links in (1, 2)
    for shape in RING_SHAPES
]


@on_fabric
@pytest.mark.skipif(not _RING_CASES, reason="no ring of >= 3 devices can be formed on this cluster")
@pytest.mark.parametrize("mesh_device,cluster_axis,topology,num_links,shape", _RING_CASES, indirect=["mesh_device"])
def test_ring(mesh_device, cluster_axis, topology, num_links, shape):
    mesh = mesh_device
    if num_links > _links_available(mesh, cluster_axis, topology == ttnn.Topology.Ring):
        pytest.skip(f"fewer than {num_links} links on some hop of the group")
    _run(mesh, shape, cluster_axis=cluster_axis, topology=topology, num_links=num_links)


@on_fabric
@pytest.mark.skipif(not RING_PLACES, reason="no ring of >= 3 devices can be formed on this cluster")
@pytest.mark.parametrize("mesh_device,axis", RING_PLACES[-1:], indirect=["mesh_device"])
def test_ring_fp32_no_gamma_and_default_links(mesh_device, axis):
    mesh = mesh_device
    _run(
        mesh,
        (1, 1, 128, 1024),
        cluster_axis=axis,
        dtype=ttnn.float32,
        gamma_dtype=None,
        topology=ttnn.Topology.Ring,
        num_links=None,
    )


@on_fabric
@pytest.mark.skipif(not ring_mock_meshes(), reason="no 1xN / Nx1 snake mesh on this cluster")
@pytest.mark.parametrize("mesh_device,axis", ring_mock_meshes()[:1], indirect=["mesh_device"])
def test_cluster_axis_of_size_one_raises(mesh_device, axis, expect_error):
    mesh = mesh_device
    other = 1 - axis  # size 1 on a 1xN / Nx1 mesh
    tx, _, _ = _inputs(mesh, (1, 1, 32, 256), ttnn.bfloat16, None, axis, 1e-6)
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(tx, cluster_axis=other)
