# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared mesh parametrization and setup for the device test suites.

**Mesh coverage** (recipe section 4). The graded shape is the spec's ``(8, 4)`` = sp8 x tp4, but
every row of every Testing table is parametrized across the shapes this pod can reach and skips
cleanly on the ones it cannot. Each chip count gets a **pair**: one shape at the target's TP
(identical per-chip tensor shapes to production) and one ``tp=1`` shape at the same chip count
(isolates SP/ring bugs from TP-interaction bugs).

TP is bounded by what divides evenly per chip for this model's attention family: 4 KV heads and
16 GDN key heads, so ``tp in {1, 2, 4}`` and 4 is the ceiling. "Every shape" below means every
shape that divisibility allows.

| chips | target-TP shape | tp=1 shape |
|-------|-----------------|------------|
| 4  (quietbox-shaped) | (2, 2) sp2 x tp2 | (4, 1) sp4 x tp1 |
| 8  (loudbox-shaped)  | (2, 4) sp2 x tp4 | (8, 1) sp8 x tp1 |
| 32 (galaxy)          | (8, 4) sp8 x tp4 — **graded** | (32, 1): not carvable, see below |

**Sub-meshes, not separate allocations.** Opening a small ``MeshShape`` directly on a galaxy fails
fabric router sync — the descriptor describes the whole 8x4 fabric, and a 2x2 allocation out of it
cannot complete the ethernet handshake. So the full mesh is opened once and the smaller shapes are
carved with ``create_submesh``, which is what the recipe prescribes and what actually maps. The
consequence: a shape must fit *inside the grid*, not merely inside the chip count, so ``(32, 1)``
— the tp=1 partner of the graded shape — has no home on an 8x4 pod and is logged as a ``skip``.
"""

from __future__ import annotations

import os

import pytest

import ttnn

from ..config import MeshConfig
from ..reference.config import Qwen35TextConfig
from ..spec import load_spec
from ..tt.ccl import CCLManager
from ..utils.general_utils import get_default_num_links

# (rows, cols) = (sp, tp). Ordered smallest-first so `-k 2x2` picks the cheap one.
COVERAGE_SHAPES: tuple[tuple[int, int], ...] = ((2, 2), (4, 1), (2, 4), (8, 1), (8, 4), (32, 1))
GRADED_SHAPE = load_spec().mesh_shape


def pod_shape() -> tuple[int, int]:
    """The full mesh the control plane exposes, e.g. ``(8, 4)`` on a single Blackhole Galaxy."""
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return (shape[0], shape[1])


def _carvable(shape: tuple[int, int], pod: tuple[int, int]) -> bool:
    """A sub-mesh must fit the pod's GRID, not just its chip count."""
    return shape[0] <= pod[0] and shape[1] <= pod[1]


def parametrize_mesh(mesh_shapes: tuple[tuple[int, int], ...] | None = None, *, graded_only: bool = False):
    """Parametrize a test over ``submesh_shape``; the full pod mesh is opened once behind it.

    Use the ``mesh`` fixture in the test body, not ``mesh_device`` — ``mesh`` is the carved
    sub-mesh of the requested shape. Shapes the pod cannot carve are emitted as explicit SKIPs
    rather than dropped, so a coverage hole is visible in the report instead of silently absent.
    ``QWEN35_MESH_SHAPES="2x2,8x4"`` narrows the set for a fast iteration loop; ``CI=true`` keeps
    only the largest shape that fits.
    """
    shapes = [GRADED_SHAPE] if graded_only else list(mesh_shapes or COVERAGE_SHAPES)
    override = os.getenv("QWEN35_MESH_SHAPES")
    if override:
        wanted = {tuple(int(v) for v in s.split("x")) for s in override.split(",")}
        shapes = [s for s in shapes if s in wanted]

    pod = pod_shape()
    fitting = [s for s in shapes if _carvable(s, pod)]
    if os.getenv("CI") == "true" and len(fitting) > 1:
        fitting = [max(fitting, key=lambda s: s[0] * s[1])]

    params = []
    for shape in shapes:
        marks = ()
        if shape not in fitting:
            marks = (
                pytest.mark.skip(reason=f"sub-mesh {shape[0]}x{shape[1]} does not fit the pod grid {pod[0]}x{pod[1]}"),
            )
        params.append(
            pytest.param(
                shape,
                pod,
                {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100000000},
                id=f"{shape[0]}x{shape[1]}",
                marks=marks,
            )
        )

    def decorator(func):
        return pytest.mark.parametrize(
            "submesh_shape, mesh_device, device_params",
            params,
            indirect=["mesh_device", "device_params"],
        )(func)

    return decorator


def mesh_setup(mesh) -> tuple[MeshConfig, CCLManager]:
    """MeshConfig + CCLManager for the mesh actually opened.

    ``Topology.Linear`` unconditionally: the plain single-galaxy descriptor has no wrap-around
    links, so ``FABRIC_1D_RING`` cannot open there, and Linear is valid on a ring-wired fabric too
    (it just leaves the wrap link unused). The torus is opted into via ``QWEN35_TORUS`` in
    ``conftest.py``, which is where the descriptor choice has to happen.
    """
    shape = tuple(mesh.shape)
    mesh_config = MeshConfig(shape, tp=shape[1])
    ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
    return mesh_config, ccl


def unit_test_config(**overrides) -> Qwen35TextConfig:
    """The real config, optionally depth- or width-reduced for a host-comparable unit test.

    Any override makes the run a **reduced** one in the recipe's sense: a diagnostic, labelled as
    such wherever its numbers appear, never a grade.
    """
    base = Qwen35TextConfig.from_json()
    if not overrides:
        return base
    fields = {k: getattr(base, k) for k in base.__dataclass_fields__ if base.__dataclass_fields__[k].init}
    fields.update(overrides)
    if "num_hidden_layers" in overrides and "layer_types" not in overrides:
        n, interval = fields["num_hidden_layers"], fields["full_attention_interval"]
        fields["layer_types"] = tuple(
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        )
    return Qwen35TextConfig(**fields)


def pcc_bars() -> tuple[float, float]:
    """``(pcc_target, pcc_lower_bound)`` from the binding spec — the assert is the lower bound."""
    acc = load_spec().acceptance
    return acc.pcc_target, acc.pcc_lower_bound
