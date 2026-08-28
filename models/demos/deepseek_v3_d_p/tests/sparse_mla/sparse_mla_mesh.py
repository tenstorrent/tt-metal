# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Hardware-adaptive Fabric2D mesh parametrization for the sparse-MLA vs-trace diagnostics.

The tests are written for a `mesh_device` whose shape is ``(sp_size, tp_size)``
(``sp_axis, tp_axis = 0, 1``). Which shapes are valid depends on the box the
suite runs on, so instead of hard-coding one box every test pulls
its parametrization from here:

    QuietBox  (4 chips):  SP=1 × TP=4, SP=2 × TP=2
    LoudBox   (8 chips):  SP=8 × TP=1, SP=4 × TP=2, SP=2 × TP=4
    Galaxy   (32 chips):  SP=8 × TP=4   (production layout)

Detection mirrors ``tests/nightly/sdpa_perf_utils.MeshConfig.detect()``: it
counts ``/dev/tenstorrent/*`` device nodes WITHOUT opening them, so it is safe to
call at pytest-collection time (no device locks, no subprocess interference).

We key the shape sets on the EXACT device count rather than emitting every shape
that "fits". A multi-chip shape smaller than the physical box opens a sub-mesh,
which the runtime does not reliably support (see the warning in the top-level
``mesh_device`` fixture), and exact-match also makes each box exercise all of its
silicon instead of an arbitrary corner of it.
"""

import glob
import os

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)


def detect_num_devices() -> int:
    """Number of TT devices, counted from /dev/tenstorrent/* without opening them.

    Device nodes are numeric (/dev/tenstorrent/0, .../1, ...); filtering to those
    avoids miscounting sibling entries such as the ``by-id/`` directory.
    """
    return len([p for p in glob.glob("/dev/tenstorrent/*") if os.path.basename(p).isdigit()])


# (sp_size, tp_size) shapes per box, keyed by exact physical device count.
# In production (Galaxy) the layout is TP=4, SP=8.
MESH_SHAPES_BY_DEVICE_COUNT = {
    4: [(1, 1), (1, 4), (2, 2)],  # QuietBox: existing single-chip diagnostic, TP ring, and SP2×TP2
    8: [(8, 1), (4, 2), (2, 4)],  # LoudBox: SP proxy plus axis-sensitive 2D profiles
    32: [(8, 4)],  # Galaxy (prod): SP=8, TP=4
}

# The kvpe cache is ND-sharded in 32-token DRAM-bank chunks
# (kv_cache_utils.NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK). With fewer than two chunks per
# SP shard the cache collapses to a layout the update_cache op rejects, so a
# per-chip sequence below this is degenerate (e.g. seq256 over SP=8 → 32 tok/chip).
KVPE_MIN_TOKENS_PER_CHIP = 64


def skip_if_seq_too_small_for_sp(seq_len: int, mesh_device) -> None:
    """Skip when SP-sharding `seq_len` leaves too few tokens/chip for the kvpe cache."""
    sp = list(mesh_device.shape)[0]
    local = seq_len // sp
    if local < KVPE_MIN_TOKENS_PER_CHIP:
        pytest.skip(
            f"seq_len {seq_len} over SP={sp} → {local} tokens/chip "
            f"(< {KVPE_MIN_TOKENS_PER_CHIP}); kvpe ND-shard cache needs ≥2 DRAM-bank chunks per shard"
        )


def _shape_id(shape) -> str:
    """Stable pytest id for a (sp, tp) shape, e.g. (2, 4) -> 'sp2xtp4'."""
    sp, tp = shape
    return f"sp{sp}xtp{tp}"


def supported_mesh_shapes(num_devices: int = None):
    """(shapes, ids) — SP×TP shape set for the detected box.

    Unknown boxes return one collection-time skipped 2x2 sentinel; they never open a degenerate
    Fabric2D mesh merely to keep the parametrization nonempty.
    """
    if num_devices is None:
        num_devices = detect_num_devices()
    shapes = MESH_SHAPES_BY_DEVICE_COUNT.get(num_devices)
    if shapes is None:
        return [pytest.param((2, 2), marks=pytest.mark.skip(reason=f"unsupported device count {num_devices}"))], [
            "unsupported"
        ]
    return shapes, [_shape_id(s) for s in shapes]


def parametrize_mesh_device():
    """`@parametrize_mesh_device()` — indirect `mesh_device` over the box's SP×TP shapes.

    Drop-in for ``@pytest.mark.parametrize("mesh_device", [...], indirect=True)``;
    the shape set is chosen from the hardware at collection time.
    """
    shapes, ids = supported_mesh_shapes()
    return pytest.mark.parametrize("mesh_device", shapes, ids=ids, indirect=True)


def parametrize_mesh_and_device_params(*, worker_l1_size, torus_xy_certified=False):
    """Pair each existing sparse-MLA mesh with its valid Fabric2D profile."""
    params = []
    for shape in MESH_SHAPES_BY_DEVICE_COUNT.get(detect_num_devices(), []):
        if shape == (1, 1):
            device_params = {"fabric_config": ttnn.FabricConfig.DISABLED}
            marker = None
            profile = "disabled"
        elif shape[1] == 1 and shape[0] > 2:
            device_params = torus_y_device_params(worker_l1_size=worker_l1_size)
            marker = "ring"
            profile = "torus-y"
        elif shape[0] == 1 and shape[1] > 2:
            device_params = torus_x_device_params(worker_l1_size=worker_l1_size)
            marker = "ring"
            profile = "torus-x"
        elif shape == (8, 4) and torus_xy_certified:
            device_params = torus_xy_device_params(worker_l1_size=worker_l1_size)
            marker = "mesh-8x4"
            profile = "torus-xy"
        else:
            device_params = fabric2d_device_params(worker_l1_size=worker_l1_size)
            marker = f"mesh-{shape[0]}x{shape[1]}"
            profile = "fabric2d"
        marks = [] if marker is None else pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=marker)
        params.append(
            pytest.param(
                shape,
                device_params,
                marks=marks,
                id=f"{profile}-sp{shape[0]}xtp{shape[1]}",
            )
        )
    if not params:
        params.append(
            pytest.param(
                (2, 2),
                fabric2d_device_params(worker_l1_size=worker_l1_size),
                marks=pytest.mark.skip(reason=f"unsupported device count {detect_num_devices()}"),
                id="unsupported",
            )
        )
    return pytest.mark.parametrize("mesh_device,device_params", params, indirect=["mesh_device", "device_params"])
