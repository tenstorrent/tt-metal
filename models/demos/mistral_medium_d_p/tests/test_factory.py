# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared test scaffolding for the Mistral-Medium-3.5 prefill blocks.

Ported from ``minimax_m3/tests/test_factory.py``. Owns the mesh + fabric parametrization every
device test uses, plus the block builders both engineers share, so neither module's tests re-derive
mesh/CCL setup.

**Shared substrate — frozen.** Attention (``tt/attention/*``, ``tt/rope*.py``) and MLP (``tt/mlp.py``)
are owned by separate engineers; this file, ``config.py``, ``tt/ccl.py`` and ``utils/`` are not owned
by either and should not change as part of block work.
"""

import json
import os

import pytest

import ttnn

from ..config import MeshConfig
from ..tt.ccl import CCLManager
from ..utils.general_utils import get_default_num_links

_CONFIG_JSON = os.path.join(os.path.dirname(__file__), "..", "configs", "Mistral-Medium-3.5-128B", "config.json")

# Mesh shapes this model is tested on: the (8,4) Blackhole Galaxy target, SP=8 x TP=4, and nothing
# else. Shapes are added back one at a time as hardware becomes reachable, so a green run always
# means something was really tested rather than skipped.
#
# The (2,4) LoudBox rung was tried and REMOVED: no proper SUBMESH of a Galaxy can bring up the fabric.
# (2,4) opens fine with FabricConfig::DISABLED, but both FABRIC_1D_RING and FABRIC_1D die in the
# ethernet router handshake ("Fabric Router Sync: Timeout ... Master chan=3 got 0xa0b0c0d0"), because
# carving chips out of the 32-chip fabric leaves their ethernet partners outside the submesh with no
# router kernel running. Not ring-specific, and not specific to (2,4) - test_mlp_vs_ref's (1,4) rung
# fails identically on this box. Only the full (8,4) allocates. Not fixable from here; a smaller rung
# needs a real smaller machine. Device-count filtering alone cannot see this, hence no (2,4) entry.
#
# Not yet re-added: the column/row-parallel QKV + o_proj tests, and the (1,1) block-math rung.
MESH_SHAPES = [(8, 4)]


def mistral_config_dims() -> dict:
    """Mistral-Medium-3.5 dimension constants from the bundled config.json (no HF import needed)."""
    with open(_CONFIG_JSON) as f:
        return json.load(f)


def _linear_fabric_default() -> bool:
    """Whether to drive the fabric as a plain mesh rather than a ring.

    Defaults to the RING, which is production and what a healthy Galaxy provides. Set
    ``MISTRAL_LINEAR_FABRIC=1`` on a **plain-mesh Galaxy** — one whose wrap-around ethernet links
    are missing or untrained, so a torus cannot be embedded. Such a box reports a grid degree
    histogram (``mesh0 {2:4, 3:16, 4:12}`` for 8x4: 4 corners, 16 perimeter, 12 interior) where a
    torus needs ``{4:32}``, and FABRIC_1D_RING dies in topology_mapper with::

        Graph specified in MGD could not fit in the discovered physical topology ... STRICT

    Correctness is topology-independent, so PCC tests are valid either way. PERFORMANCE IS NOT: the
    ring-joint SDPA gathers KV along the SP axis, so its cost is exactly what changes between a ring
    and a linear fabric. Never compare perf numbers taken under this flag against ring-fabric ones.
    """
    return os.getenv("MISTRAL_LINEAR_FABRIC", "").lower() in ("1", "true", "yes")


def parametrize_mesh_with_fabric(mesh_shapes=None, linear_fabric=None):
    """Mesh + fabric parametrization for mistral_medium_d_p tests.

    Generates a paired ``(mesh_device, device_params)`` parametrize. Each case opens a mesh of the
    requested shape directly (no submesh carving in test bodies) and configures the fabric for that
    shape. Ids are ``1x1`` / ``1x4`` / ``2x4`` / ``8x4`` so ``pytest -k 8x4`` filters cleanly.

    Auto-filters by DEVICE COUNT to the shapes that fit on the current system, so a test declaring
    ``[(8,4)]`` simply skips on a smaller box rather than failing. Under ``CI=true`` only the largest
    fitting shape runs, letting one yaml entry target several SKUs.

    Note the filter is device-count only: it cannot tell that a shape which *fits* is nonetheless
    unallocatable as a submesh of this machine's fabric (see the (2,4)-on-Galaxy note above), so
    shapes must still be declared against real SKUs, not just chip budgets.

    Fabric: ``(1,1)`` disables it (nothing to ring around). Multi-device uses ``FABRIC_1D_RING``,
    matching the ring topology the CCLManager and every collective here use; pass
    ``linear_fabric=True`` for a plain-mesh Galaxy with no wrap-around links (and use
    ``ttnn.Topology.Linear`` in the CCLManager to match).
    """
    num_devices = ttnn.get_num_devices()
    linear_fabric = _linear_fabric_default() if linear_fabric is None else linear_fabric
    mesh_shapes = MESH_SHAPES if mesh_shapes is None else mesh_shapes
    mesh_shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]

    if os.getenv("CI") == "true" and len(mesh_shapes) > 1:
        mesh_shapes = [max(mesh_shapes, key=lambda s: s[0] * s[1])]

    if not mesh_shapes:
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None, "trace_region_size": 100000000},
                id="1x1",
                marks=pytest.mark.skip(reason="No supported mistral_medium_d_p mesh shape fits on this system"),
            )
        ]
    else:
        multidev_fabric = ttnn.FabricConfig.FABRIC_1D if linear_fabric else ttnn.FabricConfig.FABRIC_1D_RING
        params = [
            pytest.param(
                shape,
                {
                    "fabric_config": (None if shape == (1, 1) else multidev_fabric),
                    "trace_region_size": 100000000,
                },
                id=f"{shape[0]}x{shape[1]}",
            )
            for shape in mesh_shapes
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator


def mesh_setup(mesh_device, linear_fabric=None):
    """Build the ``(MeshConfig, CCLManager)`` pair for this mesh.

    TP is always the full col axis (``mesh_shape[1]``), so a (2,4) mesh is TP=4/SP=2 and an (8,4)
    mesh is TP=4/SP=8 — the **same** TP sharding and the same per-chip load on both, which is what
    lets the 8-chip rung retire the ring machinery before any Galaxy time is spent.

    ``ccl_manager`` is None at TP=1 and SP=1, where no collective ever runs.
    """
    rows, cols = tuple(mesh_device.shape)
    linear_fabric = _linear_fabric_default() if linear_fabric is None else linear_fabric
    mesh_config = MeshConfig((rows, cols), tp=cols)
    needs_ccl = cols > 1 or rows > 1
    ccl_manager = (
        CCLManager(
            mesh_device,
            num_links=get_default_num_links(mesh_device),
            topology=ttnn.Topology.Linear if linear_fabric else ttnn.Topology.Ring,
        )
        if needs_ccl
        else None
    )
    return mesh_config, ccl_manager


def replicate(t, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Push a torch tensor to every device unchanged (block inputs are full-emb replicated)."""
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def gather_tp_shards(tt_out, mesh_device):
    """Reassemble a reduce-scattered block output back to full emb, on the host.

    Every block returns ``[1, 1, s, hidden/tp]`` (the sharded-residual contract). Device index is
    ``row * cols + col``, and the shards are identical down each column, so concatenating row 0's
    devices along the last dim rebuilds the full hidden dim for comparison against the torch
    reference. At TP=1 this is just device 0.
    """
    import torch

    _, cols = tuple(mesh_device.shape)
    dev = ttnn.get_device_tensors(tt_out)
    if cols == 1:
        return ttnn.to_torch(dev[0])
    return torch.cat([ttnn.to_torch(dev[c]) for c in range(cols)], dim=-1)
