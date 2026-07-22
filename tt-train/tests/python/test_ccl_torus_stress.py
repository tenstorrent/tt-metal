# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fast, model-free reproducer for the ring-CCL hang seen in the Galaxy
Llama 8B TP=4 DDP=8 training perf test.

Root cause context (see the training hang investigation): the training run
stalls inside ``ring_reduce_scatter_minimal_async`` during the TP loss /
gradient reduction. That op is only selected when the fabric is a **torus**
(``get_topology()`` returns ``Ring`` for a ``ring_ring`` mesh). On some Galaxy
runners a marginal torus wrap-around link makes the ring collective deadlock,
so the failure is intermittent and runner-specific.

This test strips away the 8B model, dataset, and optimizer and just hammers the
same collectives (``all_reduce`` on the DDP axis, ``reduce_scatter`` on the TP
axis) on the **same 8x4 ring_ring torus mesh** via the **same**
``ttml.core.distributed`` entry points the trainer uses. It reaches the ring
reduce_scatter kernels within seconds, so it can be run repeatedly to bisect
which runners are unhealthy.

Run it (ideally with the watcher, to turn a silent hang into a named assert)::

    TT_METAL_WATCHER=10 \
    TT_METAL_WATCHER_DISABLE_ETH=1 \
    pytest tt-train/tests/python/test_ccl_torus_stress.py -v --timeout=120

A healthy runner completes all iterations quickly. A bad runner hangs (the
pytest ``--timeout`` fires, and the watcher dumps the stuck reduce_scatter core
to generated/watcher/watcher.log).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pytest

import ttnn
import ttml


pytestmark = pytest.mark.requires_device

# Match the failing job's mesh: bh_galaxy_8_4_ring_ring (32-device torus).
# Axis 0 (size 8) is the DDP axis; axis 1 (size 4) is the TP axis.
MESH_SHAPE = (8, 4)
DDP_AXIS = 0
TP_AXIS = 1
DDP_SIZE = MESH_SHAPE[DDP_AXIS]
TP_SIZE = MESH_SHAPE[TP_AXIS]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RING_MGD = os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_8_4_ring_ring.textproto")


def _detect_arch() -> Optional[str]:
    try:
        name = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    if "blackhole" in name:
        return "blackhole"
    if "wormhole_b0" in name:
        return "wormhole_b0"
    return None


def _num_available_devices() -> int:
    try:
        return int(ttnn.distributed.get_num_devices())
    except Exception:  # noqa: BLE001
        return 0


def _close_device_quietly() -> None:
    try:
        ttml.autograd.AutoContext.get_instance().close_device()
    except Exception:  # noqa: BLE001
        pass


def _clear_global_mesh() -> None:
    try:
        import ttml._mesh as _mesh_mod  # type: ignore[import-not-found]

        _mesh_mod._mesh = None
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="module")
def torus_mesh():
    """Open the 8x4 ring_ring torus mesh, or skip if the host can't host it."""
    needed = DDP_SIZE * TP_SIZE
    available = _num_available_devices()
    if available and available < needed:
        pytest.skip(f"Torus CCL stress needs a {MESH_SHAPE} mesh ({needed} chips); host has {available}.")
    if _detect_arch() != "blackhole":
        pytest.skip("Torus CCL stress reproducer targets Blackhole Galaxy.")
    if not os.path.isfile(_RING_MGD):
        pytest.skip(f"Missing ring_ring MGD: {_RING_MGD}")

    previous_mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = _RING_MGD
    _close_device_quietly()
    _clear_global_mesh()
    try:
        ttml.open_device_mesh(MESH_SHAPE)
    except Exception as e:  # noqa: BLE001
        if previous_mgd is None:
            os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
        else:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous_mgd
        pytest.skip(f"Could not open {MESH_SHAPE} torus mesh: {e}")

    yield ttml.mesh()

    _close_device_quietly()
    _clear_global_mesh()
    if previous_mgd is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous_mgd


def _make_tensor(shape, placements, *, seed: int = 0):
    """Create a raw bf16 ttnn tensor with explicit per-mesh-axis ``placements``."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(list(placements)))
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32) * 0.1
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()


def _replicated(shape, *, seed: int = 0):
    return _make_tensor(shape, [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], seed=seed)


# Sweep shapes small -> large. Each is (N, C, H, W); H must be divisible by 32
# (tile) and W by 32*TP_SIZE so reduce_scatter can shard dim 3 tile-aligned
# across the TP axis. A size-dependent deadlock (e.g. only when a slice spans
# multiple pages / crosses a chunking boundary) shows up as the first shape
# that hangs.
_SHAPES = [
    (1, 1, 32, 128),  # tiny: one tile row, one tile per shard
    (1, 1, 256, 512),  # small
    (1, 1, 1024, 2048),  # medium (resembles a large gradient reduce)
    (1, 1, 2048, 4096),  # large
]
_SHAPE_IDS = [f"{h}x{w}" for (_, _, h, w) in _SHAPES]

# Fewer iters as the tensors grow so total runtime stays bounded while still
# hammering the ring many times at every size.
_ITERS_PER_SHAPE = 50


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_all_reduce_torus_stress(torus_mesh, shape):
    """Repeated ring all_reduce on the DDP axis (torus). Should never hang."""
    for i in range(_ITERS_PER_SHAPE):
        tensor = _replicated(shape, seed=i)
        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=DDP_AXIS)
        ttnn.deallocate(result)
        ttnn.deallocate(tensor)
        ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_reduce_scatter_torus_stress(torus_mesh, shape):
    """Repeated ring reduce_scatter on the TP axis (torus) — the exact op that
    tripped the watcher assert in the training hang. Should never hang."""
    scatter_dim = 3  # W / TP_SIZE per shard, tile-aligned by construction
    for i in range(_ITERS_PER_SHAPE):
        tensor = _replicated(shape, seed=i)
        result = ttml.core.distributed.reduce_scatter(tensor, scatter_dim, cluster_axis=TP_AXIS)
        ttnn.deallocate(result)
        ttnn.deallocate(tensor)
        ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_step_like_torus_stress(torus_mesh, shape):
    """Mimic one training step's collective pattern: a TP-axis reduce_scatter
    (loss/grad reduction) followed by a DDP-axis all_reduce (gradient sync),
    looped to stress both torus rings the way the trainer does."""
    scatter_dim = 3
    for i in range(_ITERS_PER_SHAPE):
        a = _replicated(shape, seed=i)
        rs = ttml.core.distributed.reduce_scatter(a, scatter_dim, cluster_axis=TP_AXIS)
        ttnn.deallocate(a)

        b = _replicated(shape, seed=i + 10_000)
        ar = ttml.core.distributed.all_reduce(b, cluster_axis=DDP_AXIS)
        ttnn.deallocate(b)

        ttnn.deallocate(rs)
        ttnn.deallocate(ar)
        ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120"])
