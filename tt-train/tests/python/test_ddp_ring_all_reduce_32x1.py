# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DDP gradient all-reduce on a 32x1 RING,RING Blackhole Galaxy mesh.

Reproduces the hang seen in the ``qwen3_1_7b_ddp32_shakespeare`` models e2e leg:
the very first gradient all-reduce after backward never completes. The
reduce-scatter half finishes, the all-gather half hangs with fabric senders
stuck waiting for receiver credits (``get_num_unprocessed_acks_from_receiver``)
and all-gather readers stuck in ``cb_reserve_back``.

Setup mirrors that run exactly:
  * MGD ``configs/mgd/bh_galaxy_32_1_ring_ring.textproto`` -> tt-train infers
    ``FABRIC_2D_TORUS_XY`` and the CCL ops pick ``Topology::Ring`` on axis 0.
  * Gradient ``[1, 1, 2048, 2048]`` bf16 TILE DRAM-interleaved, all-reduced on
    cluster axis 0, which ``all_reduce_async`` lowers to
    ``reduce_scatter_minimal_async`` (dim 3 -> ``[1, 1, 2048, 64]`` per device)
    followed by ``all_gather_async``.

Tests run from most to least isolated so a single run shows which phase hangs.
Once one hangs the device is unrecoverable and the remaining tests are skipped;
use ``-k`` to run a single case.

Run on a 32-chip Blackhole Galaxy with the operation timeout armed so a hang
fails instead of blocking forever::

    TT_METAL_OPERATION_TIMEOUT_SECONDS=120 \\
        pytest tt-train/tests/python/test_ddp_ring_all_reduce_32x1.py -v
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

import ttnn
import ttml


pytestmark = [
    pytest.mark.requires_device,
    # C++ hangs never return to Python, and closing a wedged mesh can hang too.
    pytest.mark.timeout(1800, method="thread"),
]

MESH_SHAPE = (32, 1)
RING_SIZE = MESH_SHAPE[0]
DDP_AXIS = 0
GRAD_SHAPE = (1, 1, 2048, 2048)
SCATTER_DIM = 3
NUM_ITERS = 10

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_PATH = os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_32_1_ring_ring.textproto")

_device_hung = False


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


@contextlib.contextmanager
def _fail_on_hang(what: str):
    """Turn a dispatch timeout into a clean test failure and poison later tests."""
    global _device_hung
    try:
        yield
    except Exception as e:  # noqa: BLE001
        if "TIMEOUT" in str(e):
            _device_hung = True
            pytest.fail(f"{what} hung on the {MESH_SHAPE} FABRIC_2D_TORUS_XY ring: {e}", pytrace=False)
        raise


@pytest.fixture(scope="module")
def ring_mesh():
    if "blackhole" not in ttnn.get_arch_name().lower():
        pytest.skip("Needs a Blackhole Galaxy.")
    available = int(ttnn.distributed.get_num_devices())
    if available < RING_SIZE:
        pytest.skip(f"Needs {RING_SIZE} chips; host has {available}.")

    previous_mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = _MGD_PATH
    try:
        ttml.open_device_mesh(ttml.Mesh(MESH_SHAPE, ("dp", "_1")))
        assert (
            ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_2D_TORUS_XY
        ), f"Expected FABRIC_2D_TORUS_XY inferred from {_MGD_PATH}, got {ttnn.get_fabric_config()}"
        yield ttml.mesh()
    finally:
        try:
            ttml.close_device_mesh()
        finally:
            if previous_mgd is None:
                os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
            else:
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous_mgd


@pytest.fixture(autouse=True)
def _skip_after_hang(ring_mesh):
    if _device_hung:
        pytest.skip("Device hung in an earlier test.")


def _to_device(data: np.ndarray, placements):
    mapper = ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig(list(placements)))
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()


def _read(t, composer_dims) -> np.ndarray:
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig(list(composer_dims)))
    return ttnn.to_torch(t, mesh_composer=composer).float().numpy()


def _per_device_grads(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct small-integer "gradients" per device and their exact sum.

    Magnitudes stay <= 2 * RING_SIZE, so every partial sum is exact in bf16 and
    any mismatch means lost or misrouted data rather than rounding.
    """
    rng = np.random.default_rng(seed)
    grads = rng.integers(-2, 3, size=(RING_SIZE, *GRAD_SHAPE[1:]), dtype=np.int8).astype(np.float32)
    return grads, grads.sum(axis=0, keepdims=True)


def _assert_replicated_equals(t, expected: np.ndarray, what: str):
    per_device = _read(t, (0, 1))
    assert per_device.shape == (RING_SIZE, *expected.shape[1:]), f"{what}: shape {per_device.shape}"
    for dev in range(RING_SIZE):
        np.testing.assert_array_equal(per_device[dev : dev + 1], expected, err_msg=f"{what}: device {dev}")


def test_all_gather_alone(ring_mesh):
    """all_gather of a dim-3 shard, with no reduce-scatter before it."""
    rng = np.random.default_rng(0)
    full = rng.integers(-2, 3, size=GRAD_SHAPE, dtype=np.int8).astype(np.float32)
    shard = _to_device(full, [ttnn.PlacementShard(SCATTER_DIM), ttnn.PlacementReplicate()])
    assert list(shard.shape) == [1, 1, GRAD_SHAPE[2], GRAD_SHAPE[3] // RING_SIZE]

    with _fail_on_hang("all_gather"):
        gathered = ttml.core.distributed.all_gather(shard, SCATTER_DIM, cluster_axis=DDP_AXIS)
        ttnn.synchronize_device(_device())
        _assert_replicated_equals(gathered, full, "all_gather")


def test_reduce_scatter_then_all_gather(ring_mesh):
    """The two halves of the DDP all-reduce with a device sync between them."""
    grads, expected = _per_device_grads(seed=1)
    grad = _to_device(grads, [ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

    with _fail_on_hang("reduce_scatter"):
        scattered = ttml.core.distributed.reduce_scatter(grad, SCATTER_DIM, cluster_axis=DDP_AXIS)
        ttnn.synchronize_device(_device())
        assert list(scattered.shape) == [1, 1, GRAD_SHAPE[2], GRAD_SHAPE[3] // RING_SIZE]
        np.testing.assert_array_equal(_read(scattered, (SCATTER_DIM, 1)), expected, err_msg="reduce_scatter")

    with _fail_on_hang("all_gather after reduce_scatter"):
        gathered = ttml.core.distributed.all_gather(scattered, SCATTER_DIM, cluster_axis=DDP_AXIS)
        ttnn.synchronize_device(_device())
        _assert_replicated_equals(gathered, expected, "all_gather after reduce_scatter")


def test_ddp_gradient_all_reduce(ring_mesh):
    """Exact call made by ``synchronize_gradients`` for a DDP=32 run, repeated."""
    grads, expected = _per_device_grads(seed=2)
    grad = _to_device(grads, [ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

    for it in range(NUM_ITERS):
        with _fail_on_hang(f"all_reduce iteration {it}"):
            reduced = ttml.core.distributed.all_reduce(grad, cluster_axis=DDP_AXIS)
            ttnn.synchronize_device(_device())
            _assert_replicated_equals(reduced, expected, f"all_reduce iteration {it}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
