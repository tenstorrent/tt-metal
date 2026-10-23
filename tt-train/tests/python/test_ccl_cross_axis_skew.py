# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CCLs on different mesh axes must not share semaphores while device groups are out of step.

Regression test for the hang in the ``deepseek_moe_16b_tp8_dp4`` Galaxy leg. ``CCLResources``
used to hand every collective the next entry of one 2-deep pool of global semaphores, whichever
mesh axis it ran on, and each semaphore lives at the same L1 address on every chip. Collectives
along one axis keep only that axis's group in step, so the groups along the other axis can
drift apart (the DP replicas did during MoE backward). A leading chip that moved on to a
collective along the other axis then signalled a lagging peer on the semaphore that peer was
still using for its own collective. The kernels ``wait_min`` and then reset to 0, so the
signal was consumed by the wrong op or wiped, and the later collective waited forever.

The test makes that drift deterministic on a 2D mesh. Each iteration:

  1. ``ttnn.apply_device_delay`` spins every chip outside mesh row 0 for ``LAG_CYCLES``.
  2. Two all_reduces along axis 1 (within each mesh row). Row 0 finishes both while the other
     rows are still spinning.
  3. One all_reduce along axis 0 (within each mesh column). Row 0 enters it and signals the
     lagging rows. With a single shared pool, this op gets the same semaphore set as step 2's
     first all_reduce, which the lagging rows have not run yet.

With per-axis pools the axis-0 and axis-1 collectives never share a semaphore, so every
iteration completes with exact results. With a shared pool the lagging rows consume row 0's
signal in the wrong collective: the test hangs or returns wrong sums.

Run with the operation timeout armed so a hang fails instead of blocking until the pytest
timeout kills the process::

    TT_METAL_OPERATION_TIMEOUT_SECONDS=60 pytest tt-train/tests/python/test_ccl_cross_axis_skew.py -v

After a hang the device is wedged; closing it can abort the process.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import ttnn
import ttml

pytestmark = [
    pytest.mark.requires_device,
    # Opening the mesh and compiling the CCL programs for both axes can exceed the default 300 s.
    pytest.mark.timeout(900, method="thread"),
]

MESH_SHAPE = (2, 2)
DP_AXIS = 0  # collective within each mesh column
TP_AXIS = 1  # collective within each mesh row
PER_DEVICE_SHAPE = (1, 1, 64, 256)
NUM_ITERS = 4
# Must outlast the host dispatch of the three all_reduces plus row 0 running both axis-1
# all_reduces, so row 0 reaches the axis-0 all_reduce while the other rows are still spinning.
# About 0.2 s at Blackhole's AI clock.
LAG_CYCLES = 300_000_000


@pytest.fixture(scope="module")
def mesh_2d(fresh_device_mesh):
    with fresh_device_mesh(MESH_SHAPE, ("dp", "tp"), what="CCL cross-axis skew test", require_mgd=True) as mesh:
        yield mesh


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


@contextlib.contextmanager
def _fail_on_hang(what: str):
    try:
        yield
    except Exception as e:  # noqa: BLE001
        if "TIMEOUT" in str(e):
            pytest.fail(f"{what} hung: {e}", pytrace=False)
        raise


def _per_device_data(seed: int) -> np.ndarray:
    """Small integers, one block per device, so every partial sum is exact in bf16."""
    rng = np.random.default_rng(seed)
    rows, cols = MESH_SHAPE
    return rng.integers(-2, 3, size=(rows, cols, *PER_DEVICE_SHAPE[2:]), dtype=np.int8).astype(np.float32)


def _to_device(data: np.ndarray):
    """Device ``(r, c)`` gets ``data[r, c]`` as a ``PER_DEVICE_SHAPE`` tensor."""
    mapper = ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]))
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()


def _expected_all_reduce(data: np.ndarray, cluster_axis: int) -> np.ndarray:
    """What every device ``(r, c)`` should hold after an all_reduce along ``cluster_axis``."""
    return np.broadcast_to(data.sum(axis=cluster_axis, keepdims=True), data.shape)


def _assert_per_device(tensor, expected: np.ndarray, what: str):
    # Concatenate mesh axis 0 into dim 0 and mesh axis 1 into dim 1: device (r, c) lands at [r, c].
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig([0, 1]))
    actual = ttnn.to_torch(tensor, mesh_composer=composer).float().numpy()
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape}, expected {expected.shape}"
    rows, cols = MESH_SHAPE
    for r in range(rows):
        for c in range(cols):
            np.testing.assert_array_equal(actual[r, c], expected[r, c], err_msg=f"{what}: device ({r}, {c})")


def test_cross_axis_all_reduce_with_lagging_rows(mesh_2d):
    rows, cols = MESH_SHAPE
    lag = [[0 if r == 0 else LAG_CYCLES for _ in range(cols)] for r in range(rows)]

    row_a_data = _per_device_data(seed=0)
    row_b_data = _per_device_data(seed=1)
    col_data = _per_device_data(seed=2)
    row_a_in = _to_device(row_a_data)
    row_b_in = _to_device(row_b_data)
    col_in = _to_device(col_data)
    expected = {
        "axis-1 all_reduce A": _expected_all_reduce(row_a_data, TP_AXIS),
        "axis-1 all_reduce B": _expected_all_reduce(row_b_data, TP_AXIS),
        "axis-0 all_reduce": _expected_all_reduce(col_data, DP_AXIS),
    }

    # Compile every program with a full sync after each, so the skew below is not hidden behind
    # host-side compilation and the warm-up itself cannot collide.
    with _fail_on_hang("warm-up"):
        ttnn.apply_device_delay(_device(), lag)
        ttnn.synchronize_device(_device())
        for tensor, axis in ((row_a_in, TP_AXIS), (col_in, DP_AXIS)):
            ttml.core.distributed.all_reduce(tensor, cluster_axis=axis)
            ttnn.synchronize_device(_device())

    results = []
    with _fail_on_hang("skewed cross-axis all_reduces"):
        for _ in range(NUM_ITERS):
            ttnn.apply_device_delay(_device(), lag)
            row_a = ttml.core.distributed.all_reduce(row_a_in, cluster_axis=TP_AXIS)
            row_b = ttml.core.distributed.all_reduce(row_b_in, cluster_axis=TP_AXIS)
            col = ttml.core.distributed.all_reduce(col_in, cluster_axis=DP_AXIS)
            results.append({"axis-1 all_reduce A": row_a, "axis-1 all_reduce B": row_b, "axis-0 all_reduce": col})
        ttnn.synchronize_device(_device())

    for it, outputs in enumerate(results):
        for name, tensor in outputs.items():
            _assert_per_device(tensor, expected[name], f"iteration {it} {name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
