# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CCLs on different mesh axes must not share semaphores while device groups are out of step.

Regression test for the ``deepseek_moe_16b_tp8_dp4`` Galaxy hang. A collective keeps only the
devices along its own axis in step, so when every axis drew from one round-robin semaphore pool, a
row that ran ahead signalled a lagging row on a semaphore that row was still using for a collective
on the other axis, and the later collective waited forever (see ``CCLResources``).

Each iteration delays every row except row 0, then runs a sequence of collectives: all but the last
on axis 1, the last on axis 0. Row 0 reaches the axis-0 op while the other rows are still delayed.
With one set of 2-deep pools shared across axes, every sequence puts the axis-0 op on the handshake
semaphore of its first axis-1 op, so row 0's handshake lands on a semaphore the lagging rows will use
for that op. Column 1 of each lagging row is delayed longer than column 0, so column 0's axis-1
handshake is waiting there too. Column 1 takes both as one handshake and resets the semaphore to 0,
dropping row 0's signal, and the axis-0 op hangs. Without the stagger, the stray signal stands in for
a later one, the counts balance, and the test passes even with a shared pool.

all_reduce takes its handshakes from one pool and all_gather and reduce_scatter from another, so each
op gets a sequence of its own. The mixed sequence follows the MoE expert-parallel layer: an all_gather
in, an all_reduce of the partial sums, and the reduce_scatter that is the all_gather's backward, so
one axis's pools also serve different ops in turn.

Run with the operation timeout armed so a hang fails instead of blocking::

    TT_METAL_OPERATION_TIMEOUT_SECONDS=60 pytest tt-train/tests/python/test_ccl_cross_axis_skew.py -v
"""

from __future__ import annotations

from typing import Callable, NamedTuple

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
# all_gather and reduce_scatter work on the last dim. Its gathered and scattered widths are both whole
# tiles, so neither op falls back to the composite path, which brings its own semaphores.
SPLIT_DIM = 3
NUM_ITERS = 4
# Must outlast the host dispatch of a sequence plus row 0 running its axis-1 ops, so row 0 reaches the
# axis-0 op while the other rows are still spinning. About 0.2 s at Blackhole's AI clock.
LAG_CYCLES = 300_000_000
# Column 0 of each lagging row leaves the delay first, so its axis-1 handshake is already waiting on
# column 1 next to row 0's stray axis-0 handshake, and column 1's reset to 0 drops one of them. Only
# has to cover the handshake's trip over the fabric; about 20 ms leaves room for dispatch jitter.
STAGGER_CYCLES = 30_000_000


@pytest.fixture(scope="module")
def mesh_2d(fresh_device_mesh):
    with fresh_device_mesh(MESH_SHAPE, ("dp", "tp"), what="CCL cross-axis skew test", require_mgd=True) as mesh:
        yield mesh


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _per_device_data(seed: int) -> np.ndarray:
    """Small integers, one block per device, so every partial sum is exact in bf16."""
    rng = np.random.default_rng(seed)
    return rng.integers(-2, 3, size=(*MESH_SHAPE, *PER_DEVICE_SHAPE[2:]), dtype=np.int8).astype(np.float32)


def _to_device(data: np.ndarray):
    """Device ``(r, c)`` gets ``data[r, c]`` as a ``PER_DEVICE_SHAPE`` tensor."""
    mapper = ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]))
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()


def _to_numpy(tensor) -> np.ndarray:
    """Inverse of ``_to_device``: device ``(r, c)`` lands at ``[r, c]``."""
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig([0, 1]))
    return ttnn.to_torch(tensor, mesh_composer=composer).float().numpy()


def _expected_all_reduce(data: np.ndarray, cluster_axis: int) -> np.ndarray:
    """What every device ``(r, c)`` should hold after an all_reduce along ``cluster_axis``."""
    return np.broadcast_to(data.sum(axis=cluster_axis, keepdims=True), data.shape)


def _expected_all_gather(data: np.ndarray, cluster_axis: int) -> np.ndarray:
    """After an all_gather along ``cluster_axis``, every device holds its group's blocks side by side, in group order."""
    blocks = [np.take(data, i, axis=cluster_axis) for i in range(data.shape[cluster_axis])]
    return np.stack([np.concatenate(blocks, axis=-1)] * len(blocks), axis=cluster_axis)


def _expected_reduce_scatter(data: np.ndarray, cluster_axis: int) -> np.ndarray:
    """After a reduce_scatter along ``cluster_axis``, device ``i`` of each group holds slice ``i`` of the group's sum."""
    group_size = data.shape[cluster_axis]
    return np.stack(np.split(data.sum(axis=cluster_axis), group_size, axis=-1), axis=cluster_axis)


class Collective(NamedTuple):
    run: Callable
    expected: Callable[[np.ndarray, int], np.ndarray]


COLLECTIVES = {
    "all_reduce": Collective(
        lambda tensor, axis: ttml.core.distributed.all_reduce(tensor, cluster_axis=axis), _expected_all_reduce
    ),
    "all_gather": Collective(
        lambda tensor, axis: ttml.core.distributed.all_gather(tensor, SPLIT_DIM, cluster_axis=axis),
        _expected_all_gather,
    ),
    "reduce_scatter": Collective(
        lambda tensor, axis: ttml.core.distributed.reduce_scatter(tensor, SPLIT_DIM, cluster_axis=axis),
        _expected_reduce_scatter,
    ),
}

# Every op runs on axis 1 except the last, which runs on axis 0. The first and last ops take their
# handshake from the same pool with exactly one draw from it in between, so a 2-deep pool shared across
# axes gives them the same semaphore. They are also the same op, so the last places its workers like
# the first and its stray handshake reaches the cores the lagging rows' first op waits on.
SEQUENCES = {
    "all_reduce": ("all_reduce", "all_reduce", "all_reduce"),
    "all_gather": ("all_gather", "all_gather", "all_gather"),
    "reduce_scatter": ("reduce_scatter", "reduce_scatter", "reduce_scatter"),
    # all_reduce has a handshake pool of its own, so the pool all_gather and reduce_scatter share sees
    # all_gather, reduce_scatter, all_gather.
    "mixed": ("all_gather", "all_reduce", "reduce_scatter", "all_gather"),
}


@pytest.mark.parametrize("ops", list(SEQUENCES.values()), ids=list(SEQUENCES))
def test_cross_axis_collectives_with_lagging_rows(mesh_2d, ops):
    rows, cols = MESH_SHAPE
    lag = [[0 if r == 0 else LAG_CYCLES + c * STAGGER_CYCLES for c in range(cols)] for r in range(rows)]

    axes = [TP_AXIS] * (len(ops) - 1) + [DP_AXIS]
    data = [_per_device_data(seed=i) for i in range(len(ops))]
    steps = [(COLLECTIVES[op], _to_device(d), axis) for op, d, axis in zip(ops, data, axes)]

    # Compile every program with a full sync after each, so the skew below is not hidden behind
    # host-side compilation and the warm-up itself cannot collide.
    ttnn.apply_device_delay(_device(), lag)
    ttnn.synchronize_device(_device())
    for collective, tensor, axis in steps:
        collective.run(tensor, axis)
        ttnn.synchronize_device(_device())

    results = []
    for _ in range(NUM_ITERS):
        ttnn.apply_device_delay(_device(), lag)
        results.append([collective.run(tensor, axis) for collective, tensor, axis in steps])
    ttnn.synchronize_device(_device())

    for it, outputs in enumerate(results):
        for i, (op, axis, d, tensor) in enumerate(zip(ops, axes, data, outputs)):
            np.testing.assert_array_equal(
                _to_numpy(tensor),
                COLLECTIVES[op].expected(d, axis),
                err_msg=f"iteration {it} op {i} (axis-{axis} {op})",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
