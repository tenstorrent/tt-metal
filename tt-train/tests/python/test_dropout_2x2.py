# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.ops.dropout.dropout`` on a ``[2, 2]`` ("dp", "tp") mesh, which draws one mask for the whole mesh or one per
device as its input's label says.

Copies of a tensor must draw one mask, so they stay copies; shards must draw their own, or every shard drops the same
positions. A tensor copied along one axis and sharded along the other, like the residual stream of DP x TP, needs
masks that ttnn's dropout cannot draw, so it is refused rather than masked wrongly.

Every device holds ones, so each output is that device's mask times 1 / (1 - p). Skips on a host with fewer than four
chips.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ttnn
import ttml
from tp_block import device_slices

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

PROB = 0.75  # not 0.5, so a kernel dropping 1 - p shows; its scale, 4, is exact in bf16
SCALE = 1.0 / (1.0 - PROB)
SHAPE = (4, 2, 64, 64)
R, S0, S1 = ttnn.PlacementReplicate(), ttnn.PlacementShard(0), ttnn.PlacementShard(1)
PLACEMENTS = {
    "replicated": [R, R],
    "sharded_across_both": [S0, S1],
    "sharded_across_dp": [S0, R],
    "sharded_across_tp": [R, S1],
}
# The kernel's draws are correlated, so they stray further than independent ones would: over 20 calls the drop rates
# came within 0.07 of p, and two masks (two devices', or one device's on consecutive calls) disagreed on at least 0.30
# of their positions, against 0.375 for independent masks. One seed shared by mistake gives exactly 0.
RATE_TOL = 0.15
MIN_DISAGREEMENT = 0.1


@pytest.fixture(scope="module")
def dp_tp_mesh(fresh_device_mesh):
    with fresh_device_mesh((2, 2), ("dp", "tp"), what="dropout on a [2, 2] mesh", require_mgd=True) as mesh:
        yield mesh


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def ones(layout: str) -> ttml.autograd.Tensor:
    device = ttml.autograd.AutoContext.get_instance().get_device()
    if layout == "whole_mesh":  # HSDP's batch mapper: one Shard(0) over all four devices, a collapsed 1-D label
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0, None)
    else:
        mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(PLACEMENTS[layout]))
    data = np.ones(SHAPE, np.float32)
    value = ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()
    if layout == "whole_mesh":
        assert len(value.tensor_topology().placements()) == 1, "precondition: the collapsed 1-D label"
    return ttml.autograd.create_tensor(value, True)


def disagreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a == 0) != (b == 0)).mean())


@pytest.mark.parametrize(
    "layout, one_mask", [("replicated", True), ("sharded_across_both", False), ("whole_mesh", False)]
)
def test_copies_share_a_mask_and_shards_draw_their_own(dp_tp_mesh, layout, one_mask):
    x = ones(layout)
    first = ttml.ops.dropout.dropout(x, PROB)
    second = ttml.ops.dropout.dropout(x, PROB)  # a program-cache hit, which must take the next seed

    outputs = [device_slices(out.get_value()) for out in (first, second)]
    for call, pieces in enumerate(outputs, start=1):
        for device, piece in enumerate(pieces):
            what = f"call {call}, device {device}"
            assert set(np.unique(piece)) <= {0.0, SCALE}, f"{what}: kept values are not scaled by {SCALE}"
            assert abs((piece == 0).mean() - PROB) < RATE_TOL, f"{what} dropped {(piece == 0).mean():.3f}"
        for (i, a), (j, b) in itertools.combinations(enumerate(pieces), 2):
            if one_mask:
                np.testing.assert_array_equal(a, b, err_msg=f"call {call}: devices {i} and {j} hold copies")
            else:
                assert disagreement(a, b) > MIN_DISAGREEMENT, f"call {call}: devices {i} and {j} share a mask"
    for device, (a, b) in enumerate(zip(*outputs)):
        assert disagreement(a, b) > MIN_DISAGREEMENT, f"device {device}: the second call repeated the first's mask"

    # The backward masks the gradient as the forward masked the input, though the generator has moved on.
    ttml.ops.unary.mean(first).backward(False)
    for device, (grad, out) in enumerate(zip(device_slices(x.get_grad()), outputs[0])):
        np.testing.assert_array_equal(grad != 0, out != 0, err_msg=f"device {device}")


@pytest.mark.parametrize(
    "layout, replicated_axis, sharded_axis", [("sharded_across_dp", 1, 0), ("sharded_across_tp", 0, 1)]
)
def test_copies_along_one_axis_and_shards_along_the_other_are_refused(
    dp_tp_mesh, expect_error, layout, replicated_axis, sharded_axis
):
    x = ones(layout)
    axes = f"replicated across mesh axis {replicated_axis} and sharded across mesh axis {sharded_axis}"
    with expect_error(RuntimeError, f"dropout cannot mask a tensor {axes}"):
        ttml.ops.dropout.dropout(x, PROB)
    # Probability 0, the way out the error names, passes the tensor through.
    passed = ttml.ops.dropout.dropout(x, 0.0).get_value()
    for device, (out, piece) in enumerate(zip(device_slices(passed), device_slices(x.get_value()))):
        np.testing.assert_array_equal(out, piece, err_msg=f"device {device}")
