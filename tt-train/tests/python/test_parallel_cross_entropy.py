# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.parallel.cross_entropy_loss`` against a float64 reference, for logits laid out either way across tp."""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from tp_block import gather_like

pytestmark = [pytest.mark.requires_device, pytest.mark.usefixtures("tp_mesh")]

BATCH, SEQ_LEN, VOCAB = 2, 32, 256
# bf16 rounding: the loss is near 5.6, where bf16 steps by 1/32.
LOSS_TOLERANCE = 0.05
MAX_REL_ERROR = 0.02


def reference(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """The mean cross-entropy over every position, and its gradient."""
    x = logits.astype(np.float64)
    exp = np.exp(x - x.max(-1, keepdims=True))
    probs = exp / exp.sum(-1, keepdims=True)
    index = targets[:, None, :, None].astype(np.int64)
    onehot = np.zeros_like(probs)
    np.put_along_axis(onehot, index, 1.0, -1)
    return float(-np.log(np.take_along_axis(probs, index, -1)).mean()), (probs - onehot) / targets.size


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.parametrize(
    "placement", [ttnn.PlacementShard(3), ttnn.PlacementReplicate()], ids=["vocab_sharded", "replicated"]
)
def test_the_loss_does_not_depend_on_how_the_logits_are_laid_out(placement):
    rng = np.random.default_rng(3)
    values = rng.uniform(-3.0, 3.0, (BATCH, 1, SEQ_LEN, VOCAB)).astype(np.float32)
    targets = rng.integers(0, VOCAB, (BATCH, SEQ_LEN)).astype(np.uint32)
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), placement]))
    value = ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()
    logits = ttml.autograd.create_tensor(value, True)

    loss = ttml.parallel.cross_entropy_loss(
        logits,
        ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
        ttml.ops.ReduceType.MEAN,
    )
    loss.backward(False)
    assert logits.is_grad_initialized()
    grad = gather_like(logits.get_grad(), logits)

    expected_loss, expected_grad = reference(gather_like(logits.get_value(), logits), targets)
    actual_loss = float(gather_like(loss.get_value(), loss).item())
    assert abs(actual_loss - expected_loss) <= LOSS_TOLERANCE, f"loss {actual_loss}, expected {expected_loss}"
    error = np.linalg.norm(grad - expected_grad) / np.linalg.norm(expected_grad)
    assert error <= MAX_REL_ERROR, f"gradient: relative error {error:.3g}"
