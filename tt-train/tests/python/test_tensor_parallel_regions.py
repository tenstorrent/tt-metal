# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel regions read layouts off the tensors (``ttml.parallel``).

A region's input is made whole whatever layout it arrives in, and its partial results are summed into the layout the
caller asks for. Consumers read layouts too: the loss finds the axis the vocabulary is sharded on, dropout draws one
mask per device only when no two devices hold the same values. Every case is checked against the same computation on
full, replicated weights.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models.deepseek.transformer import DeepSeekMLP
from ttml.modules import ColumnParallelLinear, LinearLayer, RowParallelLinear
from ttml.parallel import SEQUENCE_DIM, sum_partials, whole

pytestmark = [pytest.mark.requires_device, pytest.mark.usefixtures("tp_mesh")]

SEQ, DIM = 64, 64
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def tp() -> int:
    return ttml.mesh().axis_index("tp")


def tensor(data: np.ndarray, placement=None, requires_grad: bool = True):
    """``data`` placed on the mesh: replicated, or split by ``placement`` across tp."""
    mapper = None if placement is None else ttml.mesh().axis_mapper("tp", tdim=placement.dim)
    t = ttml.autograd.Tensor.from_numpy(data.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.FLOAT32, mapper)
    t.set_requires_grad(requires_grad)
    return t


def full(t) -> np.ndarray:
    """The full tensor, gathered by its own layout."""
    return ttml.Sharding.from_tensor(t).gather(t).astype(np.float64)


def full_grad(t) -> np.ndarray:
    return full(t.get_grad_tensor())


def ranks(value) -> list[np.ndarray]:
    return [ttnn.to_torch(shard).float().numpy() for shard in ttnn.get_device_tensors(value)]


def shard_dim(t) -> int | None:
    return ttml.Sharding.from_tensor(t).shard_dim(tp())


def random(shape, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape) * 0.1


def copy_weights(src, dst) -> None:
    """``dst``'s parameters set to ``src``'s full values, each placed as ``dst`` places it."""
    src_params = src.parameters()
    for name, param in dst.parameters().items():
        source = src_params[name.replace(dst.get_name(), src.get_name(), 1)]
        mapper = ttml.Sharding.from_tensor(param).derive_mapper()
        ttml.autograd.assign_numpy(param, full(source).astype(np.float32), layout=ttnn.Layout.TILE, mapper=mapper)


def assert_close(got, expected, what: str) -> None:
    np.testing.assert_allclose(got, expected, rtol=3e-2, atol=3e-2 * float(np.abs(expected).max()), err_msg=what)


class TestRegionEdges:
    def test_whole_gathers_a_sharded_input_and_passes_a_replicated_one(self):
        data = random((1, 1, SEQ, DIM), 0)
        for placement in (None, ttnn.PlacementShard(SEQUENCE_DIM)):
            x = whole(tensor(data, placement), tp())
            assert shard_dim(x) is None
            for rank in ranks(x.get_value(NATIVE)):
                assert_close(rank, data, f"whole of {placement}")

    @pytest.mark.parametrize("layout", [None, ttnn.PlacementShard(SEQUENCE_DIM)], ids=["replicated", "sequence"])
    def test_sum_partials_lands_in_the_layout_asked_for(self, layout):
        # Each rank holds a different full-shape tensor: rank r's slice r of a [2, ...] stack.
        stacked = random((2, 1, SEQ, DIM), 1)
        partial = tensor(stacked, ttnn.PlacementShard(0))
        out = sum_partials(partial, tp(), layout)
        assert shard_dim(out) == (None if layout is None else SEQUENCE_DIM)
        assert_close(full(out), stacked.sum(axis=0, keepdims=True), "sum of the partials")


class TestParallelLinears:
    def test_column_parallel_labels_its_output_features(self):
        """ttnn would carry the weight's shard dim (2) to the output, where dim 2 is the sequence."""
        layer = ColumnParallelLinear(DIM, DIM, has_bias=False)
        out = layer(tensor(random((1, 1, SEQ, DIM), 2)))
        assert shard_dim(out) == 3

    @pytest.mark.parametrize("like", [None, "sequence"])
    @pytest.mark.parametrize("input_layout", ["replicated", "features"])
    def test_row_parallel_matches_a_replicated_layer(self, input_layout, like):
        """Any input layout and any output layout: a replicated input is sliced here, and the result returns in the
        layout of ``like``. The row-parallel layer used to refuse a replicated input under sequence parallelism."""
        row = RowParallelLinear(DIM, DIM, has_bias=True)
        ref = LinearLayer(DIM, DIM, has_bias=True)
        copy_weights(row, ref)
        data = random((1, 1, SEQ, DIM), 3)
        x_ref = tensor(data)
        x = tensor(data, ttnn.PlacementShard(3) if input_layout == "features" else None)
        like_tensor = tensor(data, ttnn.PlacementShard(SEQUENCE_DIM), False) if like else None

        out = row(x, like=like_tensor)
        expected = ref(x_ref)
        assert shard_dim(out) == (SEQUENCE_DIM if like else None)
        assert_close(full(out), full(expected), "forward")

        # One loss on every rank, as a model's head gathers the sequence before it.
        ttml.ops.unary.mean(whole(out, tp())).backward(False)
        ttml.ops.unary.mean(expected).backward(False)
        ttml.sync_gradients(row.parameters())
        for name in ("weight", "bias"):
            assert_close(full_grad(getattr(row, name).tensor), full_grad(getattr(ref, name).tensor), f"{name} gradient")
        assert_close(full_grad(x), full_grad(x_ref), "input gradient")


class TestDeepSeekMLP:
    def test_tensor_parallel_matches_replicated(self):
        """The fused SwiGLU on sharded weights gives partial sums; they used to be returned unsummed."""
        tp_mlp = DeepSeekMLP(DIM, 2 * DIM, tp_axis_name="tp")
        ref = DeepSeekMLP(DIM, 2 * DIM)
        copy_weights(tp_mlp, ref)
        data = random((1, 1, SEQ, DIM), 4)
        x, x_ref = tensor(data), tensor(data)

        out, expected = tp_mlp(x), ref(x_ref)
        assert_close(full(out), full(expected), "forward")
        ttml.ops.unary.mean(out).backward(False)
        ttml.ops.unary.mean(expected).backward(False)
        assert_close(full_grad(x), full_grad(x_ref), "input gradient")


class TestCrossEntropy:
    @pytest.mark.parametrize("gather_output", [False, True], ids=["vocab_sharded", "gathered"])
    def test_follows_the_logits_layout(self, gather_output):
        """Vocab-sharded logits take the vocab-parallel loss and gathered ones the plain loss; a loss picked from the
        mesh shape gave gathered logits the vocab-parallel one (DeepSeek's LM head under TP)."""
        vocab = 128
        head = ColumnParallelLinear(DIM, vocab, has_bias=False, gather_output=gather_output)
        x = tensor(random((1, 1, SEQ, DIM), 5))
        targets_np = np.random.default_rng(6).integers(0, vocab, (1, SEQ)).astype(np.uint32)
        targets = ttml.autograd.Tensor.from_numpy(targets_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)

        logits = head(x)
        loss = ttml.parallel.cross_entropy_loss(logits, targets)
        logits_np = full(logits).reshape(SEQ, vocab)
        log_probs = logits_np - np.log(np.exp(logits_np).sum(axis=-1, keepdims=True))
        expected = -log_probs[np.arange(SEQ), targets_np[0]].mean()
        for rank in ranks(loss.get_value(NATIVE)):
            np.testing.assert_allclose(rank.reshape(-1)[0], expected, rtol=2e-2)


class TestDropout:
    def test_replicas_drop_the_same_elements_and_shards_draw_their_own(self):
        ones = np.ones((2, 1, SEQ, DIM), np.float32)
        replicated = ttml.ops.dropout.dropout(tensor(ones), 0.5)
        masks = [rank != 0 for rank in ranks(replicated.get_value(NATIVE))]
        assert np.array_equal(masks[0], masks[1]), "replicas dropped different elements"

        sharded = ttml.ops.dropout.dropout(tensor(ones, ttnn.PlacementShard(0)), 0.5)
        masks = [rank != 0 for rank in ranks(sharded.get_value(NATIVE))]
        assert not np.array_equal(masks[0], masks[1]), "shards drew the same mask"
