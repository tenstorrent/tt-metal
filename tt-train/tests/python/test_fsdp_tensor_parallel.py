# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fully sharded data parallelism composed with tensor and sequence parallelism.

On an ``("fsdp", "tp")`` mesh, a Llama whose blocks are ``fully_shard``-ed must train exactly like the same Llama kept
replicated across ``fsdp`` (plain data parallelism): FSDP shards each parameter on a dim tensor parallelism left
whole, gathers it for compute, reduce-scatters its gradient in the backward pass, and ``ttml.sync_gradients`` then
sums sequence parallelism's partial gradients over ``tp``. Synced gradients are compared parameter by parameter,
over two micro-batches so that FSDP's gradient accumulation is part of it, and with the memory-efficient runner,
whose recomputation runs a backward pass inside the outer one.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import EmbeddingPlacement, RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.parallel import TPStrategy

pytestmark = [pytest.mark.requires_device]

SEQ_LEN, HIDDEN, VOCAB, N_LAYERS = 128, 128, 128, 2


@pytest.fixture(scope="module")
def fsdp_tp_mesh(fresh_device_mesh):
    with fresh_device_mesh((2, 2), ("fsdp", "tp"), what="fsdp x tp tests", require_mgd=True) as mesh:
        yield mesh


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def llama(strategy: TPStrategy, runner: RunnerType = RunnerType.Default) -> Llama:
    ttml.manual_seed(0)
    return Llama(
        LlamaConfig(
            hidden_size=HIDDEN,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=N_LAYERS,
            intermediate_size=128,
            vocab_size=VOCAB,
            max_position_embeddings=SEQ_LEN,
            tp_strategy=strategy,
            embedding_placement=EmbeddingPlacement.VocabParallel,
            weight_tying=WeightTyingType.Disabled,
            runner_type=runner,
        )
    )


def batch(mesh, seed: int):
    """Two sequences, one per fsdp rank, and their next-token targets."""
    tokens = np.random.default_rng(seed).integers(0, VOCAB, (2, SEQ_LEN + 1)).astype(np.uint32)
    mapper = mesh.batch_mapper(tdim=0)
    ids = ttml.autograd.Tensor.from_numpy(
        tokens[:, :-1].reshape(2, 1, 1, SEQ_LEN), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper
    )
    targets = ttml.autograd.Tensor.from_numpy(tokens[:, 1:], ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper)
    return ids, targets


def causal_mask():
    mask = np.tril(np.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=np.float32))
    return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)


def train_step(model, mesh) -> None:
    model.train()
    for seed in (0, 1):  # two micro-batches accumulate
        ids, targets = batch(mesh, seed)
        loss = ttml.parallel.cross_entropy_loss(model(ids, causal_mask()), targets)
        ttml.ops.binary.mul(loss, 0.5).backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
    ttml.sync_gradients(model.parameters())


def synced_grads(model) -> dict[str, np.ndarray]:
    """Each synced gradient in full, gathered by its parameter's layout (the layout a synced gradient has)."""
    grads = {}
    for name, param in model.parameters().items():
        grads[name] = ttml.Sharding.from_tensor(param).gather(param.get_grad_tensor()).astype(np.float64)
    return grads


@pytest.mark.parametrize("runner", [RunnerType.Default, RunnerType.MemoryEfficient], ids=["default", "recompute"])
@pytest.mark.parametrize("strategy", [TPStrategy.TENSOR, TPStrategy.TENSOR_SEQUENCE], ids=["tp", "sp"])
def test_fully_sharded_matches_data_parallel(fsdp_tp_mesh, strategy, runner):
    replicated = llama(strategy, runner)
    sharded = llama(strategy, runner)
    for block in sharded.blocks:
        ttml.fsdp.fully_shard(block)
    ttml.fsdp.fully_shard(sharded)
    fsdp = fsdp_tp_mesh.axis_index("fsdp")
    assert all(ttml.Sharding.from_tensor(p).shard_dim(fsdp) is not None for p in sharded.parameters().values())

    train_step(replicated, fsdp_tp_mesh)
    train_step(sharded, fsdp_tp_mesh)

    expected, got = synced_grads(replicated), synced_grads(sharded)
    assert expected.keys() == got.keys()
    for name in expected:
        scale = float(np.abs(expected[name]).max())
        assert scale > 0, f"{name}: a zero gradient would prove nothing"
        np.testing.assert_allclose(got[name], expected[name], rtol=5e-2, atol=2e-2 * scale, err_msg=name)
