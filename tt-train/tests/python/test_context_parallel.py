# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Context parallelism (``LlamaConfig.context_parallel``) on Llama: each cp rank holds a contiguous chunk
of the sequence, and attention runs as a ring of K/V shifts along the ``"cp"`` axis.
The oracle is the same model without CP on the same mesh, built from one seed and fed the whole sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.parallel import TPStrategy
from bf16_ulp import assert_within_bf16_ulp

CP_AXIS_SIZE, TP_AXIS_SIZE = 2, 4  # the extents of conftest's cp_tp_mesh

# The ring and the single-chip kernel round differently: logits land within 1.5 ULP of the oracle
# at the peak and gradients within 3. Rotating every rank with the first chunk's positions, a plausible
# bug, reaches 30 on both.
MAX_ULP = 6.0

# Two tiles per (cp, tp) rank under SP; one tile would not exercise a multi-tile chunk.
SEQ_LEN = 32 * TP_AXIS_SIZE * CP_AXIS_SIZE * 2
HIDDEN = 256
N_HEADS = 8
N_KV_HEADS = 4
N_LAYERS = 2
INTERMEDIATE = 512
VOCAB = 256
STRATEGIES = [TPStrategy.NONE, TPStrategy.TENSOR, TPStrategy.TENSOR_SEQUENCE]


def config(tp_strategy: TPStrategy, context_parallel: bool, **overrides) -> LlamaConfig:
    fields = dict(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LEN,
        tp_strategy=tp_strategy,
        embedding_placement=EmbeddingPlacement.VocabParallel,
        weight_tying=WeightTyingType.Disabled,
        context_parallel=context_parallel,
    )
    return LlamaConfig(**{**fields, **overrides})


def paired_models(tp_strategy: TPStrategy, **overrides) -> tuple[Llama, Llama]:
    """A model without CP and one with CP, built from one seed: the same weights in independent storage."""
    models = []
    for context_parallel in (False, True):
        ttml.manual_seed(0)
        models.append(Llama(config(tp_strategy, context_parallel, **overrides)))
    return tuple(models)


def token_batch(batch: int, seed: int, context_parallel: bool):
    """Token ids ``(B, 1, 1, S)`` and next-token labels ``(B, S)``; under CP each rank gets its chunk."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB, (batch, 1, 1, SEQ_LEN)).astype(np.uint32)
    labels = rng.integers(0, VOCAB, (batch, SEQ_LEN)).astype(np.uint32)
    mesh = ttml.mesh()
    ids_mapper = mesh.shard_mapper({"cp": 3}) if context_parallel else None
    labels_mapper = mesh.shard_mapper({"cp": 1}) if context_parallel else None
    return (
        ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, ids_mapper),
        ttml.autograd.Tensor.from_numpy(labels, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, labels_mapper),
    )


def _compose(tensor, dims: list[int], shape: list[int]) -> np.ndarray:
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttnn.create_mesh_composer(
        device, ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=ttnn.MeshShape(shape))
    )
    return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=composer).astype(np.float64)


def full_sequence(logits, context_parallel: bool) -> np.ndarray:
    """``(B, tp, S, V_local)``: the cp chunks joined along the sequence, every tp rank's copy along dim 1.
    Without CP the cp ranks hold identical copies, of which the first is taken."""
    mesh = ttml.mesh()
    dims, shape = [0] * len(mesh.shape), [1] * len(mesh.shape)
    dims[mesh.axis_index("tp")], shape[mesh.axis_index("tp")] = 1, TP_AXIS_SIZE
    if context_parallel:
        dims[mesh.axis_index("cp")], shape[mesh.axis_index("cp")] = 2, CP_AXIS_SIZE
    return _compose(logits, dims, shape)


def per_device(tensor) -> np.ndarray:
    """Every device's copy of a 4-D ``(1, 1, ., .)`` tensor, mesh axis ``i`` stacked along dim ``i``."""
    mesh = ttml.mesh()
    return _compose(tensor, list(range(len(mesh.shape))), list(mesh.shape))


def loss_of(logits, labels, tp_strategy: TPStrategy):
    if tp_strategy.tensor_parallel:
        return ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
            logits, labels, cluster_axis=ttml.mesh().axis_index("tp"), reduce=ttml.ops.ReduceType.MEAN
        )
    return ttml.ops.loss.cross_entropy_loss(logits, labels, ttml.ops.ReduceType.MEAN)


def train_step(model: Llama, tp_strategy: TPStrategy, seed: int) -> None:
    """One mean cross-entropy backward on a fresh batch, then the gradient sync."""
    context_parallel = model.config.context_parallel
    ids, labels = token_batch(1, seed, context_parallel)
    model.train()
    loss_of(model(ids, None), labels, tp_strategy).backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    ttml.sync_gradients(model.parameters())


def assert_same_grads(cp_model: Llama, ref_model: Llama, label: str = "") -> None:
    ref_params, cp_params = ref_model.parameters(), cp_model.parameters()
    assert set(ref_params.keys()) == set(cp_params.keys())
    cp_axis = ttml.mesh().axis_index("cp")
    for name, ref_param in ref_params.items():
        if not ref_param.get_requires_grad():
            continue
        assert cp_params[name].is_grad_initialized(), f"{name}: CP grad missing"
        got = per_device(cp_params[name].get_grad_tensor())
        for rank in range(1, CP_AXIS_SIZE):
            np.testing.assert_array_equal(
                np.take(got, rank, axis=cp_axis), np.take(got, 0, axis=cp_axis), err_msg=f"{name}: cp ranks disagree"
            )
        assert_within_bf16_ulp(got, per_device(ref_param.get_grad_tensor()), f"grad {name} {label}", MAX_ULP)


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
@pytest.mark.usefixtures("cp_tp_mesh")
class TestMatchesFullSequence:
    @pytest.mark.parametrize("tp_strategy", STRATEGIES, ids=lambda s: s.name)
    @pytest.mark.parametrize("batch", [1, 2])
    def test_logits(self, tp_strategy, batch):
        ref_model, cp_model = paired_models(tp_strategy)
        ref_model.eval()
        cp_model.eval()
        seed = 1234 + batch
        ref_logits = full_sequence(ref_model(token_batch(batch, seed, False)[0], None), False)
        cp_logits = full_sequence(cp_model(token_batch(batch, seed, True)[0], None), True)

        assert ref_logits.std() > 1e-3, "logits are ~constant; agreement would prove nothing"
        assert_within_bf16_ulp(cp_logits, ref_logits, f"logits {tp_strategy.name} batch={batch}", MAX_ULP)

    @pytest.mark.parametrize("placement", list(EmbeddingPlacement), ids=lambda p: p.name)
    def test_logits_sequence_parallel_embeddings(self, placement):
        """Under SP each embedding placement reaches the sequence-sharded stream its own way."""
        tp_strategy = TPStrategy.TENSOR_SEQUENCE
        ref_model, cp_model = paired_models(tp_strategy, embedding_placement=placement)
        ref_model.eval()
        cp_model.eval()
        ref_logits = full_sequence(ref_model(token_batch(1, 99, False)[0], None), False)
        cp_logits = full_sequence(cp_model(token_batch(1, 99, True)[0], None), True)
        assert_within_bf16_ulp(cp_logits, ref_logits, f"logits {placement.name}", MAX_ULP)

    @pytest.mark.parametrize("tp_strategy", STRATEGIES, ids=lambda s: s.name)
    def test_gradients_after_sync(self, tp_strategy):
        ref_model, cp_model = paired_models(tp_strategy, attention_bias=True)
        for model in (ref_model, cp_model):
            train_step(model, tp_strategy, seed=77)
        assert_same_grads(cp_model, ref_model)

    def test_gradients_after_an_optimizer_step(self):
        """The optimizer rewrites each parameter's mesh topology in place; the second sync must not care.

        With a small epsilon AdamW's first update is ``lr * sign(g)``, which turns the few-ULP rounding
        differences between the two attention paths into whole-``lr`` weight differences wherever a
        gradient is near zero. A large epsilon keeps the update proportional to the gradient.
        """
        tp_strategy = TPStrategy.TENSOR_SEQUENCE
        models = paired_models(tp_strategy)
        optimizers = [
            ttml.optimizers.AdamW(m.parameters(), ttml.optimizers.AdamWConfig.make(1e-2, 0.9, 0.999, 1.0, 0.0))
            for m in models
        ]
        for step in range(2):
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad()
                train_step(model, tp_strategy, seed=step)
            assert_same_grads(models[1], models[0], f"step {step}")
            for optimizer in optimizers:
                optimizer.step()


@pytest.mark.requires_device
@pytest.mark.usefixtures("cp_tp_mesh")
class TestValidation:
    def test_rejects_a_mask(self, expect_error):
        model = Llama(config(TPStrategy.NONE, True))
        mask = ttml.autograd.Tensor.from_numpy(
            np.tril(np.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=np.float32)), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
        )
        with expect_error(ValueError, "takes no mask"):
            model(token_batch(1, 5, True)[0], mask)

    def test_rejects_a_chunk_of_the_wrong_length(self, expect_error):
        """RoPE's per-rank caches fix where each chunk sits, so the full sequence must be max_position_embeddings."""
        model = Llama(config(TPStrategy.NONE, True))
        ids = ttml.autograd.Tensor.from_numpy(
            np.zeros((1, 1, 1, SEQ_LEN), dtype=np.uint32), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
        )  # replicated: every rank sees the whole sequence instead of its chunk
        with expect_error(ValueError, "expects each rank to hold its chunk"):
            model(ids, None)

    def test_rejects_a_sequence_that_does_not_tile_the_ranks(self, expect_error):
        with expect_error(ValueError, "divisible by 256 \\(32 \\* cp_size \\* tp_size under sequence parallelism\\)"):
            config(TPStrategy.TENSOR_SEQUENCE, True, max_position_embeddings=32 * CP_AXIS_SIZE * 3)

    def test_rejects_kv_cache(self, expect_error):
        model = Llama(config(TPStrategy.NONE, True))
        kv_cache = ttml.models.KvCache(
            num_layers=N_LAYERS,
            batch_size=1,
            num_groups=N_KV_HEADS,
            max_seq_len=SEQ_LEN,
            head_dim=HIDDEN // N_HEADS,
        )
        with expect_error(NotImplementedError, "context parallelism does not support"):
            model(token_batch(1, 5, True)[0], None, kv_cache=kv_cache, new_tokens=SEQ_LEN // CP_AXIS_SIZE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
