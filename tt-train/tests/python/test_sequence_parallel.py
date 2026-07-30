# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Megatron sequence parallelism (``TPStrategy.TENSOR_SEQUENCE``) on Llama.
The oracle is classic tensor parallelism.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.models.llama.gqattn import GroupedQueryAttention
from ttml.parallel import TPStrategy
from ttml.testing import TP_AXIS_SIZE, assert_within_ulp, read_mesh_tensor

pytestmark = pytest.mark.requires_device

# Logits agree bitwise at tp=2 and the worst gradient disagreement is 1 ULP, entirely in the
# norm gammas. The limit stays above both so that summing more than two ranks, whose rounding
# genuinely is order-dependent, does not become flaky.
MAX_ULP = 2.0

# Two tiles of sequence per rank; one tile would not exercise a multi-tile shard.
SEQ_LEN = 32 * TP_AXIS_SIZE * 2
HIDDEN = 128
N_HEADS = 4
N_KV_HEADS = 2
N_LAYERS = 2
INTERMEDIATE = 128
VOCAB = 128
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


def config(tp_strategy: TPStrategy, seq_len: int = SEQ_LEN) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=seq_len,
        tp_strategy=tp_strategy,
        embedding_placement=EmbeddingPlacement.VocabParallel,
        weight_tying=WeightTyingType.Disabled,
    )


def paired_models() -> tuple[Llama, Llama]:
    """A TP model and an SP model holding bitwise-identical weights."""
    tp_model, sp_model = Llama(config(TPStrategy.TENSOR)), Llama(config(TPStrategy.TENSOR_SEQUENCE))
    source, destination = tp_model.parameters(), sp_model.parameters()
    assert set(source.keys()) == set(destination.keys())
    for name in source.keys():
        destination[name].set_value(source[name].get_value(NATIVE))
        assert destination[name].get_value(NATIVE).dtype == source[name].get_value(NATIVE).dtype, name
    return tp_model, sp_model


def token_ids(batch: int, seq_len: int, seed: int):
    ids = np.random.default_rng(seed).integers(0, VOCAB, (batch, 1, 1, seq_len)).astype(np.uint32)
    return ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)


def causal_mask(seq_len: int):
    mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.float32))
    return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)


def rope_params(seq_len: int = SEQ_LEN):
    return ttml.ops.rope.build_rope_params(seq_len, HIDDEN // N_HEADS, 500000.0, ttml.ops.rope.RopeScalingParams())


@pytest.mark.usefixtures("tp_mesh")
class TestMatchesTensorParallel:
    @pytest.mark.parametrize("batch", [1, 2])
    def test_logits(self, batch):
        tp_model, sp_model = paired_models()
        tp_model.eval()
        sp_model.eval()

        ids, mask = token_ids(batch, SEQ_LEN, seed=1234 + batch), causal_mask(SEQ_LEN)
        # Logits stay vocab-sharded under both strategies (gather_output=False).
        tp_logits = read_mesh_tensor(tp_model(ids, mask), {"tp": 3})
        sp_logits = read_mesh_tensor(sp_model(ids, mask), {"tp": 3})

        assert tp_logits.std() > 1e-3, "logits are ~constant; agreement would prove nothing"
        assert_within_ulp(sp_logits, tp_logits, f"logits batch={batch}", MAX_ULP)

    def test_gradients_after_sequence_parallel_sync(self):
        tp_model, sp_model = paired_models()
        ids, mask = token_ids(1, SEQ_LEN, seed=77), causal_mask(SEQ_LEN)
        for model in (tp_model, sp_model):
            model.train()
            model(ids, mask).backward(retain_graph=False)

        ttml.sync_sequence_parallel_gradients(sp_model.parameters(), "tp")

        tp_params, sp_params = tp_model.parameters(), sp_model.parameters()
        for name in tp_params.keys():
            assert tp_params[name].is_grad_initialized(), f"{name}: TP grad missing"
            assert sp_params[name].is_grad_initialized(), f"{name}: SP grad missing"
            assert_within_ulp(
                read_mesh_tensor(sp_params[name].get_grad_tensor()),
                read_mesh_tensor(tp_params[name].get_grad_tensor()),
                f"grad {name}",
                MAX_ULP,
            )


@pytest.mark.usefixtures("tp_mesh")
class TestValidation:
    def test_rejects_replicated_embedding(self, expect_error):
        with expect_error(ValueError, "sequence parallelism needs the embedding"):
            LlamaConfig(
                hidden_size=HIDDEN,
                num_attention_heads=N_HEADS,
                num_key_value_heads=N_KV_HEADS,
                intermediate_size=INTERMEDIATE,
                max_position_embeddings=SEQ_LEN,
                tp_strategy=TPStrategy.TENSOR_SEQUENCE,
                embedding_placement=EmbeddingPlacement.Replicated,
            )

    def test_rejects_max_positions_not_divisible_by_32_tp(self, expect_error):
        with expect_error(ValueError, "max_position_embeddings divisible by"):
            config(TPStrategy.TENSOR_SEQUENCE, seq_len=32)

    def test_rejects_input_sequence_not_divisible_by_32_tp(self, expect_error):
        """The config gate covers max_position_embeddings; a shorter input needs its own."""
        model = Llama(config(TPStrategy.TENSOR_SEQUENCE))
        with expect_error(ValueError, "input sequence length divisible by"):
            model(token_ids(1, 32, seed=5), causal_mask(32))

    def test_rejects_kv_cache_decode(self, expect_error):
        """Single-token decode has nothing to shard along the sequence."""
        attention = GroupedQueryAttention(
            embedding_size=HIDDEN,
            num_heads=N_HEADS,
            num_groups=N_KV_HEADS,
            dropout=0.0,
            rope_params=rope_params(),
            tp_strategy=TPStrategy.TENSOR_SEQUENCE,
        )
        hidden = ttml.autograd.Tensor.from_numpy(
            np.zeros((1, 1, SEQ_LEN, HIDDEN), np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
        )
        kv_cache = ttml.models.KvCache(
            num_layers=N_LAYERS,
            batch_size=1,
            num_groups=attention.num_groups,
            max_seq_len=SEQ_LEN,
            head_dim=HIDDEN // N_HEADS,
        )
        with expect_error(NotImplementedError, "sequence_parallel does not support"):
            attention(hidden, causal_mask(SEQ_LEN), kv_cache=kv_cache, layer_idx=0, new_tokens=1)


class TestTPStrategy:
    def test_from_flags(self):
        assert TPStrategy.from_flags(False) is TPStrategy.NONE
        assert TPStrategy.from_flags(True) is TPStrategy.TENSOR
        assert TPStrategy.from_flags(True, enable_sp=True) is TPStrategy.TENSOR_SEQUENCE

    def test_rejects_sp_without_tp(self, expect_error):
        with expect_error(ValueError, "requires enable_tp"):
            TPStrategy.from_flags(False, enable_sp=True)

    def test_enable_sp_is_keyword_only(self, expect_error):
        with expect_error(TypeError, "positional argument"):
            TPStrategy.from_flags(True, True)  # type: ignore

    @pytest.mark.parametrize(
        "strategy,tensor,sequence",
        [
            (TPStrategy.NONE, False, False),
            (TPStrategy.TENSOR, True, False),
            (TPStrategy.TENSOR_SEQUENCE, True, True),
        ],
    )
    def test_properties(self, strategy, tensor, sequence):
        assert strategy.tensor_parallel is tensor
        assert strategy.sequence_parallel is sequence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
