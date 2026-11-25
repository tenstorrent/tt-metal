# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama with tensor parallelism, and with sequence parallelism on top, against Llama without either.

Runs at tp=2 and tp=4; tp=4 needs four devices and the bundled ``(1, 4)`` descriptor.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.common.utils import build_causal_mask
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.parallel import TPStrategy
from tp_block import gather_like

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

NATIVE = ttml.autograd.PreferredPrecision.NATIVE
BATCH, VOCAB, N_LAYERS = 2, 128, 2
HIDDEN, N_HEADS, N_KV_HEADS, INTERMEDIATE = 256, 8, 4, 256
HEAD_DIM = HIDDEN // N_HEADS
SEQ_LEN = 256  # two tiles of sequence per rank at tp=4

# A bf16 loss near ln(128) ~ 4.9 rounds to 1/32.
LOSS_TOLERANCE = 0.05
# Relative Frobenius error: bf16 rounding, as TP sums row-parallel partial products in another order than one matmul
# does. Measured at most 0.012, in a norm gamma's gradient.
MAX_REL_ERROR = 0.04

# Fused weights whose rows are blocks: one device holds a slice of every block, so under TP the rows run block by block
# within each rank (see GroupedQueryAttention and LlamaMLP). Rows per block, keyed by the parameter name's end.
_BLOCK_ROWS = {
    "attention/qkv_linear": (N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM),
    "mlp/w_gate_up": (INTERMEDIATE, INTERMEDIATE),
}


@pytest.fixture(scope="module", params=[2, 4], ids=["tp2", "tp4"])
def tp_size(request, fresh_device_mesh):
    with fresh_device_mesh((1, request.param), ("dp", "tp"), what=f"tp={request.param}", require_mgd=True):
        yield request.param


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def config(tp_strategy: TPStrategy, placement: EmbeddingPlacement) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LEN,
        attention_bias=True,
        tp_strategy=tp_strategy,
        embedding_placement=placement,
        weight_tying=WeightTyingType.Disabled,
    )


def in_tp_layout(name: str, values: np.ndarray, tp: int) -> np.ndarray:
    """A parameter of the model without TP, as its tensor-parallel counterpart lays it out on the whole mesh."""
    block_rows = next((rows for key, rows in _BLOCK_ROWS.items() if name.endswith(f"{key}/weight")), None)
    dim = -2
    if block_rows is None:
        block_rows = next((rows for key, rows in _BLOCK_ROWS.items() if name.endswith(f"{key}/bias")), None)
        dim = -1
    if block_rows is None:
        return values
    blocks = np.split(values, np.cumsum(block_rows)[:-1], axis=dim)
    slices = [np.split(block, tp, axis=dim) for block in blocks]
    return np.concatenate([per_block[rank] for rank in range(tp) for per_block in slices], axis=dim)


def step(model: Llama) -> tuple[float, np.ndarray]:
    """Forward and backward on fixed tokens with the loss train.py picks, gradients synced; returns the loss and the
    logits."""
    rng = np.random.default_rng(7)
    ids = rng.integers(0, VOCAB, (BATCH, 1, 1, SEQ_LEN)).astype(np.uint32)
    targets = rng.integers(0, VOCAB, (BATCH, SEQ_LEN)).astype(np.uint32)
    targets = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    model.train()
    ids = ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    logits = model(ids, build_causal_mask(SEQ_LEN, device=True))
    loss = ttml.parallel.cross_entropy_loss(logits, targets, ttml.ops.ReduceType.MEAN)
    loss.backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    ttml.sync_gradients(model.parameters())
    return float(gather_like(loss.get_value(), loss).item()), gather_like(logits.get_value(), logits)


def assert_close(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    scale = np.linalg.norm(expected)
    assert scale > 0, f"{what}: all zero, so agreement would prove nothing"
    error = np.linalg.norm(actual - expected) / scale
    assert error <= MAX_REL_ERROR, f"{what}: relative error {error:.3g}"


@pytest.mark.parametrize("placement", list(EmbeddingPlacement), ids=lambda p: p.name)
@pytest.mark.parametrize("tp_strategy", [TPStrategy.TENSOR, TPStrategy.TENSOR_SEQUENCE], ids=["tp", "sp"])
def test_matches_the_model_without_tp(tp_size, tp_strategy, placement):
    ttml.manual_seed(0)
    reference = Llama(config(TPStrategy.NONE, EmbeddingPlacement.Replicated))
    model = Llama(config(tp_strategy, placement))
    params, sources = model.parameters(), reference.parameters()
    assert set(params.keys()) == set(sources.keys())
    for name, param in params.items():
        values = in_tp_layout(name, gather_like(sources[name].get_value(NATIVE), sources[name]), tp_size)
        ttml.autograd.assign_numpy(param, values, mapper=ttml.Sharding.from_tensor(param).derive_mapper())

    loss, logits = step(model)
    expected_loss, expected_logits = step(reference)
    assert abs(loss - expected_loss) <= LOSS_TOLERANCE, f"loss {loss}, without TP {expected_loss}"
    assert_close(logits, expected_logits, "logits")
    for name, param in params.items():
        assert param.is_grad_initialized(), f"{name}: no gradient"
        expected = in_tp_layout(name, gather_like(sources[name].get_grad(), sources[name]), tp_size)
        assert_close(gather_like(param.get_grad(), param), expected, f"grad of {name}")
