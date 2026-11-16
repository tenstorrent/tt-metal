# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek with tensor parallelism against DeepSeek without it, from the same weights on the same tokens."""

from __future__ import annotations

import re

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import RunnerType
from ttml.models.deepseek import DeepSeek, DeepSeekConfig
from tp_block import gather_like

pytestmark = [pytest.mark.requires_device, pytest.mark.usefixtures("tp_mesh")]

NATIVE = ttml.autograd.PreferredPrecision.NATIVE
BATCH, SEQ_LEN, VOCAB, N_EXPERTS = 2, 64, 128, 4

# Relative Frobenius error. bf16 rounding (TP sums the row-parallel partial products in another order than one matmul
# does) measured at most 0.025. Leaving k_pe's gradient partial does not rely on it: it leaves unequal copies of the
# gradients of wkv_a, the attention norms and the embedding, which gather_like refuses.
MAX_REL_ERROR = 0.04

# A sparse_ep parameter holding expert ``rank * experts_per_rank + local`` on each tp rank, and its dense linear.
_EP_EXPERT = re.compile(r"(?P<ffn>.*/ffn)/w_(?P<kind>gate|up|down)_(?P<local>\d+)$")
_DENSE_LINEAR = {"gate": "w1", "up": "w3", "down": "w2"}


def config(*, use_tp: bool, moe_type: str, q_lora_rank: int, vocab: int) -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=vocab,
        dim=128,
        inter_dim=256,
        moe_inter_dim=64,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=N_EXPERTS,
        n_expert_groups=1,
        n_limited_groups=1,
        score_func="softmax",
        route_scale=1.0,
        moe_type=moe_type,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        max_seq_len=SEQ_LEN,
        runner_type=RunnerType.Default,
        use_tp=use_tp,
    )


def from_reference(reference: DeepSeek, name: str, read) -> np.ndarray:
    """``read`` of the reference parameters that tensor-parallel parameter ``name`` is made of, in tp-rank order."""
    sources = reference.parameters()
    match = _EP_EXPERT.match(name)
    if match is None:
        return read(sources[name])
    per_rank = N_EXPERTS // ttml.mesh().axis_size("tp")
    linear, local = _DENSE_LINEAR[match["kind"]], int(match["local"])
    experts = range(local, N_EXPERTS, per_rank)
    return np.concatenate([read(sources[f"{match['ffn']}/experts/{e}/{linear}/weight"]) for e in experts])


def weight(t) -> np.ndarray:
    return gather_like(t.get_value(NATIVE), t)


def grad(t) -> np.ndarray:
    return gather_like(t.get_grad(), t)


def step(model: DeepSeek) -> np.ndarray:
    """Forward and backward on fixed tokens, gradients synced; returns the logits over the real vocab."""
    vocab = model.config.vocab_size
    rng = np.random.default_rng(7)
    ids = rng.integers(0, vocab, (BATCH, 1, 1, SEQ_LEN)).astype(np.uint32)
    targets = rng.integers(0, vocab, (BATCH, SEQ_LEN)).astype(np.uint32)
    targets = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    model.train()
    logits = model(ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32), None)
    if model.config.use_tp:
        tp = ttml.mesh().axis_index("tp")
        loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, targets, cluster_axis=tp)
    else:
        loss = ttml.ops.loss.cross_entropy_loss(logits, targets)
    loss.backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    ttml.sync_gradients(model.parameters())
    return gather_like(logits.get_value(), logits)[..., :vocab]


def assert_close(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    scale = np.linalg.norm(expected)
    assert scale > 0, f"{what}: all zero, so agreement would prove nothing"
    error = np.linalg.norm(actual - expected) / scale
    assert error <= MAX_REL_ERROR, f"{what}: relative error {error:.3g}"


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.parametrize(
    "moe_type, q_lora_rank, vocab",
    [
        ("dense", 0, VOCAB),
        ("dense", 64, VOCAB),
        ("sparse_ep", 0, VOCAB),
        ("sparse_ep", 64, VOCAB),
        # Padded to 128 either way: the reference slices the padded columns off, the TP model masks them.
        ("dense", 0, 100),
    ],
    ids=["dense-direct_q", "dense-low_rank_q", "sparse_ep-direct_q", "sparse_ep-low_rank_q", "padded_vocab"],
)
def test_matches_the_replicated_model(moe_type, q_lora_rank, vocab):
    reference = DeepSeek(config(use_tp=False, moe_type="dense", q_lora_rank=q_lora_rank, vocab=vocab))
    model = DeepSeek(config(use_tp=True, moe_type=moe_type, q_lora_rank=q_lora_rank, vocab=vocab))
    params = model.parameters()
    for name, param in params.items():
        mapper = ttml.Sharding.from_tensor(param).derive_mapper()
        ttml.autograd.assign_numpy(param, from_reference(reference, name, weight), mapper=mapper)

    assert_close(step(model), step(reference), "logits")
    for name, param in params.items():
        assert param.is_grad_initialized(), f"{name}: no gradient"
        assert_close(grad(param), from_reference(reference, name, grad), f"grad of {name}")


@pytest.mark.parametrize("placement", [ttnn.PlacementReplicate(), ttnn.PlacementShard(2)], ids=["replicated", "seq"])
def test_the_vocab_parallel_loss_refuses_logits_not_sharded_along_the_vocab(expect_error, placement):
    """The loss takes each device's columns for its own slice of the vocab. Logits gathered across tp, as this model's
    head used to return them, would count the whole vocab once per device: the loss off by log(tp), the head's
    gradient wrong."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), placement]))
    logits = np.zeros((1, 1, SEQ_LEN, VOCAB), np.float32)
    logits = ttml.autograd.Tensor.from_numpy(logits, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    targets = np.zeros((1, SEQ_LEN), np.uint32)
    targets = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    tp = ttml.mesh().axis_index("tp")
    with expect_error(RuntimeError, "takes logits sharded along the vocab"):
        ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, targets, cluster_axis=tp)
