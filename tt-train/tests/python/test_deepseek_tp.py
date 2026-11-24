# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek with ``use_tp`` on a [1, 2] mesh.

The hidden state must be identical on both TP ranks: the dense FFN runs the fused SwiGLU on sharded
weights, so its partial outputs need the all-reduce the parallel layers would do.

The vocab-parallel loss must give the same loss and gradients as a plain loss.
``examples/train/train.py`` and ``SFTTrainer`` train every TP model with
``vocab_parallel_cross_entropy_loss``, which expects each TP rank to hold its own vocab shard of
the logits. The reference here all-gathers those shards, drops the padded columns and applies the
plain ``cross_entropy_loss``. With replicated full-vocab logits, the vocab-parallel loss reads
``log(tp)`` too high and drops the target term on every rank but the first.

Vocab 96 needs padding at TP=2 (shards of 48 columns are not tile-aligned), so it also covers the
padded columns, which must get zero probability.

The sharded MLP must match a float64 SwiGLU, also when each rank's share of the hidden columns is not
whole tiles.
"""

import numpy as np
import pytest
import ttml
import ttnn
from ttml.models.deepseek import DeepSeek, DeepSeekConfig
from ttml.models.deepseek.transformer import DeepSeekMLP
from ttml.models.llama.autograd_ops import SliceLastDim

pytestmark = pytest.mark.requires_device

BATCH = 2
SEQ = 64


def _device_slices(value):
    return [ttnn.to_torch(s).double().numpy() for s in ttnn.get_device_tensors(value)]


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _model(vocab):
    cfg = DeepSeekConfig(
        vocab_size=vocab,
        dim=512,
        inter_dim=1536,
        n_layers=1,
        n_dense_layers=1,
        n_heads=8,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        max_seq_len=SEQ,
        use_tp=True,
    )
    return DeepSeek(cfg)


def _loss_and_grads(model, optimizer, loss_fn, tokens):
    optimizer.zero_grad()
    loss = loss_fn(model(tokens, None))
    loss_value = _device_slices(loss.get_value())[0].reshape(-1)[0]
    loss.backward(False)
    head_grad = np.concatenate(_device_slices(model.head.weight.tensor.get_grad()), axis=2)
    norm_grad = _device_slices(model.parameters()["DeepSeek/norm/gamma"].get_grad())[0]
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return loss_value, head_grad, norm_grad


def test_tp_hidden_state_is_replicated(tp_mesh):
    """The dense FFN's partial outputs are all-reduced, so every TP rank carries the same hidden state."""
    ttml.manual_seed(1234)
    model = _model(128)
    tokens = ttml.autograd.Tensor.from_numpy(
        np.random.default_rng(0).integers(0, 128, size=(BATCH, 1, 1, SEQ)).astype(np.uint32),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
    )
    x = model.tok_emb(tokens)
    block = model.blocks[0]
    x = ttml.ops.binary.add(x, block.attn(block.attn_norm(x)))
    ffn = _device_slices(block.ffn(block.ffn_norm(x)).get_value())
    np.testing.assert_allclose(ffn[1], ffn[0], rtol=0, atol=1e-6)
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.parametrize("vocab", [128, 96])
def test_vocab_parallel_loss_matches_plain_loss(tp_mesh, vocab):
    ttml.manual_seed(1234)
    model = _model(vocab)
    tp_axis = tp_mesh.axis_index("tp")

    rng = np.random.default_rng(0)
    tokens = ttml.autograd.Tensor.from_numpy(
        rng.integers(0, vocab, size=(BATCH, 1, 1, SEQ)).astype(np.uint32),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
    )
    labels_np = rng.integers(0, vocab, size=(BATCH, SEQ)).astype(np.uint32)
    labels = ttml.autograd.Tensor.from_numpy(labels_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)

    def vocab_parallel(logits):
        return ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
            logits, labels, cluster_axis=tp_axis, reduce=ttml.ops.ReduceType.MEAN
        )

    def plain(logits):
        full = ttml.ops.distributed.all_gather(logits, 3, tp_axis, ttml.ops.distributed.GradOutputType.REPLICATED)
        if model.padded_vocab_size != vocab:
            full = SliceLastDim.apply(full, vocab)
        return ttml.ops.loss.cross_entropy_loss(full, labels, ttml.ops.ReduceType.MEAN)

    # The vocab-parallel loss needs each rank to hold its own vocab shard of the logits.
    logits = _device_slices(model(tokens, None).get_value())
    ttml.autograd.AutoContext.get_instance().reset_graph()
    assert logits[0].shape[-1] == model.padded_vocab_size // tp_mesh.axis_size("tp")
    z = np.concatenate(logits, axis=-1).reshape(BATCH * SEQ, -1)[:, :vocab]
    target = labels_np.reshape(-1)
    log_norm = z.max(axis=1) + np.log(np.exp(z - z.max(axis=1, keepdims=True)).sum(axis=1))
    loss_host = float(np.mean(log_norm - z[np.arange(BATCH * SEQ), target]))

    # lr 0: only used to zero the gradients between the two passes.
    optimizer = ttml.optimizers.AdamW(
        model.parameters(),
        ttml.optimizers.AdamWConfig.make(lr=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0),
    )
    loss_vp, head_vp, norm_vp = _loss_and_grads(model, optimizer, vocab_parallel, tokens)
    loss_ref, head_ref, norm_ref = _loss_and_grads(model, optimizer, plain, tokens)

    # Replicated full-vocab logits would put the vocab-parallel loss log(2) ~ 0.69 above the host loss.
    assert abs(loss_vp - loss_host) < 0.05, (loss_vp, loss_host)
    assert abs(loss_ref - loss_host) < 0.05, (loss_ref, loss_host)
    assert _rel(head_vp, head_ref) < 0.05
    assert _rel(norm_vp, norm_ref) < 0.05
    # Padded rows of the LM head get no gradient.
    assert np.all(head_vp[..., vocab:, :] == 0.0)


def _swiglu_host(x, w1, w3, w2, up):
    """float64 SwiGLU y = w2(silu(w1 x) * w3 x) and, for upstream gradient ``up``, dx, dW1, dW3, dW2."""
    h1, h3 = x @ w1.T, x @ w3.T
    sig = 1.0 / (1.0 + np.exp(-h1))
    gated = h1 * sig * h3
    d_gated = up @ w2
    dh1 = d_gated * h3 * sig * (1.0 + h1 * (1.0 - sig))
    dh3 = d_gated * h1 * sig
    return gated @ w2.T, dh1 @ w1 + dh3 @ w3, dh1.T @ x, dh3.T @ x, up.T @ gated


@pytest.mark.parametrize("hidden", [128, 96])
def test_sharded_mlp_matches_host_swiglu(tp_mesh, hidden):
    """At hidden 96 each rank holds 48 hidden columns, not whole tiles, like DeepSeek-16B's dense FFN at TP=8 (1368)."""
    ttml.manual_seed(1234)
    dim = 64
    mlp = DeepSeekMLP(dim, hidden, tp_axis_name="tp")
    rng = np.random.default_rng(0)
    x = ttml.autograd.Tensor.from_numpy(
        rng.standard_normal((BATCH, 1, SEQ, dim)).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    x.set_requires_grad(True)
    up = ttml.autograd.Tensor.from_numpy(
        rng.standard_normal((BATCH, 1, SEQ, dim)).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()
    y = mlp(x)
    y.set_grad(up)
    y.backward(False)

    def gathered(value, axis):
        return np.concatenate(_device_slices(value), axis=axis)[0, 0]

    w1, w3 = (gathered(layer.weight.tensor.get_value(), 2) for layer in (mlp.w1, mlp.w3))
    w2 = gathered(mlp.w2.weight.tensor.get_value(), 3)
    assert w1.shape == (hidden, dim) and w2.shape == (dim, hidden)
    ref = _swiglu_host(
        _device_slices(x.get_value())[0].reshape(-1, dim), w1, w3, w2, _device_slices(up)[0].reshape(-1, dim)
    )
    got = (
        _device_slices(y.get_value())[0].reshape(-1, dim),
        _device_slices(x.get_grad())[0].reshape(-1, dim),
        gathered(mlp.w1.weight.tensor.get_grad(), 2),
        gathered(mlp.w3.weight.tensor.get_grad(), 2),
        gathered(mlp.w2.weight.tensor.get_grad(), 3),
    )
    ttml.autograd.AutoContext.get_instance().reset_graph()
    for name, g, r in zip(("y", "dx", "dW1", "dW3", "dW2"), got, ref):
        assert _rel(g, r) < 0.03, (name, _rel(g, r))
