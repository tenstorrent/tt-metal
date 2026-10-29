# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC harness for the Mistral-Small-24B-Base-2501 training bring-up (``ttml/models/mistral/doc/plan.md``).

Every module and layer kind at the real shapes (hidden 5120, 32/8 heads x 128, intermediate 32768, seq 1024,
vocab 131072) against a torch fp64 reference fed the same bf16-rounded inputs and weights: the forward output
and every gradient (inputs and trainable parameters), each reported with PCC and relative error. Frozen
parameters are checked to receive no gradient. The torch RoPE here uses ttml's interleaved-pair convention,
so random weights can be shared without the HF row permutation; the permutation itself is covered by the
checkpoint test (``test_logits_match_huggingface``), which compares against ``MistralForCausalLM`` on the
real weights.

Run: ``pytest tt-train/tests/python/test_mistral.py -m requires_device`` (the checkpoint tests need the HF
cache; set ``MISTRAL_CHECKPOINT`` to a directory with ``config.json`` and ``model-*.safetensors``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
import torch

import ttnn
import ttml
from ttml.models.llama.transformer import LlamaBlock, LlamaMLP, RMSNormLayer
from ttml.models.llama.gqattn import GroupedQueryAttention
from ttml.models.mistral import Mistral, MistralConfig, load_from_safetensors, useful_flops_per_step
from ttml.models.mistral.frozen_embedding import embed_tokens
from ttml.modules import Embedding, LinearLayer, LoraConfig, LoraLinear, LoraModel, transpose_frozen_weights

pytestmark = pytest.mark.requires_device

torch.set_default_dtype(torch.float64)

# Mistral-Small-24B-Base-2501
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB, EPS, THETA = 5120, 32, 8, 128, 32768, 131072, 1e-5, 1e8
SEQ, BATCH = 1024, 1
RANK, ALPHA = 16, 32.0

HF_CONFIG = {
    "model_type": "mistral",
    "hidden_size": HIDDEN,
    "intermediate_size": INTER,
    "num_hidden_layers": 40,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "vocab_size": VOCAB,
    "rms_norm_eps": EPS,
    "rope_theta": THETA,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}

# Acceptance bar (bf16 storage, fp32 accumulation, one layer): PCC and relative L2 error per tensor.
FWD_PCC, FWD_REL = 0.999, 2e-2
GRAD_PCC, GRAD_REL = 0.99, 5e-2
# Gradients that pass through the softmax of a 1024x1024 attention map accumulate more bf16 rounding.
ATTN_GRAD_PCC, ATTN_GRAD_REL = 0.98, 8e-2


@pytest.fixture(autouse=True)
def _reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


# ----------------------------------------------------------------------------------------------- helpers


def bf16(a: np.ndarray) -> np.ndarray:
    """``a`` rounded to bf16 and back, as fp64: what the device sees."""
    return a.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)


def to_device(a: np.ndarray, requires_grad: bool = False) -> ttml.autograd.Tensor:
    t = ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    t.set_requires_grad(requires_grad)
    return t


def to_host(t) -> np.ndarray:
    if hasattr(t, "get_value"):
        return t.to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


def grad_of(t: ttml.autograd.Tensor) -> np.ndarray:
    assert t.is_grad_initialized(), "no gradient was produced"
    return to_host(t.get_grad())


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got.ravel() - ref.ravel()) / max(np.linalg.norm(ref.ravel()), 1e-30))


class Report:
    """Collects (name, pcc, rel err) rows, prints them, and asserts the bar per row."""

    def __init__(self, title: str) -> None:
        self.title, self.rows = title, []

    def check(self, name: str, got: np.ndarray, ref: np.ndarray, pcc_min: float, rel_max: float) -> None:
        assert got.shape == ref.shape, f"{name}: shape {got.shape} vs reference {ref.shape}"
        p, r = pcc(got, ref), rel_err(got, ref)
        self.rows.append((name, tuple(got.shape), p, r, pcc_min, rel_max))

    def done(self) -> None:
        lines = [f"{self.title}: tensor | shape | PCC | rel err | bar"]
        failures = []
        for name, shape, p, r, pmin, rmax in self.rows:
            ok = p >= pmin and r <= rmax
            lines.append(
                f"  {name:28s} {str(shape):24s} {p:.6f} {r:.2e}  (>= {pmin}, <= {rmax}) {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(name)
        print("\n".join(lines))
        assert not failures, f"{self.title}: below the bar: {failures}"


def seeded(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def assign(param, a: np.ndarray) -> None:
    """Write ``a`` (bf16-representable) into a ttml parameter tensor, keeping its layout/dtype."""
    ttml.autograd.assign_numpy(param, a.astype(np.float32))


# --------------------------------------------------------------------------------------- torch references


def t(a: np.ndarray, grad: bool = False) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


def ref_rmsnorm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * gamma


def ref_rope(x: torch.Tensor) -> torch.Tensor:
    """RoPE on ``[B, H, T, D]`` in ttml's interleaved-pair convention: pair (2i, 2i+1) rotates by freq i."""
    T, D = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = torch.arange(T, dtype=torch.float64)[:, None] * inv_freq[None, :]  # [T, D/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def ref_attention(x: torch.Tensor, w_qkv: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    """GQA causal attention on ``x [B, 1, T, H]`` with fused ``w_qkv [(Hq + 2 Hkv) D, H]`` rows [Q | K | V]."""
    B, _, T, _ = x.shape
    qkv = x[:, 0] @ w_qkv.T  # [B, T, (Hq+2Hkv) D]
    q, k, v = torch.split(qkv, [HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
    q = q.view(B, T, HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
    q, k = ref_rope(q), ref_rope(k)
    rep = HEADS // KV_HEADS
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1) @ v  # [B, Hq, T, D]
    merged = attn.transpose(1, 2).reshape(B, 1, T, HEADS * HEAD_DIM)
    return merged @ w_out.T


def ref_mlp(x: torch.Tensor, w_gate_up: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    gu = x @ w_gate_up.T
    gate, up = gu[..., :INTER], gu[..., INTER:]
    return (torch.nn.functional.silu(gate) * up) @ w2.T


def ref_lora(x, w, a, b, scaling):
    return x @ w.T + ((x @ a.T) @ b.T) * scaling


# ------------------------------------------------------------------------------------- random parameters


def rand_weight(rng, out_f, in_f, std=None):
    std = std or 1.0 / math.sqrt(in_f)
    return bf16(rng.normal(0.0, std, (1, 1, out_f, in_f)))


def rand_x(rng, width=HIDDEN, std=1.0):
    return bf16(rng.normal(0.0, std, (BATCH, 1, SEQ, width)))


def rope_params():
    return ttml.ops.rope.build_rope_params(SEQ, HEAD_DIM, THETA)


# ------------------------------------------------------------------------------------------------ tests


def test_rmsnorm_forward_and_gradients():
    rng = seeded(1)
    x_np, g_np = rand_x(rng), bf16(rng.normal(1.0, 0.1, (1, 1, 1, HIDDEN)))
    dout_np = rand_x(rng, std=0.01)

    layer = RMSNormLayer(HIDDEN, EPS)
    assign(layer.gamma.tensor, g_np)
    x = to_device(x_np, requires_grad=True)
    out = layer(x)
    out_val = to_host(out)
    _backward(out, dout_np)

    xr, gr = t(x_np, True), t(g_np, True)
    ref = ref_rmsnorm(xr, gr)
    ref.backward(t(dout_np))

    rep = Report("rmsnorm")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("dgamma", grad_of(layer.gamma.tensor), gr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()


def _backward(out: ttml.autograd.Tensor, dout_np: np.ndarray) -> None:
    """Backward from an explicit upstream gradient (the op tests feed dL/dout rather than a loss)."""
    out.set_grad(to_device(dout_np).get_value())
    out.backward(False)


def test_rmsnorm_frozen_gamma_gets_no_gradient():
    rng = seeded(2)
    layer = RMSNormLayer(HIDDEN, EPS)
    layer.gamma.tensor.set_requires_grad(False)
    x = to_device(rand_x(rng), requires_grad=True)
    out = layer(x)
    _backward(out, rand_x(rng, std=0.01))
    assert x.is_grad_initialized()
    assert not layer.gamma.tensor.is_grad_initialized()


def test_attention_forward_and_gradients():
    rng = seeded(3)
    x_np = rand_x(rng)
    w_qkv = rand_weight(rng, (HEADS + 2 * KV_HEADS) * HEAD_DIM, HIDDEN)
    w_out = rand_weight(rng, HIDDEN, HEADS * HEAD_DIM)
    dout_np = rand_x(rng, std=0.01)

    attn = GroupedQueryAttention(HIDDEN, HEADS, KV_HEADS, 0.0, rope_params(), head_dim=HEAD_DIM)
    assign(attn.qkv_linear.weight.tensor, w_qkv)
    assign(attn.out_linear.weight.tensor, w_out)
    x = to_device(x_np, requires_grad=True)
    out = attn(x, None)
    out_val = to_host(out)
    _backward(out, dout_np)

    xr, wq, wo = t(x_np, True), t(w_qkv[0, 0], True), t(w_out[0, 0], True)
    ref = ref_attention(xr, wq, wo)
    ref.backward(t(dout_np))

    rep = Report("attention (qkv, heads, rope, causal GQA sdpa, concat, out-proj)")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), ATTN_GRAD_PCC, ATTN_GRAD_REL)
    rep.check("d w_qkv", grad_of(attn.qkv_linear.weight.tensor)[0, 0], wq.grad.numpy(), ATTN_GRAD_PCC, ATTN_GRAD_REL)
    rep.check("d w_out", grad_of(attn.out_linear.weight.tensor)[0, 0], wo.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()


def test_mlp_forward_and_gradients():
    rng = seeded(4)
    x_np = rand_x(rng)
    w_gu = rand_weight(rng, 2 * INTER, HIDDEN)
    w2 = rand_weight(rng, HIDDEN, INTER)
    dout_np = rand_x(rng, std=0.01)

    mlp = LlamaMLP(HIDDEN, INTER)
    assign(mlp.w_gate_up.weight.tensor, w_gu)
    assign(mlp.w2.weight.tensor, w2)
    x = to_device(x_np, requires_grad=True)
    out = mlp(x)
    out_val = to_host(out)
    _backward(out, dout_np)

    xr, wgu, w2r = t(x_np, True), t(w_gu[0, 0], True), t(w2[0, 0], True)
    ref = ref_mlp(xr, wgu, w2r)
    ref.backward(t(dout_np))

    rep = Report("mlp (gate_up, swiglu_packed, down)")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("d w_gate_up", grad_of(mlp.w_gate_up.weight.tensor)[0, 0], wgu.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("d w2", grad_of(mlp.w2.weight.tensor)[0, 0], w2r.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()


def test_lora_linear_forward_and_gradients_frozen_base():
    """One adapted linear at the qkv shape: base weight frozen (no gradient), A and B trained."""
    rng = seeded(5)
    out_f = (HEADS + 2 * KV_HEADS) * HEAD_DIM
    x_np, w_np = rand_x(rng), rand_weight(rng, out_f, HIDDEN)
    a_np = bf16(rng.uniform(-1 / math.sqrt(HIDDEN), 1 / math.sqrt(HIDDEN), (1, 1, RANK, HIDDEN)))
    b_np = bf16(rng.normal(0.0, 0.02, (1, 1, out_f, RANK)))  # non-zero so dA is non-trivial
    dout_np = bf16(rng.normal(0.0, 0.01, (BATCH, 1, SEQ, out_f)))

    base = LinearLayer(HIDDEN, out_f, False)
    assign(base.weight.tensor, w_np)
    layer = LoraLinear(base, LoraConfig(rank=RANK, alpha=ALPHA))
    assign(layer.lora_A.tensor, a_np)
    assign(layer.lora_B.tensor, b_np)
    x = to_device(x_np, requires_grad=True)
    out = layer(x)
    out_val = to_host(out)
    _backward(out, dout_np)

    xr, wr, ar, br = t(x_np, True), t(w_np[0, 0]), t(a_np[0, 0], True), t(b_np[0, 0], True)
    ref = ref_lora(xr, wr, ar, br, ALPHA / RANK)
    ref.backward(t(dout_np))

    rep = Report("lora linear (qkv shape)")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("d lora_A", grad_of(layer.lora_A.tensor)[0, 0], ar.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("d lora_B", grad_of(layer.lora_B.tensor)[0, 0], br.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()
    assert not layer.weight.tensor.get_requires_grad()
    assert not layer.weight.tensor.is_grad_initialized(), "frozen base weight received a gradient"


@pytest.mark.parametrize("transposed", [False, True], ids=["weights [out,in]", "frozen weights transposed"])
def test_block_forward_and_gradients_under_lora(transposed):
    """A whole decoder block at real shapes with LoRA on its four linears: output, dx, every adapter gradient;
    base weights and gammas frozen and gradient-free. ``transposed`` runs the base projections through the
    transposed-weight path (``transpose_frozen_weights``) the training example uses."""
    rng = seeded(6)
    x_np = rand_x(rng)
    gammas = [bf16(rng.normal(1.0, 0.1, (1, 1, 1, HIDDEN))) for _ in range(2)]
    w_qkv = rand_weight(rng, (HEADS + 2 * KV_HEADS) * HEAD_DIM, HIDDEN)
    w_out = rand_weight(rng, HIDDEN, HEADS * HEAD_DIM, std=0.02)
    w_gu = rand_weight(rng, 2 * INTER, HIDDEN)
    w2 = rand_weight(rng, HIDDEN, INTER, std=0.004)
    dout_np = rand_x(rng, std=0.01)
    shapes = {
        "qkv_linear": (HIDDEN, w_qkv.shape[2]),
        "out_linear": (HEADS * HEAD_DIM, HIDDEN),
        "w_gate_up": (HIDDEN, 2 * INTER),
        "w2": (INTER, HIDDEN),
    }
    adapters = {
        name: (
            bf16(rng.uniform(-1 / math.sqrt(i), 1 / math.sqrt(i), (1, 1, RANK, i))),
            bf16(rng.normal(0.0, 0.02, (1, 1, o, RANK))),
        )
        for name, (i, o) in shapes.items()
    }

    block = LlamaBlock(
        HIDDEN, HEADS, KV_HEADS, rope_params(), head_dim=HEAD_DIM, intermediate_size=INTER, rms_norm_eps=EPS
    )
    assign(block.attention_norm.gamma.tensor, gammas[0])
    assign(block.mlp_norm.gamma.tensor, gammas[1])
    assign(block.attention.qkv_linear.weight.tensor, w_qkv)
    assign(block.attention.out_linear.weight.tensor, w_out)
    assign(block.mlp.w_gate_up.weight.tensor, w_gu)
    assign(block.mlp.w2.weight.tensor, w2)
    lora = LoraModel(
        block,
        LoraConfig(
            rank=RANK,
            alpha=ALPHA,
            target_modules=[r"attention\.qkv_linear$", r"attention\.out_linear$", r"mlp\.w_gate_up$", r"mlp\.w2$"],
        ),
    )
    lora_layers = {
        "qkv_linear": block.attention.qkv_linear,
        "out_linear": block.attention.out_linear,
        "w_gate_up": block.mlp.w_gate_up,
        "w2": block.mlp.w2,
    }
    for name, layer in lora_layers.items():
        assert isinstance(layer, LoraLinear), name
        assign(layer.lora_A.tensor, adapters[name][0])
        assign(layer.lora_B.tensor, adapters[name][1])
    if transposed:
        # qkv, out and gate_up have a measured win and are stored [in, out]; w2 (32768 -> 5120) does not and stays.
        converted = transpose_frozen_weights(lora)
        assert sorted(converted) == [
            "model.attention.out_linear",
            "model.attention.qkv_linear",
            "model.mlp.w_gate_up",
        ], converted
        assert tuple(block.mlp.w_gate_up.weight.tensor.shape()) == (1, 1, HIDDEN, 2 * INTER)
        assert tuple(block.mlp.w2.weight.tensor.shape()) == (1, 1, HIDDEN, INTER)
    x = to_device(x_np, requires_grad=True)
    out = lora(x, None)
    out_val = to_host(out)
    _backward(out, dout_np)

    # torch reference, same math, adapters applied as base + scaling * (x A^T) B^T
    s = ALPHA / RANK
    A = {k: t(v[0][0, 0], True) for k, v in adapters.items()}
    Bm = {k: t(v[1][0, 0], True) for k, v in adapters.items()}
    xr = t(x_np, True)

    def lin(name, inp, w):
        return inp @ w.T + ((inp @ A[name].T) @ Bm[name].T) * s

    h = ref_rmsnorm(xr, t(gammas[0]))
    Bsz, _, T, _ = h.shape
    qkv = lin("qkv_linear", h[:, 0], t(w_qkv[0, 0]))
    q, k, v = torch.split(qkv, [HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
    q = ref_rope(q.view(Bsz, T, HEADS, HEAD_DIM).transpose(1, 2))
    k = ref_rope(k.view(Bsz, T, KV_HEADS, HEAD_DIM).transpose(1, 2)).repeat_interleave(HEADS // KV_HEADS, dim=1)
    v = v.view(Bsz, T, KV_HEADS, HEAD_DIM).transpose(1, 2).repeat_interleave(HEADS // KV_HEADS, dim=1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), float("-inf"))
    merged = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(Bsz, 1, T, HEADS * HEAD_DIM)
    h = xr + lin("out_linear", merged, t(w_out[0, 0]))
    m = ref_rmsnorm(h, t(gammas[1]))
    gu = lin("w_gate_up", m, t(w_gu[0, 0]))
    ref = h + lin("w2", torch.nn.functional.silu(gu[..., :INTER]) * gu[..., INTER:], t(w2[0, 0]))
    ref.backward(t(dout_np))

    rep = Report("decoder block under LoRA")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), ATTN_GRAD_PCC, ATTN_GRAD_REL)
    for name, layer in lora_layers.items():
        bar = (ATTN_GRAD_PCC, ATTN_GRAD_REL) if name in ("qkv_linear", "out_linear") else (GRAD_PCC, GRAD_REL)
        rep.check(f"d lora_A {name}", grad_of(layer.lora_A.tensor)[0, 0], A[name].grad.numpy(), *bar)
        rep.check(f"d lora_B {name}", grad_of(layer.lora_B.tensor)[0, 0], Bm[name].grad.numpy(), *bar)
    rep.done()
    for name, tensor in lora.parameters().items():
        if "lora_" not in name:
            assert not tensor.get_requires_grad(), name
            assert not tensor.is_grad_initialized(), f"frozen {name} received a gradient"


def test_frozen_embedding_matches_the_ttml_op():
    rng = seeded(7)
    table = bf16(rng.normal(0.0, 0.02, (1, 1, VOCAB, HIDDEN)))
    ids = rng.integers(0, VOCAB, (BATCH, 1, 1, SEQ), dtype=np.uint32)
    emb = Embedding(VOCAB, HIDDEN)
    assign(emb.weight.tensor, table)
    x = ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    reference = to_host(emb(x))
    emb.weight.tensor.set_requires_grad(False)
    out = embed_tokens(emb, x)
    rep = Report("frozen embedding")
    rep.check("out", to_host(out), reference, 1.0, 0.0)
    rep.check("out vs table rows", to_host(out), table[0, 0][ids[:, 0, 0]].reshape(BATCH, 1, SEQ, HIDDEN), 1.0, 0.0)
    rep.done()
    assert emb.weight.tensor.get_value().layout == ttnn.ROW_MAJOR_LAYOUT
    assert out.get_node() is None or not emb.weight.tensor.get_requires_grad()


@pytest.mark.parametrize("transposed", [False, True], ids=["weights [out,in]", "frozen weights transposed"])
def test_lm_head_and_cross_entropy_forward_and_gradients(transposed):
    rng = seeded(8)
    x_np = rand_x(rng)
    w_np = rand_weight(rng, VOCAB, HIDDEN, std=0.02)
    targets = rng.integers(0, VOCAB, (BATCH, SEQ), dtype=np.uint32)

    head = LinearLayer(HIDDEN, VOCAB, False)
    assign(head.weight.tensor, w_np)
    head.weight.tensor.set_requires_grad(False)
    if transposed:
        assert transpose_frozen_weights(head) == [""]
    x = to_device(x_np, requires_grad=True)
    logits = head(x)
    y = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    loss = ttml.ops.loss.cross_entropy_loss(logits, y, ttml.ops.ReduceType.MEAN)
    loss_val = float(loss.to_numpy(ttnn.DataType.FLOAT32).item())
    logits_val = to_host(logits)
    loss.backward(False)

    xr = t(x_np, True)
    ref_logits = xr @ t(w_np[0, 0]).T
    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits.reshape(-1, VOCAB), torch.tensor(targets.reshape(-1).astype(np.int64))
    )
    ref_loss.backward()

    rep = Report("lm head + cross-entropy (frozen head)")
    rep.check("logits", logits_val, ref_logits.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("loss", np.array([loss_val]), np.array([ref_loss.item()]), 0.0, 1e-2)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()
    assert not head.weight.tensor.is_grad_initialized(), "frozen LM head received a gradient"


# ---------------------------------------------------------------------------------- checkpoint tests


def _checkpoint_dir() -> Path:
    env = os.environ.get("MISTRAL_CHECKPOINT")
    if env:
        return Path(env)
    root = Path.home() / ".cache/huggingface/hub/models--mistralai--Mistral-Small-24B-Base-2501/snapshots"
    snaps = sorted(root.glob("*/config.json")) if root.exists() else []
    if not snaps:
        pytest.skip("Mistral-Small-24B checkpoint not in the HF cache; set MISTRAL_CHECKPOINT")
    return snaps[-1].parent


def test_config_from_hf_reads_mistral(expect_error):
    cfg = MistralConfig.from_hf(HF_CONFIG, max_position_embeddings=SEQ)
    assert (cfg.hidden_size, cfg.head_dim, cfg.attention_head_dim, cfg.num_key_value_heads) == (
        HIDDEN,
        HEAD_DIM,
        HEAD_DIM,
        KV_HEADS,
    )
    assert (cfg.intermediate_size, cfg.vocab_size, cfg.rope_theta, cfg.rms_norm_eps) == (INTER, VOCAB, THETA, EPS)
    with expect_error(ValueError, "sliding_window"):
        MistralConfig.from_hf({**HF_CONFIG, "sliding_window": 4096}, max_position_embeddings=SEQ)
    with expect_error(ValueError, "not a Mistral"):
        MistralConfig.from_hf({**HF_CONFIG, "model_type": "llama"}, max_position_embeddings=SEQ)


def test_useful_flops_match_the_plan():
    cfg = MistralConfig.from_hf(HF_CONFIG, max_position_embeddings=SEQ)
    flops = useful_flops_per_step(cfg, SEQ, SEQ, lora_rank=RANK)
    assert abs(flops / 1e12 - 96.37) < 0.01, flops  # doc/plan.md section 3


@pytest.mark.parametrize("layers", [2])
def test_logits_match_huggingface(layers):
    """The real checkpoint through the loader (RoPE row permutation, fused qkv / gate_up) against
    ``MistralForCausalLM`` truncated to the same layers, fp64 on CPU, on real text tokens."""
    from transformers import AutoTokenizer, MistralConfig as HFMistralConfig, MistralForCausalLM

    ckpt = _checkpoint_dir()
    cfg = MistralConfig.from_hf(ckpt, max_position_embeddings=SEQ, num_hidden_layers=layers)
    model = Mistral(cfg)
    print(load_from_safetensors(model, ckpt))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    text = (Path(os.environ["TT_METAL_HOME"]) / "tt-train/data/shakespeare.txt").read_text()[:20000]
    ids = np.asarray(tokenizer(text)["input_ids"][:SEQ], dtype=np.uint32)
    assert len(ids) == SEQ
    for param in model.parameters().values():
        param.set_requires_grad(False)
    x = ttml.autograd.Tensor.from_numpy(ids.reshape(1, 1, 1, SEQ), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    model.eval()
    logits = to_host(model(x, None))

    hf_cfg = HFMistralConfig.from_pretrained(str(ckpt), num_hidden_layers=layers, torch_dtype=torch.float64)
    hf = MistralForCausalLM.from_pretrained(str(ckpt), config=hf_cfg, torch_dtype=torch.float64)
    hf.eval()
    with torch.no_grad():
        ref = hf(input_ids=torch.tensor(ids.astype(np.int64))[None]).logits.numpy()[:, None]
    rep = Report(f"checkpoint logits, {layers} of 40 layers")
    rep.check("logits", logits, ref, FWD_PCC, 3e-2)
    rep.done()
