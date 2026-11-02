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
from ttml.modules import (
    Embedding,
    LinearLayer,
    LoraConfig,
    LoraLinear,
    LoraModel,
    fuse_lora_into_frozen_weights,
    transpose_frozen_weights,
)

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


@pytest.mark.parametrize("batches", [1, 2], ids=["32 tile-rows", "64 tile-rows: two work items per core"])
def test_rmsnorm_allcore_backward_with_frozen_gamma(batches):
    """The all-core RMSNorm backward (`rmsnorm_bw_allcore`, iteration 6) on a frozen-gamma layer matches torch.
    With 64 tile-rows the 3-slice split gives 192 work items, so cores process two items in a row (the CB
    full-block discipline of iteration 9)."""
    from ttml.models.mistral.rmsnorm_allcore import use_allcore_rmsnorm_backward

    rng = seeded(3)
    x_np = bf16(rng.normal(0.0, 1.0, (1, batches, SEQ, HIDDEN)))
    g_np = bf16(rng.normal(1.0, 0.1, (1, 1, 1, HIDDEN)))
    dout_np = bf16(rng.normal(0.0, 0.01, (1, batches, SEQ, HIDDEN)))

    layer = RMSNormLayer(HIDDEN, EPS)
    assign(layer.gamma.tensor, g_np)
    layer.gamma.tensor.set_requires_grad(False)
    assert use_allcore_rmsnorm_backward(layer) == [""], "the frozen-gamma layer itself is converted"
    x = to_device(x_np, requires_grad=True)
    out = layer(x)
    out_val = to_host(out)
    _backward(out, dout_np)
    assert not layer.gamma.tensor.is_grad_initialized()

    xr, gr = t(x_np, True), t(g_np, False)
    ref = ref_rmsnorm(xr, gr)
    ref.backward(t(dout_np))

    rep = Report("rmsnorm all-core backward")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()


@pytest.mark.parametrize("batches", [1, 2], ids=["32 tile-rows", "64 tile-rows: two work items per core"])
def test_add_rmsnorm_fused_forward_and_gradients(batches):
    """The fused residual add + RMSNorm (`add_rmsnorm_fw_allcore` and the all-core backward with the residual
    gradient folded in, iteration 9) against torch: the sum, the normalized output and the gradient reaching both
    the residual and the delta through both outputs. 64 tile-rows: cores process two work items in a row."""
    from ttml.models.mistral.fused_residual_norm import add_rmsnorm

    rng = seeded(8)
    shape = (1, batches, SEQ, HIDDEN)
    r_np, d_np = bf16(rng.normal(0.0, 1.0, shape)), bf16(rng.normal(0.0, 0.5, shape))
    g_np = bf16(rng.normal(1.0, 0.1, (1, 1, 1, HIDDEN)))
    dout_np, dsum_np = bf16(rng.normal(0.0, 0.01, shape)), bf16(rng.normal(0.0, 0.01, shape))
    layer = RMSNormLayer(HIDDEN, EPS)
    assign(layer.gamma.tensor, g_np)
    layer.gamma.tensor.set_requires_grad(False)

    rep = Report("fused residual add + RMSNorm")
    # (sum, out): a gradient reaches the sum through the next residual add and the out through the sublayer
    r, d = to_device(r_np, requires_grad=True), to_device(d_np, requires_grad=True)
    s, out = add_rmsnorm(r, d, layer)
    s_val, out_val = to_host(s), to_host(out)
    s.set_grad(to_device(dsum_np).get_value())
    out.set_grad(to_device(dout_np).get_value())
    s.backward(False)
    rr, dr, gr = t(r_np, True), t(d_np, True), t(g_np)
    s_ref = rr + dr
    out_ref = ref_rmsnorm(s_ref, gr)
    (out_ref * t(dout_np)).sum().backward(retain_graph=True)
    rep.check("sum", s_val, s_ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("out", out_val, out_ref.detach().numpy(), FWD_PCC, FWD_REL)
    # the gradient through out only (return_sum=False path has the same backward without the residual term)
    rep.check("d residual (out only)", _out_only_grad(layer, r_np, d_np, dout_np), rr.grad.numpy(), GRAD_PCC, GRAD_REL)
    (s_ref * t(dsum_np)).sum().backward()
    rep.check("d residual", grad_of(r), rr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("d delta", grad_of(d), dr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()
    assert not layer.gamma.tensor.is_grad_initialized()


def _out_only_grad(layer, r_np, d_np, dout_np):
    from ttml.models.mistral.fused_residual_norm import add_rmsnorm

    r, d = to_device(r_np, requires_grad=True), to_device(d_np, requires_grad=True)
    out = add_rmsnorm(r, d, layer, return_sum=False)
    _backward(out, dout_np)
    dr, dd = grad_of(r), grad_of(d)
    assert np.array_equal(dr, dd), "the residual and the delta receive the same gradient"
    return dr


@pytest.mark.parametrize("transpose_b", [True, False], ids=["w [N, K] (x @ A^T)", "w [K, N] (g @ B)"])
@pytest.mark.parametrize("n_cols", [16, 32, 64])
@pytest.mark.parametrize("k,batches", [(HIDDEN, 1), (8192, 2)], ids=["K=5120, 32 tile-rows", "K=8192, 64 tile-rows"])
def test_skinny_matmul_matches_torch(k, batches, n_cols, transpose_b):
    """`ttml.ops.metal.skinny_matmul` (iteration 10: K split over the cores for a rank-wide W) against torch fp64 at
    the LoRA shapes: both W layouts, one and two output tile columns (N = 16 / 32 / 64), the ragged 13-slice split of
    K = 5120 and the even one of K = 8192, and 64 tile-rows (cores with two work items)."""
    rng = seeded(10)
    x_np = bf16(rng.normal(0.0, 1.0, (1, batches, SEQ, k)))
    w_np = bf16(rng.normal(0.0, 0.05, (1, 1, n_cols, k) if transpose_b else (1, 1, k, n_cols)))
    out = ttml.ops.metal.skinny_matmul(to_device(x_np).get_value(), to_device(w_np).get_value(), transpose_b)
    out_np = to_host(out)
    wr = t(w_np[0, 0])
    ref = (t(x_np) @ (wr.T if transpose_b else wr)).numpy()
    rep = Report(f"skinny_matmul K={k} N={n_cols} transpose_b={transpose_b} batches={batches}")
    rep.check("out", out_np, ref, 0.9999, 1e-2)
    rep.done()


@pytest.mark.parametrize("n_cols", [16, 32], ids=["rank 16", "rank 32"])
def test_skinny_matmul_on_the_transposed_wide_buffer(n_cols):
    """Iteration 40: ``skinny_matmul(x_transposed, transpose_out)``: x stored ``[K + 64, T]`` (the transposed wide
    LoRA buffer, the tail rows zero), w ``[N, K]``; the K-split op streams x^T's tiles as the matmul's second operand
    and gives h^T ``[N, T]`` directly, written in place into the buffer's tail rows (``tail_out``). Against torch
    fp64; the tail rows of the buffer and the padded rows past N checked."""
    rng = seeded(40)
    k = 8192
    x_t_np = bf16(rng.normal(0.0, 1.0, (1, 1, k + 64, SEQ)))
    x_t_np[:, :, k:, :] = 0.0
    w_np = bf16(rng.normal(0.0, 0.05, (1, 1, n_cols, k)))
    x_t = to_device(x_t_np)
    out = ttml.ops.metal.skinny_matmul(
        x_t.get_value(), to_device(w_np).get_value(), True, x_t.get_value(), k // 32, 1, True, True
    )
    out_np = to_host(out)
    assert out_np.shape == (1, 1, n_cols, SEQ)
    ref = (t(w_np[0, 0]) @ t(x_t_np[0, 0, :k, :])).numpy()
    rep = Report(f"skinny_matmul on x^T [K + 64, T], N={n_cols}, tail rows written")
    rep.check("out", out_np[0, 0], ref, 0.9999, 1e-2)
    buf = to_host(x_t)
    rep.check("tail rows of the buffer", buf[0, 0, k : k + n_cols, :], ref, 0.9999, 1e-2)
    rep.done()
    assert np.all(buf[0, 0, k + n_cols : k + 32, :] == 0.0), "the padded rows past N are not zero"
    assert np.array_equal(buf[0, 0, :k, :], x_t_np[0, 0, :k, :]), "the product's rows were touched"


def test_skinny_matmul_rejects_the_transposed_output_of_a_token_major_x(expect_error):
    """Review round 12: ``transpose_out`` without ``x_transposed`` (the reduce kernel's untested DEST-transpose path)
    is refused by the op's validation."""
    x = to_device(bf16(np.zeros((1, 1, SEQ, 8192))))
    w = to_device(bf16(np.zeros((1, 1, 16, 8192))))
    with expect_error(RuntimeError, "transposed output is only produced"):
        ttml.ops.metal.skinny_matmul(x.get_value(), w.get_value(), True, None, 0, 0, False, True)


def test_skinny_contract_repeats_on_the_cached_program():
    """Review round 12: two calls of one shape on one program-cache entry (the multicast handshake re-armed by the
    override of the runtime arguments) both match torch; the second call's operands live at other addresses."""
    rng = seeded(41)
    outs = []
    alive = []
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        small_np = bf16(r.normal(0, 1, (1, 1, 16, SEQ)))
        big_np = bf16(r.normal(0, 1, (1, 1, 16384, SEQ)))
        small, big = to_device(small_np), to_device(big_np)
        got = ttml.ops.metal.skinny_contract(
            small.get_value(), big.get_value(), 2.0, True, big_transposed=True, small_transposed=True
        )
        alive.append((small, big, got))
        ref = 2.0 * (t(big_np[0, 0]) @ t(small_np[0, 0]).T)
        rep = Report(f"skinny_contract on the cached program, call {seed}")
        rep.check("out", to_host(got)[0, 0], ref.numpy(), 0.9999, 1e-2)
        rep.done()


@pytest.mark.parametrize(
    "n_cols, scale, transpose_b, images",
    [(5120, 1.0, False, 1), (1536, 2.0, True, 1), (1024, 0.5, True, 2)],
    ids=["dx-plus-dh-A", "base-plus-s-h-Bt", "two-images"],
)
def test_lowrank_update_matches_torch(n_cols, scale, transpose_b, images):
    """`acc + scale * a @ b` (rank 16, one K tile) in one metal op against torch fp64 on bf16-rounded inputs."""
    rng = seeded(12)
    rows = 1024 if images == 1 else 256
    acc_np = bf16(rng.normal(0, 1, (1, images, rows, n_cols)))
    a_np = bf16(rng.normal(0, 1, (1, images, rows, 16)))
    b_np = bf16(rng.normal(0, 0.05, (1, 1, n_cols, 16) if transpose_b else (1, 1, 16, n_cols)))
    got = to_host(
        ttml.ops.metal.lowrank_update(
            to_device(acc_np).get_value(), to_device(a_np).get_value(), to_device(b_np).get_value(), scale, transpose_b
        )
    )
    b_t = t(b_np).transpose(-1, -2) if transpose_b else t(b_np)
    ref = (t(acc_np) + scale * (t(a_np) @ b_t)).numpy()
    rep = Report(f"lowrank_update N={n_cols} scale={scale} transpose_b={transpose_b} images={images}")
    rep.check("out", got, ref, 0.9999, 1e-2)
    rep.done()


@pytest.mark.parametrize("rows, rank, scale", [(5120, 16, 1.0), (512, 32, 0.25)], ids=["gate_up-route", "tp-scaled"])
def test_transpose_split_matches_numpy(rows, rank, scale):
    """`transpose_split` of a `[rows + 64, T]` product: dx = rows[:rows]^T, tail = scale * rows[rows:rows+rank]^T."""
    rng = seeded(13)
    T = 1024
    x_np = bf16(rng.normal(0, 1, (1, 1, rows + 64, T)))
    dx, tail = ttml.ops.metal.transpose_split(to_device(x_np).get_value(), rows, rank, scale)
    rep = Report(f"transpose_split rows={rows} rank={rank} scale={scale}")
    rep.check("dx", to_host(dx), np.swapaxes(x_np[:, :, :rows, :], -1, -2), 0.9999, 1e-6)
    rep.check("tail", to_host(tail), bf16(scale * np.swapaxes(x_np[:, :, rows : rows + rank, :], -1, -2)), 0.9999, 1e-6)
    rep.done()


@pytest.mark.parametrize(
    "vocab, g, scale", [(32768, 1.0, 1.0 / 1024), (4096, 0.5, 1.0)], ids=["lm-head-shard", "small"]
)
def test_vp_cross_entropy_bw_matches_numpy(vocab, g, scale):
    """The fused vocab-parallel cross-entropy backward on one shard (first_v = 0) against numpy fp64:
    scale * g * (exp(x - logZ) - onehot(target))."""
    rng = seeded(14)
    x_np = bf16(rng.normal(0, 2, (1, 1, SEQ, vocab)))
    targets = rng.integers(0, vocab, (1, SEQ)).astype(np.int64)
    m = x_np.max(-1, keepdims=True)
    logz = np.log(np.exp(x_np - m).sum(-1, keepdims=True)) + m
    onehot = np.zeros_like(x_np)
    onehot[0, 0, np.arange(SEQ), targets[0]] = 1.0
    ref = scale * g * (np.exp(x_np - logz) - onehot)
    dev = ttml.autograd.AutoContext.get_instance().get_device()
    x = ttnn.from_torch(torch.tensor(x_np, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    lz = ttnn.from_torch(torch.tensor(logz, dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=dev)
    tg = ttnn.from_torch(
        torch.tensor(targets, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    gt = ttnn.from_torch(
        torch.full((1, 1, 32, 32), g, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev
    )  # a tile of g
    got = to_host(ttml.ops.metal.vp_cross_entropy_bw(x, lz, tg, gt, vocab, None, 0, scale))
    rep = Report(f"vp_cross_entropy_bw V={vocab} g={g} scale={scale:.2e}")
    rep.check("grad", got, ref, 0.9999, 1e-2)
    rows = np.arange(SEQ)
    # The target entries are all ~ -scale * g (a near-constant vector, PCC meaningless): relative error only.
    rep.check("grad at targets", got[0, 0, rows, targets[0]], ref[0, 0, rows, targets[0]], -1.0, 2e-2)
    rep.done()


@pytest.mark.parametrize(
    "big_transposed, small_transposed",
    [(False, False), (True, False), (True, True)],
    ids=["token-major", "big^T", "big^T, small^T"],
)
@pytest.mark.parametrize(
    "n_cols, scale, transpose_out",
    # 1536: the routed qkv dB shape; 7072: Nt = 221, the short last chunk of a core's column range (the padding
    # path); 16384: 512 column tiles over 110 cores (4 or 5 per core, iteration 40); 8256: the w2 dA shape
    [
        (16384, 2.0, True),
        (5120, 1.0, False),
        (1024, 0.5, True),
        (1536, 0.5, True),
        (7072, 1.0, True),
        (8256, 1.0, False),
    ],
    ids=["dB-gate_up", "dA-in5120", "dB-out-scaled", "dB-qkv-routed", "dB-short-last-group", "dA-w2-wide"],
)
def test_skinny_contract_matches_torch(n_cols, scale, transpose_out, big_transposed, small_transposed):
    """`scale * small^T @ big` over the token dimension (rank 16) in one op against torch fp64 on bf16 inputs; the
    small operand multicast from one core (iteration 40) and every core on a contiguous column range; the
    transposed operands of iterations 38-40 (big stored [N, T], small stored [r, T])."""
    rng = seeded(18)
    small_np = bf16(rng.normal(0, 1, (1, 1, SEQ, 16)))
    big_np = bf16(rng.normal(0, 1, (1, 1, SEQ, n_cols)))
    small_in = np.ascontiguousarray(np.swapaxes(small_np, -1, -2)) if small_transposed else small_np
    big_in = np.ascontiguousarray(np.swapaxes(big_np, -1, -2)) if big_transposed else big_np
    got = to_host(
        ttml.ops.metal.skinny_contract(
            to_device(small_in).get_value(),
            to_device(big_in).get_value(),
            scale,
            transpose_out,
            big_transposed=big_transposed,
            small_transposed=small_transposed,
        )
    )
    ref = scale * (t(small_np).transpose(-1, -2) @ t(big_np))
    if transpose_out:
        ref = ref.transpose(-1, -2)
    rep = Report(
        f"skinny_contract N={n_cols} scale={scale} transpose_out={transpose_out} big^T={big_transposed} small^T={small_transposed}"
    )
    rep.check("out", got, ref.numpy(), 0.9999, 1e-2)
    rep.done()


def test_fused_residual_norm_model_matches_the_block_path():
    """A 2-layer model on the final training path (transposed + fused LoRA, all-core norm backward) with and without
    the fused residual norm: the same loss and the same LoRA gradients; the fused forward builds the expected number
    of fused norms (2 per layer, the final norm included, the first norm plain)."""
    from ttml.common.utils import set_seed
    from ttml.models.mistral.fused_residual_norm import use_fused_residual_norm
    from ttml.models.mistral.rmsnorm_allcore import use_allcore_rmsnorm_backward

    layers = 2
    cfg = MistralConfig.from_hf(HF_CONFIG, max_position_embeddings=SEQ, num_hidden_layers=layers)
    rng = seeded(9)
    ids = rng.integers(0, VOCAB, (1, 1, 1, SEQ), dtype=np.uint32)
    targets = rng.integers(0, VOCAB, (1, SEQ), dtype=np.uint32)
    lora_B = {}
    fingerprints = []

    def build(fused_residual):
        set_seed(1234)
        model = LoraModel(
            Mistral(cfg),
            LoraConfig(
                rank=RANK,
                alpha=ALPHA,
                target_modules=[
                    r"attention\.qkv_linear$",
                    r"attention\.out_linear$",
                    r"mlp\.w_gate_up$",
                    r"mlp\.w2$",
                ],
            ),
        )
        for name, tensor in model.parameters().items():
            if "lora_B" in name:  # zero-initialised: random so that the A gradients are not zero
                if name not in lora_B:
                    lora_B[name] = bf16(rng.normal(0.0, 0.02, tuple(tensor.shape())))
                assign(tensor, lora_B[name])
        qkv = next(n for n in model.parameters() if n.endswith("blocks/0/attention/qkv_linear/weight"))
        fingerprints.append(float(np.abs(to_host(model.parameters()[qkv])).sum()))
        assert fingerprints[0] == fingerprints[-1], "the two models must start from the same random init"
        transpose_frozen_weights(model)
        fuse_lora_into_frozen_weights(model)
        use_allcore_rmsnorm_backward(model)
        if fused_residual:
            assert use_fused_residual_norm(model) == 2 * layers
        model.train()
        return model

    def run(model):
        x = ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
        y = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
        loss = ttml.ops.loss.cross_entropy_loss(model(x, None), y, ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        grads = {n: grad_of(p) for n, p in model.parameters().items() if p.get_requires_grad()}
        loss_val = float(to_host(loss).ravel()[0])
        ttml.autograd.AutoContext.get_instance().reset_graph()
        return loss_val, grads

    loss_ref, grads_ref = run(build(False))
    loss_fused, grads_fused = run(build(True))
    print(f"loss: block path {loss_ref:.5f}, fused residual norm {loss_fused:.5f}")
    assert abs(loss_fused - loss_ref) <= 0.02 + 0.01 * abs(loss_ref)
    rep = Report("fused residual norm vs the block path, 2 layers")
    assert grads_ref.keys() == grads_fused.keys() and len(grads_ref) == 8 * layers
    for name in sorted(grads_ref):
        rep.check(name, grads_fused[name], grads_ref[name], GRAD_PCC, GRAD_REL)
    rep.done()


def test_rmsnorm_allcore_path_builds_no_node_for_a_no_grad_input():
    """A no-grad input (the frozen embedding output) through the all-core norm wrapper takes the C++ op and gets
    no backward node, so the first block does not compute a gradient it would discard (review round 3)."""
    from ttml.models.mistral.rmsnorm_allcore import use_allcore_rmsnorm_backward

    rng = seeded(4)
    layer = RMSNormLayer(HIDDEN, EPS)
    layer.gamma.tensor.set_requires_grad(False)
    use_allcore_rmsnorm_backward(layer)
    x = to_device(rand_x(rng), requires_grad=False)
    out = layer(x)
    assert not out.get_requires_grad()
    assert out.get_node() is None


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


@pytest.mark.parametrize(
    "q_heads, kv_heads, seq",
    [(8, 2, 1024), (8, 2, 2048), (4, 1, 1024)],
    ids=["tp4-shard", "two-rows-per-core", "one-group"],
)
def test_sdpa_causal_gqa_forward_and_gradients(q_heads, kv_heads, seq):
    """The SDPA kernels alone (LPT row lists; head-split KV backward with the per-head reduce) against torch
    fp64 causal GQA attention: the TP=4 shard shape, a shape with several rows per core, a one-group shape."""
    rng = seeded(11)
    q_np = bf16(rng.normal(0, 1, (1, q_heads, seq, HEAD_DIM)))
    k_np = bf16(rng.normal(0, 1, (1, kv_heads, seq, HEAD_DIM)))
    v_np = bf16(rng.normal(0, 1, (1, kv_heads, seq, HEAD_DIM)))
    dout_np = bf16(rng.normal(0, 0.01, (1, q_heads, seq, HEAD_DIM)))
    q, k, v = (to_device(a, requires_grad=True) for a in (q_np, k_np, v_np))
    out = ttml.ops.attention.scaled_dot_product_attention(q, k, v)
    out_val = to_host(out)
    _backward(out, dout_np)

    qr, kr, vr = t(q_np, True), t(k_np, True), t(v_np, True)
    group = q_heads // kv_heads
    kk, vv = kr.repeat_interleave(group, dim=1), vr.repeat_interleave(group, dim=1)
    scores = (qr @ kk.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    ref = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1) @ vv
    ref.backward(t(dout_np))

    rep = Report(f"sdpa causal GQA q{q_heads} kv{kv_heads} S{seq}")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dQ", grad_of(q), qr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("dK", grad_of(k), kr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("dV", grad_of(v), vr.grad.numpy(), GRAD_PCC, GRAD_REL)
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


@pytest.mark.parametrize(
    "in_f,out_f", [(HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM // 4), (HEADS * HEAD_DIM // 4, HIDDEN)]
)
def test_lora_linear_configured_base_at_the_tp_shard_shapes(in_f, out_f):
    """Iteration 46: the composite LoRA path's frozen base at the TP=4 per-chip attention shapes (qkv 5120 -> 1536,
    out 1024 -> 5120, not transposed shapes) runs the swept 2D program configs through ConfiguredFrozenLinear:
    the routing must pick it, and output, dx and the adapter gradients match torch."""
    from ttml.modules.frozen_linear import configured_base_shape

    rng = seeded(46)
    x_np, w_np = rand_x(rng, width=in_f), rand_weight(rng, out_f, in_f)
    a_np = bf16(rng.uniform(-1 / math.sqrt(in_f), 1 / math.sqrt(in_f), (1, 1, RANK, in_f)))
    b_np = bf16(rng.normal(0.0, 0.02, (1, 1, out_f, RANK)))
    dout_np = bf16(rng.normal(0.0, 0.01, (BATCH, 1, SEQ, out_f)))

    base = LinearLayer(in_f, out_f, False)
    assign(base.weight.tensor, w_np)
    layer = LoraLinear(base, LoraConfig(rank=RANK, alpha=ALPHA))
    assign(layer.lora_A.tensor, a_np)
    assign(layer.lora_B.tensor, b_np)
    assert configured_base_shape(layer.weight.tensor, SEQ) == (in_f, out_f), "the swept config must be found"
    x = to_device(x_np, requires_grad=True)
    out = layer(x)
    out_val = to_host(out)
    _backward(out, dout_np)

    xr, wr, ar, br = t(x_np, True), t(w_np[0, 0]), t(a_np[0, 0], True), t(b_np[0, 0], True)
    ref = ref_lora(xr, wr, ar, br, ALPHA / RANK)
    ref.backward(t(dout_np))

    rep = Report(f"lora linear, configured base ({in_f} -> {out_f})")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("d lora_A", grad_of(layer.lora_A.tensor)[0, 0], ar.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("d lora_B", grad_of(layer.lora_B.tensor)[0, 0], br.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()
    assert not layer.weight.tensor.is_grad_initialized(), "frozen base weight received a gradient"
    # an input without a gradient (the first block's qkv input) gets no dx matmul and no gradient
    x2 = to_device(x_np, requires_grad=False)
    out2 = layer(x2)
    _backward(out2, dout_np)
    assert not x2.is_grad_initialized()


def test_mlp_fused_lora_wide_w2():
    """The fused-LoRA wide input path (iteration 24): an MLP at the TP=4 shard width (w2: 8192 -> 5120, a
    transposed shape) with LoRA on both linears, fused. SwiGLU writes the [T, 8192 + 64] buffer, h goes into
    its tail (no concat / pad), the wide gradient flows back through transpose_split, lowrank_update and the
    SwiGLU backward. Output, dx and every adapter gradient vs torch; the hook must be installed."""
    rng = seeded(9)
    inter = 8192
    x_np = rand_x(rng)
    w_gu = rand_weight(rng, 2 * inter, HIDDEN)
    w2 = rand_weight(rng, HIDDEN, inter, std=0.004)
    dout_np = rand_x(rng, std=0.01)
    shapes = {"w_gate_up": (HIDDEN, 2 * inter), "w2": (inter, HIDDEN)}
    adapters = {
        name: (
            bf16(rng.uniform(-1 / math.sqrt(i), 1 / math.sqrt(i), (1, 1, RANK, i))),
            bf16(rng.normal(0.0, 0.02, (1, 1, o, RANK))),
        )
        for name, (i, o) in shapes.items()
    }
    mlp = LlamaMLP(HIDDEN, inter)
    assign(mlp.w_gate_up.weight.tensor, w_gu)
    assign(mlp.w2.weight.tensor, w2)
    lora = LoraModel(mlp, LoraConfig(rank=RANK, alpha=ALPHA, target_modules=[r"w_gate_up$", r"w2$"]))
    layers = {"w_gate_up": mlp.w_gate_up, "w2": mlp.w2}
    for name, layer in layers.items():
        assert isinstance(layer, LoraLinear), name
        assign(layer.lora_A.tensor, adapters[name][0])
        assign(layer.lora_B.tensor, adapters[name][1])
    converted = transpose_frozen_weights(lora)
    fused = fuse_lora_into_frozen_weights(lora)
    assert sorted(fused) == ["model.w2", "model.w_gate_up"], (converted, fused)
    assert getattr(mlp, "swiglu_op", None) is not None, "the wide SwiGLU hook was not installed on the MLP"
    # Iteration 36: the w2 shard shape has a transposed-orientation forward config, so the wrapper carries the
    # [out, in + 64] forward copy (its tail columns rewritten every forward).
    w2_fwd = getattr(mlp.w2, "weight_cat_fwd", None)
    assert w2_fwd is not None and tuple(w2_fwd.shape)[-2:] == (HIDDEN, inter + 64), "w2 forward copy missing"
    # Iteration 38: that w2 takes its wide input transposed from the SwiGLU (h^T in a [in + 64, T] buffer).
    assert getattr(mlp.w2, "transposed_input", False)
    # Iteration 39: the gate_up shard shape has one too and hands its output transposed ([2I, T]) to the SwiGLU,
    # which runs elementwise on the transposed layout (no transpose anywhere in the MLP interior).
    gu_fwd = getattr(mlp.w_gate_up, "weight_cat_fwd", None)
    assert gu_fwd is not None and tuple(gu_fwd.shape)[-2:] == (2 * inter, HIDDEN + 64), "gate_up forward copy missing"
    assert getattr(mlp.w_gate_up, "transposed_output", False) and not getattr(mlp.w_gate_up, "transposed_input", False)
    x = to_device(x_np, requires_grad=True)
    out = lora(x)
    out_val = to_host(out)
    _backward(out, dout_np)

    s = ALPHA / RANK
    A = {k: t(v[0][0, 0], True) for k, v in adapters.items()}
    Bm = {k: t(v[1][0, 0], True) for k, v in adapters.items()}
    xr = t(x_np, True)
    gu = ref_lora(xr, t(w_gu[0, 0]), A["w_gate_up"], Bm["w_gate_up"], s)
    gate, up = gu[..., :inter], gu[..., inter:]
    ref = ref_lora(torch.nn.functional.silu(gate) * up, t(w2[0, 0]), A["w2"], Bm["w2"], s)
    ref.backward(t(dout_np))

    rep = Report("mlp, fused LoRA, wide w2 input (no K-concat)")
    rep.check("out", out_val, ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    for name, layer in layers.items():
        rep.check(f"dA {name}", grad_of(layer.lora_A.tensor)[0, 0], A[name].grad.numpy(), GRAD_PCC, GRAD_REL)
        rep.check(f"dB {name}", grad_of(layer.lora_B.tensor)[0, 0], Bm[name].grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()


def test_fused_residual_norm_reads_the_transposed_w2_output():
    """Iteration 37: the fused w2 (shard width, transposed-orientation forward copy) hands its output transposed
    (``[out, T]``, ``transposed_output``) to the fused residual norm, which reads it transposed in its kernel and
    returns that input's gradient transposed; the w2 backward takes the transposed gradient. Sum, normalized
    output, dx, the residual gradient and every adapter gradient vs torch; the same chain the model runs under
    ``use_fused_residual_norm`` at TP."""
    from ttml.models.mistral.fused_residual_norm import AddRMSNormFrozenGamma

    rng = seeded(11)
    inter = 8192
    x_np = rand_x(rng)
    res_np = rand_x(rng)
    gamma_np = bf16(rng.normal(1.0, 0.1, (1, 1, 1, HIDDEN)))
    w_gu = rand_weight(rng, 2 * inter, HIDDEN)
    w2 = rand_weight(rng, HIDDEN, inter, std=0.004)
    dout_np = rand_x(rng, std=0.01)
    dsum_np = rand_x(rng, std=0.01)
    shapes = {"w_gate_up": (HIDDEN, 2 * inter), "w2": (inter, HIDDEN)}
    adapters = {
        name: (
            bf16(rng.uniform(-1 / math.sqrt(i), 1 / math.sqrt(i), (1, 1, RANK, i))),
            bf16(rng.normal(0.0, 0.02, (1, 1, o, RANK))),
        )
        for name, (i, o) in shapes.items()
    }
    mlp = LlamaMLP(HIDDEN, inter)
    assign(mlp.w_gate_up.weight.tensor, w_gu)
    assign(mlp.w2.weight.tensor, w2)
    lora = LoraModel(mlp, LoraConfig(rank=RANK, alpha=ALPHA, target_modules=[r"w_gate_up$", r"w2$"]))
    layers = {"w_gate_up": mlp.w_gate_up, "w2": mlp.w2}
    for name, layer in layers.items():
        assign(layer.lora_A.tensor, adapters[name][0])
        assign(layer.lora_B.tensor, adapters[name][1])
    transpose_frozen_weights(lora)
    fuse_lora_into_frozen_weights(lora)
    assert getattr(mlp.w2, "weight_cat_fwd", None) is not None
    mlp.w2.transposed_output = True  # what use_fused_residual_norm sets on the model
    x = to_device(x_np, requires_grad=True)
    residual = to_device(res_np, requires_grad=True)
    gamma = to_device(gamma_np)
    y_t = lora(x)
    assert tuple(int(v) for v in y_t.get_value().shape)[-2:] == (HIDDEN, SEQ), "the w2 output is not transposed"
    s_dev, out = AddRMSNormFrozenGamma.apply(residual, y_t, gamma, EPS, True)
    s_dev.set_grad(to_device(dsum_np).get_value())
    out.set_grad(to_device(dout_np).get_value())
    out.backward(False)

    s = ALPHA / RANK
    A = {k: t(v[0][0, 0], True) for k, v in adapters.items()}
    Bm = {k: t(v[1][0, 0], True) for k, v in adapters.items()}
    xr, rr = t(x_np, True), t(res_np, True)
    gu = ref_lora(xr, t(w_gu[0, 0]), A["w_gate_up"], Bm["w_gate_up"], s)
    gate, up = gu[..., :inter], gu[..., inter:]
    y = ref_lora(torch.nn.functional.silu(gate) * up, t(w2[0, 0]), A["w2"], Bm["w2"], s)
    sum_ref = rr + y
    out_ref = ref_rmsnorm(sum_ref, t(gamma_np))
    ((out_ref * t(dout_np)).sum() + (sum_ref * t(dsum_np)).sum()).backward()

    rep = Report("fused w2 (transposed output) -> fused residual norm")
    rep.check("sum", to_host(s_dev), sum_ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("out", to_host(out), out_ref.detach().numpy(), FWD_PCC, FWD_REL)
    rep.check("dx", grad_of(x), xr.grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.check("dresidual", grad_of(residual), rr.grad.numpy(), GRAD_PCC, GRAD_REL)
    for name, layer in layers.items():
        rep.check(f"dA {name}", grad_of(layer.lora_A.tensor)[0, 0], A[name].grad.numpy(), GRAD_PCC, GRAD_REL)
        rep.check(f"dB {name}", grad_of(layer.lora_B.tensor)[0, 0], Bm[name].grad.numpy(), GRAD_PCC, GRAD_REL)
    rep.done()


@pytest.mark.parametrize(
    "transposed",
    ["plain", "transposed", "fused"],
    ids=["weights [out,in]", "frozen weights transposed", "transposed + LoRA fused into the base matmul"],
)
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
    if transposed != "plain":
        # qkv, out and gate_up have a measured win and are stored [in, out]; w2 (32768 -> 5120) does not and stays.
        converted = transpose_frozen_weights(lora)
        assert sorted(converted) == [
            "model.attention.out_linear",
            "model.attention.qkv_linear",
            "model.mlp.w_gate_up",
        ], converted
        assert tuple(block.mlp.w_gate_up.weight.tensor.shape()) == (1, 1, HIDDEN, 2 * INTER)
        assert tuple(block.mlp.w2.weight.tensor.shape()) == (1, 1, HIDDEN, INTER)
    if transposed == "fused":
        fused = fuse_lora_into_frozen_weights(lora)
        assert sorted(fused) == sorted(converted), fused
        assert tuple(block.mlp.w_gate_up.weight.tensor.shape()) == (1, 1, HIDDEN + 64, 2 * INTER)
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
    # The frozen lookup lives outside the autograd graph (review round 3, finding 1): no node, no gradient claim.
    assert not out.get_requires_grad()
    assert out.get_node() is None


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
