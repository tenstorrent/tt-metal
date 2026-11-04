# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter for tt-train (TT_PORT_PLAN.md D2), in the shape tt-blaze serves.

Reference: speculators ``DFlashDraftModel`` (``models/dflash/core.py`` and
``model_definitions.py``). Parameter names follow the speculators state dict, except
that each layer's ``k_proj``/``v_proj`` are fused into ``kv_proj`` (``[K | V]`` rows),
which ``grouped_heads_creation`` consumes directly.

Attention runs on the square fused SDPA (D1a). For every layer::

    X = concat([context ; input_layernorm(block)])          # [1, 1, S, H], S = T + n*B
    q, k, v = q_proj(X), k_proj(X), v_proj(X)               # q/k-norm, then RoPE
    attn = sdpa(q, k, v, square_mask[layer_type])[block rows]

Context rows of ``q`` are real but discarded: the square mask lets them see only
themselves and their outputs are sliced away, so they receive zero gradient. Block
rows see exactly the keys speculators' rectangular ``[n*B, T + n*B]`` mask allows.

RoPE uses HF's half-rotation layout with host-computed (YaRN) cos/sin, so HF/
speculators q/k weights load without the row permutation ttml's interleaved rope op
would need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import ttnn

import ttml
from ttml.models.deepseek.autograd_ops import autograd_concat, autograd_slice, autograd_split
from ttml.models.qwen3.transformer import Qwen3MLP, Qwen3RMSNorm
from ttml.modules import AbstractModuleBase, Embedding, LinearLayer, ModuleList

from .host import DFlashDrafterConfig, DFlashHostBatch

Capture = Callable[[str, "ttml.autograd.Tensor"], None]


def _no_capture(_name: str, _t) -> None:
    return None


# ---------------------------------------------------------------------------
# Device batch
# ---------------------------------------------------------------------------


def _const(arr: np.ndarray, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE) -> ttml.autograd.Tensor:
    t = ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(arr), layout=layout, new_type=dtype)
    t.set_requires_grad(False)
    return t


@dataclass
class DFlashDeviceBatch:
    """Device tensors for one step (all non-trainable inputs)."""

    hidden_states: ttml.autograd.Tensor  # [1, 1, T, k*H]
    teacher_hidden: ttml.autograd.Tensor  # [1, 1, n*B, H]
    noise_token_ids: ttml.autograd.Tensor  # [1, 1, 1, n*B] uint32 row-major
    cos: ttml.autograd.Tensor  # [1, 1, S, D]
    sin_signed: ttml.autograd.Tensor  # [1, 1, S, D]
    masks: dict[str, ttml.autograd.Tensor]  # [1, 1, S, S] 1/0
    slot_weights: ttml.autograd.Tensor  # [1, 1, n*B, 1]
    context_len: int
    num_slots: int

    @classmethod
    def from_host(cls, hb: DFlashHostBatch) -> "DFlashDeviceBatch":
        T, nB = hb.context_len, hb.num_slots
        S = T + nB
        return cls(
            hidden_states=_const(hb.hidden_states.reshape(1, 1, T, -1)),
            teacher_hidden=_const(hb.teacher_hidden.reshape(1, 1, nB, -1)),
            noise_token_ids=_const(
                hb.noise_token_ids.reshape(1, 1, 1, nB).astype(np.uint32),
                dtype=ttnn.DataType.UINT32,
                layout=ttnn.Layout.ROW_MAJOR,
            ),
            cos=_const(hb.cos.reshape(1, 1, S, -1)),
            sin_signed=_const(hb.sin_signed.reshape(1, 1, S, -1)),
            masks={k: _const(v.astype(np.float32).reshape(1, 1, S, S)) for k, v in hb.masks.items()},
            # Weights stay fp32: they are tiny and carry the 1/sum(mask) normalization.
            slot_weights=_const(hb.slot_weights.reshape(1, 1, nB, 1), dtype=ttnn.DataType.FLOAT32),
            context_len=T,
            num_slots=nB,
        )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def rope_half(x: ttml.autograd.Tensor, cos: ttml.autograd.Tensor, sin_signed: ttml.autograd.Tensor):
    """HF ``x * cos + rotate_half(x) * sin`` on ``[B, H, S, D]`` (cos/sin ``[1, 1, S, D]``)."""
    half = x.shape()[-1] // 2
    x1, x2 = autograd_split(x, [half, half], 3)
    swapped = autograd_concat([x2, x1], 3)
    return ttml.ops.binary.add(
        ttml.ops.binary.mul(x, cos),
        ttml.ops.binary.mul(swapped, sin_signed),
    )


class DFlashAttention(AbstractModuleBase):
    def __init__(self, cfg: DFlashDrafterConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        init = ttml.init.normal(0.0, 0.02)
        self.q_proj = LinearLayer(cfg.hidden_size, cfg.num_heads * cfg.head_dim, False, weight_init=init)
        self.kv_proj = LinearLayer(cfg.hidden_size, 2 * cfg.num_kv_heads * cfg.head_dim, False, weight_init=init)
        self.o_proj = LinearLayer(cfg.num_heads * cfg.head_dim, cfg.hidden_size, False, weight_init=init)
        self.q_norm = Qwen3RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x_square, cos, sin_signed, mask, context_len: int, capture: Capture = _no_capture):
        q = self.q_proj(x_square)
        kv = self.kv_proj(x_square)
        qh, kh, vh = ttml.ops.multi_head_utils.grouped_heads_creation(q, kv, self.num_heads, self.num_kv_heads)
        qh = rope_half(self.q_norm(qh), cos, sin_signed)
        kh = rope_half(self.k_norm(kh), cos, sin_signed)
        capture("q", qh)
        capture("k", kh)
        capture("v", vh)
        attn = ttml.ops.attention.scaled_dot_product_attention(qh, kh, vh, mask)
        batch, heads, s, d = attn.shape()
        attn = autograd_slice(attn, [0, 0, context_len, 0], [batch, heads, s, d])
        capture("sdpa", attn)
        return self.o_proj(ttml.ops.multi_head_utils.heads_fusion(attn))


class DFlashDecoderLayer(AbstractModuleBase):
    def __init__(self, cfg: DFlashDrafterConfig, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = DFlashAttention(cfg)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, context, batch: DFlashDeviceBatch, capture: Capture = _no_capture):
        h = self.input_layernorm(x)
        capture("input_norm", h)
        x_square = autograd_concat([context, h], 2)
        attn = self.self_attn(
            x_square,
            batch.cos,
            batch.sin_signed,
            batch.masks[self.layer_type],
            batch.context_len,
            capture=lambda name, t: capture(f"attn.{name}", t),
        )
        capture("attn", attn)
        x = ttml.ops.binary.add(x, attn)
        mlp = self.mlp(self.post_attention_layernorm(x))
        capture("mlp", mlp)
        return ttml.ops.binary.add(x, mlp)


class DFlashDrafter(AbstractModuleBase):
    """Trainable: fc, hidden_norm, layers, norm. Frozen (from the target): embed_tokens,
    lm_head, verifier_norm (the teacher reuses lm_head, as speculators' verifier_lm_head
    is a copy of it)."""

    def __init__(self, cfg: DFlashDrafterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.fc = LinearLayer(cfg.num_aux_layers * H, H, False, weight_init=ttml.init.normal(0.0, 0.02))
        self.hidden_norm = Qwen3RMSNorm(H, eps=cfg.rms_norm_eps)
        self.layers = ModuleList([DFlashDecoderLayer(cfg, t) for t in cfg.layer_types])
        self.norm = Qwen3RMSNorm(H, eps=cfg.rms_norm_eps)
        self.embed_tokens = Embedding(cfg.vocab_size, H)
        self.lm_head = LinearLayer(H, cfg.vocab_size, False, weight_init=ttml.init.normal(0.0, 0.02))
        self.verifier_norm = Qwen3RMSNorm(H, eps=cfg.rms_norm_eps)
        for frozen in (self.embed_tokens.weight, self.lm_head.weight, self.verifier_norm.weight):
            frozen.tensor.set_requires_grad(False)

    def forward(self, batch: DFlashDeviceBatch, capture: Capture = _no_capture):
        """Draft logits ``[1, 1, n*B, V]`` for every block slot (padded slots included)."""
        noise = self.embed_tokens(batch.noise_token_ids)  # [1, 1, n*B, H]
        capture("embed", noise)
        fc = self.fc(batch.hidden_states)
        capture("fc", fc)
        context = self.hidden_norm(fc)  # [1, 1, T, H]
        capture("hidden_norm", context)
        x = noise
        for i, layer in enumerate(self.layers):
            x = layer(x, context, batch, capture=lambda name, t, i=i: capture(f"layer{i}.{name}", t))
            capture(f"layer{i}.out", x)
        draft = self.norm(x)
        capture("norm", draft)
        logits = self.lm_head(draft)
        capture("logits", logits)
        return logits

    def teacher_logits(self, batch: DFlashDeviceBatch) -> ttml.autograd.Tensor:
        """``lm_head(verifier_norm(final state))`` at each slot's graded position (no grad)."""
        ctx = ttml.autograd.AutoContext.get_instance()
        mode = ctx.get_gradient_mode()
        ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        try:
            return self.lm_head(self.verifier_norm(batch.teacher_hidden))
        finally:
            ctx.set_gradient_mode(mode)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def dflash_kl_loss(
    logits: ttml.autograd.Tensor,
    teacher_logits: ttml.autograd.Tensor,
    slot_weights: ttml.autograd.Tensor,
) -> tuple[ttml.autograd.Tensor, float]:
    """``sum_slots w * KL(p_teacher || softmax(logits))`` (speculators ``kl_div_loss`` with
    ``loss_function``'s mask, decay and ``1/sum(mask)``, folded into ``slot_weights``).

    Gradient path, from ops with autograd (``sum p log p`` is constant w.r.t. the drafter)::

        loss_for_backward = -(N*V) * mean(log_softmax(logits) * (w * p))

    The reported value is computed separately without grad as
    ``(N*V) * mean(w * p * (log p - log q))``. Subtracting element-wise first matters:
    the two entropy-sized terms (~log V) nearly cancel to a KL of order 0.1-1, so
    differencing two bf16 scalars would lose most of the value's precision.

    Returns ``(loss_for_backward, loss_value)``.
    """
    ctx = ttml.autograd.AutoContext.get_instance()
    _, _, n, v = logits.shape()
    scale = float(n * v)
    mode = ctx.get_gradient_mode()
    ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    try:
        log_p = ttml.ops.unary.log_softmax(teacher_logits, 3)
        wp = ttml.ops.binary.mul(ttml.ops.unary.exp(log_p), slot_weights)  # [1,1,N,V] * [1,1,N,1]
    finally:
        ctx.set_gradient_mode(mode)
    wp.set_requires_grad(False)
    log_q = ttml.ops.unary.log_softmax(logits, 3)
    loss = ttml.ops.binary.mul(ttml.ops.unary.mean(ttml.ops.binary.mul(log_q, wp)), -scale)
    ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    try:
        kl = ttml.ops.unary.mean(ttml.ops.binary.mul(wp, ttml.ops.binary.sub(log_p, log_q)))
        value = _scalar(kl) * scale
    finally:
        ctx.set_gradient_mode(mode)
    return loss, value


def _scalar(t: ttml.autograd.Tensor) -> float:
    return float(np.asarray(t.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


class DFlashTrainStep:
    """speculators trainer step: zero_grad, backward, clip_grad_norm_(max_norm), AdamW.

    Split into ``forward_backward`` and ``clip_and_step`` so a parity harness can read
    the pre-clip gradients in between; ``__call__`` runs both.

    Uses ``AdamWFullPrecision`` (fp32 master weights): with bf16 weights, a 1e-3 update
    on a norm weight near 1.0 is below half a bf16 ulp and would be lost.
    """

    def __init__(
        self,
        model: DFlashDrafter,
        lr: float,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.max_grad_norm = max_grad_norm
        cfg = ttml.optimizers.AdamWFullPrecisionConfig.make(lr, betas[0], betas[1], eps, weight_decay)
        self.optimizer = ttml.optimizers.AdamWFullPrecision(model.parameters(), cfg)

    def set_lr(self, lr: float) -> None:
        self.optimizer.set_lr(lr)

    def forward_backward(self, batch: DFlashDeviceBatch, capture: Capture = _no_capture) -> float:
        """Zero grads, run forward + backward. Returns the KL objective value."""
        self.optimizer.zero_grad()
        logits = self.model(batch, capture=capture)
        teacher = self.model.teacher_logits(batch)
        capture("teacher_logits", teacher)
        loss, value = dflash_kl_loss(logits, teacher, batch.slot_weights)
        loss.backward(False)
        return value

    def clip_and_step(self) -> float:
        """Clip to ``max_grad_norm`` (global L2, frozen params skipped), AdamW step, reset
        the graph. Returns the pre-clip gradient norm."""
        norm = ttml.core.clip_grad_norm(self.model.parameters(), self.max_grad_norm, 2.0, False)
        grad_norm = _scalar(norm)
        self.optimizer.step()
        ttml.autograd.AutoContext.get_instance().reset_graph()
        return grad_norm

    def __call__(self, batch: DFlashDeviceBatch, capture: Capture = _no_capture) -> tuple[float, float]:
        """One step; returns ``(loss, pre-clip grad norm)``."""
        loss = self.forward_backward(batch, capture=capture)
        return loss, self.clip_and_step()
