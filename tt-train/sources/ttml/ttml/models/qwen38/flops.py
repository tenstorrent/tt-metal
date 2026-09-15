# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""FLOPs accounting for the Qwen3.8 hybrid stack, measured rather than derived.

The usual closed-form estimate (``6 * N + 12 * L * H * Q * T`` per token) assumes
every layer is quadratic attention.  That is wrong for Qwen3.8: 48 of its 64
layers are Gated DeltaNet, which is linear in sequence length and has a
completely different projection layout (a fused 10240-wide QKV, a depthwise
conv, and per-head scalar gates).  Hand-deriving a formula for that is
error-prone, so instead this module *measures* the reference implementation with
``torch.utils.flop_counter.FlopCounterMode``.

Two things make measuring cheap enough to do at startup:

* **meta tensors.** The reference runs on ``device="meta"``, so no memory is
  allocated and no arithmetic is performed -- ``FlopCounterMode`` counts from
  shapes alone.  The real 27B dimensions can be used directly.
* **linearity in depth.** FLOPs are exactly linear in layer count, so one
  DeltaNet layer, one attention layer and one MLP are measured once each and
  multiplied by their layer counts.

Forward *and* backward are counted, rather than assuming the usual ``3x``
forward factor.  This matters for LoRA: a frozen linear needs only the input
gradient (``dx = dy @ W^T``), not the weight gradient (``dW = x^T @ dy``), so it
costs ``2x`` forward instead of ``3x``.  Setting ``requires_grad`` to mirror the
freezing pattern makes the counter reflect that on its own.

What is counted
---------------
``FlopCounterMode`` counts matmul, conv and SDPA; it does not count elementwise
work.  That is the standard "model FLOPs" convention (the MFU denominator is
matmul-bound peak throughput), and it is what makes the number comparable to
other published MFU figures.

Two consequences worth being explicit about, since both mean reported MFU is
*conservative* -- real utilization of the hardware is higher than the number:

* The reference resolves the delta rule's intra-chunk dependencies with a
  sequential loop of elementwise multiply-and-sum, which contributes no counted
  FLOPs.  :func:`ttml.models.qwen38.delta_rule.wy_inverse` replaces it with
  ``2 * log2(chunk)`` real matmuls, which are executed but not counted here.
* The DeltaNet's elementwise gating and decay work is substantial in op count
  and also uncounted.

So this is MFU (useful architectural work / peak), not HFU (work actually
issued / peak).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Qwen38Flops", "qwen38_flops", "flops_per_token"]


@dataclass(frozen=True)
class Qwen38Flops:
    """FLOPs for one optimizer step, broken down by component."""

    delta_net: int
    attention: int
    mlp: int
    embedding_and_head: int

    @property
    def total(self) -> int:
        return self.delta_net + self.attention + self.mlp + self.embedding_and_head

    def per_token(self, batch: int, seq_len: int) -> float:
        return self.total / (batch * seq_len)


def _count(fn, *, backward: bool) -> int:
    """Run ``fn`` under FlopCounterMode on meta tensors and return total FLOPs.

    ``fn`` returns the tensor to backprop from. Backward is invoked inside the
    counter so its matmuls are included.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        out = fn()
        if backward:
            # A scalar reduction is elementwise, so it adds nothing to the count.
            out.sum().backward()
    return counter.get_total_flops()


def _meta(*shape, requires_grad: bool = False):
    import torch

    return torch.randn(*shape, device="meta", requires_grad=requires_grad)


def _delta_net_layer_flops(config, batch: int, seq_len: int, *, train_base: bool) -> int:
    """One Gated DeltaNet layer, in the checkpoint's fused layout."""
    import torch
    import torch.nn.functional as F

    from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
        chunk_gated_delta_rule,
    )

    hidden = config.hidden_size
    n_k, n_v = config.linear_num_key_heads, config.linear_num_value_heads
    d_k, d_v = config.linear_key_head_dim, config.linear_value_head_dim
    kernel = config.linear_conv_kernel_dim
    qkv_dim, z_dim = config.qkv_proj_dim, config.value_proj_dim
    key_dim = config.key_proj_dim

    # The base weights are frozen under LoRA; only the activation path carries
    # a gradient, so the counter sees dgrad without wgrad.
    rg = train_base
    x = _meta(batch, seq_len, hidden, requires_grad=True)
    w_qkv = _meta(qkv_dim, hidden, requires_grad=rg)
    w_z = _meta(z_dim, hidden, requires_grad=rg)
    w_a = _meta(n_v, hidden, requires_grad=rg)
    w_b = _meta(n_v, hidden, requires_grad=rg)
    w_out = _meta(hidden, z_dim, requires_grad=rg)
    w_conv = _meta(qkv_dim, 1, kernel, requires_grad=rg)

    def run():
        qkv = F.linear(x, w_qkv)
        # The depthwise causal conv is deliberately expressed as shifts and
        # multiplies rather than F.conv1d, for two reasons. It is what the ttml
        # implementation actually does (see gated_deltanet.causal_conv1d_silu),
        # and FlopCounterMode ignores `groups` when counting conv *backward*,
        # billing this depthwise conv's weight gradient as a dense
        # C-to-C convolution -- a ~4x overcount of the whole layer. Its true
        # cost is added analytically by _conv_flops below.
        shifted = [F.pad(qkv, (0, 0, kernel - 1 - j, 0))[:, : qkv.shape[1]] for j in range(kernel)]
        qkv = F.silu(sum(s * w_conv[:, 0, j] for j, s in enumerate(shifted)))

        q = qkv[..., :key_dim].reshape(batch, seq_len, n_k, d_k)
        k = qkv[..., key_dim : 2 * key_dim].reshape(batch, seq_len, n_k, d_k)
        v = qkv[..., 2 * key_dim :].reshape(batch, seq_len, n_v, d_v)
        repeats = n_v // n_k
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)

        beta = F.linear(x, w_b).sigmoid()
        g = -F.softplus(F.linear(x, w_a))

        o, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, chunk_size=config.delta_chunk_size, use_qk_l2norm=True
        )
        gate = F.linear(x, w_z).reshape(batch, seq_len, n_v, d_v)
        o = o * F.silu(gate)
        return F.linear(o.reshape(batch, seq_len, z_dim), w_out)

    return _count(run, backward=True) + _conv_flops(batch, seq_len, qkv_dim, kernel, train_base=train_base)


def _conv_flops(batch: int, seq_len: int, channels: int, kernel: int, *, train_base: bool) -> int:
    """Depthwise causal conv FLOPs: one multiply-add per (token, channel, tap).

    Forward plus input gradient is 2x that; a trainable weight adds a third
    pass. This is ~0.03% of a DeltaNet layer, but it is counted explicitly so
    the figure does not depend on FlopCounterMode's grouped-conv handling.
    """
    passes = 3 if train_base else 2
    return passes * 2 * batch * seq_len * channels * kernel


def _attention_layer_flops(config, batch: int, seq_len: int, *, train_base: bool) -> int:
    """One Gated Attention layer (2x-wide q_proj, GQA, partial RoPE)."""
    import torch
    import torch.nn.functional as F

    hidden = config.hidden_size
    n_h, n_kv, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q_width = n_h * d * (2 if config.attn_output_gate else 1)

    rg = train_base
    x = _meta(batch, seq_len, hidden, requires_grad=True)
    w_q = _meta(q_width, hidden, requires_grad=rg)
    w_k = _meta(n_kv * d, hidden, requires_grad=rg)
    w_v = _meta(n_kv * d, hidden, requires_grad=rg)
    w_o = _meta(hidden, n_h * d, requires_grad=rg)

    def run():
        qg = F.linear(x, w_q).view(batch, seq_len, n_h, -1)
        if config.attn_output_gate:
            q, gate = qg[..., :d], qg[..., d:]
        else:
            q, gate = qg, None
        q = q.transpose(1, 2)
        k = F.linear(x, w_k).view(batch, seq_len, n_kv, d).transpose(1, 2)
        v = F.linear(x, w_v).view(batch, seq_len, n_kv, d).transpose(1, 2)

        # GQA: SDPA sees all query heads, with KV broadcast across each group.
        groups = n_h // n_kv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        o = o.transpose(1, 2).reshape(batch, seq_len, n_h * d)
        if gate is not None:
            o = o * gate.reshape(batch, seq_len, n_h * d).sigmoid()
        return F.linear(o, w_o)

    return _count(run, backward=True)


def _mlp_layer_flops(config, batch: int, seq_len: int, *, train_base: bool) -> int:
    """One SwiGLU MLP."""
    import torch
    import torch.nn.functional as F

    hidden, inter = config.hidden_size, config.intermediate_size
    rg = train_base
    x = _meta(batch, seq_len, hidden, requires_grad=True)
    w_gate = _meta(inter, hidden, requires_grad=rg)
    w_up = _meta(inter, hidden, requires_grad=rg)
    w_down = _meta(hidden, inter, requires_grad=rg)

    def run():
        return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)

    return _count(run, backward=True)


def _head_flops(config, batch: int, seq_len: int, *, train_base: bool) -> int:
    """The LM head. The embedding lookup is a gather, so it contributes no FLOPs."""
    import torch
    import torch.nn.functional as F

    x = _meta(batch, seq_len, config.hidden_size, requires_grad=True)
    w = _meta(config.vocab_size, config.hidden_size, requires_grad=train_base)

    def run():
        return F.linear(x, w)

    return _count(run, backward=True)


# Measuring walks the reference on meta tensors; cache so repeated logging is free.
_MEASURED: dict = {}


def _measure(config, batch: int, seq_len: int, train_base: bool) -> Qwen38Flops:
    num_layers = config.num_hidden_layers
    num_full = sum(1 for i in range(num_layers) if config.is_full_attention(i))
    num_linear = num_layers - num_full

    return Qwen38Flops(
        delta_net=num_linear * _delta_net_layer_flops(config, batch, seq_len, train_base=train_base),
        attention=num_full * _attention_layer_flops(config, batch, seq_len, train_base=train_base),
        mlp=num_layers * _mlp_layer_flops(config, batch, seq_len, train_base=train_base),
        embedding_and_head=_head_flops(config, batch, seq_len, train_base=train_base),
    )


def qwen38_flops(config, batch: int, seq_len: int, *, train_base: bool = False) -> Qwen38Flops:
    """FLOPs for one forward+backward step of the full 64-layer stack.

    Args:
        config: a :class:`~ttml.models.qwen38.Qwen38Config`.
        batch: per-step batch size (local, not multiplied by data parallelism).
        seq_len: sequence length.
        train_base: ``True`` for full fine-tuning (base weights get gradients),
            ``False`` for LoRA, where the frozen base needs only the input
            gradient and therefore costs ~2x forward instead of ~3x.

    Returns:
        A :class:`Qwen38Flops` breakdown; ``.total`` is the per-step figure.
    """
    # Qwen38Config is a mutable dataclass holding a list, so it cannot be a cache
    # key; key on the fields that actually affect the count instead.
    key = (
        batch,
        seq_len,
        train_base,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.vocab_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.attn_output_gate,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_conv_kernel_dim,
        config.delta_chunk_size,
        tuple(config.layer_types),
    )
    cached = _MEASURED.get(key)
    if cached is None:
        cached = _measure(config, batch, seq_len, train_base)
        _MEASURED[key] = cached
    return cached


def flops_per_token(config, batch: int, seq_len: int, *, train_base: bool = False) -> float:
    """Per-token FLOPs, the form the throughput callback wants for MFU."""
    return qwen38_flops(config, batch, seq_len, train_base=train_base).per_token(batch, seq_len)
