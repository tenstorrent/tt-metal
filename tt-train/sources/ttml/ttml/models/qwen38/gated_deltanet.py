# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet, the ``linear_attention`` layer of Qwen3.8 (48 of 64 layers).

Structure, following the checkpoint's parameter names rather than the FLA
reference's (Qwen3.8 fuses what FLA keeps separate)::

    x -> in_proj_qkv -----> conv1d(K=4, depthwise, causal) -> silu -> split Q|K|V
      -> in_proj_b -> sigmoid ------------------------------------> beta
      -> in_proj_a -> g = -exp(A_log) * softplus(a + dt_bias) ----> g
      -> in_proj_z ----------------------------------------------> gate

    delta_rule(Q, K, V, g, beta) -> rmsnorm(.) * silu(gate) -> out_proj

Two things differ from ``torch_functional/gated_deltanet.py``: the checkpoint
has a single fused ``in_proj_qkv`` (10240 = 2*2048 + 6144) with one fused
``conv1d`` over all 10240 channels instead of three separate projections and
convs, and the gate comes from ``in_proj_z`` rather than ``g_proj``.

Tensor layout
-------------
ttml tensors are rank 4. Hidden states are ``[B, 1, T, hidden]``; the delta rule
wants one sequence per head, so the head axis is folded into the batch axis to
give ``[B * H, 1, T, head_dim]``. :func:`fold_heads` / :func:`unfold_heads` do
that round trip.
"""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter

from .autograd_ops import autograd_concat, autograd_slice
from .delta_rule import chunk_gated_delta_rule
from .parallel import make_column_linear, make_row_linear, make_sharded_parameter, tp_size

__all__ = ["Qwen38GatedDeltaNet", "fold_heads", "unfold_heads", "causal_conv1d_silu"]

_add = ttml.ops.binary.add
_mul = ttml.ops.binary.mul
_reshape = ttml.ops.reshape.reshape
_silu = ttml.ops.unary.silu
_sigmoid = ttml.ops.unary.sigmoid
_softplus = ttml.ops.unary.softplus
_exp = ttml.ops.unary.exp
_transpose = ttml.ops.unary.transpose
_shift = ttml.ops.unary.shift_along_dim
_rmsnorm = ttml.ops.rmsnorm.rmsnorm


def fold_heads(x, num_heads: int, head_dim: int):
    """``[B, 1, T, H * D]`` -> ``[B * H, 1, T, D]``.

    Both reshapes are contiguous reinterpretations; only the transpose moves
    data. Giving each head its own batch row is what lets the delta rule treat
    the head axis as independent sequences.
    """
    batch, _, seq, _ = [int(d) for d in x.shape()]
    x = _reshape(x, [batch, seq, num_heads, head_dim])
    x = _transpose(x, 1, 2)  # [B, H, T, D]
    return _reshape(x, [batch * num_heads, 1, seq, head_dim])


def unfold_heads(x, batch: int, num_heads: int):
    """``[B * H, 1, T, D]`` -> ``[B, 1, T, H * D]`` (inverse of :func:`fold_heads`)."""
    _, _, seq, head_dim = [int(d) for d in x.shape()]
    x = _reshape(x, [batch, num_heads, seq, head_dim])
    x = _transpose(x, 1, 2)  # [B, T, H, D]
    return _reshape(x, [batch, 1, seq, num_heads * head_dim])


def repeat_interleave_heads(x, num_heads: int, head_dim: int, repeats: int):
    """Repeat each head ``repeats`` times in place: ``[..., H, D] -> [..., H * R, D]``.

    ttnn has no ``repeat_interleave``, and ``ttnn.repeat`` tiles (``h0..hN`` R
    times) rather than interleaving (``h0 R times, h1 R times, ...``).  The
    interleaved order is what GVA needs, since value head ``i`` pairs with key
    head ``i // R``.

    Making the head axis explicit first turns it into a concat plus a reshape,
    both of which autograd already handles: with the heads on their own axis,
    concatenating R copies on the feature axis gives each head's row as ``D``
    repeated R times, and reinterpreting that row as R rows of ``D`` leaves each
    head repeated R times in a row.

    Concatenating the ``[B, 1, T, H*D]`` tensor directly would instead tile
    whole head groups (``h0..hN`` R times), which is the wrong pairing.
    """
    if repeats == 1:
        return x
    batch, _, seq, _ = [int(d) for d in x.shape()]
    per_head = _reshape(x, [batch, seq, num_heads, head_dim])
    wide = autograd_concat([per_head] * repeats, 3)  # [B, T, H, R * D]
    # [B, T, H, R*D] -> [B, 1, T, H*R*D]: row-major (h, r, d) == (h * R + r, d).
    return _reshape(wide, [batch, 1, seq, num_heads * repeats * head_dim])


def causal_conv1d_silu(x, weight_taps, kernel_size: int):
    """Depthwise causal conv1d over the feature axis, then SiLU.

    A depthwise causal conv is a fixed set of shifts scaled by per-channel
    weights::

        out[t, c] = sum_j w[c, j] * x[t - (K - 1 - j), c]

    so it needs no conv op: ``shift_along_dim`` supplies each tap and the
    per-channel weight is an ordinary broadcast multiply.  Gradients w.r.t. both
    the input and the conv weights come from the autograd graph.

    Args:
        x: ``[B, 1, T, C]`` input.
        weight_taps: ``K`` tensors of shape ``[1, 1, 1, C]``, tap ``j`` being
            ``weight[:, 0, j]``.
        kernel_size: ``K``.
    """
    out = None
    for j in range(kernel_size):
        # Tap j reads x[t - (K - 1 - j)], so the newest tap (j = K - 1) is unshifted.
        term = _mul(_shift(x, 2, kernel_size - 1 - j), weight_taps[j])
        out = term if out is None else _add(out, term)
    return _silu(out)


class Qwen38GatedDeltaNet(AbstractModuleBase):
    """The ``linear_attn`` submodule of a Qwen3.8 linear-attention layer."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Under TP each chip owns a slice of the heads. The GVA ratio survives
        # sharding (48/16 == 12/4 == 3), so the delta rule stays purely local.
        tp = tp_size(config)
        self.tp = tp
        self.num_k_heads = config.linear_num_key_heads // tp
        self.num_v_heads = config.linear_num_value_heads // tp
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.gva_repeats = config.gva_repeats
        self.conv_kernel = config.linear_conv_kernel_dim
        self.chunk_size = config.delta_chunk_size
        self.eps = config.rms_norm_eps

        # Local widths: what this chip's projections actually emit.
        key_dim = config.key_proj_dim // tp
        value_dim = config.value_proj_dim // tp
        hidden = config.hidden_size

        init = ttml.init.normal(0.0, 0.02)

        # Fused Q|K|V projection, matching the checkpoint's in_proj_qkv. Under
        # TP the loader permutes its rows so this chip's contiguous shard is a
        # self-consistent [q heads | k heads | v heads] group; see
        # models.qwen38.parallel.qkv_shard_permutation.
        self.in_proj_qkv = make_column_linear(config, hidden, config.qkv_proj_dim, init)
        # Output gate (the checkpoint calls it z).
        self.in_proj_z = make_column_linear(config, hidden, config.value_proj_dim, init)
        # Per-value-head scalars: write strength (beta) and decay input (a).
        self.in_proj_b = make_column_linear(config, hidden, config.linear_num_value_heads, init)
        self.in_proj_a = make_column_linear(config, hidden, config.linear_num_value_heads, init)
        # Row-parallel: consumes the sharded value heads and all-reduces.
        self.out_proj = make_row_linear(config, config.value_proj_dim, hidden, init)

        # Fused depthwise conv over all 10240 QKV channels. The checkpoint ships
        # one [C, 1, K] tensor; it is held here as K separate [1, 1, 1, C] taps
        # so each is a plain broadcast multiplicand needing no slice per step.
        #
        # Each tap is assigned to its own attribute because registration happens
        # in AbstractModuleBase.__setattr__: parameters kept only in a list are
        # never registered, so they would silently not train, load or save.
        # Depthwise, so each tap is per-channel and follows the same shard as
        # the fused projection it multiplies.
        self.conv_taps = []
        for tap_idx in range(self.conv_kernel):
            tap = make_sharded_parameter(config, init, (1, 1, 1, config.qkv_proj_dim))
            setattr(self, f"conv_tap_{tap_idx}", tap)
            self.conv_taps.append(tap)

        # Decay parameters, one per value head, laid out for broadcasting over
        # [B, 1, T, num_v_heads].
        zeros = ttml.init.zeros()
        n_v_global = config.linear_num_value_heads
        self.A_log = make_sharded_parameter(config, zeros, (1, 1, 1, n_v_global))
        self.dt_bias = make_sharded_parameter(config, zeros, (1, 1, 1, n_v_global))
        # Output RMSNorm, over head_v_dim.
        self.norm_weight = Parameter(ttml.init.ones()((1, 1, 1, self.head_v_dim)))

        self._key_dim = key_dim
        self._value_dim = value_dim

    def forward(self, hidden_states):
        batch, _, seq, _ = [int(d) for d in hidden_states.shape()]
        key_dim, value_dim = self._key_dim, self._value_dim

        # --- projections + fused causal conv -------------------------------
        qkv = self.in_proj_qkv(hidden_states)
        qkv = causal_conv1d_silu(qkv, [p.tensor for p in self.conv_taps], self.conv_kernel)

        q = autograd_slice(qkv, [0, 0, 0, 0], [batch, 1, seq, key_dim])
        k = autograd_slice(qkv, [0, 0, 0, key_dim], [batch, 1, seq, 2 * key_dim])
        v = autograd_slice(qkv, [0, 0, 0, 2 * key_dim], [batch, 1, seq, 2 * key_dim + value_dim])

        # GVA: each key head serves `gva_repeats` value heads (3 for Qwen3.8), so
        # every key head is repeated that many times. Head counts here are the
        # per-chip ones, and the ratio is the same locally as globally.
        q = repeat_interleave_heads(q, self.num_k_heads, self.head_k_dim, self.gva_repeats)
        k = repeat_interleave_heads(k, self.num_k_heads, self.head_k_dim, self.gva_repeats)

        # --- gates ---------------------------------------------------------
        beta = _sigmoid(self.in_proj_b(hidden_states))  # [B, 1, T, H_v]
        # g = -exp(A_log) * softplus(a + dt_bias), the log-space decay per step.
        a = _add(self.in_proj_a(hidden_states), self.dt_bias.tensor)
        g = _mul(_softplus(a), _mul(_exp(self.A_log.tensor), -1.0))

        # --- delta rule, one sequence per value head -----------------------
        bh = batch * self.num_v_heads
        q = fold_heads(q, self.num_v_heads, self.head_k_dim)
        k = fold_heads(k, self.num_v_heads, self.head_k_dim)
        v = fold_heads(v, self.num_v_heads, self.head_v_dim)
        # beta/g are per-head scalars: [B, 1, T, H_v] folds to [B*H_v, 1, T, 1].
        beta = fold_heads(beta, self.num_v_heads, 1)
        g = fold_heads(g, self.num_v_heads, 1)

        out = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=self.chunk_size,
            use_qk_l2norm=True,
        )

        # --- gated output norm + projection --------------------------------
        # RMSNorm is over head_v_dim, so normalize while still head-folded.
        gate = fold_heads(self.in_proj_z(hidden_states), self.num_v_heads, self.head_v_dim)
        out = _mul(_rmsnorm(out, self.norm_weight.tensor, self.eps), _silu(gate))

        out = unfold_heads(out, batch, self.num_v_heads)
        return self.out_proj(out)
