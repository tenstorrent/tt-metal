# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked gated delta rule, composed from ttml autograd ops.

Background
----------
tt-metal ships a fully fused ``ttnn.transformer.chunk_gated_delta_rule`` device
op, but it is inference-only: there is no delta-rule backward anywhere in the
repo.  Since 48 of Qwen3.8's 64 layers are Gated DeltaNet, training needs
gradients here, so this module rebuilds the same computation out of ttml ops
that already carry a backward (``matmul_op``, ``binary.add/sub/mul``,
``unary.exp``, ``reshape``).  The gradient therefore comes from the ordinary
autograd graph and no delta-rule backward has to be derived by hand.

The primitives the algorithm needs that ttml was missing -- ``cumsum``,
``l2_norm``, ``transpose``, ``sum_over_dim``, ``softplus`` and
``shift_along_dim`` -- were added as first-class ttml ops with C++ backwards
(``sources/ttml/ops/unary_ops.{hpp,cpp}``), so this module uses ttml ops
throughout.  Only slice/concat/split still come from the shared DeepSeek
wrappers, since ttnn's versions have no ttml backward.

The math follows the FLA reference kept in this repo at
``models/experimental/gated_attention_gated_deltanet/torch_functional/delta_rule_ops.py``
(``chunk_gated_delta_rule``), which is also what HuggingFace's
``torch_chunk_gated_delta_rule`` implements.

Numerical note
--------------
ttml ops fetch operands through ``get_value()``, which defaults to
``PreferredPrecision::HALF``, so this path computes in bfloat16 even though
Qwen3.8 pins the DeltaNet state to fp32 (``mamba_ssm_dtype: float32``).  The two
sensitive spots are ``v_new = v - v_prime`` (a difference of nearly-equal
vectors) and the decay differences that feed ``exp``.  See
``tests/python/test_qwen38_delta_rule.py`` for the measured PCC against the
torch reference.

The one deliberate departure: the WY transform
----------------------------------------------
The reference resolves intra-chunk dependencies with a sequential loop::

    for i in range(1, chunk_size):
        attn[..., i, :i] += (attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2)
    attn = attn + eye

That is ``chunk_size - 1`` sequential steps with an in-place slice assignment
per step -- 63 steps for a 64-wide chunk.  Expressed as an autograd graph it
would add thousands of nodes per layer, and the in-place write has no autograd
equivalent at all.

What the loop actually computes is the Neumann series of a strictly lower
triangular (hence nilpotent, ``A**C == 0``) matrix::

    I + A + A**2 + ... + A**(C-1) = (I - A)**-1

and for a nilpotent ``A`` that has a closed form by repeated squaring::

    (I - A)**-1 = prod_{j=0}^{log2(C)-1} (I + A**(2**j))

because expanding the product hits every power ``A**p`` for ``p`` in
``[0, 2**m)`` exactly once (binary expansion of ``p``).  For ``C = 64`` that is
5 squarings plus 5 products -- 10 matmuls instead of 63 sequential steps, all of
them plain batched matmuls that autograd already differentiates.
:func:`wy_inverse` implements this and the test suite pins it against the
reference loop.

Tensor layout
-------------
Everything is rank 4 because ttml tensors are 4D.  The head axis is folded into
the batch axis, so with ``BH = batch * num_v_heads``:

==================  ====================
tensor              shape
==================  ====================
``q``, ``k``        ``[BH, 1, T, K]``
``v``               ``[BH, 1, T, V]``
``beta``, ``g``     ``[BH, 1, T, 1]``
output              ``[BH, 1, T, V]``
chunked form        ``[BH, NC, C, *]``
recurrent state     ``[BH, 1, K, V]``
==================  ====================
"""

from __future__ import annotations

import math

import numpy as np

import ttnn
import ttml

from .autograd_ops import (
    autograd_concat,
    autograd_slice,
    autograd_split,
    causal_mask,
    identity,
)

__all__ = ["wy_inverse", "chunk_gated_delta_rule"]

# Local aliases for the stock ttml autograd ops this module is built from.
_mm = ttml.ops.matmul.matmul_op
_add = ttml.ops.binary.add
_sub = ttml.ops.binary.sub
_mul = ttml.ops.binary.mul
_exp = ttml.ops.unary.exp
_reshape = ttml.ops.reshape.reshape
_cumsum = ttml.ops.unary.cumsum
_l2_norm = ttml.ops.unary.l2_norm
_transpose = ttml.ops.unary.transpose


def _zeros(shape):
    """Zero tensor that participates in the graph as a constant."""
    return ttml.autograd.Tensor.from_numpy(
        np.zeros(shape, dtype=np.float32),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
    )


def wy_inverse(a, chunk_size: int, batch_shape):
    """``(I - a)**-1`` for strictly lower triangular ``a``, by repeated squaring.

    ``a`` is ``[*batch_shape, C, C]`` and strictly lower triangular, so it is
    nilpotent with ``a**C == 0`` and the Neumann series terminates.  See the
    module docstring for why this replaces the reference's sequential loop.

    Requires ``chunk_size`` to be a power of two, which every supported chunk
    size is (32, 64, 128).
    """
    exponent = int(math.log2(chunk_size))
    if 2**exponent != chunk_size:
        raise ValueError(f"chunk_size must be a power of two, got {chunk_size}")

    eye = identity(chunk_size, batch_shape, dtype=ttnn.DataType.BFLOAT16)

    # prod_{j} (I + a**(2**j)) == I + a + a**2 + ... + a**(C-1)
    product = _add(eye, a)
    power = a
    for _ in range(1, exponent):
        power = _mm(power, power)
        product = _mm(product, _add(eye, power))
    return product


def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    chunk_size: int = 64,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
):
    """Gated delta rule over a full sequence, chunk-parallel within / recurrent across.

    Args:
        q, k: ``[BH, 1, T, K]`` query / key (already GVA-expanded to ``BH``).
        v: ``[BH, 1, T, V]`` value.
        g: ``[BH, 1, T, 1]`` log-space decay gate (negative).
        beta: ``[BH, 1, T, 1]`` write strength.
        chunk_size: tokens per chunk; must divide ``T`` and be a power of two.
        scale: query scale, defaults to ``K ** -0.5``.
        use_qk_l2norm: L2-normalize q/k, as Qwen3.8 does.

    Returns:
        ``[BH, 1, T, V]`` output.  The final recurrent state is not returned:
        training never needs it, and keeping it would pin an extra tensor per
        layer in the graph.
    """
    bh, _, seq_len, key_dim = [int(d) for d in q.shape()]
    val_dim = int(v.shape()[3])

    if seq_len % chunk_size != 0:
        raise ValueError(f"sequence length {seq_len} must be a multiple of chunk_size {chunk_size}")
    num_chunks = seq_len // chunk_size

    if use_qk_l2norm:
        q = _l2_norm(q)
        k = _l2_norm(k)
    if scale is None:
        scale = key_dim**-0.5
    q = _mul(q, float(scale))

    # --- reshape into chunks -------------------------------------------------
    # [BH, 1, T, D] -> [BH, NC, C, D] is a contiguous reshape (T == NC * C).
    chunked = (bh, num_chunks, chunk_size)
    q_c = _reshape(q, list(chunked + (key_dim,)))
    k_c = _reshape(k, list(chunked + (key_dim,)))
    v_c = _reshape(v, list(chunked + (val_dim,)))
    beta_c = _reshape(beta, list(chunked + (1,)))
    g_c = _reshape(g, list(chunked + (1,)))

    v_beta = _mul(v_c, beta_c)
    k_beta = _mul(k_c, beta_c)

    # --- within-chunk decay --------------------------------------------------
    batch_all = (bh, num_chunks)
    bf16 = ttnn.DataType.BFLOAT16
    decay = _cumsum(g_c, dim=2)  # [BH, NC, C, 1]
    decay_exp = _exp(decay)

    keep_diag = causal_mask(chunk_size, batch_all, diagonal=1, dtype=bf16)  # j <= i
    keep_strict = causal_mask(chunk_size, batch_all, diagonal=0, dtype=bf16)  # j <  i

    # decay_mask[i, j] = exp(decay_i - decay_j) for j <= i, else 0.
    # Mask *before* the exp: for j > i the difference is positive and would
    # overflow, which is also why the reference does .tril().exp().tril().
    diff = _sub(decay, _transpose(decay, -1, -2))  # [BH, NC, C, C]
    decay_mask = _mul(_exp(_mul(diff, keep_diag)), keep_diag)

    # --- WY transform: resolve intra-chunk dependencies ---------------------
    # a = -(k_beta @ k^T * decay_mask), strictly lower triangular.
    a = _mm(k_beta, k_c, transpose_b=True)
    a = _mul(a, decay_mask)
    a = _mul(a, -1.0)
    a = _mul(a, keep_strict)

    t_mat = wy_inverse(a, chunk_size, batch_all)

    v_corrected = _mm(t_mat, v_beta)
    k_cumdecay = _mm(t_mat, _mul(k_beta, decay_exp))

    # --- recurrent scan across chunks ---------------------------------------
    # One split per tensor rather than a slice per chunk: the split's backward
    # is a single concat instead of NC zero-padded scatters.
    ones = [1] * num_chunks
    q_chunks = autograd_split(q_c, ones, 1)
    k_chunks = autograd_split(k_c, ones, 1)
    v_chunks = autograd_split(v_corrected, ones, 1)
    kcd_chunks = autograd_split(k_cumdecay, ones, 1)
    decay_chunks = autograd_split(decay, ones, 1)
    decay_exp_chunks = autograd_split(decay_exp, ones, 1)
    mask_chunks = autograd_split(decay_mask, ones, 1)

    keep_diag_1 = causal_mask(chunk_size, (bh, 1), diagonal=1, dtype=bf16)

    state = _zeros((bh, 1, key_dim, val_dim))
    outputs = []

    for i in range(num_chunks):
        q_i = q_chunks[i]
        k_i = k_chunks[i]
        v_i = v_chunks[i]
        decay_i = decay_chunks[i]

        # Within-chunk attention, causal including the diagonal.
        intra = _mul(_mm(q_i, k_i, transpose_b=True), mask_chunks[i])
        intra = _mul(intra, keep_diag_1)

        # Remove what the carried state already accounts for, then read it.
        v_new = _sub(v_i, _mm(kcd_chunks[i], state))
        o_inter = _mm(_mul(q_i, decay_exp_chunks[i]), state)
        outputs.append(_add(o_inter, _mm(intra, v_new)))

        # Carry the state to the next chunk: decay it by the whole chunk's gate,
        # then add this chunk's contribution weighted by the remaining decay.
        last = autograd_slice(
            decay_i,
            [0, 0, chunk_size - 1, 0],
            [bh, 1, chunk_size, 1],
        )  # [BH, 1, 1, 1]
        decayed = _mul(state, _exp(last))
        k_weighted = _mul(k_i, _exp(_sub(last, decay_i)))
        state = _add(decayed, _mm(k_weighted, v_new, transpose_a=True))

    out = autograd_concat(outputs, 1)  # [BH, NC, C, V]
    return _reshape(out, [bh, 1, seq_len, val_dim])
