# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Residual add fused with RMSNorm for the training path (iteration 9).

A pre-norm block ends with ``s = residual + delta`` and the next sublayer starts with ``rmsnorm(s)``; the composite
path runs the add (one memory-bound op), the norm forward on one core per tile-row, and in the backward the norm
backward plus an add that accumulates the two gradients reaching ``s``. :class:`AddRMSNormFrozenGamma` runs
``metal/ops/add_rmsnorm_fw_allcore`` (sum, rms and the normalized output in two all-core phases) and the all-core
backward with the residual gradient folded in (``rmsnorm_bw_allcore(..., residual_grad=)``): no standalone adds,
no 32-core norm forward. Gamma is frozen (LoRA), so it gets no gradient and the frozen-gamma kernels apply.

:func:`fused_forward` is the decoder forward with the layer boundaries re-paired: the add that closes a sublayer is
computed together with the norm that opens the next one (the final norm closes the last block). The first norm
reads the frozen embedding output and stays the plain op (no gradient upstream). :func:`use_fused_residual_norm`
switches a :class:`ttml.models.mistral.Mistral` (possibly inside a ``LoraModel``) to that forward.
"""

from __future__ import annotations

import ttml
import ttnn
from ttml.autograd import Function
from ttml.models.llama.transformer import RMSNormLayer

__all__ = ["AddRMSNormFrozenGamma", "add_rmsnorm", "fused_forward", "use_fused_residual_norm"]


class AddRMSNormFrozenGamma(Function):
    """``(residual, delta, gamma, epsilon, return_sum) -> (sum, out)`` or ``out`` alone.

    ``sum = residual + delta`` is the next residual stream; ``out = rmsnorm(sum) * gamma``. The backward returns the
    same ``dsum = rmsnorm_bw(sum, grad_out) + grad_sum`` for both ``residual`` and ``delta`` (the add's gradient is
    the identity on both), computed in the backward kernel. With ``return_sum=False`` (the final norm) the sum is
    not an output and ``grad_sum`` is zero."""

    @staticmethod
    def forward(ctx, residual, delta, gamma, epsilon, return_sum):
        r, d, g = residual.get_value(), delta.get_value(), gamma.get_value()
        # Iteration 37: a delta that is the [C, T] transpose of the residual (the fused w2's transposed-orientation
        # output) is read transposed by the kernel; its gradient comes back transposed too.
        rs, ds = tuple(int(v) for v in r.shape), tuple(int(v) for v in d.shape)
        transposed = ds != rs and ds[:2] == rs[:2] and ds[2] == rs[3] and ds[3] == rs[2]
        s, out, rms = ttml.ops.metal.add_rmsnorm_fw_allcore(r, d, g, epsilon, delta_transposed=transposed)
        ctx.save_for_backward(s, g, rms)
        ctx.return_sum = return_sum
        ctx.delta_transposed = transposed
        return (s, out) if return_sum else out

    @staticmethod
    def backward(ctx, *grads):
        s, g, rms = ctx.saved_tensors
        ctx.save_for_backward()
        if ctx.return_sum:
            grad_sum, grad_out = grads
        else:
            (grad_out,) = grads
            grad_sum = None
        res = ttml.ops.metal.rmsnorm_bw_allcore(
            s, g, rms, grad_out, False, grad_sum, transposed_grad=ctx.delta_transposed
        )
        dsum = res[0]
        return dsum, (res[2] if ctx.delta_transposed else dsum), None


def add_rmsnorm(residual, delta, norm: RMSNormLayer, return_sum: bool = True):
    """``norm(residual + delta)`` through the fused kernels (frozen gamma); returns ``(sum, out)`` or ``out``."""
    assert not norm.gamma.tensor.get_requires_grad(), "the fused residual norm needs a frozen gamma"
    return AddRMSNormFrozenGamma.apply(residual, delta, norm.gamma.tensor, norm.epsilon, return_sum)


def fused_forward(model, input, mask):
    """The ``Llama.forward`` training path (no KV cache, default runner) with every residual add fused into the
    norm that follows it. ``model`` is the ``Llama`` / ``Mistral`` instance."""
    TILE_SIZE = 32
    actual_seq_len = input.shape()[-1]
    padded_seq_len = ((actual_seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    input_padded = input
    if padded_seq_len != actual_seq_len:
        padding = [(0, 0), (0, 0), (0, 0), (0, padded_seq_len - actual_seq_len)]
        input_padded = ttml.autograd.create_tensor(ttnn.pad(input.get_value(), padding=padding, value=0.0))
    out = model._embed(input_padded)
    if padded_seq_len != actual_seq_len:
        shape = out.shape()
        out = ttml.autograd.create_tensor(
            ttnn.slice(out.get_value(), [0, 0, 0, 0], [shape[0], shape[1], actual_seq_len, shape[3]], [1, 1, 1, 1])
        )

    blocks = list(model.blocks)
    residual = out
    h = blocks[0].attention_norm(residual)  # no gradient upstream (frozen embedding): the plain op
    for i, block in enumerate(blocks):
        delta = block.attention(h, mask)
        residual, h = add_rmsnorm(residual, delta, block.mlp_norm)
        delta = block.mlp(h)
        if i + 1 < len(blocks):
            residual, h = add_rmsnorm(residual, delta, blocks[i + 1].attention_norm)
        else:
            h = add_rmsnorm(residual, delta, model.ln_fc, return_sum=False)
    logits = model.fc(h)
    if not model.config.use_tp and model.padded_vocab_size != model.config.vocab_size:
        from ttml.models.llama.autograd_ops import SliceLastDim

        logits = SliceLastDim.apply(logits, model.config.vocab_size)
    return logits


def use_fused_residual_norm(model) -> int:
    """Switch the :class:`ttml.models.mistral.Mistral` inside ``model`` (or ``model`` itself) to
    :func:`fused_forward`. Every RMSNorm gamma must be frozen. Returns the number of fused norms."""
    from ttml.models.mistral import Mistral

    targets = [m for _, m in model.named_modules() if isinstance(m, Mistral)]
    if isinstance(model, Mistral):
        targets.append(model)
    assert len(targets) == 1, f"expected one Mistral model, found {len(targets)}"
    llama = targets[0]
    norms = [m for _, m in llama.named_modules() if isinstance(m, RMSNormLayer)]
    frozen = [m for m in norms if not m.gamma.tensor.get_requires_grad()]
    assert len(frozen) == len(norms), "the fused residual norm needs every RMSNorm gamma frozen (LoRA)"
    llama.fused_residual_norm = True
    # Iteration 37: fused linears with a transposed-orientation forward copy hand their output transposed to the
    # residual norm (no transpose op); only valid under this forward, whose norms read it transposed.
    for _, m in llama.named_modules():
        if getattr(m, "weight_cat_fwd", None) is not None:
            m.transposed_output = True
    return len(norms) - 1  # the first norm stays the plain op
