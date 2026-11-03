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
    def forward(ctx, residual, delta, gamma, epsilon, return_sum, out_tail_cols=0):
        r, d, g = residual.get_value(), delta.get_value(), gamma.get_value()
        # Iteration 37: a delta that is the [C, T] transpose of the residual (the fused w2's transposed-orientation
        # output) is read transposed by the kernel; its gradient comes back transposed too.
        rs, ds = tuple(int(v) for v in r.shape), tuple(int(v) for v in d.shape)
        transposed = ds != rs and ds[:2] == rs[:2] and ds[2] == rs[3] and ds[3] == rs[2]
        # Iteration 49: ``out_tail_cols`` (the LoRA tail width) makes out wide, [.., C + tail] with the tail zeroed, so
        # the consumer's fused LoRA layer writes its h into the tail (the wide path: no K-concat, no pad); the
        # gradient comes back wide and the backward kernel reads it at its own stride.
        s, out, rms = ttml.ops.metal.add_rmsnorm_fw_allcore(
            r, d, g, epsilon, delta_transposed=transposed, out_tail_cols=out_tail_cols
        )
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


def add_rmsnorm(residual, delta, norm: RMSNormLayer, return_sum: bool = True, out_tail_cols: int = 0):
    """``norm(residual + delta)`` through the fused kernels (frozen gamma); returns ``(sum, out)`` or ``out``.
    ``out_tail_cols``: a wide output for a fused LoRA consumer (iteration 49)."""
    assert not norm.gamma.tensor.get_requires_grad(), "the fused residual norm needs a frozen gamma"
    return AddRMSNormFrozenGamma.apply(residual, delta, norm.gamma.tensor, norm.epsilon, return_sum, out_tail_cols)


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
    # Sequence parallel (iteration 42): the residual stream lives row-sharded over the TP group. Every TP all-reduce
    # is a reduce-scatter followed by an all-gather; the norm (and the residual add fused into it) runs between the
    # two on the chip's quarter of the rows, the all-gather hands the normalized rows to the column-parallel layers
    # (its backward reduce-scatters their partial input gradients: no broadcast node, no dx all-reduce), the
    # row-parallel layers return their reduce-scattered shard. Same bytes on the links, a quarter of the norm work.
    sp_axis = getattr(model, "sequence_parallel_axis", None)
    sp = sp_axis is not None
    overlap = getattr(model, "collective_overlap", None)  # iteration 47
    # Iteration 49: the fused norms feeding a fused LoRA layer (qkv, gate_up) write a wide output whose tail takes the
    # layer's LoRA h (no K-concat, no pad); the first norm (the plain op) and the final norm (the LM head) stay plain.
    tail = getattr(model, "norm_out_tail_cols", 0)

    def gather(t, replicated_grad=False):
        # SHARDED: the consumer is a LoRA column-parallel layer without its broadcast node, so the gather's backward
        # reduce-scatters the partial input gradients; REPLICATED: the consumer (the LM head's base column-parallel
        # linear) still broadcasts and all-reduces its dx itself, the gather's backward only takes the local rows.
        if not sp:
            return t
        kind = (
            ttml.ops.distributed.GradOutputType.REPLICATED
            if replicated_grad
            else ttml.ops.distributed.GradOutputType.SHARDED
        )
        return ttml.ops.distributed.all_gather(t, 2, sp_axis, kind)

    if sp:
        residual = ttml.ops.distributed.scatter(residual, 2, sp_axis)  # the replicated embedding rows -> the shard
    # Iteration 48: with the collective overlap the fused qkv takes the shard and gathers it itself (as gate_up does).
    qkv_overlapped = (
        overlap is not None and getattr(blocks[0].attention.qkv_linear, "collective_overlap", None) is not None
    )
    h = blocks[0].attention_norm(residual)  # no gradient upstream (frozen embedding): the plain op
    h = h if qkv_overlapped else gather(h)
    for i, block in enumerate(blocks):
        delta = block.attention(h, mask)
        residual, h = add_rmsnorm(residual, delta, block.mlp_norm, out_tail_cols=tail)
        # Iteration 47: with the collective overlap the fused gate_up takes the shard, gathers it itself and
        # reduce-scatters its input gradient inside its backward, overlapped with its adapter products.
        delta = block.mlp(h if overlap is not None else gather(h))
        if i + 1 < len(blocks):
            residual, h = add_rmsnorm(residual, delta, blocks[i + 1].attention_norm, out_tail_cols=tail)
            h = h if qkv_overlapped else gather(h)
        else:
            h = gather(add_rmsnorm(residual, delta, model.ln_fc, return_sum=False), replicated_grad=True)
    logits = model.fc(h)
    if not model.config.use_tp and model.padded_vocab_size != model.config.vocab_size:
        from ttml.models.llama.autograd_ops import SliceLastDim

        logits = SliceLastDim.apply(logits, model.config.vocab_size)
    return logits


def use_fused_residual_norm(
    model,
    sequence_parallel: bool = True,
    collective_overlap: bool = False,
    overlap_serial: bool = False,
    overlap_serial_async: bool = False,
) -> int:
    """Switch the :class:`ttml.models.mistral.Mistral` inside ``model`` (or ``model`` itself) to
    :func:`fused_forward`. Every RMSNorm gamma must be frozen. Returns the number of fused norms.

    ``sequence_parallel`` (iteration 42, TP only): the residual stream row-sharded over the TP group, the norms on the
    shard between the reduce-scatter and the all-gather halves of each all-reduce (see :func:`fused_forward`); the
    row-parallel LoRA layers return their reduce-scattered shard and the column-parallel ones drop the broadcast
    node (the all-gather's backward reduces their partial input gradients).

    ``collective_overlap`` (iteration 47, TP + sequence parallel; off by default since iteration 48): the fused
    column-parallel layers run the reduce-scatter of their input gradient on a collective sub-device while their
    adapter products run on the compute rows (:class:`ttml.modules.ccl_overlap.CollectiveOverlap`); the model's
    ``collective_overlap`` holds the manager. Correct with persistent buffers (iteration 48), but slower: +2.4 ms per
    step over 32 alternating pairs (`doc/optimization/evidence/iter48b_alternate_summary.txt`)."""
    from ttml.models.mistral import Mistral
    from ttml.modules.lora import LoraColumnParallelLinear, LoraLinear, LoraRowParallelLinear

    targets = [m for _, m in model.named_modules() if isinstance(m, Mistral)]
    if isinstance(model, Mistral):
        targets.append(model)
    assert len(targets) == 1, f"expected one Mistral model, found {len(targets)}"
    llama = targets[0]
    norms = [m for _, m in llama.named_modules() if isinstance(m, RMSNormLayer)]
    frozen = [m for m in norms if not m.gamma.tensor.get_requires_grad()]
    assert len(frozen) == len(norms), "the fused residual norm needs every RMSNorm gamma frozen (LoRA)"
    llama.fused_residual_norm = True
    # Iteration 49: the wide norm output for the fused LoRA consumers (the wide path's skinny_matmul writes h into the
    # tail); a composite consumer would see a wider input, so the tail is set only when every qkv and gate_up is a
    # fused LoRA layer.
    from ttml.modules.fused_lora import TAIL_ROWS

    consumers = [
        m
        for name, m in llama.named_modules()
        if name.endswith(("qkv_linear", "w_gate_up")) and isinstance(m, (LoraLinear, LoraColumnParallelLinear))
    ]
    llama.norm_out_tail_cols = TAIL_ROWS if consumers and all(getattr(m, "lora_fused", False) for m in consumers) else 0
    # Iteration 37: a fused w2 with a transposed-orientation forward copy hands its output transposed to the
    # residual norm (no transpose op); only valid under this forward, whose norms read it transposed. (A fused
    # gate_up with such a copy hands its output transposed to the SwiGLU instead, iteration 39, set at fusion.)
    for name, m in llama.named_modules():
        if (name.endswith("w2") or name.endswith("out_linear")) and getattr(m, "weight_cat_fwd", None) is not None:
            m.transposed_output = True  # iteration 48: the out-projection in the transposed orientation too
    row_parallel = [m for _, m in llama.named_modules() if isinstance(m, LoraRowParallelLinear)]
    if sequence_parallel and row_parallel:
        llama.sequence_parallel_axis = row_parallel[0].cluster_axis
        for _, m in llama.named_modules():
            if isinstance(m, (LoraRowParallelLinear, LoraColumnParallelLinear)):
                m.sequence_parallel = True
        column_parallel = [
            m
            for name, m in llama.named_modules()
            if (name.endswith("w_gate_up") or name.endswith("qkv_linear"))
            and isinstance(m, LoraColumnParallelLinear)
            and getattr(m, "lora_fused", False)
        ]
        if collective_overlap and column_parallel:
            from ttml.modules.ccl_overlap import CollectiveOverlap

            llama.collective_overlap = CollectiveOverlap(
                ttml.autograd.AutoContext.get_instance().get_device(),
                serial=overlap_serial,
                serial_async=overlap_serial_async,
            )
            for m in column_parallel:  # iteration 47: gate_up; iteration 48: the fused qkv as well
                m.collective_overlap = llama.collective_overlap
    return len(norms) - 1  # the first norm stays the plain op
