# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA update folded into the frozen base matmul by K-concatenation.

For a frozen linear stored transposed (``W_t = [in, out]``, see :mod:`.frozen_linear`) with a LoRA adapter
``A = [r, in]``, ``B = [out, r]`` and scale ``s``:

    y = x @ W_t + s * (x @ A^T) @ B^T  =  [x | h] @ [[W_t], [s * B^T]],   h = x @ A^T

so the base matmul with K extended by the (tile-padded) rank computes base and update in one pass: the separate
B matmul, the ``addalpha`` and, in the backward, the ``dh = g @ B`` matmul disappear (for gate_up: 0.83 + 0.78 +
0.77 ms per layer at 1-5 % of peak). ``W_cat_t = [in + pad, out]`` holds the frozen weight with ``s * B^T`` in the
first ``r`` rows of its tail, rewritten every step (the tail is a contiguous tile-row block, written with
``ttml.ops.metal.write_tile_block``); the tail rows beyond ``r`` stay zero. The backward reads
``dX_cat^T = W_cat_t @ g^T``: its first ``in`` rows are ``dx_base^T``, its tail rows ``s * dh^T``.

The adapter stays a separate trainable ``B`` (``[out, r]``, the usual layout for the optimizer and checkpoints);
only its scaled transpose is copied into the tail. Under TP: a column-parallel layer (``out`` sharded) keeps
``B`` sharded on ``out`` and the tail is the local shard; a row-parallel layer (``in`` sharded, ``B`` replicated)
writes ``s / tp * B^T`` so the output all-reduce sums to ``s * h B^T`` once, and scales the tail gradient back by
``tp``. Measured costs and gains: ``ttml/models/mistral/doc/perf_analysis.md`` (iteration 3).
"""

from __future__ import annotations

import ttnn
import ttml
from ttml.autograd import Function

from .frozen_linear import (
    dx_matmul_kwargs,
    TRANSPOSED_SHAPES,
    _compute_config,
    forward_matmul_kwargs,
    full_core_grid,
    local_in_out,
)
from .lora import lowrank_update, skinny_contract, skinny_matmul

__all__ = ["FusedLoraLinearT", "fuse_lora_into_frozen_weights", "reduce_partial_lora_grads", "TAIL_ROWS"]

# Two tile rows: the first holds s * B^T (rank 16, zero-padded to 32), the second stays zero. K then grows by two
# tiles (162 for in = 5120), which ttnn's block heuristic handles like 160; one extra tile (161, prime) cost 5 % on
# the gate_up forward (iter3 tail measurement: K 5120 / 5152 / 5184 -> 7.51 / 7.92 / 7.59 ms).
TAIL_ROWS = 64


def write_tail(weight_cat, tail_rows, in_features):
    """Overwrite the first tail block (rows ``in_features .. in_features + 32``) of ``weight_cat`` with ``tail_rows``
    (``[1, 1, 32, out]``), in place. The remaining tail rows stay zero.

    ``ttml.ops.metal.write_tile_block`` copies the tile block with every core taking a share of its tiles. The earlier
    ``ttnn.fill_cache`` through the ``[R / 32, 1, 32, C]`` block view parallelised over the input's tile-rows, so a
    one-tile-row tail ran on a single core (0.79 ms for gate_up); narrowing the view is not possible because
    ``ttnn.reshape`` copies whenever the last dimension changes (iteration 4)."""
    ttml.ops.metal.write_tile_block(weight_cat, tail_rows, in_features // 32)


class FusedLoraLinearT(Function):
    """``[x | x A^T] @ W_cat_t`` with ``W_cat_t`` holding ``s * B^T`` (times ``1/tp`` for a row-parallel layer) in its
    tail, and the backward reading ``dx`` and ``dh`` from one ``W_cat_t @ g^T``."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight_cat,
        lora_A,
        lora_B,
        scaling,
        in_features,
        out_features,
        tail_scale,
        dh_scale,
        reduce_axis,
        h_reduce_axis,
    ):
        cfg = _compute_config()
        x_val, a_val, b_val, w_val = x.get_value(), lora_A.get_value(), lora_B.get_value(), weight_cat.get_value()
        rank = int(b_val.shape[-1])
        # Tail block = scaled B^T, zero-padded from rank to 32 rows, written into the frozen weight in place.
        bt = ttnn.transpose(b_val, -2, -1)  # [1, 1, rank, out]
        if scaling * tail_scale != 1.0:
            bt = ttnn.multiply(bt, scaling * tail_scale)
        if rank != 32:
            bt = ttnn.pad(bt, padding=[(0, 0), (0, 0), (0, 32 - rank), (0, 0)], value=0.0)
        write_tail(w_val, bt, in_features)
        h = skinny_matmul(x_val, a_val, transpose_b=True)  # [T, rank], K split over the cores (iteration 10)
        if h_reduce_axis is not None:  # row-parallel: A is sharded on in, h is a partial sum
            h = ttml.core.distributed.all_reduce(h, h_reduce_axis)
        h_pad = ttnn.pad(h, padding=[(0, 0), (0, 0), (0, 0), (0, TAIL_ROWS - rank)], value=0.0)
        x_cat = ttnn.concat([x_val, h_pad], dim=3)  # [T, in + TAIL_ROWS]
        ttnn.deallocate(h_pad)
        y = ttnn.matmul(
            x_cat,
            w_val,
            compute_kernel_config=cfg,
            **forward_matmul_kwargs(in_features, out_features, x_cat.shape[-2], k=x_cat.shape[-1]),
        )
        ctx.save_for_backward(x_val, a_val, h, w_val)
        ctx.meta = (scaling, in_features, out_features, rank, dh_scale, reduce_axis)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_val, a_val, h, w_val = ctx.saved_tensors
        ctx.save_for_backward()
        scaling, in_features, out_features, rank, dh_scale, reduce_axis = ctx.meta
        cfg = _compute_config()
        # One matmul: [in + TAIL, out] @ [out, T] -> [in + TAIL, T]; rows [:in] = dx_base^T, rows [in:in+rank] = s*dh^T.
        dcat_t = ttnn.matmul(
            w_val,
            grad_output,
            transpose_b=True,
            compute_kernel_config=cfg,
            **dx_matmul_kwargs(w_val.shape[-2], w_val.shape[-1], grad_output.shape[-2]),
        )
        # dx = rows [:in]^T and s_dh = dh_scale * rows [in:in+rank]^T ([T, rank], already scaled by s and 1/tp) in
        # one pass over the product (iteration 13; was two slices, two transposes and a multiply).
        dx, s_dh = ttml.ops.metal.transpose_split(dcat_t, in_features, rank, dh_scale)
        ttnn.deallocate(dcat_t)
        # dx += (s dh) @ A ; dB = s * g^T h ; dA = (s dh)^T x (dh all-reduced first when B is column-sharded)
        dx = lowrank_update(dx, s_dh, a_val)
        grad_b = skinny_contract(h, grad_output, scaling, transpose_out=True)  # s * g^T h -> [out, r]
        if reduce_axis is not None:
            s_dh = ttml.core.distributed.all_reduce(s_dh, reduce_axis)
        grad_a = skinny_contract(s_dh, x_val)  # (s dh)^T x -> [r, in]
        return dx, None, grad_a, grad_b


def fused_lora_linear_t(
    x, weight_cat, lora_A, lora_B, scaling, shape, tail_scale=1.0, dh_scale=1.0, reduce_axis=None, h_reduce_axis=None
):
    in_features, out_features = shape
    return FusedLoraLinearT.apply(
        x,
        weight_cat,
        lora_A,
        lora_B,
        scaling,
        in_features,
        out_features,
        tail_scale,
        dh_scale,
        reduce_axis,
        h_reduce_axis,
    )


def fuse_lora_into_frozen_weights(model, batched_tp_grads: bool = True) -> list[str]:
    """Extend every transposed frozen weight of a LoRA wrapper by a zero tail of ``TAIL_ROWS`` rows and switch the
    wrapper to :class:`FusedLoraLinearT`. Call after :func:`transpose_frozen_weights`.

    ``batched_tp_grads``: under TP, drop the per-layer all-reduces of the LoRA ``h`` / ``dh`` (``[T, r]``, four per
    layer, latency-bound at ~150 us each on the 1x4 ring) and leave the replicated adapter of each parallel layer
    (``A`` of a column-parallel, ``B`` of a row-parallel layer) with a partial gradient, which
    :func:`reduce_partial_lora_grads` sums over the TP axis in two collectives per step. This marks every TP LoRA
    wrapper, fused or on the composite adapter (the shapes not on the transposed path). The training step must
    call it after the backward and before the optimizer step."""
    from .lora import LoraColumnParallelLinear, LoraLinear, LoraRowParallelLinear

    converted: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, (LoraLinear, LoraColumnParallelLinear, LoraRowParallelLinear)):
            continue
        if batched_tp_grads and isinstance(module, LoraColumnParallelLinear) and not module.gather_output:
            module.tp_partial_grads = ("lora_A",)
        elif batched_tp_grads and isinstance(module, LoraRowParallelLinear):
            module.tp_partial_grads = ("lora_B",)
        if not getattr(module, "weight_transposed", False) or getattr(module, "lora_fused", False):
            continue
        if isinstance(module, LoraColumnParallelLinear) and module.gather_output:
            continue  # the gathered path keeps the composite adapter
        if module.bias is not None:
            continue
        weight = module.weight.tensor
        value = weight.get_value()
        rows, cols = int(value.shape[-2]), int(value.shape[-1])
        zeros = ttnn.zeros([1, 1, TAIL_ROWS, cols], dtype=value.dtype, layout=ttnn.TILE_LAYOUT, device=value.device())
        extended = ttnn.concat([value, zeros], dim=2)
        weight.set_value(extended)
        ttnn.deallocate(value)
        ttnn.deallocate(zeros)
        module.lora_fused = True
        converted.append(name)
    return converted


def reduce_partial_lora_grads(model, cluster_axis) -> int:
    """Sum the partial LoRA adapter gradients left by the batched fused TP path over ``cluster_axis``: the ``[r, in]``
    ``dA`` partials are stacked on their row dim and the ``[out, r]`` ``dB`` partials side by side on their column
    dim, so a step reduces two tensors instead of four ``[T, r]`` ones per layer. Returns the number of gradients
    reduced; a no-op for models fused without ``batched_tp_grads``."""
    groups: dict[tuple, list] = {}
    for _, module in model.named_modules():
        for pname in getattr(module, "tp_partial_grads", ()):
            param = getattr(module, pname).tensor
            if not param.is_grad_initialized():
                continue
            grad = param.get_grad()
            rows, cols = int(grad.shape[-2]), int(grad.shape[-1])
            dim = 2 if rows <= cols else 3  # [r, in] stack rows, [out, r] stack columns
            key = (dim, cols if dim == 2 else rows)
            groups.setdefault(key, []).append((param, grad))
    reduced = 0
    for (dim, _), entries in groups.items():
        if len(entries) == 1:
            cat = entries[0][1]
        else:
            cat = ttnn.concat([g for _, g in entries], dim=dim)
        cat = ttml.core.distributed.all_reduce(cat, cluster_axis=cluster_axis)
        offset = 0
        for param, grad in entries:
            n = int(grad.shape[dim])
            if len(entries) == 1:
                part = cat
            else:
                start = [0, 0, 0, 0]
                end = list(int(d) for d in cat.shape)
                start[dim], end[dim] = offset, offset + n
                part = ttnn.slice(cat, start, end)
            param.set_grad(part)
            offset += n
            reduced += 1
    return reduced
