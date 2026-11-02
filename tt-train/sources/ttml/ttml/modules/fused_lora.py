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
    forward_transposed_matmul_kwargs,
    full_core_grid,
    local_in_out,
)
from .lora import skinny_contract, skinny_matmul, skinny_matmul_fits

__all__ = [
    "FusedLoraLinearT",
    "SwigluPackedWide",
    "fuse_lora_into_frozen_weights",
    "reduce_partial_lora_grads",
    "TAIL_ROWS",
]

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


def write_tail_columns(weight_cat_fwd, tail_cols, in_features):
    """Overwrite the first tail tile-column block (columns ``in_features .. in_features + 32``) of the transposed
    forward copy ``weight_cat_fwd`` (``[1, 1, out, in + TAIL_ROWS]``) with ``tail_cols`` (``[1, 1, out, 32]``), in
    place (iteration 36: the same scaled ``B`` the row tail of ``weight_cat`` holds, as columns)."""
    ttml.ops.metal.write_tile_block(weight_cat_fwd, tail_cols, 0, in_features // 32)


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
        weight_cat_fwd=None,
        transposed_output=False,
        lowrank_sink=None,
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
        # Transposed-orientation forward (iteration 36): the module keeps ``W_cat^T`` ([out, in + tail], a plain
        # ttnn tensor, no gradient) whose tail columns get the same scaled B; ``y^T = W_cat^T @ x_cat^T`` runs the
        # weight as the row-multicast operand on the full grid, then one transpose. Only for shapes with a config.
        # Transposed wide input (iteration 38): the SwiGLU wrote h^T into a [in + TAIL_ROWS, T] buffer (its tail rows
        # zeroed); the LoRA h^T goes into its tail rows and the transposed-orientation matmul takes it as a plain
        # [K, N] operand. The buffer's gradient comes back in the same layout (the SwiGLU backward folds the
        # low-rank term in, see the backward).
        wide_t = (
            weight_cat_fwd is not None
            and int(x_val.shape[-2]) == in_features + TAIL_ROWS
            and int(x_val.shape[-1]) != in_features + TAIL_ROWS
        )
        rows = int(x_val.shape[-1]) if wide_t else int(x_val.shape[-2])
        transposed_kwargs = (
            forward_transposed_matmul_kwargs(in_features + TAIL_ROWS, out_features, rows)
            if weight_cat_fwd is not None
            else None
        )
        assert not wide_t or transposed_kwargs is not None, "the transposed wide input needs the transposed config"
        if transposed_kwargs is not None:
            b_cols = ttnn.transpose(bt, -2, -1)  # [1, 1, out, 32]
            write_tail_columns(weight_cat_fwd, b_cols, in_features)
            ttnn.deallocate(b_cols)
        # Wide input (iteration 24): the producer (SwiGLU) wrote x into a [T, in + TAIL_ROWS] buffer, so h goes
        # straight into its tail (zero-padded to TAIL_ROWS) and the K-concat and pad disappear. Otherwise the
        # concat path.
        wide = (
            not wide_t
            and int(x_val.shape[-1]) == in_features + TAIL_ROWS
            and h_reduce_axis is None
            and skinny_matmul_fits(in_features, rank, a_val)
        )
        if wide_t:
            # h^T [32, T] = A_wide @ x_cat^T: A zero-padded to 32 rows and to the buffer's K (the tail rows are zero,
            # so the extra columns contribute nothing), written into the tail rows of the buffer.
            a_wide = ttnn.pad(a_val, padding=[(0, 0), (0, 0), (0, 32 - rank), (0, TAIL_ROWS)], value=0.0)
            h = ttnn.matmul(a_wide, x_val, compute_kernel_config=cfg, core_grid=full_core_grid())  # [1, 1, 32, T]
            ttnn.deallocate(a_wide)
            ttml.ops.metal.write_tile_block(x_val, h, in_features // 32, 0)
            x_cat = x_val
        elif wide:
            h = skinny_matmul(
                x_val, a_val, transpose_b=True, tail_out=x_val, tail_col=in_features // 32, tail_tiles=TAIL_ROWS // 32
            )
            x_cat = x_val
        else:
            h = skinny_matmul(x_val, a_val, transpose_b=True)  # [T, rank], K split over the cores (iteration 10)
            if h_reduce_axis is not None:  # row-parallel: A is sharded on in, h is a partial sum
                h = ttml.core.distributed.all_reduce(h, h_reduce_axis)
            h_pad = ttnn.pad(h, padding=[(0, 0), (0, 0), (0, 0), (0, TAIL_ROWS - rank)], value=0.0)
            x_cat = ttnn.concat([x_val, h_pad], dim=3)  # [T, in + TAIL_ROWS]
            ttnn.deallocate(h_pad)
        transposed_out = transposed_kwargs is not None and transposed_output
        if transposed_kwargs is not None:
            y_t = ttnn.matmul(
                weight_cat_fwd, x_cat, transpose_b=not wide_t, compute_kernel_config=cfg, **transposed_kwargs
            )
            if transposed_out:
                y = y_t  # iteration 37: the consumer (the fused residual norm) reads [out, T]; its gradient is [out, T]
            else:
                y = ttnn.transpose(y_t, -2, -1)
                ttnn.deallocate(y_t)
        else:
            y = ttnn.matmul(
                x_cat,
                w_val,
                compute_kernel_config=cfg,
                **forward_matmul_kwargs(in_features, out_features, x_cat.shape[-2], k=x_cat.shape[-1]),
            )
        ctx.save_for_backward(x_val, a_val, h, w_val)
        ctx.meta = (scaling, in_features, out_features, rank, dh_scale, reduce_axis, wide, transposed_out, wide_t)
        ctx.lowrank_sink = lowrank_sink
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_val, a_val, h, w_val = ctx.saved_tensors
        ctx.save_for_backward()
        scaling, in_features, out_features, rank, dh_scale, reduce_axis, wide, transposed_out, wide_t = ctx.meta
        cfg = _compute_config()
        # One matmul: [in + TAIL, out] @ [out, T] -> [in + TAIL, T]; rows [:in] = dx_base^T, rows [in:in+rank] = s*dh^T.
        # With the transposed output (iteration 37) the gradient arrives as [out, T]: the same product without the
        # free in1 transpose, and dB from it with transpose_b instead (both exact).
        if transposed_out:
            dcat_t = ttnn.matmul(w_val, grad_output, compute_kernel_config=cfg, core_grid=full_core_grid())
        else:
            dcat_t = ttnn.matmul(
                w_val,
                grad_output,
                transpose_b=True,
                compute_kernel_config=cfg,
                **dx_matmul_kwargs(w_val.shape[-2], w_val.shape[-1], grad_output.shape[-2]),
            )
        if wide_t:
            # Transposed wide input (iteration 38): the gradient of the [in + TAIL, T] buffer is the product itself
            # plus the low-rank term dh^T += A^T s_dh^T, which the SwiGLU backward folds in while it transposes the
            # tiles (it reads A and the scale from the wrapper); here only the adapter gradients. h is h^T [32, T].
            assert ctx.lowrank_sink is not None, "the transposed wide input needs the wrapper as the low-rank sink"
            T = int(dcat_t.shape[-1])
            s_dh_t = ttnn.slice(dcat_t, [0, 0, in_features, 0], [1, 1, in_features + 32, T], [1, 1, 1, 1])
            if dh_scale != 1.0:
                s_dh_t = ttnn.multiply(s_dh_t, dh_scale)
            if reduce_axis is not None:
                s_dh_t = ttml.core.distributed.all_reduce(s_dh_t, reduce_axis)
            # dA = (s dh)^T x -> [32, in + TAIL] (the padded rows / tail columns dropped) and dB^T = h^T g -> [32, out]
            # through ttnn's free in1 transpose (the all-core op's transposed-big mode is no faster at these shapes,
            # `O/iter38_contract_check.log`).
            da_wide = ttnn.matmul(
                s_dh_t, x_val, transpose_b=True, compute_kernel_config=cfg, core_grid=full_core_grid()
            )
            grad_a = ttnn.slice(da_wide, [0, 0, 0, 0], [1, 1, rank, in_features], [1, 1, 1, 1])
            ttnn.deallocate(da_wide)
            if transposed_out:
                db_t = ttnn.matmul(
                    h, grad_output, transpose_b=True, compute_kernel_config=cfg, core_grid=full_core_grid()
                )
            else:
                db_t = ttnn.matmul(h, grad_output, compute_kernel_config=cfg, core_grid=full_core_grid())
            if scaling != 1.0:
                db_t = ttnn.multiply(db_t, scaling)
            db_full = ttnn.transpose(db_t, -2, -1)  # [out, 32]
            ttnn.deallocate(db_t)
            grad_b = ttnn.slice(db_full, [0, 0, 0, 0], [1, 1, out_features, rank], [1, 1, 1, 1])
            ttnn.deallocate(db_full)
            ttnn.deallocate(s_dh_t)
            ctx.lowrank_sink._lowrank_grad = (a_val, dh_scale)
            return dcat_t, None, grad_a, grad_b
        # dx = rows [:in]^T and s_dh = dh_scale * rows [in:in+rank]^T ([T, rank], already scaled by s and 1/tp) in
        # one pass over the product (iteration 13; was two slices, two transposes and a multiply).
        # With the wide input dx is [T, in + TAIL] (its tail columns are the unscaled tail rows, which the
        # producer's backward never reads) so the autograd shapes match.
        # dx += (s dh) @ A is fused into the transpose (iteration 25): the tail's s_dh tile and A's tile column
        # multiply in the same DEST pass as the transposed tile. dB = s * g^T h ; dA = (s dh)^T x (dh all-reduced
        # first when B is column-sharded).
        dx, s_dh = ttml.ops.metal.transpose_split(
            dcat_t, in_features, rank, dh_scale, int(x_val.shape[-1]) if wide else 0, a_val
        )
        ttnn.deallocate(dcat_t)
        grad_b = skinny_contract(
            h, grad_output, scaling, transpose_out=True, big_transposed=transposed_out
        )  # s * g^T h -> [out, r]
        if reduce_axis is not None:
            s_dh = ttml.core.distributed.all_reduce(s_dh, reduce_axis)
        grad_a = skinny_contract(s_dh, x_val)  # (s dh)^T x -> [r, in] (+ the tail columns when x is wide)
        if wide:
            grad_a = ttnn.slice(grad_a, [0, 0, 0, 0], [1, 1, rank, in_features], [1, 1, 1, 1])
        return dx, None, grad_a, grad_b


class SwigluPackedWide(Function):
    """``silu(gate) * up`` written into a ``[.., R, out_cols]`` buffer (``out_cols = I + TAIL_ROWS``): the wide input
    of a fused LoRA ``w2`` (iteration 24). The backward takes the wide gradient and reads its first ``I`` columns."""

    @staticmethod
    def forward(ctx, packed, out_cols, sink=None, transposed=False):
        p = packed.get_value()
        ctx.save_for_backward(p)
        ctx.sink, ctx.transposed = sink, transposed
        # Iteration 38: ``transposed`` writes h^T into a [out_cols, R] buffer (tail rows zeroed) for a
        # transposed-orientation w2, whose backward leaves (A, scale) on ``sink`` for the low-rank term.
        return ttml.ops.metal.swiglu_packed_fw(p, None, out_cols, transposed_output=transposed)

    @staticmethod
    def backward(ctx, grad_h):
        (p,) = ctx.saved_tensors
        ctx.save_for_backward()
        if ctx.transposed:
            a_val, scale = ctx.sink._lowrank_grad  # left by FusedLoraLinearT.backward (the same backward pass)
            del ctx.sink._lowrank_grad
            return ttml.ops.metal.swiglu_packed_bw(
                p, grad_h, None, lora_a=a_val, transposed_grad=True, lowrank_scale=scale
            )
        return ttml.ops.metal.swiglu_packed_bw(p, grad_h)  # one gradient: `out_cols` is not a tensor


def _install_wide_swiglu(model, w2_names: set[str]) -> list[str]:
    """Give every MLP whose ``w2`` is a fused wrapper on the wide path a ``swiglu_op`` that writes its output into
    the ``[T, in + TAIL_ROWS]`` buffer ``FusedLoraLinearT`` expects (``MLP.forward`` honours the hook)."""
    installed: list[str] = []
    for name, module in model.named_modules():
        w2 = getattr(module, "w2", None)
        w2_name = f"{name}.w2" if name else "w2"
        if w2 is None or w2_name not in w2_names:
            continue
        out_cols = int(w2.frozen_shape[0]) + TAIL_ROWS

        def swiglu_op(gu, out_cols=out_cols, w2=w2):
            return SwigluPackedWide.apply(gu, out_cols, w2, getattr(w2, "transposed_input", False))

        module.swiglu_op = swiglu_op
        installed.append(w2_name)
    return installed


def fused_lora_linear_t(
    x,
    weight_cat,
    lora_A,
    lora_B,
    scaling,
    shape,
    tail_scale=1.0,
    dh_scale=1.0,
    reduce_axis=None,
    h_reduce_axis=None,
    weight_cat_fwd=None,
    transposed_output=False,
    lowrank_sink=None,
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
        weight_cat_fwd,
        transposed_output,
        lowrank_sink,
    )


def fuse_lora_into_frozen_weights(model, batched_tp_grads: bool = True, transposed_rows_hint: int = 1024) -> list[str]:
    """Extend every transposed frozen weight of a LoRA wrapper by a zero tail of ``TAIL_ROWS`` rows and switch the
    wrapper to :class:`FusedLoraLinearT`. Call after :func:`transpose_frozen_weights`.

    ``batched_tp_grads``: under TP, drop the per-layer all-reduces of the LoRA ``h`` / ``dh`` (``[T, r]``, four per
    layer, latency-bound at ~150 us each on the 1x4 ring) and leave the replicated adapter of each parallel layer
    (``A`` of a column-parallel, ``B`` of a row-parallel layer) with a partial gradient, which
    :func:`reduce_partial_lora_grads` sums over the TP axis in two collectives per step. This marks every TP LoRA
    wrapper, fused or on the composite adapter (the shapes not on the transposed path). The training step must
    call it after the backward and before the optimizer step.

    ``transposed_rows_hint``: the token count the transposed-orientation forward configs are checked against when
    deciding which fused weights get the ``W_cat^T`` copy (iteration 36)."""
    from .lora import LoraColumnParallelLinear, LoraLinear, LoraRowParallelLinear, skinny_matmul_fits

    converted: list[str] = []
    wide_w2: set[str] = set()
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
        # Transposed-orientation forward copy (iteration 36): ``W_cat^T`` for the shapes with a config (the TP w2
        # shard); its tail columns are rewritten with the scaled B every forward.
        if forward_transposed_matmul_kwargs(rows + TAIL_ROWS, cols, transposed_rows_hint) is not None:
            module.weight_cat_fwd = ttnn.transpose(extended, -2, -1)
            # Iteration 38: the wide producer (SwiGLU) hands the input over transposed ([in + tail, T]) so the
            # transposed-orientation matmul takes it as a plain operand and the backward needs no transpose_split.
            module.transposed_input = True
        # The wide input path (iteration 24) needs h unreduced on the rank: every wrapper but a row-parallel one
        # without the batched gradients (that one all-reduces h before the tail).
        if (
            name.endswith(".w2")
            and (not isinstance(module, LoraRowParallelLinear) or getattr(module, "tp_partial_grads", None))
            and skinny_matmul_fits(
                int(module.frozen_shape[0]),
                int(module.lora_B.tensor.get_value().shape[-1]),
                module.lora_A.tensor.get_value(),
            )
        ):
            wide_w2.add(name)
    _install_wide_swiglu(model, wide_w2)
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
