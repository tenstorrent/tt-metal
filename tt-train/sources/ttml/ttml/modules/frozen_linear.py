# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A frozen linear layer whose weight is stored transposed, ``[in, out]``, so the forward is a plain matmul.

ttml stores a linear weight as ``[out, in]`` and issues ``ttnn.linear(x, W, transpose_b=True)``. On Blackhole the
transposed-B path is 14-28 % slower than the same matmul on a pre-transposed weight for the wide projections of
Mistral-Small-24B (``[1024, 5120] x [5120, 65536]``: 11.75 -> 8.45 ms on an 8x11 grid), and the input gradient
is 17 % faster as ``(W_t @ g^T)^T`` than as ``g @ W``. A frozen weight (LoRA base, LM head) can be transposed
once after loading; a trainable one cannot, since the optimizer and the checkpoint formats expect ``[out, in]``.

:func:`transpose_frozen_weights` converts every frozen linear of a model in place; the layers' forwards then take
this path. Measurements: ``ttml/models/mistral/doc/optimization/evidence/iter2_matmul_bench*.log``.
"""

from __future__ import annotations

import ttnn
import ttml
from ttml.autograd import Function

__all__ = [
    "FrozenLinearT",
    "frozen_linear_t",
    "forward_core_grid",
    "transpose_frozen_weights",
    "FORWARD_CORE_GRID",
    "TRANSPOSED_SHAPES",
]

# Core grid for the forward matmul by (in, out): the full grid unless a measurement says otherwise. ttnn's default
# program config splits N over the grid's x dimension; 65536 / 32 = 2048 tiles do not divide by 11 but do by 8,
# which is where the 8x11 win comes from, while N = 131072 on 8 columns overflows L1 and falls back to a slow
# path. Tuned on Blackhole p300 for the Mistral-Small-24B shapes (see the module docstring).
FORWARD_CORE_GRID: dict[tuple[int, int], tuple[int, int]] = {
    (5120, 65536): (8, 11),  # (y, x): gate_up, 11.78 -> 9.07 ms (full grid 10.53)
    (4096, 5120): (8, 11),  # out-proj, 0.51 -> 0.49 ms
}

# (in, out) shapes (local to the chip under TP) that are transposed at all: those where the forward on W_t
# and/or the input gradient as (W_t @ g^T)^T beat today's path (iter2_matmul_bench3.log, iter2_matmul_bench4.log,
# wall ms, same clock within a row): gate_up fwd 11.78 -> 9.07 and dX 7.71 -> 7.19; LM head fwd 23.66 -> 21.48 and
# dX 15.88 -> 15.00; out dX 0.47 -> 0.43; qkv dX 0.74 -> 0.65; TP-local gate_up dX 1.89 -> 1.58, w2 0.93 -> 0.85,
# LM head fwd 4.24 -> 3.55 and dX 4.34 -> 3.14. Not w2 (32768 -> 5120: fwd equal, every dX route slower:
# 3.39 -> 4.59 / 5.85) nor the small TP qkv / out shards (equal or slower). Every transposed shape uses the
# (W_t @ g^T)^T gradient route, with the full core grid: without an explicit core_grid ttnn's default program
# config is 1.5-1.8x slower on these shapes (bench4), which is what the first iteration-2 profile showed.
TRANSPOSED_SHAPES: set[tuple[int, int]] = {
    (5120, 6144),
    (4096, 5120),
    (5120, 65536),
    (5120, 131072),  # single chip
    (5120, 16384),
    (8192, 5120),
    (5120, 32768),  # TP=4 shards
}


def _compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )


def full_core_grid():
    grid = ttml.autograd.AutoContext.get_instance().get_device().compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def local_in_out(weight) -> tuple[int, int] | None:
    """(in, out) of the weight shard one chip holds, if it is a shape in :data:`TRANSPOSED_SHAPES`, else None.

    A mesh tensor may report its global ``[out, in]`` or the per-chip shard depending on how it was built, so both
    the reported shape and the reported shape divided along the dim sharded over ``tp`` are tried; the local
    candidate wins when both are listed (the single-chip and TP-local tables do not collide on a shape).
    """
    value = weight.get_value()
    out_features, in_features = int(value.shape[-2]), int(value.shape[-1])
    candidates = [(in_features, out_features)]
    mesh = ttml.maybe_mesh()
    if mesh is not None and mesh.has_axis("tp"):
        tp = mesh.axis_size("tp")
        axis = mesh.axis_index("tp")
        placements = ttml.Sharding.from_tensor(weight).placements
        placement = placements[axis] if placements and axis < len(placements) else None
        if isinstance(placement, ttnn.PlacementShard) and placement.dim == 2 and out_features % tp == 0:
            candidates.insert(0, (in_features, out_features // tp))
        elif isinstance(placement, ttnn.PlacementShard) and placement.dim == 3 and in_features % tp == 0:
            candidates.insert(0, (in_features // tp, out_features))
    for shape in candidates:
        if shape in TRANSPOSED_SHAPES:
            return shape
    return None


def forward_core_grid(in_features: int, out_features: int):
    grid = ttml.autograd.AutoContext.get_instance().get_device().compute_with_storage_grid_size()
    y, x = FORWARD_CORE_GRID.get((in_features, out_features), (grid.y, grid.x))
    return ttnn.CoreGrid(y=min(y, grid.y), x=min(x, grid.x))


class FrozenLinearT(Function):
    """``y = x @ W_t`` for a frozen ``W_t = [1, 1, in, out]``; no weight gradient, ``dx = (W_t @ g^T)^T``."""

    @staticmethod
    def forward(ctx, x, weight_t, bias, shape):
        """``shape`` is the (in, out) of the local shard, resolved once at conversion time."""
        w = weight_t.get_value()
        in_features, out_features = shape
        ctx.save_for_backward(w)
        ctx.has_bias = bias is not None
        y = ttnn.matmul(
            x.get_value(),
            w,
            compute_kernel_config=_compute_config(),
            core_grid=forward_core_grid(in_features, out_features),
        )
        if bias is not None:
            y = ttnn.add(y, bias.get_value())
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        ctx.save_for_backward()
        # [in, out] @ [out, T] -> [in, T], then back to [T, in]: faster than g @ W for every transposed shape.
        dx_t = ttnn.matmul(
            w, grad_output, transpose_b=True, compute_kernel_config=_compute_config(), core_grid=full_core_grid()
        )
        dx = ttnn.transpose(dx_t, -2, -1)
        ttnn.deallocate(dx_t)
        # The weight is frozen (no dW); a bias here would be frozen too (the Mistral linears have none).
        # The Function expects one gradient per tensor input, and a None bias is not a tensor input.
        return (dx, None, None) if ctx.has_bias else (dx, None)


def frozen_linear_t(x, weight_t, bias, shape):
    return FrozenLinearT.apply(x, weight_t, bias, shape)


def transpose_frozen_weights(model) -> list[str]:
    """Store every frozen linear weight of ``model`` transposed and switch those layers to :class:`FrozenLinearT`.

    Covers ``LinearLayer``, ``ColumnParallelLinear``, ``RowParallelLinear`` and their LoRA wrappers (they share the
    base ``weight`` Parameter). A weight that requires a gradient is left alone. Returns the converted module names.
    Call after loading the checkpoint and after LoRA injection (both expect ``[out, in]``).
    """
    from .linear import LinearLayer, ColumnParallelLinear, RowParallelLinear
    from .lora import LoraLinear, LoraColumnParallelLinear, LoraRowParallelLinear

    kinds = (
        LinearLayer,
        ColumnParallelLinear,
        RowParallelLinear,
        LoraLinear,
        LoraColumnParallelLinear,
        LoraRowParallelLinear,
    )
    converted: list[str] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        if not isinstance(module, kinds) or getattr(module, "weight_transposed", False):
            continue
        weight = module.weight.tensor
        if weight.get_requires_grad():
            continue
        shape = local_in_out(weight)
        if shape is None:
            continue
        if id(weight) not in seen:
            value = weight.get_value()
            transposed = ttnn.transpose(value, -2, -1)
            weight.set_value(transposed)
            ttnn.deallocate(value)
            seen.add(id(weight))
        module.weight_transposed = True
        module.frozen_shape = shape
        converted.append(name)
    return converted
