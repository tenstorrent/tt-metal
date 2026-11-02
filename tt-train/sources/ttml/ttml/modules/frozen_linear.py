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
    "ConfiguredFrozenLinear",
    "configured_linear",
    "configured_base_shape",
    "forward_core_grid",
    "transpose_frozen_weights",
    "FORWARD_CORE_GRID",
    "FORWARD_PROGRAM_CONFIG",
    "FORWARD_TRANSPOSED_CONFIG",
    "forward_transposed_matmul_kwargs",
    "PROGRAM_CONFIG_2D",
    "matmul_kwargs_2d",
    "dx_plain_matmul_kwargs",
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
    for shape in shard_candidates(weight):
        if shape in TRANSPOSED_SHAPES:
            return shape
    return None


def shard_candidates(weight) -> list[tuple[int, int]]:
    """The (in, out) shapes the weight ``[.., out, in]`` may hold on one chip: the reported shape divided along the
    dim sharded over ``tp`` first (when a mesh with that axis is open), then the reported shape itself."""
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
    return candidates


def forward_core_grid(in_features: int, out_features: int):
    grid = ttml.autograd.AutoContext.get_instance().get_device().compute_with_storage_grid_size()
    y, x = FORWARD_CORE_GRID.get((in_features, out_features), (grid.y, grid.x))
    return ttnn.CoreGrid(y=min(y, grid.y), x=min(x, grid.x))


# Explicit 1D-mcast program configs for the forward matmul ``[rows, in] x [in, out]``, from the iteration-4 sweep
# (``examples/mistral_train/matmul_sweep1d.py``, HiFi4 / fp32 accumulate, bit-exact against ttnn's default choice):
# gate_up 7.92 -> 7.35 ms (ttnn's pick: 8x11 grid, per_core_N 24, in0_block_w 2, out_subblock 2x2, out_block 16x12).
# ``grid`` is (x, y); ``per_core_M`` follows from ``rows``. Shapes not listed fall back to ``forward_core_grid``.
FORWARD_PROGRAM_CONFIG: dict[tuple[int, int], dict] = {
    (5120, 65536): dict(
        grid=(10, 10), in0_block_w=2, out_subblock_h=4, out_subblock_w=1, out_block_h=16, out_block_w=7, per_core_N=21
    ),
    # gate_up forward at the TP=4 shard width [1024, 5184] x [5184, 16384] (iteration 35, ``iter35_matmul_sweep1d.py``):
    # nine K tiles per block instead of ttnn's two (36 matmuls per DEST acquire, the per-block partial round trip
    # amortised 4.5x) at the same 16x5 output block; 1.60 vs 1.68 ms standalone, outputs within one bf16 ulp of
    # ttnn's choice (the K blocking changes the fp32 accumulation order). ``in0_block_w`` must divide K in tiles
    # (162 = 2 x 3^4 with the LoRA tail; ``forward_matmul_kwargs`` falls back when it does not).
    (5120, 16384): dict(
        grid=(11, 10), in0_block_w=9, out_subblock_h=4, out_subblock_w=1, out_block_h=16, out_block_w=5, per_core_N=5
    ),
}


# Explicit 2D (block-multicast) program configs by (M, K, N, transpose_b) of the base matmul, from the iteration-19
# sweep (``doc/optimization/evidence/iter19_matmul_sweep2d.py``, HiFi4 / fp32 accumulate, outputs within one bf16
# ulp of ttnn's default choice: a different K blocking changes the fp32 accumulation order). ``grid`` is (x, y):
# x splits N into ``per_core_N`` tiles per core, y splits M into ``per_core_M``. Shapes not listed keep ttnn's choice.
PROGRAM_CONFIG_2D: dict[tuple[int, int, int, bool], dict] = {
    # w2 forward [1024, 8192 + 64] x [8256, 5120]: 981 -> 882 us standalone (ttnn's choice: 88 cores, 1D).
    (1024, 8256, 5120, False): dict(
        grid=(11, 8), in0_block_w=6, out_subblock_h=4, out_subblock_w=1, per_core_M=4, per_core_N=15
    ),
    # gate_up dX^T [5184, 16384] x [16384, 1024] with the transposed SwiGLU gradient as a plain operand (iteration
    # 39): 1.531 -> 1.499 ms standalone against ttnn's choice (K blocks of 4), `O/iter39_scoping.log`.
    (5184, 16384, 1024, False): dict(
        grid=(11, 10), in0_block_w=8, out_subblock_h=1, out_subblock_w=3, per_core_M=17, per_core_N=3
    ),
    # The attention projections' base matmuls at the TP=4 per-chip shapes on the composite LoRA path (iteration 46,
    # `O/iter46_sweep2d_*.log`, confirmed in alternating rounds against ttnn's default choice in
    # `O/iter46_attn_configs_abab.log`): qkv forward [1024, 5120] x [1536, 5120]^T 207 -> 192 us (-7 %), out forward
    # [1024, 1024] x [5120, 1024]^T 158 -> 143 (-9 %), qkv dX [1024, 1536] x [1536, 5120] 210 -> 188 (-11 %), out dX
    # [1024, 5120] x [5120, 1024] 128 -> 118 (-8 %); every output within one bf16 ulp of the default's (the deeper
    # K blocks change the fp32 accumulation order). Served by :class:`ConfiguredFrozenLinear`.
    (1024, 5120, 1536, True): dict(
        grid=(11, 8), in0_block_w=16, out_subblock_h=4, out_subblock_w=1, per_core_M=4, per_core_N=5
    ),
    (1024, 1024, 5120, True): dict(
        grid=(11, 8), in0_block_w=8, out_subblock_h=4, out_subblock_w=1, per_core_M=4, per_core_N=15
    ),
    (1024, 1536, 5120, False): dict(
        grid=(11, 10), in0_block_w=12, out_subblock_h=4, out_subblock_w=1, per_core_M=4, per_core_N=15
    ),
    (1024, 5120, 1024, False): dict(
        grid=(11, 8), in0_block_w=16, out_subblock_h=2, out_subblock_w=1, per_core_M=4, per_core_N=3
    ),
}


# Transposed-orientation forward (iteration 36): ``y^T = W_cat^T @ x_cat^T`` with the weight copy ``[out, in + tail]`` as
# the row-multicast operand and the activation as the free-transposed column operand. The output dim then splits over
# the grid's rows (M = out) and the token dim over its columns (N = rows / 32 tiles), so the w2 forward runs on all
# 110 cores (0.730 vs 0.883 ms standalone for [5120, 8256] x [1024, 8256]^T against the 88-core 2D config) at the
# cost of a second weight copy and a transpose of the [out, rows] result. Keyed by (K = in + tail, out); ``rows``
# fixes per_core_N at call time. Shapes not listed keep the plain orientation.
FORWARD_TRANSPOSED_CONFIG: dict[tuple[int, int], dict] = {
    (8256, 5120): dict(grid=(11, 10), in0_block_w=6, out_subblock_h=1, out_subblock_w=3, per_core_M=16),
    # gate_up (iteration 39): [16384, 5184] x [1024, 5184]^T on all 110 cores, 1.497 ms standalone against the
    # committed 1D config's 1.62 ms (`O/iter38_scoping.log`); M = 512 tiles over 10 rows needs per_core_M 52 with
    # the output block cut to 13 rows (the full 52 x 3 block at K depth 9 overflows L1).
    (5184, 16384): dict(
        grid=(11, 10), in0_block_w=9, out_subblock_h=1, out_subblock_w=3, per_core_M=52, out_block_h=13
    ),
}


def forward_transposed_matmul_kwargs(k: int, out_features: int, rows: int) -> dict | None:
    """``program_config=`` for the transposed-orientation forward ``[out, k] x [rows, k]^T`` (``transpose_b``), or
    ``None`` when the shape is not listed, the grid is too small or ``rows`` does not fill the config's columns."""
    spec = FORWARD_TRANSPOSED_CONFIG.get((k, out_features))
    if spec is None or rows % 32 != 0:
        return None
    grid = ttml.autograd.AutoContext.get_instance().get_device().compute_with_storage_grid_size()
    gx, gy = spec["grid"]
    per_core_n = -(-(rows // 32) // gx)
    if gx > grid.x or gy > grid.y or per_core_n % spec["out_subblock_w"] != 0 or (k // 32) % spec["in0_block_w"] != 0:
        return None
    return {
        "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=spec["grid"],
            in0_block_w=spec["in0_block_w"],
            out_subblock_h=spec["out_subblock_h"],
            out_subblock_w=spec["out_subblock_w"],
            out_block_h=spec.get("out_block_h", spec["per_core_M"]),
            out_block_w=per_core_n,
            per_core_M=spec["per_core_M"],
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
    }


def matmul_kwargs_2d(m: int, k: int, n: int, transpose_b: bool) -> dict | None:
    """``program_config=`` for a swept 2D config of the ``[m, k] x [k, n]`` matmul (``transpose_b``: the second
    operand is stored ``[n, k]``), or ``None`` when the shape is not listed or the grid is too small."""
    spec = PROGRAM_CONFIG_2D.get((m, k, n, transpose_b))
    if spec is None:
        return None
    grid = ttml.autograd.AutoContext.get_instance().get_device().compute_with_storage_grid_size()
    if spec["grid"][0] > grid.x or spec["grid"][1] > grid.y:
        return None
    return {
        "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=spec["grid"],
            in0_block_w=spec["in0_block_w"],
            out_subblock_h=spec["out_subblock_h"],
            out_subblock_w=spec["out_subblock_w"],
            per_core_M=spec["per_core_M"],
            per_core_N=spec["per_core_N"],
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
    }


def dx_matmul_kwargs(m: int, k: int, n: int) -> dict:
    """Keyword arguments for the transposed-gradient matmul ``[m, k] x [n, k]^T`` (``transpose_b=True``)."""
    return matmul_kwargs_2d(m, k, n, True) or {"core_grid": full_core_grid()}


def dx_plain_matmul_kwargs(m: int, k: int, n: int) -> dict:
    """Keyword arguments for the transposed-gradient matmul ``[m, k] x [k, n]`` whose second operand already is
    the transposed upstream gradient (iteration 37/39: a consumer that hands the gradient over as ``[out, T]``)."""
    return matmul_kwargs_2d(m, k, n, False) or {"core_grid": full_core_grid()}


def forward_matmul_kwargs(in_features: int, out_features: int, rows: int, k: int | None = None) -> dict:
    """``program_config=`` (a swept 2D or 1D config) or ``core_grid=`` keyword arguments for the forward matmul
    ``[rows, k] x [k, out]``; ``k`` is the actual K of the call (``in + tail`` on the fused LoRA path), ``in_features``
    the layer's input width that keys the 1D table."""
    kwargs_2d = matmul_kwargs_2d(rows, k if k is not None else in_features, out_features, False)
    if kwargs_2d is not None:
        return kwargs_2d
    spec = FORWARD_PROGRAM_CONFIG.get((in_features, out_features))
    grid = ttml.autograd.AutoContext.get_instance().get_device().compute_with_storage_grid_size()
    k_tiles = (k if k is not None else in_features) // 32
    if (
        spec is None
        or rows % 32 != 0
        or (rows // 32) % spec["out_block_h"] != 0
        or k_tiles % spec["in0_block_w"] != 0
        or spec["grid"][0] > grid.x
        or spec["grid"][1] > grid.y
    ):
        return {"core_grid": forward_core_grid(in_features, out_features)}
    return {
        "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=spec["grid"],
            in0_block_w=spec["in0_block_w"],
            out_subblock_h=spec["out_subblock_h"],
            out_subblock_w=spec["out_subblock_w"],
            out_block_h=spec["out_block_h"],
            out_block_w=spec["out_block_w"],
            per_core_M=rows // 32,
            per_core_N=spec["per_core_N"],
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    }


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
            **forward_matmul_kwargs(in_features, out_features, x.get_value().shape[-2]),
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
            w,
            grad_output,
            transpose_b=True,
            compute_kernel_config=_compute_config(),
            **dx_matmul_kwargs(w.shape[-2], w.shape[-1], grad_output.shape[-2]),
        )
        dx = ttnn.transpose(dx_t, -2, -1)
        ttnn.deallocate(dx_t)
        # The weight is frozen (no dW); a bias here would be frozen too (the Mistral linears have none).
        # The Function expects one gradient per tensor input, and a None bias is not a tensor input.
        return (dx, None, None) if ctx.has_bias else (dx, None)


def frozen_linear_t(x, weight_t, bias, shape):
    return FrozenLinearT.apply(x, weight_t, bias, shape)


def configured_base_shape(weight, rows: int) -> tuple[int, int] | None:
    """(in, out) of the frozen weight shard ``[.., out, in]`` whose forward ``[rows, in] x [out, in]^T`` has a swept
    2D config in :data:`PROGRAM_CONFIG_2D` (iteration 46: the attention projections at the TP=4 shapes), else None."""
    for in_features, out_features in shard_candidates(weight):
        if matmul_kwargs_2d(rows, in_features, out_features, True) is not None:
            return in_features, out_features
    return None


class ConfiguredFrozenLinear(Function):
    """``y = x @ W^T`` for a frozen ``W = [1, 1, out, in]`` kept in its natural layout, through the swept 2D
    program configs (iteration 46): the composite LoRA path's base projection where ttnn's default choice loses
    7-11 % at the small TP attention shapes. No weight gradient; ``dx = g @ W`` on the swept config of that shape
    (or the full grid)."""

    @staticmethod
    def forward(ctx, x, weight, bias, shape):
        """``shape`` is the (in, out) of the local shard from :func:`configured_base_shape`."""
        w = weight.get_value()
        in_features, out_features = shape
        ctx.save_for_backward(w)
        ctx.has_bias = bias is not None
        ctx.shape = shape
        # The first block's qkv input (the gathered norm of the frozen embedding) needs no gradient: the ttml linear
        # skips that dx, so does this (the first profile computed it: 3 -> 4 dX matmuls per 4 layers).
        ctx.x_needs_grad = x.get_requires_grad()
        y = ttnn.matmul(
            x.get_value(),
            w,
            transpose_b=True,
            compute_kernel_config=_compute_config(),
            **matmul_kwargs_2d(int(x.get_value().shape[-2]), in_features, out_features, True),
        )
        if bias is not None:
            y = ttnn.add(y, bias.get_value())
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        ctx.save_for_backward()
        if not ctx.x_needs_grad:
            return (None, None, None) if ctx.has_bias else (None, None)
        in_features, out_features = ctx.shape
        rows = int(grad_output.shape[-2])
        dx = ttnn.matmul(
            grad_output,
            w,
            compute_kernel_config=_compute_config(),
            **(matmul_kwargs_2d(rows, out_features, in_features, False) or {"core_grid": full_core_grid()}),
        )
        return (dx, None, None) if ctx.has_bias else (dx, None)


def configured_linear(x, weight, bias, shape):
    return ConfiguredFrozenLinear.apply(x, weight, bias, shape)


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
