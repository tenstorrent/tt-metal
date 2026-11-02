# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import ttnn
import ttml

from .linear import LinearLayer, ColumnParallelLinear, RowParallelLinear
from .module_base import AbstractModuleBase, ModuleDict, ModuleList
from .parameter import Parameter
from _ttml.modules import RunMode
from ..autograd.function import Function


def lora_matmul_config():
    """Compute config matching ttml's linear (HiFi4, fp32 accumulation)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )


# `x @ A^T` / `g @ B` produce rank-wide outputs ([T, r], one tile column): ttnn's matmul parallelises over output
# tiles and runs them on T/32 cores (33 of 110 at T=1024, 75-120 us for K=5120-8192). `skinny_matmul` splits K over
# the cores instead (iteration 10); below this K the two are equal within dispatch noise.
SKINNY_MIN_K = 2048
SKINNY_CONTRACT_MAX_N = (
    5120  # token-major big operands below this N go to skinny_contract (qkv dB 1536: 0.089 -> 0.065 ms per 4 layers
    # in the step, iteration 40); at the out-projection's 5120 the op reads 0.138 against ttnn's 0.119 + 0.010
    # transpose in device time (standalone 36 vs 47 us wall counts a launch, not device time: review round 12), so
    # that shape keeps ttnn; the transposed operands (iterations 38-40) always take the op
)


def lowrank_update(acc, a, b, scale=1.0, transpose_b=False):
    """``acc + scale * (a @ b)`` for a rank <= 32 ``a`` (``b`` is ``[N, rank]`` with ``transpose_b``): one metal op
    (`ttml.ops.metal.lowrank_update`, iteration 12) in place of ttnn's 33-core matmul plus a separate add when the
    shapes fit, else the two ops."""
    k = a.shape[-1]
    if (
        k <= 32
        and acc.shape[-2] <= 4096
        and b.shape[0] == 1
        and b.shape[1] == 1
        and list(acc.shape)[:3] == list(a.shape)[:3]
    ):
        return ttml.ops.metal.lowrank_update(acc, a, b, scale, transpose_b)
    update = ttnn.matmul(a, b, transpose_b=transpose_b, compute_kernel_config=lora_matmul_config())
    return ttnn.addalpha(acc, update, scale)


def skinny_contract(small, big, scale=1.0, transpose_out=False, big_transposed=False, small_transposed=False):
    """``scale * small^T @ big`` over the token dimension (``small`` ``[T, r <= 32]``, or ``[r, T]`` with
    ``small_transposed``; with ``big_transposed`` the second operand is stored ``[N, T]``).

    Routing: a transposed operand (iterations 38-40: the transposed MLP interior's dA / dB on ``g^T`` / ``x^T``)
    always takes the all-core op (its small operand multicast from one core, every core on a contiguous column range,
    iteration 40: the fastest of the measured routes, `O/iter40_lora_products.log`). Token-major operands: a
    narrow ``transpose_out`` product (``N < SKINNY_CONTRACT_MAX_N``: the qkv dB) takes the op; the rest (the
    out-projection dB at N = 5120, any ``[r, N]`` product) ttnn's transposed-a matmul plus a transpose."""
    r, T = (small.shape[-2], small.shape[-1]) if small_transposed else (small.shape[-1], small.shape[-2])
    N = big.shape[-2] if big_transposed else big.shape[-1]
    cfg = lora_matmul_config()
    if big_transposed or small_transposed:  # the op's native operands (iteration 40: faster than ttnn's routes)
        return ttml.ops.metal.skinny_contract(
            small, big, scale, transpose_out, big_transposed=big_transposed, small_transposed=small_transposed
        )
    if (
        transpose_out
        and N < SKINNY_CONTRACT_MAX_N
        and r <= 16
        and T % 32 == 0
        and T <= 4096
        and small.shape[0] == 1
        and small.shape[1] == 1
        and big.shape[0] == 1
        and big.shape[1] == 1
    ):
        return ttml.ops.metal.skinny_contract(small, big, scale, transpose_out)
    out = ttnn.matmul(small, big, transpose_a=True, transpose_b=big_transposed, compute_kernel_config=cfg)  # [r, N]
    if scale != 1.0:
        out = ttnn.multiply(out, scale)
    return ttnn.transpose(out, -2, -1) if transpose_out else out


def skinny_matmul_fits(k, n, b):
    """Whether the K-split op takes a ``[.., M, >= k] @ b`` product with ``n`` output columns."""
    return n <= 64 and k >= SKINNY_MIN_K and k % 32 == 0 and b.shape[0] == 1 and b.shape[1] == 1


def skinny_matmul(
    a, b, transpose_b=False, tail_out=None, tail_col=0, tail_tiles=0, x_transposed=False, transpose_out=False
):
    """``a @ (b^T if transpose_b else b)`` (HiFi4, fp32 accumulation) for a rank-wide ``b``: the K-split op when it
    pays, ttnn's matmul otherwise. ``a`` may be wider than ``b``'s K (only its first K columns enter). With
    ``tail_out`` the op also writes the product into tile column ``tail_col`` of that tensor in place, followed by
    zero tiles up to ``tail_tiles`` columns (iteration 24: the LoRA ``h`` into the wide input buffer's tail); the
    caller checks :func:`skinny_matmul_fits` first. ``x_transposed`` (iteration 40): ``a`` is stored ``[K', M]``
    (the transposed wide buffer) and the result comes out transposed (``[N, M]``, the tail rows of ``tail_out``)."""
    k = b.shape[-1] if transpose_b else b.shape[-2]
    n = b.shape[-2] if transpose_b else b.shape[-1]
    if skinny_matmul_fits(k, n, b):
        return ttml.ops.metal.skinny_matmul(
            a, b, transpose_b, tail_out, tail_col, tail_tiles, x_transposed, transpose_out or x_transposed
        )
    assert (
        tail_out is None and not x_transposed
    ), "skinny_matmul: the tail write / transposed buffer need the K-split op"
    return ttnn.matmul(a, b, transpose_b=transpose_b, compute_kernel_config=lora_matmul_config())


class LoraApply(Function):
    """``base + scaling * (x @ A^T) @ (B * mask)^T`` with a TP-correct backward.

    One A matmul, one B matmul and one ``addalpha`` per pass; the naive
    ``base + linear(linear(x, A), B) * scaling`` spends a scale pass and an add
    pass over the full ``[T, out]`` update, and a scale pass over its gradient.
    Here the scale rides inside ``addalpha`` forward and on the small ``[T, r]``
    / ``[out, r]`` tensors backward.

    ``mask`` (optional 0/1 of B's shape) keeps B block-structured when several
    adapters are packed into one B; it is applied to B forward and to dB backward.

    ``reduce_axis``: when B is column-sharded over a mesh axis (a column-parallel
    layer), each chip's ``dh = scaling * g_c @ B_c`` is only that chip's share of
    the output columns. ``dx = dh_c @ A`` must stay partial (the input's own
    broadcast all-reduces it), but ``dA = dh^T @ x`` needs the full ``dh``, so it
    is all-reduced over ``reduce_axis`` first (``[T, r]``, a few KB). Without
    this every chip trains a different A and the replicated copies drift apart.
    """

    @staticmethod
    def forward(ctx, base, x, lora_A, lora_B, mask, scaling, reduce_axis):
        cfg = lora_matmul_config()
        x_val, a_val = x.get_value(), lora_A.get_value()
        b_val = lora_B.get_value() if mask is None else ttnn.multiply(lora_B.get_value(), mask)
        h = skinny_matmul(x_val, a_val, transpose_b=True)
        ctx.save_for_backward(x_val, a_val, b_val, h)
        ctx.mask, ctx.scaling, ctx.reduce_axis = mask, scaling, reduce_axis
        return lowrank_update(base.get_value(), h, b_val, scaling, transpose_b=True)

    @staticmethod
    def backward(ctx, grad_output):
        x_val, a_val, b_val, h = ctx.saved_tensors
        ctx.save_for_backward()
        cfg = lora_matmul_config()
        s = ctx.scaling
        grad_h = ttnn.multiply(skinny_matmul(grad_output, b_val), s)
        grad_b = skinny_contract(h, grad_output, s, transpose_out=True)  # s * g^T h -> [out, r]
        if ctx.mask is not None:
            grad_b = ttnn.multiply(grad_b, ctx.mask)
        grad_x = ttnn.matmul(grad_h, a_val, compute_kernel_config=cfg)
        if ctx.reduce_axis is not None:
            grad_h = ttml.core.distributed.all_reduce(grad_h, ctx.reduce_axis)
        grad_a = skinny_contract(grad_h, x_val)  # dh^T x -> [r, in]
        return grad_output, grad_x, grad_a, grad_b


class _LoraUpdateAdd(Function):
    """``base + scaling * (h @ B^T)``: the B half of :class:`LoraApply`, for a precomputed (all-reduced) ``h``."""

    @staticmethod
    def forward(ctx, base, h, lora_B, scaling):
        h_val, b_val = h.get_value(), lora_B.get_value()
        ctx.save_for_backward(h_val, b_val)
        ctx.scaling = scaling
        return lowrank_update(base.get_value(), h_val, b_val, scaling, transpose_b=True)

    @staticmethod
    def backward(ctx, grad_output):
        h_val, b_val = ctx.saved_tensors
        ctx.save_for_backward()
        cfg = lora_matmul_config()
        grad_h = ttnn.multiply(skinny_matmul(grad_output, b_val), ctx.scaling)
        grad_b = skinny_contract(h_val, grad_output, ctx.scaling, transpose_out=True)  # s * g^T h -> [out, r]
        return grad_output, grad_h, grad_b


def lora_apply(base, x, lora_A, lora_B, scaling, *, mask=None, reduce_axis=None):
    """``base + scaling * linear(linear(x, lora_A), lora_B)`` in three ops, TP-correct (see :class:`LoraApply`)."""
    return LoraApply.apply(base, x, lora_A, lora_B, mask, scaling, reduce_axis)


@dataclass
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adaptation) injection.

    Attributes:
        rank: Rank of the low-rank decomposition (A and B matrices).
        alpha: Scaling numerator.  The effective scaling factor is
            ``alpha / sqrt(rank)`` when *use_rslora* is ``True``, else
            ``alpha / rank``.
        target_modules: List of regex patterns matched against fully-qualified
            module names.  Matching linear layers are replaced with their LoRA
            counterparts.
        use_rslora: Use rank-stabilized LoRA scaling (``alpha / sqrt(rank)``)
            instead of the original ``alpha / rank``.
        is_bias_trainable: If ``True``, bias parameters in matched layers remain
            trainable after injection.
        trainable_modules: List of name substrings; any parameter whose path
            contains one of these substrings is unfrozen after injection.  This
            is useful for keeping specific non-LoRA parameters trainable
            (e.g. layer norms).
        lora_dropout: Dropout probability applied to the LoRA input during
            training.
    """

    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = field(default_factory=list)
    use_rslora: bool = False
    is_bias_trainable: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    lora_dropout: float = 0.0
    verbose: bool = False


def _create_lora_A(in_features: int, rank: int, mapper=None):
    """Initialize LoRA A (down-projection) with kaiming uniform, shape (1, 1, rank, in_features)."""
    bound = 1.0 / np.sqrt(in_features)
    weight_np = np.random.uniform(
        low=-bound,
        high=bound,
        size=(1, 1, rank, in_features),
    ).astype(np.float32)
    return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _create_lora_B(rank: int, out_features: int, mapper=None):
    """Initialize LoRA B (up-projection) with zeros, shape (1, 1, out_features, rank)."""
    weight_np = np.zeros((1, 1, out_features, rank), dtype=np.float32)
    return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _base_linear(module, x, bias):
    """The frozen base projection of a LoRA wrapper: the ttml linear, or the transposed-weight matmul once
    :func:`ttml.modules.frozen_linear.transpose_frozen_weights` has converted it."""
    if getattr(module, "weight_transposed", False):
        from .frozen_linear import frozen_linear_t

        return frozen_linear_t(x, module.weight.tensor, bias, module.frozen_shape)
    return ttml.ops.linear.linear(x, module.weight.tensor, bias)


class LoraLinear(AbstractModuleBase):
    """LinearLayer with frozen base weights and a trainable low-rank path.

    Computes: linear(x, weight, bias) + linear(linear(x, lora_A), lora_B) * scaling
    """

    def __init__(self, base_layer: LinearLayer, config: LoraConfig):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.dropout_prob = config.lora_dropout
        self.scaling = config.alpha / math.sqrt(config.rank) if config.use_rslora else config.alpha / config.rank

        self.weight = base_layer.weight
        self.bias = base_layer.bias

        # Freeze weights
        self.weight.tensor.set_requires_grad(False)
        if self.bias is not None:
            self.bias.tensor.set_requires_grad(config.is_bias_trainable)

        self.lora_A = Parameter(_create_lora_A(self.in_features, config.rank))
        self.lora_B = Parameter(_create_lora_B(config.rank, self.out_features))

    def forward(self, x):
        bias = self.bias.tensor if self.bias is not None else None
        if getattr(self, "lora_fused", False) and not (
            self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0
        ):
            from .fused_lora import fused_lora_linear_t

            return fused_lora_linear_t(
                x,
                self.weight.tensor,
                self.lora_A.tensor,
                self.lora_B.tensor,
                self.scaling,
                self.frozen_shape,
                weight_cat_fwd=getattr(self, "weight_cat_fwd", None),
                transposed_output=getattr(self, "transposed_output", False),
                lowrank_sink=self,
            )
        base = _base_linear(self, x, bias)
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        return lora_apply(base, lora_input, self.lora_A.tensor, self.lora_B.tensor, self.scaling)


class LoraColumnParallelLinear(AbstractModuleBase):
    """ColumnParallelLinear with frozen base weights and a trainable low-rank path.

    lora_A is replicated (full in_features).
    lora_B is column-sharded on dim 2 (out_features split across TP devices).
    """

    def __init__(self, base_layer: ColumnParallelLinear, config: LoraConfig):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.gather_output = base_layer.gather_output
        self.axis_name = base_layer.axis_name
        self.cluster_axis = base_layer.cluster_axis

        self.dropout_prob = config.lora_dropout
        self.scaling = config.alpha / math.sqrt(config.rank) if config.use_rslora else config.alpha / config.rank

        self.weight = base_layer.weight
        self.bias = base_layer.bias

        # Freeze weights
        self.weight.tensor.set_requires_grad(False)
        if self.bias is not None:
            self.bias.tensor.set_requires_grad(config.is_bias_trainable)

        self.lora_A = Parameter(_create_lora_A(self.in_features, config.rank))
        lora_B_mapper = ttml.mesh().axis_mapper(self.axis_name, tdim=2)
        self.lora_B = Parameter(_create_lora_B(config.rank, self.out_features, mapper=lora_B_mapper))

    def forward(self, x):
        bias_t = self.bias.tensor if self.bias is not None else None
        x = ttml.ops.distributed.broadcast(x, self.cluster_axis)
        if getattr(self, "lora_fused", False) and not (
            self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0
        ):
            from .fused_lora import fused_lora_linear_t

            # B is column-sharded: the tail is the local shard; dA needs the all-reduced dh, or, in batched mode,
            # every rank keeps its partial dA and `reduce_partial_lora_grads` sums them once per step.
            return fused_lora_linear_t(
                x,
                self.weight.tensor,
                self.lora_A.tensor,
                self.lora_B.tensor,
                self.scaling,
                self.frozen_shape,
                reduce_axis=None if getattr(self, "tp_partial_grads", None) else self.cluster_axis,
                weight_cat_fwd=getattr(self, "weight_cat_fwd", None),
                transposed_output=getattr(self, "transposed_output", False),
                lowrank_sink=self,
            )
        base = _base_linear(self, x, bias_t)
        if self.gather_output:
            base = ttml.ops.distributed.all_gather(
                base, 3, self.cluster_axis, ttml.ops.distributed.GradOutputType.REPLICATED
            )
        # lora_A is replicated so it operates on the full input; lora_B is
        # column-sharded, producing a sharded update that must be gathered
        # whenever the base path is gathered.
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        if self.gather_output:
            h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
            lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
            lora_update = ttml.ops.distributed.all_gather(
                lora_update, 3, self.cluster_axis, ttml.ops.distributed.GradOutputType.REPLICATED
            )
            return base + lora_update * self.scaling
        # lora_B is column-sharded: dA needs the all-reduced dh (see LoraApply).
        # Batched mode (`fused_lora.reduce_partial_lora_grads`): dA stays the rank's partial, no per-layer all-reduce.
        return lora_apply(
            base,
            lora_input,
            self.lora_A.tensor,
            self.lora_B.tensor,
            self.scaling,
            reduce_axis=None if getattr(self, "tp_partial_grads", None) else self.cluster_axis,
        )


class LoraRowParallelLinear(AbstractModuleBase):
    """RowParallelLinear with frozen base weights and a trainable low-rank path.

    lora_A is row-sharded on dim 3 (in_features split across TP devices).
    lora_B is replicated (full out_features).
    """

    def __init__(self, base_layer: RowParallelLinear, config: LoraConfig):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.input_is_parallel = base_layer.input_is_parallel
        self.axis_name = base_layer.axis_name
        self.cluster_axis = base_layer.cluster_axis

        self.dropout_prob = config.lora_dropout
        self.scaling = config.alpha / math.sqrt(config.rank) if config.use_rslora else config.alpha / config.rank

        self.weight = base_layer.weight
        self.bias = base_layer.bias

        # Freeze weights
        self.weight.tensor.set_requires_grad(False)
        if self.bias is not None:
            self.bias.tensor.set_requires_grad(config.is_bias_trainable)

        lora_A_mapper = ttml.mesh().axis_mapper(self.axis_name, tdim=3)
        self.lora_A = Parameter(_create_lora_A(self.in_features, config.rank, mapper=lora_A_mapper))
        self.lora_B = Parameter(_create_lora_B(config.rank, self.out_features))

    def forward(self, x):
        if not self.input_is_parallel:
            x = ttml.ops.distributed.scatter(x, 3, self.cluster_axis)
        if getattr(self, "lora_fused", False) and not (
            self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0
        ):
            from .fused_lora import fused_lora_linear_t

            if getattr(self, "tp_partial_grads", None):
                # Batched mode: h stays the rank's partial sum, the output all-reduce sums s * h_r B^T over the
                # ranks (linear in h), dh = s g B is full on every rank, and dB_r = s g^T h_r is a partial that
                # `reduce_partial_lora_grads` sums once per step.
                out = fused_lora_linear_t(
                    x,
                    self.weight.tensor,
                    self.lora_A.tensor,
                    self.lora_B.tensor,
                    self.scaling,
                    self.frozen_shape,
                    weight_cat_fwd=getattr(self, "weight_cat_fwd", None),
                    transposed_output=getattr(self, "transposed_output", False),
                    lowrank_sink=self,
                )
            else:
                # A is row-sharded (h partial, all-reduced inside), B replicated: the tail carries s/tp * B^T so
                # the output all-reduce sums the update once, and the tail gradient is scaled back by tp.
                tp = ttml.mesh().axis_size(self.axis_name)
                out = fused_lora_linear_t(
                    x,
                    self.weight.tensor,
                    self.lora_A.tensor,
                    self.lora_B.tensor,
                    self.scaling,
                    self.frozen_shape,
                    tail_scale=1.0 / tp,
                    dh_scale=float(tp),
                    h_reduce_axis=self.cluster_axis,
                    weight_cat_fwd=getattr(self, "weight_cat_fwd", None),
                    transposed_output=getattr(self, "transposed_output", False),
                    lowrank_sink=self,
                )
            out = ttml.ops.distributed.all_reduce(out, self.input_is_parallel, self.cluster_axis)
            if self.bias is not None:
                out = ttml.ops.binary.add(out, self.bias.tensor)
            return out
        base = _base_linear(self, x, None)
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        # lora_A is row-sharded (each device sees a slice of in_features), so h is a partial sum.
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
        if getattr(self, "tp_partial_grads", None):
            # Batched mode: the partial update s * h_r B^T joins the partial base before the one output all-reduce
            # (linear in h); dh = s g B is full on every rank, dB_r = s g^T h_r is a partial that
            # `fused_lora.reduce_partial_lora_grads` sums once per step.
            out = _LoraUpdateAdd.apply(base, h, self.lora_B.tensor, self.scaling)
            out = ttml.ops.distributed.all_reduce(out, self.input_is_parallel, self.cluster_axis)
            if self.bias is not None:
                out = ttml.ops.binary.add(out, self.bias.tensor)
            return out
        base = ttml.ops.distributed.all_reduce(base, self.input_is_parallel, self.cluster_axis)
        if self.bias is not None:
            base = ttml.ops.binary.add(base, self.bias.tensor)
        # all-reduce h before B (B is replicated and then sees the full h, so dB is already complete).
        h = ttml.ops.distributed.all_reduce(h, self.input_is_parallel, self.cluster_axis)
        return _LoraUpdateAdd.apply(base, h, self.lora_B.tensor, self.scaling)


_LORA_MODULE_MAP: dict[type, type] = {
    ColumnParallelLinear: LoraColumnParallelLinear,
    RowParallelLinear: LoraRowParallelLinear,
    LinearLayer: LoraLinear,
}


def register_lora_module(base_cls: type, lora_cls: type) -> None:
    """Let :class:`LoraModel` wrap ``base_cls`` layers with ``lora_cls``.

    ``lora_cls(base_layer, config)`` must freeze the base weights and expose the
    adapters as ``lora_*`` parameters, like the built-in wrappers above. Model
    packages use this for layer types of their own (e.g. a fused multi-projection
    linear) without ``LoraModel`` having to know them.
    """
    _LORA_MODULE_MAP[base_cls] = lora_cls


class LoraModel(AbstractModuleBase):
    """Wraps a model, freezes its parameters, and injects LoRA adapters.

    Linear layers whose fully-qualified name matches any pattern in
    ``config.target_modules`` are replaced with their LoRA counterpart via
    ``_LORA_MODULE_MAP`` (``LinearLayer`` -> ``LoraLinear``,
    ``ColumnParallelLinear`` -> ``LoraColumnParallelLinear``, etc.).
    """

    def __init__(self, model: AbstractModuleBase, config: LoraConfig) -> None:
        super().__init__()
        self.model = model
        verbose = config.verbose

        if verbose and not config.target_modules:
            print("[LoRA] target_modules is empty. No injection will occur.")

        if verbose:
            print(
                f"[LoRA] Config: rank={config.rank}, alpha={config.alpha}, "
                f"targets={config.target_modules}, dropout={config.lora_dropout}, "
                f"use_rslora={config.use_rslora}, is_bias_trainable={config.is_bias_trainable}"
            )

        if verbose:
            print("[LoRA] Freezing weights")
        for _, tensor in model.named_parameters():
            tensor.set_requires_grad(False)

        patterns = [re.compile(p) for p in config.target_modules]
        injected = self._inject(model, "", patterns, config)
        if verbose:
            print("[LoRA] Injecting LoRA")
            for name in injected:
                print(f"  - {name}")

        unmatched = [p.pattern for p in patterns if not any(p.search(name) for name in injected)]
        if unmatched:
            raise ValueError(f"[LoRA] target_modules pattern(s) matched no LinearLayer in the model: {unmatched}")

        if config.trainable_modules:
            self._unfreeze_trainable(model, config.trainable_modules)

        if verbose:
            unfrozen = [name for name, t in model.parameters().items() if t.get_requires_grad()]
            print("[LoRA] Unfreezing weights:")
            for name in unfrozen:
                print(f"  - {name}")

    def _inject(
        self,
        module: AbstractModuleBase,
        prefix: str,
        patterns: list[re.Pattern],
        config: LoraConfig,
    ) -> list[str]:
        """Recursively replace matching linear layers with their LoRA wrappers.

        ModuleList and ModuleDict require index/key-based assignment instead of
        ``setattr`` because their ``__setitem__`` maintains internal bookkeeping.
        """
        injected: list[str] = []
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if any(p.search(full_name) for p in patterns):
                lora_cls = _LORA_MODULE_MAP.get(type(child))
                if lora_cls is None:
                    raise RuntimeError(
                        f"Cannot apply LoRA to '{full_name}': "
                        f"'{type(child).__name__}' is not a supported layer type. "
                        f"Supported: {', '.join(cls.__name__ for cls in _LORA_MODULE_MAP)}"
                    )
                lora_layer = lora_cls(child, config)

                if isinstance(module, ModuleList):
                    module[int(name)] = lora_layer
                elif isinstance(module, ModuleDict):
                    module[name] = lora_layer
                else:
                    setattr(module, name, lora_layer)
                injected.append(full_name)
            elif isinstance(child, AbstractModuleBase):
                injected.extend(self._inject(child, full_name, patterns, config))
        return injected

    @staticmethod
    def _unfreeze_trainable(model: AbstractModuleBase, trainable_modules: list[str]) -> None:
        """Unfreeze parameters whose full path starts with any of the given prefixes."""
        for param_path, tensor in model.parameters().items():
            if any(prefix in param_path for prefix in trainable_modules):
                tensor.set_requires_grad(True)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
