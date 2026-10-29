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
        h = ttnn.matmul(x_val, a_val, transpose_b=True, compute_kernel_config=cfg)
        update = ttnn.matmul(h, b_val, transpose_b=True, compute_kernel_config=cfg)
        ctx.save_for_backward(x_val, a_val, b_val, h)
        ctx.mask, ctx.scaling, ctx.reduce_axis = mask, scaling, reduce_axis
        return ttnn.addalpha(base.get_value(), update, scaling)

    @staticmethod
    def backward(ctx, grad_output):
        x_val, a_val, b_val, h = ctx.saved_tensors
        ctx.save_for_backward()
        cfg = lora_matmul_config()
        s = ctx.scaling
        grad_h = ttnn.multiply(ttnn.matmul(grad_output, b_val, compute_kernel_config=cfg), s)
        grad_b = ttnn.multiply(ttnn.matmul(grad_output, h, transpose_a=True, compute_kernel_config=cfg), s)
        if ctx.mask is not None:
            grad_b = ttnn.multiply(grad_b, ctx.mask)
        grad_x = ttnn.matmul(grad_h, a_val, compute_kernel_config=cfg)
        if ctx.reduce_axis is not None:
            grad_h = ttml.core.distributed.all_reduce(grad_h, ctx.reduce_axis)
        grad_a = ttnn.matmul(grad_h, x_val, transpose_a=True, compute_kernel_config=cfg)
        return grad_output, grad_x, grad_a, grad_b


class _LoraUpdateAdd(Function):
    """``base + scaling * (h @ B^T)``: the B half of :class:`LoraApply`, for a precomputed (all-reduced) ``h``."""

    @staticmethod
    def forward(ctx, base, h, lora_B, scaling):
        h_val, b_val = h.get_value(), lora_B.get_value()
        ctx.save_for_backward(h_val, b_val)
        ctx.scaling = scaling
        update = ttnn.matmul(h_val, b_val, transpose_b=True, compute_kernel_config=lora_matmul_config())
        return ttnn.addalpha(base.get_value(), update, scaling)

    @staticmethod
    def backward(ctx, grad_output):
        h_val, b_val = ctx.saved_tensors
        ctx.save_for_backward()
        cfg = lora_matmul_config()
        grad_h = ttnn.multiply(ttnn.matmul(grad_output, b_val, compute_kernel_config=cfg), ctx.scaling)
        grad_b = ttnn.multiply(
            ttnn.matmul(grad_output, h_val, transpose_a=True, compute_kernel_config=cfg), ctx.scaling
        )
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
                x, self.weight.tensor, self.lora_A.tensor, self.lora_B.tensor, self.scaling, self.frozen_shape
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

            # B is column-sharded: the tail is the local shard; dA needs the all-reduced dh.
            return fused_lora_linear_t(
                x,
                self.weight.tensor,
                self.lora_A.tensor,
                self.lora_B.tensor,
                self.scaling,
                self.frozen_shape,
                reduce_axis=self.cluster_axis,
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
        return lora_apply(
            base, lora_input, self.lora_A.tensor, self.lora_B.tensor, self.scaling, reduce_axis=self.cluster_axis
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

            # A is row-sharded (h partial, all-reduced inside), B replicated: the tail carries s/tp * B^T so the
            # output all-reduce sums the update once, and the tail gradient is scaled back by tp.
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
            )
            out = ttml.ops.distributed.all_reduce(out, self.input_is_parallel, self.cluster_axis)
            if self.bias is not None:
                out = ttml.ops.binary.add(out, self.bias.tensor)
            return out
        base = _base_linear(self, x, None)
        base = ttml.ops.distributed.all_reduce(base, self.input_is_parallel, self.cluster_axis)
        if self.bias is not None:
            base = ttml.ops.binary.add(base, self.bias.tensor)
        # lora_A is row-sharded (each device sees a slice of in_features), so
        # an all_reduce is needed after lora_A to sum partial projections before
        # the replicated lora_B can be applied.
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        # A is row-sharded, so h is a partial sum: all-reduce it before B (B is
        # replicated and then sees the full h, so dB is already complete).
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
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
