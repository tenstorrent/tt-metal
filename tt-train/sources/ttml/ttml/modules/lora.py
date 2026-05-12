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
        base = ttml.ops.linear.linear(x, self.weight.tensor, bias)
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
        lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        return base + lora_update * self.scaling


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
        base = ttml.ops.linear.linear(x, self.weight.tensor, bias_t)
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
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
        lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.gather_output:
            lora_update = ttml.ops.distributed.all_gather(
                lora_update, 3, self.cluster_axis, ttml.ops.distributed.GradOutputType.REPLICATED
            )
        return base + lora_update * self.scaling


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
        base = ttml.ops.linear.linear(x, self.weight.tensor, None)
        base = ttml.ops.distributed.all_reduce(base, self.input_is_parallel, self.cluster_axis)
        if self.bias is not None:
            base = ttml.ops.binary.add(base, self.bias.tensor)
        # lora_A is row-sharded (each device sees a slice of in_features), so
        # an all_reduce is needed after lora_A to sum partial projections before
        # the replicated lora_B can be applied.
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
        h = ttml.ops.distributed.all_reduce(h, self.input_is_parallel, self.cluster_axis)
        lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        return base + lora_update * self.scaling


_LORA_MODULE_MAP: dict[type, type] = {
    ColumnParallelLinear: LoraColumnParallelLinear,
    RowParallelLinear: LoraRowParallelLinear,
    LinearLayer: LoraLinear,
}


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

        for _, tensor in model.named_parameters():
            tensor.set_requires_grad(False)

        patterns = [re.compile(p) for p in config.target_modules]
        self._inject(model, "", patterns, config)

        if config.trainable_modules:
            self._unfreeze_trainable(model, config.trainable_modules)

    def _inject(
        self,
        module: AbstractModuleBase,
        prefix: str,
        patterns: list[re.Pattern],
        config: LoraConfig,
    ) -> None:
        """Recursively replace matching linear layers with their LoRA wrappers.

        ModuleList and ModuleDict require index/key-based assignment instead of
        ``setattr`` because their ``__setitem__`` maintains internal bookkeeping.
        """
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
            elif isinstance(child, AbstractModuleBase):
                self._inject(child, full_name, patterns, config)

    @staticmethod
    def _unfreeze_trainable(model: AbstractModuleBase, trainable_modules: list[str]) -> None:
        """Unfreeze parameters whose full path starts with any of the given prefixes."""
        for param_path, tensor in model.parameters().items():
            if any(prefix in param_path for prefix in trainable_modules):
                tensor.set_requires_grad(True)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
