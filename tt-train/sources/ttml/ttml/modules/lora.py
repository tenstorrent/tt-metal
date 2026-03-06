# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ml_dtypes

import ttnn
import ttml

from .linear import LinearLayer
from .module_base import AbstractModuleBase, ModuleDict, ModuleList
from .parameter import Parameter
from .._ttml.modules import RunMode


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = field(default_factory=list)
    use_rslora: bool = False
    is_bias_trainable: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    lora_dropout: float = 0.0


def _create_lora_A(in_features: int, rank: int):
    """Initialize LoRA A (down-projection) with kaiming uniform, shape (1, 1, rank, in_features)."""
    bound = 1.0 / np.sqrt(in_features)
    weight_np = np.random.uniform(
        low=-bound,
        high=bound,
        size=(1, 1, rank, in_features),
    ).astype(ml_dtypes.bfloat16)
    return ttml.autograd.Tensor.from_numpy(weight_np, layout=ttnn.Layout.TILE)


def _create_lora_B(rank: int, out_features: int):
    """Initialize LoRA B (up-projection) with zeros, shape (1, 1, out_features, rank)."""
    weight_np = np.zeros((1, 1, out_features, rank), dtype=ml_dtypes.bfloat16)
    return ttml.autograd.Tensor.from_numpy(weight_np, layout=ttnn.Layout.TILE)


class LoraLinear(AbstractModuleBase):
    """Low-Rank Adaptation wrapper around a frozen LinearLayer.

    Computes: linear(x, weight, bias) + linear(linear(x, lora_A), lora_B) * scaling
    """

    def __init__(self, linear: LinearLayer, config: LoraConfig) -> None:
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = linear.weight
        self.weight.tensor.set_requires_grad(False)

        self.bias = linear.bias
        if self.bias is not None:
            self.bias.tensor.set_requires_grad(config.is_bias_trainable)

        self.lora_A = Parameter(_create_lora_A(self.in_features, config.rank))
        self.lora_B = Parameter(_create_lora_B(config.rank, self.out_features))

        self.dropout_prob = config.lora_dropout

        self.scaling = (
            config.alpha / math.sqrt(config.rank)
            if config.use_rslora
            else config.alpha / config.rank
        )

    def forward(self, x: Any) -> Any:
        bias = self.bias.tensor if self.bias is not None else None
        base = ttml.ops.linear.linear(x, self.weight.tensor, bias)
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
        lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        return base + lora_update * self.scaling


class LoraModel(AbstractModuleBase):
    """Wraps a model, freezes its parameters, and injects LoraLinear modules."""

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
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, LinearLayer) and any(
                p.search(full_name) for p in patterns
            ):
                lora_linear = LoraLinear(child, config)
                if isinstance(module, ModuleList):
                    module[int(name)] = lora_linear
                elif isinstance(module, ModuleDict):
                    module[name] = lora_linear
                else:
                    setattr(module, name, lora_linear)
            elif isinstance(child, AbstractModuleBase):
                self._inject(child, full_name, patterns, config)

    @staticmethod
    def _unfreeze_trainable(
        model: AbstractModuleBase, trainable_modules: list[str]
    ) -> None:
        """Unfreeze parameters whose full path starts with any of the given prefixes."""
        for param_path, tensor in model.parameters().items():
            if any(prefix in param_path for prefix in trainable_modules):
                tensor.set_requires_grad(True)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
