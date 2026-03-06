# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import re
from typing import Any

import numpy as np
import ml_dtypes

import ttnn
import ttml

from .adapter import Adapter, ForwardInvocation
from .linear import LinearLayer
from .parameter import Parameter


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


class LoRA(Adapter):
    """Low-Rank Adaptation: adds lora_B(lora_A(x)) * scaling to the base output."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        use_rslora: bool = False,
    ) -> None:
        self.scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank

        self.lora_A = Parameter(_create_lora_A(in_features, rank))
        self.lora_B = Parameter(_create_lora_B(rank, out_features))

    def parameters(self) -> dict:
        """Return lora_A and lora_B as trainable parameters."""
        return {"lora_A": self.lora_A, "lora_B": self.lora_B}

    def __call__(self, fwd: ForwardInvocation) -> Any:
        """Add scaled low-rank update to the base output."""
        x = fwd.args[0]
        h = ttml.ops.linear.linear(x, self.lora_A.tensor, None)
        lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        return fwd.output + lora_update * self.scaling


def inject_lora(
    model,
    rank: int,
    alpha: float,
    target_modules: list[str],
    use_rslora: bool = False,
):
    """Attach a LoRA adapter to every LinearLayer whose name matches any pattern in target_modules."""
    patterns = [re.compile(p) for p in target_modules]
    for name, module in model.named_modules():
        if not isinstance(module, LinearLayer):
            continue
        if not any(p.search(name) for p in patterns):
            continue
        module.adapter = LoRA(
            module.in_features, module.out_features, rank, alpha, use_rslora
        )
    return model
