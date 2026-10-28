# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-weight schema for the Kimi-K3 vision tower.

Every source of weights produces the same dataclass: HF layout (`[out, in]`), torch, on host. A
module converts from there to whatever it needs (a ttnn module transposes and uploads). Changing how
weights are loaded or streamed means adding a producer here, not touching a module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from models.experimental.kimi_k3_moonvit_v2.reference.config import KimiK3VisionConfig

# Upstream MoonViT names; the MLP is `fc0` then `fc1`.
FFN_WEIGHT_NAMES = ("fc0.weight", "fc1.weight")


def expected_ffn_weight_shapes(cfg: type = KimiK3VisionConfig) -> dict[str, tuple[int, int]]:
    return {
        "fc0.weight": (cfg.VT_INTERMEDIATE_SIZE, cfg.VT_HIDDEN_SIZE),
        "fc1.weight": (cfg.VT_HIDDEN_SIZE, cfg.VT_INTERMEDIATE_SIZE),
    }


@dataclass(frozen=True)
class VisionFfnWeights:
    """One vision block's FFN weights, HF layout `[out, in]`."""

    fc0: torch.Tensor  # [intermediate, hidden]
    fc1: torch.Tensor  # [hidden, intermediate]

    def validate(self, cfg: type = KimiK3VisionConfig) -> None:
        shapes = expected_ffn_weight_shapes(cfg)
        for name, tensor in zip(FFN_WEIGHT_NAMES, (self.fc0, self.fc1)):
            if tuple(tensor.shape) != shapes[name]:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}; expected {shapes[name]}")

    @classmethod
    def from_state_dict(
        cls, state_dict: Mapping[str, torch.Tensor], prefix: str = "", cfg: type = KimiK3VisionConfig
    ) -> "VisionFfnWeights":
        """From a checkpoint state dict; `prefix` is the block's MLP path, e.g. `"...blocks.0.mlp."`."""
        weights = cls(*(state_dict[prefix + name] for name in FFN_WEIGHT_NAMES))
        weights.validate(cfg)
        return weights

    @classmethod
    def synthetic(cls, seed: int = 0, cfg: type = KimiK3VisionConfig) -> "VisionFfnWeights":
        """Random weights with std 1/sqrt(fan_in), so a unit-variance input stays O(1) through both
        projections and GELU sees values in its non-linear range rather than saturating."""
        gen = torch.Generator().manual_seed(seed)
        tensors = []
        for name in FFN_WEIGHT_NAMES:
            shape = expected_ffn_weight_shapes(cfg)[name]
            tensors.append(torch.randn(shape, generator=gen) / shape[1] ** 0.5)
        return cls(*tensors)
