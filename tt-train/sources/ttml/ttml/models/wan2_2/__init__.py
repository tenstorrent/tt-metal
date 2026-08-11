# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 DiT for ttml. Defaults are Wan2.2-T2V-A14B; one expert per instance."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList, Parameter

from .. import RunnerType, memory_efficient_runner
from .conditioning import WanConditioning, timestep_features
from .patch_embed import (
    conv3d_weight_to_linear,
    grid_size,
    patch_features,
    patchify,
    patchify_output_order,
    unpatchify,
)
from .rope import build_rope_params, build_tables
from .transformer import WanTransformerBlock
from .weights import load_expert_from_safetensors, to_ttml_name

_FINAL_CHUNKS = 2


@dataclass(frozen=True)
class WanConfig:
    dim: int = 5120
    ffn_dim: int = 13824
    num_layers: int = 40
    num_heads: int = 40
    patch_size: tuple = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    model_type: str = "t2v"
    runner_type: RunnerType = RunnerType.Default
    init_weights: bool = True
    use_tp: bool = False

    def weight_init(self):
        return ttml.init.normal(0.0, 0.02) if self.init_weights else ttml.init.zeros()

    def __post_init__(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError(f"dim {self.dim} must be divisible by num_heads {self.num_heads}")
        if self.model_type not in ("t2v", "i2v"):
            raise ValueError(f"model_type must be t2v or i2v, got {self.model_type!r}")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


class WanTransformer3D(AbstractModuleBase):
    """One MoE expert. Input and output are patch tokens; see patch_embed for the layout."""

    def __init__(self, config: WanConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_embed = LinearLayer(
            patch_features(config.in_channels, config.patch_size),
            config.dim,
            True,
            weight_init=config.weight_init(),
        )
        self.condition_embedder = WanConditioning(config)
        self.blocks = ModuleList([WanTransformerBlock(config) for _ in range(config.num_layers)])
        self.scale_shift_table = Parameter(ttml.init.zeros()((1, 1, _FINAL_CHUNKS, config.dim)))
        self.proj_out = LinearLayer(
            config.dim,
            patch_features(config.out_channels, config.patch_size),
            True,
            weight_init=config.weight_init(),
        )

    def _final_modulation(self, temb):
        """Two chunks here, unlike the six per block. Frozen, so no grad."""
        shifted = ttnn.add(self.scale_shift_table.tensor.get_value(), temb.get_value())
        shift, scale = ttnn.chunk(shifted, _FINAL_CHUNKS, dim=2)
        gamma = ttml.autograd.create_tensor(ttnn.add(scale, 1.0), False)
        beta = ttml.autograd.create_tensor(shift, False)
        return gamma, beta

    def forward(self, tokens, timesteps, text_embed, rope_params, mask=None):
        """(B, 1, S, in*prod(patch)) -> (B, 1, S, out*prod(patch)) in proj_out order.

        Compare against patchify_output_order(target); the model does not unpatchify.
        """
        x = self.patch_embed(tokens)
        timestep_proj, temb, prompt = self.condition_embedder(timesteps, text_embed)

        checkpointed = self.config.runner_type == RunnerType.MemoryEfficient
        for block in self.blocks:
            if checkpointed:
                x = memory_efficient_runner(block, x, mask, prompt, timestep_proj, rope_params)
            else:
                x = block(x, mask, prompt, timestep_proj, rope_params)

        gamma, beta = self._final_modulation(temb)
        x = ttml.ops.layernorm.layernorm(x, gamma, beta)
        return self.proj_out(x)


__all__ = [
    "WanConditioning",
    "WanConfig",
    "WanTransformer3D",
    "WanTransformerBlock",
    "build_rope_params",
    "build_tables",
    "conv3d_weight_to_linear",
    "grid_size",
    "load_expert_from_safetensors",
    "patch_features",
    "patchify",
    "patchify_output_order",
    "timestep_features",
    "to_ttml_name",
    "unpatchify",
]
