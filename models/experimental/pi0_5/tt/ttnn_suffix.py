# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 suffix embedding (TTNN).

Drops state_proj and the action+time MLP. Time is encoded with sincos and
pushed through `time_mlp_in -> silu -> time_mlp_out` to produce
`adarms_cond` for the action expert. Suffix = action_in_proj(noisy_actions).
"""

from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import SuffixConfig
from models.experimental.pi0.tt.ttnn_common import (
    create_sinusoidal_pos_embedding_ttnn,
    tensor_1d_to_2d_ttnn,
)
from models.experimental.pi0.tt.ttnn_suffix import SuffixEmbeddingTTNN


class Pi0_5SuffixEmbeddingTTNN(SuffixEmbeddingTTNN):
    """
    PI0.5 TTNN suffix.

    Expected weight keys (TTNN tensors):
      - action_in_proj.{weight,bias}
      - action_out_proj.{weight,bias}
      - time_mlp_in.{weight,bias}
      - time_mlp_out.{weight,bias}
    """

    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, "ttnn.Tensor"],
        device: "ttnn.Device",
    ):
        assert config.pi05, "Pi0_5SuffixEmbeddingTTNN requires config.pi05=True"
        self.config = config
        self.device = device
        self.weights = weights

        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # Pre-compute attention mask pattern: pi0.5 has only action tokens,
        # so the pattern is [1, 0, 0, ..., 0] of length action_horizon.
        suffix_len = config.action_horizon
        pad_len = ((suffix_len + 31) // 32) * 32
        att = ttnn.zeros((1, pad_len), device=device, dtype=ttnn.bfloat16)
        att = ttnn.to_layout(att, ttnn.TILE_LAYOUT)
        ones = ttnn.ones((1, 1), device=device, dtype=ttnn.bfloat16)
        ones = ttnn.to_layout(ones, ttnn.TILE_LAYOUT)
        ones_padded = ttnn.pad(ones, [(0, 0), (0, pad_len - 1)], value=0.0)
        ttnn.deallocate(ones)
        ttnn.deallocate(att)
        self._att_mask_pattern = ones_padded
        self._att_mask_suffix_len = suffix_len

        self.indices = ttnn.arange(0, 512, 1, device=device, dtype=ttnn.float32)

    def embed_adarms_cond(self, timestep: "ttnn.Tensor") -> "ttnn.Tensor":
        """sincos(t) -> Linear -> silu -> Linear  -> adarms_cond."""
        sincos = create_sinusoidal_pos_embedding_ttnn(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
            device=self.device,
            indices=self.indices,
        )
        x = ttnn.linear(
            sincos,
            self.weights["time_mlp_in.weight"],
            bias=self.weights.get("time_mlp_in.bias"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
        )
        x = ttnn.silu(x)
        x = ttnn.linear(
            x,
            self.weights["time_mlp_out.weight"],
            bias=self.weights.get("time_mlp_out.bias"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
        )
        return x

    def embed_suffix(
        self,
        state: Optional["ttnn.Tensor"],
        noisy_actions: "ttnn.Tensor",
        timestep: "ttnn.Tensor",
    ) -> Tuple["ttnn.Tensor", Optional["ttnn.Tensor"], "ttnn.Tensor", "ttnn.Tensor"]:
        batch_size = noisy_actions.shape[0]

        adarms_cond = self.embed_adarms_cond(timestep)
        suffix_embs = self.embed_actions(noisy_actions)

        suffix_len = suffix_embs.shape[1]
        if batch_size == 1:
            suffix_att_masks = ttnn.slice(self._att_mask_pattern, [0, 0], [1, suffix_len])
        else:
            att_sliced = ttnn.slice(self._att_mask_pattern, [0, 0], [1, suffix_len])
            suffix_att_masks = ttnn.repeat(att_sliced, (batch_size, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return suffix_embs, None, suffix_att_masks, adarms_cond


def convert_pi0_5_suffix_weights_to_ttnn(
    torch_weights: Dict[str, torch.Tensor],
    device: "ttnn.Device",
    dtype: Optional["ttnn.DataType"] = None,
) -> Dict[str, "ttnn.Tensor"]:
    """
    Convert PI0.5 suffix weights from PyTorch to TTNN.

    Accepts only keys that pi0.5 actually uses (action_in_proj, action_out_proj,
    time_mlp_in, time_mlp_out). Other keys in the checkpoint are ignored.
    """
    keep_prefixes = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    if dtype is None:
        weight_dtype = ttnn.bfloat8_b
        bias_dtype = ttnn.bfloat16
    else:
        weight_dtype = dtype
        bias_dtype = dtype

    out: Dict[str, "ttnn.Tensor"] = {}
    for key, value in torch_weights.items():
        if not any(key.startswith(p) for p in keep_prefixes):
            continue
        if "bias" in key:
            out[key] = tensor_1d_to_2d_ttnn(value, device, dtype=bias_dtype)
        else:
            out[key] = ttnn.from_torch(
                value.T.contiguous(),
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    return out
