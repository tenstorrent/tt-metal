# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic KDA test configuration and weights."""

import torch

import ttnn
from models.experimental.kimi_delta_attention.config import KDAConfig


def make_config(*, recurrent_state_dtype: ttnn.DataType = ttnn.float32) -> KDAConfig:
    return KDAConfig(
        hidden_size=64,
        num_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
        recurrent_state_dtype=recurrent_state_dtype,
        chunk_size=4,
    )


def random_weights(config: KDAConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260723)

    def normal(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return scale * torch.randn(*shape, generator=generator)

    hidden = config.hidden_size
    key_rank, value_rank = config.head_k_dim, config.head_v_dim
    return {
        "q_proj.weight": normal(config.q_dim, hidden),
        "k_proj.weight": normal(config.k_dim, hidden),
        "v_proj.weight": normal(config.v_dim, hidden),
        "q_conv1d.weight": normal(config.q_dim, 1, config.conv_kernel_size, scale=0.2),
        "k_conv1d.weight": normal(config.k_dim, 1, config.conv_kernel_size, scale=0.2),
        "v_conv1d.weight": normal(config.v_dim, 1, config.conv_kernel_size, scale=0.2),
        "A_log": torch.log(torch.linspace(1.0, 4.0, config.num_heads)).reshape(1, 1, config.num_heads, 1),
        "f_a_proj.weight": normal(key_rank, hidden),
        "f_b_proj.weight": normal(config.num_heads * key_rank, key_rank),
        "dt_bias": normal(config.num_heads * key_rank),
        "b_proj.weight": normal(config.num_heads, hidden),
        "g_a_proj.weight": normal(value_rank, hidden),
        "g_b_proj.weight": normal(config.num_heads * value_rank, value_rank),
        "o_norm.weight": 1.0 + normal(value_rank),
        "o_proj.weight": normal(hidden, config.num_heads * value_rank),
    }
