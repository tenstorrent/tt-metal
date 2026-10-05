# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small deterministic fixtures for CPU-only GDN reference tests."""

import torch

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig

# Two K heads with three V heads each: the smallest shape that exercises the K/V head grouping.
TINY = GDNConfig(
    hidden_size=64,
    num_key_heads=2,
    num_value_heads=6,
    head_k_dim=16,
    head_v_dim=16,
    conv_kernel_size=4,
    norm_eps=1e-6,
    output_gate_activation="silu",
)


def random_weights(config: GDNConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    """bf16 layer-local weights in the canonical schema, magnitudes that keep every gate away from saturation."""
    g = torch.Generator().manual_seed(seed)

    def w(*dims, scale):
        return (torch.randn(*dims, generator=g) * scale).to(torch.bfloat16)

    return {
        "in_proj_qkv.weight": w(config.conv_dim, config.hidden_size, scale=0.2),
        "in_proj_z.weight": w(config.v_dim, config.hidden_size, scale=0.2),
        "in_proj_a.weight": w(config.num_value_heads, config.hidden_size, scale=0.2),
        "in_proj_b.weight": w(config.num_value_heads, config.hidden_size, scale=0.2),
        "out_proj.weight": w(config.hidden_size, config.v_dim, scale=0.2),
        "conv1d.weight": w(config.conv_dim, 1, config.conv_kernel_size, scale=0.5),
        "A_log": w(config.num_value_heads, scale=1.0),
        "dt_bias": w(config.num_value_heads, scale=1.0),
        "norm.weight": (1 + w(config.head_v_dim, scale=0.1).float()).to(torch.bfloat16),
    }
