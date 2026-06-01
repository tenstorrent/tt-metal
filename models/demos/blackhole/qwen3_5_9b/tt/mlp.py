# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP for Qwen3.5-9B: down(silu(gate(x)) * up(x))."""
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj  [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj  [in, out], bfloat8_b
    w3: ttnn.Tensor  # up_proj    [in, out], bfloat4_b


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None) -> MLPWeights:
    """state_dict is the per-layer mlp substate: keys 'gate_proj.weight', 'down_proj.weight', 'up_proj.weight'."""

    def load(name, dtype):
        t = state_dict[f"{name}.weight"].T.contiguous()  # [in, out] for ttnn.linear
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mlp.{name}.weight") if tensor_cache_path else None,
        )

    # Gate and up projections use bfloat4_b (halves memory bandwidth for these large matmuls).
    # Down projection stays at bfloat8_b (on the critical accuracy path).
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat4_b),
        w2=load("down_proj", ttnn.bfloat8_b),
        w3=load("up_proj", ttnn.bfloat4_b),
    )


class Qwen35MLP:
    """SwiGLU feed-forward network for Qwen3.5-9B."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x):
        w = self.weights
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        output = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        return output
