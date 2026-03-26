# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU MLP for Qwen3.5-9B.

Standard gated MLP: output = down_proj(silu(gate_proj(x)) * up_proj(x))
Same structure as Llama MLP — gate/up/down projections with SiLU activation.
"""
import ttnn


class Qwen35MLP:
    """SwiGLU feed-forward network for Qwen3.5-9B."""

    def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
        self.device = device
        prefix = f"layers.{layer_num}.mlp"

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def load_weight(name, dtype=ttnn.bfloat8_b):
            """Load 2D weight, transpose to [in, out] for ttnn.linear."""
            t = state_dict[f"{prefix}.{name}"].T.contiguous()
            return ttnn.as_tensor(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
            )

        # Gate and up projections use bfloat4_b (halves memory bandwidth for these large matmuls).
        # Down projection stays at bfloat8_b (on the critical accuracy path).
        self.w1 = load_weight("gate_proj.weight", dtype=ttnn.bfloat4_b)
        self.w2 = load_weight("down_proj.weight")
        self.w3 = load_weight("up_proj.weight", dtype=ttnn.bfloat4_b)

    def forward(self, x):
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        # Use L1 for short sequences (decode/small prefill), DRAM for long sequences
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, self.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, self.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        output = ttnn.linear(hidden, self.w2, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        return output
