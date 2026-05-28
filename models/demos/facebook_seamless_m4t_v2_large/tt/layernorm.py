# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the standard LayerNorm used throughout SeamlessM4T-v2.

The PyTorch reference is the standard layer-norm op (matches
`models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::layernorm_forward`).

This module wraps `ttnn.layer_norm` and follows the same loading pattern as
`models/demos/qwen3_vl/tt/vision_layernorm.py`, simplified for the unsharded
(interleaved DRAM) path used by SeamlessM4T-v2's many LayerNorm sites
(conformer, NLLB encoder/decoder, T2U, variance predictor).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class LayerNorm(LightweightModule):
    """Standard nn.LayerNorm in TTNN over the last dim.

    Args:
        device: ttnn device or mesh device.
        dim: last-dim size (`normalized_shape[-1]`).
        weight: torch.Tensor of shape (dim,) — gamma.
        bias: torch.Tensor of shape (dim,) — beta.
        eps: layer_norm epsilon (default 1e-5, matches HF
             `SeamlessM4Tv2Config.layer_norm_eps`).
        weight_dtype: storage dtype for weight/bias on device.
        weight_memory_config: where to place weight/bias.
    """

    def __init__(
        self,
        device,
        dim: int,
        weight,
        bias,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        self.dim = dim

        # LayerNorm expects weight/bias broadcastable to the last dim of input.
        # We store them as 1D row-tiled tensors in DRAM.
        self.weight = ttnn.from_torch(
            weight.reshape(1, 1, 1, dim),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, dim),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply LayerNorm over the last dim of `x`.

        `x` must be a TILE_LAYOUT ttnn tensor whose last dim equals `self.dim`.
        """
        return ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_kernel_config,
        )
