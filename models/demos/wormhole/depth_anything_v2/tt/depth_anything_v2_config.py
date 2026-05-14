# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Depth-Anything-V2-Large Configuration for TTNN Implementation.

Depth-Anything-V2-Large uses DINOv2 ViT-L/14 as backbone with a DPT
(Dense Prediction Transformer) decoder head for monocular depth estimation.

Architecture:
  - Backbone: DINOv2 ViT-L/14 (1024 embed_dim, 24 layers, 16 heads)
  - Decoder: DPT head (4 intermediate layers at indices [4, 11, 17, 23])
  - Input: 518x518x3 RGB images
  - Output: Single-channel depth map (518x518)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DepthAnythingV2Config:
    """Configuration for Depth-Anything-V2-Large TTNN model."""

    # === ViT-L/14 Backbone (DINOv2) ===
    image_size: int = 518
    patch_size: int = 14
    in_channels: int = 3
    embed_dim: int = 1024  # ViT-L hidden size
    num_layers: int = 24  # ViT-L depth
    num_heads: int = 16  # ViT-L attention heads
    mlp_ratio: float = 4.0  # MLP hidden dim = embed_dim * mlp_ratio
    mlp_hidden_dim: int = 4096  # 1024 * 4
    head_dim: int = 64  # 1024 // 16
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    init_values: float = 1.0  # LayerScale init
    num_register_tokens: int = 0
    layer_norm_eps: float = 1e-6

    # === Intermediate layer indices for DPT ===
    intermediate_layer_indices: List[int] = field(default_factory=lambda: [4, 11, 17, 23])

    # === DPT Decoder Head ===
    dpt_features: int = 256
    dpt_out_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
    use_clstoken: bool = False
    use_bn: bool = False

    # === TTNN Optimization Parameters ===
    weights_dtype: str = "bfloat16"
    use_lofi: bool = True
    use_l1_memory: bool = False

    # Derived
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_h(self) -> int:
        return self.image_size // self.patch_size

    @property
    def patch_w(self) -> int:
        return self.image_size // self.patch_size

    @classmethod
    def from_huggingface(cls, hf_config, **kwargs):
        """Create config from HuggingFace model config."""
        return cls(
            image_size=hf_config.img_size if hasattr(hf_config, "img_size") else 518,
            patch_size=hf_config.patch_size if hasattr(hf_config, "patch_size") else 14,
            embed_dim=hf_config.embed_dim if hasattr(hf_config, "embed_dim") else 1024,
            num_layers=hf_config.depth if hasattr(hf_config, "depth") else 24,
            num_heads=hf_config.num_heads if hasattr(hf_config, "num_heads") else 16,
            **kwargs,
        )

    def get_compute_kernel_config(self, device):
        """Get compute kernel config for Wormhole."""
        import ttnn

        if self.use_lofi:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            )
        else:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            )

    def get_memory_config(self):
        """Get memory config based on optimization settings."""
        import ttnn

        if self.use_l1_memory:
            return ttnn.L1_MEMORY_CONFIG
        else:
            return ttnn.DRAM_MEMORY_CONFIG
