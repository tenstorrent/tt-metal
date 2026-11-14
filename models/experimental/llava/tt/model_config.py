# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ttnn


@dataclass
class VisionModelArgs:
    """
    Configuration for LLaVA multimodal model on Tenstorrent hardware

    Reference: models/demos/qwen25_vl/tt/model_config.py
    """

    # Vision encoder configuration
    vision_encoder_type: str = "clip"  # "clip", "siglip", "dino", etc.
    vision_hidden_size: int = 1024  # CLIP ViT-L/14 output dim
    vision_num_patches: int = 576  # 24x24 grid at 336px resolution
    vision_resolution: int = 336  # Input image resolution

    # Projector configuration
    projector_type: str = "mlp"  # "mlp" or "moe" (future)
    projector_hidden_size: int = 4096  # LLM hidden size

    # LLM configuration
    llm_backbone: str = "llama3.1"
    llm_hidden_size: int = 4096
    max_seq_len: int = 2048
    max_batch_size: int = 1  # Prefill limitation

    # Device configuration
    mesh_device: Optional[ttnn.MeshDevice] = None
    dtype: ttnn.DataType = ttnn.bfloat16
    is_galaxy: bool = False

    # Weight caching
    weight_cache_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize derived properties"""
        if self.mesh_device:
            self.is_galaxy = self.mesh_device.get_num_devices() > 8

        # Set default weight cache path
        if self.weight_cache_path is None:
            import os

            cache_root = Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal"))
            self.weight_cache_path = cache_root / "models" / "llava"

        # Encoder-specific defaults: when caller picks "siglip" but doesn't override
        # the per-encoder fields, fill them from the canonical SigLIP SO400M @ 384
        # config so downstream code (projector, tests) sees the right shapes.
        if self.vision_encoder_type == "siglip":
            if self.vision_hidden_size == 1024:
                self.vision_hidden_size = 1152
            if self.vision_num_patches == 576:
                self.vision_num_patches = 729
            if self.vision_resolution == 336:
                self.vision_resolution = 384

        # MoE-LLaVA-Qwen-1.8B-4e defaults: the projector maps CLIP features
        # (1024) into the Qwen-1.8B hidden size (2048), not LLaMA 3.1's 4096.
        # Only fill when the caller kept the LLaMA defaults; explicit values win.
        if self.llm_backbone == "qwen-moe":
            if self.projector_hidden_size == 4096:
                self.projector_hidden_size = 2048
            if self.llm_hidden_size == 4096:
                self.llm_hidden_size = 2048

    def prepare_residual_tensor_prefill(self, x, force_replicated=True):
        """
        Convert torch tensor to ttnn tensor with proper sharding for prefill

        Critical: Use this for all tensors going to device

        Args:
            x: torch.Tensor to convert
            force_replicated: If True, replicate across mesh (single-device mode)
                            If False, use DRAM_MEMORY_CONFIG

        Returns:
            ttnn.Tensor with proper layout and sharding

        Reference: models/demos/qwen25_vl/tt/model_config.py:prepare_residual_tensor_prefill
        """
        return ttnn.as_tensor(
            x,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG if force_replicated else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def get_vision_output_dim(self):
        """
        Get vision encoder output dimension

        Handles different encoders with variable output sizes
        """
        # For now, all encoders output to vision_hidden_size
        # SigLIP may differ - handle in encoder implementation
        return self.vision_hidden_size
