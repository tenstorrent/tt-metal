# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MLP Projector - TTNN Implementation

Implements the LLaVA 1.5 multimodal projector that maps vision encoder
features into the LLM embedding space.

Architecture (mlp2x_gelu):
    Linear(vision_hidden_size -> llm_hidden_size) -> GELU -> Linear(llm_hidden_size -> llm_hidden_size)

For LLaVA 1.5 7B:
    Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 4096)

Input:  [B, num_patches, 1024]  (CLIP encoder output)
Output: [B, num_patches, 4096]  (LLaMA hidden size)

Weight Loading:
    State dict keys (with "mm_projector." prefix stripped):
        - "0.weight" [4096, 1024], "0.bias" [4096]   (first Linear)
        - "2.weight" [4096, 4096], "2.bias" [4096]   (second Linear)

Reference:
    - LLaVA/llava/model/multimodal_projector/builder.py (build_vision_projector)
    - models/demos/qwen25_vl/tt/patch_merger.py (PatchMerger pattern)
    - models/demos/llava/tt/vision_encoders/clip_encoder.py (weight loading pattern)
"""

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.common.lightweightmodule import LightweightModule


class MLPProjector(LightweightModule):
    """
    MLP Projector - TTNN Implementation

    2-layer MLP with GELU activation: Linear -> GELU -> Linear

    Args:
        args: VisionModelArgs with projector configuration
        dtype: ttnn data type (ttnn.bfloat16)
        state_dict: Model weights dictionary
        weight_cache_path: Path to cached weights (optional)

    Attributes:
        mesh_device: ttnn.MeshDevice for tensor placement
        vision_hidden_size: Input dimension (1024 for CLIP)
        llm_hidden_size: Output dimension (4096 for LLaMA 3.1)
    """

    def __init__(self, args, dtype, state_dict: Dict[str, Any], weight_cache_path: Optional[Path] = None):
        super().__init__()

        self.args = args
        self.mesh_device = args.mesh_device
        self.dtype = dtype

        # Dimensions
        self.vision_hidden_size = args.vision_hidden_size  # 1024 (CLIP output)
        self.llm_hidden_size = args.projector_hidden_size  # 4096 (LLaMA hidden)

        logger.info(f"MLP Projector: {self.vision_hidden_size} -> {self.llm_hidden_size} -> {self.llm_hidden_size}")

        # Load weights
        self._load_weights(state_dict, weight_cache_path)

    def _normalize_keys(self, state_dict):
        """
        Normalize state dict keys by stripping 'mm_projector.' prefix.

        LLaVA stores projector weights as:
            mm_projector.0.weight, mm_projector.0.bias
            mm_projector.2.weight, mm_projector.2.bias

        We normalize to:
            0.weight, 0.bias
            2.weight, 2.bias
        """
        normalized = {}
        for key, val in state_dict.items():
            if key.startswith("mm_projector."):
                normalized[key[len("mm_projector.") :]] = val
            elif key.startswith("0.") or key.startswith("2."):
                normalized[key] = val
        return normalized

    def _load_weights(self, state_dict, weight_cache_path):
        """Load and preprocess weights from PyTorch state_dict"""

        # Normalize keys
        sd = self._normalize_keys(state_dict)

        # Verify expected keys exist
        expected_keys = ["0.weight", "0.bias", "2.weight", "2.bias"]
        for key in expected_keys:
            if key not in sd:
                raise KeyError(f"Missing projector weight: {key}. " f"Available keys: {list(sd.keys())[:10]}...")

        # First Linear: [1024 -> 4096]
        # PyTorch weight shape: [out_features, in_features] = [4096, 1024]
        self.fc1_weight = preprocess_linear_weight(
            sd["0.weight"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.fc1_weight = ttnn.to_device(self.fc1_weight, self.mesh_device)

        self.fc1_bias = preprocess_linear_bias(
            sd["0.bias"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.fc1_bias = ttnn.to_device(self.fc1_bias, self.mesh_device)

        # Second Linear: [4096 -> 4096]
        self.fc2_weight = preprocess_linear_weight(
            sd["2.weight"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.fc2_weight = ttnn.to_device(self.fc2_weight, self.mesh_device)

        self.fc2_bias = preprocess_linear_bias(
            sd["2.bias"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.fc2_bias = ttnn.to_device(self.fc2_bias, self.mesh_device)

        logger.info("MLP Projector weights loaded successfully")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: Linear -> GELU -> Linear

        Args:
            x: Input tensor [B, num_patches, vision_hidden_size] on device

        Returns:
            Output tensor [B, num_patches, llm_hidden_size] on device
        """
        # First Linear: [B, 576, 1024] -> [B, 576, 4096]
        x = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # GELU activation
        x = ttnn.gelu(x)

        # Second Linear: [B, 576, 4096] -> [B, 576, 4096]
        x = ttnn.linear(
            x,
            self.fc2_weight,
            bias=self.fc2_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x

    @property
    def hidden_size(self) -> int:
        """Return output hidden dimension"""
        return self.llm_hidden_size


def create_mlp_projector(args, dtype, state_dict, weight_cache_path=None):
    """
    Factory function to create MLP projector

    Args:
        args: VisionModelArgs
        dtype: ttnn data type
        state_dict: Pre-trained weights
        weight_cache_path: Optional weight cache path

    Returns:
        MLPProjector
    """
    return MLPProjector(args, dtype, state_dict, weight_cache_path)
