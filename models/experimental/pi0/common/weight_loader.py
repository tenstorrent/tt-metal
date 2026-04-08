# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Weight loader for PI0 model.

This module handles loading weights from HuggingFace (lerobot/pi0_base) or local
safetensors files and converting them to TTNN format.

Weight Structure:
    - action_in_proj, action_out_proj: PI0-specific action projections
    - action_time_mlp_in, action_time_mlp_out: Time conditioning MLP
    - state_proj: State projection
    - paligemma_with_expert.paligemma.*: VLM backbone (Gemma 2B + SigLIP)
    - paligemma_with_expert.gemma_expert.*: Action expert (Gemma 300M)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from safetensors.torch import load_file as safetensors_load_file

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


@dataclass
class PI0Config:
    """Configuration for PI0 model."""

    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    precision: str = "bfloat16"

    # Gemma 2B config (PaliGemma backbone)
    vlm_width: int = 2048
    vlm_depth: int = 18
    vlm_mlp_dim: int = 16384
    vlm_num_heads: int = 8
    vlm_num_kv_heads: int = 1
    vlm_head_dim: int = 256

    # Gemma 300M config (Action expert)
    expert_width: int = 1024
    expert_depth: int = 18
    expert_mlp_dim: int = 4096
    expert_num_heads: int = 8
    expert_num_kv_heads: int = 1
    expert_head_dim: int = 256

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "PI0Config":
        """Load config from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)


def load_pi0_state_dict(
    model_path: Union[str, Path],
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load PI0 model weights from safetensors file or HuggingFace.

    Args:
        model_path: Path to model directory or HuggingFace model ID
        device: Device to load weights to ("cpu" or "cuda")

    Returns:
        Dictionary of weight tensors
    """
    model_path = Path(model_path)

    # Check if it's a local path
    if model_path.exists():
        safetensors_path = model_path / "model.safetensors"
        if safetensors_path.exists():
            state_dict = safetensors_load_file(str(safetensors_path))
        else:
            raise FileNotFoundError(f"No model.safetensors found in {model_path}")
    else:
        # Try to download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download

            # Download safetensors file
            safetensors_path = hf_hub_download(
                repo_id=str(model_path),
                filename="model.safetensors",
                token=os.environ.get("HF_TOKEN"),
            )
            state_dict = safetensors_load_file(safetensors_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    return state_dict


def categorize_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Categorize weights into different components.

    Args:
        state_dict: Full model state dict

    Returns:
        Dictionary with categorized weights:
            - "pi0_projections": action_in_proj, action_out_proj, state_proj, time_mlp
            - "vlm_language": PaliGemma language model weights
            - "vlm_vision": SigLIP vision tower weights
            - "vlm_projector": Multimodal projector weights
            - "action_expert": Gemma expert weights
    """
    categorized = {
        "pi0_projections": {},
        "vlm_language": {},
        "vlm_vision": {},
        "vlm_projector": {},
        "action_expert": {},
    }

    for key, value in state_dict.items():
        if key.startswith("action_in_proj") or key.startswith("action_out_proj"):
            categorized["pi0_projections"][key] = value
        elif key.startswith("action_time_mlp"):
            categorized["pi0_projections"][key] = value
        elif key.startswith("state_proj"):
            categorized["pi0_projections"][key] = value
        elif "paligemma.model.language_model" in key:
            # Transform: paligemma_with_expert.paligemma.model.language_model.layers.X -> model.layers.X
            new_key = key.replace("paligemma_with_expert.paligemma.model.language_model.", "model.")
            categorized["vlm_language"][new_key] = value
        elif "paligemma.model.vision_tower" in key:
            # Remove prefix but keep vision_model structure for now
            # The specific weight extraction methods will handle further transformations
            new_key = key.replace("paligemma_with_expert.paligemma.model.vision_tower.", "")
            categorized["vlm_vision"][new_key] = value
        elif "paligemma.model.multi_modal_projector" in key:
            new_key = key.replace("paligemma_with_expert.paligemma.model.multi_modal_projector.", "")
            categorized["vlm_projector"][new_key] = value
        elif "gemma_expert" in key:
            # Remove prefix but keep "model." for consistency with block weight extraction
            new_key = key.replace("paligemma_with_expert.gemma_expert.", "")
            categorized["action_expert"][new_key] = value
        elif "paligemma.lm_head" in key:
            new_key = key.replace("paligemma_with_expert.paligemma.", "")
            categorized["vlm_language"][new_key] = value

    return categorized


def convert_linear_weight_to_ttnn(
    weight: torch.Tensor,
    device: "ttnn.Device",
    dtype: "ttnn.DataType" = None,
    layout: "ttnn.Layout" = None,
) -> "ttnn.Tensor":
    """
    Convert a PyTorch linear weight to TTNN format.

    Args:
        weight: PyTorch weight tensor [out_features, in_features]
        device: TTNN device
        dtype: TTNN data type (default: bfloat8_b)
        layout: TTNN layout (default: TILE_LAYOUT)

    Returns:
        TTNN tensor
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    if dtype is None:
        dtype = ttnn.bfloat8_b
    if layout is None:
        layout = ttnn.TILE_LAYOUT

    # Transpose for TTNN format [in_features, out_features]
    weight_t = weight.T.contiguous()

    return ttnn.from_torch(
        weight_t,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def convert_linear_bias_to_ttnn(
    bias: torch.Tensor,
    device: "ttnn.Device",
    dtype: "ttnn.DataType" = None,
    layout: "ttnn.Layout" = None,
) -> "ttnn.Tensor":
    """
    Convert a PyTorch linear bias to TTNN format.

    Args:
        bias: PyTorch bias tensor [out_features]
        device: TTNN device
        dtype: TTNN data type (default: bfloat16)
        layout: TTNN layout (default: TILE_LAYOUT)

    Returns:
        TTNN tensor
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    if dtype is None:
        dtype = ttnn.bfloat16
    if layout is None:
        layout = ttnn.TILE_LAYOUT

    # Expand bias to [1, out_features]
    bias_expanded = bias.unsqueeze(0)

    return ttnn.from_torch(
        bias_expanded,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def fuse_qkv_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fuse Q, K, V weights for efficient computation.

    Args:
        q_weight: Query weight [q_dim, hidden_dim]
        k_weight: Key weight [kv_dim, hidden_dim]
        v_weight: Value weight [kv_dim, hidden_dim]

    Returns:
        Fused QKV weight [q_dim + kv_dim + kv_dim, hidden_dim]
    """
    return torch.cat([q_weight, k_weight, v_weight], dim=0)


def fuse_gate_up_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fuse gate and up projection weights for GeGLU.

    Args:
        gate_weight: Gate projection weight [mlp_dim, hidden_dim]
        up_weight: Up projection weight [mlp_dim, hidden_dim]

    Returns:
        Fused gate_up weight [2*mlp_dim, hidden_dim]
    """
    return torch.cat([gate_weight, up_weight], dim=0)


class PI0WeightLoader:
    """
    Weight loader for PI0 model with TTNN conversion support.

    Example:
        loader = PI0WeightLoader("lerobot/pi0_base")
        config = loader.config

        # Get PyTorch weights
        torch_weights = loader.get_torch_weights()

        # Convert to TTNN (requires device)
        ttnn_weights = loader.get_ttnn_weights(device)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        cache_path: Optional[Path] = None,
    ):
        """
        Initialize weight loader.

        Args:
            model_path: Path to model directory or HuggingFace model ID
            cache_path: Optional path for caching converted weights
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.cache_path = cache_path

        # Load config
        config_path = self.model_path / "config.json" if self.model_path.exists() else None
        if config_path and config_path.exists():
            self.config = PI0Config.from_json(config_path)
        else:
            self.config = PI0Config()

        # Lazy load weights
        self._state_dict = None
        self._categorized = None

    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get full state dict (lazy loaded)."""
        if self._state_dict is None:
            self._state_dict = load_pi0_state_dict(self.model_path)
        return self._state_dict

    @property
    def categorized_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get categorized weights (lazy loaded)."""
        if self._categorized is None:
            self._categorized = categorize_weights(self.state_dict)
        return self._categorized

    def get_pi0_projections(self) -> Dict[str, torch.Tensor]:
        """Get PI0-specific projection weights."""
        return self.categorized_weights["pi0_projections"]

    def get_vlm_language_weights(self) -> Dict[str, torch.Tensor]:
        """Get VLM language model weights."""
        return self.categorized_weights["vlm_language"]

    def get_vlm_vision_weights(self) -> Dict[str, torch.Tensor]:
        """Get VLM vision tower weights."""
        return self.categorized_weights["vlm_vision"]

    def get_action_expert_weights(self) -> Dict[str, torch.Tensor]:
        """Get action expert weights."""
        return self.categorized_weights["action_expert"]

    def get_layer_weights(
        self,
        layer_idx: int,
        component: str = "action_expert",
    ) -> Dict[str, torch.Tensor]:
        """
        Get weights for a specific transformer layer.

        Args:
            layer_idx: Layer index (0-17 for both VLM and expert)
            component: "action_expert" or "vlm_language"

        Returns:
            Dictionary with layer weights:
                - input_layernorm.weight
                - self_attn.q_proj.weight
                - self_attn.k_proj.weight
                - self_attn.v_proj.weight
                - self_attn.o_proj.weight
                - post_attention_layernorm.weight
                - mlp.gate_proj.weight
                - mlp.up_proj.weight
                - mlp.down_proj.weight
        """
        if component == "action_expert":
            weights = self.get_action_expert_weights()
            prefix = f"model.layers.{layer_idx}."
        else:
            weights = self.get_vlm_language_weights()
            prefix = f"model.layers.{layer_idx}."

        layer_weights = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                layer_weights[new_key] = value

        return layer_weights

    def get_fused_qkv_weight(
        self,
        layer_idx: int,
        component: str = "action_expert",
    ) -> torch.Tensor:
        """
        Get fused QKV weight for a layer.

        Args:
            layer_idx: Layer index
            component: "action_expert" or "vlm_language"

        Returns:
            Fused QKV weight tensor
        """
        layer_weights = self.get_layer_weights(layer_idx, component)

        q_weight = layer_weights["self_attn.q_proj.weight"]
        k_weight = layer_weights["self_attn.k_proj.weight"]
        v_weight = layer_weights["self_attn.v_proj.weight"]

        return fuse_qkv_weights(q_weight, k_weight, v_weight)

    def get_fused_gate_up_weight(
        self,
        layer_idx: int,
        component: str = "action_expert",
    ) -> torch.Tensor:
        """
        Get fused gate+up weight for a layer's MLP.

        Args:
            layer_idx: Layer index
            component: "action_expert" or "vlm_language"

        Returns:
            Fused gate_up weight tensor
        """
        layer_weights = self.get_layer_weights(layer_idx, component)

        gate_weight = layer_weights["mlp.gate_proj.weight"]
        up_weight = layer_weights["mlp.up_proj.weight"]

        return fuse_gate_up_weights(gate_weight, up_weight)
