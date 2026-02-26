# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Model configuration for Molmo2-8B TTNN implementation.

This module provides configuration classes for the three main components:
- VisionModelArgs: ViT encoder configuration (27 layers, hidden=1152)
- AdapterModelArgs: Vision adapter configuration (pooling + projector)
- TextModelArgs: Language model configuration (36 layers, hidden=4096)
- Molmo2ModelArgs: Combined configuration for full model
"""

import math
from dataclasses import dataclass
from typing import Tuple

from loguru import logger

from models.tt_transformers.tt.model_config import ModelArgs


def nearest_multiple(x: int, multiple: int) -> int:
    """Round x up to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


@dataclass
class Molmo2VisionConfig:
    """Configuration for the Molmo2 Vision Transformer (ViT) encoder."""

    hidden_size: int = 1152
    intermediate_size: int = 4304  # 1152 * 3.74 (GELU MLP ratio)
    num_hidden_layers: int = 27  # Total layers in ViT (25 used)
    num_layers_used: int = 25  # Actually used layers
    num_attention_heads: int = 16
    head_dim: int = 72  # 1152 / 16
    patch_size: int = 14
    image_size: int = 378
    num_patches: int = 729  # (378 / 14)^2 = 27^2 = 729
    layer_norm_eps: float = 1e-6
    # Layers to extract features from (for multi-scale concat)
    feature_layers: Tuple[int, int] = (18, 24)  # 0-indexed: layers 18 and 24


@dataclass
class Molmo2AdapterConfig:
    """Configuration for the Molmo2 vision adapter (pooling + projector)."""

    # Image pooling (cross-attention)
    pooling_input_dim: int = 2304  # 1152 * 2 (concat of two ViT layer outputs)
    pooling_hidden_dim: int = 1152
    pooling_num_heads: int = 16

    # Image projector (SwiGLU)
    projector_input_dim: int = 1152
    projector_intermediate_dim: int = 12288
    projector_output_dim: int = 4096


@dataclass
class Molmo2TextConfig:
    """Configuration for the Molmo2 language model."""

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA 32/8
    head_dim: int = 128  # 4096 / 32
    vocab_size: int = 152064  # 151936 + 128 special tokens
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    use_qk_norm: bool = True
    qk_norm_type: str = "qwen3"  # Per-head RMSNorm

    # Special token IDs
    image_patch_id: int = 151938  # Token marking image patch positions
    bos_token_id: int = 151643
    eos_token_id: int = 151645


class Molmo2VisionModelArgs(ModelArgs):
    """
    Model arguments for the Molmo2 Vision Transformer encoder.

    Extends ModelArgs to provide vision-specific configuration and utilities.
    """

    def __init__(self, *args, **kwargs):
        # Don't call super().__init__() yet - we need to set up vision config first
        # Store mesh_device for later
        self._init_args = args
        self._init_kwargs = kwargs

        # Vision config from Molmo2
        self.vision_config = Molmo2VisionConfig()

        # Now initialize parent with mesh_device
        mesh_device = kwargs.get("mesh_device") or (args[0] if args else None)
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 1
        self.mesh_device = mesh_device
        self.tile_size = 32

        # Core dimensions from vision config
        self.dim = self.vision_config.hidden_size
        self.unpadded_hidden_dim = self.vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(self.unpadded_hidden_dim, self.tile_size * self.num_devices)
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"Padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")

        self.head_dim = self.vision_config.head_dim
        self.n_heads = self.vision_config.num_attention_heads
        self.n_kv_heads = self.vision_config.num_attention_heads  # Full attention (not GQA)
        self.n_layers = self.vision_config.num_layers_used

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size
        if self.padded_head_dim != self.head_dim:
            logger.info(f"Padding head dim from {self.head_dim} to {self.padded_head_dim}")

        # QKV size for attention
        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)

        # Model config dict for TTNN operations
        self.model_config = {}

    def get_state_dict_prefix(self, module_name: str, layer_num: int = None) -> str:
        """Get the state dict key prefix for a vision module."""
        layer_prefix = f"image_vit.transformer.resblocks.{layer_num}." if layer_num is not None else ""

        module_map = {
            "VisionBlock": "",
            "VisionAttention": "attn.",
            "VisionMLP": "mlp.",
            "VisionTransformer": "image_vit.",
            "ln_1": "ln_1.",
            "ln_2": "ln_2.",
            "": "",
        }
        return layer_prefix + module_map.get(module_name, "")

    def reference_vision_model(self, depth: int = None):
        """Load reference HuggingFace vision model for testing."""
        from models.demos.molmo2.reference.model import Molmo2Reference

        ref = Molmo2Reference(self.CKPT_DIR)
        return ref.image_vit

    def reference_vision_block(self, layer_num: int = 0):
        """Get reference vision block for testing."""
        return self.reference_vision_model().transformer.resblocks[layer_num]


class Molmo2AdapterModelArgs:
    """
    Model arguments for the Molmo2 vision adapter (pooling + projector).
    """

    def __init__(self, mesh_device, **kwargs):
        self.adapter_config = Molmo2AdapterConfig()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 1
        self.tile_size = 32

        # Pooling dimensions
        self.pooling_input_dim = self.adapter_config.pooling_input_dim
        self.pooling_hidden_dim = self.adapter_config.pooling_hidden_dim
        self.pooling_num_heads = self.adapter_config.pooling_num_heads
        self.pooling_head_dim = self.pooling_hidden_dim // self.pooling_num_heads

        # Projector dimensions
        self.projector_input_dim = self.adapter_config.projector_input_dim
        self.projector_intermediate_dim = self.adapter_config.projector_intermediate_dim
        self.projector_output_dim = self.adapter_config.projector_output_dim

        # Model config dict
        self.model_config = {}

    def get_state_dict_prefix(self, module_name: str) -> str:
        """Get the state dict key prefix for an adapter module."""
        module_map = {
            "ImagePooling": "image_pooling_2d.",
            "ImageProjector": "image_projector.",
            "": "",
        }
        return module_map.get(module_name, "")


class Molmo2TextModelArgs(ModelArgs):
    """
    Model arguments for the Molmo2 language model.

    Extends ModelArgs to use Molmo2-specific configuration.
    This should largely reuse tt_transformers infrastructure.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent ModelArgs
        super().__init__(*args, **kwargs)

        # Override with Molmo2-specific text config
        self.text_config = Molmo2TextConfig()

        # Core dimensions
        self.dim = self.text_config.hidden_size
        self.hidden_dim = self.text_config.intermediate_size
        self.n_layers = self.text_config.num_hidden_layers
        self.n_heads = self.text_config.num_attention_heads
        self.n_kv_heads = self.text_config.num_key_value_heads
        self.head_dim = self.text_config.head_dim
        self.vocab_size = self.text_config.vocab_size

        # RoPE and normalization
        self.rope_theta = self.text_config.rope_theta
        self.norm_eps = self.text_config.rms_norm_eps
        self.use_qk_norm = self.text_config.use_qk_norm
        self.qk_norm_type = self.text_config.qk_norm_type

        # Special tokens
        self.image_patch_id = self.text_config.image_patch_id
        self.bos_token_id = self.text_config.bos_token_id
        self.eos_token_id = self.text_config.eos_token_id

    def get_state_dict_prefix(self, module_name: str, layer_num: int = None) -> str:
        """Get the state dict key prefix for a text model module."""
        layer_prefix = f"model.layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "Transformer": "model.",
            "Attention": "attention.",
            "MLP": "feed_forward.",
            "attention_norm": "attention_norm.",
            "ffn_norm": "ffn_norm.",
            "": "",
        }
        return layer_prefix + module_map.get(module_name, "")


class Molmo2ModelArgs:
    """
    Combined model arguments for the full Molmo2-8B model.

    Provides access to vision, adapter, and text model configurations.
    """

    # Model-specific parameters
    MODEL_NAME = "Molmo2-8B"
    TOTAL_PARAMS = 8.66e9  # 8.66B parameters
    VIT_PARAMS = 383e6  # 383M (4.4%)
    ADAPTER_PARAMS = 88e6  # 88M (1.0%)
    TEXT_PARAMS = 8192e6  # 8,192M (94.6%)

    def __init__(
        self,
        mesh_device,
        max_batch_size: int = 1,
        max_seq_len: int = 128 * 1024,
        **kwargs,
    ):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Component configurations
        self.vision_config = Molmo2VisionConfig()
        self.adapter_config = Molmo2AdapterConfig()
        self.text_config = Molmo2TextConfig()

        # Store for creating component-specific ModelArgs
        self._kwargs = kwargs

    def get_vision_args(self) -> Molmo2VisionModelArgs:
        """Get model args for the vision encoder."""
        return Molmo2VisionModelArgs(
            mesh_device=self.mesh_device,
            max_batch_size=self.max_batch_size,
            **self._kwargs,
        )

    def get_adapter_args(self) -> Molmo2AdapterModelArgs:
        """Get model args for the vision adapter."""
        return Molmo2AdapterModelArgs(
            mesh_device=self.mesh_device,
            **self._kwargs,
        )

    def get_text_args(self) -> Molmo2TextModelArgs:
        """Get model args for the text model."""
        return Molmo2TextModelArgs(
            mesh_device=self.mesh_device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            **self._kwargs,
        )

    @property
    def image_size(self) -> int:
        """Input image size for the vision encoder."""
        return self.vision_config.image_size

    @property
    def patch_size(self) -> int:
        """Patch size for vision encoder."""
        return self.vision_config.patch_size

    @property
    def num_patches(self) -> int:
        """Number of patches per image."""
        return self.vision_config.num_patches

    @property
    def feature_layers(self) -> Tuple[int, int]:
        """ViT layers to extract features from."""
        return self.vision_config.feature_layers

    @property
    def text_hidden_size(self) -> int:
        """Hidden size of the language model."""
        return self.text_config.hidden_size

    @property
    def image_patch_id(self) -> int:
        """Special token ID marking image patch positions."""
        return self.text_config.image_patch_id
