# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and conversion utilities for TTTv2"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import safetensors
import torch
from tqdm import tqdm

import ttnn


class WeightConverter(ABC):
    """
    Abstract base class for weight conversion.

    Handles conversion between different model formats.
    """

    @abstractmethod
    def convert(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_config: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert weights from source format to target format.

        Args:
            source_weights: Dictionary of source weight tensors
            target_config: Target model configuration

        Returns:
            Dictionary of converted weight tensors
        """

    @abstractmethod
    def get_weight_map(self) -> Dict[str, str]:
        """Get mapping from source weight names to target names"""


class HuggingFaceConverter(WeightConverter):
    """Converter for HuggingFace model weights"""

    def __init__(self, source_model_type: str, target_model_type: str = "transformer"):
        self.source_model_type = source_model_type
        self.target_model_type = target_model_type
        self.weight_map = self._create_weight_map()

    def convert(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_config: Any,
    ) -> Dict[str, torch.Tensor]:
        """Convert HuggingFace weights to TTT format"""
        converted_weights = {}

        for source_name, target_name in self.weight_map.items():
            if source_name in source_weights:
                weight = source_weights[source_name]

                # Apply any necessary transformations
                weight = self._transform_weight(source_name, weight, target_config)

                converted_weights[target_name] = weight

        return converted_weights

    def get_weight_map(self) -> Dict[str, str]:
        """Get weight name mapping"""
        return self.weight_map

    def _create_weight_map(self) -> Dict[str, str]:
        """Create mapping from HF names to TTT names"""
        if self.source_model_type == "llama":
            return {
                # Embeddings
                "model.embed_tokens.weight": "embeddings.token_embedding.weight",
                # Attention weights
                "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq",
                "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk",
                "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv",
                "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo",
                # MLP weights
                "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1",
                "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3",
                "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2",
                # Normalization
                "model.layers.{}.input_layernorm.weight": "layers.{}.norm1.weight",
                "model.layers.{}.post_attention_layernorm.weight": "layers.{}.norm2.weight",
                # Final norm and output
                "model.norm.weight": "final_norm.weight",
                "lm_head.weight": "lm_head.weight",
            }
        elif self.source_model_type == "gpt2":
            return {
                # Embeddings
                "transformer.wte.weight": "embeddings.token_embedding.weight",
                "transformer.wpe.weight": "embeddings.position_embedding.weight",
                # Attention weights
                "transformer.h.{}.attn.c_attn.weight": "layers.{}.attention.wqkv",
                "transformer.h.{}.attn.c_proj.weight": "layers.{}.attention.wo",
                # MLP weights
                "transformer.h.{}.mlp.c_fc.weight": "layers.{}.mlp.w1",
                "transformer.h.{}.mlp.c_proj.weight": "layers.{}.mlp.w2",
                # Normalization
                "transformer.h.{}.ln_1.weight": "layers.{}.norm1.weight",
                "transformer.h.{}.ln_1.bias": "layers.{}.norm1.bias",
                "transformer.h.{}.ln_2.weight": "layers.{}.norm2.weight",
                "transformer.h.{}.ln_2.bias": "layers.{}.norm2.bias",
                # Final norm and output
                "transformer.ln_f.weight": "final_norm.weight",
                "transformer.ln_f.bias": "final_norm.bias",
            }
        else:
            # Default identity mapping
            return {}

    def _transform_weight(
        self,
        name: str,
        weight: torch.Tensor,
        config: Any,
    ) -> torch.Tensor:
        """Apply any necessary transformations to weights"""
        # Handle combined QKV weights for GPT2
        if "c_attn.weight" in name and hasattr(config, "hidden_size"):
            # Split into Q, K, V
            hidden_size = config.hidden_size
            q, k, v = weight.split(hidden_size, dim=0)
            # Return concatenated for now, actual splitting happens in model
            return weight

        return weight


class WeightLoader:
    """
    Unified weight loader for TTT models.

    Handles loading from various sources and formats.
    """

    def __init__(
        self,
        device: ttnn.Device,
        dtype: ttnn.DataType = ttnn.bfloat16,
        cache_dir: Optional[Path] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ttt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Register converters
        self.converters = {
            "huggingface": HuggingFaceConverter,
        }

    def load_pretrained_weights(
        self,
        model_name_or_path: str,
        model_config: Any,
        weight_map: Optional[Dict[str, str]] = None,
        converter: Optional[WeightConverter] = None,
    ) -> Dict[str, ttnn.Tensor]:
        """
        Load pretrained weights for a model.

        Args:
            model_name_or_path: Model name or path to weights
            model_config: Model configuration
            weight_map: Optional custom weight mapping
            converter: Optional weight converter

        Returns:
            Dictionary of TTNN tensors
        """
        # Load raw weights
        raw_weights = self._load_raw_weights(model_name_or_path)

        # Convert if needed
        if converter:
            raw_weights = converter.convert(raw_weights, model_config)
        elif weight_map:
            raw_weights = self._apply_weight_map(raw_weights, weight_map)

        # Convert to TTNN tensors
        ttnn_weights = self._convert_to_ttnn(raw_weights, model_config)

        return ttnn_weights

    def _load_raw_weights(self, model_name_or_path: str) -> Dict[str, torch.Tensor]:
        """Load raw weights from file or model hub"""
        path = Path(model_name_or_path)

        if path.is_file():
            # Load from single file
            if path.suffix == ".pt" or path.suffix == ".pth":
                return torch.load(path, map_location="cpu")
            elif path.suffix == ".safetensors":
                return safetensors.torch.load_file(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        elif path.is_dir():
            # Load from directory
            return self._load_from_directory(path)
        else:
            # Try to load from model hub
            return self._load_from_hub(model_name_or_path)

    def _load_from_directory(self, directory: Path) -> Dict[str, torch.Tensor]:
        """Load weights from a directory"""
        weights = {}

        # Check for index file
        index_file = directory / "pytorch_model.bin.index.json"
        if index_file.exists():
            # Sharded weights
            with open(index_file) as f:
                index = json.load(f)

            weight_map = index["weight_map"]
            loaded_files = set()

            for weight_name, filename in weight_map.items():
                if filename not in loaded_files:
                    filepath = directory / filename
                    if filepath.suffix == ".safetensors":
                        file_weights = safetensors.torch.load_file(filepath)
                    else:
                        file_weights = torch.load(filepath, map_location="cpu")
                    weights.update(file_weights)
                    loaded_files.add(filename)
        else:
            # Single file
            for pattern in ["*.bin", "*.pt", "*.pth", "*.safetensors"]:
                files = list(directory.glob(pattern))
                if files:
                    for file in files:
                        if file.suffix == ".safetensors":
                            file_weights = safetensors.torch.load_file(file)
                        else:
                            file_weights = torch.load(file, map_location="cpu")
                        weights.update(file_weights)
                    break

        return weights

    def _load_from_hub(self, model_name: str) -> Dict[str, torch.Tensor]:
        """Load weights from model hub"""
        # This would integrate with HuggingFace Hub or other model registries
        raise NotImplementedError(f"Model hub loading not implemented for: {model_name}")

    def _apply_weight_map(
        self,
        weights: Dict[str, torch.Tensor],
        weight_map: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """Apply weight name mapping"""
        mapped_weights = {}
        for old_name, new_name in weight_map.items():
            if old_name in weights:
                mapped_weights[new_name] = weights[old_name]
        return mapped_weights

    def _convert_to_ttnn(
        self,
        weights: Dict[str, torch.Tensor],
        model_config: Any,
    ) -> Dict[str, ttnn.Tensor]:
        """Convert PyTorch tensors to TTNN tensors"""
        ttnn_weights = {}

        for name, weight in tqdm(weights.items(), desc="Converting weights to TTNN"):
            # Get target layout and memory config
            layout = self._get_layout_for_weight(name, weight)
            memory_config = self._get_memory_config_for_weight(name, weight, model_config)

            # Convert to TTNN tensor
            ttnn_weight = ttnn.from_torch(
                weight,
                device=self.device,
                dtype=self.dtype,
                layout=layout,
                memory_config=memory_config,
            )

            ttnn_weights[name] = ttnn_weight

        return ttnn_weights

    def _get_layout_for_weight(self, name: str, weight: torch.Tensor) -> ttnn.Layout:
        """Determine layout for a weight tensor"""
        # Use TILE layout for matrix operations
        if len(weight.shape) >= 2:
            return ttnn.TILE_LAYOUT
        else:
            return ttnn.ROW_MAJOR_LAYOUT

    def _get_memory_config_for_weight(
        self,
        name: str,
        weight: torch.Tensor,
        model_config: Any,
    ) -> ttnn.MemoryConfig:
        """Determine memory configuration for a weight tensor"""
        # Default to DRAM for weights
        return ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        )

    def save_weights(
        self,
        weights: Dict[str, ttnn.Tensor],
        save_path: Path,
        save_format: str = "safetensors",
    ):
        """
        Save TTNN weights to disk.

        Args:
            weights: Dictionary of TTNN tensors
            save_path: Path to save weights
            save_format: Format to save in ("safetensors", "pt")
        """
        # Convert TTNN tensors back to PyTorch
        torch_weights = {}
        for name, weight in weights.items():
            torch_weights[name] = ttnn.to_torch(weight)

        # Save in requested format
        if save_format == "safetensors":
            safetensors.torch.save_file(torch_weights, save_path)
        else:
            torch.save(torch_weights, save_path)

    def load_cached_weights(
        self,
        cache_key: str,
    ) -> Optional[Dict[str, ttnn.Tensor]]:
        """Load weights from cache if available"""
        cache_path = self.cache_dir / f"{cache_key}.safetensors"
        if cache_path.exists():
            return self.load_pretrained_weights(str(cache_path), None)
        return None

    def cache_weights(
        self,
        weights: Dict[str, ttnn.Tensor],
        cache_key: str,
    ):
        """Save weights to cache"""
        cache_path = self.cache_dir / f"{cache_key}.safetensors"
        self.save_weights(weights, cache_path)
