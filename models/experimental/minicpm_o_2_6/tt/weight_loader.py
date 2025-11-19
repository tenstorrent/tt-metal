# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight Loading Utilities for MiniCPM-o-2_6

Loads and converts weights from HuggingFace checkpoints to TTNN format:
- Whisper encoder weights
- Qwen2.5 LLM weights
- Audio Projector weights
- DVAE weights
"""

import torch
import transformers
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional, Union, List

# Import weight generators for fallback
from weight_generator import (
    generate_whisper_weights,
    generate_qwen_weights,
    generate_audio_projector_weights,
    generate_dvae_weights,
)


class MiniCPMWeightLoader:
    """Weight loader for MiniCPM-o-2_6 components"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize weight loader

        Args:
            cache_dir: Directory to cache downloaded weights
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "minicpm_weights"

        # Model configurations
        self.whisper_model_id = "openai/whisper-base"  # Smaller model for testing
        self.qwen_model_id = "Qwen/Qwen2.5-7B-Instruct"  # Use 7B for manageable size

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Weight loader initialized with cache dir: {self.cache_dir}")

    def load_minicpm_config(
        self, checkpoint_path="openbmb/MiniCPM-o-2_6", trust_remote_code: bool = True
    ) -> Dict[str, Any]:
        """
        Load MiniCPM-o config to get actual cross-attention layers and audio pool step.

        Args:
            checkpoint_path: HuggingFace model path or local path

        Returns:
            Dict with configuration parameters
        """
        try:
            logger.info(f"Loading MiniCPM-o config from {checkpoint_path}")

            # Try to load from HuggingFace
            config = transformers.AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code)

            # Extract key configuration parameters
            minicpm_config = {
                "cross_attention_layers": getattr(config, "cross_attention_layers", [8, 16, 24]),
                "audio_pool_step": getattr(config, "audio_pool_step", 2),
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                "max_position_embeddings": config.max_position_embeddings,
                "rope_theta": getattr(config, "rope_theta", 1000000.0),
                "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
                "attention_dropout": getattr(config, "attention_dropout", 0.0),
                # Audio config (Whisper)
                "audio_hidden_size": getattr(config, "audio_hidden_size", 1024),
                "audio_seq_len": getattr(config, "audio_seq_len", 128),
                # Vision config (SigLip)
                "vision_hidden_size": getattr(config, "vision_hidden_size", 1152),
                "vision_seq_len": getattr(config, "vision_seq_len", 256),
                # TTS config
                "tts_hidden_size": getattr(config, "tts_hidden_size", 768),
            }

            logger.info(
                f"✅ Loaded config: cross_attention_layers={minicpm_config['cross_attention_layers']}, "
                f"audio_pool_step={minicpm_config['audio_pool_step']}"
            )

            return minicpm_config

        except Exception as e:
            logger.warning(f"Failed to load config from HuggingFace: {e}")
            logger.info("Using default configuration")

            # Return default configuration matching reference implementation
            return {
                "cross_attention_layers": [8, 16, 24],  # Default from minicpm_o_model.py
                "audio_pool_step": 2,  # Default from modeling_minicpmo.py
                "vocab_size": 151700,
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "audio_hidden_size": 1024,
                "audio_seq_len": 128,
                "vision_hidden_size": 1152,
                "vision_seq_len": 256,
                "tts_hidden_size": 768,
            }

    def load_whisper_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Whisper encoder weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of Whisper weights
        """
        if use_huggingface:
            try:
                logger.info(f"Loading Whisper weights from {self.whisper_model_id} using safetensors")
                from safetensors import safe_open
                from huggingface_hub import hf_hub_download

                # Download and load safetensors file directly (more memory efficient)
                safetensors_path = hf_hub_download(self.whisper_model_id, "model.safetensors")
                weights = {}

                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    # Load only encoder weights (skip decoder weights)
                    for key in f.keys():
                        if key.startswith("model.encoder."):
                            weights[key] = f.get_tensor(key)

                logger.info(f"Loaded {len(weights)} Whisper encoder weights from safetensors")

            except Exception as e:
                logger.warning(f"Failed to load Whisper weights from safetensors: {e}")
                logger.info("Falling back to generated weights")
                weights = generate_whisper_weights()
        else:
            logger.info("Using generated Whisper weights")
            weights = generate_whisper_weights()

        return weights

    def load_qwen_weights(
        self, use_huggingface: bool = False, cross_attention_layers: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load Qwen2.5 LLM weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of Qwen weights
        """
        if use_huggingface:
            try:
                logger.info(f"Loading Qwen weights from {self.qwen_model_id}")
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.qwen_model_id,
                    torch_dtype=torch.float16,  # Load in half precision to save memory
                    device_map="cpu",  # Load on CPU first
                )

                weights = {}

                # Token embeddings
                weights["model.embed_tokens.weight"] = model.model.embed_tokens.weight

                # Layers
                for i, layer in enumerate(model.model.layers):
                    # Input layer norm
                    weights[f"model.layers.{i}.input_layernorm.weight"] = layer.input_layernorm.weight

                    # Self-attention
                    weights[f"model.layers.{i}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight
                    weights[f"model.layers.{i}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight
                    weights[f"model.layers.{i}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight
                    weights[f"model.layers.{i}.self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight

                    # Post-attention layer norm
                    weights[f"model.layers.{i}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight

                    # MLP
                    weights[f"model.layers.{i}.mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight
                    weights[f"model.layers.{i}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight
                    weights[f"model.layers.{i}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight

                    # Cross-attention (if present in MiniCPM layers)
                    # Note: Standard Qwen2.5 doesn't have cross-attention, but MiniCPM adds it
                    # We'll handle this in the generation logic

                # Final layer norm
                weights["model.norm.weight"] = model.model.norm.weight

                # LM head
                weights["lm_head.weight"] = model.lm_head.weight

                logger.info(f"Loaded Qwen weights: {len(weights)} parameters")

                # If MiniCPM added cross-attention layers, try to pull those parameters
                # from the MiniCPM snapshot in the HuggingFace cache.
                if cross_attention_layers:
                    try:
                        import re
                        from safetensors import safe_open

                        # Try to locate the MiniCPM snapshot directory in the HF cache
                        snapshot_root = (
                            Path.home() / ".cache" / "huggingface" / "models--openbmb--MiniCPM-o-2_6" / "snapshots"
                        )
                        found_cross = {}
                        if snapshot_root.exists():
                            # Search safetensors files in snapshot root and subdirectories
                            for fpath in snapshot_root.rglob("*.safetensors"):
                                try:
                                    with safe_open(str(fpath), framework="pt", device="cpu") as f:
                                        for key in f.keys():
                                            if (
                                                "cross_attn" in key
                                                or "cross-attn" in key
                                                or "crossattn" in key
                                                or ".cross." in key
                                            ):
                                                tensor = f.get_tensor(key)
                                                # Normalize layer index and suffix
                                                m = re.search(r"layers\.(\d+)\.(.*)", key)
                                                if m:
                                                    layer_idx = int(m.group(1))
                                                    suffix = m.group(2)
                                                    normalized = f"model.layers.{layer_idx}.cross_attn.{suffix}"
                                                    found_cross[normalized] = tensor
                                except Exception:
                                    # ignore safetensors that fail to read
                                    continue

                        # Merge found cross-attention keys into weights
                        if found_cross:
                            weights.update(found_cross)
                            logger.info(f"Found and merged {len(found_cross)} cross-attention parameters from snapshot")
                        else:
                            # If none found, raise to trigger fallback or explicit error handling
                            logger.warning("No cross-attention parameters found in local MiniCPM snapshots")
                    except Exception as e_inner:
                        logger.warning(f"Failed to extract MiniCPM cross-attention params: {e_inner}")

            except Exception as e:
                logger.warning(f"Failed to load Qwen weights from HuggingFace: {e}")
                logger.info("Falling back to generated weights")
                weights = generate_qwen_weights(cross_attention_layers=cross_attention_layers)
        else:
            logger.info("Using generated Qwen weights")
            weights = generate_qwen_weights(cross_attention_layers=cross_attention_layers)

        return weights

    def load_audio_projector_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Audio Projector weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of Audio Projector weights
        """
        if use_huggingface:
            # Placeholder - would need actual MiniCPM checkpoint
            logger.warning("HuggingFace Audio Projector weights not available, using generated weights")
            weights = generate_audio_projector_weights()
        else:
            logger.info("Using generated Audio Projector weights")
            weights = generate_audio_projector_weights()

        return weights

    def load_dvae_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load DVAE weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of DVAE weights
        """
        if use_huggingface:
            # Placeholder - would need actual MiniCPM checkpoint
            logger.warning("HuggingFace DVAE weights not available, using generated weights")
            weights = generate_dvae_weights()
        else:
            logger.info("Using generated DVAE weights")
            weights = generate_dvae_weights()

        return weights

    def load_vision_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Vision (SigLip) weights
        """
        if use_huggingface:
            logger.warning("HuggingFace Vision weights not implemented yet")
            use_huggingface = False

        if not use_huggingface:
            logger.info("Using generated Vision weights")
            from weight_generator import generate_vision_weights

            weights = generate_vision_weights()

        return weights

    def load_resampler_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Resampler weights
        """
        if use_huggingface:
            logger.warning("HuggingFace Resampler weights not implemented yet")
            use_huggingface = False

        if not use_huggingface:
            logger.info("Using generated Resampler weights")
            from weight_generator import generate_resampler_weights

            weights = generate_resampler_weights()

        return weights

    def load_chattts_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load ChatTTS decoder weights
        """
        if use_huggingface:
            logger.warning("HuggingFace ChatTTS weights not implemented yet")
            use_huggingface = False

        if not use_huggingface:
            logger.info("Using generated ChatTTS weights")
            from weight_generator import generate_chattts_weights

            weights = generate_chattts_weights()

        return weights

    def load_from_official_checkpoint(
        self,
        checkpoint_path: str = "openbmb/MiniCPM-o-2_6",
        components: Optional[List[str]] = None,
        trust_remote_code: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load weights directly from official MiniCPM-o-2_6 checkpoint.

        This method downloads the safetensors files and extracts component weights,
        then converts them to TTNN-compatible formats.

        Args:
            checkpoint_path: HuggingFace model path
            components: List of components to load (default: all)

        Returns:
            Dictionary with component weights in TTNN format
        """
        from weight_converter import convert_component_weights

        if components is None:
            # Use 'siglip' for vision component to match downloader naming
            components = ["qwen", "whisper", "audio_projector", "siglip", "chattts", "dvae"]

        logger.info(f"Loading official checkpoint weights for components: {components}")

        all_weights = {}

        try:
            # Try to use the component downloader script
            from scripts.download_component_weights import download_component_weights

            for component in components:
                logger.info(f"Loading {component} weights...")

                try:
                    # Download component weights
                    component_weights = download_component_weights(
                        component=component,
                        checkpoint_path=checkpoint_path,
                        output_dir=self.cache_dir,
                        trust_remote_code=trust_remote_code,
                    )

                    if component_weights:
                        # Convert to TTNN format
                        ttnn_weights = convert_component_weights(component_weights, component)
                        all_weights[component] = ttnn_weights
                        logger.info(f"✅ Loaded and converted {component}: {len(ttnn_weights)} tensors")
                    else:
                        logger.warning(f"No {component} weights found, using generated weights")
                        all_weights[component] = self._get_fallback_weights(component)

                except Exception as e:
                    logger.warning(f"Failed to load {component} from checkpoint: {e}")
                    logger.info(f"Using generated weights for {component}")
                    all_weights[component] = self._get_fallback_weights(component)

        except ImportError:
            logger.warning("Component downloader not available, using generated weights for all components")
            for component in components:
                all_weights[component] = self._get_fallback_weights(component)

        total_params = sum(len(component_weights) for component_weights in all_weights.values())
        logger.info(f"Loaded weights for {len(all_weights)} components: {total_params} total parameters")

        return all_weights

    def _get_fallback_weights(self, component: str) -> Dict[str, torch.Tensor]:
        """Get fallback generated weights for a component."""
        if component == "qwen":
            config = self.load_minicpm_config(trust_remote_code=True)
            cross_attention_layers = config.get("cross_attention_layers", [8, 16, 24])
            return self.load_qwen_weights(use_huggingface=False, cross_attention_layers=cross_attention_layers)
        elif component == "whisper":
            return self.load_whisper_weights(use_huggingface=False)
        elif component == "audio_projector":
            return self.load_audio_projector_weights(use_huggingface=False)
        elif component == "vision" or component == "siglip":
            return self.load_vision_weights(use_huggingface=False)
        elif component == "resampler":
            return self.load_resampler_weights(use_huggingface=False)
        elif component == "chattts":
            return self.load_chattts_weights(use_huggingface=False)
        elif component == "dvae":
            return self.load_dvae_weights(use_huggingface=False)
        else:
            logger.warning(f"Unknown component: {component}")
            return {}

    def load_all_weights(self, use_huggingface: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load weights for all MiniCPM components

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary containing weights for all components
        """
        logger.info("Loading all MiniCPM-o-2_6 weights...")

        if use_huggingface:
            # Try to load from official checkpoint
            try:
                return self.load_from_official_checkpoint()
            except Exception as e:
                logger.warning(f"Failed to load from official checkpoint: {e}")
                logger.info("Falling back to generated weights")

        # Load config to get correct cross-attention layers
        config = self.load_minicpm_config(trust_remote_code=True)
        cross_attention_layers = config["cross_attention_layers"]

        weights = {
            "whisper": self.load_whisper_weights(use_huggingface),
            "qwen": self.load_qwen_weights(use_huggingface, cross_attention_layers),
            "audio_projector": self.load_audio_projector_weights(use_huggingface),
            "dvae": self.load_dvae_weights(use_huggingface),
        }

        total_params = sum(len(component_weights) for component_weights in weights.values())
        logger.info(f"Loaded weights for all components: {total_params} total parameters")
        logger.info(f"Using cross-attention layers: {cross_attention_layers}")

        return weights

    def save_weights(self, weights: Dict[str, Dict[str, torch.Tensor]], output_dir: Union[str, Path]):
        """
        Save weights to disk

        Args:
            weights: Weights dictionary from load_all_weights()
            output_dir: Directory to save weights
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving weights to {output_dir}")

        for component_name, component_weights in weights.items():
            component_file = output_dir / f"{component_name}_weights.pt"
            torch.save(component_weights, component_file)
            logger.info(f"Saved {component_name} weights: {len(component_weights)} parameters")

    def load_weights_from_disk(self, weights_dir: Union[str, Path]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load weights from disk

        Args:
            weights_dir: Directory containing saved weights

        Returns:
            Dictionary containing weights for all components
        """
        weights_dir = Path(weights_dir)

        weights = {}
        for component_name in ["whisper", "qwen", "audio_projector", "dvae"]:
            weight_file = weights_dir / f"{component_name}_weights.pt"
            if weight_file.exists():
                weights[component_name] = torch.load(weight_file, map_location="cpu")
                logger.info(f"Loaded {component_name} weights from disk: {len(weights[component_name])} parameters")
            else:
                logger.warning(f"Weight file not found: {weight_file}")
                weights[component_name] = {}

        return weights


def create_weight_loader(cache_dir: Optional[str] = None) -> MiniCPMWeightLoader:
    """
    Create and return a MiniCPM weight loader

    Args:
        cache_dir: Cache directory for downloaded weights

    Returns:
        Configured MiniCPMWeightLoader instance
    """
    return MiniCPMWeightLoader(cache_dir)


# Convenience functions for quick weight loading
def load_minicpm_config(
    checkpoint_path: str = "openbmb/MiniCPM-o-2_6", cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load MiniCPM configuration

    Args:
        checkpoint_path: HuggingFace model path
        cache_dir: Cache directory

    Returns:
        MiniCPM configuration dictionary
    """
    loader = create_weight_loader(cache_dir)
    return loader.load_minicpm_config(checkpoint_path)


def load_minicpm_weights(
    use_huggingface: bool = False, cache_dir: Optional[str] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convenience function to load all MiniCPM weights

    Args:
        use_huggingface: Whether to load from HuggingFace
        cache_dir: Cache directory

    Returns:
        All MiniCPM weights
    """
    loader = create_weight_loader(cache_dir)
    return loader.load_all_weights(use_huggingface)


def load_minicpm_component_weights(
    component: str, use_huggingface: bool = False, cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load weights for a specific component

    Args:
        component: Component name ('whisper', 'qwen', 'audio_projector', 'dvae')
        use_huggingface: Whether to load from HuggingFace
        cache_dir: Cache directory

    Returns:
        Component weights
    """
    loader = create_weight_loader(cache_dir)

    if component == "whisper":
        return loader.load_whisper_weights(use_huggingface)
    elif component == "qwen":
        return loader.load_qwen_weights(use_huggingface)
    elif component == "audio_projector":
        return loader.load_audio_projector_weights(use_huggingface)
    elif component == "dvae":
        return loader.load_dvae_weights(use_huggingface)
    else:
        raise ValueError(f"Unknown component: {component}")


"""
Weight Loading Utilities for MiniCPM-o-2_6

Loads and converts weights from HuggingFace checkpoints to TTNN format:
- Whisper encoder weights
- Qwen2.5 LLM weights
- Audio Projector weights
- DVAE weights
"""

import torch
import transformers
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional, Union, List

# Import weight generators for fallback
from weight_generator import (
    generate_whisper_weights,
    generate_qwen_weights,
    generate_audio_projector_weights,
    generate_dvae_weights,
)


class MiniCPMWeightLoader:
    """Weight loader for MiniCPM-o-2_6 components"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize weight loader

        Args:
            cache_dir: Directory to cache downloaded weights
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "minicpm_weights"

        # Model configurations
        self.whisper_model_id = "openai/whisper-base"  # Smaller model for testing
        self.qwen_model_id = "Qwen/Qwen2.5-7B-Instruct"  # Use 7B for manageable size

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Weight loader initialized with cache dir: {self.cache_dir}")

    def load_minicpm_config(
        self, checkpoint_path="openbmb/MiniCPM-o-2_6", trust_remote_code: bool = True
    ) -> Dict[str, Any]:
        """
        Load MiniCPM-o config to get actual cross-attention layers and audio pool step.

        Args:
            checkpoint_path: HuggingFace model path or local path

        Returns:
            Dict with configuration parameters
        """
        try:
            logger.info(f"Loading MiniCPM-o config from {checkpoint_path}")

            # Try to load from HuggingFace
            config = transformers.AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code)

            # Extract key configuration parameters
            minicpm_config = {
                "cross_attention_layers": getattr(config, "cross_attention_layers", [8, 16, 24]),
                "audio_pool_step": getattr(config, "audio_pool_step", 2),
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                "max_position_embeddings": config.max_position_embeddings,
                "rope_theta": getattr(config, "rope_theta", 1000000.0),
                "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
                "attention_dropout": getattr(config, "attention_dropout", 0.0),
                # Audio config (Whisper)
                "audio_hidden_size": getattr(config, "audio_hidden_size", 1024),
                "audio_seq_len": getattr(config, "audio_seq_len", 128),
                # Vision config (SigLip)
                "vision_hidden_size": getattr(config, "vision_hidden_size", 1152),
                "vision_seq_len": getattr(config, "vision_seq_len", 256),
                # TTS config
                "tts_hidden_size": getattr(config, "tts_hidden_size", 768),
            }

            logger.info(
                f"✅ Loaded config: cross_attention_layers={minicpm_config['cross_attention_layers']}, "
                f"audio_pool_step={minicpm_config['audio_pool_step']}"
            )

            return minicpm_config

        except Exception as e:
            logger.warning(f"Failed to load config from HuggingFace: {e}")
            logger.info("Using default configuration")

            # Return default configuration matching reference implementation
            return {
                "cross_attention_layers": [8, 16, 24],  # Default from minicpm_o_model.py
                "audio_pool_step": 2,  # Default from modeling_minicpmo.py
                "vocab_size": 151700,
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "audio_hidden_size": 1024,
                "audio_seq_len": 128,
                "vision_hidden_size": 1152,
                "vision_seq_len": 256,
                "tts_hidden_size": 768,
            }

    def load_whisper_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Whisper encoder weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of Whisper weights
        """
        if use_huggingface:
            try:
                logger.info(f"Loading Whisper weights from {self.whisper_model_id} using safetensors")
                from safetensors import safe_open
                from huggingface_hub import hf_hub_download

                # Download and load safetensors file directly (more memory efficient)
                safetensors_path = hf_hub_download(self.whisper_model_id, "model.safetensors")
                weights = {}

                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    # Load only encoder weights (skip decoder weights)
                    for key in f.keys():
                        if key.startswith("model.encoder."):
                            weights[key] = f.get_tensor(key)

                logger.info(f"Loaded {len(weights)} Whisper encoder weights from safetensors")

            except Exception as e:
                logger.warning(f"Failed to load Whisper weights from safetensors: {e}")
                logger.info("Falling back to generated weights")
                weights = generate_whisper_weights()
        else:
            logger.info("Using generated Whisper weights")
            weights = generate_whisper_weights()

        return weights

    def load_qwen_weights(
        self, use_huggingface: bool = False, cross_attention_layers: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load Qwen2.5 LLM weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of Qwen weights
        """
        if use_huggingface:
            try:
                logger.info(f"Loading Qwen weights from {self.qwen_model_id}")
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.qwen_model_id,
                    torch_dtype=torch.float16,  # Load in half precision to save memory
                    device_map="cpu",  # Load on CPU first
                )

                weights = {}

                # Token embeddings
                weights["model.embed_tokens.weight"] = model.model.embed_tokens.weight

                # Layers
                for i, layer in enumerate(model.model.layers):
                    # Input layer norm
                    weights[f"model.layers.{i}.input_layernorm.weight"] = layer.input_layernorm.weight

                    # Self-attention
                    weights[f"model.layers.{i}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight
                    weights[f"model.layers.{i}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight
                    weights[f"model.layers.{i}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight
                    weights[f"model.layers.{i}.self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight

                    # Post-attention layer norm
                    weights[f"model.layers.{i}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight

                    # MLP
                    weights[f"model.layers.{i}.mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight
                    weights[f"model.layers.{i}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight
                    weights[f"model.layers.{i}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight

                    # Cross-attention (if present in MiniCPM layers)
                    # Note: Standard Qwen2.5 doesn't have cross-attention, but MiniCPM adds it
                    # We'll handle this in the generation logic

                # Final layer norm
                weights["model.norm.weight"] = model.model.norm.weight

                # LM head
                weights["lm_head.weight"] = model.lm_head.weight

                logger.info(f"Loaded Qwen weights: {len(weights)} parameters")

            except Exception as e:
                logger.warning(f"Failed to load Qwen weights from HuggingFace: {e}")
                logger.info("Falling back to generated weights")
                weights = generate_qwen_weights(cross_attention_layers=cross_attention_layers)
        else:
            logger.info("Using generated Qwen weights")
            weights = generate_qwen_weights(cross_attention_layers=cross_attention_layers)

        return weights

    def load_audio_projector_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Audio Projector weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of Audio Projector weights
        """
        if use_huggingface:
            # Placeholder - would need actual MiniCPM checkpoint
            logger.warning("HuggingFace Audio Projector weights not available, using generated weights")
            weights = generate_audio_projector_weights()
        else:
            logger.info("Using generated Audio Projector weights")
            weights = generate_audio_projector_weights()

        return weights

    def load_dvae_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load DVAE weights

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary of DVAE weights
        """
        if use_huggingface:
            # Placeholder - would need actual MiniCPM checkpoint
            logger.warning("HuggingFace DVAE weights not available, using generated weights")
            weights = generate_dvae_weights()
        else:
            logger.info("Using generated DVAE weights")
            weights = generate_dvae_weights()

        return weights

    def load_vision_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Vision (SigLip) weights
        """
        if use_huggingface:
            logger.warning("HuggingFace Vision weights not implemented yet")
            use_huggingface = False

        if not use_huggingface:
            logger.info("Using generated Vision weights")
            from weight_generator import generate_vision_weights

            weights = generate_vision_weights()

        return weights

    def load_resampler_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load Resampler weights
        """
        if use_huggingface:
            logger.warning("HuggingFace Resampler weights not implemented yet")
            use_huggingface = False

        if not use_huggingface:
            logger.info("Using generated Resampler weights")
            from weight_generator import generate_resampler_weights

            weights = generate_resampler_weights()

        return weights

    def load_chattts_weights(self, use_huggingface: bool = False) -> Dict[str, torch.Tensor]:
        """
        Load ChatTTS decoder weights
        """
        if use_huggingface:
            logger.warning("HuggingFace ChatTTS weights not implemented yet")
            use_huggingface = False

        if not use_huggingface:
            logger.info("Using generated ChatTTS weights")
            from weight_generator import generate_chattts_weights

            weights = generate_chattts_weights()

        return weights

    def load_from_official_checkpoint(
        self,
        checkpoint_path: str = "openbmb/MiniCPM-o-2_6",
        components: Optional[List[str]] = None,
        trust_remote_code: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load weights directly from official MiniCPM-o-2_6 checkpoint.

        This method downloads the safetensors files and extracts component weights,
        then converts them to TTNN-compatible formats.

        Args:
            checkpoint_path: HuggingFace model path
            components: List of components to load (default: all)

        Returns:
            Dictionary with component weights in TTNN format
        """
        from weight_converter import convert_component_weights

        if components is None:
            # Use 'siglip' for vision component to match downloader naming
            components = ["qwen", "whisper", "audio_projector", "siglip", "chattts", "dvae"]

        logger.info(f"Loading official checkpoint weights for components: {components}")

        all_weights = {}

        try:
            # Try to use the component downloader script
            from scripts.download_component_weights import download_component_weights

            for component in components:
                logger.info(f"Loading {component} weights...")

                try:
                    # Download component weights
                    component_weights = download_component_weights(
                        component=component,
                        checkpoint_path=checkpoint_path,
                        output_dir=self.cache_dir,
                        trust_remote_code=trust_remote_code,
                    )

                    if component_weights:
                        # Convert to TTNN format
                        ttnn_weights = convert_component_weights(component_weights, component)
                        all_weights[component] = ttnn_weights
                        logger.info(f"✅ Loaded and converted {component}: {len(ttnn_weights)} tensors")
                    else:
                        logger.warning(f"No {component} weights found, using generated weights")
                        all_weights[component] = self._get_fallback_weights(component)

                except Exception as e:
                    logger.warning(f"Failed to load {component} from checkpoint: {e}")
                    logger.info(f"Using generated weights for {component}")
                    all_weights[component] = self._get_fallback_weights(component)

        except ImportError:
            logger.warning("Component downloader not available, using generated weights for all components")
            for component in components:
                all_weights[component] = self._get_fallback_weights(component)

        total_params = sum(len(component_weights) for component_weights in all_weights.values())
        logger.info(f"Loaded weights for {len(all_weights)} components: {total_params} total parameters")

        return all_weights

    def _get_fallback_weights(self, component: str) -> Dict[str, torch.Tensor]:
        """Get fallback generated weights for a component."""
        if component == "qwen":
            config = self.load_minicpm_config(trust_remote_code=True)
            cross_attention_layers = config.get("cross_attention_layers", [8, 16, 24])
            return self.load_qwen_weights(use_huggingface=False, cross_attention_layers=cross_attention_layers)
        elif component == "whisper":
            return self.load_whisper_weights(use_huggingface=False)
        elif component == "audio_projector":
            return self.load_audio_projector_weights(use_huggingface=False)
        elif component == "vision" or component == "siglip":
            return self.load_vision_weights(use_huggingface=False)
        elif component == "resampler":
            return self.load_resampler_weights(use_huggingface=False)
        elif component == "chattts":
            return self.load_chattts_weights(use_huggingface=False)
        elif component == "dvae":
            return self.load_dvae_weights(use_huggingface=False)
        else:
            logger.warning(f"Unknown component: {component}")
            return {}

    def load_all_weights(self, use_huggingface: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load weights for all MiniCPM components

        Args:
            use_huggingface: Whether to load from HuggingFace (True) or generate random (False)

        Returns:
            Dictionary containing weights for all components
        """
        logger.info("Loading all MiniCPM-o-2_6 weights...")

        if use_huggingface:
            # Try to load from official checkpoint
            try:
                return self.load_from_official_checkpoint()
            except Exception as e:
                logger.warning(f"Failed to load from official checkpoint: {e}")
                logger.info("Falling back to generated weights")

        # Load config to get correct cross-attention layers
        config = self.load_minicpm_config(trust_remote_code=True)
        cross_attention_layers = config["cross_attention_layers"]

        weights = {
            "whisper": self.load_whisper_weights(use_huggingface),
            "qwen": self.load_qwen_weights(use_huggingface, cross_attention_layers),
            "audio_projector": self.load_audio_projector_weights(use_huggingface),
            "dvae": self.load_dvae_weights(use_huggingface),
        }

        total_params = sum(len(component_weights) for component_weights in weights.values())
        logger.info(f"Loaded weights for all components: {total_params} total parameters")
        logger.info(f"Using cross-attention layers: {cross_attention_layers}")

        return weights

    def save_weights(self, weights: Dict[str, Dict[str, torch.Tensor]], output_dir: Union[str, Path]):
        """
        Save weights to disk

        Args:
            weights: Weights dictionary from load_all_weights()
            output_dir: Directory to save weights
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving weights to {output_dir}")

        for component_name, component_weights in weights.items():
            component_file = output_dir / f"{component_name}_weights.pt"
            torch.save(component_weights, component_file)
            logger.info(f"Saved {component_name} weights: {len(component_weights)} parameters")

    def load_weights_from_disk(self, weights_dir: Union[str, Path]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load weights from disk

        Args:
            weights_dir: Directory containing saved weights

        Returns:
            Dictionary containing weights for all components
        """
        weights_dir = Path(weights_dir)

        weights = {}
        for component_name in ["whisper", "qwen", "audio_projector", "dvae"]:
            weight_file = weights_dir / f"{component_name}_weights.pt"
            if weight_file.exists():
                weights[component_name] = torch.load(weight_file, map_location="cpu")
                logger.info(f"Loaded {component_name} weights from disk: {len(weights[component_name])} parameters")
            else:
                logger.warning(f"Weight file not found: {weight_file}")
                weights[component_name] = {}

        return weights


def create_weight_loader(cache_dir: Optional[str] = None) -> MiniCPMWeightLoader:
    """
    Create and return a MiniCPM weight loader

    Args:
        cache_dir: Cache directory for downloaded weights

    Returns:
        Configured MiniCPMWeightLoader instance
    """
    return MiniCPMWeightLoader(cache_dir)


# Convenience functions for quick weight loading
def load_minicpm_config(
    checkpoint_path: str = "openbmb/MiniCPM-o-2_6", cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load MiniCPM configuration

    Args:
        checkpoint_path: HuggingFace model path
        cache_dir: Cache directory

    Returns:
        MiniCPM configuration dictionary
    """
    loader = create_weight_loader(cache_dir)
    return loader.load_minicpm_config(checkpoint_path)


def load_minicpm_weights(
    use_huggingface: bool = False, cache_dir: Optional[str] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convenience function to load all MiniCPM weights

    Args:
        use_huggingface: Whether to load from HuggingFace
        cache_dir: Cache directory

    Returns:
        All MiniCPM weights
    """
    loader = create_weight_loader(cache_dir)
    return loader.load_all_weights(use_huggingface)


def load_minicpm_component_weights(
    component: str, use_huggingface: bool = False, cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load weights for a specific component

    Args:
        component: Component name ('whisper', 'qwen', 'audio_projector', 'dvae')
        use_huggingface: Whether to load from HuggingFace
        cache_dir: Cache directory

    Returns:
        Component weights
    """
    loader = create_weight_loader(cache_dir)

    if component == "whisper":
        return loader.load_whisper_weights(use_huggingface)
    elif component == "qwen":
        return loader.load_qwen_weights(use_huggingface)
    elif component == "audio_projector":
        return loader.load_audio_projector_weights(use_huggingface)
    elif component == "dvae":
        return loader.load_dvae_weights(use_huggingface)
    else:
        raise ValueError(f"Unknown component: {component}")
