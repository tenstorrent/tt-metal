"""
MiniCPM Weight Bridge

Converts MiniCPM safetensors weights to tt_transformers-compatible format.
Handles key renaming and component filtering.
"""

import torch
from typing import Dict
from loguru import logger
import sys
from pathlib import Path

# Add reference_pytorch to path
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_pytorch"))
from weight_loader import DiskBasedWeightLoader


class MiniCPMWeightBridge:
    """
    Bridge between MiniCPM safetensors and tt_transformers format.

    Component Weight Prefixes:
    - LLM: 'llm.' → strip to '' (tt_transformers expects no prefix)
    - Vision Encoder: 'vpm.' → keep as is
    - Vision Resampler: 'resampler.' → keep as is
    - Audio Encoder: 'apm.' → keep as is
    - Audio Projector: within 'apm.' → separate into audio_projection_layer
    - TTS: 'tts.' → keep as is
    """

    def __init__(self, model_name: str = "openbmb/MiniCPM-o-2_6"):
        self.weight_loader = DiskBasedWeightLoader()
        self.model_name = model_name

    def get_qwen_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get base Qwen LLM weights in tt_transformers format.

        Conversion:
        - 'llm.model.embed_tokens.weight' → 'model.embed_tokens.weight'
        - 'llm.model.layers.0.self_attn.q_proj.weight' → 'model.layers.0.self_attn.q_proj.weight'
        - Excludes: cross_attn, resampler, vpm, apm components

        Returns:
            Dict compatible with tt_transformers TtModelArgs.load_state_dict()
        """
        logger.info("Loading Qwen LLM weights from MiniCPM checkpoint...")

        # Get filtered base LLM weights (excludes multimodal components)
        base_weights = self.weight_loader.get_base_llm_weights()

        # Convert key format from MiniCPM to tt_transformers expected format
        tt_weights = {}
        for key, tensor in base_weights.items():
            if key.startswith("llm."):
                # Convert MiniCPM format to tt_transformers format
                new_key = key[4:]  # Remove 'llm.' prefix
                new_key = new_key.replace("model.embed_tokens.weight", "tok_embeddings.weight")
                new_key = new_key.replace("model.norm.weight", "norm.weight")
                new_key = new_key.replace("model.layers.", "layers.")
                new_key = new_key.replace("lm_head.weight", "output.weight")

                # Convert attention weights from self_attn.* to attention.*
                new_key = new_key.replace("self_attn.q_proj.weight", "attention.wq.weight")
                new_key = new_key.replace("self_attn.k_proj.weight", "attention.wk.weight")
                new_key = new_key.replace("self_attn.v_proj.weight", "attention.wv.weight")
                new_key = new_key.replace("self_attn.o_proj.weight", "attention.wo.weight")

                # Convert attention biases
                new_key = new_key.replace("self_attn.q_proj.bias", "attention.wq.bias")
                new_key = new_key.replace("self_attn.k_proj.bias", "attention.wk.bias")
                new_key = new_key.replace("self_attn.v_proj.bias", "attention.wv.bias")
                new_key = new_key.replace("self_attn.o_proj.bias", "attention.wo.bias")

                # Convert MLP weights from mlp.* to feed_forward.*
                new_key = new_key.replace("mlp.gate_proj.weight", "feed_forward.w1.weight")
                new_key = new_key.replace("mlp.up_proj.weight", "feed_forward.w3.weight")
                new_key = new_key.replace("mlp.down_proj.weight", "feed_forward.w2.weight")

                # Convert MLP biases
                new_key = new_key.replace("mlp.gate_proj.bias", "feed_forward.w1.bias")
                new_key = new_key.replace("mlp.up_proj.bias", "feed_forward.w3.bias")
                new_key = new_key.replace("mlp.down_proj.bias", "feed_forward.w2.bias")

                # Convert layer norms
                new_key = new_key.replace("input_layernorm.weight", "attention_norm.weight")
                new_key = new_key.replace("input_layernorm.bias", "attention_norm.bias")
                new_key = new_key.replace("post_attention_layernorm.weight", "ffn_norm.weight")
                new_key = new_key.replace("post_attention_layernorm.bias", "ffn_norm.bias")

                tt_weights[new_key] = tensor
                logger.debug(f"Converted: {key} → {new_key}")
            else:
                logger.warning(f"Unexpected key format: {key}")

        logger.info(f"Converted {len(tt_weights)} Qwen weights for tt_transformers")
        # Ensure embedding/output vocab size matches HF model config (pad/trim if necessary)
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            expected_vocab_size = getattr(config, "vocab_size", None)
        except Exception:
            expected_vocab_size = None

        if expected_vocab_size is not None:
            # Check tok_embeddings.weight and output.weight
            for key in ("tok_embeddings.weight", "output.weight"):
                if key in tt_weights:
                    logger.info(f"Found {key} in weights with shape {tt_weights[key].shape}")
                    tensor = tt_weights[key]
                    if tensor.ndim == 2 and tensor.shape[0] != expected_vocab_size:
                        hidden_dim = tensor.shape[1]
                        if tensor.shape[0] < expected_vocab_size:
                            logger.warning(
                                f"Vocab size mismatch for {key}: {tensor.shape[0]} -> padding to {expected_vocab_size}"
                            )
                            new_tensor = torch.zeros((expected_vocab_size, hidden_dim), dtype=tensor.dtype)
                            new_tensor[: tensor.shape[0], :] = tensor
                            tt_weights[key] = new_tensor
                        else:
                            logger.warning(
                                f"Vocab size mismatch for {key}: {tensor.shape[0]} -> trimming to {expected_vocab_size}"
                            )
                            tt_weights[key] = tensor[:expected_vocab_size, :].contiguous()
                    else:
                        logger.info(f"{key} vocab size matches: {tensor.shape[0]}")
                else:
                    logger.error(f"❌ {key} not found in converted weights!")

        return tt_weights

    def get_standard_qwen_weights(self, model_name="Qwen2.5-7B"):
        """
        Get standard Qwen weights directly from HuggingFace, converting to tt_transformers format.

        Args:
            model_name (str): Name of the Qwen model to load

        Returns:
            Dict[str, torch.Tensor]: Qwen weights in tt_transformers format
        """
        logger.info(f"Loading standard Qwen weights from {model_name}...")

        from transformers import AutoModelForCausalLM

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)

        # Get state dict
        state_dict = model.state_dict()

        logger.info("Converting standard Qwen weight keys to tt_transformers format...")
        # Convert key format from standard Qwen to tt_transformers expected format
        tt_weights = {}
        for key, tensor in state_dict.items():
            # Convert standard Qwen format to tt_transformers format
            new_key = key
            new_key = new_key.replace("model.embed_tokens.weight", "tok_embeddings.weight")
            new_key = new_key.replace("model.norm.weight", "norm.weight")
            new_key = new_key.replace("model.layers.", "layers.")
            new_key = new_key.replace("lm_head.weight", "output.weight")

            # Convert attention weights from self_attn.* to attention.*
            new_key = new_key.replace("self_attn.q_proj.weight", "attention.wq.weight")
            new_key = new_key.replace("self_attn.k_proj.weight", "attention.wk.weight")
            new_key = new_key.replace("self_attn.v_proj.weight", "attention.wv.weight")
            new_key = new_key.replace("self_attn.o_proj.weight", "attention.wo.weight")

            # Convert attention biases
            new_key = new_key.replace("self_attn.q_proj.bias", "attention.wq.bias")
            new_key = new_key.replace("self_attn.k_proj.bias", "attention.wk.bias")
            new_key = new_key.replace("self_attn.v_proj.bias", "attention.wv.bias")
            new_key = new_key.replace("self_attn.o_proj.bias", "attention.wo.bias")

            # Convert MLP weights
            new_key = new_key.replace("mlp.gate_proj.weight", "feed_forward.w1.weight")
            new_key = new_key.replace("mlp.up_proj.weight", "feed_forward.w3.weight")
            new_key = new_key.replace("mlp.down_proj.weight", "feed_forward.w2.weight")

            # Convert MLP biases
            new_key = new_key.replace("mlp.gate_proj.bias", "feed_forward.w1.bias")
            new_key = new_key.replace("mlp.up_proj.bias", "feed_forward.w3.bias")
            new_key = new_key.replace("mlp.down_proj.bias", "feed_forward.w2.bias")

            # Convert layer norms
            new_key = new_key.replace("input_layernorm.weight", "attention_norm.weight")
            new_key = new_key.replace("input_layernorm.bias", "attention_norm.bias")
            new_key = new_key.replace("post_attention_layernorm.weight", "ffn_norm.weight")
            new_key = new_key.replace("post_attention_layernorm.bias", "ffn_norm.bias")

            tt_weights[new_key] = tensor
            logger.debug(f"Converted: {key} → {new_key}")

        logger.info(f"Converted {len(tt_weights)} standard Qwen weights for tt_transformers")
        return tt_weights

    def get_vision_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get SigLIP vision encoder weights.

        Prefix: 'vpm.' (Vision Pre-processing Module)
        Output dim: 1152 (hidden_size)
        """
        logger.info("Loading SigLIP vision encoder weights...")
        vision_weights = self.weight_loader.get_component_weights(["vpm."])
        logger.info(f"Loaded {len(vision_weights)} vision encoder weights")
        return vision_weights

    def get_vision_resampler_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get vision resampler weights.

        Prefix: 'resampler.'
        Architecture: Perceiver-style cross-attention
        Input: (batch, vision_seq, 1152)
        Output: (batch, 32, 3584) ← Qwen embedding dim
        """
        logger.info("Loading vision resampler weights...")
        resampler_weights = self.weight_loader.get_component_weights(["resampler."])
        logger.info(f"Loaded {len(resampler_weights)} resampler weights")
        return resampler_weights

    def get_audio_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get Whisper audio encoder weights.

        Prefix: 'apm.' (Audio Processing Module)
        Output dim: 1024 (whisper hidden_size)
        """
        logger.info("Loading Whisper audio encoder weights...")
        audio_weights = self.weight_loader.get_component_weights(["apm."])

        # Separate encoder from projection layers
        encoder_weights = {}
        for key, tensor in audio_weights.items():
            # Keep only Whisper encoder weights (exclude projection layers)
            if not any(x in key for x in ["audio_projection_layer", "audio_avg_pooler"]):
                encoder_weights[key] = tensor

        logger.info(f"Loaded {len(encoder_weights)} audio encoder weights")
        return encoder_weights

    def get_audio_projector_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get audio projection layer weights.

        Components:
        - audio_projection_layer: Linear(896, 3584) ← Projects to Qwen dim
        - audio_avg_pooler: AvgPool1d(kernel=2, stride=2)

        Flow: Whisper(1024) → Linear(256) → AvgPool → Linear(3584)
        """
        logger.info("Loading audio projector weights...")
        audio_weights = self.weight_loader.get_component_weights(["apm."])

        # Extract only projection components
        projector_weights = {}
        for key, tensor in audio_weights.items():
            if "audio_projection_layer" in key or "audio_avg_pooler" in key:
                projector_weights[key] = tensor

        logger.info(f"Loaded {len(projector_weights)} audio projector weights")
        return projector_weights

    def get_tts_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get TTS module weights (DVAE + ChatTTS decoder).

        Prefix: 'tts.'
        Components: DVAE encoder/decoder + GFSQ quantizer
        """
        logger.info("Loading TTS weights...")
        tts_weights = self.weight_loader.get_component_weights(["tts."])
        logger.info(f"Loaded {len(tts_weights)} TTS weights")
        return tts_weights

    def get_all_component_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load all component weights at once.

        Returns:
            Dict with keys: 'qwen', 'vision_encoder', 'vision_resampler',
                          'audio_encoder', 'audio_projector', 'tts'
        """
        logger.info("Loading ALL MiniCPM component weights...")
        return {
            "qwen": self.get_qwen_weights(),
            "vision_encoder": self.get_vision_encoder_weights(),
            "vision_resampler": self.get_vision_resampler_weights(),
            "audio_encoder": self.get_audio_encoder_weights(),
            "audio_projector": self.get_audio_projector_weights(),
            "tts": self.get_tts_weights(),
        }
