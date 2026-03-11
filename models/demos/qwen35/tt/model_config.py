# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from models.tt_transformers.tt.model_config import ModelArgs


class Qwen35ModelArgs(ModelArgs):
    """
    Model arguments for Qwen3.5-9B.

    Extends the base ModelArgs to add Qwen3.5-specific parameters:
    - Hybrid attention layers (linear + full)
    - Linear attention parameters (GatedDeltaNet)
    - M-RoPE (Multimodal RoPE) parameters
    """

    def __init__(self, mesh_device, *args, **kwargs):
        # Set model-specific defaults before calling super
        kwargs.setdefault("instruct", True)

        super().__init__(mesh_device, *args, **kwargs)

        # Parse Qwen3.5-specific config from HuggingFace
        self._set_qwen35_params()

        logger.info(f"Initialized Qwen35ModelArgs for {self.model_name}")
        logger.info(f"Total layers: {self.n_layers}")
        logger.info(f"Layer types: {self.layer_types[:8]}... (showing first 8)")
        logger.info(
            f"Linear attention - Key heads: {self.linear_num_key_heads}, Value heads: {self.linear_num_value_heads}"
        )
        logger.info(f"Full attention - Q heads: {self.n_heads}, KV heads: {self.n_kv_heads}")

    def _set_qwen35_params(self):
        """Set Qwen3.5-specific parameters from HF config."""
        # Access the text config (merged by parent class)
        text_config = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config

        # === Layer Types (Hybrid Attention) ===
        # Pattern: ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]
        if hasattr(text_config, "layer_types") and text_config.layer_types:
            self.layer_types = text_config.layer_types
        else:
            # Fallback: generate default pattern if not in config
            logger.warning("layer_types not found in config, generating default pattern (3 linear + 1 full)")
            interval = getattr(text_config, "full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if (i + 1) % interval else "full_attention" for i in range(self.n_layers)
            ]

        # === Linear Attention Parameters ===
        self.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
        self.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        self.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        self.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)

        # Computed dimensions
        self.linear_key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_value_dim = self.linear_num_value_heads * self.linear_value_head_dim

        # === M-RoPE (Multimodal RoPE) Parameters ===
        rope_params = getattr(text_config, "rope_parameters", {})
        if isinstance(rope_params, dict):
            self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
            self.partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)
            self.mrope_interleaved = rope_params.get("mrope_interleaved", True)
        else:
            # Fallback defaults
            self.mrope_section = [11, 11, 10]
            self.partial_rotary_factor = 0.25
            self.mrope_interleaved = True

        # Validate M-RoPE section sum equals rotary dim
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        mrope_sum = sum(self.mrope_section)
        if mrope_sum != rotary_dim:
            logger.warning(
                f"M-RoPE section sum ({mrope_sum}) != rotary_dim ({rotary_dim}). " f"Adjusting mrope_section to match."
            )
            # Distribute the difference evenly
            diff = rotary_dim - mrope_sum
            self.mrope_section[0] += diff

    def get_layer_type(self, layer_idx: int) -> str:
        """Get the attention type for a specific layer."""
        if layer_idx < 0 or layer_idx >= len(self.layer_types):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(self.layer_types)})")
        return self.layer_types[layer_idx]

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """Check if layer uses linear attention."""
        return self.get_layer_type(layer_idx) == "linear_attention"

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Check if layer uses full attention."""
        return self.get_layer_type(layer_idx) == "full_attention"

    def count_layer_types(self):
        """Count the number of each layer type."""
        from collections import Counter

        return Counter(self.layer_types)

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        """
        Map module names to HuggingFace state dict keys.

        Matches parent ModelArgs signature with required layer_num parameter.
        For top-level components (Embedding, LMHead, Norm), pass None as layer_num.

        Args:
            module_name: Component name (e.g., "MLP", "LinearAttention", "FullAttention")
            layer_num: Layer index (required; use None for top-level components)
            is_vision: Whether this is a vision model component

        Returns:
            String prefix for accessing weights in state_dict
        """
        if is_vision:
            # Vision model state dict prefix (if we implement vision later)
            return self._get_vision_state_dict_prefix(module_name, layer_num)

        # Text model state dict prefix
        # Note: For AutoModelForCausalLM, keys use "model.layers" not "model.language_model.layers"
        layer_prefix = f"model.layers.{layer_num}." if layer_num is not None else "model."

        module_map = {
            "Embedding": "embed_tokens",
            "LMHead": "lm_head",
            "Norm": "norm",
            "MLP": "mlp",
            "FullAttention": "self_attn",
            "LinearAttention": "linear_attn",
            "InputLayerNorm": "input_layernorm",
            "PostAttentionLayerNorm": "post_attention_layernorm",
        }

        return layer_prefix + module_map.get(module_name, "")

    def _get_vision_state_dict_prefix(self, module_name, layer_num=None):
        """Get state dict prefix for vision components."""
        vision_prefix = "visual"
        if layer_num is not None:
            block_prefix = f"{vision_prefix}.blocks.{layer_num}"
            module_map = {
                "VisionAttention": f"{block_prefix}.attn",
                "VisionMLP": f"{block_prefix}.mlp",
                "Norm1": f"{block_prefix}.norm1",
                "Norm2": f"{block_prefix}.norm2",
            }
            return module_map.get(module_name, block_prefix)
        else:
            module_map = {
                "VisionTransformer": vision_prefix,
                "PatchMerger": f"{vision_prefix}.merger",
            }
            return module_map.get(module_name, vision_prefix)

    # === Reference Model Accessors (for testing) ===

    def reference_model(self):
        """Load full HuggingFace model."""
        try:
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        except ImportError:
            # Fallback to AutoModel if Qwen3_5 not available (older transformers)
            logger.warning("Qwen3_5ForConditionalGeneration not found, using AutoModel with trust_remote_code=True")
            from transformers import AutoModelForCausalLM

            model_class = AutoModelForCausalLM

        if self.cached_hf_model is None or not self.cache_hf_flag:
            logger.info(f"Loading HuggingFace Qwen3.5 model from {self.CKPT_DIR}")
            self.cached_hf_model = model_class.from_pretrained(
                self.CKPT_DIR,
                torch_dtype="auto",
                device_map="cpu",  # Load to CPU to avoid CUDA memory issues
                trust_remote_code=True,  # Allow loading custom model code if present
            )
        return self.cached_hf_model

    def reference_text_model(self):
        """Get the text (language) model component."""
        return self.reference_model().model.language_model

    def reference_vision_model(self):
        """Get the vision model component."""
        if hasattr(self.reference_model(), "visual"):
            return self.reference_model().visual
        else:
            logger.warning("Vision model not available in this Qwen3.5 variant")
            return None

    def reference_linear_attention(self, layer_idx=0):
        """Get a linear attention layer from reference model."""
        if not self.is_linear_attention_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} is not a linear attention layer")
        return self.reference_text_model().layers[layer_idx].linear_attn

    def reference_full_attention(self, layer_idx=3):
        """Get a full attention layer from reference model."""
        if not self.is_full_attention_layer(layer_idx):
            # Find the first full attention layer
            for i, layer_type in enumerate(self.layer_types):
                if layer_type == "full_attention":
                    logger.warning(f"Layer {layer_idx} is not full attention, using layer {i} instead")
                    layer_idx = i
                    break
        return self.reference_text_model().layers[layer_idx].self_attn

    def reference_mlp(self, layer_idx=0):
        """Get an MLP layer from reference model."""
        return self.reference_text_model().layers[layer_idx].mlp

    def reference_decoder_layer(self, layer_idx=0):
        """Get a full decoder layer from reference model."""
        return self.reference_text_model().layers[layer_idx]
