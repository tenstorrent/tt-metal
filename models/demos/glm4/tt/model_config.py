# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import ModelArgs as BaseModelArgs
from loguru import logger  # Import logger


class Glm4ModelArgs(BaseModelArgs):
    """
    glm4 specific model arguments, inheriting from base ModelArgs
    handles loading glm4 specific parameters from config.json
    """

    def __init__(self, *args, **kwargs):
        # Initialize attributes before calling super() in case super() needs them
        self.is_glm4 = False
        self.partial_rotary_factor = 1.0  # Default to standard RoPE
        self.attention_bias = False  # Default for Llama-like models

        super().__init__(*args, **kwargs)
        # Note: super().__init__ will call _set_hf_params or _set_params if needed,
        # which populates self.config

        # Now parse the potentially populated self.config for GLM-4 specifics
        self._parse_glm4_config()

    def _parse_glm4_config(self):
        """Parses the loaded config for GLM-4 specific parameters."""
        if hasattr(self, "config") and self.config:
            config_dict = self.config  # Assuming self.config is the dict loaded from config.json

            is_glm_model_type = config_dict.get("model_type", "").lower() == "glm4"
            is_glm_name = "glm-4" in config_dict.get("_name_or_path", "").lower()

            if is_glm_model_type or is_glm_name:
                self.is_glm4 = True
                # GLM-4 default partial rotary factor is 0.5 if not specified
                self.partial_rotary_factor = config_dict.get("partial_rotary_factor", 0.5)
                # Check for attention bias in GLM-4 config
                self.attention_bias = config_dict.get("attention_bias", False)
                logger.info(
                    f"Detected GLM-4 model. Partial Rotary Factor: {self.partial_rotary_factor}, Attention Bias: {self.attention_bias}"
                )
            else:
                # Ensure defaults are set if not detected as GLM-4
                self.is_glm4 = False
                self.partial_rotary_factor = 1.0
                self.attention_bias = False
        else:
            logger.warning("Model config not found or empty, cannot parse for GLM-4 specifics.")

    # Override get_state_dict_prefix to handle Glm4Attention
    def get_state_dict_prefix(self, module_name, layer_num):
        text_prefix = "text_model." if self.is_vision() else ""
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "Glm4Attention": "attention",  # Add mapping for Glm4Attention
            "TransformerBlock": "",
            "Glm4TransformerBlock": "",  # Add mapping for Glm4TransformerBlock
            "": "",  # If no module is given, just get layer prefix
        }
        if module_name not in module_map:
            logger.warning(
                f"Module name {module_name} not found in default state dict prefix map. Returning empty prefix."
            )
            # Fallback or raise error? For now, return base prefix.
            return text_prefix + layer_prefix

        return text_prefix + layer_prefix + module_map[module_name]

    # TODO: Implement the .load() method assumed by demo.py
    # This method should read config.json from ckpt_dir and populate self attributes
    # similar to how _set_hf_params works in the base class.
    def load(self, ckpt_dir):
        logger.info(f"Loading config from {ckpt_dir}")
        # Simplified load - assumes config.json exists in ckpt_dir
        import json
        import os

        config_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
            self.config = loaded_config  # Store loaded config
            # Call _set_params_from_dict to populate attributes based on loaded config
            # This reuses the logic from the base class init
            self._set_params_from_dict(self.config)
            # Re-parse for GLM-4 specifics after populating base args
            self._parse_glm4_config()
        except Exception as e:
            logger.error(f"Failed to load or parse config from {config_path}: {e}")
            raise
        return self  # Return self for chaining
