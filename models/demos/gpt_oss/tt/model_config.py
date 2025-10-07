# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS ModelArgs class that's compatible with tt_transformers interface
"""

import os
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.gpt_oss.reference.hf_utils import get_state_dict, load_tokenizer


class ModelArgs:
    """GPT-OSS ModelArgs compatible with tt_transformers create_tt_model interface"""

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
    ):
        self.mesh_device = mesh_device
        self.instruct = instruct
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.optimizations = optimizations
        self.cache_hf = cache_hf

        # GPT-OSS specific paths - use single GPT_DIR environment variable
        # Default paths for available models
        default_models = [
            "/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B",
            "/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-120B",
        ]

        # Use first available model as default, or GPT_DIR environment variable override
        default_path = None
        for model_path in default_models:
            if os.path.exists(model_path):
                default_path = model_path
                break

        if default_path is None:
            default_path = default_models[-1]  # Fallback to first in list

        # Use single GPT_DIR environment variable for all paths
        gpt_dir = os.getenv("GPT_DIR", default_path)
        self.model_path = gpt_dir
        self.weights_path = gpt_dir

        logger.info(f"Using GPT-OSS model from: {self.model_path}")

        if self.dummy_weights:
            # Skip loading HF config for testing - use default values
            logger.info("Using dummy weights mode - skipping HuggingFace config loading")
            # Default GPT-OSS config values for testing
            self.hf_config = None
            self.vocab_size = 201088  # GPT-OSS vocab size
            self.n_layers = 32  # Default layer count
            self.head_dim = 64  # hidden_size // num_attention_heads = 2048 // 32
            self.rope_theta = 10000.0
            self.rope_scaling = None
        else:
            # Load HF config to get model parameters
            self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            # Set key attributes that tt_transformers expects
            self.vocab_size = self.hf_config.vocab_size
            self.n_layers = getattr(self.hf_config, "num_hidden_layers", 32)
            self.head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
            self.rope_theta = getattr(self.hf_config, "rope_theta", 10000.0)
            self.rope_scaling = None  # Keep simple like original GPT-OSS

        # Add missing attributes that Generator expects
        self.max_prefill_chunk_size = 2048  # Standard chunk size for prefill
        self.model_name = "GPT-OSS-120B" if "GPT-OSS-120B" in self.model_path else "GPT-OSS-20B"  # Model identifier
        self.max_context_len = max_seq_len  # Context length for tt_transformers compatibility

        if self.dummy_weights:
            # Skip tokenizer loading for testing
            self.tokenizer = None
            self.processor = None
        else:
            # Load tokenizer
            self.tokenizer = load_tokenizer(self.weights_path)
            self.processor = None  # GPT-OSS doesn't use vision processor

            # Add meta-compatible stop token list to the HF tokenizer (like tt_transformers does)
            if not "stop_tokens" in self.tokenizer.__dict__:
                self.tokenizer.stop_tokens = [self.tokenizer.eos_token_id]
                # Add common stop tokens for GPT-OSS
                if hasattr(self.tokenizer, "encode"):
                    try:
                        # Try to add <|eot_id|> if it exists (common in instruction models)
                        eot_tokens = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
                        if eot_tokens:
                            self.tokenizer.stop_tokens.extend(eot_tokens)
                    except:
                        pass  # Not all tokenizers have <|eot_id|>

    def encode_prompt(self, prompt_text, instruct=True, system_prompt_text=None):
        """
        Encode prompts using HuggingFace tokenizer with chat template
        Compatible with tt_transformers interface
        """
        chat = []
        if isinstance(prompt_text, str):
            if system_prompt_text:
                chat.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                chat.append({"role": "user", "content": prompt_text})
            return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
        else:
            # prompt_text is already a list of chat messages
            return self.tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)

    def load_state_dict(self):
        """Load model state dict compatible with tt_transformers"""
        if self.dummy_weights:
            # Return dummy state dict for testing
            return {}
        else:
            # Load actual GPT-OSS weights
            return get_state_dict(self.weights_path, "", dtype=torch.bfloat16)

    def weight_cache_path(self, dtype):
        """Return weight cache path for the model"""
        cache_dir = Path(self.model_path)  # Use same directory as model
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]

        if self.instruct:
            cache_path = cache_dir / f"tensor_cache_instruct_{dtype_str}_{self.mesh_device.shape}"
        else:
            cache_path = cache_dir / f"tensor_cache_{dtype_str}_{self.mesh_device.shape}"

        cache_path.mkdir(parents=True, exist_ok=True)
        return str(cache_path)

    def get_model_config(self):
        """Return model configuration dict"""
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
        }

    def get_state_dict_prefix(self, prefix, layer_idx):
        """Get state dict prefix for layer weights"""
        if layer_idx is None:
            return prefix
        return f"{prefix}layers.{layer_idx}."

    @property
    def max_grid_size(self):
        """Return maximum grid size for the device"""
        return ttnn.CoreGrid(y=8, x=8)  # Standard grid size
