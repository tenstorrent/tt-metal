# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS ModelArgs class that's compatible with tt_transformers interface
"""

import os
from glob import glob
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import ttnn


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

        # GPT-OSS specific paths - use HF_MODEL environment variable (tt_transformers standard)
        # Default paths are internal CI paths for automated testing
        default_models = [
            "/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B",  # Internal CI path
            "/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-120B",  # Internal CI path
        ]

        # Use first available model as default, or HF_MODEL environment variable override
        default_path = None
        for model_path in default_models:
            if os.path.exists(model_path):
                default_path = model_path
                break

        if default_path is None:
            default_path = default_models[-1]  # Fallback to first in list

        # Use HF_MODEL environment variable (consistent with tt_transformers)
        dir = os.getenv("HF_MODEL", default_path)
        self.model_path = dir
        self.weights_path = dir

        logger.info(f"Using GPT-OSS model from: {self.model_path}")

        if self.dummy_weights:
            # Skip loading HF config for testing - use default values
            logger.info("Using dummy weights mode - skipping HuggingFace config loading")

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
        self.max_prefill_chunk_size = 128 * 1024
        self.model_name = "GPT-OSS-120B" if "GPT-OSS-120B" in self.model_path else "GPT-OSS-20B"  # Model identifier
        self.max_context_len = max_seq_len  # Context length for tt_transformers compatibility

        if self.dummy_weights:
            # Skip tokenizer loading for testing
            self.tokenizer = None
            self.processor = None
        else:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True)
            self.processor = None  # GPT-OSS doesn't use vision processor

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
            # Load actual GPT-OSS weights directly from safetensors files
            # Check if we have a cached torch_state_dict.pt file
            torch_state_dict_path = os.path.join(self.weights_path, "torch_state_dict.pt")

            if os.path.exists(torch_state_dict_path):
                # Load from cached file
                weights_dict = torch.load(torch_state_dict_path)
            else:
                # Load from safetensors files
                safetensors_filepaths = sorted(glob(f"{self.weights_path}/*.safetensors"))
                weights_dict = {}
                for filepath in tqdm(safetensors_filepaths, desc="Loading weights"):
                    weights_dict.update(load_file(filepath))

                # Cache for future use
                torch.save(weights_dict, torch_state_dict_path)

            # Convert to bfloat16 if needed
            if torch.bfloat16 != torch.float32:
                weights_dict = {
                    k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                    for k, v in tqdm(weights_dict.items(), desc="Converting to bfloat16")
                }

            return weights_dict

    def weight_cache_path(self, dtype):
        """Return weight cache path for the model"""
        cache_dir = Path(self.model_path)  # Use same directory as model
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]

        if self.instruct:
            cache_path = cache_dir / f"tensor_cache_instruct_{dtype_str}_{self.mesh_device.shape}"
        else:
            cache_path = cache_dir / f"tensor_cache_{dtype_str}_{self.mesh_device.shape}"

        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

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
