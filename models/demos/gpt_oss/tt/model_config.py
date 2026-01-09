# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS ModelArgs class that's compatible with tt_transformers interface
"""

import os
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.tt_transformers.tt.common import (
    calculate_prefill_warmup_seq_lens,
    cap_seq_lens_to_max_prefill_chunk_size,
    get_base_model_name,
)
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format


class ModelArgs:
    """GPT-OSS ModelArgs compatible with tt_transformers create_tt_model interface"""

    def __init__(
        self,
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
    ):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        if optimizations is not None:
            logger.warning("GPT-OSS doesn't support any performance optimizations - ignoring optimizations argument")
        self.optimizations = None
        self.cache_hf = cache_hf

        # GPT-OSS specific paths - use HF_MODEL environment variable (tt_transformers standard)
        # Default paths are internal CI paths for automated testing
        default_models = [
            "/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b",  # Internal CI path
            "/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-120b",  # Internal CI path
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
        self.model_name = Path(self.model_path).name
        assert self.model_name in [
            "gpt-oss-20b",
            "gpt-oss-120b",
        ], f"Unrecognized model name {self.model_name} inferred from model path {self.model_path}. Make sure you're using standard huggingface naming convention for your model checkpoint e.g openai/gpt-oss-20b"  # Model identifier
        self.max_context_len = max_seq_len  # Context length for tt_transformers compatibility

        if self.dummy_weights:
            # Skip tokenizer loading for testing
            self.tokenizer = None
            self.processor = None
        else:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True)
            self.processor = None  # GPT-OSS doesn't use vision processor

        self.capped_warmup_seq_len = min(self.max_prefill_chunk_size, self.max_seq_len)
        self.trace_prefill_supported_seq_lens = self.get_trace_prefill_supported_seq_lens()

    def get_warmup_prefill_supported_seq_lens(self):
        DEFAULT_VALUE = self.capped_warmup_seq_len
        # This dictionary is used to override the default ceil warmup prefill value
        model_specific_ceil_warmup_lengths = {
            # e.g. "gpt-oss-120b": 4096
        }

        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(self.base_model_name, DEFAULT_VALUE)
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )

        to_warmup_seq_lens = self.filter_warmup_seq_lens(to_warmup_seq_lens)

        return to_warmup_seq_lens

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        # TODO: Add more model-specific filtering here
        # This filtering is based on the current PR's (https://github.com/tenstorrent/tt-metal/pull/33143) sequence lengths that are used for warmup

        # TODO: https://github.com/tenstorrent/tt-metal/issues/33722
        if self.model_name == "gpt-oss-120b":
            if 6144 in to_warmup_seq_lens:
                to_warmup_seq_lens.remove(6144)

        for seq_len in to_warmup_seq_lens:
            if seq_len >= 64 * 1024:
                to_warmup_seq_lens = to_warmup_seq_lens[: to_warmup_seq_lens.index(seq_len)]
                break
        return to_warmup_seq_lens

    @property
    def base_model_name(self):
        return get_base_model_name(self.model_name)

    def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
        """
        This function is used to determine if trace should be enabled for the prefill.
        Tracing is used only for certain sequence lengths, because for bigger sequence lengths, op2op gaps are already small, so we don't need tracing.
        # TODO: Support chunked prefill with tracing - https://github.com/tenstorrent/tt-metal/issues/32056
        """

        allowed_seq_lens = self.trace_prefill_supported_seq_lens

        return (
            prefill_seq_len in allowed_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
            and num_cached_tokens == 0
        )

    def get_trace_prefill_supported_seq_lens(self):
        # No supported sequence lengths for GPT-OSS model, see issue below
        # TODO: https://github.com/tenstorrent/tt-metal/issues/32818
        default_supported_seq_lens = {}

        # TODO: If no specific sequence lengths are listed for a model and device, the default one will be used (from the default_supported_seq_lens dictionary)
        model_specific_supported_seq_lens = {
            # exmaple : #base_model_name : {device_name : [sequence_lengths]}
        }

        model_name = self.model_name
        device_name = determine_device_name(self.mesh_device)

        # Try model-specific sequence lengths first
        result = model_specific_supported_seq_lens.get(model_name, {}).get(device_name)
        if result:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)

        # Fall back to default sequence lengths
        result = default_supported_seq_lens.get(device_name)
        if result:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)

        # No supported sequence lengths found, return empty list
        return []

    def encode_prompt(self, prompt_text, instruct=False, system_prompt_text=None):
        """
        Encode prompts using HuggingFace tokenizer with chat template
        Compatible with tt_transformers interface
        """
        assert not instruct, "GPT-OSS does not support instruct mode"
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

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True):
        """Load model state dict compatible with tt_transformers

        Args:
            weights_path (str or Path): Path to the model weights directory or file.
            dummy_weights (bool): If True, returns a dummy state dict for testing purposes.
            convert_to_meta_format (bool): If True, convert HF QKV weights to Meta format for RoPE.
                Set to False when loading for HuggingFace reference models.
        """
        if dummy_weights:
            # Return dummy state dict for testing
            return {}
        else:
            # Load actual GPT-OSS weights directly from safetensors files
            # Check if we have a cached torch_state_dict.pt file
            model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                torch_dtype="auto"
                # Note that the default setting is torch.dtype.float32, but model weights are
                # may come in any dtype. If the model's weights are in torch.dtype.bfloat16, this would result in 2x memory usage from an
                # unnecessary cast.
            )
            state_dict = model.state_dict()
            # Convert HF QKV weights to Meta format for RoPE compatibility (if requested)
            if convert_to_meta_format:
                logger.info("Converting QKV weights from HuggingFace to Meta format for RoPE")
                state_dict = convert_hf_qkv_to_meta_format(state_dict, model.config.head_dim)
            if state_dict["model.norm.weight"].dtype != torch.bfloat16:
                # Convert to bfloat16 if needed
                state_dict = {
                    k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                    for k, v in tqdm(state_dict.items(), desc="Converting to bfloat16")
                }
            return state_dict

    def weight_cache_path(self, dtype):
        """Return weight cache path for the model"""
        cache_dir = os.getenv("TT_CACHE_PATH")
        if cache_dir:
            cache_dir = Path(cache_dir)  # If we specify a TT_CACHE_PATH, use that for the cache
        else:
            cache_dir = Path(self.model_path)  # Use same directory as model
        logger.info(f"Cache directory: {cache_dir}")
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
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


def determine_device_name(mesh_device):
    """
    Determine device name based on number of devices and architecture.

    Args:
        mesh_device (MeshDevice): MeshDevice object

    Returns:
        str: Device name (e.g., "CPU", "N150", "P100", etc.)

    Raises:
        ValueError: If architecture or device count is unsupported
    """
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None  # CoreCoord with (x, y)

    if num_devices == 0:
        return "CPU"

    if is_blackhole():
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",  # P100 DRAM grid is 7x1, P150 is 8x1
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif is_wormhole_b0():
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices in dict_device_names:
        return dict_device_names[num_devices]
    else:
        raise ValueError(f"Unsupported number of devices: {num_devices} for {arch_name}")
