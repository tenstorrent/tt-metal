# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS ModelArgs class that's compatible with tt_transformers interface
"""

import gc
import json
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
        if self.max_batch_size > 32:
            assert (
                self.max_batch_size % self.mesh_device.shape[0] == 0
            ), "max_batch_size must be divisible by the number of device rows"
            self.max_local_batch_size = self.max_batch_size // self.mesh_device.shape[0]
        else:
            self.max_local_batch_size = self.max_batch_size
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

        logger.info(
            f"Using GPT-OSS model from: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

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

        self.disable_batched_prefill = True
        self.capped_warmup_seq_len = 2048
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
            "gpt-oss-120b": {
                "T3K": [128],
                "TG": [128],
            },
            "gpt-oss-20b": {
                "T3K": [128],
                "TG": [128],
            }
            # exmaple : #base_model_name : {device_name : [sequence_lengths]}
        }

        model_name = self.model_name
        device_name = determine_device_name(self.mesh_device)

        # If there is no entry for a model in model_specific_supported_seq_lens, use the entry in default_supported_seq_lens
        result = model_specific_supported_seq_lens.get(model_name, {}).get(
            device_name, default_supported_seq_lens.get(device_name)
        )

        if result is not None:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)
        else:
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
            encoded = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
        else:
            # prompt_text is already a list of chat messages
            encoded = self.tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)

        # Normalize whatever apply_chat_template(tokenize=True) returns into a flat List[int].
        # Across tokenizer/transformers versions this may be a List[int], a tokenizers.Encoding,
        # a list of Encodings, or a BatchEncoding/dict ({"input_ids": ...}). The GPT-OSS fast
        # tokenizer in CI returns a non-list form, which made downstream torch.tensor(...) raise
        # "Could not infer dtype of tokenizers.Encoding".
        raw_type = type(encoded).__name__
        # BatchEncoding / dict -> take input_ids
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"] if "input_ids" in encoded else getattr(encoded, "input_ids")

        def _to_ids(obj):
            if hasattr(obj, "ids"):  # tokenizers.Encoding
                return list(obj.ids)
            if isinstance(obj, (list, tuple)):
                flat = []
                for item in obj:
                    flat.append(item) if isinstance(item, int) else flat.extend(_to_ids(item))
                return flat
            return obj

        encoded = _to_ids(encoded)
        if not (isinstance(encoded, list) and (len(encoded) == 0 or isinstance(encoded[0], int))):
            logger.warning(
                f"[gpt-oss encode_prompt] unexpected token container: raw={raw_type}, "
                f"normalized={type(encoded).__name__}"
            )
        return encoded

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
            # Load actual GPT-OSS weights directly from safetensors files.
            #
            # GPT-OSS ships MXFP4-quantized weights (see configs/*/config.json). On a CPU host
            # (CI loads weights on host before pushing them to the Tenstorrent device) transformers
            # dequantizes the MXFP4 experts during from_pretrained. With torch_dtype="auto" that
            # dequant intermediate lands in fp32, ~doubling the resident host footprint AND forcing
            # the bf16 conversion pass below to allocate a second full copy of the state dict --
            # enough to OOM / hang the host for the 20B & 120B demos (#48509, #48508). Loading
            # straight to bf16 makes dequant target bf16 and turns that pass into a no-op (the dict
            # comprehension then rebinds references rather than copying tensors).
            model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            head_dim = model.config.head_dim
            state_dict = model.state_dict()
            # Drop the HF module graph/buffers now that we hold the weight tensors. state_dict shares
            # storage with the params, so after this the peak host footprint is bounded by the bf16
            # weights themselves rather than weights + a live HF model object.
            del model
            gc.collect()
            # Convert HF QKV weights to Meta format for RoPE compatibility (if requested)
            if convert_to_meta_format:
                logger.info("Converting QKV weights from HuggingFace to Meta format for RoPE")
                state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
            # Safety net: ensure bf16. With the bf16 load above this is a no-op for every tensor
            # (references are reused); it only casts genuine fp32 stragglers.
            if state_dict["model.norm.weight"].dtype != torch.bfloat16:
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

    # Name of the marker file dropped into a weight-cache directory once every weight
    # for that (model, dtype, mesh shape) has been materialized to disk.
    WEIGHT_CACHE_MARKER = ".weights_complete"
    # Cache-format version embedded in the marker. Bump this whenever the set/naming/layout of
    # cached weight tensors changes in a way that an existing cache would not satisfy (e.g. a
    # weight tensor is added or renamed without changing the layer count). A marker written by an
    # older format is then rejected -> the run cold-loads and regenerates the cache, rather than
    # skipping the load and hard-failing in ttnn.as_tensor(None, ...) on a missing .tensorbin.
    # (Mirrors DeepSeek's WEIGHT_CACHE_FORMAT_VERSION in deepseek_v3/utils/weight_config.py.)
    WEIGHT_CACHE_FORMAT_VERSION = 1

    def weight_cache_is_complete(self, dtype):
        """True when the on-disk ttnn weight cache for this (model, dtype, mesh shape) was
        fully built by a previous run.

        When True, ttnn.as_tensor loads every weight from its cached .tensorbin and the HF
        state_dict is never read, so the caller can skip the expensive from_pretrained host
        load entirely (the load that OOMs/hangs during prefill, #48509) without needing the
        manual --skip-model-load flag. Set GPT_OSS_FORCE_MODEL_LOAD=1 to force a fresh load
        (e.g. to regenerate the cache)."""
        if os.getenv("GPT_OSS_FORCE_MODEL_LOAD") == "1":
            return False
        cache_path = self.weight_cache_path(dtype)
        marker = cache_path / self.WEIGHT_CACHE_MARKER
        if not marker.is_file():
            return False
        try:
            meta = json.loads(marker.read_text())
        except (ValueError, OSError):
            return False
        # Reject a stale marker: an older cache format, a different model, or a partial
        # (num_layers-limited) build whose cache does not cover the full model we are about to
        # construct. A rejected marker falls back to a cold load (which regenerates the cache)
        # rather than skipping the load and crashing on a missing/renamed .tensorbin.
        if meta.get("format_version") != self.WEIGHT_CACHE_FORMAT_VERSION:
            return False
        if meta.get("model_name") != self.model_name or meta.get("n_layers") != self.n_layers:
            return False
        # Belt-and-suspenders: the cache dir must still actually hold tensor files.
        return any(cache_path.glob("*.tensorbin"))

    def mark_weight_cache_complete(self, dtype):
        """Record that the ttnn weight cache for this (model, dtype, mesh shape) was fully
        built, so subsequent runs can skip the HF state_dict load (see weight_cache_is_complete)."""
        cache_path = self.weight_cache_path(dtype)
        marker = cache_path / self.WEIGHT_CACHE_MARKER
        try:
            marker.write_text(
                json.dumps(
                    {
                        "format_version": self.WEIGHT_CACHE_FORMAT_VERSION,
                        "model_name": self.model_name,
                        "n_layers": self.n_layers,
                        "dtype": str(dtype),
                    }
                )
            )
            logger.info(f"Marked ttnn weight cache complete: {marker}")
        except OSError as e:
            logger.warning(f"Could not write weight-cache completion marker {marker}: {e}")

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
