# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import math
import os
from enum import Enum, auto
from pathlib import Path
from typing import Tuple

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0, nearest_32
from models.tt_transformers.tt.common import (
    calculate_hidden_dim,
    calculate_prefill_warmup_seq_lens,
    cap_seq_lens_to_max_prefill_chunk_size,
    encode_prompt_hf,
    get_base_model_name,
    get_out_subblock_w,
    nearest_multiple,
    num_to_core_range_set,
    rope_scaling_model_factory,
)
from models.tt_transformers.tt.load_checkpoints import convert_vision_meta_to_hf  # Minimal addition for Mistral vision
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_hf_to_meta_mllama,
    convert_meta_to_hf,
    convert_vision_hf_to_meta,
    load_hf_state_dict,
    load_meta_state_dict,
    reverse_permute,
    standardize_hf_keys,
    standardize_hf_keys_multimodal,
)

# file names for performance and accuracy mode override files
PERFORMANCE_DECODER_CONFIG_FILENAME = "performance_decoder_config.json"
ACCURACY_DECODER_CONFIG_FILENAME = "accuracy_decoder_config.json"


class CheckpointType(Enum):
    Meta = auto()
    HuggingFace = auto()


class TensorGroup(Enum):
    FF1_FF3 = "ff1_3"
    FF2 = "ff2"
    WQKV = "wqkv"
    WO = "wo"
    KV_CACHE = "kv_cache"
    ACTIVATION = "activation"


class PrecisionSetting(Enum):
    BFP4 = "bfp4"
    BFP8 = "bfp8"
    BF16 = "bf16"


class OpGroup(Enum):
    """
    LI_* are linear operator groups
    SDPA_* are scaled_dot_product_attention operator groups
    """

    LI_FF1_FF3 = "li_ff1_3"
    LI_FF2 = "li_ff2"
    LI_QKV_DECODE = "li_qkv_decode"
    LI_O_DECODE = "li_o_decode"
    SDPA_DECODE = "sdpa_decode"
    LI_QKV_PREFILL = "li_qkv_prefill"
    LI_O_PREFILL = "li_o_prefill"
    SDPA_PREFILL = "sdpa_prefill"
    ACCURACY = "accuracy"  # This is a special group for accuracy mode, not an actual operator group


class MathFidelitySetting(Enum):
    LOFI = "lofi"
    HIFI2 = "hifi2"
    HIFI2_NA = "hifi2na"  # na specified `packer_l1_acc=False` and `fp32_dest_acc_en=False` in compute kernel config
    HIFI2_FP16 = "hifi2fp16"  # fp16 specified `fp32_dest_acc_en=False` in compute kernel config
    HIFI4 = "hifi4"
    HIFI4_FP32 = "hifi4fp32"


class ModelOptimizations:
    @classmethod
    def accuracy(cls, model_name):
        """Configuration optimized for accuracy
        70B+ models still use bfp4 MLPs and BFP8 attention in this configuration
        """
        base_model_name = get_base_model_name(model_name)
        if base_model_name in ["Llama-3.1-70B", "Llama-3.2-90B", "DeepSeek-R1-Distill-Llama-70B", "Qwen2.5-72B"]:
            logger.info(
                f"{model_name} is >70B and large models test insensitive precision, using BFP4 MLPs and BFP8 attention even in accuracy mode"
            )
            inst = cls(
                {
                    "TensorPrecision": {TensorGroup.FF1_FF3: PrecisionSetting.BFP4},
                    "OpFidelity": {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI},
                }
            )
        else:
            if (
                base_model_name.startswith("Llama-3")
                or base_model_name.startswith("Mistral-7B")
                or base_model_name.startswith("Phi-3-mini")
                or base_model_name.startswith("phi-4")
            ):
                if model_name.startswith("phi-4"):
                    logger.info(
                        f"Model {model_name} is running out of DRAM memory for weight fetching under standard accuracy settings, using BFP8 for WQKV"
                    )
                logger.info(
                    f"Llama 3, Mistral 7B and Phi3-mini models test insensitive to attention precision, using BFP8 attention and kv-cache with FP16 MLP accumulation even in accuracy mode"
                )
                settings = {
                    "TensorPrecision": {
                        TensorGroup.WQKV: PrecisionSetting.BFP8,
                        TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
                        TensorGroup.WO: PrecisionSetting.BFP8,
                    },
                    "OpFidelity": {
                        OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                        OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
                    },
                }
                if model_name.startswith("Phi-3-mini"):  # TODO: Only do this for N150
                    logger.info(
                        f"Model {model_name} is running out of L1 memory under standard accuracy settings, using FP16 accumulate in attention prefill QKV Matmul"
                    )
                    settings["OpFidelity"][OpGroup.LI_QKV_PREFILL] = MathFidelitySetting.HIFI2_FP16
                inst = cls(settings)
            else:
                inst = cls(
                    {
                        "TensorPrecision": {
                            TensorGroup.WQKV: PrecisionSetting.BF16,
                            TensorGroup.KV_CACHE: PrecisionSetting.BF16,
                            TensorGroup.WO: PrecisionSetting.BF16,
                        },
                        "OpFidelity": {
                            OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                            OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                            OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                            OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                            OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                            OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
                        },
                    }
                )
        inst.__name__ = "accuracy"
        return inst

    @classmethod
    def performance(cls, model_name):
        """Configuration optimized for performance
        All models use bfp4 in FF1 and FF3 MLPs in this configuration
        """
        base_model_name = get_base_model_name(model_name)
        if base_model_name in ["Qwen2.5-7B", "Qwen2.5-VL-7B"]:
            logger.info(
                f"Model {model_name} is degraded under standard high-performance settings, using BF16 attention and BFP8 MLP"
            )
            inst = cls(
                {
                    "TensorPrecision": {
                        TensorGroup.WQKV: PrecisionSetting.BF16,
                        TensorGroup.KV_CACHE: PrecisionSetting.BF16,
                        TensorGroup.WO: PrecisionSetting.BF16,
                    },
                    "OpFidelity": {
                        OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                        OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                        OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                        OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                        OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                        OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
                    },
                }
            )
        else:
            settings = {
                "TensorPrecision": {TensorGroup.FF1_FF3: PrecisionSetting.BFP4},
                "OpFidelity": {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI},
            }
            if model_name.startswith("Phi-3-mini"):  # TODO: Only do this for N150
                logger.info(
                    f"Model {model_name} is running out of L1 memory under standard high-performance settings, using FP16 accumulate in attention prefill QKV Matmul"
                )
                settings["OpFidelity"][OpGroup.LI_QKV_PREFILL] = MathFidelitySetting.HIFI2_FP16
            inst = cls(settings)
        inst.__name__ = "performance"
        return inst

    def __init__(self, settings: dict = None):
        if settings:
            self._validate_settings(settings)

        self._opt_settings = self._default_settings()
        self._names = {}
        for key, enum_type in (("TensorPrecision", TensorGroup), ("OpFidelity", OpGroup)):
            self._opt_settings[key].update((settings or {}).get(key, {}))
            curr = self._opt_settings[key]
            self._names[key] = ", ".join(
                [f"{k.value}: {curr[k].value if curr[k] else 'mixed'}" for k in list(enum_type)]
            )

        self._full_name = (
            "precision_cfg = {"
            + self._names["TensorPrecision"]
            + "}, fidelity_cfg = {"
            + self._names["OpFidelity"]
            + "}"
        )
        # NOTE: self.__name__ is used as section header in PERF.md; It is also used by, for example test_llama_accuracy.py to look for comparative results in PERF.md
        self.__name__ = self._full_name

        # TODO: maybe we could warn about some unwanted settings here

    def _validate_settings(self, settings: dict):
        # Check that only valid top-level keys are used
        valid_keys = {"TensorPrecision", "OpFidelity"}
        invalid_keys = set(settings.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid settings keys: {invalid_keys}. Must be one of {valid_keys}")

        # Validate TensorPrecision settings
        if "TensorPrecision" in settings:
            for key, value in settings["TensorPrecision"].items():
                if not isinstance(key, TensorGroup):
                    raise ValueError(f"Invalid TensorPrecision key: {key}. Must be a TensorGroup enum value")
                if not isinstance(value, PrecisionSetting):
                    raise ValueError(f"Invalid TensorPrecision value: {value}. Must be a PrecisionSetting enum value")

        # Validate OpFidelity settings
        if "OpFidelity" in settings:
            for key, value in settings["OpFidelity"].items():
                if not isinstance(key, OpGroup):
                    raise ValueError(f"Invalid OpFidelity key: {key}. Must be an OpGroup enum value")
                if not isinstance(value, MathFidelitySetting):
                    raise ValueError(f"Invalid OpFidelity value: {value}. Must be a MathFidelitySetting enum value")

    def _default_settings(self):
        """Default is BFP8/HIFI2 everywhere, activation follows input type (usually BF16)
        Only exceptions:
        - SDPA runs in HIFI4 during prefill (still HIFI2 during decode)
        """
        return {
            "TensorPrecision": {
                # MLP
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                # Attention
                TensorGroup.WQKV: PrecisionSetting.BFP8,
                TensorGroup.WO: PrecisionSetting.BFP8,
                TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
                # Activation across whole model
                TensorGroup.ACTIVATION: None,  # this signals that original dtype should be used
            },
            "OpFidelity": {
                # MLP linear operators - BFP8 with FP16 accumulation to save L1
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
                # Attention operators -- linear and scaled_dot_product_attention, in decode and prefill modes
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI2,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI2,  # FP32 accumulate is important here
                OpGroup.ACCURACY: MathFidelitySetting.HIFI4_FP32,
            },
        }

    @property
    def tensor_dtype_settings(self):
        return self._opt_settings["TensorPrecision"]

    @property
    def op_fidelity_settings(self):
        return self._opt_settings["OpFidelity"]


def parse_optimizations(string):
    """
    Parse the optimizations full name and return a ModelOptimizations instance.
    """
    # Find the precision and fidelity config sections
    precision_start = string.find("precision_cfg")
    fidelity_start = string.find("fidelity_cfg")

    if precision_start == -1 and fidelity_start == -1:
        raise ValueError("String must contain either precision_cfg or fidelity_cfg")

    # Extract the config dictionaries between { }
    def extract_config(start_idx, cfg_name):
        open_brace = string.find("{", start_idx)
        if open_brace == -1:
            raise ValueError(f"Missing opening brace for {cfg_name}")

        close_brace = string.find("}", open_brace)
        if close_brace == -1:
            raise ValueError(f"Missing closing brace for {cfg_name}")

        return string[open_brace + 1 : close_brace].strip()

    precision_dict = extract_config(precision_start, "precision_cfg") if precision_start != -1 else {}
    fidelity_dict = extract_config(fidelity_start, "fidelity_cfg") if fidelity_start != -1 else {}

    # Create ModelOptimizations instance with the parsed configs
    settings = {"TensorPrecision": {}, "OpFidelity": {}}

    # Parse precision config
    for pair in precision_dict.split(","):
        if ":" not in pair:
            raise ValueError("Invalid format - missing ':' separator")
        key, value = pair.split(":")
        key = TensorGroup(key.strip())
        value = value.strip()
        if key == TensorGroup.ACTIVATION and value == "mixed":
            # special case for activation's mixed precision, which is the default configuration
            continue

        settings["TensorPrecision"][key] = PrecisionSetting(value)

    # Parse fidelity config
    for pair in fidelity_dict.split(","):
        if ":" not in pair:
            raise ValueError("Invalid format - missing ':' separator")
        key, value = pair.split(":")
        key = OpGroup(key.strip())
        value = MathFidelitySetting(value.strip())
        settings["OpFidelity"][key] = value

    model_opt = ModelOptimizations(settings)

    def apply_settings(model_args):
        return DecodersPrecision(model_args.n_layers, model_args.model_name, model_opt)

    apply_settings.__name__ = model_opt.__name__
    return apply_settings


def parse_decoder_json(json_file_path, default_optimization=ModelOptimizations.performance):
    """
    Reads a JSON file and returns a DecodersPrecision instance.
    """
    if not json_file_path:
        return None

    json_file_path = Path(json_file_path)
    if not json_file_path.exists():
        raise FileNotFoundError(f"JSON configuration file not found: {json_file_path}")

    try:
        with open(json_file_path, "r") as f:
            config_data = json.load(f)

        if "decoders" not in config_data:
            raise ValueError("Invalid JSON format: Missing 'decoders' key")

        num_decoders = max(int(decoder_id) for decoder_id in config_data["decoders"].keys()) + 1
        placeholder_model_name = "model"
        decoder_conf = default_optimization(placeholder_model_name)
        default_tensor_dtype_settings = decoder_conf.tensor_dtype_settings
        default_op_fidelity_settings = decoder_conf.op_fidelity_settings
        decoders_precision = DecodersPrecision(num_decoders, placeholder_model_name, decoder_conf)

        for decoder_id, settings in config_data["decoders"].items():
            decoder_id = int(decoder_id)

            tensor_precision = (
                {TensorGroup[key]: PrecisionSetting[value] for key, value in settings.get("precision_cfg").items()}
                if "precision_cfg" in settings
                else default_tensor_dtype_settings
            )

            op_fidelity = (
                {OpGroup[key]: MathFidelitySetting[value] for key, value in settings.get("fidelity_cfg").items()}
                if "fidelity_cfg" in settings
                else default_op_fidelity_settings
            )

            custom_opt = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            decoders_precision.set_decoder_conf(decoder_id, custom_opt)

        return decoders_precision

    except Exception as e:
        raise ValueError(f"Error loading JSON configuration: {e}")


class ModelArgs:
    OP_KEYS = (
        # Embedding
        "EMB_WEIGHTS",
        # Feed forward
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        # Attention
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "ATTN_OUTPUT",
        "ATTN_W_LAYOUT",
        # Decoder
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
        # MoE
        "GATE_W_LAYOUT",
        "GATE_WEIGHTS",
        "GATE_MM_OUTPUT",
    )

    LOCAL_LLAMA_PARAMS = {
        "LLAMA3_2_1B_PARAMS": "models/tt_transformers/model_params/Llama-3.2-1B-Instruct",
        "LLAMA3_2_3B_PARAMS": "models/tt_transformers/model_params/Llama-3.2-3B-Instruct",
        "LLAMA3_1_8B_PARAMS": "models/tt_transformers/model_params/Llama-3.1-8B-Instruct",
        "LLAMA3_2_11B_PARAMS": "models/tt_transformers/model_params/Llama-3.2-11B-Vision-Instruct",
        "LLAMA3_1_70B_PARAMS": "models/tt_transformers/model_params/Llama-3.1-70B-Instruct",
        "LLAMA3_2_90B_PARAMS": "models/tt_transformers/model_params/Llama-3.2-90B-Vision-Instruct",
    }

    LOCAL_HF_PARAMS = {
        "Llama-3.1-8B-Instruct": "models/tt_transformers/model_params/Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct": "models/tt_transformers/model_params/Llama-3.1-70B-Instruct",
        "Llama-3.2-1B-Instruct": "models/tt_transformers/model_params/Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct": "models/tt_transformers/model_params/Llama-3.2-3B-Instruct",
        "Llama-3.2-11B-Instruct": "models/tt_transformers/model_params/Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-11B-Vision-Instruct": "models/tt_transformers/model_params/Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Instruct": "models/tt_transformers/model_params/Llama-3.2-90B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct": "models/tt_transformers/model_params/Llama-3.2-90B-Vision-Instruct",
        "Mistral-7B-Instruct-v0.3": "models/tt_transformers/model_params/Mistral-7B-Instruct-v0.3",
        "Qwen2.5-VL-3B-Instruct": "models/tt_transformers/model_params/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-32B-Instruct": "models/tt_transformers/model_params/Qwen2.5-VL-32B-Instruct",
        "Qwen2.5-VL-72B-Instruct": "models/tt_transformers/model_params/Qwen2.5-VL-72B-Instruct",
    }

    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,  # Set to False to reduce memory usage by not caching HF model
    ):
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.arch_name = ttnn.get_arch_name()
        self.dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None  # CoreCoord with (x, y)

        self.device_name = determine_device_name(self.mesh_device)

        logger.info(f"Inferring device name: {self.device_name}")
        device = mesh_device if mesh_device is not None else None
        self.cluster_shape = list(mesh_device.shape) if mesh_device is not None else None
        self.is_galaxy = self.num_devices == 32

        self.model_name = "Unknown"  # Llama model name will be dependent on the checkpoint directory
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.tile_size = 32
        self.is_70b = False
        self.is_90b = False
        self.fuse_qkv = False
        self.fuse_mlp = False
        self.trust_remote_code_hf = False
        self.from_hf_url = False  # updated below if true
        self.prefill_len_cutoff = 512 if is_blackhole() else 1024
        self.dummy_weights = dummy_weights
        self.cache_hf_flag = cache_hf  # Whether to cache HF model to avoid multiple loads (uses extra memory)
        self.cached_hf_model = None  # Save any HF model object to avoid loading it multiple times for reference methods

        self.rms_norm_add_unit_offset = False
        self.embed_scale = None

        assert not os.getenv(
            "FAKE_DEVICE"
        ), "FAKE_DEVICE has been renamed to MESH_DEVICE for consistency with vLLM, please update your environment variables and run again."

        # Remove trailing slashes so basename gets the right model name
        HF_MODEL = os.getenv("HF_MODEL")
        self.CACHE_PATH = os.getenv("TT_CACHE_PATH")
        if HF_MODEL:
            self.CKPT_DIR = HF_MODEL
            self.TOKENIZER_PATH = HF_MODEL
            self.from_hf_url = True

            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join("model_cache", HF_MODEL, self.device_name)
            else:  # For HF models, always append the device name (e.g. N150/N300/T3K/TG) to the cache path
                self.CACHE_PATH = os.path.join(self.CACHE_PATH, self.device_name)
            self.model_name = HF_MODEL.strip("/").split("/")[
                -1
            ]  # HF model names use / even on windows. May be overridden by config.
        else:
            assert False, "Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct"

        logger.info(f"Checkpoint directory: {self.CKPT_DIR}")
        logger.info(f"Tokenizer file: {self.TOKENIZER_PATH + '/tokenizer.model'}")
        logger.info(f"Cache directory: {self.CACHE_PATH}")
        logger.info(f"Model name: {self.model_name}")

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.CKPT_DIR)
        self.model_cache_path = Path(self.CACHE_PATH)

        # Load weights and tokenizer
        self.consolidated_weights_path = self.CKPT_DIR + "/consolidated.00.pth"
        self.tokenizer_path = self.TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct
        # If the weights file contain the keyword `instruct` also set self.instruct to true
        if any(keyword in self.CKPT_DIR.lower() for keyword in ("instruct", "it")):
            self.instruct = True

        # Check for supported batches since previous logic that contained the check was removed because it was unused
        supported_batches = {1, 2, 4, 8, 16, 32}
        if self.max_batch_size not in supported_batches:
            raise ValueError(f"Batch size {self.max_batch_size} not supported")

        # Load model params
        if self.base_model_name in ["Phi-3-mini-128k-instruct"]:
            self.trust_remote_code_hf = True

        # Set checkpoint type - always HuggingFace since we only support HF_MODEL now
        self.checkpoint_type = CheckpointType.HuggingFace
        self._set_hf_params(self.CKPT_DIR)

        # Set the max number of tokens for each prefill chunk based on the model and device
        max_prefill_chunk_size_div1024 = os.getenv("MAX_PREFILL_CHUNK_SIZE")
        if max_prefill_chunk_size_div1024 is None:
            # TODO Improve this to be more general to more devices and models
            MAX_PREFILL_CHUNK_SIZES_DIV1024 = {
                "Llama-3.2-1B": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "Llama-3.2-3B": {"N150": 8, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "Llama-3.1-8B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
                "Llama-3.2-11B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
                "Llama-3.1-70B": {"N150": None, "N300": None, "T3K": 32, "TG": 128, "P150x4": 128},
                "Llama-3.2-90B": {"N150": None, "N300": None, "T3K": 32, "TG": 128, "P150x4": 128},
                "DeepSeek-R1-Distill-Llama-70B": {"N150": None, "N300": None, "T3K": 32, "TG": 128, "P150x4": 128},
                "Qwen2.5-7B": {"N150": 4, "N300": 32, "T3K": 128, "TG": 128, "P150x4": 128},
                "Qwen2.5-72B": {"N150": None, "N300": None, "T3K": 16, "TG": 128, "P150x4": 128},
                "Qwen2.5-VL-3B": {"N150": 128, "N300": 128, "T3K": None, "TG": None, "P150x4": None},
                "Qwen2.5-VL-7B": {"N150": 64, "N300": 128, "T3K": None, "TG": None, "P150x4": None},
                "Qwen2.5-VL-32B": {"N150": None, "N300": None, "T3K": 64, "TG": None, "P150x4": None},
                "Qwen2.5-VL-72B": {"N150": None, "N300": None, "T3K": 32, "TG": None, "P150x4": None},
                "DeepSeek-R1-Distill-Qwen-14B": {"N150": 4, "N300": 64, "T3K": 128, "TG": None, "P150x4": None},
                "Phi-3.5-mini-instruct": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "Phi-3-mini-128k-instruct": {"N150": 32, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
                "QwQ-32B": {"N150": None, "N300": None, "T3K": 64, "TG": 128, "P150x4": 128},
                "Qwen3-32B": {"N150": None, "N300": None, "T3K": 64, "TG": 128, "P150x4": 128},
                "Mistral-Small-3.1-24B": {
                    "N150": 32,
                    "N300": 64,
                    "T3K": 128,
                    "TG": 128,
                    "P150x4": 128,
                },  # Conservative: Allow on all devices
            }
            try:
                max_prefill_chunk_size_div1024 = MAX_PREFILL_CHUNK_SIZES_DIV1024[self.base_model_name][self.device_name]
            except KeyError:
                logger.warning(
                    f"Unknown model {self.model_name} on device {self.device_name}, setting MAX_PREFILL_CHUNK_SIZE to 4 for compatibility"
                )
                logger.warning(
                    f"Try setting MAX_PREFILL_CHUNK_SIZE to larger powers of 2 up to e.g. 128 for faster performance (if you run out of L1 memory it was too high)"
                )
                max_prefill_chunk_size_div1024 = 4
            assert (
                max_prefill_chunk_size_div1024 is not None
            ), f"Unsupported model {self.model_name} on device {self.device_name}"
        else:
            max_prefill_chunk_size_div1024 = int(max_prefill_chunk_size_div1024)
        self.max_prefill_chunk_size = max_prefill_chunk_size_div1024 * 1024

        if (self.base_model_name in ["Llama-3.1-8B", "Llama-3.2-11B", "Mistral-7B"] and self.device_name == "N150") or (
            self.base_model_name in ["Qwen2.5-7B", "Qwen2.5-VL-7B"] and self.device_name == "N300"
        ):
            logger.info(f"Reducing prefill_len_cutoff to 512 for {self.model_name} on {self.device_name}")
            self.prefill_len_cutoff = 512
        elif self.base_model_name in ["Mixtral-8x7B"] and self.device_name == "T3K":
            self.prefill_len_cutoff = 512

        if callable(optimizations):
            self.optimizations = optimizations(self)
        else:
            self.optimizations = optimizations

        # Configure data precision and math fidelity for tensors and kernels
        if self.optimizations is None:
            self.optimizations = DecodersPrecision.accuracy(num_decoders=self.n_layers, model_name=self.model_name)

        self.dummy_weights = dummy_weights
        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))

        # Enable workarounds by default until di/dt issues are fixed
        self.di_dt_workaround = os.getenv("DISABLE_DI_DT_WORKAROUND") != "1"
        if not self.di_dt_workaround:
            logger.info("Disabling di/dt workaround, re-enable if you see hangs")

        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        # Update memory configs (weights->DRAM, activations->L1)
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        self.tokenizer = None if dummy_weights else self.create_tokenizer()
        self.processor = None if dummy_weights else self.create_processor()

        # Flag to indicate whether we use fused version of QK ops (rotary embedding + page cached update)
        # We currently disable this fusion of ops for vision-capable or multimodal models
        self.use_qk_fused = not self.is_multimodal

        if device is not None:  # Avoid issue with test_torch.py not having a device
            self.n_local_heads = self.n_heads // self.cluster_shape[1]

            grid = device.compute_with_storage_grid_size()
            self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)

            # DRAM weight grid specs for dram sharding matmuls
            self.dram_weight_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(self.dram_grid_size.x - 1, self.dram_grid_size.y - 1),
                    )
                }
            )

            # Compute kernels. FP32 acc does not appear to be needed for accuracy in model tests or demo runs.
            self.compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.compute_kernel_config_hifi4_fp32 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
                dst_full_sync_en=False,
            )
            self.compute_kernel_config_hifi2_na = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            # Configure data precision and math fidelity for tensors and kernels
            self.model_config["COMPUTE_KERNEL_CONFIG_HIFI2"] = self.compute_kernel_config_hifi2
            # Mixtral
            self.model_config["MIXTRAL_PREFILL_MLP_COMPUTE_CONFIG"] = self.compute_kernel_config_lofi
            self.model_config["MIXTRAL_GATE_MM_OUTPUT_KERNEL_CONFIG"] = self.compute_kernel_config_lofi
            # end mixtral

            self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

            # Create memory config for sharded tensors
            residual_grid = self.dram_shard_core_grid_for_k(self.dim // self.num_devices)
            self.model_config["DECODE_RESIDUAL_MEMCFG"] = (
                ttnn.L1_MEMORY_CONFIG  # FIXME: when residual add support typecasting for sharded tensors
                if self.is_galaxy
                else ttnn.create_sharded_memory_config(
                    (
                        self.tile_padded_batch_rows,
                        self.dim // residual_grid.num_cores // self.num_devices,
                    ),
                    residual_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen, chunk_start_idx=None: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                # We want 256 if seqlen >= 2048 else 64. BUT:
                # SPDA limitation: chunk_start_idx must be a multiple of q_chunk_size
                # Here (x & -x) is the highest power of 2 that divides x.
                # When chunk_start_idx=0, we use default values since 0 is a multiple of any number.
                q_chunk_size=256
                if seqlen >= 2048 and (chunk_start_idx is None or chunk_start_idx == 0)
                else 64
                if seqlen < 2048 and (chunk_start_idx is None or chunk_start_idx == 0)
                else min(256, chunk_start_idx & -chunk_start_idx)
                if seqlen >= 2048
                else min(64, chunk_start_idx & -chunk_start_idx),
                # Original:
                # k_chunk_size=256 if seqlen >= 2048 else 64,
                # Workaround for https://github.com/tenstorrent/tt-metal/issues/35225 :
                k_chunk_size=256
                if seqlen >= 2048 and (chunk_start_idx is None or chunk_start_idx == 0)
                else 64
                if seqlen < 2048 and (chunk_start_idx is None or chunk_start_idx == 0)
                else min(256, chunk_start_idx & -chunk_start_idx)
                if seqlen >= 2048
                else min(64, chunk_start_idx & -chunk_start_idx),
            )

            # nlp_concat_heads_decode will shard the data across this number of cores
            assert (
                self.n_heads % self.cluster_shape[1] == 0
            ), f"n_heads must be divisible by num_devices: {self.n_heads} % {self.cluster_shape[1]}"

            # Note: for some models (e.g. Mistral-Small) n_heads * head_dim != dim
            self.model_config["ATTN_OUTPUT_PROGCFG"] = (
                None
                if self.is_galaxy
                else self.dram_matmul_config(
                    m=self.tile_padded_batch_rows,
                    k=(self.n_heads * self.head_dim) // self.num_devices,
                    n=self.dim,
                    num_cores=self.n_heads // self.num_devices,
                )
            )

            # All Gather Matmul for Dense Out (DO)
            # TODO: Is there a better way to decide if fused all gather matmul should be used? And is there a better way to use the flag, instead of passing it into model_config?
            # NOTE: Fused all gather matmul only suppports a core grid of size num_devices x 1
            # TODO: #26657 (self.num_devices == 8 and os.getenv("ACTUAL_DEVICE", "") != "TG") should be refactored, and investigate if ACTUAL_DEVICE environment variable is still used
            self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] = (
                self.num_devices == 8
                and os.getenv("ACTUAL_DEVICE", "") != "TG"
                and (self.dim // self.tile_size // self.num_devices) % self.num_devices == 0
                and self.num_devices > 1
            )

            if self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]:
                do_core_grid_size = (8, 1)
                do_per_core_N = (
                    self.dim // self.num_devices // self.tile_size // (do_core_grid_size[0] * do_core_grid_size[1])
                )
                self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=do_core_grid_size,
                    in0_block_w=self.dim
                    // self.tile_size
                    // (do_core_grid_size[0] * do_core_grid_size[1]),  # [32 x 8k] x [8k x 1k] = [32 x 1k]
                    out_subblock_h=1,
                    out_subblock_w=get_out_subblock_w(
                        do_per_core_N, out_subblock_h=1
                    ),  # Max out_subblock_w = 4, needs to be divisible by per_core_N
                    per_core_M=self.tile_padded_batch_rows // self.tile_size,
                    per_core_N=do_per_core_N,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            else:
                self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"] = None

            # For maximum performance, set the prefill grid row to 8, even if it can fit in a smaller grid
            # prefill_rows = lambda seq_len: min(seq_len, 1024) // self.tile_size
            prefill_rows = 8  # TODO if BH = 10, if wh = 8
            mlp1_3_grid = lambda seq_len: (
                (8, min(min(seq_len, 1024) // 32, 4))
                if self.is_galaxy
                else self.find_prefill_grid(prefill_rows, self.dim // self.tile_size)
            )
            mlp2_grid = lambda seq_len: (
                (8, min(min(seq_len, 1024) // 32, 4))
                if self.is_galaxy
                else self.find_prefill_grid(prefill_rows, self.hidden_dim // self.tile_size)
            )

            mlp_w_dram_sharded = not self.is_galaxy
            n_w1_w3 = self.hidden_dim // self.cluster_shape[1]
            # Using dram_shard_grid_width to ensure per_core_N matches DRAM shard width for P100, otherwise matmuls silently give bad PCC
            dram_shard_grid_width = 8 if is_wormhole_b0() else self.dram_grid_size.x  # 7 for P100, 8 for P150
            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
                k=self.dim // self.cluster_shape[0],
                n=n_w1_w3,
                grid_size=mlp1_3_grid(seq_len),
                per_core_N=math.ceil(n_w1_w3 / (self.tile_size * dram_shard_grid_width))
                if mlp_w_dram_sharded
                else None,
            )
            n_w2 = self.dim
            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
                k=self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1),
                n=n_w2,
                grid_size=mlp2_grid(seq_len),
                per_core_N=math.ceil(n_w2 / (self.tile_size * dram_shard_grid_width)) if mlp_w_dram_sharded else None,
            )
            self.model_config["PREFILL_MIXTRAL_MLP_W1_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
                k=self.dim // self.cluster_shape[0],
                n=n_w1_w3,
                grid_size=mlp1_3_grid(min(seq_len, self.prefill_len_cutoff)),
                per_core_M=math.ceil(min(seq_len, self.prefill_len_cutoff) / self.tile_size / self.cluster_shape[1]),
                per_core_N=math.ceil(n_w1_w3 / self.tile_size / self.cluster_shape[0]),
                fused_activation=ttnn.UnaryOpType.SILU,
            )
            self.model_config["PREFILL_MIXTRAL_MLP_W3_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
                k=self.dim // self.cluster_shape[0],
                n=n_w1_w3,
                grid_size=mlp1_3_grid(min(seq_len, self.prefill_len_cutoff)),
                per_core_M=math.ceil(min(seq_len, self.prefill_len_cutoff) / self.tile_size / self.cluster_shape[1]),
                per_core_N=math.ceil(n_w1_w3 / self.tile_size / self.cluster_shape[0]),
            )
            # Attention output is not necessarily the same dimension as the self.dim, e.g. in Mistral
            k_dim = (
                (self.n_heads * self.head_dim) // self.cluster_shape[0]
                if self.is_galaxy
                else (self.n_heads * self.head_dim) // self.num_devices
            )
            # TODO: #26657 (if self.num_devices == 8 and os.getenv("ACTUAL_DEVICE", "") != "TG") should be refactored, and investigate if ACTUAL_DEVICE environment variable is still used
            n_dim = (
                self.dim // self.cluster_shape[1]
                if self.is_galaxy
                else (
                    1024
                    if self.num_devices == 8
                    and os.getenv("ACTUAL_DEVICE", "") != "TG"
                    and 1024 % (self.dim / self.num_devices) == 0
                    else self.dim
                )
            )
            num_rows = lambda seq_len: min(seq_len, 1024)
            dram_sharded_wo = not (self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] or self.is_galaxy)
            self.model_config["WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=num_rows(seq_len),
                k=k_dim,
                n=n_dim,
                grid_size=self.find_prefill_grid(prefill_rows, k_dim // self.tile_size),
                in0_block_w=1 if self.is_galaxy else None,
                fuse_batch=seq_len <= 1024,
                per_core_N=math.ceil(n_dim / (self.tile_size * dram_shard_grid_width)) if dram_sharded_wo else None,
            )

            # Calculate largest number of lm_head_num_rows such that self.dim % (lm_head_num_rows * lm_head_cores_per_row) == 0
            if self.num_devices == 32:
                lm_head_num_rows = 4
                while self.dim % (32 * 32 * lm_head_num_rows) != 0:
                    lm_head_num_rows -= 1
            else:
                lm_head_num_rows = 8
            lm_head_cores_per_row = 8
            while self.dim % (32 * lm_head_num_rows * lm_head_cores_per_row) != 0:
                lm_head_num_rows -= 1
                if lm_head_num_rows == 0:
                    lm_head_cores_per_row -= 1
                    if lm_head_cores_per_row == 0:
                        raise ValueError(
                            f"Could not find a lm_head_num_rows such that self.dim(={self.dim}) % (lm_head_num_rows * 8) == 0"
                        )
                    lm_head_num_rows = 8
            self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=lm_head_cores_per_row)
            # 128256 comes from original llama 3 vocab size. 128256 / 4 was experimentally the maximum columns that worked per device.
            # The LM head for that was on 48 cores, so we know 128256 / 4 / 48 = 668 columns per core is close to the L1 limit.
            # FIXME: Update blackhole figure to be per-core as well.
            self.max_columns_per_device_lm_head = (
                128256 // 8 if is_blackhole() else 668 * self.lm_head_core_grid.num_cores
            )

            self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    nearest_32((self.dim // (4 if self.is_galaxy else 1)) // self.lm_head_core_grid.num_cores),
                ),  # Shard shape: [32, 128] -> 1 shard per core
                self.lm_head_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
            self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / (self.n_kv_heads // self.cluster_shape[1])
            self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 10) if is_blackhole() else (8, 8),
                in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=7
                if self.device_name == "P100"
                else (
                    max(  # NOTE: P100 runs OOM in L1 with 8 per_core_M
                        1,
                        8 if seq_len >= self.MAX_QKV_MM_SEQ_LEN else math.ceil(seq_len / self.tile_size / 8),  # 8 rows
                    )
                ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=math.ceil(
                    self.qkv_size / self.cluster_shape[1] / 32 / dram_shard_grid_width
                ),  # N / TILE_WIDTH / grid width
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= self.MAX_QKV_MM_SEQ_LEN,
            )

            assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"
            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: self.get_xqkv_prefill_mem_cfg(seq_len)

            self.model_config["CREATE_QKV_DECODE_SHARD"] = (
                ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if is_blackhole()
                else ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )

            self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )

            self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            self.model_config[
                "SCORES_BATCHED_MM_OUTPUT_MEMCFG"
            ] = lambda batch_size_per_device_group: ttnn.create_sharded_memory_config(
                shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),  # self.n_heads padded to tile size
                core_grid=ttnn.CoreRangeSet({num_to_corerange(batch_size_per_device_group)}),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # MLP configs
            mlp_core_grid = (
                self.dram_shard_core_grid_for_k(self.dim)
                if self.is_galaxy
                else self.dram_shard_core_grid_for_k_and_n(self.dim, self.hidden_dim // self.num_devices)
            )

            self.model_config["SHARDED_MLP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.dim // mlp_core_grid.num_cores,
                ),  # Shard shape: [32, 128] -> 1 shard per core
                mlp_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=self.hidden_dim // self.cluster_shape[1],
                num_cores=mlp_core_grid.num_cores,
            )

            mlp2_core_grid = (
                ttnn.CoreGrid(y=1, x=8)
                if self.is_galaxy
                else self.dram_shard_core_grid_for_k_and_n(self.hidden_dim // self.num_devices, self.dim)
            )

            self.model_config["SHARDED_MLP2_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    32 if self.is_galaxy else self.tile_padded_batch_rows,
                    self.hidden_dim // self.cluster_shape[1] // mlp2_core_grid.num_cores,
                ),
                mlp2_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.hidden_dim // self.cluster_shape[1],
                n=self.dim,
                num_cores=mlp2_core_grid.num_cores,
            )
            attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
            self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, nearest_32(self.dim // (8 * lm_head_num_rows) // 4)),
                    core_grid=ttnn.CoreGrid(y=lm_head_num_rows, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.is_galaxy
                else ttnn.create_sharded_memory_config(
                    (
                        self.tile_padded_batch_rows,
                        self.dim // attn_input_grid.num_cores,
                    ),  # Shard shape: [32, 128] -> 1 shard per core
                    attn_input_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            # glx doesn't support DRAM sharded matmuls yet
            self.model_config["XQKV_DECODE_PROGCFG"] = (
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(8, 5 if self.is_70b or self.is_90b else lm_head_num_rows),
                    in0_block_w=2 if self.is_70b or self.is_90b else 1,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                if self.is_galaxy
                else self.dram_matmul_config(
                    m=self.tile_padded_batch_rows,
                    k=self.dim,
                    n=self.qkv_size // self.num_devices,
                    num_cores=attn_input_grid.num_cores,
                )
            )

            full_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    )
                }
            )
            self.model_config["FULL_GRID_MEMCFG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    full_grid,
                    [
                        32,
                        nearest_32(56),
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

            self.model_config["MLP_ACT_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, self.dim // 4 // 16),  # dim / num devices / 16 cores
                    core_grid=ttnn.CoreGrid(x=8, y=2),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.dim >= 4096
                else self.model_config["FULL_GRID_MEMCFG"]
            )

            if self.is_galaxy:
                self.model_config["FF1_3_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                    (
                        1,
                        1,
                        32,
                        self.dim // 4,
                    ),
                    (
                        1,
                        1,
                        self.dim // 4,
                        self.hidden_dim // 8,
                    ),
                    grid=ttnn.CoreGrid(x=8, y=2),
                    overwrite_subblock_h=1,
                    overwrite_subblock_w=1,
                )

                self.model_config["FF2_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                    (
                        1,
                        1,
                        32,
                        self.hidden_dim // 8,
                    ),
                    (
                        1,
                        1,
                        self.hidden_dim // 8,
                        self.dim // 4,
                    ),
                    grid=ttnn.CoreGrid(x=8, y=2),
                    overwrite_subblock_h=1,
                    overwrite_subblock_w=1,
                )

            self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, self.hidden_dim // 28 // 8),  # shard_grid_cores = 28, num_devices=8
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 3))}),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )  # if self.dim==8192 else ttnn.DRAM_MEMORY_CONFIG

            self.model_config["FF1_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32 * 4, self.hidden_dim // 8 // 8),
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, self.dim // 8 // 4),  # shard_grid_cores = 8, num_devices=4
                    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.dim == 8192
                else ttnn.create_sharded_memory_config(
                    shape=(32 * 8, self.dim // 4 // 8),
                    core_grid=ttnn.CoreGrid(y=1, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            self.model_config["SELF_OUT_REDUCE_SCATTER_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, 2048 // 8 // 8),  # mesh_rows = 8, num_cores=8
                    core_grid=ttnn.CoreGrid(y=1, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.dim == 8192
                else ttnn.create_sharded_memory_config(
                    shape=(32 * 8, nearest_32(self.dim // 4 // 32)),  # mesh_rows = 8
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            self.model_config["FF2_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32 * 8, self.dim // 4 // 8),
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Vision model configs
            self.model_config["IMAGE_MLP_FC_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_dim,
                n=self.vision_hidden_dim // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_MLP_PROJ_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_hidden_dim // self.num_devices,
                n=self.vision_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_ATTN_QKV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_dim,
                n=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3)
                // self.num_devices,  # Head dim was padded to nearest 32
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_ATTN_OUT_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3) // self.num_devices,
                n=self.vision_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["VISION_XATTN_Q_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.dim,
                n=(self.head_dim * self.n_heads) // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= 1024,
            )
            self.model_config["VISION_XATTN_KV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.dim,
                n=(self.head_dim * self.n_kv_heads) // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["VISION_XATTN_SCORE_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                m=seq_len,
                k=self.head_dim,
                n=cache_seq_len,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )
            self.model_config["VISION_XATTN_OUTPUT_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                m=seq_len,
                k=cache_seq_len,
                n=self.head_dim,
                grid_size=(8, 8),
                # in0_block_w=1, # TODO: Remove this when we get non-causal FlashDecode
                fuse_batch=False,
            )
            self.model_config["VISION_XATTN_DENSE_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.dim // self.num_devices,
                n=self.dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )

            self.model_config["VISION_PROJ_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=seq_len,
                k=self.vision_dim * 6,
                n=self.dim // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )

            self.model_config["CROSS_TRANSFORMER_TEXT_OUTPUT_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.dim,
                n=self.vocab_size // 8,  # Magic number. LM Head always contains 8 splits
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )

            def _get_xattn_kv_prefill_mem_cfg(seq_len):
                M = (self.n_kv_heads // self.num_devices) * seq_len
                cores_x, cores_y = self.find_grid(M // self.tile_size)
                return ttnn.create_sharded_memory_config(
                    (
                        nearest_32(M // (cores_x * cores_y)),
                        self.head_dim,
                    ),
                    ttnn.CoreGrid(y=cores_y, x=cores_x),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

            self.model_config["XATTN_KV_PREFILL_MEM_CFG"] = _get_xattn_kv_prefill_mem_cfg

            if self.is_multimodal:
                self.VISION_MAX_MM_SEQ = nearest_32(self.vision_chunk_ntok)

            # RMS NORM
            self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = self.create_sharded_norm_config(attn_input_grid)
            self.model_config["SHARDED_NORM_MLP_PRGM_CFG"] = self.create_sharded_norm_config(mlp_core_grid)
            self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"] = self.create_sharded_norm_config(self.lm_head_core_grid)

            # All gather matmuls currently only supported on T3K
            # We need it sharded on num_cores = num_devices
            self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    num_to_core_range_set(self.num_devices),
                    [
                        self.tile_padded_batch_rows,
                        self.dim // self.num_devices,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

            self.set_tg_attention_config()

            self.is_multichip = self.num_devices > 1
            self.num_reduce_scatter_links = 1
            self.num_all_gather_links = (
                2 if self.is_galaxy else 1
            )  # TODO: try out 3 for short axis and 4 for long axis (TG only) <- should work but untested in model
            self.ccl_dtype = ttnn.bfloat8_b

            logger.info(f"Attention grid: {attn_input_grid}")
            logger.info(f"MLP grid: {mlp_core_grid}")
            logger.info(f"MLP prefill grids @ 32: w1/w3: {mlp1_3_grid(32)}, w2: {mlp2_grid(32)}")
            logger.info(
                f"MLP prefill grids @ max_seq_len({self.max_seq_len}): w1/w3: {mlp1_3_grid(self.max_seq_len)}, w2: {mlp2_grid(self.max_seq_len)}"
            )
            logger.info(f"LM head grid: {self.lm_head_core_grid}")

        self.capped_warmup_seq_len = min(self.max_prefill_chunk_size, self.max_seq_len)
        self.trace_prefill_supported_seq_lens = self.get_trace_prefill_supported_seq_lens()

    def get_warmup_prefill_supported_seq_lens(self):
        assert (
            self.capped_warmup_seq_len > 0 and (self.capped_warmup_seq_len & (self.capped_warmup_seq_len - 1)) == 0
        ), f"capped_warmup_seq_len must be a power of 2, but got {self.capped_warmup_seq_len}"

        DEFAULT_VALUE = self.capped_warmup_seq_len
        # This dictionary is used to override the default ceil warmup prefill value
        model_specific_ceil_warmup_lengths = {
            # Qwen3-32B hangs at 8192, so we cap at 4096
            "Qwen3-32B": 4096,
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

        # TODO: https://github.com/tenstorrent/tt-metal/issues/33991 - for P100 only, P150 has assert for ISL > 1K
        if self.base_model_name == "Llama-3.1-8B" and self.device_name == "P100":
            for seq_len in to_warmup_seq_lens:
                if seq_len > 1024:
                    to_warmup_seq_lens = to_warmup_seq_lens[: to_warmup_seq_lens.index(seq_len)]
                    break
        return to_warmup_seq_lens

    def get_trace_prefill_supported_seq_lens(self):
        default_supported_seq_lens = {
            "N150": [128],
            "N300": [128, 1024],
            "T3K": [128, 1024],
            "TG": [128, 1024],
            "P150": [128, 1024],
            "P300": [128, 1024],
            "P150x4": [128, 1024],
            "P150x8": [128, 1024],
        }

        # TODO: If no specific sequence lengths are listed for a model and device, the default one will be used (from the default_supported_seq_lens dictionary)
        model_specific_supported_seq_lens = {
            "Llama-3.1-8B": {
                "P100": [128, 1024],
                "N150": [128, 1024],
                "N300": [128, 1024, 2048, 4096, 8192],
                "T3K": [128, 1024, 2048, 4096, 8192],
                "TG": [128, 1024, 2048, 4096, 8192],
            },
            "Llama-3.1-70B": {
                "T3K": [128, 1024, 2048, 4096, 8192],
                "TG": [128, 1024, 2048, 4096, 8192],
            },
            "Llama-3.3-70B": {
                "T3K": [128, 1024, 2048, 4096, 8192],
                "TG": [128, 1024, 2048, 4096, 8192],
            },
        }

        model_name = self.base_model_name
        device_name = self.device_name

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

    @staticmethod
    def __get_llama_local_params_name(model_name):
        if "3.2-1B" in model_name:
            local_params = "LLAMA3_2_1B_PARAMS"
        elif "3.2-3B" in model_name:
            local_params = "LLAMA3_2_3B_PARAMS"
        elif "3.1-8B" in model_name:
            local_params = "LLAMA3_1_8B_PARAMS"
        elif "3.2-11B" in model_name:
            local_params = "LLAMA3_2_11B_PARAMS"
        elif "3.1-70B" in model_name:
            local_params = "LLAMA3_1_70B_PARAMS"
        elif "3.2-90B" in model_name:
            local_params = "LLAMA3_2_90B_PARAMS"
        else:
            local_params = None
        return local_params

    def get_xqkv_prefill_mem_cfg(self, seq_len):
        return ttnn.create_sharded_memory_config(
            (((self.n_kv_heads // self.cluster_shape[1]) * seq_len // (8 * 8)), self.head_dim),
            ttnn.CoreGrid(y=8, x=8),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def is_distributed_norm(self, mode):
        if not self.is_multichip:
            return False
        if all([dim > 1 for dim in list(self.mesh_device.shape)]):  # 2D grid
            return True
        elif self.dim > 4096 and mode == "prefill":  # Somewhere between 4k and 8k WH runs out of L1 if not distributed
            return True
        return False

    def ccl_topology(self):
        # Use ring on a T3K or 6U galaxy submesh
        if self.num_devices == 8 and ttnn.cluster.get_cluster_type() in [
            ttnn.cluster.ClusterType.T3K,
            ttnn.cluster.ClusterType.GALAXY,
        ]:
            return ttnn.Topology.Ring
        elif self.num_devices > 1:  # All other multi chip devices
            return ttnn.Topology.Linear
        return None

    def prepare_residual_tensor_decode(self, x, input_mem_cfg, force_replicated=False, on_host=False):
        """
        Prepare inputs for decode mode.
        x: (batch, seq, dim)
        """
        dims = (None, None) if force_replicated else (None, -1)
        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=self.cluster_shape)

        if len(x.shape) == 3:
            batch = x.shape[0]
            seq_len = x.shape[1]
            assert x.shape[2] == self.dim
        elif len(x.shape) == 4:
            seq_len = x.shape[0]
            assert x.shape[1] == 1
            batch = x.shape[2]
            assert x.shape[3] == self.dim

        assert seq_len == 1, "Only supporting decode mode"

        # Support input on device
        if torch.is_tensor(x):  # Input on host -> Use torch
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, dim]
            # Pad small batches to 32
            if batch < 32:
                zeros = torch.zeros(1, seq_len, 32, self.dim)
                zeros[:, :, :batch, :] = x
                x = zeros
        elif len(x.shape) == 3:  # Input on device -> Use ttnn
            x = ttnn.reshape(x, (batch, seq_len, 1, self.dim))  # [batch, seqlen, dim] -> [batch, seqlen, 1, dim]
            x = ttnn.permute(x, (1, 2, 0, 3))  # [seq_len, 1, batch, dim]
        elif len(x.shape) == 4:
            pass  # already in [seq_len, 1, batch, dim]

        if torch.is_tensor(x):
            x = ttnn.from_torch(
                x,
                device=self.mesh_device if not on_host else None,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=input_mem_cfg if not on_host else None,
            )
        else:  # Convert the row major layout from embedding back to tile layout
            x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        return x

    def prepare_residual_tensor_prefill(self, x_bsh, force_replicated=False):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim
        """

        x_1BSH = x_bsh.unsqueeze(0)
        dims = (None, None) if force_replicated else (None, -1)

        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=self.cluster_shape)

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

    def _get_text_prefix(self):
        if self.is_llama_vision():
            return "text_model."
        else:
            return ""

    def _get_vision_prefix(self):
        return "visual."

    def _get_hidden_activation_type(self, config):
        activation_map = {
            "gelu": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0),
            "gelu_pytorch_tanh": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
            "swish": ttnn.UnaryOpType.SILU,
        }

        hidden_activation = config.get("hidden_act") or config.get("hidden_activation")
        if not hidden_activation:
            # Default to SILU if no activation is specified
            return ttnn.UnaryOpType.SILU

        hidden_activation = hidden_activation.lower()
        if hidden_activation not in activation_map:
            raise NotImplementedError(f"Unsupported activation '{hidden_activation}'")

        return activation_map.get(hidden_activation, ttnn.UnaryOpType.SILU)

    def _set_model_specific_params(self):
        return

    def _set_params_from_dict(self, config):
        eos_token_id = config.get("eos_token_id", None)
        self.image_token_index = config.get("image_token_index", None)

        # Try to get text_config, if it doesn't exist everything is text config
        text_config = config.get("text_config", config)
        self.eos_token_id = None if isinstance(eos_token_id, int) else eos_token_id
        layer_types = text_config["layer_types"] if "layer_types" in text_config else None

        # Common params with different names between Meta and HF
        self.dim = text_config.get("dim", text_config.get("hidden_size"))
        self.n_heads = text_config.get("n_heads", text_config.get("num_attention_heads"))
        self.n_kv_heads = text_config.get("n_kv_heads", text_config.get("num_key_value_heads"))
        self.n_layers = text_config.get("n_layers", text_config.get("num_hidden_layers"))
        # multimodal llama additionally adds cross attention layers
        # they are calculated in HF but not calculated in Meta
        self.n_layers -= len(text_config.get("cross_attention_layers", ()))

        self.sliding_window_pattern = (
            [lt == "sliding_attention" for lt in layer_types] if layer_types is not None else [False] * self.n_layers
        )

        self.full_model_n_layers = self.n_layers
        self.norm_eps = text_config.get("norm_eps", text_config.get("rms_norm_eps"))
        self.vocab_size = text_config["vocab_size"]
        self.padded_vocab_size = 128 * 1024 if self.is_galaxy else None
        self.head_dim = text_config.get("head_dim", self.dim // self.n_heads) or self.dim // self.n_heads
        self.num_experts_per_tok = text_config.get("num_experts_per_tok", 0)
        self.max_context_len = text_config.get("max_position_embeddings")

        # Handle different MLP dimension specifications
        if "intermediate_size" in text_config:
            self.hidden_dim = text_config["intermediate_size"]
            self.ffn_dim_multiplier = None
            self.multiple_of = None

            # temporary solution for using HF_MODEL for LLaMA until llama_model references are removed
            local_params = self.__get_llama_local_params_name(self.model_name)
            if local_params in self.LOCAL_LLAMA_PARAMS:
                params_file = os.path.join(self.LOCAL_LLAMA_PARAMS[local_params], "params.json")
                if os.path.exists(params_file):
                    with open(params_file, "r") as f:
                        params = json.load(f)
                    self.ffn_dim_multiplier = params["ffn_dim_multiplier"]
                    self.multiple_of = params["multiple_of"]
        else:
            self.ffn_dim_multiplier = text_config["ffn_dim_multiplier"]
            self.multiple_of = text_config["multiple_of"]
            self.hidden_dim = calculate_hidden_dim(self.dim, self.ffn_dim_multiplier, self.multiple_of)

        if "_name_or_path" in config and config["_name_or_path"]:
            normalized_path = os.path.normpath(config["_name_or_path"])
            # For HF paths, they might end with `<model_name>/snapshots/<snapshot_id>/`
            if "snapshots" in normalized_path:
                full_model_name = normalized_path.split(os.path.sep)[-3]
                self.model_name = full_model_name.split("--")[-1]
            else:
                self.model_name = os.path.basename(normalized_path)
            logger.info(f"Model name from config: {self.model_name}")

        if self.base_model_name in ["Qwen2.5-7B", "Qwen2.5-VL-7B"] and self.num_devices not in [0, 2, 4]:
            raise AssertionError(
                "Qwen2.5-7B and Qwen2.5-VL-7B is only supported on 2 or 4 devices, run on an N300 or use MESH_DEVICE=N150x4"
            )

        self.unpadded_hidden_dim = self.hidden_dim
        # Don't need to pad for CPU runs
        if self.num_devices:
            # Default padding cores for each model, 0 if not set here
            default_padded_cores = {
                "Qwen2.5-VL-72B": 32,
                "Qwen2.5-VL-32B": 16,
                "Qwen2.5-72B": 32,
                "Qwen2.5-7B": 16,
                "QwQ-32B": 16,
            }.get(self.base_model_name, 0)

            # Override MLP padding cores from env var
            mlp_padded_cores = int(os.environ.get("PAD_MLP_CORES", default_padded_cores))

            # Only pad if MLP_PADDED_CORES is non-zero
            if mlp_padded_cores > 0:
                padded_hidden_dim = nearest_multiple(
                    self.hidden_dim, mlp_padded_cores * self.tile_size * self.num_devices
                )
                if padded_hidden_dim != self.hidden_dim:
                    logger.info(
                        f"PAD_MLP_CORES={mlp_padded_cores}, padding hidden dim from {self.hidden_dim} to {padded_hidden_dim}"
                    )
                    self.hidden_dim = padded_hidden_dim

        self.layer_types = text_config.get("layer_types", None)

        # RoPE params
        self.rope_theta = text_config.get("rope_theta")
        self.rope_theta_local = text_config.get("rope_local_base_freq", None)

        rope_scaling_params = text_config.get("rope_scaling", None)
        self.original_max_context_len = text_config.get("original_max_position_embeddings", None)
        self.rope_scaling = (
            rope_scaling_model_factory(rope_scaling_params, original_max_context_len=self.original_max_context_len)
            if rope_scaling_params
            else None
        )

        self.query_pre_attn_scalar = text_config.get("query_pre_attn_scalar", None)

        # Sliding window attention
        self.sliding_window = text_config.get("sliding_window", None)

        # Configurable MLP activation type
        self.mlp_activation_type = self._get_hidden_activation_type(text_config)

        self._set_vision_params(config)
        self.is_multimodal = "vision_config" in config or self.is_vision()

        self.state_dict_text_prefix = self._get_text_prefix()
        self.state_dict_vision_prefix = self._get_vision_prefix()

        self._set_model_specific_params()

    @property
    def use_scaled_rope(self):
        return self.rope_scaling is not None

    @property
    def base_model_name(self):
        return get_base_model_name(self.model_name)

    @property
    def vision_chunk_ntok(self):
        """
        Returns the number of tokens per chunk, accounting for the extra class token
        """
        return (self.vision_chunk_size // self.vision_patch_size) ** 2 + 1

    def _set_vision_params(self, config):
        vision_config = config.get("vision_config", config)

        # Get vision_dim from config (same key for all models)
        self.vision_dim = vision_config.get("hidden_size", 1280)

        # Get vision_head_dim - Mistral has it in config, others calculate it
        if "head_dim" in vision_config:
            self.vision_head_dim = vision_config["head_dim"]
        else:
            num_heads = vision_config.get("num_attention_heads") or vision_config.get("num_heads") or 16
            self.vision_head_dim = self.vision_dim // num_heads

        # Get image_size from config (same key for all models)
        self.image_size = vision_config.get("image_size", -1)

        # Optional values with reasonable fallbacks
        chunk_size_fallback = self.image_size if self.image_size != -1 else vision_config.get("image_size", -1)
        self.vision_chunk_size = vision_config.get("vision_chunk_size", chunk_size_fallback)
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", vision_config.get("max_num_tiles", 4))
        self.vision_num_cross_attention_layers = vision_config.get(
            "vision_num_cross_attention_layers", vision_config.get("num_global_layers", 8)
        )

        # Common vision parameters for all models
        intermediate_size = vision_config.get("intermediate_size", self.vision_dim * 4)
        self.vision_image_size = vision_config.get("image_size", 1540)
        self.vision_rope_theta = vision_config.get("rope_theta", 10000.0)
        self.image_token_index = vision_config.get("image_token_index", 10)

        self.vision_mlp_ratio = intermediate_size // self.vision_dim
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads") or vision_config.get("num_heads") or 16
        # Note: vision_head_dim is already set above (from config for Mistral, calculated for others)

        # Default to 32 layers - this is the standard for Llama vision models (e.g., Llama-3.2-11B-Vision uses 32)
        # This default is only used when the config doesn't specify num_hidden_layers or depth
        # Models that specify these values in their config (e.g., Mistral-Small-3.1-24B-Instruct-2503 uses 24)
        # will use their specified values, not this default
        # The default of 32 comes from the main branch and matches Llama vision model architecture
        self.vision_n_layers = vision_config.get("num_hidden_layers") or vision_config.get("depth") or 32
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_in_channels = vision_config.get("num_channels", 3)

        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.mm_tokens_per_image = vision_config.get("mm_tokens_per_image", config.get("mm_tokens_per_image", 256))

        # Optional vision activation layer, defaults to GELU
        act_layer = vision_config.get("act_layer", "gelu").lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)

        # Optional tuning knobs
        self.vision_max_num_tiles = vision_config.get("max_num_tiles", 4)
        self.vision_n_global_layers = vision_config.get("n_global_layers", vision_config.get("num_global_layers", 8))

    def _set_hf_params(self, checkpoint_dir):
        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            # Merge non-nested keys into text_config
            text_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return text_config

        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            # Merge non-nested keys into vision_config
            vision_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return vision_config

        if self.from_hf_url:
            from transformers import AutoConfig

            if self.dummy_weights:
                logger.info(
                    f"Loading state param for dummy {self.model_name} from {self.LOCAL_HF_PARAMS[self.model_name]}"
                )
                self.hf_config = AutoConfig.from_pretrained(
                    self.LOCAL_HF_PARAMS[self.model_name], trust_remote_code=self.trust_remote_code_hf
                )
            else:
                self.hf_config = AutoConfig.from_pretrained(
                    self.CKPT_DIR,
                    trust_remote_code=self.trust_remote_code_hf,
                    local_files_only=os.getenv("CI") == "true",
                )

            config = self.hf_config.to_dict()

            if "text_config" in config or "vision_config" in config:
                merged_text_config = merge_text_config(config)
                self._set_params_from_dict(merged_text_config)

                if "vision_config" in config:
                    # Merge vision config (merge_vision_config is safe for all models - it only adds missing keys)
                    merged_vision_config = merge_vision_config(config)
                    self._set_vision_params({"vision_config": merged_vision_config})
            else:
                self._set_params_from_dict(config)
        else:
            config_file = os.path.join(checkpoint_dir, "config.json")
            assert os.path.exists(config_file), f"config.json file not found at {config_file}"
            with open(config_file, "r") as f:
                config = json.load(f)
            self._set_params_from_dict(config)

        # compatibility with _set_params
        if "llama" in self.model_name.lower():
            if "3.2-11B" in checkpoint_dir:
                logger.warning(f"-Vision is removed from model_name {self.model_name}")
                # TODO: do not remove "-Vision" part
                self.model_name = "Llama-3.2-11B" + ("-Instruct" if self.instruct else "")
            elif "3.1-70B" in checkpoint_dir:
                self.is_70b = True  # self.dim == 8192 and self.n_layers == 80
            elif "3.2-90B" in checkpoint_dir:
                logger.warning(f"-Vision is removed from model_name {self.model_name}")
                # TODO: do not remove "-Vision" part
                self.model_name = "Llama-3.2-90B" + ("-Instruct" if self.instruct else "")
                self.is_90b = True

    def __repr__(self):
        return f"""ModelArgs(
    dim={self.dim},
    n_layers={self.n_layers},
    n_heads={self.n_heads},
    n_kv_heads={self.n_kv_heads},
    vocab_size={self.vocab_size},
    multiple_of={self.multiple_of},
    ffn_dim_multiplier={self.ffn_dim_multiplier},
    norm_eps={self.norm_eps},
    rope_theta={self.rope_theta},
    rope_scaling_factor={self.rope_scaling.factor if self.rope_scaling is not None else None},
    max_batch_size={self.max_batch_size},
    max_seq_len={self.max_seq_len},
    vision_chunk_size={self.vision_chunk_size},
    vision_max_num_chunks={self.vision_max_num_chunks},
    vision_num_cross_attention_layers={self.vision_num_cross_attention_layers}
)"""

    def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
        """
        This function is used to determine if trace should be enabled for the prefill.
        Tracing is used only for certain sequence lengths, because for bigger sequence lengths, op2op gaps are already small, so we don't need tracing.
        # TODO: Support chunked prefill with tracing - https://github.com/tenstorrent/tt-metal/issues/32056
        # TODO: Support prefix caching with tracing
        """

        allowed_seq_lens = self.trace_prefill_supported_seq_lens

        return (
            prefill_seq_len in allowed_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
            and num_cached_tokens == 0
        )

    def is_llama_vision(self):
        return self.CKPT_DIR is not None and ("llama" in self.CKPT_DIR.lower()) and ("vision" in self.CKPT_DIR.lower())

    def is_vision(self):
        """Check if this is a vision-capable model (Llama vision or Mistral multimodal)"""
        return self.is_llama_vision() or (
            "mistral" in self.model_name.lower()
            and (
                (self.CKPT_DIR is not None and "vision" in self.CKPT_DIR.lower())
                or "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name
            )
        )

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        # Llama vision models use "text_model." prefix for text keys
        # Other vision models (Mistral, etc.) don't use text_model prefix for text
        if self.is_llama_vision():
            text_prefix = self.state_dict_text_prefix
        else:
            # Standard models and non-Llama vision: no prefix for text, prefix for vision
            text_prefix = "" if not is_vision else self.state_dict_text_prefix

        vision_prefix = self.state_dict_vision_prefix if is_vision else ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        text_module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",  # If no module is given, just get layer prefix
        }

        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "self_attn.",
            "TransformerBlock": "",
            "": "",
        }

        module_map = vision_module_map if is_vision else text_module_map
        prefix = vision_prefix if is_vision else text_prefix

        return prefix + layer_prefix + module_map[module_name]

    def weight_cache_path(self, dtype):
        # Keep the weight cache separate for generative and instruct weights
        if self.instruct:
            return (
                self.model_cache_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        else:
            return (
                self.model_cache_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            )

    def get_model_config(self):
        return self.model_config

    def get_hf_model_cls(self):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

        if not self.is_multimodal:
            return AutoModelForCausalLM

        for model_cls in (AutoModelForVision2Seq, AutoModelForImageTextToText):
            if type(self.hf_config) in model_cls._model_mapping:
                return model_cls

        raise ValueError(f"Unknown model for config {type(self.hf_config)}")

    # TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
    def load_state_dict(self):
        # by default, the model is not a mixture-of-expert. This will be set to True if we find any `.experts.` in the keys
        if self.dummy_weights:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                self.LOCAL_HF_PARAMS[self.model_name], trust_remote_code=self.trust_remote_code_hf
            )
            if hasattr(config, "text_config"):
                config.text_config.num_layers = self.n_layers
                config.text_config.num_hidden_layers = self.n_layers
            else:
                config.num_layers = self.n_layers
                config.num_hidden_layers = self.n_layers

            model_cls = self.get_hf_model_cls()

            try:
                # .from_pretrained + _init_weights works faster than .from_config
                model = model_cls.from_pretrained(
                    self.CKPT_DIR,
                    config=config,
                    torch_dtype="auto",
                    trust_remote_code=self.trust_remote_code_hf,
                    local_files_only=True,
                )
                model.apply(model._init_weights)
            except Exception as e:
                logger.info(f"Error loading dummy weights using .from_pretrained. Using .from_config. Error: {e}")
                model = model_cls.from_config(config, trust_remote_code=self.trust_remote_code_hf)

            # model.load_state_dict({k: torch.randn_like(v) for k, v in model.state_dict().items()})
            state_dict = model.state_dict()
        elif self.checkpoint_type == CheckpointType.Meta:
            state_dict = load_meta_state_dict(self.CKPT_DIR, self.n_layers)
            self.is_mixture_of_experts = any(["experts" in k for k in state_dict.keys()])
        else:
            assert self.checkpoint_type == CheckpointType.HuggingFace
            if self.from_hf_url:
                # Use get_hf_model_cls() from main branch, but handle special cases
                # Special case Qwen2.5-VL models until they are fully integrated into a HF release
                if "Qwen2.5-VL" in self.model_name:
                    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                        Qwen2_5_VLForConditionalGeneration as AutoModelForCausalLM,
                    )

                    model_cls = AutoModelForCausalLM
                    print("Loading Qwen2.5-VL model: ", AutoModelForCausalLM)
                elif "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
                    # Special case Mistral-Small-3.1-24B-Instruct-2503: HF's AutoModel doesn't work,
                    # similar to Qwen2.5-VL, until fully integrated into a HF release
                    from transformers import Mistral3ForConditionalGeneration

                    model_cls = Mistral3ForConditionalGeneration
                else:
                    model_cls = self.get_hf_model_cls()
                model = model_cls.from_pretrained(
                    self.CKPT_DIR,
                    torch_dtype="auto",
                    trust_remote_code=self.trust_remote_code_hf,
                    local_files_only=os.getenv("CI") == "true"
                    # Note that the default setting is torch.dtype.float32, but model weights are
                    # may come in any dtype. If the model's weights are in torch.dtype.bfloat16, this would result in 2x memory usage from an
                    # unnecessary cast.
                )
                if self.cache_hf_flag:
                    self.cached_hf_model = model
                state_dict = model.state_dict()
                self.is_mixture_of_experts = any([".experts." in k for k in state_dict.keys()])
            else:
                state_dict = load_hf_state_dict(self.CKPT_DIR)
                self.is_mixture_of_experts = any([".experts." in k for k in state_dict.keys()])

        if self.is_multimodal:
            state_dict = standardize_hf_keys_multimodal(state_dict)
            if self.is_llama_vision():
                state_dict = convert_hf_to_meta_mllama(state_dict, self.head_dim, self.hf_config)
            else:
                state_dict = convert_vision_hf_to_meta(state_dict, self.head_dim)
        else:
            self.fuse_qkv = any(["qkv" in layer_name for layer_name in state_dict.keys()])
            self.fuse_mlp = any(["gate_up" in layer_name for layer_name in state_dict.keys()])
            state_dict = standardize_hf_keys(state_dict)
            state_dict = convert_hf_to_meta(state_dict, self.head_dim, self.n_heads, self.n_kv_heads)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)
        if getattr(self, "is_mixture_of_experts", False):
            self.initialize_mixture_of_experts_configs()
            self.moe = True
            self.num_experts = max([int(item[-11]) + 1 for item in keys_dict if "block_sparse_moe.experts" in item])
        return state_dict

    def initialize_mixture_of_experts_configs(self):
        # Porting mixtral to llama
        self.model_config["FF1_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 4096 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=7,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.SILU,
            mcast_in0=True,
        )
        self.model_config["FF3_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 4096 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=7,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.model_config["FF2_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=7,  # K = 14336 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            # Issue #8959: Increasing subblock to 2 results in hangs -> Potentially related to di/dt hangs.
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=2,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.model_config["PREFILL_MLP_W1_PRG_CONFIG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # 32, #16,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=56,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=ttnn.UnaryOpType.SILU,
            fuse_batch=False,
        )
        self.model_config["PREFILL_MLP_W3_PRG_CONFIG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=56,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        # end Porting mixtral to llama

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors"""
        dram_cores = self.dram_grid_size.x  # WH has 12 dram cores, P150 has 8, P100 has 7
        assert self.dram_grid_size.y == 1, "Current dram sharding assumes y dim is 1"
        padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def matmul_config(
        self,
        m: int,
        k: int,
        n: int,
        grid_size: Tuple[int, int],
        in0_block_w: int = None,
        fuse_batch: bool = False,
        fused_activation=None,
        per_core_M=None,
        per_core_N=None,
    ):
        if per_core_M is None:
            per_core_M = math.ceil(m / (self.tile_size * grid_size[1]))
        if per_core_N is None:
            per_core_N = math.ceil(n / (self.tile_size * grid_size[0]))

        out_subblock_h = 1
        out_subblock_w = (
            get_out_subblock_w(per_core_N, out_subblock_h) if not self.is_galaxy else 1
        )  # TODO: Needed for TG hang workaround

        if in0_block_w is None:
            assert (
                k % (self.tile_size * grid_size[1]) == 0
            ), f"Input width must be divisible by tile size times grid size"
            in0_block_w = self.find_largest_divisor(k // (self.tile_size * grid_size[1]))

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=fuse_batch,
        )

    def dram_shard_core_grid_for_k(self, k: int) -> Tuple[int, int]:
        rows, cols = self.find_grid(k // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid(self, N):
        """
        Find the number of rows and columns for a grid of cores such that
        the total number of tiles N can be evenly divided among the cores.
        Each core will have the same integer number of tiles.
        The grid size is limited to a maximum of 2 rows and 8 columns.

        Parameters:
            N (int): Total number of tiles to be distributed.

        Returns:
            tuple: A tuple (rows, cols) representing the grid dimensions.

        Raises:
            AssertionError: If it's not possible to find such a grid configuration.
        """
        max_rows = 8
        max_cols = 8
        max_cores = max_rows * max_cols

        # Find all possible numbers of cores that divide N and are less than or equal to max_cores
        target = 32
        possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
        possible_cores.sort(key=lambda x: abs(x - target))  # Sort by closest to target

        for cores in possible_cores:
            # Try to find a grid configuration with the current number of cores
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        # If no configuration is found, assert an error
        raise AssertionError(
            f"Cannot find a grid configuration for {N} tiles that evenly divides into {max_cores} cores of max size {max_rows}x{max_cols}."
        )

    def find_prefill_grid(self, row_tiles, col_tiles):
        """Find a grid such that the number of row tiles evenly divides into the number
        of rows and the number of column tiles evenly divides into the number of columns
        """
        max_rows = 8
        max_cols = 8
        # TODO Improve configuration for BH (higher core grid than WH)

        # Find number of cols that evenly divides into the number of columns
        cols = None
        rows = None

        for i in range(max_cols, 0, -1):
            if col_tiles % i == 0:
                cols = i
                break

        for i in range(max_rows, 0, -1):
            if row_tiles % i == 0:
                rows = i
                break

        assert cols is not None, f"Cannot find a number of columns that evenly divides into {col_tiles}, not even 1(!)."
        assert rows is not None, f"Cannot find a number of rows that evenly divides into {row_tiles}, not even 1(!)."
        return rows, cols

    def dram_shard_core_grid_for_k_and_n(self, k: int, n: int) -> Tuple[int, int]:
        rows, cols = self.find_grid_k_n(k // self.tile_size, n // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid_k_n(self, K, N):
        """
        Find the number of rows and columns for a grid of cores such that
        the total number of tiles N can be evenly divided among the cores.
        Each core will have the same integer number of tiles.

        Parameters:
            N (int): Total number of tiles to be distributed.

        Returns:
            tuple: A tuple (rows, cols) representing the grid dimensions.

        Raises:
            AssertionError: If it's not possible to find such a grid configuration.
        """
        max_rows = 8
        max_cols = 8  # Maximum number of rows or columns
        max_cores = max_rows * max_cols  # Maximum number of cores

        # Find all possible numbers of cores that divide N and are less than or equal to max_cores
        possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
        possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

        for cores in possible_cores:
            # Try to find a grid configuration with the current number of cores
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        # If no configuration is found, assert an error
        raise AssertionError(
            f"Cannot find a grid configuration such that both {K} and {N} tiles evenly divide into cores of max size {max_rows}x{max_cols}."
        )

    def find_largest_divisor(self, n, max_divisor=8):
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1  # Fallback to 1 if no divisor found

    def dram_matmul_config(self, m: int, k: int, n: int, num_cores=None, fused_activation=None):
        # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
        if num_cores is None:
            # num_cores = self.dram_shard_core_grid_for_k(k).num_cores
            num_cores = self.dram_shard_core_grid_for_k_and_n(k, n).num_cores
            assert (
                k % (self.tile_size * num_cores) == 0
            ), f"k must be divisible by tile_size * num_cores: {k} % {self.tile_size * num_cores} != 0"
            # assert n % (self.tile_size * num_cores) == 0, f"n must be divisible by tile_size * num_cores: {n} % {self.tile_size * num_cores} != 0"
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.find_largest_divisor(k // (self.tile_size * num_cores)),
            per_core_M=math.ceil(m / self.tile_size),
            per_core_N=math.ceil(n / (self.tile_size * num_cores)),
            fused_activation=fused_activation,
        )

    def matmul_1d_config(
        self,
        m,
        k,
        n,
        grid=ttnn.CoreGrid(x=8, y=8),
        act=None,
        is_fp32_accumulate=False,
        overwrite_per_core_k=None,
        overwrite_subblock_w=None,
        overwrite_subblock_h=None,
    ):
        tile_width = 32
        tile_height = 32

        if (
            n // tile_width // grid.num_cores < 1
        ):  # use less number of cores in case we have more N num tiles than cores
            # assert (n // tile_width) % grid.x == 0
            grid_y = n // tile_width // grid.x
            grid = ttnn.CoreGrid(x=grid.x, y=grid_y)

        per_core_m = m // tile_height
        per_core_k = self.find_largest_divisor(k // (self.tile_size * grid.num_cores))
        per_core_n = math.ceil(n / tile_width / grid.num_cores)

        if is_fp32_accumulate:
            max_subblock_w_h = 4
        else:
            max_subblock_w_h = 8

        # find the largest value between 1 and 8 that is a factor of per_core_n
        # e.g. if per_core_n is 14, then out_subblock_w = 7
        out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

        # find the largest value that is a factor of per_core_m such that
        # out_subblock_w * out_subblock_h <= 8
        out_subblock_h = max(
            [
                i
                for i in range(1, max_subblock_w_h + 1)
                if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h
            ]
        )

        if overwrite_per_core_k is not None:
            per_core_k = overwrite_per_core_k

        if overwrite_subblock_w is not None:
            out_subblock_w = overwrite_subblock_w

        if overwrite_subblock_h is not None:
            out_subblock_h = overwrite_subblock_h

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=per_core_k,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=act,
            mcast_in0=True,
        )

    def matmul_1d_config_from_tensor_shapes(
        self,
        in0_shape,
        in1_shape,
        grid=ttnn.CoreGrid(x=8, y=8),
        act=None,
        is_fp32_accumulate=False,
        overwrite_subblock_w=None,
        overwrite_subblock_h=None,
    ):
        m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
        return self.matmul_1d_config(
            m,
            k,
            n,
            grid,
            act,
            is_fp32_accumulate,
            overwrite_subblock_w=overwrite_subblock_w,
            overwrite_subblock_h=overwrite_subblock_h,
        )

    def create_sharded_norm_config(self, grid):
        """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

        Args:
            grid (ttnn.CoreGrid): Grid specification for the norm operation
        """
        block_w = self.dim // grid.num_cores // self.tile_size
        # Find largest value <= 4 that evenly divides block_w
        subblock_w = 4
        while subblock_w > 0:
            if block_w % subblock_w == 0:
                break
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=self.tile_padded_batch_rows // self.tile_size,
            block_w=block_w,
            inplace=False,
        )

    def create_tokenizer(self):
        from transformers import AutoTokenizer

        # Mapping of base model names to their known tokenizer paths
        # These are the original models that have proper tokenizers
        base_model_tokenizer_mapping = {
            "Qwen2.5-0.5B": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
            "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
            "Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
            "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
            "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
            "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
            "Llama-3.1-70B": "meta-llama/Llama-3.1-70B-Instruct",
            "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
            "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
            "Llama-3.2-11B": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "Llama-3.2-90B": "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
            "Mistral-Small-3.1-24B": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "Phi-3-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
        }

        logger.info(f"Tokenizer path: {self.TOKENIZER_PATH}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Base model name: {self.base_model_name}")

        tokenizer = None
        try:
            # Try to load tokenizer from the original model path
            # If there is no Processor, it will return Tokenizer (useful for multimodal models)
            tokenizer = AutoTokenizer.from_pretrained(
                self.TOKENIZER_PATH,
                local_files_only=os.getenv("CI") == "true",
                trust_remote_code=self.trust_remote_code_hf,
            )
            logger.info(f"Successfully loaded tokenizer from {self.TOKENIZER_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {self.TOKENIZER_PATH}: {e}")

        # Only try fallback if initial load failed
        if tokenizer is None:
            # Try to use base model tokenizer as fallback
            fallback_tokenizer_path = base_model_tokenizer_mapping.get(self.base_model_name)

            # If no direct match, try to infer from model name patterns
            if not fallback_tokenizer_path:
                model_name_lower = self.model_name.lower()
                if "qwen2.5" in model_name_lower and "0.5b" in model_name_lower:
                    fallback_tokenizer_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
                elif "qwen2.5" in model_name_lower and "1.5b" in model_name_lower:
                    fallback_tokenizer_path = "Qwen/Qwen2.5-1.5B-Instruct"
                elif "qwen2.5" in model_name_lower and "3b" in model_name_lower:
                    fallback_tokenizer_path = "Qwen/Qwen2.5-3B-Instruct"
                elif "qwen2.5" in model_name_lower and "7b" in model_name_lower:
                    fallback_tokenizer_path = "Qwen/Qwen2.5-7B-Instruct"
                elif "qwen2.5" in model_name_lower and "14b" in model_name_lower:
                    fallback_tokenizer_path = "Qwen/Qwen2.5-14B-Instruct"
                elif "qwen2.5" in model_name_lower and "32b" in model_name_lower:
                    fallback_tokenizer_path = "Qwen/Qwen2.5-32B-Instruct"
                elif "qwen2.5" in model_name_lower and "72b" in model_name_lower:
                    fallback_tokenizer_path = "Qwen/Qwen2.5-72B-Instruct"
                elif "llama" in model_name_lower and "3.1" in model_name_lower and "8b" in model_name_lower:
                    fallback_tokenizer_path = "meta-llama/Llama-3.1-8B-Instruct"
                elif "llama" in model_name_lower and "3.1" in model_name_lower and "70b" in model_name_lower:
                    fallback_tokenizer_path = "meta-llama/Llama-3.1-70B-Instruct"
                elif "llama" in model_name_lower and "3.2" in model_name_lower and "1b" in model_name_lower:
                    fallback_tokenizer_path = "meta-llama/Llama-3.2-1B-Instruct"
                elif "llama" in model_name_lower and "3.2" in model_name_lower and "3b" in model_name_lower:
                    fallback_tokenizer_path = "meta-llama/Llama-3.2-3B-Instruct"
                elif "mistral" in model_name_lower and "7b" in model_name_lower:
                    fallback_tokenizer_path = "mistralai/Mistral-7B-Instruct-v0.3"
                elif "mistral" in model_name_lower and "small" in model_name_lower and "24b" in model_name_lower:
                    fallback_tokenizer_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
                elif "phi-3-mini" in model_name_lower and "128k" in model_name_lower and "instruct" in model_name_lower:
                    fallback_tokenizer_path = "microsoft/Phi-3-mini-128k-instruct"

            if fallback_tokenizer_path:
                logger.info(f"Attempting to use fallback tokenizer: {fallback_tokenizer_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        fallback_tokenizer_path, local_files_only=os.getenv("CI") == "true"
                    )
                    logger.info(f"Successfully loaded fallback tokenizer from {fallback_tokenizer_path}")
                except Exception as fallback_e:
                    logger.error(f"Failed to load fallback tokenizer from {fallback_tokenizer_path}: {fallback_e}")
                    raise fallback_e
            else:
                logger.error(f"No fallback tokenizer found for base model: {self.base_model_name}")
                raise Exception(f"No fallback tokenizer found for base model: {self.base_model_name}")

        # Add meta-compatible stop token list to the HF tokenizer
        if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
            tokenizer.stop_tokens = [tokenizer.eos_token_id]
            # Phi-3-mini uses "<|end|>" as EOS token
            if "phi-3-mini" in self.base_model_name.lower():
                tokenizer.stop_tokens.append(tokenizer.encode("<|end|>")[0])
        return tokenizer

    def create_processor(self):
        from transformers import AutoProcessor

        processor = None
        try:
            processor = AutoProcessor.from_pretrained(self.TOKENIZER_PATH, local_files_only=os.getenv("CI") == "true")
            logger.info(f"Successfully loaded processor from {self.TOKENIZER_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load processor from {self.TOKENIZER_PATH}: {e}")

        return processor

    def encode_prompt(self, prompt_text, system_prompt_text=None, instruct=True):
        if instruct:
            try:
                return encode_prompt_hf(self.tokenizer, prompt_text, system_prompt_text)
            except ValueError as e:
                logger.warning(f"Failed to encode chat prompt, are you sure this is an instruct model? Error: {e}")
                logger.warning(f"Falling back to base model encoding with no chat template")
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)

    def reference_lm_head(self):
        model = self.reference_transformer(wrap=False)
        layer = model.lm_head
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_transformer(self, wrap=True, load_checkpoint=False):
        from transformers import AutoConfig

        model_cls = self.get_hf_model_cls()

        # HF is much faster at loading from a checkpoint than generating from config
        # so use that by preference unless we don't have a checkpoint
        if self.dummy_weights and not load_checkpoint:
            config = AutoConfig.from_pretrained(
                self.LOCAL_HF_PARAMS[self.model_name],
                trust_remote_code=self.trust_remote_code_hf,
                local_files_only=os.getenv("CI") == "true",
            )
            if hasattr(config, "text_config"):
                config.text_config.num_layers = self.n_layers
                config.text_config.num_hidden_layers = self.n_layers
            else:
                config.num_layers = self.n_layers
                config.num_hidden_layers = self.n_layers

            try:
                # .from_pretrained + _init_weights works faster than .from_config
                model = model_cls.from_pretrained(
                    self.CKPT_DIR,
                    config=config,
                    torch_dtype="auto",
                    trust_remote_code=self.trust_remote_code_hf,
                    local_files_only=True,
                )
                model.apply(model._init_weights)
            except Exception as e:
                logger.info(f"Error loading dummy weights using .from_pretrained. Using .from_config. Error: {e}")
                model = model_cls.from_config(config, trust_remote_code=self.trust_remote_code_hf)
            # model.load_state_dict({k: torch.randn_like(v) for k, v in model.state_dict().items()})
        else:
            # Special case Qwen2.5-VL models until they are fully integrated into a HF release
            if "Qwen/Qwen2.5-VL" in self.model_name:
                from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig as AutoConfig
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                    Qwen2_5_VLForConditionalGeneration as AutoModelForCausalLM,
                )

                model_cls = AutoModelForCausalLM
            elif "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
                from transformers import AutoConfig
                from transformers import Mistral3ForConditionalGeneration as AutoModelForCausalLM

                model_cls = AutoModelForCausalLM
            else:
                from transformers import AutoConfig, AutoModelForCausalLM

                model_cls = AutoModelForCausalLM  # Conservative: Use AutoModelForCausalLM for standard models

            # HF is much faster at loading from a checkpoint than generating from config
            # so use that by preference unless we don't have a checkpoint
            if self.dummy_weights and not load_checkpoint:
                config = AutoConfig.from_pretrained(
                    self.LOCAL_HF_PARAMS[self.model_name],
                    trust_remote_code=self.trust_remote_code_hf,
                    local_files_only=os.getenv("CI") == "true",
                )
                if hasattr(config, "text_config"):
                    config.text_config.num_layers = self.n_layers
                    config.text_config.num_hidden_layers = self.n_layers
                else:
                    config.num_layers = self.n_layers
                    config.num_hidden_layers = self.n_layers

                try:
                    # .from_pretrained + _init_weights works faster than .from_config
                    model = model_cls.from_pretrained(
                        self.CKPT_DIR,
                        config=config,
                        torch_dtype="auto",
                        trust_remote_code=self.trust_remote_code_hf,
                        local_files_only=True,
                    )
                    model.apply(model._init_weights)
                except Exception as e:
                    logger.info(f"Error loading dummy weights using .from_pretrained. Using .from_config. Error: {e}")
                    model = model_cls.from_config(config, trust_remote_code=self.trust_remote_code_hf)
                # model.load_state_dict({k: torch.randn_like(v) for k, v in model.state_dict().items()})
            else:
                if self.cache_hf_flag and self.cached_hf_model is None:
                    model = model_cls.from_pretrained(
                        self.CKPT_DIR,
                        torch_dtype="auto",
                        local_files_only=os.getenv("CI") == "true",
                        trust_remote_code=self.trust_remote_code_hf,
                    )
                    self.cached_hf_model = model
                elif self.cache_hf_flag and self.cached_hf_model is not None:
                    model = self.cached_hf_model
                else:
                    # No caching - load fresh each time
                    model = model_cls.from_pretrained(
                        self.CKPT_DIR,
                        torch_dtype="auto",
                        trust_remote_code=self.trust_remote_code_hf,
                        local_files_only=os.getenv("CI") == "true",
                    )

        # HACK: Assume that we want the language model layers only
        if hasattr(model, "language_model"):
            model.model = model.language_model
            # We keep language_model because transformers don't let us change or delete it
        model.model.layers = model.model.layers[: self.n_layers]
        if wrap:
            wrapper = HfModelWrapper(model, self.head_dim, config=self.hf_config)
            return wrapper
        else:
            return model

    def reference_vision_multi_modal(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.multi_modal_projector
        return layer

    def reference_vision_rms_norm(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.multi_modal_projector.mm_soft_emb_norm
        # layer._load_state_dict = layer.load_state_dict
        # layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_rms_norm(self):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm

            return RMSNorm(self.dim, self.norm_eps)
        else:
            model = self.reference_transformer(wrap=False)
            layers = getattr(model, "layers", getattr(model, "model", {}).layers)
            layer = layers[0].input_layernorm
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
            return layer

    def reference_vision_transformer(self, wrap=True, load_checkpoint=False):
        if self.checkpoint_type == CheckpointType.HuggingFace:
            from transformers import AutoConfig

            model_cls = self.get_hf_model_cls()

            if self.dummy_weights and not load_checkpoint:
                config = AutoConfig.from_pretrained(self.LOCAL_HF_PARAMS[self.model_name])
                if hasattr(config, "text_config"):
                    config.text_config.num_layers = self.n_layers
                    config.text_config.num_hidden_layers = self.n_layers
                else:
                    config.num_layers = self.n_layers
                    config.num_hidden_layers = self.n_layers

                try:
                    # .from_pretrained + _init_weights works faster than .from_config
                    model = model_cls.from_pretrained(
                        self.CKPT_DIR,
                        config=config,
                        torch_dtype="auto",
                        trust_remote_code=self.trust_remote_code_hf,
                        local_files_only=True,
                    )
                    model.apply(model._init_weights)
                except Exception as e:
                    logger.info(f"Error loading dummy weights using .from_pretrained. Using .from_config. Error: {e}")
                    model = model_cls.from_config(config, trust_remote_code=self.trust_remote_code_hf)
                # model.load_state_dict({k: torch.randn_like(v) for k, v in model.state_dict().items()})
            else:
                if "gemma-3" in self.model_name:
                    from transformers import Gemma3ForConditionalGeneration

                    model = Gemma3ForConditionalGeneration.from_pretrained(self.CKPT_DIR)
                elif "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:  # Minimal addition
                    from transformers import Mistral3ForConditionalGeneration

                    model = Mistral3ForConditionalGeneration.from_pretrained(self.CKPT_DIR, torch_dtype=torch.bfloat16)
                else:
                    from transformers import AutoModelForCausalLM

                    if self.cached_hf_model is None:
                        model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR)
                        self.cached_hf_model = model
                    else:
                        model = self.cached_hf_model
                    model.model.layers = model.model.layers[: self.n_layers]
            if wrap:
                wrapper = HfModelWrapper(model, self.head_dim)
                return wrapper
            else:
                return model

    def reference_gemma_model(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_model(self):
        model = self.reference_vision_transformer(wrap=False)
        if "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
            # Mistral-Small-3.1-24B-Instruct-2503 has a different structure
            layer = model.vision_tower
        else:
            layer = model.vision_tower.vision_model
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_mlp(self, layer_idx=0):
        model = self.reference_vision_transformer(wrap=False)
        if "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
            layer = model.vision_tower.transformer.layers[layer_idx].feed_forward
        else:
            layer = model.vision_tower.vision_model.encoder.layers[0].mlp
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_siglip_patch_embed(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings.patch_embedding
        # layer._load_state_dict = layer.load_state_dict
        # layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_pos_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings.position_embedding
        # layer._load_state_dict = layer.load_state_dict
        # layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings
        # layer._load_state_dict = layer.load_state_dict
        # layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_layernorm(self, layer_name="layer_norm1"):
        model = self.reference_vision_transformer(wrap=False)
        if layer_name == "layer_norm1":
            layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm1
        elif layer_name == "layer_norm2":
            layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm2
        else:
            layer = model.vision_tower.vision_model.post_layernorm
        # layer._load_state_dict = layer.load_state_dict
        # layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_attention(self, layer_idx=0):
        model = self.reference_vision_transformer(wrap=False)
        if "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
            layer = model.vision_tower.transformer.layers[layer_idx].attention
        else:
            layer = model.vision_tower.vision_model.encoder.layers[0].self_attn  # Common naming
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_encoder_block(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder.layers[0]
        # layer._load_state_dict = layer.load_state_dict
        # layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_encoder(self):
        model = self.reference_vision_transformer(wrap=False)
        if "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
            # For Mistral: vision_tower is the PixtralVisionModel directly
            layer = model.vision_tower.transformer
        else:
            # For other models: vision_tower.vision_model.encoder
            layer = model.vision_tower.vision_model.encoder
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    # Minimal addition for Mistral vision support
    def reference_pixtral_image_block(self, layer_num=0):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.transformer.layers[layer_num]
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    # Minimal addition for Mistral vision support
    def reference_vision_rms(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.transformer.layers[0].ffn_norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    # Minimal addition for Mistral vision support
    def reference_conv2d_patch(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.patch_conv
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    # Minimal addition for Mistral vision support
    def reference_vision_rot_emb(self):
        model = self.reference_vision_transformer(wrap=False)
        if "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
            layer = model.vision_tower.patch_positional_embedding
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_mlp(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].mlp
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(
            convert_meta_to_hf(x, self.head_dim, fuse_mlp=self.fuse_mlp)
        )
        return layer

    def reference_embedding(self, reference_model=None):
        if reference_model is None:
            model = self.reference_transformer(wrap=False)
            layer = model.model.embed_tokens
        else:
            layer = reference_model.model.model.embed_tokens

        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_decoder(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0]
        use_position_embeddings = layer.__class__.__name__ != "Phi3DecoderLayer" or self.base_model_name in ("phi-4",)
        if hasattr(model.model, "rotary_emb_local"):
            rotary_emb_local = model.model.rotary_emb_local
        else:
            rotary_emb_local = None
        wrapper = HfDecoderWrapper(
            layer, self.head_dim, model.model.rotary_emb if use_position_embeddings else None, rotary_emb_local
        )
        return wrapper

    def reference_attention(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].self_attn
        use_position_embeddings = "position_embeddings" in inspect.signature(layer.forward).parameters
        wrapper = HfAttentionWrapper(layer, self.head_dim, model.model.rotary_emb if use_position_embeddings else None)
        return wrapper

    def set_tg_attention_config(self):
        shard_spec_n_cores_grid = ttnn.CoreRangeSet({num_to_corerange(40)})

        self.model_config["CREATE_HEAD_INPUT_MEMCFG"] = (
            None
            if self.dim < 4096
            else ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec_n_cores_grid,
                    [
                        32,
                        32,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        )

        if self.is_galaxy:
            num_cores = 40 if self.dim == 8192 else (24 if self.dim == 4096 else (20 if self.dim == 3072 else 12))

            self.model_config["QKV_OUT_GATHERED_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
                shape=(32 * mesh_cols, 32),  # mesh_cols = 4
                core_grid=num_to_coregrid(num_cores),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["SELF_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
                shape=(32 * mesh_rows, self.dim // 4 // min(32, self.dim // 4 // 32)),
                core_grid=num_to_coregrid(min(32, self.dim // 4 // 32)),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
                shape=(32 * mesh_cols, 32),  # mesh_cols = 4
                core_grid=num_to_coregrid(min(32, self.dim // 8 // 32)),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            qkv_core_grid = self.dram_shard_core_grid_for_k(self.dim)
            self.model_config["QKV_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
                (
                    self.tile_size * mesh_rows,
                    self.dim // qkv_core_grid.num_cores,
                ),  # Shard shape: [32, 128] -> 1 shard per core
                core_grid=qkv_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            gather_core_grid = self.dram_shard_core_grid_for_k(self.dim // 4)
            self.model_config["SELF_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
                (
                    self.tile_size * mesh_rows,
                    self.dim // 4 // gather_core_grid.num_cores,
                ),
                core_grid=gather_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            users_core_grid = self.dram_shard_core_grid_for_k(self.dim // 8)
            self.model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
                (
                    self.tile_size * mesh_cols,
                    self.dim // 8 // users_core_grid.num_cores,
                ),
                core_grid=users_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )


class HfAttentionWrapper:
    def __init__(self, attention, head_dim, rotary_emb):
        from transformers import DynamicCache

        super().__init__()
        self.attention = attention
        self.past_key_value = DynamicCache()
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])

        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(x, position_ids)
            output, *_ = self.attention(
                x,
                position_embeddings=position_embeddings,
                past_key_value=self.past_key_value,
                use_cache=True,
                attention_mask=mask,
            )
        else:
            output, _, self.past_key_value = self.attention(
                x,
                past_key_value=self.past_key_value,
                use_cache=True,
                position_ids=position_ids,
                attention_mask=mask,
            )
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        try:  # Checking for fused qkv layer
            fuse_qkv = hasattr(self.attention, "qkv_proj")
        except:
            fuse_qkv = False
        return self.attention.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim, fuse_qkv))

    @property
    def cache_k(self):
        [(k, v)] = self.past_key_value.to_legacy_cache()
        hf_k = k.permute(0, 2, 1, 3)  # match meta-style reference which uses (batch_size, seq, n_kv_heads, head_dim)
        batch_size, seq_len, n_heads, head_dim = hf_k.shape

        meta_k = torch.zeros_like(hf_k)
        for b in range(batch_size):
            for s in range(seq_len):
                # Flatten just heads and head_dim
                flat = hf_k[b, s].flatten()
                # Apply reverse_permute
                transformed = reverse_permute(flat.unsqueeze(-1), n_heads, flat.shape[0], 1).squeeze(-1)
                # Restore heads and head_dim shape
                meta_k[b, s] = transformed.reshape(n_heads, head_dim)

        return meta_k

    @property
    def cache_v(self):
        [(k, v)] = self.past_key_value.to_legacy_cache()
        return v.permute(0, 2, 1, 3)  # match meta-style reference which uses (batch_size, seq, n_kv_heads, head_dim)


class HfDecoderWrapper:
    def __init__(self, decoder, head_dim, rotary_emb, rotary_emb_local=None):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.rotary_emb_local = rotary_emb_local
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(x, position_ids)

        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

        if self.rotary_emb_local is not None:
            position_embeddings_local = self.rotary_emb_local(x, position_ids)
            result = self.decoder.forward(
                x,
                position_embeddings_global=position_embeddings,
                position_embeddings_local=position_embeddings_local,
                past_key_value=self.past_key_values,
                use_cache=True,
                position_ids=position_ids,
                attention_mask=mask,
            )
        else:
            result = self.decoder.forward(
                x,
                position_embeddings=position_embeddings,
                past_key_value=self.past_key_values,
                use_cache=True,
                position_ids=position_ids,
                attention_mask=mask,
            )

        output = result[0]
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        try:  # Checking for fused qkv and mlp layers
            fuse_qkv = hasattr(self.decoder.self_attn, "qkv_proj")
            fuse_mlp = hasattr(self.decoder.mlp, "gate_up_proj")
        except:
            fuse_qkv, fuse_mlp = False, False
        return self.decoder.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim, fuse_qkv, fuse_mlp))


class HfModelWrapper:
    def __init__(self, model, head_dim, config=None):
        from transformers import DynamicCache

        self.model = model
        self.head_dim = head_dim
        self.config = config
        self.past_key_values = DynamicCache()

    def forward(self, inputs_embeds, start_pos, mode="decode"):
        position_ids = torch.tensor(
            [list(range(start_pos, start_pos + inputs_embeds.shape[1]))] * inputs_embeds.shape[0]
        )
        logits, new_cache, hidden_states = self.model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict=False,
            output_hidden_states=True,
        )
        self.past_key_values = new_cache
        return logits if mode == "decode" else hidden_states[-2]  # last hidden state is final norm

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        try:  # Checking for fused qkv and mlp layers
            fuse_qkv = hasattr(self.model.model.layers[0].self_attn, "qkv_proj")
            fuse_mlp = hasattr(self.model.model.layers[0].mlp, "gate_up_proj")
        except:
            fuse_qkv, fuse_mlp = False, False
        return self.model.load_state_dict(
            convert_meta_to_hf(state_dict, self.head_dim, fuse_qkv, fuse_mlp, self.config)
        )

    def eval(self):
        self.model.eval()

    @property
    def cache_k(self):
        kvs = self.past_key_values.to_legacy_cache()
        meta_ks = []
        for k, v in kvs:
            hf_k = k.permute(
                0, 2, 1, 3
            )  # match meta-style reference which uses (batch_size, seq, n_kv_heads, head_dim)
            batch_size, seq_len, n_heads, head_dim = hf_k.shape

            meta_k = torch.zeros_like(hf_k)
            for b in range(batch_size):
                for s in range(seq_len):
                    # Flatten just heads and head_dim
                    flat = hf_k[b, s].flatten()
                    # Apply reverse_permute
                    transformed = reverse_permute(flat.unsqueeze(-1), n_heads, flat.shape[0], 1).squeeze(-1)
                    # Restore heads and head_dim shape
                    meta_k[b, s] = transformed.reshape(n_heads, head_dim)

            meta_ks.append(meta_k)

        return meta_ks

    @property
    def cache_v(self):
        kvs = self.past_key_values.to_legacy_cache()
        return [
            v.permute(0, 2, 1, 3) for k, v in kvs
        ]  # match meta-style reference which uses (batch_size, seq, n_kv_heads, head_dim)


class DecodersPrecision:
    @classmethod
    def from_string(cls, optimizations: str):
        if optimizations == "performance":
            return cls.performance
        elif optimizations == "accuracy":
            return cls.accuracy
        else:
            raise ValueError(
                f"Invalid optimization configuration: {optimizations}. Allowed values are 'performance' or 'accuracy'"
            )

    @classmethod
    def accuracy(cls, num_decoders, model_name):
        inst = cls._precision_factory(num_decoders, model_name, ModelOptimizations.accuracy)
        inst.__name__ = "accuracy"
        return inst

    @classmethod
    def performance(cls, num_decoders, model_name):
        inst = cls._precision_factory(num_decoders, model_name, ModelOptimizations.performance)
        inst.__name__ = "performance"
        return inst

    def __init__(self, num_decoders, model_name, decoder_conf: dict = None):
        if decoder_conf is None:
            decoder_conf = ModelOptimizations.accuracy(model_name)
        self.decoder_optimizations = {decoder_id: decoder_conf for decoder_id in range(num_decoders)}
        self._update_full_name()

    def set_decoder_conf(self, decoder_id, conf: ModelOptimizations):
        self.decoder_optimizations[decoder_id] = conf
        self._update_full_name()

    def get_tensor_dtype(self, decoder_id, tensor: TensorGroup):
        precision_setting_lookup = {
            PrecisionSetting.BFP4: ttnn.bfloat4_b,
            PrecisionSetting.BFP8: ttnn.bfloat8_b,
            PrecisionSetting.BF16: ttnn.bfloat16,
            None: None,  # this signals that original dtype should be used
        }
        if (
            decoder_id not in self.decoder_optimizations
            or tensor not in self.decoder_optimizations[decoder_id].tensor_dtype_settings
        ):
            return None

        key = self.decoder_optimizations[decoder_id].tensor_dtype_settings[tensor]

        if key is None or key not in precision_setting_lookup:
            return None

        return precision_setting_lookup[key]

    def get_math_fidelity(self, decoder_id, op: OpGroup, configuration: ModelArgs):
        math_fidelity_setting_lookup = {
            MathFidelitySetting.LOFI: configuration.compute_kernel_config_lofi,
            MathFidelitySetting.HIFI2: configuration.compute_kernel_config_hifi2,
            MathFidelitySetting.HIFI2_NA: configuration.compute_kernel_config_hifi2_na,
            MathFidelitySetting.HIFI2_FP16: configuration.compute_kernel_config_hifi2_fp16,
            MathFidelitySetting.HIFI4: configuration.compute_kernel_config_hifi4,
            MathFidelitySetting.HIFI4_FP32: configuration.compute_kernel_config_hifi4_fp32,
        }
        return math_fidelity_setting_lookup[self.decoder_optimizations[decoder_id].op_fidelity_settings[op]]

    def _update_full_name(self):
        self._full_name = " | ".join(
            f"Decoder {decoder_id}: {conf._full_name}" for decoder_id, conf in self.decoder_optimizations.items()
        )

    @classmethod
    def _precision_factory(cls, num_decoders, model_name, optimization_level):
        # use respective configuration for each optimization level
        decoder_config_filename = None
        match optimization_level:
            case ModelOptimizations.accuracy:
                decoder_config_filename = ACCURACY_DECODER_CONFIG_FILENAME
            case ModelOptimizations.performance:
                decoder_config_filename = PERFORMANCE_DECODER_CONFIG_FILENAME
            case _:
                raise ValueError(f"optimization_level ({optimization_level}) not implemented")

        # check if decoder config exists, if it exists load it else use optimization_level
        model_params_dir = Path(__file__).parent.parent
        decoder_config_path = model_params_dir / "model_params" / model_name / decoder_config_filename
        inst = None
        if decoder_config_path.exists():
            inst = parse_decoder_json(decoder_config_path, default_optimization=optimization_level)
            logger.info(
                f"Model {model_name} requires specific TensorPrecision and OpFidelity configuration, using {decoder_config_path}"
            )
        else:
            inst = cls(num_decoders, model_name, optimization_level(model_name))

        return inst


def num_to_corerange(
    x: int,
    start_core: ttnn.CoreCoord = ttnn.CoreCoord(0, 0),
    grid_x: int = 8,
    grid_y: int = 8,
) -> ttnn.CoreRange:
    """
    Construct a rectangular CoreRange of exactly ``x`` cores starting at
    ``start_core`` on a ``grid_x × grid_y`` core grid.

    The CoreRange is allocated in row-major order semantics but must form
    a single contiguous rectangle representable by ``ttnn.CoreRange``.

    Defaults to an 8×8 grid for backward compatibility.
    """

    # --- basic sanity ---
    assert x > 0, "x must be positive"
    assert grid_x > 0 and grid_y > 0
    assert 0 <= start_core.x < grid_x
    assert 0 <= start_core.y < grid_y

    sx, sy = start_core.x, start_core.y

    # --- linear availability (row-major correctness) ---
    remaining_linear_cores = (grid_x - sx) + (grid_y - sy - 1) * grid_x  # remainder of start row  # full rows below
    assert remaining_linear_cores >= x, (
        f"Not enough cores from start_core {start_core} "
        f"to allocate {x} cores (only {remaining_linear_cores} available)"
    )

    # --- rectangular availability ---
    remaining_x = grid_x - sx
    remaining_y = grid_y - sy

    # --- shape rule ---
    assert x < grid_x or x % grid_x == 0, f"x must be < grid_x ({grid_x}) or a multiple of grid_x"

    # --- choose rectangle dimensions ---
    num_x = min(x, remaining_x)
    num_y = x // num_x

    assert num_x * num_y == x, f"x={x} cannot form a rectangular CoreRange starting at {start_core}"

    # --- bounds check ---
    assert num_y <= remaining_y, f"CoreRange height {num_y} exceeds available rows {remaining_y}"

    end_x = sx + num_x - 1
    end_y = sy + num_y - 1

    return ttnn.CoreRange(
        start_core,
        ttnn.CoreCoord(end_x, end_y),
    )


def num_to_coregrid(x):
    if x % 8 == 0:
        return ttnn.CoreGrid(y=x // 8, x=8)
    if x == 12:
        return ttnn.CoreGrid(y=2, x=6)
    if x == 20:
        return ttnn.CoreGrid(y=4, x=5)


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
