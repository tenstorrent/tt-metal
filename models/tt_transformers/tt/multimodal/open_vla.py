"""
open_vla.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
    => https://huggingface.co/openvla/openvla-7b/blob/main/modeling_prismatic.py
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from PIL import Image
from safetensors import safe_open
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vit.tt import ttnn_optimized_vit_highres_gs as ttnn_optimized_vit_highres
from models.tt_transformers.demo.simple_text_demo import prepare_generator_args
from models.tt_transformers.tt.common import (
    create_tt_model,
    get_block_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)


def ttnn_to_torch_safe(tensor, mesh_device=None):
    """
    Convert ttnn tensor to torch, handling multi-device mesh (N300/T3K).
    For replicated tensors, gets from first device to avoid doubling.
    Works for both single-device (P150/N150) and multi-device (N300/T3K).
    """
    # Check if tensor is multi-device by trying to get device tensors
    try:
        device_tensors = ttnn.get_device_tensors(tensor)
        if len(device_tensors) > 1:
            # Multi-device tensor - get from first device
            return ttnn.to_torch(device_tensors[0]).float()
        else:
            # Single device tensor wrapped in list
            return ttnn.to_torch(device_tensors[0]).float()
    except (RuntimeError, TypeError, AttributeError):
        # Not a multi-device tensor, convert directly
        return ttnn.to_torch(tensor).float()


"""
configuration_prismatic.py

HuggingFace-style configuration definition for Prismatic VLMs, inheriting from `transformers.PretrainedConfig`.
Default configuration specifies `siglip-224px+7b`.
"""

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "siglip-vit-so400m": [224], "dinov2-vit-l": [224],
    "dinosiglip-vit-so-224px": [224, 224],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "dinov2-vit-l": [None],
    "siglip-vit-so400m": [None], "siglip-vit-so400m-384px": [None],
    "dinosiglip-vit-so-224px": [None, None],
}

LLM_BACKBONE_TO_HF_PATH = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf",
}
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama",
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)
# fmt: on


class PerfCheckpoints:
    checkpoints: ClassVar[List[Dict[str, int]]] = None

    def __init__(self):
        self.times = {}
        if self.checkpoints is None:
            self.checkpoints = []
        self.present_keys_counter = {}

    def checkpoint(self, key):
        if key not in self.present_keys_counter:
            self.present_keys_counter[key] = 0
            new_key = f"{key}_{self.present_keys_counter[key]}"
            self.times[new_key] = time.time()
        else:
            self.present_keys_counter[key] += 1
            new_key = f"{key}_{self.present_keys_counter[key]}"
            self.times[new_key] = time.time()

    def get_pairs(self):
        pairs = []
        keys = list(key for key in self.present_keys_counter if key.startswith("start"))
        for key in keys:
            end_key = key.replace("start", "end")
            if end_key in self.present_keys_counter:
                pairs.append((key, end_key))
        return pairs

    def analyze(self, pairs=None):
        results = {}
        if pairs is None:
            pairs = self.get_pairs()
        for pair in pairs:
            assert len(pair) == 2, "Each pair must contain exactly two keys."
            assert pair[0] in self.present_keys_counter, f"Key {pair[0]} not found in checkpoints."
            assert pair[1] in self.present_keys_counter, f"Key {pair[1]} not found in checkpoints."
            assert (
                self.present_keys_counter[pair[0]] == self.present_keys_counter[pair[1]]
            ), f"Key {pair[0]} and {pair[1]} must have the same number of occurrences."
            for counter in range(self.present_keys_counter[pair[0]] + 1):
                key1 = f"{pair[0]}_{counter}"
                key2 = f"{pair[1]}_{counter}"
                results[f"{key1}->{key2}"] = self.times[key2] - self.times[key1]
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        return results

    def reset(self):
        if len(self.times) > 0:
            self.checkpoints.append(self.times)
            self.times = {}
            self.present_keys_counter = {}


CHECKPOINTS = PerfCheckpoints()


def map_openvla_hf_to_meta_keys(loaded_weights):
    hf_to_meta = {
        # Top level mappings
        "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
        "language_model.model.norm.weight": "norm.weight",
        "language_model.lm_head.weight": "output.weight",
        # Layer level mappings
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        # Attention module mappings
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "self_attn.q_proj.bias": "attention.wq.bias",
        "self_attn.k_proj.bias": "attention.wk.bias",
        "self_attn.v_proj.bias": "attention.wv.bias",
        # Feed forward module mappings
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        # Direct module mappings
        "gate_proj.weight": "w1.weight",
        "down_proj.weight": "w2.weight",
        "up_proj.weight": "w3.weight",
        "q_proj.weight": "wq.weight",
        "k_proj.weight": "wk.weight",
        "v_proj.weight": "wv.weight",
        "o_proj.weight": "wo.weight",
        "q_proj.bias": "wq.bias",
        "k_proj.bias": "wk.bias",
        "v_proj.bias": "wv.bias",
        "weight": "emb.weight",  # For host embeddings
        # Full path layer mappings
        "language_model.model.layers.{layer}.input_layernorm.weight": "layers.{layer}.attention_norm.weight",
        "language_model.model.layers.{layer}.post_attention_layernorm.weight": "layers.{layer}.ffn_norm.weight",
        "language_model.model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attention.wq.weight",
        "language_model.model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attention.wk.weight",
        "language_model.model.layers.{layer}.self_attn.v_proj.weight": "layers.{layer}.attention.wv.weight",
        "language_model.model.layers.{layer}.self_attn.o_proj.weight": "layers.{layer}.attention.wo.weight",
        "language_model.model.layers.{layer}.self_attn.q_proj.bias": "layers.{layer}.attention.wq.bias",
        "language_model.model.layers.{layer}.self_attn.k_proj.bias": "layers.{layer}.attention.wk.bias",
        "language_model.model.layers.{layer}.self_attn.v_proj.bias": "layers.{layer}.attention.wv.bias",
        "language_model.model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.feed_forward.w1.weight",
        "language_model.model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.feed_forward.w3.weight",
        "language_model.model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.feed_forward.w2.weight",
    }

    meta_state_dict = {}
    for key, tensor in loaded_weights.items():
        if key in hf_to_meta:
            # Direct match for top-level keys
            meta_state_dict[hf_to_meta[key]] = tensor
        elif "language_model.model.layers." in key:
            # Extract layer number and form a template key
            parts = key.split(".")
            layer_num = parts[3]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "language_model.model.layers.{layer}." + ".".join(parts[4:])
            if template_key in hf_to_meta:
                meta_state_dict[hf_to_meta[template_key].format(layer=layer_num)] = tensor

    return meta_state_dict


def get_LLama2OpenVLAArgs(state_dict):
    class LLama2OpenVLAArgs(ModelArgs):
        # CRITICAL: Use BF16 for LM head to avoid token logit corruption
        # BFP8 causes specific tokens (like 31872) to have inflated logits
        lm_head_dtype = ttnn.bfloat16

        def __init__(self, *args, **kwargs):
            HF_MODEL = os.getenv("HF_MODEL")
            assert (
                HF_MODEL == "meta-llama/Llama-2-7b-hf"
            ), f"When LLama2OpenVLAArgs is used, HF_MODEL must be meta-llama/Llama-2-7b-hf"
            super().__init__(*args, **kwargs)

        def _set_params_from_dict(self, config):
            new_config = {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "initializer_range": 0.02,
                "intermediate_size": 11008,
                "max_position_embeddings": 2048,
                "model_type": "llama",
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "num_key_value_heads": 32,
                "pad_token_id": 32000,
                "pretraining_tp": 1,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
                "torch_dtype": "float16",
                "transformers_version": "4.38.0",
                "use_cache": True,
                "vocab_size": 32000 if state_dict is None else 32064,
            }
            text_config = config.get("text_config", config)
            for key, value in text_config.items():
                if key not in new_config:
                    new_config[key] = value

            return super()._set_params_from_dict(new_config)

        def load_state_dict(self):
            if state_dict is None:
                return super().load_state_dict()
            new_state_dict = map_openvla_hf_to_meta_keys(state_dict)
            return new_state_dict

        def weight_cache_path(self, dtype):
            """Use a SEPARATE cache path for OpenVLA weights to avoid conflicts with base Llama."""
            import ttnn

            # When OpenVLA weights are provided, use a different cache directory
            if state_dict is not None:
                cache_suffix = {
                    ttnn.bfloat16: "tensor_cache_openvla_bf16",
                    ttnn.bfloat8_b: "tensor_cache_openvla_bfp8",
                }[dtype]
            else:
                # Fall back to default base Llama cache
                cache_suffix = {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            return self.model_cache_path / cache_suffix

    return LLama2OpenVLAArgs


class OpenVLALanguageModel(GenerationMixin):
    def __init__(self, device, local_state_dict=None):
        # ============================================================================
        # OPTION 1: BFP8 with BF16 KV cache only (works on N150, single device)
        # This has precision issues with action tokens but fits in L1 on single device
        # ============================================================================
        # def custom_kv_bf16_optimization(model_args):
        #     """Custom optimization: BF16 for KV cache only (N150 compatible)."""
        #     base_opt = ModelOptimizations.performance(model_args.model_name)
        #     base_opt.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BF16
        #     return DecodersPrecision(model_args.n_layers, model_args.model_name, base_opt)

        # ============================================================================
        # OPTION 2: Full BF16 attention like Qwen2.5-7B (requires N300 / 2 devices)
        # This is needed for action token prediction which has very small gaps
        # between correct and incorrect tokens (0.06-0.25), while BFP8 error is ~0.5-1.5.
        # ============================================================================
        def qwen_style_bf16_attention(model_args):
            """
            Custom optimization: BF16 attention only (FFN stays BFP8).
            Fits on N300 (2 devices).
            LM head precision controlled separately via dtype parameter.
            """
            settings = {
                "TensorPrecision": {
                    TensorGroup.WQKV: PrecisionSetting.BF16,
                    TensorGroup.KV_CACHE: PrecisionSetting.BF16,
                    TensorGroup.WO: PrecisionSetting.BF16,
                    TensorGroup.ACTIVATION: PrecisionSetting.BF16,  # Q in SDPA must be BF16 too!
                    # FFN stays at BFP8 to fit in memory
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
            model_opt = ModelOptimizations(settings)
            return DecodersPrecision(model_args.n_layers, model_args.model_name, model_opt)

        def bf16_lmhead_bfp8_attn(model_args):
            """
            TEST CONFIG: BF16 KV cache only, everything else BFP8.
            LM head BF16 is achieved by passing dtype=ttnn.bfloat16 to create_tt_model.
            This tests if LM head precision alone can fix instruction sensitivity.
            """
            settings = {
                "TensorPrecision": {
                    TensorGroup.KV_CACHE: PrecisionSetting.BF16,
                    # WQKV, WO, FFN all stay BFP8 to save memory
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
            model_opt = ModelOptimizations(settings)
            return DecodersPrecision(model_args.n_layers, model_args.model_name, model_opt)

        def full_bf16_config(model_args):
            """
            FULL BF16 CONFIG (like Qwen2.5-7B) - Requires T3K (8 devices) or more memory.
            All weights in BF16: WQKV, WO, FF1, FF2, FF3, KV_CACHE, LM_HEAD.
            This should give best precision for instruction sensitivity.
            """
            settings = {
                "TensorPrecision": {
                    TensorGroup.WQKV: PrecisionSetting.BF16,
                    TensorGroup.WO: PrecisionSetting.BF16,
                    TensorGroup.FF1_FF3: PrecisionSetting.BF16,
                    TensorGroup.FF2: PrecisionSetting.BF16,
                    TensorGroup.KV_CACHE: PrecisionSetting.BF16,
                    TensorGroup.ACTIVATION: PrecisionSetting.BF16,  # Q in SDPA must be BF16 too!
                    # LM head controlled by dtype parameter in create_tt_model
                },
                "OpFidelity": {
                    OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                    OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                    OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                    OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                    OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                    OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
                    OpGroup.LI_FF1_FF3_DECODE: MathFidelitySetting.HIFI4,
                    OpGroup.LI_FF1_FF3_PREFILL: MathFidelitySetting.HIFI4,
                    OpGroup.LI_FF2_DECODE: MathFidelitySetting.HIFI4,
                    OpGroup.LI_FF2_PREFILL: MathFidelitySetting.HIFI4,
                },
            }
            model_opt = ModelOptimizations(settings)
            return DecodersPrecision(model_args.n_layers, model_args.model_name, model_opt)

        # Detect device type for automatic config selection
        num_devices = device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1
        is_t3k_or_larger = num_devices >= 8
        is_multi_device = num_devices > 1
        mesh_shape = tuple(device.shape) if isinstance(device, ttnn.MeshDevice) else (1, 1)

        # Debug trace for N300/multi-device setup
        print(f"🔧 OpenVLA LLM Config:")
        print(f"   - Device type: {'MeshDevice' if isinstance(device, ttnn.MeshDevice) else 'SingleDevice'}")
        print(f"   - num_devices={num_devices}, mesh_shape={mesh_shape}")
        print(f"   - is_multi_device={is_multi_device}, is_t3k_or_larger={is_t3k_or_larger}")
        print(f"   - Precision: {'FULL BF16 (T3K)' if is_t3k_or_larger else 'BF16 attention + BFP8 FFN (N300/P150)'}")
        print(f"   - LM Head dtype: BF16 (via LLama2OpenVLAArgs.lm_head_dtype)")

        self.generator_args_config = {
            "num_devices": num_devices,
            "data_parallel": 1,
            "mesh_device": device,
            "instruct": False,
            "global_batch_size": 1,
            # Auto-select config based on device:
            # - T3K (8+ devices): Full BF16 for best instruction sensitivity
            # - N300 (2 devices): BF16 attention only (FFN/LM head BFP8 to fit)
            "optimizations": full_bf16_config if is_t3k_or_larger else qwen_style_bf16_attention,
            "max_seq_len": 1024,  # Vision (~512) + text gets padded to 1024
            "page_params": {"page_block_size": 32, "page_max_num_blocks_per_dp": 512},  # Reduced blocks
            "paged_attention": True,
            "num_layers": 32,  # Default number of layers for LLaMA model
        }

        def model_factory_fn(*args, **kwargs):
            # T3K: Full BF16 (including LM head) for best precision
            # N300: BFP8 LM head (default), BF16 attention via optimization
            if is_t3k_or_larger:
                kwargs.pop("dtype", None)  # Remove default BFP8 dtype
                return create_tt_model(
                    *args, **kwargs, dtype=ttnn.bfloat16, ModelArgsClass=get_LLama2OpenVLAArgs(local_state_dict)
                )
            else:
                return create_tt_model(*args, **kwargs, ModelArgsClass=get_LLama2OpenVLAArgs(local_state_dict))

        (
            self.model_args,
            self.model,
            self.page_table,
            self.tt_kv_cache,
            self.tokenizer,
            self.processor,
        ) = prepare_generator_args(**self.generator_args_config, model_factory_fn=model_factory_fn)
        self.generator = Generator(self.model, self.model_args, device, self.tokenizer)
        self.num_actions = 1

    def predict_text(self, input_prompts, max_generated_tokens=200):
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            self.tokenizer,
            self.model_args,
            self.generator_args_config["instruct"],
            max_generated_tokens,
            max_prefill_len=self.generator_args_config["max_seq_len"],
        )
        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_generated_tokens + max_encoded_prompt_len <= self.generator_args_config["max_seq_len"]
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({self.generator_args_config['max_seq_len']})"
        paged_cache_max_seq_len = (
            self.generator_args_config["page_params"]["page_block_size"]
            * self.generator_args_config["page_params"]["page_max_num_blocks_per_dp"]
            / self.generator_args_config["global_batch_size"]
        )
        assert (
            max_generated_tokens + max_encoded_prompt_len <= paged_cache_max_seq_len
        ), f"max_generated_tokens ({max_generated_tokens}) needs to be <= than paged_cache_max_seq_len ({paged_cache_max_seq_len})"
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)
        logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        prefilled_token = torch.argmax(logits, dim=-1)

        output = encoded_prompts[0][: prefill_lens[0]]
        user_tok = int(prefilled_token[0].item())
        output.append(user_tok)
        prompt = input_prompts[0]

        device_sampling_params = None
        sampling_params = {"temperature": 0, "top_p": 0.08}
        stop_at_eos = True
        user_done = False
        # Initial positions
        current_pos = torch.tensor([decoding_pos[0]])

        # Start decoding
        iteration = 0
        user_decoding = True

        out_tok = prefilled_token
        while user_decoding:
            # Run decode forward
            decode_output = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=device_sampling_params,
            )
            # decode_forward_text returns (logits, log_probs) tuple
            logits = decode_output[0] if isinstance(decode_output, tuple) else decode_output

            # Get the next token
            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)
            else:
                # TODO Fix use case with temperature > 0
                _, out_tok = sample_host(
                    logits,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )

            current_pos += 1
            # Save output token to print out later
            user_tok = out_tok.item()
            if (
                user_tok not in self.tokenizer.stop_tokens and not user_done
            ):  # Read until an eos token (e.g. <|eot_id|>); create_tokenizer adds stop_tokens to HF tokenizers
                output.append(user_tok)
            else:
                if (
                    stop_at_eos
                ):  # For performance gathering in CI, we want to sometimes force decoding for a fixed number of iterations
                    user_done = True
                    user_decoding = False

            # Print out generated outputs for each user at the end of every iteration
            text = "".join(self.tokenizer.decode(output))
            if len(text) > 100:
                text = "..." + text[-97:]
            text = text.replace("\n", " ")
            print(text)
            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= max_generated_tokens:
                user_decoding = False
        text = self.tokenizer.decode(output)
        prompt_including_assistant_tags = self.tokenizer.decode(
            self.model_args[0].encode_prompt(prompt, instruct=self.generator_args_config["instruct"])
        )
        text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
        return text_after_prompt

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.model[0].embd

    def set_input_embeddings(self, value: nn.Module) -> None:
        pass

    def get_output_embeddings(self) -> nn.Module:
        pass

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        pass

    def get_decoder(self) -> nn.Module:
        pass

    def set_decoder(self, decoder: nn.Module) -> None:
        pass

    def tie_weights(self) -> None:
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        pass

    def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len):
        # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(prefill_len, block_size)
        return page_table[:, :num_blocks]

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        assert input_ids is None, f"{self.__class__.__name__} does not accept input_ids"
        assert position_ids is None, f"{self.__class__.__name__} does not accept position_ids"
        assert past_key_values is None, f"{self.__class__.__name__} does not accept past_key_values"
        assert labels is None, f"{self.__class__.__name__} does not accept labels"
        assert not use_cache, f"{self.__class__.__name__} does not accept use_cache"
        assert not output_attentions, f"{self.__class__.__name__} does not accept output_attentions"
        assert not output_hidden_states, f"{self.__class__.__name__} does not accept output_hidden_states"
        assert return_dict, f"{self.__class__.__name__} does not accept return_dict=False"

        # FIX: Handle both 3D [B,S,D] and 4D [1,1,S,D] input shapes
        # Always squeeze to get correct seq_len before computing positions
        emb_shape = inputs_embeds.shape
        if len(emb_shape) == 4 and emb_shape[1] == 1:
            # Shape is [1, 1, S, D] - seq_len is dim 2
            seq_len = emb_shape[2]
        elif len(emb_shape) == 3:
            # Shape is [B, S, D] - seq_len is dim 1, need to unsqueeze
            seq_len = emb_shape[1]
            inputs_embeds = ttnn.unsqueeze(inputs_embeds, dim=1)  # -> [B, 1, S, D]
        else:
            raise ValueError(f"Unexpected inputs_embeds shape: {emb_shape}. Expected [B,S,D] or [1,1,S,D]")

        if getattr(self, "_debug_trace", False):
            print(f"DEBUG: inputs_embeds shape={emb_shape}, computed seq_len={seq_len}")

        CHECKPOINTS.checkpoint("start_PREFILL")
        padding = get_padded_prefill_len(seq_len) - inputs_embeds.shape[2]
        padded_seq_len = seq_len + padding
        if padding != 0:
            inputs_embeds = ttnn.pad(inputs_embeds, [(0, 0), (0, 0), (0, padding), (0, 0)], 0)

        tt_rot_mats_prefill_global = [
            self.model[0].rope_setup.cos_matrix[:, :, : inputs_embeds.shape[2], :],
            self.model[0].rope_setup.sin_matrix[:, :, : inputs_embeds.shape[2], :],
        ]
        page_table_user = self._get_prefill_user_page_table(self.page_table, self.tt_kv_cache[0], seq_len)
        tt_page_table = ttnn.from_torch(
            page_table_user,
            device=inputs_embeds.device(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(inputs_embeds.device()),
        )
        # DEBUG: Check inputs_embeds before LLM forward
        if getattr(self, "_debug_trace", False):
            _mesh_device = self.generator_args_config["mesh_device"]
            _emb_in = ttnn_to_torch_safe(inputs_embeds, _mesh_device)
            print(f"\n=== DEBUG: LLM Forward Input ===")
            print(
                f"DEBUG inputs_embeds before LLM: shape={_emb_in.shape}, mean={_emb_in.mean():.6f}, std={_emb_in.std():.6f}"
            )
            print(f"DEBUG inputs_embeds checksum: {_emb_in.sum().item():.2f} (MUST differ per image!)")
            # Check the ACTUAL last token position (seq_len-1), not the padding (-1)
            actual_last_pos = seq_len - 1
            print(
                f"DEBUG ACTUAL last position ({actual_last_pos}): mean={_emb_in[0, 0, actual_last_pos, :].mean():.6f}, std={_emb_in[0, 0, actual_last_pos, :].std():.6f}"
            )
            # Also check text region checksum specifically (positions 257 to seq_len-1)
            text_region = _emb_in[0, 0, 257:seq_len, :]
            print(
                f"DEBUG text region ({257} to {seq_len-1}): checksum={text_region.sum().item():.4f}, shape={text_region.shape}"
            )

        tt_logits = self.model[0].forward(
            inputs_embeds,
            None,
            rot_mats_global=tt_rot_mats_prefill_global,
            mode="prefill",
            page_table=tt_page_table,
            kv_cache=self.tt_kv_cache[0],
            get_last_token=((seq_len - 1) // 32) * 32,
        )

        last_token_idx = seq_len - 1

        # Move tensor from device to host before processing
        tt_logits = ttnn.from_device(tt_logits)

        # DEBUG: Check logits after LLM forward
        if getattr(self, "_debug_trace", False):
            _logits_debug = self.model[0].concat_host_output(tt_logits)
            print(f"\n=== DEBUG: LLM Forward Output ===")
            print(f"DEBUG raw logits shape: {_logits_debug.shape}")
            print(f"DEBUG logits checksum: {_logits_debug.sum().item():.2f} (should differ if inputs differ)")
            # Check action token logits specifically (31744-32063)
            action_logits = _logits_debug[0, 0, last_token_idx % 32, 31744:32000]
            print(
                f"DEBUG action token logits (31744-32000): mean={action_logits.mean():.4f}, std={action_logits.std():.4f}, max={action_logits.max():.4f}"
            )

        # Since we give unpadded_seq_len, only the tile containing the last token is returned
        tile_start = ((seq_len - 1) // 32) * 32
        idx_in_tile = last_token_idx % 32
        absolute_pos = tile_start + idx_in_tile

        # DEBUG: Verify we're reading from the correct position
        if getattr(self, "_debug_trace", False):
            print(f"\n=== CRITICAL POSITION CHECK ===")
            print(f"seq_len={seq_len}, last_token_idx={last_token_idx}")
            print(f"tile_start={tile_start}, idx_in_tile={idx_in_tile}")
            print(f"absolute_pos={absolute_pos} (MUST equal seq_len-1={seq_len-1})")
            assert absolute_pos == seq_len - 1, f"POSITION BUG! absolute_pos={absolute_pos} != seq_len-1={seq_len-1}"

        output_logits = self.model[0].process_output_prefill(tt_logits, last_token_idx=idx_in_tile)
        prefilled_token = torch.argmax(output_logits.cpu(), dim=-1).unsqueeze(0)

        # DEBUG: Print first token to trace if prefill is working correctly
        if getattr(self, "_debug_trace", False):
            logits_cpu = output_logits.cpu().float()
            top10_vals, top10_toks = torch.topk(logits_cpu.flatten(), 10)
            gap = top10_vals[0] - top10_vals[1]
            print(f"\n=== TT PREFILL TOP-10 ===")
            for i, (tok, val) in enumerate(zip(top10_toks.tolist(), top10_vals.tolist())):
                print(f"  {i+1}. token={tok}, logit={val:.4f}")
            print(f"TT PREFILL ARGMAX: {top10_toks[0].item()}")
            print(f"GAP (top1-top2): {gap:.4f}")
            print(
                f"DEBUG __call__: prefilled_token={prefilled_token}, shape={prefilled_token.shape}, dtype={prefilled_token.dtype}"
            )

        # Initial positions - use int32 as TT kernels expect int32 tokens/positions
        current_pos = torch.tensor([seq_len], dtype=torch.int32)
        # Convert prefilled_token to int32 for TT compatibility
        out_tok = prefilled_token.to(torch.int32).view(1)  # Ensure shape [1] and int32
        output_toks = []
        CHECKPOINTS.checkpoint("end_PREFILL")
        decode_tokens = [prefilled_token.item()]  # Track all generated tokens for debug

        # DEBUG: Verify KV cache has content after prefill
        if getattr(self, "_debug_trace", False):
            try:
                # Get first layer's KV cache and compute checksum
                kv_layer0 = self.tt_kv_cache[0][0][0]  # First layer, first part
                if hasattr(kv_layer0, "shape"):
                    _mesh_device = self.generator_args_config["mesh_device"]
                    kv_torch = ttnn_to_torch_safe(kv_layer0, _mesh_device)
                    kv_sum = kv_torch.abs().sum().item()
                    kv_mean = kv_torch.mean().item()
                    kv_std = kv_torch.std().item()
                    print(f"DEBUG KV after prefill: sum={kv_sum:.2f}, mean={kv_mean:.6f}, std={kv_std:.6f}")
            except Exception as e:
                print(f"DEBUG KV check failed: {e}")

            # DEBUG: Check LLM embedding weights for token 31872
            try:
                embd = self.model[0].embd
                if hasattr(embd, "weight"):
                    # PyTorch embedding
                    embd_weight = embd.weight
                    token_embd = embd_weight[31872].float()
                    print(
                        f"DEBUG LLM embedding[31872]: mean={token_embd.mean():.6f}, std={token_embd.std():.6f}, first5={token_embd[:5].tolist()}"
                    )
                else:
                    print(f"DEBUG LLM embedding: type={type(embd)}")
            except Exception as e:
                print(f"DEBUG LLM embedding check failed: {e}")

        # ==========================================================================
        # KV CACHE STRUCTURE DIAGNOSTIC
        # Check if prefill and decode are using the same KV cache structure
        # ==========================================================================
        if getattr(self, "_debug_trace", False):
            print(f"\n=== KV CACHE STRUCTURE DIAGNOSTIC ===")
            kv = self.tt_kv_cache
            print(f"type(tt_kv_cache) = {type(kv)}")
            print(f"len(tt_kv_cache) = {len(kv)}")
            print(f"type(tt_kv_cache[0]) = {type(kv[0])}")
            try:
                print(f"len(tt_kv_cache[0]) = {len(kv[0])}")
                if len(kv[0]) > 0:
                    print(f"type(tt_kv_cache[0][0]) = {type(kv[0][0])}")
                    try:
                        print(f"len(tt_kv_cache[0][0]) = {len(kv[0][0])}")
                    except:
                        pass
            except Exception as e:
                print(f"Cannot inspect deeper: {e}")

            # Check what prefill vs decode are passing
            print(f"\nPREFILL uses: kv_cache=self.tt_kv_cache[0] (32-layer list)")
            print(f"DECODE  uses: kv_cache=[self.tt_kv_cache[0]] (wrapped for generator)")
            print(f"Generator accesses [0] → both pass same 32-layer cache object ✓")
            print(f"=== END KV STRUCTURE ===\n")

        # ==========================================================================
        # KV HANDLE A/B TEST: Verify which KV cache form works correctly
        # Test: [self.tt_kv_cache[0]] vs self.tt_kv_cache at same position
        # ==========================================================================
        if getattr(self, "_debug_trace", False):
            print(f"\n=== KV HANDLE A/B TEST at pos={seq_len} ===")
            test_pos = torch.tensor([seq_len], dtype=torch.int32)
            test_tok = torch.tensor([31872], dtype=torch.int32)

            # Form A: [self.tt_kv_cache[0]] - wrapped for generator
            try:
                logitsA_out = self.generator.decode_forward_text(
                    test_tok,
                    test_pos,
                    page_table=page_table_user,
                    kv_cache=[self.tt_kv_cache[0]],  # Form A
                    sampling_params=None,
                    enable_trace=False,
                )
                logitsA = logitsA_out[0] if isinstance(logitsA_out, tuple) else logitsA_out
                logitsA_cpu = logitsA.cpu().float().flatten()
                top5A_vals, top5A_toks = torch.topk(logitsA_cpu, 5)
                print(
                    f"Form A [tt_kv_cache[0]]: top5={top5A_toks.tolist()}, argmax={top5A_toks[0].item()}, sum={logitsA_cpu.sum().item():.2f}"
                )
            except Exception as e:
                print(f"Form A failed: {e}")
                logitsA_cpu = None

            # Form B: self.tt_kv_cache - full list
            try:
                test_pos_B = torch.tensor([seq_len], dtype=torch.int32)
                logitsB_out = self.generator.decode_forward_text(
                    test_tok,
                    test_pos_B,
                    page_table=page_table_user,
                    kv_cache=self.tt_kv_cache,  # Form B
                    sampling_params=None,
                    enable_trace=False,
                )
                logitsB = logitsB_out[0] if isinstance(logitsB_out, tuple) else logitsB_out
                logitsB_cpu = logitsB.cpu().float().flatten()
                top5B_vals, top5B_toks = torch.topk(logitsB_cpu, 5)
                print(
                    f"Form B [tt_kv_cache]:    top5={top5B_toks.tolist()}, argmax={top5B_toks[0].item()}, sum={logitsB_cpu.sum().item():.2f}"
                )
            except Exception as e:
                print(f"Form B failed: {e}")
                logitsB_cpu = None

            # Compare
            if logitsA_cpu is not None and logitsB_cpu is not None:
                diff = (logitsA_cpu - logitsB_cpu).abs()
                print(f"A vs B: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
                if diff.max().item() < 0.001:
                    print(f"✅ Both forms produce identical results")
                else:
                    print(f"⚠️ Forms differ! Use the one that matches prefill behavior")
            print(f"=== END KV A/B TEST ===\n")

        # FIX: Use cropped page_table_user for decode (same as prefill)
        # This ensures decode reads from the correct KV cache pages
        for i in range(self.num_actions):
            # DEBUG: Print what we're feeding to decode
            if getattr(self, "_debug_trace", False) and i == 0:
                print(f"\n=== DECODE LOOP STARTING ===")
                print(f"DEBUG prefill_seq_len={seq_len}, starting_pos={current_pos.item()}")
                print(f"DEBUG num_actions={self.num_actions}")
                print(f"DEBUG page_table_user shape={page_table_user.shape}")
                print(f"DEBUG kv_cache type={type(self.tt_kv_cache)}, len={len(self.tt_kv_cache)}")
            if getattr(self, "_debug_trace", False):
                print(f"DEBUG decode[{i}] INPUT: out_tok={out_tok.item()}, pos={current_pos.item()}")

            # Run decode forward
            # CRITICAL: Use self.tt_kv_cache[0] to match what prefill wrote to!
            # Prefill: model[0].forward(..., kv_cache=self.tt_kv_cache[0], ...)
            # Decode must read from the SAME KV cache object
            CHECKPOINTS.checkpoint("start_LLM_DECODE")
            decode_output = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                page_table=page_table_user,  # FIX: Use cropped page table, not self.page_table
                kv_cache=[self.tt_kv_cache[0]],  # FIX: Wrap in list if generator expects list, but use [0] content
                sampling_params=None,
                enable_trace=False,
            )
            # decode_forward_text returns (logits, log_probs) tuple
            logits = decode_output[0] if isinstance(decode_output, tuple) else decode_output
            CHECKPOINTS.checkpoint("end_LLM_DECODE")

            # DEBUG: Print raw decode output
            if getattr(self, "_debug_trace", False):
                print(
                    f"DEBUG decode[{i}] OUTPUT: logits type={type(logits)}, shape={logits.shape if hasattr(logits, 'shape') else 'N/A'}"
                )

            current_pos += 1
            output_toks.append(logits)

            # FIX: Update out_tok for next decode step (autoregressive!)
            # This is CRITICAL - each decode step must feed the previous token
            logits_cpu = logits.cpu().float()
            # logits shape is [1, 1, vocab] - take last position and argmax
            next_token = torch.argmax(logits_cpu[:, -1, :], dim=-1).to(torch.int32)  # [1] shape, int32

            # DEBUG: Show what we're updating out_tok to
            if getattr(self, "_debug_trace", False):
                print(
                    f"DEBUG decode[{i}] NEXT: argmax={next_token.item()}, shape={next_token.shape}, dtype={next_token.dtype}, updating out_tok for next step"
                )

            out_tok = next_token  # Keep [1] shape and int32 dtype

            # Debug trace
            if getattr(self, "_debug_trace", False):
                decode_tokens.append(next_token.item())
                # Top-5 analysis for each decode step with GAP
                top5_vals, top5_toks = torch.topk(logits_cpu.flatten(), 5)
                gap = top5_vals[0] - top5_vals[1]
                # Check if logits are changing at all
                logits_hash = logits_cpu.sum().item()
                print(
                    f"DEBUG decode[{i}] SUMMARY: pos={current_pos.item()}, top1={top5_toks[0].item()} ({top5_vals[0]:.4f}), top2={top5_toks[1].item()} ({top5_vals[1]:.4f}), GAP={gap:.4f}, logits_sum={logits_hash:.2f}"
                )

        # DEBUG: Print all generated tokens
        if getattr(self, "_debug_trace", False):
            print(f"DEBUG __call__: all_tokens={decode_tokens}")

        # Save tokens for external access
        self._last_generated_tokens = decode_tokens

        return output_toks


class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "llama2-7b-pure",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Vision backbone `{vision_backbone_id}` not in {VALID_VISION_BACKBONES = }")

        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"LLM backbone `{llm_backbone_id}` not in {VALID_LLM_BACKBONES = }")

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )

        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[self.vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](**text_config)
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAConfig(PrismaticConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins
        super().__init__(**kwargs)


# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        # get_intermediate_layers returns a tuple or list, unpack the first element
        if isinstance(result, (tuple, list)) and len(result) >= 1:
            return result[0]
        return result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


def ttnn_featurizer(embedding, encoder, pixel):
    embd = embedding(pixel)
    tiled_hidden_states = ttnn.to_layout(embd, layout=ttnn.TILE_LAYOUT)
    encoder = encoder(tiled_hidden_states)
    return encoder


# =============================================================================
# NEW: Proven Vision Encoder Implementation (0.99 PCC verified)
# =============================================================================


def _upchannel_attn_weight_bias_new(qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads):
    """Pad attention weights so head_dim is multiple of 32 (for SigLIP: 72 -> 96)."""
    qkv = 3
    head_dim = qkv_weight.shape[0] // (num_heads * qkv)
    is_padding_required = head_dim % 32 != 0

    if is_padding_required:
        padded_head_dim = int(np.ceil(head_dim / 32) * 32)
        padded_val = padded_head_dim * num_heads * qkv

        new_qkv_weight = torch.zeros((padded_val, qkv_weight.shape[1]), dtype=qkv_weight.dtype)
        new_qkv_weight = new_qkv_weight.reshape(qkv, num_heads, padded_head_dim, qkv_weight.shape[1])
        reshaped_qkv_weight = qkv_weight.reshape(qkv, num_heads, head_dim, qkv_weight.shape[1])
        new_qkv_weight[:, :, :head_dim, :] = reshaped_qkv_weight
        new_qkv_weight = new_qkv_weight.reshape(padded_val, qkv_weight.shape[1])

        new_qkv_bias = torch.zeros((padded_val,), dtype=qkv_bias.dtype)
        new_qkv_bias = new_qkv_bias.reshape(qkv, num_heads, padded_head_dim)
        reshaped_qkv_bias = qkv_bias.reshape(qkv, num_heads, head_dim)
        new_qkv_bias[:, :, :head_dim] = reshaped_qkv_bias
        new_qkv_bias = new_qkv_bias.reshape(-1)

        new_proj_weight = torch.zeros((proj_weight.shape[0], padded_head_dim * num_heads), dtype=proj_weight.dtype)
        new_proj_weight = new_proj_weight.reshape(proj_weight.shape[0], num_heads, padded_head_dim)
        reshaped_proj = proj_weight.reshape(proj_weight.shape[0], num_heads, head_dim)
        new_proj_weight[:, :, :head_dim] = reshaped_proj
        new_proj_weight = new_proj_weight.reshape(proj_weight.shape[0], padded_head_dim * num_heads)

        return new_qkv_weight, new_qkv_bias, new_proj_weight, proj_bias

    return qkv_weight, qkv_bias, proj_weight, proj_bias


def _preprocess_patch_embed_new(weight, bias, device, mesh_mapper=None):
    """Preprocess patch embedding conv weights for TTNN."""
    out_channels, in_channels, _, _ = weight.shape
    pad_value = 4 - in_channels
    preprocessed = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
    preprocessed = preprocessed.permute(2, 3, 1, 0)  # [H, W, C, hidden]
    preprocessed = preprocessed.reshape(-1, out_channels)  # [H*W*C, hidden]

    return (
        ttnn.from_torch(
            preprocessed.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        ),
        ttnn.from_torch(
            bias.unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        ),
    )


def _dinov2_patch_embeddings_new(pixel_values, weight, bias, pos_embed, cls_token, reg_token, patch_size=14):
    """DINOv2 patch embedding with CLS + 4 register tokens."""
    batch_size = pixel_values.shape[0]

    pixel_values = ttnn.reshape(pixel_values, (batch_size, 16, patch_size, 16, patch_size, 4))
    pixel_values = ttnn.permute(pixel_values, (0, 1, 3, 2, 4, 5))
    pixel_values = ttnn.reshape(pixel_values, (batch_size, 256, patch_size * patch_size * 4))
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    patch_embeds = ttnn.linear(
        pixel_values, weight, bias=bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    patch_embeds = ttnn.add(patch_embeds, pos_embed, dtype=ttnn.bfloat16)

    cls_reg = ttnn.concat([cls_token, reg_token], dim=1)
    cls_reg = ttnn.typecast(cls_reg, dtype=ttnn.bfloat16)

    embeddings = ttnn.concat([cls_reg, patch_embeds], dim=1)
    return embeddings


def _dinov2_attention_new(hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias, ls_scale, num_heads=16):
    """DINOv2 attention with LayerScale."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    head_dim = hidden_dim // num_heads
    scale = 1.0 / (head_dim**0.5)

    qkv = ttnn.linear(
        hidden_states, qkv_weight, bias=qkv_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=num_heads
    )
    ttnn.deallocate(qkv)

    attn_scores = ttnn.matmul(query, key, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attn_scores = ttnn.mul(attn_scores, scale)
    attn_probs = ttnn.softmax_in_place(attn_scores)

    context = ttnn.matmul(attn_probs, value, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)

    context = ttnn.transformer.concatenate_heads(context, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.linear(context, proj_weight, bias=proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(context)
    output = ttnn.mul(output, ls_scale)

    return output


def _dinov2_mlp_new(hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, ls_scale):
    """DINOv2 MLP with LayerScale."""
    output = ttnn.linear(
        hidden_states, fc1_weight, bias=fc1_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    output = ttnn.gelu(output, fast_and_approximate_mode=True)
    output = ttnn.linear(output, fc2_weight, bias=fc2_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    output = ttnn.mul(output, ls_scale)
    return output


def _dinov2_block_new(hidden_states, params):
    """Single DINOv2 transformer block."""
    normed = ttnn.layer_norm(
        hidden_states, weight=params["norm1_weight"], bias=params["norm1_bias"], memory_config=ttnn.L1_MEMORY_CONFIG
    )

    attn_out = _dinov2_attention_new(
        normed,
        params["qkv_weight"],
        params["qkv_bias"],
        params["proj_weight"],
        params["proj_bias"],
        params["ls1_scale"],
    )
    hidden_states = ttnn.add(hidden_states, attn_out)

    normed = ttnn.layer_norm(
        hidden_states, weight=params["norm2_weight"], bias=params["norm2_bias"], memory_config=ttnn.L1_MEMORY_CONFIG
    )

    mlp_out = _dinov2_mlp_new(
        normed, params["fc1_weight"], params["fc1_bias"], params["fc2_weight"], params["fc2_bias"], params["ls2_scale"]
    )
    hidden_states = ttnn.add(hidden_states, mlp_out)

    return hidden_states


def _siglip_patch_embeddings_new(pixel_values, weight, bias, pos_embed, patch_size=14):
    """SigLIP patch embedding (no CLS token)."""
    batch_size = pixel_values.shape[0]

    pixel_values = ttnn.reshape(pixel_values, (batch_size, 16, patch_size, 16, patch_size, 4))
    pixel_values = ttnn.permute(pixel_values, (0, 1, 3, 2, 4, 5))
    pixel_values = ttnn.reshape(pixel_values, (batch_size, 256, patch_size * patch_size * 4))
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    patch_embeds = ttnn.linear(
        pixel_values, weight, bias=bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    patch_embeds = ttnn.add(patch_embeds, pos_embed, dtype=ttnn.bfloat16)

    return patch_embeds


def _siglip_attention_new(
    hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads=16, padded_head_dim=96
):
    """SigLIP attention with padded head_dim."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    scale = 1.0 / (padded_head_dim**0.5)

    qkv = ttnn.linear(
        hidden_states, qkv_weight, bias=qkv_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=num_heads
    )
    ttnn.deallocate(qkv)

    attn_scores = ttnn.matmul(query, key, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attn_scores = ttnn.mul(attn_scores, scale)
    attn_probs = ttnn.softmax_in_place(attn_scores)

    context = ttnn.matmul(attn_probs, value, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)

    context = ttnn.transformer.concatenate_heads(context, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.linear(context, proj_weight, bias=proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(context)

    return output


def _siglip_mlp_new(hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    """SigLIP MLP (no LayerScale)."""
    output = ttnn.linear(
        hidden_states, fc1_weight, bias=fc1_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    output = ttnn.gelu(output, fast_and_approximate_mode=True)
    output = ttnn.linear(output, fc2_weight, bias=fc2_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    return output


def _siglip_block_new(hidden_states, params):
    """Single SigLIP transformer block."""
    normed = ttnn.layer_norm(
        hidden_states, weight=params["norm1_weight"], bias=params["norm1_bias"], memory_config=ttnn.L1_MEMORY_CONFIG
    )

    attn_out = _siglip_attention_new(
        normed, params["qkv_weight"], params["qkv_bias"], params["proj_weight"], params["proj_bias"]
    )
    hidden_states = ttnn.add(hidden_states, attn_out)

    normed = ttnn.layer_norm(
        hidden_states, weight=params["norm2_weight"], bias=params["norm2_bias"], memory_config=ttnn.L1_MEMORY_CONFIG
    )

    mlp_out = _siglip_mlp_new(
        normed, params["fc1_weight"], params["fc1_bias"], params["fc2_weight"], params["fc2_bias"]
    )
    hidden_states = ttnn.add(hidden_states, mlp_out)

    return hidden_states


class OpenVLAVisionEncoderNew:
    """New proven vision encoder implementation (0.99 PCC verified)."""

    def __init__(self, device, state_dict):
        """Initialize with OpenVLA state dict."""
        self.device = device
        # For multi-device (N300/T3K), use mesh_mapper to replicate weights
        if hasattr(device, "shape") and tuple(device.shape) != (1, 1):
            self.mesh_mapper = ttnn.ReplicateTensorToMesh(device)
            self.is_multi_device = True
            mesh_shape = tuple(device.shape)
        else:
            self.mesh_mapper = None
            self.is_multi_device = False
            mesh_shape = (1, 1)

        # Debug trace for N300/multi-device vision encoder setup
        print(f"🔧 OpenVLA Vision Encoder Config:")
        print(f"   - is_multi_device={self.is_multi_device}, mesh_shape={mesh_shape}")
        print(f"   - mesh_mapper={'ReplicateTensorToMesh' if self.mesh_mapper else 'None'}")

        self._preprocess_weights(state_dict)

    def _to_device(self, tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        """Helper to load tensor to device with proper mesh_mapper for multi-device."""
        return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=self.device, mesh_mapper=self.mesh_mapper)

    def _preprocess_weights(self, state_dict):
        """Preprocess all vision encoder weights for TTNN."""
        print("   [NEW] Preprocessing DINOv2 weights...")

        # DINOv2 patch embedding
        dinov2_patch_w = state_dict["vision_backbone.featurizer.patch_embed.proj.weight"]
        dinov2_patch_b = state_dict["vision_backbone.featurizer.patch_embed.proj.bias"]
        self.dinov2_patch_weight, self.dinov2_patch_bias = _preprocess_patch_embed_new(
            dinov2_patch_w, dinov2_patch_b, self.device, self.mesh_mapper
        )

        # Position embedding [1, 256, 1024]
        pos_embed = state_dict["vision_backbone.featurizer.pos_embed"]
        self.dinov2_pos_embed = self._to_device(pos_embed.to(torch.bfloat16))
        self.dinov2_cls_token = self._to_device(state_dict["vision_backbone.featurizer.cls_token"].to(torch.bfloat16))
        self.dinov2_reg_token = self._to_device(state_dict["vision_backbone.featurizer.reg_token"].to(torch.bfloat16))

        # Final layer norm
        self.dinov2_final_norm_w = self._to_device(
            state_dict["vision_backbone.featurizer.norm.weight"].unsqueeze(0).to(torch.bfloat16)
        )
        self.dinov2_final_norm_b = self._to_device(
            state_dict["vision_backbone.featurizer.norm.bias"].unsqueeze(0).to(torch.bfloat16)
        )

        # DINOv2 blocks
        self.dinov2_blocks = []
        for i in range(24):
            prefix = f"vision_backbone.featurizer.blocks.{i}"
            qkv_w = state_dict[f"{prefix}.attn.qkv.weight"]
            qkv_b = state_dict[f"{prefix}.attn.qkv.bias"]

            block_params = {
                "norm1_weight": self._to_device(state_dict[f"{prefix}.norm1.weight"].unsqueeze(0).to(torch.bfloat16)),
                "norm1_bias": self._to_device(state_dict[f"{prefix}.norm1.bias"].unsqueeze(0).to(torch.bfloat16)),
                "qkv_weight": self._to_device(qkv_w.t().contiguous().to(torch.bfloat16)),
                "qkv_bias": self._to_device(qkv_b.unsqueeze(0).to(torch.bfloat16)),
                "proj_weight": self._to_device(
                    state_dict[f"{prefix}.attn.proj.weight"].t().contiguous().to(torch.bfloat16)
                ),
                "proj_bias": self._to_device(state_dict[f"{prefix}.attn.proj.bias"].unsqueeze(0).to(torch.bfloat16)),
                "ls1_scale": self._to_device(
                    state_dict[f"{prefix}.ls1.scale_factor"].reshape(1, 1, -1).to(torch.bfloat16)
                ),
                "norm2_weight": self._to_device(state_dict[f"{prefix}.norm2.weight"].unsqueeze(0).to(torch.bfloat16)),
                "norm2_bias": self._to_device(state_dict[f"{prefix}.norm2.bias"].unsqueeze(0).to(torch.bfloat16)),
                "fc1_weight": self._to_device(
                    state_dict[f"{prefix}.mlp.fc1.weight"].t().contiguous().to(torch.bfloat16)
                ),
                "fc1_bias": self._to_device(state_dict[f"{prefix}.mlp.fc1.bias"].unsqueeze(0).to(torch.bfloat16)),
                "fc2_weight": self._to_device(
                    state_dict[f"{prefix}.mlp.fc2.weight"].t().contiguous().to(torch.bfloat16)
                ),
                "fc2_bias": self._to_device(state_dict[f"{prefix}.mlp.fc2.bias"].unsqueeze(0).to(torch.bfloat16)),
                "ls2_scale": self._to_device(
                    state_dict[f"{prefix}.ls2.scale_factor"].reshape(1, 1, -1).to(torch.bfloat16)
                ),
            }
            self.dinov2_blocks.append(block_params)

        print("   [NEW] Preprocessing SigLIP weights...")

        # SigLIP patch embedding
        siglip_patch_w = state_dict["vision_backbone.fused_featurizer.patch_embed.proj.weight"]
        siglip_patch_b = state_dict["vision_backbone.fused_featurizer.patch_embed.proj.bias"]
        self.siglip_patch_weight, self.siglip_patch_bias = _preprocess_patch_embed_new(
            siglip_patch_w, siglip_patch_b, self.device, self.mesh_mapper
        )

        # Position embedding [1, 256, 1152]
        siglip_pos_embed = state_dict["vision_backbone.fused_featurizer.pos_embed"]
        self.siglip_pos_embed = self._to_device(siglip_pos_embed.to(torch.bfloat16))

        # Final layer norm
        self.siglip_final_norm_w = self._to_device(
            state_dict["vision_backbone.fused_featurizer.norm.weight"].unsqueeze(0).to(torch.bfloat16)
        )
        self.siglip_final_norm_b = self._to_device(
            state_dict["vision_backbone.fused_featurizer.norm.bias"].unsqueeze(0).to(torch.bfloat16)
        )

        # SigLIP blocks (with head_dim padding)
        self.siglip_blocks = []
        for i in range(27):
            prefix = f"vision_backbone.fused_featurizer.blocks.{i}"

            qkv_w = state_dict[f"{prefix}.attn.qkv.weight"]
            qkv_b = state_dict[f"{prefix}.attn.qkv.bias"]
            proj_w = state_dict[f"{prefix}.attn.proj.weight"]
            proj_b = state_dict[f"{prefix}.attn.proj.bias"]

            # Pad to head_dim=96
            qkv_w, qkv_b, proj_w, proj_b = _upchannel_attn_weight_bias_new(qkv_w, qkv_b, proj_w, proj_b, num_heads=16)

            block_params = {
                "norm1_weight": self._to_device(state_dict[f"{prefix}.norm1.weight"].unsqueeze(0).to(torch.bfloat16)),
                "norm1_bias": self._to_device(state_dict[f"{prefix}.norm1.bias"].unsqueeze(0).to(torch.bfloat16)),
                "qkv_weight": self._to_device(qkv_w.t().contiguous().to(torch.bfloat16)),
                "qkv_bias": self._to_device(qkv_b.unsqueeze(0).to(torch.bfloat16)),
                "proj_weight": self._to_device(proj_w.t().contiguous().to(torch.bfloat16)),
                "proj_bias": self._to_device(proj_b.unsqueeze(0).to(torch.bfloat16)),
                "norm2_weight": self._to_device(state_dict[f"{prefix}.norm2.weight"].unsqueeze(0).to(torch.bfloat16)),
                "norm2_bias": self._to_device(state_dict[f"{prefix}.norm2.bias"].unsqueeze(0).to(torch.bfloat16)),
                "fc1_weight": self._to_device(
                    state_dict[f"{prefix}.mlp.fc1.weight"].t().contiguous().to(torch.bfloat16)
                ),
                "fc1_bias": self._to_device(state_dict[f"{prefix}.mlp.fc1.bias"].unsqueeze(0).to(torch.bfloat16)),
                "fc2_weight": self._to_device(
                    state_dict[f"{prefix}.mlp.fc2.weight"].t().contiguous().to(torch.bfloat16)
                ),
                "fc2_bias": self._to_device(state_dict[f"{prefix}.mlp.fc2.bias"].unsqueeze(0).to(torch.bfloat16)),
            }
            self.siglip_blocks.append(block_params)

        print("   [NEW] Weight preprocessing complete!")

    def forward(self, pixel_values):
        """
        Forward pass.

        Args:
            pixel_values: tuple of (dinov2_input, siglip_input) TTNN tensors
                          Each is [B, H, W, 4] in NHWC format with 4 padded channels
                          For multi-device: inputs should be mesh tensors (replicated)

        Returns:
            [batch, 256, 2176] - concatenated DINOv2 + SigLIP features
            For multi-device: returns mesh tensor (use ttnn_to_torch_safe to extract)
        """
        dinov2_in, siglip_in = pixel_values

        # NOTE: For multi-device (N300/T3K), both inputs and weights are mesh tensors
        # (replicated via ReplicateTensorToMesh). The computation runs on all devices
        # identically, and we extract from first device at output time using ttnn_to_torch_safe.

        # DINOv2 forward (23 layers - skip last to match get_intermediate_layers(n={22}))
        hidden_states = _dinov2_patch_embeddings_new(
            dinov2_in,
            self.dinov2_patch_weight,
            self.dinov2_patch_bias,
            self.dinov2_pos_embed,
            self.dinov2_cls_token,
            self.dinov2_reg_token,
        )
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        # Process only first 23 layers (0-22), skip last layer (23) to match PyTorch
        for i, block_params in enumerate(self.dinov2_blocks):
            if i >= 23:  # Skip last layer
                break
            hidden_states = _dinov2_block_new(hidden_states, block_params)

        # NO final layer norm - PyTorch get_intermediate_layers doesn't apply it

        # Skip CLS + 4 REG tokens
        dinov2_features = hidden_states[:, 5:, :]  # [B, 256, 1024]

        # SigLIP forward (26 layers - skip last to match get_intermediate_layers(n={25}))
        hidden_states = _siglip_patch_embeddings_new(
            siglip_in, self.siglip_patch_weight, self.siglip_patch_bias, self.siglip_pos_embed
        )
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        # Process only first 26 layers (0-25), skip last layer (26) to match PyTorch
        for i, block_params in enumerate(self.siglip_blocks):
            if i >= 26:  # Skip last layer
                break
            hidden_states = _siglip_block_new(hidden_states, block_params)

        # NO final layer norm - PyTorch get_intermediate_layers doesn't apply it

        siglip_features = hidden_states  # [B, 256, 1152]

        # Concatenate (ensure same dtype)
        dinov2_features = ttnn.typecast(dinov2_features, dtype=ttnn.bfloat16)
        siglip_features = ttnn.typecast(siglip_features, dtype=ttnn.bfloat16)
        output = ttnn.concat([dinov2_features, siglip_features], dim=2)  # [B, 256, 2176]

        return output

    def __call__(self, pixel_values):
        return self.forward(pixel_values)


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
        ttnn_device: Optional[Any] = None,
        local_state_dict=None,
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.ttnn_device = ttnn_device
        self.local_state_dict = local_state_dict

        # NEW: Use proven vision encoder when we have local_state_dict and ttnn_device
        # Set USE_OLD_ENCODER=1 to force using the OLD encoder for comparison
        force_old = os.environ.get("USE_OLD_ENCODER", "0") == "1"
        self.use_new_encoder = (
            local_state_dict is not None and ttnn_device is not None and use_fused_vision_backbone and not force_old
        )
        self.new_encoder = None

        if force_old:
            print("[OLD ENCODER] Forced to use OLD encoder via USE_OLD_ENCODER=1")

        if self.use_new_encoder:
            print("[NEW ENCODER] Using OpenVLAVisionEncoderNew (0.99 PCC verified)")
            self.new_encoder = OpenVLAVisionEncoderNew(ttnn_device, local_state_dict)

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )

        # Match HuggingFace: use get_intermediate_layers to return SECOND-TO-LAST layer
        # This is what the original OpenVLA model uses - no final norm, 2nd-to-last layer
        def unpack_tuple_local(fn):
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                return result[0] if isinstance(result, (tuple, list)) else result

            return wrapper

        self.featurizer.forward = unpack_tuple_local(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)
        if self.local_state_dict is not None:
            featurizer_state_dict = {
                k.replace("vision_backbone.featurizer.", ""): v
                for k, v in self.local_state_dict.items()
                if k.startswith("vision_backbone.featurizer.")
            }
            self.featurizer.load_state_dict(featurizer_state_dict, strict=True)
            print("Loaded local state dict into PrismaticVisionBackbone.featurizer")
        self.embed_dim = self.featurizer.embed_dim
        if self.ttnn_device is not None:
            CHECKPOINTS.checkpoint("start_DINOINIT")
            self.ttnn_featurizer = ttnn_optimized_vit_highres.dinov2_encoder(self.featurizer, self.ttnn_device)
            CHECKPOINTS.checkpoint("end_DINOINIT")

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )

            # Match HuggingFace: use get_intermediate_layers to return SECOND-TO-LAST layer
            def unpack_tuple_local(fn):
                def wrapper(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    return result[0] if isinstance(result, (tuple, list)) else result

                return wrapper

            self.fused_featurizer.forward = unpack_tuple_local(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim
            if self.local_state_dict is not None:
                fused_featurizer_state_dict = {
                    k.replace("vision_backbone.fused_featurizer.", ""): v
                    for k, v in self.local_state_dict.items()
                    if k.startswith("vision_backbone.fused_featurizer.")
                }
                self.fused_featurizer.load_state_dict(fused_featurizer_state_dict, strict=True)
                print("Loaded local state dict into PrismaticVisionBackbone.fused_featurizer")
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)
            if self.ttnn_device is not None:
                CHECKPOINTS.checkpoint("start_SIGLIPINIT")
                self.featurize_parameters_2 = preprocess_model_parameters(
                    initialize_model=lambda: self.fused_featurizer.to(torch.bfloat16),
                    device=self.ttnn_device,
                    custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor_siglip,
                )

                self.head_masks_2 = [
                    ttnn.from_torch(
                        torch.zeros(1, 1, 1, 1, dtype=torch.float32),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.ttnn_device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    for _ in self.featurize_parameters_2.blocks
                ]
                # Match HuggingFace: run up to 2nd-to-last layer, NO final norm
                # This matches get_intermediate_layers(n={total_blocks - 2})
                siglip_total_blocks = len(self.fused_featurizer.blocks)
                siglip_layer_end = siglip_total_blocks - 1  # Process all but last (2nd-to-last output)

                def siglip_forward_no_norm(x2):
                    output = ttnn_featurizer(
                        embedding=lambda x: ttnn_optimized_vit_highres.siglip_patch_embeddings(
                            x,
                            parameters=self.featurize_parameters_2.patch_embed.patch_embeddings,
                        ),
                        encoder=lambda x: ttnn_optimized_vit_highres.siglip_encoder(
                            x,
                            self.head_masks_2,
                            parameters=self.featurize_parameters_2.blocks,
                            layer_end_index=siglip_layer_end,  # Stop at 2nd-to-last layer
                        ),
                        pixel=x2,
                    )
                    # NOTE: Do NOT apply final layer norm - HF get_intermediate_layers returns raw layer output
                    return output

                self.ttnn_fused_featurizer = siglip_forward_no_norm
                CHECKPOINTS.checkpoint("end_SIGLIPINIT")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # NEW: Use proven encoder if available (TTNN path only)
        if self.use_new_encoder and self.new_encoder is not None and not isinstance(pixel_values, torch.Tensor):
            # pixel_values is already (dinov2_input, siglip_input) tuple for TTNN
            CHECKPOINTS.checkpoint("start_DINOFORWARD")
            output = self.new_encoder(pixel_values)
            CHECKPOINTS.checkpoint("end_SIGLIPFORWARD")
            # Debug: print stats (use mesh_composer ONLY for multi-device N300/T3K)
            output_torch = ttnn_to_torch_safe(output, self.ttnn_device)
            print(
                f"DEBUG TT [NEW]: shape={output_torch.shape}, mean={output_torch.mean():.4f}, std={output_torch.std():.4f}"
            )
            return output

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        if self.ttnn_device is None:
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            # PyTorch: get_intermediate_layers already returns ONLY patch tokens (no CLS/registers)
            # DINOv2: [B, 256, 1024] (get_intermediate_layers excludes CLS + 4 registers)
            # SigLIP: [B, 256, 1152] (no CLS token in SigLIP)
            patches = self.featurizer(img)  # No token dropping - get_intermediate_layers handles it
            patches_fused = self.fused_featurizer(img_fused)  # SigLIP has no CLS
            # Debug: print pos_embed and first few values
            print(
                f"DEBUG PT pos_embed: shape={self.featurizer.pos_embed.shape}, mean={self.featurizer.pos_embed.mean():.6f}, first5={self.featurizer.pos_embed[0,0,:5].tolist()}"
            )
            # Debug: print individual encoder stats
            print(
                f"DEBUG PT DINOv2: shape={patches.shape}, mean={patches.float().mean():.4f}, std={patches.float().std():.4f}"
            )
            print(
                f"DEBUG PT SigLIP: shape={patches_fused.shape}, mean={patches_fused.float().mean():.4f}, std={patches_fused.float().std():.4f}"
            )
        else:
            # OLD path (kept for backwards compatibility)
            img, img_fused = pixel_values
            CHECKPOINTS.checkpoint("start_DINOFORWARD")
            patches = self.ttnn_featurizer(img)[:, 5:, :]  # Drop CLS + 4 register tokens for DINOv2
            CHECKPOINTS.checkpoint("end_DINOFORWARD")
            CHECKPOINTS.checkpoint("start_SIGLIPFORWARD")
            # Note: SigLIP encoder already returns patch tokens without CLS (verified by shape match)
            patches_fused = self.ttnn_fused_featurizer(img_fused)
            CHECKPOINTS.checkpoint("end_SIGLIPFORWARD")
            # Debug: print individual encoder stats
            patches_torch = ttnn_to_torch_safe(patches, self.ttnn_device)
            patches_fused_torch = ttnn_to_torch_safe(patches_fused, self.ttnn_device)
            print(
                f"DEBUG TT DINOv2 [OLD]: shape={patches_torch.shape}, mean={patches_torch.mean():.4f}, std={patches_torch.std():.4f}"
            )
            print(
                f"DEBUG TT SigLIP [OLD]: shape={patches_fused_torch.shape}, mean={patches_fused_torch.mean():.4f}, std={patches_fused_torch.std():.4f}"
            )
        if self.ttnn_device is None:
            return torch.cat([patches, patches_fused], dim=2)
        return ttnn.concat([patches, ttnn.typecast(patches_fused, patches.dtype)], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        vision_dim: int,
        llm_dim: int,
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


class TTNNPrismaticProjector:
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        vision_dim: int,
        llm_dim: int,
        ttnn_device: Optional[Any] = None,
        params=None,
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim
        self.ttnn_device = ttnn_device
        self.params = params
        assert self.ttnn_device is not None, "TTNNPrismaticProjector requires a ttnn_device"
        assert self.use_fused_vision_backbone, "TTNNPrismaticProjector only supports fused vision backbones"
        assert self.params is not None, "TTNNPrismaticProjector requires params for TTNN linear layers"

    def forward(self, img_patches):
        # Keep BFP8 for projector to fit in memory (projector PCC is already 0.99)
        projected_features = ttnn.linear(
            img_patches,
            self.params.fc1.weight,
            bias=self.params.fc1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            activation="gelu",
        )
        projected_features = ttnn.linear(
            projected_features,
            self.params.fc2.weight,
            bias=self.params.fc2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            activation="gelu",
        )
        projected_features = ttnn.linear(
            projected_features,
            self.params.fc3.weight,
            bias=self.params.fc3.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig, ttnn_device=None, local_state_dict=None) -> None:
        super().__init__(config, ttnn_device=ttnn_device, local_state_dict=local_state_dict)

        # Debug trace for N300/multi-device main model setup
        if ttnn_device is not None:
            num_devices = ttnn_device.get_num_devices() if isinstance(ttnn_device, ttnn.MeshDevice) else 1
            mesh_shape = tuple(ttnn_device.shape) if isinstance(ttnn_device, ttnn.MeshDevice) else (1, 1)
            print(f"🔧 OpenVLA Main Model (PrismaticForConditionalGeneration):")
            print(f"   - ttnn_device type: {type(ttnn_device).__name__}")
            print(f"   - num_devices={num_devices}, mesh_shape={mesh_shape}")
            print(f"   - MESH_DEVICE env: {os.environ.get('MESH_DEVICE', 'not set')}")

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16", "1.0.22"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 or == 1.0.22 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        CHECKPOINTS.checkpoint("start_VISIONINIT")
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
            ttnn_device,
            local_state_dict=local_state_dict,
        )
        CHECKPOINTS.checkpoint("end_VISIONINIT")

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )
        if local_state_dict is not None:
            self.projector.load_state_dict(
                {k.replace("projector.", ""): v for k, v in local_state_dict.items() if k.startswith("projector.")},
                strict=True,
            )
        if ttnn_device is not None:
            CHECKPOINTS.checkpoint("start_PROJECTORINIT")
            projector_params = preprocess_model_parameters(
                initialize_model=lambda: self.projector.to(torch.bfloat16),
                device=ttnn_device,
            )
            self.ttnn_projector = TTNNPrismaticProjector(
                config.use_fused_vision_backbone,
                vision_dim=self.vision_backbone.embed_dim,
                llm_dim=config.text_config.hidden_size,
                ttnn_device=ttnn_device,
                params=projector_params,
            )
            CHECKPOINTS.checkpoint("end_PROJECTORINIT")
        # Instantiate LLM Backbone
        if ttnn_device is not None:
            CHECKPOINTS.checkpoint("start_LLama2INIT")
            self.language_model = OpenVLALanguageModel(ttnn_device, local_state_dict=local_state_dict)
            CHECKPOINTS.checkpoint("end_LLama2INIT")
        else:
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config, attn_implementation=config._attn_implementation
            )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.ttnn_device = ttnn_device
        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()
        self.cached_output = (None, None)

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        if self.cached_output[0] is not None:
            token = self.cached_output[0][0]
            cached_projector_features = self.cached_output[1]  # Get cached projector features
            language_model_output = self.cached_output[0][1:]
            if len(language_model_output) == 0:
                self.cached_output = (None, None)
            else:
                self.cached_output = (language_model_output, cached_projector_features)
            if not return_dict:
                if output_projector_features and (cached_projector_features is not None):
                    return token, cached_projector_features
                return token
            return PrismaticCausalLMOutputWithPast(
                loss=None,
                logits=token,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                projector_features=cached_projector_features,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        if (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multi-modal forward!"
            assert labels is None

            # Visual Feature Extraction
            if self.ttnn_device is not None:
                CHECKPOINTS.checkpoint("start_PREPROCESS")
                pixel_values = torch.permute(pixel_values, (0, 2, 3, 1))
                img, img_fused = torch.split(pixel_values, [3, 3], dim=3)
                pixel_values1 = torch.nn.functional.pad(img, (0, 1, 0, 0, 0, 0, 0, 0))
                pixel_values2 = torch.nn.functional.pad(img_fused, (0, 1, 0, 0, 0, 0, 0, 0))
                pixel_values1 = ttnn.from_torch(
                    pixel_values1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ttnn_device
                )
                pixel_values2 = ttnn.from_torch(
                    pixel_values2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ttnn_device
                )
                pixel_values = [pixel_values1, pixel_values2]
                CHECKPOINTS.checkpoint("end_PREPROCESS")
                CHECKPOINTS.checkpoint("start_VISIONFORWARD")
                # DEBUG: Confirm which encoder is being used
                if getattr(self, "_debug_trace", False):
                    encoder_type = "NEW (OpenVLAVisionEncoderNew)" if self.vision_backbone.use_new_encoder else "OLD"
                    print(f"DEBUG: Using {encoder_type} vision encoder")
                ttnn_patch_features = self.vision_backbone(pixel_values)
                CHECKPOINTS.checkpoint("end_VISIONFORWARD")

                # DEBUG: Trace vision features
                if getattr(self, "_debug_trace", False):
                    _debug_vision = ttnn_to_torch_safe(ttnn_patch_features, self.ttnn_device)
                    print(
                        f"DEBUG vision_features: shape={_debug_vision.shape}, mean={_debug_vision.mean():.6f}, std={_debug_vision.std():.6f}"
                    )

                CHECKPOINTS.checkpoint("start_PROJECTORFORWARD")
                projected_patch_embeddings = self.ttnn_projector.forward(ttnn_patch_features)
                projected_patch_embeddings = ttnn.mesh_partition(
                    projected_patch_embeddings,
                    -1,
                )
                CHECKPOINTS.checkpoint("end_PROJECTORFORWARD")

                # DEBUG: Trace projector output
                if getattr(self, "_debug_trace", False):
                    _debug_proj = ttnn_to_torch_safe(projected_patch_embeddings, self.ttnn_device)
                    print(
                        f"DEBUG projector_output: shape={_debug_proj.shape}, mean={_debug_proj.mean():.6f}, std={_debug_proj.std():.6f}"
                    )

                # Save vision + projector output for cross-testing with PyTorch LLM
                if getattr(self, "_save_vision_output", None):
                    save_path = self._save_vision_output
                    vision_torch = ttnn_to_torch_safe(ttnn_patch_features, self.ttnn_device).cpu()
                    proj_torch = ttnn_to_torch_safe(projected_patch_embeddings, self.ttnn_device).cpu()
                    torch.save(
                        {
                            "vision_output": vision_torch,
                            "projector_output": proj_torch,
                            "input_ids": input_ids.cpu() if input_ids is not None else None,
                        },
                        save_path,
                    )
                    print(f"✅ Saved TTNN vision+projector to: {save_path}")
            else:
                patch_features = self.vision_backbone(pixel_values)
                # Projection Logic =>> Update Attention Mask
                projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                if self.ttnn_device is not None:
                    projected_patch_attention_mask = ttnn.from_torch(
                        projected_patch_attention_mask,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.ttnn_device,
                    )

            # Get Input Embeddings (from Language Model Embeddings)
            if self.ttnn_device is not None:
                CHECKPOINTS.checkpoint("start_LLMINPUTEMBEDDINGS")

                # DEBUG: Check input_ids before embedding
                if getattr(self, "_debug_trace", False):
                    print(f"\n=== DEBUG: Text Input Processing ===")
                    print(f"DEBUG input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}")
                    print(f"DEBUG input_ids values: {input_ids[0, :10].tolist()}... (first 10)")
                    print(f"DEBUG input_ids range: min={input_ids.min().item()}, max={input_ids.max().item()}")
                    # Check embedding layer
                    emb_layer = self.get_input_embeddings()
                    if hasattr(emb_layer, "weights"):
                        emb_shape = emb_layer.weights.shape
                        print(f"DEBUG embedding layer weights shape: {emb_shape}")
                    print(f"DEBUG vocab_size from config: {self.config.text_config.vocab_size}")

                # Token IDs must be uint32 and ROW_MAJOR (TTNN embedding requires UINT32 or BFLOAT16)
                ttnn_input_ids = ttnn.from_torch(
                    input_ids.to(torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.ttnn_device,
                )
                input_embeddings = self.get_input_embeddings()(ttnn_input_ids)

                # DEBUG: Check text embeddings
                if getattr(self, "_debug_trace", False):
                    _text_emb = ttnn_to_torch_safe(input_embeddings, self.ttnn_device)
                    print(
                        f"DEBUG text_embeddings: shape={_text_emb.shape}, mean={_text_emb.mean():.6f}, std={_text_emb.std():.6f}"
                    )
                    # Check BOS embedding (should be same for all inputs)
                    print(
                        f"DEBUG BOS embedding (pos 0): mean={_text_emb[0, 0, :].mean():.6f}, std={_text_emb[0, 0, :].std():.6f}"
                    )
                    # Check last text token embedding
                    print(
                        f"DEBUG last text token embedding: mean={_text_emb[0, -1, :].mean():.6f}, std={_text_emb[0, -1, :].std():.6f}"
                    )

                CHECKPOINTS.checkpoint("end_LLMINPUTEMBEDDINGS")
            else:
                input_embeddings = self.get_input_embeddings()(input_ids)

            # ============================================
            # GUARDRAIL: Verify prompt length matches expected format
            # ============================================
            text_len = input_ids.shape[1]
            expected_multimodal_len = 1 + 256 + (text_len - 1)  # BOS + vision + text_without_BOS
            if getattr(self, "_debug_trace", False):
                print(f"\n=== PROMPT LENGTH GUARDRAIL ===")
                print(f"TEXT_LEN={text_len}, TEXT_TAIL={input_ids[0, -5:].tolist()}")
                print(f"Expected multimodal_len={expected_multimodal_len}")
                # Verify 29871 was added (required for OpenVLA training format)
                if input_ids[0, -1].item() == 29871:
                    print(f"✅ Empty token 29871 present at end (correct)")
                else:
                    print(f"⚠️  Empty token 29871 NOT at end - last token is {input_ids[0, -1].item()}")

            if self.ttnn_device is not None:
                CHECKPOINTS.checkpoint("start_VISIONLLMCONCAT")

                # EXPERIMENT: Scale up text embeddings to match visual magnitude
                # Visual std ~0.46, Text std ~0.016, ratio ~29x
                # Set TEXT_EMBED_SCALE env var to experiment (default=1.0, try 10-30)
                text_scale = float(os.environ.get("TEXT_EMBED_SCALE", "1.0"))
                if text_scale != 1.0:
                    print(f"🔧 TEXT_EMBED_SCALE={text_scale} - scaling text embeddings to boost instruction signal")
                    # Scale text embeddings (excluding BOS which stays at position 0)
                    scaled_text_embeddings = ttnn.mul(input_embeddings[:, 1:, :], text_scale)
                    multimodal_embeddings = ttnn.concat(
                        [input_embeddings[:, :1, :], projected_patch_embeddings, scaled_text_embeddings], dim=1
                    )
                else:
                    multimodal_embeddings = ttnn.concat(
                        [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
                    )

                # GUARDRAIL: Verify multimodal length matches expected
                actual_mm_len = multimodal_embeddings.shape[1]
                assert actual_mm_len == expected_multimodal_len, (
                    f"Multimodal length mismatch! Got {actual_mm_len}, expected {expected_multimodal_len}. "
                    f"This indicates a prompt construction bug (check 29871 token handling)."
                )

                multimodal_embeddings = ttnn.unsqueeze(multimodal_embeddings, dim=0)
                multimodal_embeddings = ttnn.to_layout(multimodal_embeddings, layout=ttnn.TILE_LAYOUT)
                CHECKPOINTS.checkpoint("end_VISIONLLMCONCAT")

                # DEBUG: Trace multimodal embeddings structure
                if getattr(self, "_debug_trace", False):
                    _debug_mm = ttnn_to_torch_safe(multimodal_embeddings, self.ttnn_device)
                    print(f"\n=== DEBUG: Multimodal Embeddings Structure ===")
                    print(
                        f"DEBUG multimodal_embeddings: shape={_debug_mm.shape}, mean={_debug_mm.mean():.6f}, std={_debug_mm.std():.6f}"
                    )
                    # Break down the structure: [BOS] + [256 visual] + [text tokens]
                    seq_len = _debug_mm.shape[2]
                    txt_portion = _debug_mm[0, 0, 257:, :]  # Text tokens after vision
                    print(
                        f"DEBUG  Position 0 (BOS): mean={_debug_mm[0, 0, 0, :].mean():.6f}, std={_debug_mm[0, 0, 0, :].std():.6f}"
                    )
                    print(
                        f"DEBUG  Positions 1-256 (Visual): mean={_debug_mm[0, 0, 1:257, :].mean():.6f}, std={_debug_mm[0, 0, 1:257, :].std():.6f}"
                    )
                    if seq_len > 257:
                        print(
                            f"DEBUG  Positions 257+ (Text): mean={txt_portion.mean():.6f}, std={txt_portion.std():.6f}, len={txt_portion.shape[0]}"
                        )
                        print(
                            f"DEBUG  Last position ({seq_len-1}): mean={_debug_mm[0, 0, -1, :].mean():.6f}, std={_debug_mm[0, 0, -1, :].std():.6f}"
                        )

                    # Store text embeddings for comparison between prompts
                    txt_checksum = txt_portion.sum().item() if seq_len > 257 else 0.0
                    visual_checksum = _debug_mm[0, 0, 1:257, :].sum().item()
                    print(f"DEBUG  TEXT checksum: {txt_checksum:.2f} (MUST differ for different prompts!)")
                    print(f"DEBUG  VISUAL checksum: {visual_checksum:.2f} (should differ per image)")
            else:
                # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
                multimodal_embeddings = torch.cat(
                    [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
                )
            multimodal_attention_mask = None
            if attention_mask is not None:
                if self.ttnn_device is not None:
                    attention_mask = ttnn.from_torch(
                        attention_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ttnn_device
                    )
                    multimodal_attention_mask = ttnn.concat(
                        [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                    )
                    multimodal_attention_mask = ttnn.unsqueeze(multimodal_attention_mask, dim=0)
                    multimodal_attention_mask = ttnn.to_layout(multimodal_attention_mask, layout=ttnn.TILE_LAYOUT)
                else:
                    multimodal_attention_mask = torch.cat(
                        [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                    )
            CHECKPOINTS.checkpoint("start_LLMFORWARD")
            # Propagate debug flag to language model (preserve if already set)
            if not getattr(self.language_model, "_debug_trace", False):
                self.language_model._debug_trace = getattr(self, "_debug_trace", False)
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            CHECKPOINTS.checkpoint("end_LLMFORWARD")

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            token = language_model_output[0]
            language_model_output = language_model_output[1:]
            if len(language_model_output) == 0:
                self.cached_output = (None, None)
            else:
                self.cached_output = (language_model_output, projected_patch_embeddings)
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output
        if self.ttnn_device is not None:
            token = language_model_output[0]
            language_model_output = language_model_output[1:]
            if len(language_model_output) == 0:
                self.cached_output = (None, None)
            else:
                self.cached_output = (language_model_output, projected_patch_embeddings)
            return PrismaticCausalLMOutputWithPast(
                loss=None,
                logits=token,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                projector_features=projected_patch_embeddings,
            )
        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}  # Note: HF expects "inputs_embeds" (plural)
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


def get_final_action(generated_ids, action_dims, bin_centers, vocab_size, action_norm_stats):
    # Extract predicted action tokens and translate into (normalized) continuous actions
    predicted_action_token_ids = generated_ids[0, -action_dims:].cpu().numpy()
    discretized_actions = vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
    normalized_actions = bin_centers[discretized_actions]

    # Unnormalize actions
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


class TTOpenVLAForActionPrediction(PrismaticForConditionalGeneration, GenerationMixin):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig, ttnn_device=None, local_state_dict=None) -> None:
        super().__init__(config, ttnn_device=ttnn_device, local_state_dict=local_state_dict)
        self.norm_stats = config.norm_stats
        self.ttnn_device = ttnn_device
        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.local_state_dict = local_state_dict
        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def reset_kv_cache(self):
        """Reset the KV cache to prevent state leakage between runs.

        This is CRITICAL for correct multi-run inference:
        - The KV cache stores attention key/values from previous forward passes
        - Without reset, subsequent runs read stale K/V values from earlier runs
        - This causes the model to produce identical outputs for different inputs

        KV cache structure: tt_kv_cache[model_id][layer_id] -> list of [k_cache, v_cache] tensors
        """
        if self.ttnn_device is None:
            return

        if not hasattr(self, "language_model"):
            return

        if not hasattr(self.language_model, "tt_kv_cache"):
            return

        reset_count = 0
        error_count = 0

        # tt_kv_cache is structured as: [model_id][layer_tensors]
        # where layer_tensors can be a list of [k_cache, v_cache] for each layer
        for model_idx, kv_cache_per_model in enumerate(self.language_model.tt_kv_cache):
            if kv_cache_per_model is None:
                continue

            # kv_cache_per_model is a list of layer caches
            # Each layer cache is typically [k_cache, v_cache]
            def reset_tensor(tensor, path=""):
                nonlocal reset_count, error_count
                if tensor is None:
                    return
                if isinstance(tensor, (list, tuple)):
                    for idx, t in enumerate(tensor):
                        reset_tensor(t, f"{path}[{idx}]")
                    return
                # It's an actual tensor
                try:
                    shape = tensor.shape
                    dtype = tensor.dtype
                    layout = tensor.layout
                    # Create zeros and copy to cache
                    zeros = ttnn.zeros(shape, dtype=dtype, device=self.ttnn_device, layout=layout)
                    ttnn.copy(zeros, tensor)
                    ttnn.deallocate(zeros)
                    reset_count += 1
                except Exception as e:
                    error_count += 1
                    if getattr(self, "_debug_trace", False):
                        print(f"DEBUG: Could not reset KV cache tensor {path}: {e}")

            reset_tensor(kv_cache_per_model, f"[{model_idx}]")

        if getattr(self, "_debug_trace", False):
            print(f"DEBUG reset_kv_cache: Reset {reset_count} tensors, {error_count} errors")
            # Verify the reset worked by checking the first KV tensor
            try:
                kv_first = self.language_model.tt_kv_cache[0][0][0]
                kv_torch = ttnn_to_torch_safe(kv_first, self.ttnn_device)
                kv_sum = kv_torch.abs().sum().item()
                print(f"DEBUG reset_kv_cache: Verification - KV[0][0][0] sum after reset: {kv_sum:.2f} (should be ~0)")
            except Exception as e:
                print(f"DEBUG reset_kv_cache: Verification failed: {e}")

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # DEBUG: Trace predict_action start
        if getattr(self, "_debug_trace", False):
            print(f"DEBUG predict_action: Starting, resetting cached_output and KV cache")

        # Reset cached output to ensure fresh start (prevents stale state between runs)
        self.cached_output = (None, None)

        # Reset KV cache to prevent state leakage between predict_action calls
        # NOTE: This is SLOW (resets 64 tensors) but REQUIRED for correct multi-inference
        # For single-inference deployment, you can disable this by setting _skip_kv_reset=True
        if not getattr(self, "_skip_kv_reset", False):
            self.reset_kv_cache()

        if getattr(self, "_debug_trace", False):
            print(f"DEBUG predict_action: KV cache reset complete")

        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        if self.ttnn_device is not None:
            self.language_model.num_actions = self.get_action_dim(unnorm_key)
        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)

        # get final actions
        actions = get_final_action(
            generated_ids,
            self.get_action_dim(unnorm_key),
            self.bin_centers,
            self.vocab_size,
            self.get_action_stats(unnorm_key),
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt",
    [
        "In: What action should the robot take to {<INSTRUCTION>}?\nOut:",  # OpenVLA robot instruction format
    ],
)
def test_language_model(mesh_device, prompt):
    language_model = OpenVLALanguageModel(mesh_device)
    # Reduce max_generated_tokens to fit within memory constraints (max_seq_len=512)
    predicted_text = language_model.predict_text([prompt], max_generated_tokens=50)
    print("Prompt -> ", prompt)
    print("Final Result -> ", predicted_text)
    print("DONE")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "iterations",
    [1],  # Set to 1 for quick test, change to 100 for perf benchmarking
)
def test_openvla_model(mesh_device, iterations):
    ##  Download model checkpoints HuggingFace
    #   ```
    #   huggingface-cli download openvla/openvla-7b model.safetensors.index.json
    #   huggingface-cli download openvla/openvla-7b model-00001-of-00003.safetensors
    #   huggingface-cli download openvla/openvla-7b model-00002-of-00003.safetensors
    #   huggingface-cli download openvla/openvla-7b model-00003-of-00003.safetensors
    #   export OPENVLA_WEIGHTS="<path_to_downloaded_files>/"
    #   ```
    # Load all tensors

    # List all shard files you want to load
    weight_path = os.getenv("OPENVLA_WEIGHTS", None)
    merged_tensors = None
    if weight_path is not None and os.path.exists(weight_path):
        shard_files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        merged_tensors = {}
        for path in shard_files:
            assert os.path.exists(
                weight_path + path
            ), f"Provided OPENVLA_WEIGHTS path {weight_path + path} does not exist!"
            with safe_open(weight_path + path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_tensors[key] = f.get_tensor(key)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Create test image (use LeRobot image if available, else synthetic)
    LEROBOT_IMAGES_DIR = os.path.expanduser("~/teja/smolvla/demo/images")
    # Try different images to test behavior
    image_options = [
        "lerobot_sample_2.png",  # Try sample 2 first
        "lerobot_sample_3.png",  # Then sample 3
        "lerobot_sample_1.png",  # Fallback to sample 1
    ]
    image = None
    for img_name in image_options:
        image_path = os.path.join(LEROBOT_IMAGES_DIR, img_name)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            print(f"✅ Using image: {image_path}")
            break
    if image is None:
        # Use a different synthetic image with more contrast
        image = Image.new("RGB", (224, 224), color=(255, 100, 50))  # Bright orange
        print(f"⚠️  No LeRobot images found, using synthetic orange image")

    # Try a different prompt
    prompt = "In: What action should the robot take to move forward?\nOut:"
    print(f"📝 Prompt: {prompt}")
    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": "openvla/openvla-7b",
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, kwargs = OpenVLAConfig.from_dict(config_dict, **kwargs)
    vla = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
    results: List[Dict[str, float]] = []
    for i in range(iterations):
        # Enable debug trace only on first iteration
        vla._debug_trace = i == 0

        CHECKPOINTS.reset()
        CHECKPOINTS.checkpoint("start_ACTIONPREDICTION")
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        CHECKPOINTS.checkpoint("end_ACTIONPREDICTION")
        results.append(CHECKPOINTS.analyze())

        if i == 0:
            print(f"\n--- First iteration complete, disabling verbose debug ---\n")

    # combine results
    combined_results = {k: 0.0 for k in results[0].keys()}
    for r in results[min(iterations - 1, 1) :]:
        for k, v in r.items():
            combined_results[k] += v
    results = {k: round(v / len(results), 6) for k, v in combined_results.items()}
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    print(f"Predicted Action: {action}")
    print(f"Timings after running {iterations} iterations: {json.dumps(results, indent=4)}")
