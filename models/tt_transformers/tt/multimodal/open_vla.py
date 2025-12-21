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
from models.tt_transformers.tt.multimodal.ttnn_openvla_vision_bh import OpenVLAVisionEncoderBH


# #region agent log
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str):
    """Write debug log to NDJSON file for debugging TT vs PyTorch divergence."""
    import json as _json

    log_entry = {
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "timestamp": int(time.time() * 1000),
        "sessionId": "openvla-debug",
    }
    try:
        with open("/local/ttuser/teja/METAL_OPENVLA/tt-metal/.cursor/debug.log", "a") as f:
            f.write(_json.dumps(log_entry) + "\n")
    except Exception:
        pass


# #endregion
from models.tt_transformers.tt.common import (
    get_block_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.model_config import ModelArgs

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

    return LLama2OpenVLAArgs


class OpenVLALanguageModel(GenerationMixin):
    def _create_hybrid_precision(self):
        """Create hybrid precision: BFP8 for most layers, BF16 for last N layers to reduce drift."""
        from models.tt_transformers.tt.model_config import (
            DecodersPrecision,
            ModelOptimizations,
            PrecisionSetting,
            TensorGroup,
        )

        num_layers = 32
        bf16_last_n = int(os.environ.get("OPENVLA_BF16_LAYERS", "0"))  # How many last layers use BF16

        # Start with performance mode (BFP8/BFP4)
        hybrid = DecodersPrecision.performance(num_decoders=num_layers, model_name="Llama-2-7b-hf")

        if bf16_last_n > 0:
            # Minimal BF16 config - ONLY WO weight (smallest impact on L1)
            bf16_config = ModelOptimizations(
                {
                    "TensorPrecision": {
                        TensorGroup.WO: PrecisionSetting.BF16,  # Just output proj - smallest overhead
                    },
                    # Keep all OpFidelity at default (no change from base)
                }
            )

            # Set last N layers to BF16
            for layer_id in range(num_layers - bf16_last_n, num_layers):
                hybrid.set_decoder_conf(layer_id, bf16_config)
            print(f"[HYBRID PRECISION] Using BF16 for layers {num_layers - bf16_last_n}-{num_layers-1}, BFP8 for rest")
        else:
            print("[HYBRID PRECISION] Using full BFP8 performance mode (set OPENVLA_BF16_LAYERS=N for hybrid)")

        return hybrid

    def __init__(self, device, local_state_dict=None):
        self.device = device  # Store device reference for profiler flushing

        self.generator_args_config = {
            "num_devices": device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1,
            "data_parallel": 1,
            "mesh_device": device,
            "instruct": False,
            "global_batch_size": 1,
            "optimizations": None,  # Default to accuracy mode (BF16/HIFI4) - should work on T3K
            "max_seq_len": 1024,
            "page_params": {"page_block_size": 32, "page_max_num_blocks_per_dp": 64},  # reduced to fit L1
            "paged_attention": True,
            "num_layers": 32,
        }

        # Create model directly using custom ModelArgs class for OpenVLA
        from models.tt_transformers.demo.simple_text_demo import create_tt_page_table
        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model import Transformer

        LLama2OpenVLAArgs = get_LLama2OpenVLAArgs(local_state_dict)

        paged_attention_config = PagedAttentionConfig(
            block_size=self.generator_args_config["page_params"]["page_block_size"],
            max_num_blocks=self.generator_args_config["page_params"]["page_max_num_blocks_per_dp"],
        )

        # Create model args and model
        tt_model_args = LLama2OpenVLAArgs(
            device,
            instruct=self.generator_args_config["instruct"],
            max_batch_size=self.generator_args_config["global_batch_size"],
            optimizations=self.generator_args_config["optimizations"],
            max_seq_len=self.generator_args_config["max_seq_len"],
        )
        tt_model_args.n_layers = self.generator_args_config["num_layers"]

        # FIX: Set LM head output dtype to bfloat16 for better precision
        tt_model_args.lm_head_dtype = ttnn.bfloat16

        state_dict = tt_model_args.load_state_dict()

        model = Transformer(
            args=tt_model_args,
            mesh_device=device,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
            paged_attention_config=paged_attention_config,
        )

        tt_kv_cache = [l.attention.layer_past for l in model.layers]

        self.model_args = [tt_model_args]
        self.model = [model]
        self.tt_kv_cache = [tt_kv_cache]

        self.page_table = create_tt_page_table(
            global_batch_size=self.generator_args_config["global_batch_size"],
            data_parallel=self.generator_args_config["data_parallel"],
            paged_attention_config=paged_attention_config,
        )

        self.tokenizer = tt_model_args.tokenizer
        self.processor = tt_model_args.processor

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
            logits, _ = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=device_sampling_params,
            )

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
        seq_len = inputs_embeds.shape[2]
        CHECKPOINTS.checkpoint("start_PREFILL")
        padding = get_padded_prefill_len(seq_len) - inputs_embeds.shape[2]
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
        tt_logits = self.model[0].forward(
            inputs_embeds,
            None,
            rot_mats_global=tt_rot_mats_prefill_global,
            mode="prefill",
            page_table=tt_page_table,
            kv_cache=self.tt_kv_cache[0],
            get_last_token=((seq_len - 1) // 32) * 32,
        )

        # Flush profiler buffer after prefill (prevent overflow for large models)
        ttnn.synchronize_device(self.device)
        ttnn.ReadDeviceProfiler(self.device)

        last_token_idx = seq_len - 1

        # Since we give unpadded_seq_len, only the tile containing the last token is returned
        # Main branch API: process_output_prefill expects host tensor
        tt_logits_host = tt_logits.cpu()
        output_logits = self.model[0].process_output_prefill(tt_logits_host, last_token_idx=(last_token_idx % 32))
        prefilled_token = torch.argmax(output_logits.cpu(), dim=-1).unsqueeze(0)
        # #region agent log
        _debug_log(
            "open_vla.py:prefill_output",
            "TT prefill output logits",
            {
                "shape": list(output_logits.shape),
                "mean": float(output_logits.float().mean()),
                "std": float(output_logits.float().std()),
                "argmax_token": int(prefilled_token[0].item()),
                "top5_tokens": output_logits.topk(5).indices.tolist(),
                "top5_values": [float(v) for v in output_logits.topk(5).values.tolist()],
            },
            "D",
        )
        # #endregion
        # Initial positions
        current_pos = torch.tensor([seq_len])
        out_tok = prefilled_token
        output_toks = []
        print(f"  [DECODE DEBUG] Starting decode loop: seq_len={seq_len}, num_actions={self.num_actions}")
        print(f"  [DECODE DEBUG] Initial: current_pos={current_pos.item()}, out_tok={out_tok.tolist()}")
        CHECKPOINTS.checkpoint("end_PREFILL")

        # If num_actions=0, return prefill logits only (for debugging/teacher forcing)
        if self.num_actions == 0:
            print(f"  [DECODE DEBUG] num_actions=0, returning prefill logits only")
            return output_logits.unsqueeze(0)  # Shape: [1, vocab_size]

        for i in range(self.num_actions):
            # Run decode forward
            CHECKPOINTS.checkpoint("start_LLM_DECODE")
            print(f"  [DECODE DEBUG] Step {i}: input_tok={out_tok.flatten().tolist()}, pos={current_pos.item()}")
            logits, _ = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=None,
                enable_trace=False,
            )
            # Flush profiler buffer after each decode step to prevent overflow
            ttnn.synchronize_device(self.device)
            ttnn.ReadDeviceProfiler(self.device)
            CHECKPOINTS.checkpoint("end_LLM_DECODE")
            current_pos += 1
            output_toks.append(logits)

            # Temperature sampling to break attractor token pattern
            temperature = float(os.environ.get("OPENVLA_TEMPERATURE", "0.0"))
            top_k = int(os.environ.get("OPENVLA_TOP_K", "0"))

            if temperature > 0:
                scaled_logits = logits.view(-1) / temperature
                if top_k > 0:
                    # Top-k filtering
                    top_k_vals, top_k_idx = scaled_logits.topk(top_k)
                    filtered_logits = torch.full_like(scaled_logits, float("-inf"))
                    filtered_logits.scatter_(0, top_k_idx, top_k_vals)
                    probs = torch.softmax(filtered_logits, dim=-1)
                else:
                    probs = torch.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(logits, dim=-1)

            top5 = logits.view(-1).topk(5)
            print(
                f"  [DECODE DEBUG] Step {i}: output_tok={next_tok.flatten().tolist()}, top5={top5.indices.tolist()}, top5_vals=[{', '.join([f'{v:.2f}' for v in top5.values.tolist()])}]",
                f"(temp={temperature})" if temperature > 0 else "",
            )
            out_tok = next_tok.unsqueeze(0)
        print(f"  [DECODE DEBUG] Final tokens: {[torch.argmax(l, dim=-1).item() for l in output_toks]}")
        return output_toks

    def decode_with_teacher_forcing(self, inputs_embeds, seq_len, teacher_tokens):
        """
        Decode using teacher forcing - feed PT's tokens instead of TT's predictions.
        This isolates whether decode divergence is due to snowball effect vs fundamental bug.

        Args:
            inputs_embeds: TTNN tensor of multimodal embeddings from prefill
            seq_len: Starting sequence length (after prefill)
            teacher_tokens: List of 7 ground truth tokens from PT model

        Returns:
            List of (logits, pcc_with_pt) tuples for each decode step
        """
        print(f"  [TEACHER FORCING] Starting with seq_len={seq_len}, teacher_tokens={teacher_tokens}")

        # Run prefill first
        logits = self.generator.prefill_forward_text(
            None,  # input_ids (not used when embeddings provided)
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=[seq_len],
            embeddings=inputs_embeds,
        )
        prefilled_token = torch.argmax(logits, dim=-1)
        print(f"  [TEACHER FORCING] Prefill token: {prefilled_token.item()}")

        current_pos = torch.tensor([seq_len])
        output_logits = []

        for i in range(len(teacher_tokens)):
            # Use TEACHER token (from PT), not TT's prediction
            teacher_tok = torch.tensor([[teacher_tokens[i]]])

            print(f"  [TEACHER FORCING] Step {i}: feeding teacher_tok={teacher_tokens[i]}, pos={current_pos.item()}")

            step_logits, _ = self.generator.decode_forward_text(
                teacher_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=None,
                enable_trace=False,
            )
            ttnn.synchronize_device(self.device)

            current_pos += 1
            tt_pred = torch.argmax(step_logits, dim=-1).item()
            top5 = step_logits.view(-1).topk(5)
            print(f"  [TEACHER FORCING] Step {i}: TT predicts {tt_pred}, top5={top5.indices.tolist()}")

            output_logits.append(step_logits)

        return output_logits


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
        return result[0] if isinstance(result, tuple) else result

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
        self.featurizer.forward = unpack_tuple(
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

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
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

        # Use high-PCC Blackhole-optimized vision encoder for TTNN
        self.bh_vision_encoder = None
        if self.ttnn_device is not None and self.local_state_dict is not None:
            CHECKPOINTS.checkpoint("start_BH_VISION_INIT")
            print("Creating Blackhole-optimized vision encoder (high PCC)...")
            self.bh_vision_encoder = OpenVLAVisionEncoderBH(self.ttnn_device, self.local_state_dict)
            CHECKPOINTS.checkpoint("end_BH_VISION_INIT")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Use Blackhole-optimized vision encoder for TTNN path
        if self.bh_vision_encoder is not None:
            CHECKPOINTS.checkpoint("start_BH_VISION_FORWARD")
            # BH encoder expects [B, 6, 224, 224] torch tensor directly
            result = self.bh_vision_encoder(pixel_values)
            CHECKPOINTS.checkpoint("end_BH_VISION_FORWARD")
            # #region agent log
            result_torch = ttnn.to_torch(result)
            _debug_log(
                "open_vla.py:vision_backbone_tt",
                "TT vision backbone output (BH)",
                {
                    "shape": list(result_torch.shape),
                    "mean": float(result_torch.float().mean()),
                    "std": float(result_torch.float().std()),
                    "min": float(result_torch.float().min()),
                    "max": float(result_torch.float().max()),
                },
                "A",
            )
            # #endregion
            return result

        # PyTorch path (when no BH vision encoder)
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        # TIMM get_intermediate_layers returns list/tuple - extract last element
        patches = patches[-1] if isinstance(patches, (list, tuple)) else patches
        patches_fused = patches_fused[-1] if isinstance(patches_fused, (list, tuple)) else patches_fused
        result = torch.cat([patches, patches_fused], dim=2)
        # #region agent log
        _debug_log(
            "open_vla.py:vision_backbone_pt",
            "PyTorch vision backbone output",
            {
                "shape": list(result.shape),
                "mean": float(result.float().mean()),
                "std": float(result.float().std()),
                "min": float(result.float().min()),
                "max": float(result.float().max()),
            },
            "A",
        )
        # #endregion
        # Convert to ttnn if device exists (for projector compatibility)
        if self.ttnn_device is not None:
            result = ttnn.from_torch(result, dtype=ttnn.bfloat16, device=self.ttnn_device, layout=ttnn.TILE_LAYOUT)
        return result


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

        # #region agent log
        _debug_log(
            "open_vla.py:projector_pt",
            "PyTorch projector output",
            {
                "shape": list(projected_features.shape),
                "mean": float(projected_features.float().mean()),
                "std": float(projected_features.float().std()),
                "min": float(projected_features.float().min()),
                "max": float(projected_features.float().max()),
            },
            "B",
        )
        # #endregion
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
        # Use bfloat16 for higher precision in projector (bfloat8_b loses too much information)
        projected_features = ttnn.linear(
            img_patches,
            self.params.fc1.weight,
            bias=self.params.fc1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            activation="gelu",
        )
        projected_features = ttnn.linear(
            projected_features,
            self.params.fc2.weight,
            bias=self.params.fc2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            activation="gelu",
        )
        projected_features = ttnn.linear(
            projected_features,
            self.params.fc3.weight,
            bias=self.params.fc3.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        # #region agent log
        pf_torch = ttnn.to_torch(projected_features)
        _debug_log(
            "open_vla.py:projector_tt",
            "TT projector output",
            {
                "shape": list(pf_torch.shape),
                "mean": float(pf_torch.float().mean()),
                "std": float(pf_torch.float().std()),
                "min": float(pf_torch.float().min()),
                "max": float(pf_torch.float().max()),
                "dtype": "bfloat16",
            },
            "B",
        )
        # #endregion
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
        # Load projector weights BEFORE preprocess_model_parameters to get correct TTNN params
        self._projector_state_dict = None
        if local_state_dict is not None:
            projector_sd = {
                k.replace("projector.", ""): v for k, v in local_state_dict.items() if k.startswith("projector.")
            }
            self.projector.load_state_dict(projector_sd, strict=True)
            self._projector_state_dict = projector_sd  # Save for reload after post_init()
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
        # Save state dict for weight loading after post_init()
        self._local_state_dict = local_state_dict

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

        # Load projector weights AFTER post_init() to avoid being reset by _init_weights()
        if self._projector_state_dict is not None:
            self.projector.load_state_dict(self._projector_state_dict, strict=True)
            del self._projector_state_dict  # Clean up

        # CRITICAL: Load LLM weights for PT model (ttnn_device=None) AFTER post_init()
        # The TT model loads weights in OpenVLALanguageModel, but PT model uses from_config() which is random!
        if ttnn_device is None and local_state_dict is not None:
            llm_state_dict = {
                k.replace("language_model.", ""): v
                for k, v in local_state_dict.items()
                if k.startswith("language_model.")
            }
            if llm_state_dict:
                missing, unexpected = self.language_model.load_state_dict(llm_state_dict, strict=False)
                print(
                    f"  PT LLM weights loaded: {len(llm_state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}"
                )

        # CRITICAL: Reload vision backbone weights AFTER post_init() as _init_weights() resets them!
        if local_state_dict is not None:
            featurizer_state_dict = {
                k.replace("vision_backbone.featurizer.", ""): v
                for k, v in local_state_dict.items()
                if k.startswith("vision_backbone.featurizer.")
            }
            if featurizer_state_dict:
                self.vision_backbone.featurizer.load_state_dict(featurizer_state_dict, strict=True)

            if hasattr(self.vision_backbone, "fused_featurizer"):
                fused_featurizer_state_dict = {
                    k.replace("vision_backbone.fused_featurizer.", ""): v
                    for k, v in local_state_dict.items()
                    if k.startswith("vision_backbone.fused_featurizer.")
                }
                if fused_featurizer_state_dict:
                    self.vision_backbone.fused_featurizer.load_state_dict(fused_featurizer_state_dict, strict=True)

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
            language_model_output = self.cached_output[0][1:]
            if len(language_model_output) == 0:
                self.cached_output = (None, None)
            else:
                self.cached_output = (language_model_output, self.cached_output[1])
            if not return_dict:
                if output_projector_features and (projected_patch_embeddings is not None):
                    return token, projected_patch_embeddings
                return token
            return PrismaticCausalLMOutputWithPast(
                loss=None,
                logits=token,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                projector_features=self.cached_output[1],
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
                # BH vision encoder handles preprocessing internally - pass raw NCHW tensor
                CHECKPOINTS.checkpoint("start_VISIONFORWARD")
                ttnn_patch_features = self.vision_backbone(pixel_values)
                # Flush profiler buffer after vision processing
                ttnn.ReadDeviceProfiler(self.ttnn_device)
                CHECKPOINTS.checkpoint("end_VISIONFORWARD")
                CHECKPOINTS.checkpoint("start_PROJECTORFORWARD")
                projected_patch_embeddings = self.ttnn_projector.forward(ttnn_patch_features)
                projected_patch_embeddings = ttnn.mesh_partition(
                    projected_patch_embeddings,
                    -1,
                )
                # Flush profiler buffer after projector
                ttnn.ReadDeviceProfiler(self.ttnn_device)
                CHECKPOINTS.checkpoint("end_PROJECTORFORWARD")
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
                # FIX: Token IDs must be uint32 for embedding lookup, not bfloat16!
                ttnn_input_ids = ttnn.from_torch(
                    input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ttnn_device
                )
                input_embeddings = self.get_input_embeddings()(ttnn_input_ids)
                # Flush profiler buffer after embeddings
                ttnn.ReadDeviceProfiler(self.ttnn_device)
                CHECKPOINTS.checkpoint("end_LLMINPUTEMBEDDINGS")
            else:
                input_embeddings = self.get_input_embeddings()(input_ids)

            if self.ttnn_device is not None:
                CHECKPOINTS.checkpoint("start_VISIONLLMCONCAT")
                multimodal_embeddings = ttnn.concat(
                    [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
                )
                multimodal_embeddings = ttnn.unsqueeze(multimodal_embeddings, dim=0)
                multimodal_embeddings = ttnn.to_layout(multimodal_embeddings, layout=ttnn.TILE_LAYOUT)
                # Flush profiler buffer after concat
                ttnn.ReadDeviceProfiler(self.ttnn_device)
                CHECKPOINTS.checkpoint("end_VISIONLLMCONCAT")
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
            # #region agent log
            if self.ttnn_device is not None:
                mm_torch = ttnn.to_torch(multimodal_embeddings)
                _debug_log(
                    "open_vla.py:multimodal_embeds_tt",
                    "TT multimodal embeddings before LLM",
                    {
                        "shape": list(mm_torch.shape),
                        "mean": float(mm_torch.float().mean()),
                        "std": float(mm_torch.float().std()),
                        "min": float(mm_torch.float().min()),
                        "max": float(mm_torch.float().max()),
                    },
                    "C",
                )
            else:
                _debug_log(
                    "open_vla.py:multimodal_embeds_pt",
                    "PyTorch multimodal embeddings before LLM",
                    {
                        "shape": list(multimodal_embeddings.shape),
                        "mean": float(multimodal_embeddings.float().mean()),
                        "std": float(multimodal_embeddings.float().std()),
                        "min": float(multimodal_embeddings.float().min()),
                        "max": float(multimodal_embeddings.float().max()),
                    },
                    "C",
                )
            # #endregion
            CHECKPOINTS.checkpoint("start_LLMFORWARD")
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
            # Flush profiler buffer after LLM forward (prefill + decode)
            if self.ttnn_device is not None:
                ttnn.ReadDeviceProfiler(self.ttnn_device)
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
        # FIX: HF expects "inputs_embeds" (with 's'), not "input_embeds"
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
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

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
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
        "What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        "In: What action should the robot take to {<INSTRUCTION>}?\nOut:",
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
    [1],
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

    # Create random init image
    image: Image.Image = Image.new("RGB", (224, 224))
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
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
        CHECKPOINTS.reset()
        CHECKPOINTS.checkpoint("start_ACTIONPREDICTION")
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        CHECKPOINTS.checkpoint("end_ACTIONPREDICTION")
        results.append(CHECKPOINTS.analyze())

    # combine results
    combined_results = {k: 0.0 for k in results[0].keys()}
    for r in results[min(iterations - 1, 1) :]:
        for k, v in r.items():
            combined_results[k] += v
    results = {k: round(v / len(results), 6) for k, v in combined_results.items()}
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    print(f"Predicted Action: {action}")
    print(f"Timings after running {iterations} iterations: {json.dumps(results, indent=4)}")


def _compute_pcc(a, b):
    """Compute Pearson correlation coefficient between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean
    num = (a_centered * b_centered).sum()
    denom = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    return (num / denom).item() if denom > 0 else 0.0


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
        {"N150": (1, 1), "N300": (1, 2), "P150": (1, 1), "P300": (1, 2)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_openvla_module_pcc(mesh_device):
    """Run each module separately and compute PCC between TT and PyTorch."""
    import numpy as np

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
            with safe_open(weight_path + path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_tensors[key] = f.get_tensor(key)

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    image: Image.Image = Image.new("RGB", (224, 224))
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

    config_dict, _ = OpenVLAConfig.get_config_dict(
        return_unused_kwargs=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        name_or_path="openvla/openvla-7b",
        pretrained_model_name_or_path="openvla/openvla-7b",
    )
    vla_config, _ = OpenVLAConfig.from_dict(config_dict, return_unused_kwargs=True)

    print("\n" + "=" * 70)
    print("MODULE-LEVEL PCC COMPARISON: TT vs PyTorch")
    print("=" * 70)

    # Create PyTorch reference model (no TT device)
    print("\nCreating PyTorch reference model...")
    pt_model = TTOpenVLAForActionPrediction(vla_config, ttnn_device=None, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )

    # Create TT model
    print("Creating TT model...")
    tt_model = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )

    inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
    pixel_values = inputs["pixel_values"]

    # Split pixel values: first 3 channels for DinoV2, next 3 for SigLIP
    img_dino, img_siglip = torch.split(pixel_values, [3, 3], dim=1)

    print(f"\nInput shapes: DinoV2={img_dino.shape}, SigLIP={img_siglip.shape}")

    # ========== 1. Vision Backbone ==========
    print("\n" + "-" * 50)
    print("[1] VISION BACKBONE (PT vs TT)")
    print("-" * 50)

    pcc_dino = None
    pcc_siglip = None
    pcc_vision = None

    # PT Vision: Use forward_features (all layers + final norm)
    with torch.no_grad():
        pt_dino_out = pt_model.vision_backbone.featurizer.forward_features(img_dino)[:, 5:, :]  # Skip CLS+REG
        pt_siglip_out = pt_model.vision_backbone.fused_featurizer.forward_features(img_siglip)
        pt_vision_out = torch.cat([pt_dino_out, pt_siglip_out], dim=2)

    # TT Vision: BH encoder
    tt_vision_out_raw = tt_model.vision_backbone(pixel_values)
    tt_vision_out = ttnn.to_torch(tt_vision_out_raw)
    tt_dino_out = tt_vision_out[:, :, :1024]
    tt_siglip_out = tt_vision_out[:, :, 1024:]

    # DinoV2 comparison
    print(f"\n  DinoV2:")
    print(
        f"    PT: shape={pt_dino_out.shape}, mean={pt_dino_out.float().mean():.6f}, std={pt_dino_out.float().std():.6f}"
    )
    print(
        f"    TT: shape={tt_dino_out.shape}, mean={tt_dino_out.float().mean():.6f}, std={tt_dino_out.float().std():.6f}"
    )
    pcc_dino = _compute_pcc(pt_dino_out, tt_dino_out)
    print(f"    PCC: {pcc_dino:.6f}")

    # SigLIP comparison
    print(f"\n  SigLIP:")
    print(
        f"    PT: shape={pt_siglip_out.shape}, mean={pt_siglip_out.float().mean():.6f}, std={pt_siglip_out.float().std():.6f}"
    )
    print(
        f"    TT: shape={tt_siglip_out.shape}, mean={tt_siglip_out.float().mean():.6f}, std={tt_siglip_out.float().std():.6f}"
    )
    pcc_siglip = _compute_pcc(pt_siglip_out, tt_siglip_out)
    print(f"    PCC: {pcc_siglip:.6f}")

    # Combined vision comparison
    print(f"\n  Combined Vision:")
    print(
        f"    PT: shape={pt_vision_out.shape}, mean={pt_vision_out.float().mean():.6f}, std={pt_vision_out.float().std():.6f}"
    )
    print(
        f"    TT: shape={tt_vision_out.shape}, mean={tt_vision_out.float().mean():.6f}, std={tt_vision_out.float().std():.6f}"
    )
    pcc_vision = _compute_pcc(pt_vision_out, tt_vision_out)
    print(f"    PCC: {pcc_vision:.6f}")

    # ========== 2. Projector ==========
    print("\n" + "-" * 50)
    print("[2] PROJECTOR")
    print("-" * 50)

    # PyTorch projector
    with torch.no_grad():
        pt_proj_out = pt_model.projector(pt_vision_out)
    print(
        f"  PT Projector: shape={pt_proj_out.shape}, mean={pt_proj_out.float().mean():.6f}, std={pt_proj_out.float().std():.6f}"
    )

    # TT projector (using PT vision output to isolate projector)
    tt_input = ttnn.from_torch(pt_vision_out, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_proj_out_raw = tt_model.ttnn_projector.forward(tt_input)
    tt_proj_out = ttnn.to_torch(tt_proj_out_raw)
    print(
        f"  TT Projector: shape={tt_proj_out.shape}, mean={tt_proj_out.float().mean():.6f}, std={tt_proj_out.float().std():.6f}"
    )

    pcc_proj = _compute_pcc(pt_proj_out, tt_proj_out)
    print(f"  Projector PCC: {pcc_proj:.6f}")

    # ========== 3. Full Model Forward (1 token) ==========
    print("\n" + "-" * 50)
    print("[3] FULL MODEL FORWARD (first token generation)")
    print("-" * 50)

    # Run full forward pass to get first token logits
    with torch.no_grad():
        pt_outputs = pt_model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs.get("attention_mask", None),
        )
    pt_logits = pt_outputs.logits[:, -1, :]  # Last position logits
    pt_first_token = pt_logits.argmax(dim=-1).item()
    print(
        f"  PT Logits: shape={pt_logits.shape}, mean={pt_logits.float().mean():.6f}, std={pt_logits.float().std():.6f}"
    )
    print(f"  PT First predicted token: {pt_first_token}")

    # TT model forward - use the language_model's prefill
    # Note: TT model uses different flow, so we test via generate
    print("  (TT forward tested via generate below)")

    # ========== 4. LLM PREFILL COMPARISON ==========
    print("\n" + "-" * 50)
    print("[4] LLM PREFILL COMPARISON (no decode)")
    print("-" * 50)

    # Get multimodal embeddings for both PT and TT
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # Debug: Show what tokens we're feeding
    print(f"  Input IDs: shape={input_ids.shape}, tokens={input_ids[0].tolist()[:10]}...{input_ids[0].tolist()[-5:]}")
    print(f"  Last token: {input_ids[0, -1].item()} (should be 29871 for action generation)")

    # Fix: Add trailing space token (29871) for proper action generation
    if input_ids[0, -1].item() != 29871:
        print(f"  Adding trailing space token (29871) for action generation...")
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=input_ids.dtype)], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=1)
        print(f"  New input shape: {input_ids.shape}, last token: {input_ids[0, -1].item()}")

    # PT: Get projected vision features and multimodal embeddings
    with torch.no_grad():
        # Use forward_features for PT vision to match BH encoder
        pt_vis_full = torch.cat(
            [
                pt_model.vision_backbone.featurizer.forward_features(img_dino)[:, 5:, :],
                pt_model.vision_backbone.fused_featurizer.forward_features(img_siglip),
            ],
            dim=2,
        )
        pt_proj_full = pt_model.projector(pt_vis_full)

        # Get text embeddings
        pt_text_embeds = pt_model.language_model.model.embed_tokens(input_ids)

        print(
            f"  PT Projector out: shape={pt_proj_full.shape}, mean={pt_proj_full.float().mean():.6f}, std={pt_proj_full.float().std():.6f}"
        )
        print(f"  PT Text embeds:   shape={pt_text_embeds.shape}, mean={pt_text_embeds.float().mean():.6f}")

    # TT: Get projected vision features
    tt_vis_out = tt_model.vision_backbone(pixel_values)
    tt_vis_torch = ttnn.to_torch(tt_vis_out)

    tt_proj_input = ttnn.from_torch(tt_vis_torch, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_proj_out = tt_model.ttnn_projector.forward(tt_proj_input)
    tt_proj_torch = ttnn.to_torch(tt_proj_out)

    print(
        f"  TT Projector out: shape={tt_proj_torch.shape}, mean={tt_proj_torch.float().mean():.6f}, std={tt_proj_torch.float().std():.6f}"
    )

    # Compute PCC for projected features (this is what goes into LLM)
    pcc_proj_full = _compute_pcc(pt_proj_full, tt_proj_torch)
    print(f"  Projector (full pipeline) PCC: {pcc_proj_full:.6f}")

    # ========== LLM PREFILL DEBUG ==========
    print("\n  --- LLM Prefill Debug ---")

    # PT: Run full forward to get logits from multimodal input
    with torch.no_grad():
        pt_outputs = pt_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    pt_logits = pt_outputs.logits
    pt_last_hidden = pt_outputs.hidden_states[-1] if pt_outputs.hidden_states else None

    print(
        f"  PT Logits: shape={pt_logits.shape}, mean={pt_logits.float().mean():.6f}, std={pt_logits.float().std():.6f}"
    )
    print(f"  PT Last token logits: argmax={pt_logits[0, -1, :].argmax().item()}")
    if pt_last_hidden is not None:
        print(f"  PT Last hidden: shape={pt_last_hidden.shape}, mean={pt_last_hidden.float().mean():.6f}")

    # Get PT first predicted token
    pt_first_token = pt_logits[0, -1, :].argmax().item()
    print(f"  PT First predicted token: {pt_first_token}")

    # TT: Run forward to get first token
    # We need to use the TT model's forward path
    print("\n  TT Forward (prefill):")
    try:
        # TT model forward returns logits in a different format
        # Let's trace what happens in generate()
        tt_outputs = tt_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        if hasattr(tt_outputs, "logits"):
            tt_logits = tt_outputs.logits
            if isinstance(tt_logits, ttnn.Tensor):
                tt_logits = ttnn.to_torch(tt_logits)
            print(
                f"  TT Logits: shape={tt_logits.shape}, mean={tt_logits.float().mean():.6f}, std={tt_logits.float().std():.6f}"
            )
            tt_first_token = (
                tt_logits[0, -1, :].argmax().item() if len(tt_logits.shape) == 3 else tt_logits.argmax().item()
            )
            print(f"  TT First predicted token: {tt_first_token}")

            # Compare logits
            if pt_logits.shape == tt_logits.shape:
                pcc_logits = _compute_pcc(pt_logits, tt_logits)
                print(f"  Logits PCC: {pcc_logits:.6f}")
            else:
                print(f"  Logits shape mismatch: PT={pt_logits.shape}, TT={tt_logits.shape}")
                # Compare last position logits only
                pt_last_logits = pt_logits[0, -1, :].float()
                tt_last_logits = (
                    tt_logits.flatten()[: pt_last_logits.shape[0]].float()
                    if tt_logits.numel() >= pt_last_logits.shape[0]
                    else tt_logits.flatten().float()
                )

                # Detailed comparison of last position
                print(f"\n  --- Last Position Logits Comparison ---")
                print(
                    f"  PT last logits: mean={pt_last_logits.mean():.6f}, std={pt_last_logits.std():.6f}, min={pt_last_logits.min():.6f}, max={pt_last_logits.max():.6f}"
                )
                print(
                    f"  TT last logits: mean={tt_last_logits.mean():.6f}, std={tt_last_logits.std():.6f}, min={tt_last_logits.min():.6f}, max={tt_last_logits.max():.6f}"
                )

                if tt_last_logits.shape[0] == pt_last_logits.shape[0]:
                    pcc_last_logits = _compute_pcc(pt_last_logits, tt_last_logits)
                    print(f"  Last position logits PCC: {pcc_last_logits:.6f}")

                # Top-5 tokens comparison
                pt_top5 = pt_last_logits.topk(5)
                tt_top5 = tt_last_logits.topk(5)
                print(f"\n  PT Top-5 tokens: {pt_top5.indices.tolist()}")
                print(f"  PT Top-5 values: {[f'{v:.2f}' for v in pt_top5.values.tolist()]}")
                print(f"  TT Top-5 tokens: {tt_top5.indices.tolist()}")
                print(f"  TT Top-5 values: {[f'{v:.2f}' for v in tt_top5.values.tolist()]}")
        else:
            print(f"  TT outputs type: {type(tt_outputs)}")
            if isinstance(tt_outputs, tuple):
                print(
                    f"  TT outputs[0] shape: {tt_outputs[0].shape if hasattr(tt_outputs[0], 'shape') else type(tt_outputs[0])}"
                )
    except Exception as e:
        import traceback

        print(f"  TT forward failed: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()

    # ========== 5. FULL ACTION PREDICTION (7 tokens) ==========
    print("\n" + "-" * 50)
    print("[5] FULL ACTION PREDICTION (all 7 action tokens)")
    print("-" * 50)

    # PT: Generate all 7 action tokens
    print("\n  PT predict_action():")
    with torch.no_grad():
        pt_generated = pt_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=7,
            do_sample=False,
        )
    pt_action_tokens = pt_generated[0, -7:].tolist()
    print(f"  PT Generated tokens (last 7): {pt_action_tokens}")

    # TT: Generate all 7 action tokens
    print("\n  TT predict_action():")
    try:
        # CRITICAL: Set num_actions before generate() - this tells the TT model
        # to generate all 7 tokens in one prefill+decode pass
        tt_model.language_model.num_actions = 7
        print(f"  Set tt_model.language_model.num_actions = 7")

        tt_generated = tt_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=7,
            do_sample=False,
        )
        if isinstance(tt_generated, ttnn.Tensor):
            tt_generated = ttnn.to_torch(tt_generated)
        tt_action_tokens = tt_generated[0, -7:].tolist()
        print(f"  TT Generated tokens (last 7): {tt_action_tokens}")

        # Compare
        print("\n  --- Token by Token Comparison ---")
        for i, (pt_tok, tt_tok) in enumerate(zip(pt_action_tokens, tt_action_tokens)):
            match = "✓" if pt_tok == tt_tok else "✗"
            print(f"  Action {i}: PT={pt_tok}, TT={tt_tok} {match}")

        # Check if TT is constant
        if len(set(tt_action_tokens)) == 1:
            print(f"\n  ⚠️  WARNING: TT outputs CONSTANT token {tt_action_tokens[0]} for all 7 actions!")

        tokens_match = pt_action_tokens == tt_action_tokens
        print(f"\n  All tokens match: {tokens_match}")

        # ========== Convert tokens to actual action vectors ==========
        print("\n  --- Final Action Vectors (7 DoF) ---")

        # Get bin_centers and vocab_size from model
        bin_centers = pt_model.bin_centers
        vocab_size = pt_model.vocab_size

        # Convert tokens to normalized actions
        def tokens_to_actions(tokens, vocab_size, bin_centers):
            tokens_np = np.array(tokens)
            discretized = vocab_size - tokens_np
            discretized = np.clip(discretized - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
            return bin_centers[discretized]

        pt_actions_norm = tokens_to_actions(pt_action_tokens, vocab_size, bin_centers)
        tt_actions_norm = tokens_to_actions(tt_action_tokens, vocab_size, bin_centers)

        print(f"  Vocab size: {vocab_size}, Num bins: {bin_centers.shape[0]}")
        print(f"\n  PT normalized actions: [{', '.join([f'{a:.4f}' for a in pt_actions_norm])}]")
        print(f"  TT normalized actions: [{', '.join([f'{a:.4f}' for a in tt_actions_norm])}]")

        # Compute action error
        action_diff = np.abs(pt_actions_norm - tt_actions_norm)
        print(f"\n  Action error (|PT - TT|): [{', '.join([f'{d:.4f}' for d in action_diff])}]")
        print(f"  Mean action error: {action_diff.mean():.4f}")
        print(f"  Max action error:  {action_diff.max():.4f}")

        # Unnormalize to actual robot actions (if norm_stats available)
        try:
            unnorm_key = "bridge_orig"
            action_stats = pt_model.get_action_stats(unnorm_key)
            mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_stats["q99"]), np.array(action_stats["q01"])

            pt_actions_unnorm = np.where(
                mask, 0.5 * (pt_actions_norm + 1) * (action_high - action_low) + action_low, pt_actions_norm
            )
            tt_actions_unnorm = np.where(
                mask, 0.5 * (tt_actions_norm + 1) * (action_high - action_low) + action_low, tt_actions_norm
            )

            print(f"\n  PT unnormalized actions ({unnorm_key}):")
            print(f"    [{', '.join([f'{a:.4f}' for a in pt_actions_unnorm])}]")
            print(f"  TT unnormalized actions ({unnorm_key}):")
            print(f"    [{', '.join([f'{a:.4f}' for a in tt_actions_unnorm])}]")

            # Show what each DoF represents
            dof_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            print(f"\n  DoF comparison:")
            for i, (pt_a, tt_a, name) in enumerate(zip(pt_actions_unnorm, tt_actions_unnorm, dof_names)):
                diff = abs(pt_a - tt_a)
                status = "✓" if diff < 0.01 else "✗"
                print(f"    {name:8s}: PT={pt_a:+.4f}, TT={tt_a:+.4f}, diff={diff:.4f} {status}")
        except Exception as e:
            print(f"  (Could not unnormalize: {e})")

    except Exception as e:
        import traceback

        print(f"  TT generate failed: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
        tt_action_tokens = []

    # ========== 6. TEACHER FORCING TEST ==========
    # DISABLED: Causes L1 memory overflow due to accumulated tensors from sections [1]-[5]
    # This section does additional forward passes that exceed L1 capacity
    print("\n" + "-" * 50)
    print("[6] TEACHER FORCING TEST - SKIPPED (L1 memory constraint)")
    print("-" * 50)

    if False and pt_action_tokens:  # Disabled
        try:
            # ========== Reset TT model for fresh KV cache ==========
            print("\n  --- Running TT PREFILL ONLY (num_actions=0) ---")

            # Set num_actions=0 to run prefill only, no decode
            tt_model.language_model.num_actions = 0

            # Run TT forward - this does prefill and returns prefill logits only
            tt_out_prefill = tt_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

            # Get prefill logits - handle different return types
            if hasattr(tt_out_prefill, "logits"):
                tt_prefill_logits = tt_out_prefill.logits
            else:
                tt_prefill_logits = tt_out_prefill  # Direct tensor return from language_model

            if isinstance(tt_prefill_logits, ttnn.Tensor):
                tt_prefill_logits = ttnn.to_torch(tt_prefill_logits)

            # Keep full shape for row analysis before flattening
            tt_prefill_full = tt_prefill_logits.float()
            print(f"  TT Prefill raw shape: {list(tt_prefill_full.shape)}")

            # Reshape to [32, vocab] for row analysis
            if tt_prefill_full.numel() >= 32 * 32064:
                tt_32rows = tt_prefill_full.view(-1)[: 32 * 32064].view(32, 32064)
                # CORRECT ROW: prefill last token is at position (seq_len - 1)
                prefill_row = (input_ids.shape[1] - 1) % 32
                print(f"  TT Prefill has 32 rows, extracting row {prefill_row} (pos={input_ids.shape[1]-1} % 32)")
                tt_prefill_logits = tt_32rows[prefill_row]

                # Also show what row 0 would give for comparison
                row0_argmax = tt_32rows[0].argmax().item()
                print(f"  TT row 0 argmax={row0_argmax}, row {prefill_row} argmax={tt_prefill_logits.argmax().item()}")
            else:
                tt_32rows = tt_prefill_full.view(-1, 32064)
                tt_prefill_logits = tt_32rows[0]
                print(f"  TT Prefill already flattened (not 32 rows), using row 0")

            tt_tok0 = tt_prefill_logits.argmax().item()
            print(f"  TT Prefill (correct row): argmax={tt_tok0}, mean={tt_prefill_logits.mean():.4f}")

            # CRITICAL: Compare logit values at PT vs TT argmax positions
            pt_argmax_token = pt_step_logits[0].argmax().item() if pt_step_logits else 31874
            tt_argmax_token = tt_tok0
            print(f"\n  --- CRITICAL: Logit Value Comparison ---")
            print(f"    PT picks token {pt_argmax_token}, TT picks token {tt_argmax_token}")
            print(
                f"    PT logit at PT's token ({pt_argmax_token}): {pt_step_logits[0][pt_argmax_token].item():.4f}"
                if pt_step_logits
                else "    (no pt_step_logits)"
            )
            print(
                f"    PT logit at TT's token ({tt_argmax_token}): {pt_step_logits[0][tt_argmax_token].item():.4f}"
                if pt_step_logits
                else ""
            )
            print(f"    TT logit at PT's token ({pt_argmax_token}): {tt_prefill_logits[pt_argmax_token].item():.4f}")
            print(f"    TT logit at TT's token ({tt_argmax_token}): {tt_prefill_logits[tt_argmax_token].item():.4f}")
            print(
                f"    PT gap (PT_tok - TT_tok): {(pt_step_logits[0][pt_argmax_token] - pt_step_logits[0][tt_argmax_token]).item():.4f}"
                if pt_step_logits
                else ""
            )
            print(
                f"    TT gap (PT_tok - TT_tok): {(tt_prefill_logits[pt_argmax_token] - tt_prefill_logits[tt_argmax_token]).item():.4f}"
            )

            # Show top-5 for both
            pt_top5 = pt_step_logits[0].topk(5) if pt_step_logits else None
            tt_top5 = tt_prefill_logits.topk(5)
            if pt_top5:
                print(f"    PT top-5 tokens: {pt_top5.indices.tolist()}")
                print(f"    PT top-5 values: {[f'{v:.3f}' for v in pt_top5.values.tolist()]}")
            print(f"    TT top-5 tokens: {tt_top5.indices.tolist()}")
            print(f"    TT top-5 values: {[f'{v:.3f}' for v in tt_top5.values.tolist()]}")

            # ========== MULTIMODAL EMBEDDINGS COMPARISON ==========
            print("\n  --- Multimodal Embeddings Comparison (INPUT to LLM) ---")

            # PT multimodal embeddings
            with torch.no_grad():
                pt_vis = torch.cat(
                    [
                        pt_model.vision_backbone.featurizer.forward_features(img_dino)[:, 5:, :],
                        pt_model.vision_backbone.fused_featurizer.forward_features(img_siglip),
                    ],
                    dim=2,
                )
                pt_proj = pt_model.projector(pt_vis)
                pt_text_embeds = pt_model.language_model.model.embed_tokens(input_ids)
                pt_multimodal = torch.cat([pt_proj, pt_text_embeds], dim=1)

            # TT multimodal embeddings (reconstruct the same way TT does internally)
            with torch.no_grad():
                # TT vision
                tt_vis_out = tt_model.vision_backbone.bh_vision_encoder(inputs["pixel_values"])
                tt_vis_torch = ttnn.to_torch(tt_vis_out).squeeze(0)  # Remove batch dim if needed

                # TT projector (use .forward() since TTNNPrismaticProjector is not directly callable)
                tt_proj_torch = ttnn.to_torch(tt_model.ttnn_projector.forward(tt_vis_out)).squeeze(0)

                # TT text embeddings - need to get from TT language model's embedding layer
                # Use PT text embeddings for now as they should be identical
                tt_text_embeds = pt_text_embeds  # Same tokenization

                # TT would concat: [proj, text_embeds]
                tt_multimodal_approx = torch.cat([tt_proj_torch, tt_text_embeds], dim=1)

            seq_len = pt_multimodal.shape[1]

            print(f"    PT multimodal shape: {list(pt_multimodal.shape)}")
            print(f"    TT multimodal shape (approx): {list(tt_multimodal_approx.shape)}")

            # Compare multimodal embeddings
            pt_mm_flat = pt_multimodal.view(-1).float()
            tt_mm_flat = tt_multimodal_approx.view(-1)[: pt_mm_flat.shape[0]].float()
            mm_pcc = _compute_pcc(pt_mm_flat, tt_mm_flat)

            print(f"    PT multimodal: mean={pt_multimodal.float().mean():.6f}, std={pt_multimodal.float().std():.6f}")
            print(
                f"    TT multimodal: mean={tt_multimodal_approx.float().mean():.6f}, std={tt_multimodal_approx.float().std():.6f}"
            )
            print(f"    MULTIMODAL EMBEDDINGS PCC: {mm_pcc:.6f}")

            if mm_pcc > 0.95:
                print(f"    ✓ Multimodal embeddings are correct → Issue is in TT LLM itself")
            elif mm_pcc > 0.8:
                print(f"    ⚠ Multimodal embeddings have some divergence")
            else:
                print(f"    ✗ Multimodal embeddings diverge significantly → Issue is before LLM")

            # ========== PT LLM: Get step-by-step logits ==========
            print("\n  --- PT LLM (step-by-step with teacher forcing) ---")

            pt_step_logits = []
            with torch.no_grad():
                # Step 0: Prefill
                pt_out = pt_model.language_model.model(
                    inputs_embeds=pt_multimodal,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = pt_out.past_key_values
                pt_lm_head = pt_model.language_model.lm_head

                step0_logits = pt_lm_head(pt_out.last_hidden_state[:, -1:, :]).squeeze(1)
                pt_step_logits.append(step0_logits)
                pt_tok0 = step0_logits.argmax().item()
                print(f"  PT Prefill: argmax={pt_tok0}, mean={step0_logits.float().mean():.4f}")
                # Compare PT logits at key tokens
                print(
                    f"  PT logit@31874={step0_logits[0, 31874].item():.4f}, logit@31852={step0_logits[0, 31852].item():.4f}"
                )
                print(f"  PT gap (31874 - 31852) = {step0_logits[0, 31874].item() - step0_logits[0, 31852].item():.4f}")

                # Steps 1-6: Decode with teacher forcing (feed PT's tokens)
                for i in range(6):
                    tok = torch.tensor([[pt_action_tokens[i]]], dtype=torch.long)
                    tok_embed = pt_model.language_model.model.embed_tokens(tok)
                    pt_out = pt_model.language_model.model(
                        inputs_embeds=tok_embed,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = pt_out.past_key_values
                    step_logits = pt_lm_head(pt_out.last_hidden_state[:, -1:, :]).squeeze(1)
                    pt_step_logits.append(step_logits)

            # ========== Compare Prefill ==========
            prefill_row_used = (input_ids.shape[1] - 1) % 32
            print(f"\n  --- Step 0 (Prefill) Comparison ---")
            print(f"    PT: argmax={pt_tok0}, mean={pt_step_logits[0].float().mean():.4f}")
            print(f"    TT (row {prefill_row_used}): argmax={tt_tok0}, mean={tt_prefill_logits.mean():.4f}")

            # ========== PREFILL PCC ==========
            print(f"\n  --- Prefill Logits Comparison ---")
            pt_flat = pt_step_logits[0].view(-1)[:32064].float()
            tt_flat = tt_prefill_logits.view(-1)[:32064].float()
            pcc_step0 = _compute_pcc(pt_flat, tt_flat)

            print(f"    PT: argmax={pt_tok0}, mean={pt_flat.mean():.4f}, std={pt_flat.std():.4f}")
            print(f"    TT: argmax={tt_tok0}, mean={tt_flat.mean():.4f}, std={tt_flat.std():.4f}")
            print(f"    PREFILL PCC: {pcc_step0:.6f}")

            # Check if the prefill output is already flattened or packed
            print(f"\n    Raw TT prefill shape: {list(tt_prefill_full.shape)}")
            if tt_prefill_full.numel() > 32064:
                print(f"    Note: TT prefill output may be packed, checking rows...")
                try:
                    tt_32rows = tt_prefill_full.view(-1)[: 32 * 32064].view(32, 32064)
                    for r in [0, 22, 23]:
                        pcc_r = _compute_pcc(pt_flat, tt_32rows[r])
                        print(f"    Row {r} PCC: {pcc_r:.6f}, argmax={int(tt_32rows[r].argmax())}")
                except:
                    print(f"    Could not reshape to 32 rows")
            else:
                print(f"    Prefill output is already extracted to single token (not packed)")

            # ========== Decode with teacher forcing ==========
            print(f"\n  --- Decode Steps (Teacher Forcing) ---")
            print(f"  seq_len={seq_len}, feeding PT tokens to TT decode")

            tt_step_logits = [tt_prefill_logits]
            current_pos = torch.tensor([seq_len])

            for i in range(6):
                teacher_tok = torch.tensor([[pt_action_tokens[i]]], dtype=torch.long)

                tt_logits, _ = tt_model.language_model.generator.decode_forward_text(
                    teacher_tok,
                    current_pos,
                    page_table=tt_model.language_model.page_table,
                    kv_cache=tt_model.language_model.tt_kv_cache,
                    sampling_params=None,
                    enable_trace=False,
                )
                ttnn.synchronize_device(tt_model.ttnn_device)

                # CORRECT ROW EXTRACTION: decode output at row = current_pos % 32
                decode_row = current_pos.item() % 32
                tt_logits_full = tt_logits.view(-1)

                if tt_logits_full.numel() >= 32 * 32064:
                    tt_32rows_decode = tt_logits_full[: 32 * 32064].view(32, 32064)
                    tt_logits_torch = tt_32rows_decode[decode_row].float()
                    row0_argmax = tt_32rows_decode[0].argmax().item()
                else:
                    tt_logits_torch = tt_logits_full[:32064].float()
                    row0_argmax = tt_logits_torch.argmax().item()
                    decode_row = 0  # Already flattened

                tt_step_logits.append(tt_logits_torch)

                pcc_i = _compute_pcc(pt_step_logits[i + 1].float(), tt_logits_torch)
                pt_pred = pt_step_logits[i + 1].argmax().item()
                tt_pred = tt_logits_torch.argmax().item()

                print(f"  Step {i+1}: input={pt_action_tokens[i]}, pos={current_pos.item()}")
                print(f"    Extracting from row {decode_row} (pos % 32), row0_argmax={row0_argmax}")
                print(f"    PT: argmax={pt_pred}, mean={pt_step_logits[i+1].float().mean():.4f}")
                print(f"    TT: argmax={tt_pred}, mean={tt_logits_torch.mean():.4f}")
                print(f"    PCC={pcc_i:.6f}, match={'✓' if pt_pred == tt_pred else '✗'}")

                current_pos += 1

            # ========== Summary ==========
            print(f"\n  --- Teacher Forcing Summary ---")
            all_pccs = [pcc_step0] + [
                _compute_pcc(pt_step_logits[i + 1].float(), tt_step_logits[i + 1]) for i in range(6)
            ]
            print(f"  Per-step PCCs: [{', '.join([f'{p:.4f}' for p in all_pccs])}]")
            print(f"  Mean PCC: {np.mean(all_pccs):.4f}")

            if np.mean(all_pccs) > 0.9:
                print(f"\n  ✓ HIGH PCC → TT decode is correct, issue is snowball effect")
            elif np.mean(all_pccs) > 0.7:
                print(f"\n  ⚠ MEDIUM PCC - some decode issues but close")
            else:
                print(f"\n  ✗ LOW PCC → FUNDAMENTAL TT DECODE BUG")
                print(f"  → Check: RoPE, SDPA, KV cache update, LM head")

        except Exception as e:
            import traceback

            print(f"  Teacher forcing test failed: {e}")
            traceback.print_exc()

    # ========== 7. COMPREHENSIVE DECODE DEBUG ==========
    # DISABLED: Causes L1 memory overflow due to accumulated tensors
    print("\n" + "=" * 70)
    print("[7] COMPREHENSIVE DECODE DEBUG - SKIPPED (L1 memory constraint)")
    print("=" * 70)

    if False:  # Disabled
        seq_len = 279  # From prefill

        # ============================================
        # DEBUG A: LLM WEIGHTS COMPARISON (PT vs TT)
        # ============================================
        print("\n  --- DEBUG A: LLM Weights Comparison ---")

        # Check embedding weights
        pt_embed_weight = pt_model.language_model.model.embed_tokens.weight.data
        tt_transformer = tt_model.language_model.generator.model[0]
        tt_embed_weight = ttnn.to_torch(tt_transformer.embd.weights)

        # Reshape TT weight if needed
        tt_embed_flat = tt_embed_weight.view(-1)[: pt_embed_weight.numel()].view(pt_embed_weight.shape)
        pcc_embed_weight = _compute_pcc(pt_embed_weight.float(), tt_embed_flat.float())
        print(f"    Embedding weights: PT shape={list(pt_embed_weight.shape)}, TT shape={list(tt_embed_weight.shape)}")
        print(f"    Embedding weights PCC: {pcc_embed_weight:.6f}")

        # Check LM head weights
        pt_lm_head = pt_model.language_model.lm_head.weight.data
        # TT LMHead uses output_weights list (may be split across devices)
        if hasattr(tt_transformer.lm_head, "output_weights") and len(tt_transformer.lm_head.output_weights) > 0:
            tt_lm_head = ttnn.to_torch(tt_transformer.lm_head.output_weights[0])
            # LM head in TT is transposed [hidden, vocab] vs PT [vocab, hidden]
            print(f"    LM head weights: PT shape={list(pt_lm_head.shape)}, TT shape={list(tt_lm_head.shape)}")
            # Just check that TT LM head has reasonable stats (can't directly compare due to transpose/split)
            print(f"    LM head PT: mean={pt_lm_head.float().mean():.6f}, std={pt_lm_head.float().std():.6f}")
            print(f"    LM head TT: mean={tt_lm_head.float().mean():.6f}, std={tt_lm_head.float().std():.6f}")
        else:
            print(f"    LM head: Could not access TT weights")

        # Check layer 0 attention weights (wqkv)
        pt_wq = pt_model.language_model.model.layers[0].self_attn.q_proj.weight.data
        pt_wk = pt_model.language_model.model.layers[0].self_attn.k_proj.weight.data
        pt_wv = pt_model.language_model.model.layers[0].self_attn.v_proj.weight.data
        print(
            f"    PT Layer0 Q weight: shape={list(pt_wq.shape)}, mean={pt_wq.float().mean():.6f}, std={pt_wq.float().std():.6f}"
        )
        print(f"    PT Layer0 K weight: shape={list(pt_wk.shape)}, mean={pt_wk.float().mean():.6f}")
        print(f"    PT Layer0 V weight: shape={list(pt_wv.shape)}, mean={pt_wv.float().mean():.6f}")

        # ============================================
        # DEBUG B: Q ROW INDEXING CHECK
        # ============================================
        print("\n  --- DEBUG B: Q Row Indexing Check ---")
        print("    Testing if Q is read from correct row (pos % 32)")

        # Fresh prefill
        tt_model.language_model.num_actions = 0
        _ = tt_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        test_token = torch.tensor([[31874]], dtype=torch.long)
        current_pos = torch.tensor([seq_len])  # pos=279

        # Get the transformer model
        model = tt_model.language_model.generator.model[0]

        # Prepare inputs manually to inspect Q
        tokens_tt, current_pos_tt, rope_idxs, page_table_tt = model.prepare_decode_inputs_host(
            test_token, current_pos, tt_model.language_model.page_table
        )

        # Move to device
        tokens_tt = ttnn.to_device(tokens_tt, mesh_device)
        current_pos_tt = ttnn.to_device(current_pos_tt, mesh_device)
        rope_idxs = ttnn.to_device(rope_idxs, mesh_device)
        if page_table_tt is not None:
            page_table_tt = ttnn.to_device(page_table_tt, mesh_device)

        # Get embeddings
        tt_embed = model.embd(tokens_tt)
        tt_embed = ttnn.unsqueeze_to_4D(tt_embed)
        tt_embed = ttnn.to_memory_config(tt_embed, model.args.model_config["DECODE_RESIDUAL_MEMCFG"])

        # Get embedding as torch for inspection
        embed_torch = ttnn.to_torch(tt_embed)
        print(f"    Input embedding shape: {list(embed_torch.shape)}")
        print(f"    Input embedding mean: {embed_torch.float().mean():.6f}")
        print(f"    Expected Q row index: {seq_len % 32} (pos={seq_len} % 32)")

        # Check which row has non-zero values
        embed_flat = embed_torch.view(embed_torch.shape[-2], -1)
        row_norms = [embed_flat[i].float().norm().item() for i in range(min(32, embed_flat.shape[0]))]
        nonzero_rows = [i for i, n in enumerate(row_norms) if n > 0.01]
        print(f"    Non-zero rows in embedding: {nonzero_rows}")
        print(f"    Row norms (first 8): {[f'{n:.4f}' for n in row_norms[:8]]}")

        # ============================================
        # DEBUG C: KV CACHE LAYOUT CHECK
        # ============================================
        print("\n  --- DEBUG C: KV Cache Layout Check ---")

        kv_cache = tt_model.language_model.tt_kv_cache
        if kv_cache:
            k_cache = kv_cache[0][0]  # Layer 0 K cache
            v_cache = kv_cache[0][1]  # Layer 0 V cache

            # Handle multi-device case (list of tensors)
            if isinstance(k_cache, list):
                k_cache = k_cache[0]
                v_cache = v_cache[0]

            k_torch = ttnn.to_torch(k_cache)
            v_torch = ttnn.to_torch(v_cache)

            print(f"    K cache shape: {list(k_torch.shape)}")
            print(f"    V cache shape: {list(v_torch.shape)}")
            print(f"    Expected layout: [num_blocks, n_kv_heads, block_size, head_dim]")
            print(f"                 or: [batch, n_kv_heads, max_seq_len, head_dim]")

            # Check if cache has data after prefill
            k_nonzero = (k_torch.abs() > 1e-6).sum().item()
            v_nonzero = (v_torch.abs() > 1e-6).sum().item()
            k_total = k_torch.numel()
            print(f"    K cache non-zero: {k_nonzero}/{k_total} ({100*k_nonzero/k_total:.2f}%)")
            print(f"    V cache non-zero: {v_nonzero}/{k_total} ({100*v_nonzero/k_total:.2f}%)")

            # Check the first few positions
            print(f"    K cache [0,0,:5,:4] (first 5 positions, first 4 dims):")
            print(f"      {k_torch[0,0,:5,:4].float()}")
        else:
            print(f"    ✗ KV cache is None!")

        # ============================================
        # DEBUG D: PAGE TABLE CHECK
        # ============================================
        print("\n  --- DEBUG D: Page Table Check ---")

        page_table = tt_model.language_model.page_table
        if page_table is not None:
            pt_torch = ttnn.to_torch(page_table)
            print(f"    Page table shape: {list(pt_torch.shape)}")
            print(f"    Page table dtype: {pt_torch.dtype}")
            print(f"    Page table first 10 entries: {pt_torch.view(-1)[:10].tolist()}")
            print(f"    Page table unique values: {torch.unique(pt_torch).tolist()[:20]}")

            # Check if sequential
            pt_flat = pt_torch.view(-1)
            is_sequential = all(pt_flat[i] == i for i in range(min(10, len(pt_flat))))
            print(f"    Is sequential (0,1,2,...): {is_sequential}")

            # Calculate expected blocks for seq_len
            block_size = 64  # Typical block size
            expected_blocks = (seq_len + block_size - 1) // block_size
            print(f"    Expected blocks for seq_len={seq_len}: {expected_blocks} (block_size={block_size})")
        else:
            print(f"    Page table is None (non-paged KV cache)")

        # ============================================
        # DEBUG E: STEP-BY-STEP FORWARD COMPARISON
        # ============================================
        print("\n  --- DEBUG E: Step-by-Step Forward Comparison ---")
        print("    Running PT and TT with SAME input, comparing at each stage")

        # Fresh prefill for TT
        tt_model.language_model.num_actions = 0
        _ = tt_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        test_tok = torch.tensor([[31874]], dtype=torch.long)

        # PT: Get embedding and run through layer 0
        with torch.no_grad():
            pt_embed = pt_model.language_model.model.embed_tokens(test_tok)
            print(f"    PT embedding: shape={list(pt_embed.shape)}, mean={pt_embed.float().mean():.6f}")

            # Run PT full decode forward to get the actual logits
            # We need past_key_values from prefill first
            pt_prefill_out = pt_model(
                input_ids=input_ids,
                pixel_values=inputs["pixel_values"],
                attention_mask=attention_mask,
                use_cache=True,
            )
            pt_past_kv = pt_prefill_out.past_key_values

            # Now decode with the test token
            pt_decode_out = pt_model.language_model(
                input_ids=test_tok,
                past_key_values=pt_past_kv,
                use_cache=True,
            )
            pt_decode_logits = pt_decode_out.logits[0, -1, :].float()
            pt_decode_argmax = pt_decode_logits.argmax().item()
            pt_top5_vals, pt_top5_idx = torch.topk(pt_decode_logits, 5)

            print(f"    PT decode logits: argmax={pt_decode_argmax}, mean={pt_decode_logits.mean():.4f}")
            print(f"    PT decode top-5: {[idx.item() for idx in pt_top5_idx]}")
            print(f"    PT decode top-5 vals: {[f'{v:.3f}' for v in pt_top5_vals]}")

        # TT: Decode with same token
        tt_decode_logits, _ = tt_model.language_model.generator.decode_forward_text(
            test_tok,
            torch.tensor([seq_len]),
            page_table=tt_model.language_model.page_table,
            kv_cache=tt_model.language_model.tt_kv_cache,
            sampling_params=None,
            enable_trace=False,
        )
        ttnn.synchronize_device(tt_model.ttnn_device)
        tt_decode_logits_torch = tt_decode_logits.view(-1)[:32064].float()
        tt_decode_argmax = tt_decode_logits_torch.argmax().item()
        tt_top5_vals, tt_top5_idx = torch.topk(tt_decode_logits_torch, 5)

        print(f"    TT decode logits: argmax={tt_decode_argmax}, mean={tt_decode_logits_torch.mean():.4f}")
        print(f"    TT decode top-5: {[idx.item() for idx in tt_top5_idx]}")
        print(f"    TT decode top-5 vals: {[f'{v:.3f}' for v in tt_top5_vals]}")

        # Compare
        pcc_decode = _compute_pcc(pt_decode_logits, tt_decode_logits_torch)
        print(f"\n    DECODE LOGITS PCC: {pcc_decode:.6f}")

        if pcc_decode < 0.7:
            print(f"    ✗ LOW DECODE PCC - Fundamental issue in decode path")
        elif pcc_decode < 0.9:
            print(f"    ⚠ Medium decode PCC - Some divergence")
        else:
            print(f"    ✓ High decode PCC - Decode path looks OK")

        # Check specific token logits
        print(f"\n    Token 31874 logit: PT={pt_decode_logits[31874]:.4f}, TT={tt_decode_logits_torch[31874]:.4f}")
        print(f"    Token 31852 logit: PT={pt_decode_logits[31852]:.4f}, TT={tt_decode_logits_torch[31852]:.4f}")
        print(f"    PT prefers 31874 by: {pt_decode_logits[31874] - pt_decode_logits[31852]:.4f}")
        print(f"    TT prefers 31852 by: {tt_decode_logits_torch[31852] - tt_decode_logits_torch[31874]:.4f}")

        # ============================================
        # DEBUG F: DIFFERENT TOKENS SENSITIVITY
        # ============================================
        print("\n  --- DEBUG F: Token Sensitivity Check ---")

        # Fresh prefill
        tt_model.language_model.num_actions = 0
        _ = tt_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        # Test with token 31874
        logits_a, _ = tt_model.language_model.generator.decode_forward_text(
            torch.tensor([[31874]], dtype=torch.long),
            torch.tensor([seq_len]),
            page_table=tt_model.language_model.page_table,
            kv_cache=tt_model.language_model.tt_kv_cache,
            sampling_params=None,
            enable_trace=False,
        )
        ttnn.synchronize_device(tt_model.ttnn_device)
        logits_a_torch = logits_a.view(-1)[:32064].float()

        # Fresh prefill again
        tt_model.language_model.num_actions = 0
        _ = tt_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        # Test with token 31852
        logits_b, _ = tt_model.language_model.generator.decode_forward_text(
            torch.tensor([[31852]], dtype=torch.long),
            torch.tensor([seq_len]),
            page_table=tt_model.language_model.page_table,
            kv_cache=tt_model.language_model.tt_kv_cache,
            sampling_params=None,
            enable_trace=False,
        )
        ttnn.synchronize_device(tt_model.ttnn_device)
        logits_b_torch = logits_b.view(-1)[:32064].float()

        pcc_sensitivity = _compute_pcc(logits_a_torch, logits_b_torch)
        print(f"    Token 31874 → argmax={logits_a_torch.argmax().item()}, mean={logits_a_torch.mean():.4f}")
        print(f"    Token 31852 → argmax={logits_b_torch.argmax().item()}, mean={logits_b_torch.mean():.4f}")
        print(f"    PCC between different tokens: {pcc_sensitivity:.6f}")

        if pcc_sensitivity > 0.99:
            print(f"    ✗ CRITICAL: TT IGNORES INPUT TOKEN!")
            print(f"    → Bug is in: token embedding, Q projection, or attention mask")
        elif pcc_sensitivity > 0.95:
            print(f"    ⚠ Very weak token sensitivity")
        else:
            print(f"    ✓ TT responds to different tokens")

        # ============================================
        # SUMMARY
        # ============================================
        print("\n" + "=" * 70)
        print("DECODE DEBUG SUMMARY")
        print("=" * 70)
        print(f"  Embedding weights PCC:    {pcc_embed_weight:.6f} {'✓' if pcc_embed_weight > 0.99 else '✗'}")
        print(f"  LM head weights PCC:      {pcc_lm_head:.6f} {'✓' if pcc_lm_head > 0.99 else '✗'}")
        print(f"  Decode logits PCC:        {pcc_decode:.6f} {'✓' if pcc_decode > 0.9 else '✗'}")
        print(
            f"  Token sensitivity PCC:    {pcc_sensitivity:.6f} {'✗ IGNORES TOKEN' if pcc_sensitivity > 0.99 else '✓'}"
        )

        if pcc_embed_weight < 0.99:
            print(f"\n  → ISSUE: Embedding weights don't match!")
        if pcc_lm_head < 0.99:
            print(f"\n  → ISSUE: LM head weights don't match!")
        if pcc_sensitivity > 0.99:
            print(f"\n  → ISSUE: Decode ignores input token - check Q row indexing!")

    # ========== 8. TIMING TEST (100 iterations) ==========
    print("\n" + "-" * 50)
    print("[8] TIMING TEST (100 iterations - vision + projector)")
    print("-" * 50)

    import time as time_module

    # Warmup
    for _ in range(3):
        _ = tt_model.vision_backbone(pixel_values)

    # Time vision + projector
    n_iters = 100
    start_time = time_module.perf_counter()
    for _ in range(n_iters):
        tt_vis = tt_model.vision_backbone(pixel_values)
        tt_proj_in = ttnn.typecast(tt_vis, ttnn.bfloat16)
        _ = tt_model.ttnn_projector.forward(tt_proj_in)
    end_time = time_module.perf_counter()

    total_ms = (end_time - start_time) * 1000
    per_iter_ms = total_ms / n_iters
    print(f"  Total time for {n_iters} iterations: {total_ms:.2f} ms")
    print(f"  Per iteration: {per_iter_ms:.2f} ms")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  DinoV2 PCC:       {pcc_dino:.6f}" if pcc_dino is not None else "  DinoV2 PCC:       N/A")
    print(f"  SigLIP PCC:       {pcc_siglip:.6f}" if pcc_siglip is not None else "  SigLIP PCC:       N/A")
    print(f"  Vision PCC:       {pcc_vision:.6f}" if pcc_vision is not None else "  Vision PCC:       N/A")
    print(f"  Projector PCC:    {pcc_proj:.6f}")
    tokens_match = pt_action_tokens == tt_action_tokens if tt_action_tokens else False
    print(f"  Tokens match:     {tokens_match}")
    print(f"  PT action tokens: {pt_action_tokens}")
    print(f"  TT action tokens: {tt_action_tokens}")

    # Warn if issues
    if pcc_dino is not None and pcc_dino < 0.95:
        print(f"  ⚠️ WARNING: DinoV2 PCC is low: {pcc_dino}")
    if pcc_siglip is not None and pcc_siglip < 0.95:
        print(f"  ⚠️ WARNING: SigLIP PCC is low: {pcc_siglip}")
    if pcc_vision is not None and pcc_vision < 0.95:
        print(f"  ⚠️ WARNING: Vision PCC is low: {pcc_vision}")
    if pcc_proj < 0.95:
        print(f"  ⚠️ WARNING: Projector PCC is low: {pcc_proj}")
    if not tokens_match:
        print(f"  ⚠️ WARNING: Generated tokens do not match!")
