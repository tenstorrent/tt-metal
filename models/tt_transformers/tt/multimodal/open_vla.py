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

import logging
import os
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
    num_blocks_in_seq,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator
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

        def _set_params_from_dict(self, config, is_hf=False):
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

            return super()._set_params_from_dict(
                new_config,
                is_hf,
            )

        def load_state_dict(self):
            if state_dict is None:
                return super().load_state_dict()
            new_state_dict = map_openvla_hf_to_meta_keys(state_dict)
            return new_state_dict

    return LLama2OpenVLAArgs


class OpenVLALanguageModel(GenerationMixin):
    def __init__(self, device, local_state_dict=None):
        self.generator_args_config = {
            "num_devices": device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1,
            "data_parallel": 1,
            "mesh_device": device,
            "instruct": False,
            "global_batch_size": 1,
            "optimizations": None,
            "max_seq_len": 1024,
            "page_params": {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
            "paged_attention": True,
        }
        self.model_args, self.model, self.page_table, self.tt_kv_cache, self.tokenizer = prepare_generator_args(
            **self.generator_args_config,
            model_factory_fn=lambda *args, **kwargs: create_tt_model(
                *args, **kwargs, ModelArgsClass=get_LLama2OpenVLAArgs(local_state_dict)
            ),
        )
        self.generator = Generator(self.model, self.model_args, device, self.tokenizer)

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
            logits = self.generator.decode_forward_text(
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
        # Make width multiple of 128
        if inputs_embeds.shape[2] % 128 != 0:
            padding = int(np.ceil(inputs_embeds.shape[2] / 128) * 128) - inputs_embeds.shape[2]
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
        device = inputs_embeds.device()
        # TODO: Find a workaround for ttnn.aggregate_tensor bug
        inputs_embeds = ttnn.aggregate_tensor(inputs_embeds, ttnn.ConcatMeshToTensor(device, dim=3))[
            :, :, :, : inputs_embeds.shape[-1]
        ]
        inputs_embeds = ttnn.distribute_tensor(
            inputs_embeds,
            mapper=ttnn.ShardTensorToMesh(device, dim=3),
            mesh_device=device,
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

        # Since we give unpadded_seq_len, only the tile containing the last token is returned
        output_logits = self.model[0].process_output_prefill(tt_logits, last_token_idx=(last_token_idx % 32))

        return output_logits.unsqueeze(0).unsqueeze(0)


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
        if self.ttnn_device is not None:
            self.ttnn_featurizer = ttnn_optimized_vit_highres.dinov2_encoder(self.featurizer, self.ttnn_device)

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
            if self.ttnn_device is not None:
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
                self.ttnn_fused_featurizer = lambda x2: ttnn_featurizer(
                    embedding=lambda x: ttnn_optimized_vit_highres.siglip_patch_embeddings(
                        x,
                        parameters=self.featurize_parameters_2.patch_embed.patch_embeddings,
                    ),
                    encoder=lambda x: ttnn_optimized_vit_highres.siglip_encoder(
                        x, self.head_masks_2, parameters=self.featurize_parameters_2.blocks
                    ),
                    pixel=x2,
                )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        if self.ttnn_device is None:
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        else:
            img, img_fused = pixel_values
            patches, patches_fused = self.ttnn_featurizer(img)[:, 5:, :], self.ttnn_fused_featurizer(img_fused)

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

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
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
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
            ttnn_device,
            local_state_dict=local_state_dict,
        )

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

        # Instantiate LLM Backbone
        if ttnn_device is not None:
            self.language_model = OpenVLALanguageModel(ttnn_device, local_state_dict=local_state_dict)
        else:
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config, attn_implementation=config._attn_implementation
            )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.ttnn_device = ttnn_device
        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

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
                ttnn_patch_features = self.vision_backbone(pixel_values)
                projected_patch_embeddings = self.ttnn_projector.forward(ttnn_patch_features)
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
                ttnn_input_ids = ttnn.from_torch(
                    input_ids, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.ttnn_device
                )
                input_embeddings = self.get_input_embeddings()(ttnn_input_ids)
                input_embeddings = ttnn.aggregate_tensor(
                    input_embeddings, ttnn.ConcatMeshToTensor(self.ttnn_device, dim=2)
                )
                input_embeddings = ttnn.distribute_tensor(
                    input_embeddings,
                    mapper=ttnn.replicate_tensor_to_mesh_mapper(self.ttnn_device),
                    mesh_device=self.ttnn_device,
                )

            else:
                input_embeddings = self.get_input_embeddings()(input_ids)

            if self.ttnn_device is not None:
                multimodal_embeddings = ttnn.concat(
                    [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
                )
                multimodal_embeddings = ttnn.unsqueeze(multimodal_embeddings, dim=0)
                multimodal_embeddings = ttnn.to_layout(multimodal_embeddings, layout=ttnn.TILE_LAYOUT)
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
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output
        if self.ttnn_device is not None:
            return PrismaticCausalLMOutputWithPast(
                loss=None,
                logits=language_model_output,
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

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
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


class TTOpenVLAForActionPrediction(PrismaticForConditionalGeneration):
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
    predicted_text = language_model.predict_text([prompt])
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
def test_openvla_model(mesh_device):
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
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(action)
