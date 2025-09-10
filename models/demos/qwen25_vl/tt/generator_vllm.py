# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import torch
from loguru import logger
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as Ref_Qwen2_5_VLForConditionalGeneration,
)
from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal

import ttnn
from models.demos.qwen25_vl.tt.common import merge_vision_tokens, multimodal_rope_from_hf, preprocess_inputs_prefill
from models.demos.qwen25_vl.tt.generator import Generator as QwenVLGenerator
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.generator_vllm import input_processor_for_multimodal
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: Transformer, model_args: ModelArgs, tt_cache_path):
    for layer_idx in range(num_layers):
        cache_k = torch.zeros(kv_cache_shape, dtype=dtype)
        cache_v = torch.zeros(kv_cache_shape, dtype=dtype)

        model.layers[layer_idx].attention.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                device=model.mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=model_args.model_config["ATTN_W_LAYOUT_TILE"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                cache_file_name=f"{tt_cache_path}/kvcache_{k_or_v.shape}",
            )
            for k_or_v in [cache_k, cache_v]
        ]

    return [l.attention.layer_past for l in model.layers]


def get_platform_specific_optimizations(model_name):
    is_72B = "72B" in model_name
    max_seq_len = 4096 if is_72B else 12288

    performance_opt = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    return performance_opt, max_seq_len


def initialize_vllm_text_transformer(
    hf_config,
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat8_b,
    optimizations=None,
):
    tt_model_args = ModelArgs(
        mesh_device,
        instruct=("Instruct" in hf_config.name_or_path),
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    assert tt_model_args.model_name.replace("-", "").endswith(
        hf_config.name_or_path.split("/")[-1].replace("-", "")
    ), f"The model specified in vLLM ({hf_config.name_or_path}) does not match the model name ({tt_model_args.model_name}) with model weights ({tt_model_args.CKPT_DIR})."
    state_dict = tt_model_args.load_state_dict()

    model = Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        use_paged_kv_cache=True,  # [INFO] use paged kv cache provided by this generator
    )

    return tt_model_args, model


class CustomNamespace(SimpleNamespace):
    def __contains__(self, key):
        return key in self.__dict__


@INPUT_REGISTRY.register_input_processor(input_processor_for_multimodal)
class Qwen2_5_VLForConditionalGeneration(QwenVLGenerator, SupportsMultiModal):
    def __init__(self, *args, **kwargs):
        self.reference_model = kwargs.pop("reference_model", None)
        self.visual_model = kwargs.pop("visual_model", None)
        assert (
            self.reference_model is not None and self.visual_model is not None
        ), "Reference model and visual model must be provided for vLLM"

        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1):
        optimizations, max_seq_len_native = get_platform_specific_optimizations(hf_config.name_or_path)
        if max_seq_len > max_seq_len_native:
            logger.warning(
                f"max_seq_len {max_seq_len} is not supported for {hf_config.name_or_path}, using {max_seq_len_native} instead"
            )
            max_seq_len = max_seq_len_native
        model_args, model = initialize_vllm_text_transformer(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat8_b,
            optimizations=optimizations,
        )

        ref_model_name = model_args.CKPT_DIR  # allows for local model loading as well
        config = Ref_Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(ref_model_name)
        # config.vision_config.depth = 1 # [INFO] useful for debugging
        reference_model = Ref_Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ref_model_name, config=config, torch_dtype="auto", device_map="auto"
        )
        # Create the TorchVisionTransformer wrapper using the original vision model as reference
        vision_model_args = VisionModelArgs(
            mesh_device,
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            optimizations=optimizations,
        )
        vision_model_args.hf_config.vision_config.depth = config.vision_config.depth
        visual_model = DropInVisionTransformer(reference_model.visual, vision_model_args)

        return cls(
            model,
            model_args,
            mesh_device,
            tokenizer=model_args.tokenizer,
            reference_model=reference_model,
            visual_model=visual_model,
        )

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(
            *args, **kwargs, model=self.model, model_args=self.model_args, tt_cache_path=self.cache_path
        )

    def prefill_forward(
        self,
        tokens,
        images,
        page_table,
        kv_cache,
        prompt_lens,  # [INFO] prompt_lens is pre-padding number of tokens after text-image processing
    ):
        # [INFO] tokens are padded to the same length by appending 0s; change the padding to use pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        padded_seq_len = tokens.shape[-1]
        for i in range(tokens.shape[0]):  # for each user, fix their padding
            tokens[i][prompt_lens[i] :] = pad_token_id

        # reconstruct the inputs that Qwen2.5-VL expects
        inputs = CustomNamespace()
        inputs.input_ids = tokens.to(images[0].attention_mask.dtype)
        inputs.attention_mask = torch.concat(
            [
                torch.nn.functional.pad(im.attention_mask, (0, padded_seq_len - im.attention_mask.shape[-1]), value=0)
                for im in images
            ],
            dim=0,
        )
        if "pixel_values" in images[0]:
            # we currently do not support mixed inputs of text-only users and text-image users; hence checking images[0] is enough
            inputs.pixel_values = torch.concat([im.pixel_values for im in images], dim=0)
            inputs.image_grid_thw = torch.concat([im.image_grid_thw for im in images], dim=0)
            # Vision prefill
            image_embeds = self.visual_model(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
        else:
            # text-only users
            image_embeds = torch.tensor([], dtype=torch.bfloat16)

        # Prepare text + vision inputs for decoder model
        text_embeds = self.reference_model.model.language_model.embed_tokens(inputs.input_ids)
        input_embeds = merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, self.reference_model.config)
        (
            input_prefill_pt,
            decoding_pos,  # Position where decoding should start for each user
            _prefill_lens,  # [INFO] _prefill_lens is post-padding number of tokens after text-image processing
        ) = preprocess_inputs_prefill(
            input_embeds,
            self.model_args,
            inputs.attention_mask,
            pad_embedding=self.reference_model.model.language_model.embed_tokens(torch.tensor(pad_token_id)),
        )
        # Get user-specific rotary position embeddings
        cos, sin = multimodal_rope_from_hf(
            inputs, input_embeds, self.reference_model, self.model_args, pad_token_id=pad_token_id
        )
        rot_mats = (cos, sin)

        logits = self.prefill_forward_text(
            input_prefill_pt,
            rot_mats=rot_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=decoding_pos,
        )

        return logits, rot_mats

    def decode_forward(self, *args, **kwargs):
        rot_mats_list: list = kwargs.pop("rot_mats_all_users", None)
        assert rot_mats_list is not None, "rot_mats_all_users must be provided for Qwen2.5-VL"
        # [INFO] update the cos/sin matrices for the current users in the batch
        super().update_cos_sin_rows(rot_mats_list)

        return super().decode_forward_text(*args, **kwargs)
