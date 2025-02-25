# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from types import SimpleNamespace
from typing import Union

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as Ref_Qwen2_5_VLForConditionalGeneration,
)

from models.demos.qwen25_vl.tt.generator import Generator as QwenVLGenerator
from models.demos.qwen25_vl.tt.common import (
    PagedAttentionConfig,
    preprocess_inputs_prefill,
    merge_vision_tokens,
    multimodal_rope_from_hf,
)
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.model_config import ModelArgs, ModelOptimizations

from vllm.inputs import INPUT_REGISTRY, DecoderOnlyInputs, EncoderDecoderInputs, InputContext
from vllm.model_executor.models.interfaces import SupportsMultiModal


def get_platform_specific_optimizations(model_name):
    is_72B = "72B" in model_name

    optimizations = ModelOptimizations.performance if is_72B else ModelOptimizations.accuracy
    max_seq_len = 4096 if is_72B else 12288

    return optimizations, max_seq_len


def initialize_vllm_text_transformer(
    hf_config,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
    optimizations=ModelOptimizations.accuracy,
):
    tt_model_args = ModelArgs(
        mesh_device,
        instruct=("Instruct" in hf_config.name_or_path),
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    assert tt_model_args.model_name.replace("-", "").endswith(
        hf_config.name_or_path.replace("-", "")
    ), f"The model specified in vLLM ({hf_config.name_or_path}) does not match the model name ({tt_model_args.model_name}) with model weights ({tt_model_args.CKPT_DIR})."
    if n_layers is not None:
        tt_model_args.n_layers = n_layers
    state_dict = tt_model_args.load_state_dict()

    page_table = None
    paged_attention_config = None
    tt_kv_cache = None

    block_size = 32  # [INFO] block size is fixed to 32 for now
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size,
        max_num_blocks=max_seq_len * max_batch_size // block_size,
    )
    # Implied shuffling of blocks
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    # Page table which maps virtual blocks to physical
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        tt_model_args.max_batch_size, paged_attention_config.max_num_blocks // tt_model_args.max_batch_size
    )

    model = Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers]

    return tt_model_args, model, page_table, tt_kv_cache


def input_processor_for_qwen25_vl(ctx: InputContext, inputs: Union[DecoderOnlyInputs, EncoderDecoderInputs]):
    input_processor = ctx.get_hf_processor()
    if "prompt" in inputs:
        prompt_text = inputs["prompt"]
    else:
        # [INFO] with current version of vLLM, in server mode, inputs["prompt"] gives KeyError; only inputs['prompt_token_ids'] is available
        assert "prompt_token_ids" in inputs, "prompt_token_ids must be available in server mode"
        prompt_text = input_processor.decode(inputs["prompt_token_ids"], skip_special_tokens=False)
    images = inputs["multi_modal_data"]["image"]

    processed_inputs = input_processor(
        text=prompt_text,  # [INFO] Qwen2VLProcessor handles the case where text is a string or a list of strings
        images=images,
        videos=None,  # [INFO] videos are not supported yet
        return_tensors="pt",
    )

    assert processed_inputs.input_ids.shape[0] == 1, "Only one image is processed at a time by vLLM"
    return {
        "type": inputs["type"],
        "prompt_token_ids": processed_inputs.input_ids[0].tolist(),
        "prompt": prompt_text,
        "multi_modal_data": {"image": processed_inputs},  # [INFO] add processed_inputs
    }


class CustomNamespace(SimpleNamespace):
    def __contains__(self, key):
        return key in self.__dict__


@INPUT_REGISTRY.register_input_processor(input_processor_for_qwen25_vl)
class Qwen2_5_VLForConditionalGeneration(QwenVLGenerator, SupportsMultiModal):
    def __init__(self, *args, **kwargs):
        self.page_table = kwargs.pop("page_table", None)
        self.kv_cache = kwargs.pop("kv_cache", None)
        assert (
            self.page_table is not None and self.kv_cache is not None
        ), "Page table and kv cache must be provided for vLLM"
        self.reference_model = kwargs.pop("reference_model", None)
        self.visual_model = kwargs.pop("visual_model", None)
        assert (
            self.reference_model is not None and self.visual_model is not None
        ), "Reference model and visual model must be provided for vLLM"

        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, n_layers=None, n_vision_layers=None, tt_data_parallel=1
    ):
        # Enable async mode todo)) remove this when qwen2.5-vl-rebase-main is merged
        mesh_device.enable_async(True)

        optimizations, max_seq_len = get_platform_specific_optimizations(hf_config.name_or_path)
        model_args, model, page_table, kv_cache = initialize_vllm_text_transformer(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=optimizations,
        )

        config = Ref_Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(model_args.model_name)
        if n_vision_layers is not None:
            config.vision_config.depth = n_vision_layers
        reference_model = Ref_Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name, config=config, torch_dtype="auto", device_map="auto"
        )
        # Create the TorchVisionTransformer wrapper using the original vision model as reference
        vision_model_args = VisionModelArgs(
            mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=None),
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            optimizations=optimizations,
        )
        vision_model_args.hf_config.vision_config.depth = config.vision_config.depth
        visual_model = DropInVisionTransformer(
            reference_model.visual, vision_model_args, debug=False
        )  # debug=True to show PCC

        return cls(
            model,
            model_args,
            mesh_device,
            tokenizer=model_args.tokenizer,
            page_table=page_table,
            kv_cache=kv_cache,
            reference_model=reference_model,
            visual_model=visual_model,
        )

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def allocate_kv_cache(self, *args, **kwargs):
        self.zero_out_kv_cache()
        return self.kv_cache

    def zero_out_kv_cache(self):
        for layer in self.kv_cache:
            k_cache, v_cache = layer
            k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
            v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

    def prefill_forward(
        self,
        tokens,
        images,
        page_table,  # [INFO] page_table is incorrectly generated for the tokens before the image tokens were processed
        kv_cache,  # [INFO] id(kv_cache) == id(self.kv_cache) due to allocate_kv_cache returning self.kv_cache
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
        inputs.pixel_values = torch.concat([im.pixel_values for im in images], dim=0)
        inputs.attention_mask = torch.concat(
            [
                torch.nn.functional.pad(im.attention_mask, (0, padded_seq_len - im.attention_mask.shape[-1]), value=0)
                for im in images
            ],
            dim=0,
        )
        inputs.image_grid_thw = torch.concat([im.image_grid_thw for im in images], dim=0)

        # Vision prefill
        image_embeds = self.visual_model(inputs.pixel_values, grid_thw=inputs.image_grid_thw)

        # Prepare text + vision inputs for decoder model
        # FIXME: on-host embeddings - run as part of vision model prefill when merge_vision_tokens is ported to ttnn
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
        self.model.rope_setup.set_cos_sin(cos, sin)

        logits = self.prefill_forward_text(
            input_prefill_pt[0].unsqueeze(0),  # Just warmup prefill for 1 user
            page_table=self.page_table,
            kv_cache=kv_cache,
            prompt_lens=decoding_pos,
        )

        logits = self.prefill_forward_text(
            input_prefill_pt,
            page_table=self.page_table,
            kv_cache=kv_cache,
            prompt_lens=decoding_pos,
        )

        return logits

    def decode_forward(
        self,
        start_pos,
        tokens,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
    ):
        return super().decode_forward_text(
            tokens=tokens,
            start_pos=start_pos,
            page_table=self.page_table,
            kv_cache=kv_cache,
            enable_trace=False,  # [INFO] work around tracing bug
            read_from_device=read_from_device,
            argmax_on_device=False,
        )

    # todo)) remove this after qwen2.5-vl-rebase-main is merged
    def read_decode_output(self, tt_logits, unpadded_batch, is_tokens=False):
        return super().read_decode_output(tt_logits, unpadded_batch, argmax_on_device=is_tokens)
