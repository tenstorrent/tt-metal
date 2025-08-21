# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Union

import PIL
import torch
from llama_models.llama3.api.chat_format import create_vision_mask
from tqdm import tqdm
from vllm.inputs import EncoderDecoderInputs, InputContext, TokenInputs, token_inputs
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsV0Only
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from models.utility_functions import is_wormhole_b0, nearest_32


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, dp_model: List[Transformer], tt_cache_path):
    submesh_devices = [model.mesh_device for model in dp_model]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for _ in tqdm(range(num_layers), desc=f"Allocating TT kv caches for each layer (submesh {mesh_idx+1})"):
            kv_tt_i = [
                ttnn.as_tensor(
                    lp,
                    device=submesh,
                    # TODO: this could be ShardTensorToMesh, removing the need for vLLM to know about TP for num_kv_heads.
                    # Could affect other calculations which use TTCacheEngine.num_kv_heads, though.
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    cache_file_name=tt_cache_path / f"empty_cache_paged_attention{kv_cache_shape}",
                )
                for lp in (cache_kv, cache_kv)
            ]

            kv_tt.append(kv_tt_i)
        kv_cache.append(kv_tt)
    return kv_cache


def initialize_vllm_text_transformer(
    hf_config,
    tt_data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
    optimizations=DecodersPrecision.performance,
):
    submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
    # Load model args, weights
    model_args = []
    for submesh in submesh_devices:
        model_args_i = ModelArgs(
            submesh,
            instruct=(
                "Instruct" in hf_config._name_or_path or "DeepSeek-R1-Distill-Llama-70B" in hf_config._name_or_path
            ),
            max_batch_size=max_batch_size // tt_data_parallel,
            optimizations=lambda model_args: optimizations(model_args.n_layers, model_args.model_name),
            max_seq_len=max_seq_len,
        )

        assert model_args_i.model_name.replace("-", "") in hf_config._name_or_path.replace(
            "-", ""
        ), f"The model specified in vLLM ({hf_config._name_or_path}) does not match the model name ({model_args_i.model_name}) with model weights ({model_args_i.CKPT_DIR})."
        if n_layers is not None:
            model_args_i.n_layers = n_layers

        model_args.append(model_args_i)

    state_dict = model_args[0].load_state_dict()

    tt_model = []
    for i, submesh in enumerate(submesh_devices):
        tt_model_i = Transformer(
            args=model_args[i],
            mesh_device=submesh,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args[i].weight_cache_path(dtype),
            use_paged_kv_cache=True,
        )
        tt_model.append(tt_model_i)

    return tt_model, model_args


class MllamaProcessingInfo:
    """Processing information for Mllama multi-modal models."""

    def __init__(self, ctx):
        self.ctx = ctx

    def get_hf_config(self):
        """Get the HuggingFace config for the model."""
        from transformers import MllamaConfig

        return self.ctx.get_hf_config(MllamaConfig)

    def get_hf_processor(self, **kwargs):
        """Get the HuggingFace processor for the model."""
        from transformers.models.mllama.processing_mllama import MllamaProcessor

        return self.ctx.get_hf_processor(MllamaProcessor, **kwargs)

    def get_supported_mm_limits(self):
        """Get the supported multimodal limits."""
        return {"image": None}

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.ctx.tokenizer

    @property
    def model_id(self):
        return self.ctx.model_config.model


class MllamaDummyInputsBuilder:
    """Builder for dummy inputs used in profiling."""

    def __init__(self, info):
        self.info = info

    def get_dummy_text(self, mm_counts):
        """Generate dummy text with appropriate image tokens."""
        num_images = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        return image_token * num_images

    def get_dummy_mm_data(self, seq_len, mm_counts):
        """Generate dummy multimodal data for profiling."""
        import numpy as np
        from PIL import Image

        num_images = mm_counts.get("image", 0)

        # Create dummy images with reasonable size
        target_width = 1120
        target_height = 1120

        images = []
        for _ in range(num_images):
            dummy_image_array = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            images.append(Image.fromarray(dummy_image_array, "RGB"))

        return {"image": images if len(images) > 1 else (images[0] if images else None)}


class MllamaMultiModalProcessor:
    """Multi-modal processor for Mllama that handles encoder-decoder inputs."""

    def __init__(self, info, dummy_inputs, **kwargs):
        self.info = info
        self.dummy_inputs = dummy_inputs

    def apply(self, prompt, mm_data, hf_processor_mm_kwargs=None, tokenization_kwargs=None, return_mm_hashes=False):
        """Apply processing to multimodal inputs."""
        if hf_processor_mm_kwargs is None:
            hf_processor_mm_kwargs = {}
        if tokenization_kwargs is None:
            tokenization_kwargs = {}

        # Create encoder-decoder inputs in the format expected by input_processor_for_mllama
        encoder_decoder_inputs = EncoderDecoderInputs(
            encoder={
                "type": "token",
                "prompt_token_ids": [],  # Will be filled by tokenization
                "prompt": prompt if isinstance(prompt, str) else "",
                "multi_modal_data": mm_data,
            },
            decoder={
                "type": "token",
                "prompt_token_ids": [],
            },
        )

        # For now, delegate to the existing input_processor_for_mllama function
        processed = input_processor_for_mllama(ctx=self.info.ctx, inputs=encoder_decoder_inputs)

        # Convert back to the expected format
        return {
            "type": "multimodal",
            "prompt_token_ids": processed["decoder"]["prompt_token_ids"],
            "prompt": processed["decoder"]["prompt"],
            "multi_modal_data": processed["decoder"].get("multi_modal_data"),
            "encoder_prompt_token_ids": processed["encoder"]["prompt_token_ids"],
            "encoder_prompt": processed["encoder"]["prompt"],
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


# TODO: Update input processor to inherit from EncDecMultiModalProcessor as is done in vllm.model_executor.models.mllama.py
def input_processor_for_mllama(
    ctx: InputContext,
    inputs: EncoderDecoderInputs,
) -> EncoderDecoderInputs:
    """
    This was based on a previous version of vllm.model_executor.models.mllama.py::input_processor_for_mllama()
    without the additional processing for computing num_tiles (here it is fixed).
    """
    # Example input to processor:
    # {
    #     'encoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128000, 128256, 128000, 3923, 374, 279, 2262, 315, 420, 2217, 30],  # noqa: E501
    #         'prompt': '<|image|><|begin_of_text|>What is the content of this image?',  # noqa: E501
    #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
    #     },
    #     'decoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128000],
    #     },
    # }

    # Move encoder_prompt to prompt. If the user does not explicitly provide separate
    # encoder and decoder prompts, vLLM by default will treat the prompt as the encoder prompt.
    # For the block manager to allocate enough blocks and add them to the block table, the decoder prompt
    # must contain the full text prompt.
    dec_inputs = TokenInputs(**inputs["encoder"])

    if os.environ.get("MESH_DEVICE") == "N300":
        prompt_len = len(dec_inputs.get("prompt_token_ids"))
        MAX_PROMPT_LEN = 8192
        if prompt_len > MAX_PROMPT_LEN:
            raise ValueError(
                f"TT-LLama11B-Vision does not support prompts longer than {MAX_PROMPT_LEN} tokens on N300 (received prompt with {prompt_len} tokens)"
            )

    multi_modal_data = dec_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        # text-only
        return EncoderDecoderInputs(
            encoder=token_inputs([]),
            decoder=dec_inputs,
        )

    # Set encoder prompt length based on the number of vision tokens so block manager allocates enough blocks (cross block tables).
    hf_config = ctx.model_config.hf_config
    vision_config = hf_config.vision_config
    assert vision_config.image_size % 14 == 0, "chunk size should be multiple of 14"
    token_per_chunk = nearest_32(
        (vision_config.image_size // 14) ** 2 + 1
    )  # Note: we use nearest 32 while vLLM does not by default
    num_vision_tokens = (
        vision_config.max_num_tiles * token_per_chunk
    )  # Note: we use max_num_tiles while vLLM uses num_tiles by default

    # Example output from processor:
    # {
    #     'encoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128256, 128256, ..., 128256],
    #         'prompt': '<|image|><|image|>...<|image|>',
    #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
    #     },
    #     'decoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128000, 128256, 128000, 3923, 374, 279, 2262, 315, 420, 2217, 30],  # noqa: E501
    #         'prompt': '<|image|><|begin_of_text|>What is the content of this image?',  # noqa: E501
    #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
    #     },
    # }
    MLLAMA_IMAGE_TOKEN_ID = 128256
    MLLAMA_IMAGE_TOKEN = "<|image|>"

    # Create encoder inputs without multi_modal_data in token_inputs
    encoder_token_inputs = token_inputs(
        prompt_token_ids=[MLLAMA_IMAGE_TOKEN_ID] * num_vision_tokens,
        prompt=MLLAMA_IMAGE_TOKEN * num_vision_tokens,
    )

    # Add multi_modal_data separately
    encoder_inputs = {**encoder_token_inputs, "multi_modal_data": multi_modal_data}

    return EncoderDecoderInputs(
        encoder=encoder_inputs,
        decoder=dec_inputs,
    )


# def input_processor_for_multimodal(ctx: InputContext, inputs: Union[DecoderOnlyInputs, EncoderDecoderInputs]):
#     mm_processor_kwargs = getattr(ctx.model_config, "mm_processor_kwargs", None) or {}
#     input_processor = ctx.get_hf_processor(**mm_processor_kwargs)

#     if "prompt" in inputs:
#         prompt_text = inputs["prompt"]
#     else:
#         # [INFO] with current version of vLLM, in server mode, inputs["prompt"] gives KeyError; only inputs['prompt_token_ids'] is available
#         assert "prompt_token_ids" in inputs, "prompt_token_ids must be available in server mode"
#         prompt_text = input_processor.decode(inputs["prompt_token_ids"], skip_special_tokens=False)

#     multi_modal_data = inputs.get("multi_modal_data", None)

#     processed_inputs = input_processor(
#         text=prompt_text,  # [INFO] Qwen2VLProcessor handles the case where text is a string or a list of strings
#         images=multi_modal_data["image"] if multi_modal_data is not None else None,
#         videos=None,  # [INFO] videos are not supported yet
#         return_tensors="pt",
#     )

#     assert processed_inputs.input_ids.shape[0] == 1, "Only one image is processed at a time by vLLM"
#     return {
#         "type": inputs["type"],
#         "prompt_token_ids": processed_inputs.input_ids[0].tolist(),
#         "prompt": prompt_text,
#         "multi_modal_data": {"image": processed_inputs},  # [INFO] add processed_inputs
#     }


@MULTIMODAL_REGISTRY.register_processor(
    MllamaMultiModalProcessor, info=MllamaProcessingInfo, dummy_inputs=MllamaDummyInputsBuilder
)
class MllamaForConditionalGeneration(Generator, SupportsMultiModal, SupportsV0Only):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MLLAMA_IMAGE_TOKEN_ID = 128256
        self.max_gen_len = self.model_args[0].max_seq_len - 1  # TODO: double check what this should be

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1):
        from models.tt_transformers.demo.simple_vision_demo import create_multimodal_model

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None

        for submesh in submesh_devices:
            model_args_i, model_i, state_dict = create_multimodal_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                use_paged_kv_cache=True,
                checkpoint=state_dict,
            )
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    @property
    def max_cross_attn_tokens(self):
        return self.model_args[0].vision_max_num_chunks * nearest_32(self.model_args[0].vision_chunk_ntok)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        images: Union[List[PIL.Image.Image], List[List[PIL.Image.Image]]],
        page_table: torch.Tensor,
        kv_cache,
        prompt_lens,
        cross_page_table: torch.Tensor,
    ):
        """
        Replaces prefill_forward from Generator with a version that supports mask creation.
        """
        batch = tokens.shape[0]

        vision_images = []
        vision_masks = []
        total_lens = []
        for user_id in range(batch):
            image = images[user_id]
            if isinstance(image, list):
                assert len(image) == 1, "Only one image is supported for each user in the batch"
                image = image[0]
            vision_images.append([image] if image else None)
            prompt_tokens = [int(tokens[user_id, i]) for i in range(prompt_lens[user_id])]
            vision_masks.append(create_vision_mask(prompt_tokens, self.MLLAMA_IMAGE_TOKEN_ID) if image else None)
            total_lens.append(prompt_lens[user_id] + self.max_gen_len)

        return super().prefill_forward(
            vision_images,
            vision_masks,
            tokens,
            None,
            total_lens,
            prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            cross_page_table=cross_page_table,
        )

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class LlamaForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, n_layers=None, tt_data_parallel=1
    ):
        hf_model_name = hf_config._name_or_path
        if (
            ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
            and mesh_device.get_num_devices() == 1
            and is_wormhole_b0()
        ):
            MAX_PROMPT_LEN = 65536
            if max_seq_len > MAX_PROMPT_LEN:
                raise ValueError(
                    f"TT-LLama8B and TT-Llama11B do not support max_model_len greater than {MAX_PROMPT_LEN} on N150 "
                    f"(received {max_seq_len}). Set --max_model_len to {MAX_PROMPT_LEN} or lower in vLLM."
                )

        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class QwenForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, n_layers=None, tt_data_parallel=1
    ):
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class MistralForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, n_layers=None, tt_data_parallel=1
    ):
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


# @INPUT_REGISTRY.register_input_processor(input_processor_for_multimodal) # TODO: replace with MllamaMultiModalProcessor
class Gemma3ForConditionalGeneration(Generator, SupportsMultiModal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=131072, n_layers=None, tt_data_parallel=1
    ):
        from models.demos.gemma3.demo.vision_demo import create_multimodal_model

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None

        for submesh in submesh_devices:
            model_args_i, model_i, state_dict = create_multimodal_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                use_paged_kv_cache=True,
                checkpoint=state_dict,
            )
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        data = kwargs.get("images", None)
        pixel_values = [im.pixel_values if hasattr(im, "pixel_values") else None for im in data] if data else None

        return super().prefill_forward_text(
            pixel_values=pixel_values,
            **kwargs,
        )

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)
