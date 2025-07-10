# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Union

import PIL
import torch
from llama_models.llama3.api.chat_format import create_vision_mask
from tqdm import tqdm
from vllm.inputs import INPUT_REGISTRY, DecoderOnlyInputs, EncoderDecoderInputs, InputContext, TokenInputs, token_inputs
from vllm.model_executor.models.interfaces import SupportsMultiModal

import ttnn
from models.tt_transformers.demo.simple_vision_demo import create_multimodal_model
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from models.utility_functions import nearest_32


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
    return EncoderDecoderInputs(
        encoder=token_inputs(
            prompt_token_ids=[MLLAMA_IMAGE_TOKEN_ID] * num_vision_tokens,
            prompt=MLLAMA_IMAGE_TOKEN * num_vision_tokens,
            multi_modal_data=multi_modal_data,
        ),
        decoder=dec_inputs,
    )


def input_processor_for_llama_text(ctx: InputContext, inputs: Union[DecoderOnlyInputs, EncoderDecoderInputs]):
    hf_model_name = ctx.model_config.hf_config._name_or_path
    if ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name) and os.environ.get("MESH_DEVICE") == "N150":
        prompt_len = len(inputs.get("prompt_token_ids"))
        MAX_PROMPT_LEN = 65536
        if prompt_len > MAX_PROMPT_LEN:
            raise ValueError(
                f"TT-LLama8B and TT-Llama11B do not support prompts longer than {MAX_PROMPT_LEN} tokens on N150 (received prompt with {prompt_len} tokens)"
            )
    return inputs


# @MULTIMODAL_REGISTRY.register_image_input_mapper()  # TODO: Add once model can accept inputs from multi_modal_input_mapper (raw pixel values)
@INPUT_REGISTRY.register_input_processor(input_processor_for_mllama)
class MllamaForConditionalGeneration(Generator, SupportsMultiModal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MLLAMA_IMAGE_TOKEN_ID = 128256
        self.max_gen_len = self.model_args[0].max_seq_len - 1  # TODO: double check what this should be

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=131072, tt_data_parallel=1):
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


@INPUT_REGISTRY.register_input_processor(input_processor_for_llama_text)
class LlamaForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=131072, n_layers=None, tt_data_parallel=1
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


class QwenForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=131072, n_layers=None, tt_data_parallel=1
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
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=32768, n_layers=None, tt_data_parallel=1
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
