# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Mapping, Optional, Sequence, Union

import torch
import vllm.envs as envs
from llama_models.llama3.api.chat_format import create_vision_mask
from loguru import logger
from PIL.Image import Image
from tqdm import tqdm
from transformers import BatchFeature
from vllm.model_executor.models.gemma3_mm import (
    Gemma3DummyInputsBuilder,
    Gemma3MultiModalProcessor,
    Gemma3ProcessingInfo,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsV0Only
from vllm.model_executor.models.mllama import MllamaProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalEncDecInputs,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, EncDecMultiModalProcessor, PromptUpdate
from vllm.multimodal.profiling import BaseDummyInputsBuilder

import ttnn
from models.common.utility_functions import is_wormhole_b0, nearest_32
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, TensorGroup


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, dp_model: List[Transformer], tt_cache_path):
    submesh_devices = [model.mesh_device for model in dp_model]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for layer_num in tqdm(range(num_layers), desc=f"Allocating TT kv caches for each layer (submesh {mesh_idx+1})"):
            # Get the dtype for the kv cache based on the configured optimizations in the model
            if dp_model[mesh_idx].args.optimizations is not None:
                kv_cache_dtype = dp_model[mesh_idx].args.optimizations.get_tensor_dtype(
                    decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
                )
            else:
                logger.info("No dtype specified for the model KV cache - defaulting to ttnn.bfloat8_b.")
                kv_cache_dtype = None
            # Set default to bfloat8_b when no optimizations are configured
            kv_cache_dtype = ttnn.bfloat8_b if kv_cache_dtype is None else kv_cache_dtype
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    # TODO: this could be ShardTensorToMesh, removing the need for vLLM to know about TP for num_kv_heads.
                    # Could affect other calculations which use TTCacheEngine.num_kv_heads, though.
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_cache_dtype,
                    # Separate cache files for K and V to avoid collision.
                    cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
                )
                for kv in ["k", "v"]
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


class TT_MllamaProcessingInfo(MllamaProcessingInfo):
    def get_supported_mm_limits(self):
        return {"image": 1}  # TT implementation currently only supports 1 image


class DummyInputsBuilder(BaseDummyInputsBuilder):
    """
    We don't need to implement a dummy input builder since we don't do profiling in vLLM.
    Create callable class just for processor registration.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        raise NotImplementedError

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        raise NotImplementedError


# TODO: This multi-modal processor currently bypasses vLLM's mm processing on the images
# and passes the images directly to the model. In the future, the apply() function should
# call super().apply() (similar to vllm.model_executor.models.mllama.py::MllamaMultiModalProcessor)
# and _get_mm_fields_config / _get_prompt_updates should be implemented.
class MllamaMultiModalProcessor(EncDecMultiModalProcessor[TT_MllamaProcessingInfo]):
    """Multi-modal processor for Llama3.2-Vision that handles encoder-decoder inputs."""

    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        data = mm_data.get("image", [])
        num_images = 1 if isinstance(data, Image) else len(data)
        image_token_id = self.info.get_hf_config().image_token_index
        return [image_token_id] * num_images

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Unused, defined to satisfy abstract method requirement."""
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        """Unused, defined to satisfy abstract method requirement."""
        raise NotImplementedError

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalEncDecInputs:
        """
        Based on vllm.model_executor.models.mllama.py::MllamaMultiModalProcessor
        without performing processing on the images inputs or computing num_tiles (here it is fixed).
        """

        # In vLLM's mllama.py, super().apply() is called which also processes images,
        # while here only prompts are tokenized.
        encoder_prompt = self.create_encoder_prompt(prompt, mm_data)
        encoder_inputs = MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=encoder_prompt,
            mm_kwargs=mm_data,  # We pass the image directly
            mm_hashes={},
            mm_placeholders={},
        )
        mm_inputs = self._get_enc_dec_inputs(
            prompt=prompt,
            mm_data=mm_data,
            encoder_inputs=encoder_inputs,
        )

        image_token_id = self.info.get_hf_config().image_token_index
        # Check that the number of image tokens in the decoder prompt matches
        # the number of images provided in mm_data
        num_image_tokens = mm_inputs["prompt_token_ids"].count(image_token_id)
        image_data = mm_data.get("image", [])
        num_images = 1 if isinstance(image_data, Image) else len(image_data)
        if num_image_tokens != num_images:
            raise ValueError(
                f"The number of image tokens ({num_image_tokens}) must be"
                f" the same as the number of images ({num_images})"
            )

        if os.environ.get("MESH_DEVICE") == "N300":
            prompt_len = len(mm_inputs["prompt_token_ids"])
            MAX_PROMPT_LEN = 8192
            if prompt_len > MAX_PROMPT_LEN:
                raise ValueError(
                    f"TT-LLama11B-Vision does not support prompts longer than {MAX_PROMPT_LEN} tokens on N300 (received prompt with {prompt_len} tokens)"
                )

        # Example input to encoder and decoder:
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

        if mm_data:
            # Set encoder prompt length based on the number of vision tokens so block manager allocates enough blocks (cross block tables).
            vision_config = self.info.get_hf_config().vision_config
            assert vision_config.image_size % 14 == 0, "chunk size should be multiple of 14"
            token_per_chunk = nearest_32(
                (vision_config.image_size // 14) ** 2 + 1
            )  # Note: we use nearest 32 while vLLM does not by default
            num_vision_tokens = (
                vision_config.max_num_tiles * token_per_chunk
            )  # Note: we use max_num_tiles while vLLM uses num_tiles by default

            hf_processor = self.info.get_hf_processor()
            image_token: str = hf_processor.image_token
            mm_inputs["encoder_prompt_token_ids"] = [image_token_id] * num_vision_tokens
            mm_inputs["encoder_prompt"] = image_token * num_vision_tokens

        return mm_inputs


@MULTIMODAL_REGISTRY.register_processor(
    MllamaMultiModalProcessor, info=TT_MllamaProcessingInfo, dummy_inputs=DummyInputsBuilder
)
class MllamaForConditionalGeneration(Generator, SupportsMultiModal, SupportsV0Only):
    # Class-level capabilities
    # Note: Mllama doesn't support prefix caching (it's V0 only)
    model_capabilities = {
        "supports_prefix_caching": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MLLAMA_IMAGE_TOKEN_ID = 128256
        self.max_gen_len = self.model_args[0].max_seq_len - 1  # TODO: double check what this should be

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations: str = None
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
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
        images: Union[List[Image], List[List[Image]]],
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
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        hf_model_name = hf_config._name_or_path
        if (
            ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
            and mesh_device.get_num_devices() == 1
            and is_wormhole_b0()
        ):
            MAX_PROMPT_LEN = 32768
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
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
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
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
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
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
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


class MultiModalProcessor(BaseMultiModalProcessor):
    """Multi-modal processor for Gemma3 / Qwen-VL."""

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Unused, defined to satisfy abstract method requirement."""
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        """Unused, defined to satisfy abstract method requirement."""
        raise NotImplementedError

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        input_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # WORKAROUND
        # When using /v1/chat/completions endpoint prompt is already tokenized
        # Processor requires text, so we decode tokens back to text
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            # Use the processor's tokenizer to decode tokens back to text
            tokenizer = (
                getattr(input_processor, "tokenizer") if hasattr(input_processor, "tokenizer") else input_processor
            )
            text_prompt = tokenizer.decode(prompt, skip_special_tokens=False)
            logger.warning(f"Applied workaround: decoded {len(prompt)} tokens back to text for processor compatibility")
        else:
            text_prompt = prompt

        processed_inputs = input_processor(
            text=text_prompt,  # [INFO] Qwen2VLProcessor handles the case where text is a string or a list of strings
            images=mm_data["image"] if mm_data else None,
            videos=None,  # [INFO] videos are not supported yet
            return_tensors="pt",
        )

        assert processed_inputs.input_ids.shape[0] == 1, "Expected to process one input prompt at a time in processor"
        prompt_token_ids = processed_inputs.input_ids[0].tolist()

        mm_inputs = MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            mm_kwargs={"image": processed_inputs},  # [INFO] add processed_inputs,
            mm_hashes={},
            mm_placeholders={},
        )
        return mm_inputs


@MULTIMODAL_REGISTRY.register_processor(
    Gemma3MultiModalProcessor if envs.VLLM_USE_V1 else MultiModalProcessor,
    info=Gemma3ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class Gemma3ForConditionalGeneration(Generator, SupportsMultiModal):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=131072,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        from models.demos.gemma3.demo.vision_demo import create_multimodal_model

        optimizations = (
            DecodersPrecision.from_string(optimizations) if optimizations is not None else DecodersPrecision.performance
        )

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
                optimizations=lambda model_args: optimizations(model_args.n_layers, model_args.model_name),
            )
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        if not envs.VLLM_USE_V1:
            data = kwargs.get("images", None)
            kwargs["pixel_values"] = (
                [im.pixel_values if hasattr(im, "pixel_values") else None for im in data] if data else None
            )

        return super().prefill_forward_text(**kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)


class GptOssForCausalLM(Generator):
    """GPT-OSS model for vLLM integration"""

    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,  # Sliding window => no prefix caching
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        from models.demos.gpt_oss.tt.common import create_tt_model

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None

        for submesh in submesh_devices:
            # Use the existing create_tt_model function
            model_args_i, model_i, _, state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                paged_attention_config=None,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
                num_layers=n_layers,
                mesh_config=None,
                create_kv_cache=False,
            )

            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat8_b)

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)
