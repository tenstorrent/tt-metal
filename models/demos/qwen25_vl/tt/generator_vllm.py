# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace
from typing import Mapping, Optional

import torch
import vllm.envs as envs
from loguru import logger
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as Ref_Qwen2_5_VLForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.qwen25_vl.tt.common import merge_vision_tokens, multimodal_rope_from_hf, preprocess_inputs_prefill
from models.demos.qwen25_vl.tt.generator import Generator as QwenVLGenerator
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.generator_vllm import DummyInputsBuilder, MultiModalProcessor
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: Transformer, model_args: ModelArgs, tt_cache_path):
    for layer_idx in range(num_layers):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)

        model.layers[layer_idx].attention.layer_past = [
            ttnn.as_tensor(
                cache_kv,
                device=model.mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=model_args.model_config["ATTN_W_LAYOUT_TILE"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                # Separate cache files for K and V to avoid collision.
                cache_file_name=f"{tt_cache_path}/{kv}cache_{kv_cache_shape}",
            )
            for kv in ["k", "v"]
        ]

    return [l.attention.layer_past for l in model.layers]


def get_platform_specific_optimizations(model_name):
    max_seq_len = 131072

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
    tt_model_args.use_qk_fused = False  # Qwen2.5-VL doesn't use qk fused ops
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


class TT_Qwen2_5_VLProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1, "video": 0}  # [INFO] videos are not supported yet, only supporting 1 image for now


# TODO: Eventually replace MultiModalProcessor with vllm.model_executor.models.qwen2_5_vl::Qwen2_5_VLMultiModalProcessor
@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor if envs.VLLM_USE_V1 else MultiModalProcessor,
    info=TT_Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder if envs.VLLM_USE_V1 else DummyInputsBuilder,
)
class Qwen2_5_VLForConditionalGeneration(QwenVLGenerator, SupportsMultiModal):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,
    }

    def __init__(self, *args, **kwargs):
        self.reference_model = kwargs.pop("reference_model", None)
        self.visual_model = kwargs.pop("visual_model", None)
        assert (
            self.reference_model is not None and self.visual_model is not None
        ), "Reference model and visual model must be provided for vLLM"

        # Performance tracking
        self._decode_iteration = 0
        self._last_decode_time = None

        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
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
            optimizations=DecodersPrecision.performance(config.vision_config.depth, ref_model_name),
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
        page_table,
        kv_cache,
        prompt_lens,  # [INFO] prompt_lens is pre-padding number of tokens after text-image processing
        enable_trace,
        **kwargs,  # images for V0, pixel_values and image_grid_thw for V1,
    ):
        start_pos = kwargs.get("start_pos", None)
        assert (start_pos is None) or all(
            x == 0 for x in start_pos
        ), f"Prefix caching is not supported for Qwen2_5_VL, got start_pos: {start_pos}"
        # Must add this so that vLLM can call without errors
        enable_trace = False
        logger.warning("Tracing in prefill mode is not supported for Qwen2_5_VL")

        # [INFO] tokens are padded to the same length by appending 0s; change the padding to use pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        padded_seq_len = tokens.shape[-1]
        for i in range(tokens.shape[0]):  # for each user, fix their padding
            tokens[i][prompt_lens[i] :] = pad_token_id

        # reconstruct the inputs that Qwen2.5-VL expects
        inputs = CustomNamespace()
        if envs.VLLM_USE_V1:
            inputs.input_ids = tokens.to(torch.int64)  # TODO: Derive dtype, like V0 does (see below)?
            # Construct inputs.attention_mask with shape [batch_size, padded_seq_len] like tokens,
            # where each row has ones in the first prompt_lens[i] positions and zeros elsewhere
            inputs.attention_mask = torch.zeros(
                (tokens.shape[0], padded_seq_len), dtype=inputs.input_ids.dtype, device=tokens.device
            )
            for i, plen in enumerate(prompt_lens):
                inputs.attention_mask[i, :plen] = 1

            if (
                "pixel_values" in kwargs
                and len(kwargs["pixel_values"]) > 0
                and kwargs["pixel_values"][0] is not None
                # kwargs["pixel_values"] is a list,
                # each element is a list of images for one user
                # We only check if the first user's pixel_values is not None
                # as we currently do not support mixed inputs of text-only
                # users and text-image users
            ):
                inputs.pixel_values = torch.concat(
                    [im for user_pixel_values in kwargs["pixel_values"] for im in user_pixel_values], dim=0
                )
                inputs.image_grid_thw = torch.concat(
                    [im for user_image_grid_thw in kwargs["image_grid_thw"] for im in user_image_grid_thw], dim=0
                )
                # Vision prefill
                vision_prefill_start = time.perf_counter()
                image_embeds = self.visual_model(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
                vision_prefill_time = time.perf_counter() - vision_prefill_start
                batch_size = tokens.shape[0]
                vision_prefill_time_per_user = vision_prefill_time / batch_size
                # Calculate image tokens for logging
                try:
                    # Get spatial_merge_size from vision config if available
                    spatial_merge_size = getattr(self.reference_model.config.vision_config, "spatial_merge_size", None)
                    if spatial_merge_size is None:
                        # Try to get merge_size from processor config (fallback)
                        spatial_merge_size = getattr(self.reference_model.config, "spatial_merge_size", 14)
                    merge_length = spatial_merge_size**2
                    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw.numel() > 0:
                        # Calculate number of image tokens: grid_thw.prod() / merge_size^2
                        num_image_tokens = (inputs.image_grid_thw.prod() / merge_length).item()
                        vision_tok_s = num_image_tokens / vision_prefill_time if vision_prefill_time > 0 else 0
                        vision_tok_s_per_user = vision_tok_s / batch_size
                        logger.info(
                            f"[PERF] Vision prefill: {vision_prefill_time*1000:.2f}ms "
                            f"({vision_prefill_time_per_user*1000:.2f}ms/user) @ "
                            f"{vision_tok_s_per_user:.1f} tok/s/user ({vision_tok_s:.1f} tok/s throughput)"
                        )
                    else:
                        logger.info(
                            f"[PERF] Vision prefill: {vision_prefill_time*1000:.2f}ms "
                            f"({vision_prefill_time_per_user*1000:.2f}ms/user)"
                        )
                except Exception:
                    # Fallback to simple time logging if token calculation fails
                    logger.info(
                        f"[PERF] Vision prefill: {vision_prefill_time*1000:.2f}ms "
                        f"({vision_prefill_time_per_user*1000:.2f}ms/user)"
                    )
            else:
                # text-only users
                image_embeds = torch.tensor([], dtype=torch.bfloat16, device=tokens.device)
        else:  # V0
            if (
                "images" in kwargs
                and isinstance(kwargs["images"], list)
                and len(kwargs["images"]) > 0
                and kwargs["images"][0] is not None
                and "attention_mask" in kwargs["images"][0]
            ):
                inputs.input_ids = tokens.to(kwargs["images"][0].attention_mask.dtype)
            else:
                inputs.input_ids = tokens

            inputs.attention_mask = torch.concat(
                [
                    torch.nn.functional.pad(
                        im.attention_mask, (0, padded_seq_len - im.attention_mask.shape[-1]), value=0
                    )
                    if im is not None
                    else torch.ones_like(tokens[i : i + 1], dtype=tokens.dtype)
                    for i, im in enumerate(kwargs["images"])
                ],
                dim=0,
            )
            if (
                "images" in kwargs
                and len(kwargs["images"]) > 0
                and kwargs["images"][0] is not None
                and "pixel_values" in kwargs["images"][0]
            ):
                # we currently do not support mixed inputs of text-only users and text-image users; hence checking images[0] is enough
                inputs.pixel_values = torch.concat([im.pixel_values for im in kwargs["images"]], dim=0)
                inputs.image_grid_thw = torch.concat([im.image_grid_thw for im in kwargs["images"]], dim=0)

                # Vision prefill
                vision_prefill_start = time.perf_counter()
                image_embeds = self.visual_model(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
                vision_prefill_time = time.perf_counter() - vision_prefill_start
                batch_size = tokens.shape[0]
                vision_prefill_time_per_user = vision_prefill_time / batch_size
                # Calculate image tokens for logging
                try:
                    # Get spatial_merge_size from vision config if available
                    spatial_merge_size = getattr(self.reference_model.config.vision_config, "spatial_merge_size", None)
                    if spatial_merge_size is None:
                        # Try to get merge_size from processor config (fallback)
                        spatial_merge_size = getattr(self.reference_model.config, "spatial_merge_size", 14)
                    merge_length = spatial_merge_size**2
                    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw.numel() > 0:
                        # Calculate number of image tokens: grid_thw.prod() / merge_size^2
                        num_image_tokens = (inputs.image_grid_thw.prod() / merge_length).item()
                        vision_tok_s = num_image_tokens / vision_prefill_time if vision_prefill_time > 0 else 0
                        vision_tok_s_per_user = vision_tok_s / batch_size
                        logger.info(
                            f"[PERF] Vision prefill: {vision_prefill_time*1000:.2f}ms "
                            f"({vision_prefill_time_per_user*1000:.2f}ms/user) @ "
                            f"{vision_tok_s_per_user:.1f} tok/s/user ({vision_tok_s:.1f} tok/s throughput)"
                        )
                    else:
                        logger.info(
                            f"[PERF] Vision prefill: {vision_prefill_time*1000:.2f}ms "
                            f"({vision_prefill_time_per_user*1000:.2f}ms/user)"
                        )
                except Exception:
                    # Fallback to simple time logging if token calculation fails
                    logger.info(
                        f"[PERF] Vision prefill: {vision_prefill_time*1000:.2f}ms "
                        f"({vision_prefill_time_per_user*1000:.2f}ms/user)"
                    )
            else:
                # text-only users
                image_embeds = torch.tensor([], dtype=torch.bfloat16, device=tokens.device)

        # Prepare text + vision inputs for decoder model
        text_embeds = self.reference_model.model.language_model.embed_tokens(inputs.input_ids)

        input_embeds = merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, self.reference_model.config)
        (
            input_prefill_pt,
            decoding_pos,  # Position where decoding should start for each user
            prefill_lens,  # [INFO] prefill_lens is post-padding number of tokens after text-image processing
        ) = preprocess_inputs_prefill(
            input_embeds,
            self.model_args,
            inputs.attention_mask,
            pad_embedding=self.reference_model.model.language_model.embed_tokens(torch.tensor(pad_token_id)),
        )
        # Get user-specific rotary position embeddings
        cos, sin, rope_deltas = multimodal_rope_from_hf(
            inputs, input_embeds, self.reference_model, self.model_args, pad_token_id=pad_token_id
        )
        rot_mats = (cos, sin)

        # Text prefill timing (TTFT)
        prefill_start = time.perf_counter()
        logits = self.prefill_forward_text(
            input_prefill_pt,
            rot_mats=rot_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=decoding_pos,
        )
        prefill_time = time.perf_counter() - prefill_start
        batch_size = tokens.shape[0]
        ttft = prefill_time / batch_size
        # Calculate prefill tokens per second using prefill_lens (actual prefill lengths)
        if isinstance(prefill_lens, torch.Tensor):
            prefill_tokens = prefill_lens.sum().item()
        elif isinstance(prefill_lens, (list, tuple)):
            prefill_tokens = sum(prefill_lens)
        else:
            # Fallback: use decoding_pos as approximation
            if isinstance(decoding_pos, torch.Tensor):
                prefill_tokens = decoding_pos.sum().item()
            elif isinstance(decoding_pos, (list, tuple)):
                prefill_tokens = sum(decoding_pos)
            else:
                prefill_tokens = batch_size * padded_seq_len
        prefill_tok_s = prefill_tokens / prefill_time if prefill_time > 0 else 0
        logger.info(
            f"[PERF] Text prefill (TTFT): {prefill_time*1000:.2f}ms "
            f"({ttft*1000:.2f}ms/user) @ {prefill_tok_s:.1f} tok/s"
        )

        # Reset decode iteration counter for new prefill
        self._decode_iteration = 0

        return logits, rope_deltas

    def decode_forward(self, *args, **kwargs):
        rope_deltas_list: list = kwargs.pop(
            "rope_deltas_all_users", None
        )  # [INFO] update the cos/sin matrices for the current users in the batch
        if rope_deltas_list is not None:
            super().update_rope_deltas(rope_deltas_list)

        # Decode timing
        decode_start = time.perf_counter()
        result = super().decode_forward_text(*args, **kwargs)
        decode_time = time.perf_counter() - decode_start

        self._decode_iteration += 1
        self._last_decode_time = decode_time

        # Calculate tokens per second
        # Assume batch_size from the first argument (tokens) if available
        batch_size = 1
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            batch_size = args[0].shape[0] if len(args[0].shape) > 0 else 1

        tok_s_per_user = 1.0 / decode_time if decode_time > 0 else 0
        tok_s_throughput = tok_s_per_user * batch_size

        logger.info(
            f"[PERF] Decode iteration {self._decode_iteration}: {decode_time*1000:.2f}ms @ "
            f"{tok_s_per_user:.1f} tok/s/user ({tok_s_throughput:.1f} tok/s throughput)"
        )

        return result
