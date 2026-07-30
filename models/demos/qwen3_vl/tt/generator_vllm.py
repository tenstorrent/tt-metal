# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace
from typing import Mapping, Optional

import torch
from loguru import logger
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as Ref_Qwen3VLForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.qwen3_vl.tt.common import (
    get_hf_visual,
    get_pad_embedding,
    merge_vision_tokens_single_user_ttnn,
    multimodal_rope_single_user_from_hf,
    preprocess_inputs_prefill_single_user_ttnn,
)
from models.demos.qwen3_vl.tt.generator import Generator as QwenVLGenerator
from models.demos.qwen3_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen3_vl.tt.model_config import VisionModelArgs
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


def _prefill_single_user_with_sliced_page_table(
    generator,
    input_prefill,
    page_table,
    user_id,
    decoding_pos,
    rot_mats,
    kv_cache,
    deepstack_visual_embeds,
):
    """Run one user prefill without exposing unallocated page-table entries."""
    user_page_table = generator._ttt_generator._get_prefill_user_page_table(page_table, kv_cache, decoding_pos)
    return generator.prefill_forward_single_user_text(
        input_prefill,
        page_table=user_page_table,
        user_id=user_id,
        last_token_idx=decoding_pos - 1,
        rot_mats=rot_mats,
        kv_cache=kv_cache,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )


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
    # Prefer the original HF model id (`_name_or_path`) since `name_or_path` may be rewritten to a local cache snapshot path.
    hf_model_id = getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", "")

    tt_model_args = ModelArgs(
        mesh_device,
        instruct=("Instruct" in hf_model_id),
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    assert tt_model_args.model_name.replace("-", "") in hf_model_id.replace(
        "-", ""
    ), f"The model specified in vLLM ({hf_model_id}) does not match the model name ({tt_model_args.model_name}) with model weights ({tt_model_args.CKPT_DIR})."
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


class TT_Qwen3VLProcessingInfo(Qwen3VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1, "video": 0}  # [INFO] videos are not supported yet, only supporting 1 image for now


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3VLProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen3VLForConditionalGeneration(QwenVLGenerator, SupportsMultiModal):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
    }

    def __init__(self, *args, **kwargs):
        self.reference_model = kwargs.pop("reference_model", None)
        self.visual_model = kwargs.pop("visual_model", None)
        assert (
            self.reference_model is not None and self.visual_model is not None
        ), "Reference model and visual model must be provided for vLLM"

        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        # Prefer the original HF model id (`_name_or_path`) since `name_or_path` may be rewritten to a local cache snapshot path.
        hf_model_id = getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", "")
        optimizations, max_seq_len_native = get_platform_specific_optimizations(hf_model_id)
        if max_seq_len > max_seq_len_native:
            logger.warning(
                f"max_seq_len {max_seq_len} is not supported for {hf_model_id}, using {max_seq_len_native} instead"
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
        config = Ref_Qwen3VLForConditionalGeneration.config_class.from_pretrained(ref_model_name)
        # config.vision_config.depth = 1 # [INFO] useful for debugging
        reference_model = Ref_Qwen3VLForConditionalGeneration.from_pretrained(
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
        visual_model = DropInVisionTransformer(get_hf_visual(reference_model), vision_model_args)

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
        **kwargs,  # pixel_values and image_grid_thw
    ):
        """
        Prefill forward pass that processes users one at a time to reduce memory pressure
        and avoid expensive concat/stack operations.
        """
        start_pos = kwargs.get("start_pos", None)
        assert (start_pos is None) or all(
            x == 0 for x in start_pos
        ), f"Prefix caching is not supported for Qwen3VL, got start_pos: {start_pos}"
        enable_trace = False
        logger.warning("Tracing in prefill mode is not supported for Qwen3VL")

        pad_token_id = self.tokenizer.pad_token_id
        batch_size = tokens.shape[0]

        # Fix padding for all users
        for i in range(batch_size):
            tokens[i][prompt_lens[i] :] = pad_token_id

        # Get pad embedding once (reused for all users)
        pad_embedding_tt = get_pad_embedding(self.reference_model, pad_token_id, self.model_args)

        # Output storage
        output_logits = torch.zeros(batch_size, 1, self.model_args.vocab_size)
        all_rope_deltas = []

        # Process each user one at a time
        for user_id in range(batch_size):
            logger.info(f"Processing User {user_id + 1}/{batch_size}")

            # Get this user's input_ids and attention_mask
            user_input_ids = tokens[user_id].to(torch.int64)
            user_prompt_len = prompt_lens[user_id]
            user_attention_mask = torch.zeros(user_input_ids.shape[0], dtype=torch.int64, device=tokens.device)
            user_attention_mask[:user_prompt_len] = 1

            # Process vision for this user if they have images

            # Check if we have image inputs
            has_images = (
                "pixel_values" in kwargs
                and len(kwargs["pixel_values"]) > user_id
                and kwargs["pixel_values"][user_id] is not None
            )

            if has_images and kwargs["pixel_values"][user_id] is not None:
                # Get this user's pixel_values and image_grid_thw
                user_pixel_values = kwargs["pixel_values"][user_id]
                user_image_grid_thw = kwargs["image_grid_thw"][user_id]

                # Concatenate all images for this user (if multiple images per user)
                if isinstance(user_pixel_values, list) and len(user_pixel_values) > 0:
                    user_pixel_values = torch.concat(user_pixel_values, dim=0)
                    # Stack grid_thw for all images of this user
                    user_image_grid_thw = torch.stack(
                        [g.to(device=tokens.device, dtype=torch.int32) for g in user_image_grid_thw],
                        dim=0,
                    )

                # Run vision model for this single user
                image_embeds, deepstack_visual_embeds = self.visual_model.forward_single_user(
                    user_pixel_values, grid_thw=user_image_grid_thw
                )
            else:
                # Text-only user
                image_embeds = ttnn.from_torch(
                    torch.tensor([], dtype=torch.bfloat16), device=self.model_args.mesh_device, dtype=ttnn.bfloat16
                )
                deepstack_visual_embeds = None
                user_image_grid_thw = None

            # Compute text embeddings for this user
            user_text_embeds = self.reference_model.model.language_model.embed_tokens(user_input_ids.unsqueeze(0))
            user_text_embeds_tt = ttnn.from_torch(
                user_text_embeds.squeeze(0),  # Remove batch dim for single-user processing
                device=self.model_args.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.model_args.mesh_device, dims=(None, 1), mesh_shape=self.model_args.cluster_shape
                ),
            )

            # Merge vision tokens for this user
            input_embeds, deepstack_visual_embeds = merge_vision_tokens_single_user_ttnn(
                user_input_ids,
                user_text_embeds_tt,
                image_embeds,
                self.reference_model.config,
                deepstack_visual_embeds,
                self.model_args,
            )
            ttnn.deallocate(user_text_embeds_tt)
            ttnn.deallocate(image_embeds)

            # Preprocess for this user
            (
                input_prefill,
                deepstack_visual_embeds,
                decoding_pos,
                prefill_len,
            ) = preprocess_inputs_prefill_single_user_ttnn(
                input_embeds,
                self.model_args,
                user_attention_mask,
                pad_embedding=pad_embedding_tt,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

            # Compute rope embeddings for this user
            cos, sin, rope_deltas = multimodal_rope_single_user_from_hf(
                user_input_ids,
                user_image_grid_thw,
                self.reference_model,
                self.model_args,
                pad_token_id=pad_token_id,
            )
            rot_mats = (cos, sin)
            all_rope_deltas.append(rope_deltas)

            # The persistent vLLM page table is padded to max_model_len. The
            # attention prefill path uses its width to decide how much K/V to
            # write, so expose only blocks allocated for this user's sequence.
            # Otherwise padded zero block IDs redirect padded K/V writes into
            # physical block 0 and can corrupt this request's prompt cache.
            logits = _prefill_single_user_with_sliced_page_table(
                self,
                ttnn.unsqueeze(input_prefill, 0),
                page_table,
                user_id,
                decoding_pos,
                rot_mats,
                kv_cache,
                deepstack_visual_embeds,
            )

            # Store output
            output_logits[user_id] = logits

            # Deallocate tensors for this user to free memory
            ttnn.deallocate(input_prefill)
            if deepstack_visual_embeds is not None:
                for dsve in deepstack_visual_embeds:
                    ttnn.deallocate(dsve)

        # Deallocate pad embedding
        ttnn.deallocate(pad_embedding_tt)

        # Stack rope_deltas from all users
        rope_deltas = torch.stack([rd.squeeze(0) for rd in all_rope_deltas], dim=0)

        logger.info(f"Finished prefill for all {batch_size} users")
        return output_logits, rope_deltas

    def decode_forward(self, *args, **kwargs):
        rope_deltas_list: list = kwargs.pop(
            "rope_deltas_all_users", None
        )  # [INFO] update the cos/sin matrices for the current users in the batch
        if rope_deltas_list is not None:
            super().update_rope_deltas(rope_deltas_list)

        return super().decode_forward(*args, **kwargs)
