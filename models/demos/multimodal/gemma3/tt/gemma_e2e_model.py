# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import re
from typing import List

import torch
from loguru import logger

import ttnn
from models.demos.multimodal.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.generator import Generator, max_prefill_chunk_size_cutoff
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.multimodal.llama_vision_model import _stack_images

try:
    from PIL.Image import Image

    from models.tt_transformers.tt.generator_vllm import (
        Gemma3ForConditionalGeneration as TtTransformersGemma3ForConditionalGeneration,
    )
except ModuleNotFoundError:
    Image = None
    TtTransformersGemma3ForConditionalGeneration = None


def _is_insufficient_trace_region_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "trace" not in msg:
        return False
    return "trace_region" in msg or "get_trace_buffers_size" in msg or ("allocated" in msg and "region" in msg)


def _trace_buffer_bytes_from_fatal(exc: BaseException) -> int | None:
    m = re.search(r"Creating trace buffers of size (\d+)B", str(exc), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _stack_images(
    images: List[List[torch.Tensor]],  # batch of samples, each with list of image embeddings
) -> List[torch.Tensor]:
    """
    Concatenate image embeddings per sample into a single 2D tensor.

    Args:
        images: List of samples, each being a list of [num_patches, hidden_dim] tensors

    Returns:
        List of [total_patches, hidden_dim] tensors, one per sample
    """
    return [torch.cat(image_list, dim=0) for image_list in images]


class TtGemmaModel(Transformer):
    """When ``supports_vision_prefill_host_trace`` is True, prefill host-trace replays
    ``transform_and_embed`` + ``ttnn_prefill_forward`` with a **pre-merged** embedding
    (text embed + vision scatter + residual prep). The vision tower still runs before
    each prefill; only the transformer block graph is traced."""

    supports_vision_prefill_host_trace = True

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args,
            dtype,
            mesh_device,
            state_dict,
            weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        self.vision_model = TtGemmaTransformerVision(
            mesh_device=mesh_device,
            state_dict=state_dict,
            tt_ccl=self.tt_ccl,
            state_dict_prefix="model.vision_tower.vision_model.",
            dtype=dtype,
            configuration=args,
            weight_cache_path=weight_cache_path,
        )

    def _embed_with_vision_scatter_and_residual_4d(self, tokens, start_pos, **kwargs):
        """Device path: token embed, optional vision scatter, residual prep → 4D tensor for prefill forward."""
        device = self.mesh_device
        S = tokens.shape[-1]
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        assert mat_len >= start_pos + S, f"Padded prefill end idx {start_pos + S} exceeds max seq len {mat_len}"

        ttnn_tokens = ttnn.from_torch(
            tokens.reshape(1, 1, 1, -1),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        tokens_embd = self.embd(ttnn_tokens)
        tokens_embd = ttnn.to_torch(tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))

        if "pixel_values" in kwargs and kwargs.get("pixel_values", None) is not None:
            vision_output = self.compute_vision_token(kwargs.get("pixel_values", None))
            comp_vision_output = ttnn.to_torch(
                vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[: vision_output.shape[0], :]

            image_features = comp_vision_output.squeeze(0)
            special_image_mask = (tokens == self.args.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(tokens_embd)
            image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
            tokens_embd = tokens_embd.masked_scatter(special_image_mask, image_features)

        tokens_embd = self.args.prepare_residual_tensor_prefill(
            tokens_embd,
        )
        return ttnn.unsqueeze_to_4D(tokens_embd)

    def transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
        """Text trace: uint32 → embed. Vision e2e trace: first arg is already merged bf16 activations."""
        if tokens.get_dtype() != ttnn.uint32:
            return tokens, tt_page_table, tt_chunk_page_table
        return super().transform_and_embed_prefill_inputs_device(tokens, tt_page_table, tt_chunk_page_table)

    def prepare_prefill_inputs_trace(
        self, tokens, page_table=None, chunk_page_table=None, batch_size=1, user_id=0, **kwargs
    ):
        return self.prepare_inputs_prefill(
            tokens,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            trace_enabled=True,
            batch_size=batch_size,
            user_id=user_id,
            **kwargs,
        )

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        trace_enabled=False,
        last_token_idx=None,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        **kwargs,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device (trace_enabled=False) or on host (trace_enabled=True).

        When ``trace_enabled=True`` without images, the uint32 token path matches
        ``Transformer.prepare_inputs_prefill`` (see ``model.py``).

        When ``trace_enabled=True`` with ``pixel_values`` (batch 1), RoPE and page
        tables still come from the base trace path, but the first returned tensor is
        a **host** copy of the same merged 4D embedding the untraced e2e path would
        feed into ``ttnn_prefill_forward``. The vision tower runs in Python before
        each prefill; the captured graph is ``(identity embed) + decoder``.

        The trace branch defers to ``Transformer.prepare_inputs_prefill`` for RoPE
        and paging so the standard tt-Transformers prefill-trace path stays aligned.
        """

        if trace_enabled:
            # Text-only or vision e2e: drop multimodal fields for base trace RoPE / page prep.
            trace_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("pixel_values", "image_sizes", "image_grid_thw", "image_grid")
            }
            base = super().prepare_inputs_prefill(
                tokens,
                start_pos=start_pos,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                trace_enabled=True,
                last_token_idx=last_token_idx,
                global_user_id=global_user_id,
                batch_size=batch_size,
                user_id=user_id,
                **trace_kwargs,
            )
            if (
                self.supports_vision_prefill_host_trace
                and batch_size == 1
                and kwargs.get("pixel_values", None) is not None
            ):
                x_dev = self._embed_with_vision_scatter_and_residual_4d(tokens, start_pos, **kwargs)
                ttnn.synchronize_device(self.mesh_device)
                x_host = ttnn.from_device(x_dev)
                return (x_host, base[1], base[2], base[3], base[4])
            return base

        # --- Non-trace: Gemma multimodal (embed on device, optional vision scatter, residual prep) ---
        device = self.mesh_device
        S = tokens.shape[-1]
        tokens_embd = self._embed_with_vision_scatter_and_residual_4d(tokens, start_pos, **kwargs)

        prefill_start_pos = start_pos
        slice_end = start_pos + S

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :],
            self.rope_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :],
        ]

        tt_rot_mats_prefill_local = [
            self.rope_local_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :],
            self.rope_local_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill_global, tt_rot_mats_prefill_local, tt_page_table, tt_chunk_page_table

    def compute_vision_token(self, pixel_values, batch_size=3):
        """
        Process vision tokens in batches to avoid OOM for large number of images.

        Args:
            pixel_values: torch.Tensor of shape (B, C, H, W) where B is number of images
            batch_size: Number of images to process in one batch (max 3, or else device runs OOM)

        Returns:
            Combined vision output tensor
        """

        assert 0 < batch_size <= 3, "Device runs OOM with batch size > 3"

        if not isinstance(pixel_values, list):
            pixel_values = [pixel_values]

        pixel_values_batches = []
        total_num_images = 0
        for image in pixel_values:
            num_images = image.shape[0]
            total_num_images += num_images
            if num_images < batch_size:
                pixel_values_batches.append(image)
            else:
                # If image was too big it was split into several, but still in one tensor
                for i in range(0, num_images, batch_size):
                    end_idx = min(i + batch_size, num_images)
                    pixel_values_batches.append(image[i:end_idx])

        logger.info(f"Starting vision encoder for {total_num_images} image(s) in {len(pixel_values_batches)} batch(es)")

        # Process images in batches
        vision_outputs = []
        for batch_idx, batch_pixel_values in enumerate(pixel_values_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(pixel_values_batches)}")
            batch_vision_output = self.vision_model(batch_pixel_values)
            vision_outputs.append(batch_vision_output)

        # Combine all vision outputs along the batch dimension
        combined_vision_output = ttnn.concat(vision_outputs, dim=1)
        logger.info(f"Vision encoder done")
        return combined_vision_output


class Gemma3GeneratorMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decode_trace_unavailable = False

    @staticmethod
    def _has_pixel_values(kwargs):
        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is None:
            return False
        if isinstance(pixel_values, (list, tuple)):
            return any(x is not None for x in pixel_values)
        return True

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device):
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        warmup_use_trace = enable_trace and getattr(self.model_args[0], "warmup_prefill_capture_trace", True)
        sequence_lengths_to_warmup = self.model_args[0].get_warmup_prefill_supported_seq_lens()
        warmup_batch_sizes = (1,)
        skip_sequence_lengths = False
        sampling_parameters_sweeped = False

        for model_id in range(self.data_parallel):
            for supported_length in sequence_lengths_to_warmup:
                if model_id != 0 and (
                    supported_length not in self.model_args[0].trace_prefill_supported_seq_lens or not warmup_use_trace
                ):
                    continue

                for batch_size in warmup_batch_sizes:
                    if batch_size > 1 and batch_size * supported_length >= 128 * 1024:
                        logger.info(
                            f"Skipping batched prefill warmup for batch_size={batch_size}, "
                            f"seq_len={supported_length}: exceeds token limit"
                        )
                        continue

                    warmup_args = self._mock_tokens(batch_size, supported_length, kv_cache, model_id)

                    if warmup_args["page_table"] is None and max_prefill_chunk_size_cutoff(
                        supported_length, self.model_args[0].max_prefill_chunk_size
                    ):
                        logger.warning(
                            f"Skipping warmup for sequence lengths after: {supported_length} because they are greater than the max prefill chunk size and paged attention is disabled"
                        )
                        skip_sequence_lengths = True
                        break

                    if not sampling_parameters_sweeped:
                        sampling_params = self._create_sampling_params(
                            can_sample_on_device=can_sample_on_device,
                            non_greedy_decoding_on_device=non_greedy_decoding_on_device,
                            batch_size=batch_size,
                        )
                    else:
                        sampling_params = [None]

                    for param in sampling_params:
                        logger.info(
                            f"Warming up prefill for sequence length: {supported_length} for batch size: {batch_size} with sampling params: {param}"
                        )
                        self.prefill_forward_text(
                            **warmup_args,
                            kv_cache=kv_cache,
                            enable_trace=warmup_use_trace,
                            model_id_warmup=model_id,
                            sampling_params=param,
                        )

                    sampling_parameters_sweeped = True

                if skip_sequence_lengths:
                    break

        if getattr(self.model_args[0], "is_multimodal", False):
            vision_chunk_size = getattr(self.model_args[0], "vision_chunk_size", 896)
            vision_channels = getattr(self.model_args[0], "vision_in_channels", 3)
            model_id = 0
            warmup_pixel_values = [torch.zeros((1, vision_channels, vision_chunk_size, vision_chunk_size))]
            prefill_forward_args = self._mock_tokens(1, 128, kv_cache, model_id)

            logger.info(f"Warming up vision encoder with image size {vision_chunk_size}x{vision_chunk_size}")
            self.prefill_forward_text(
                **prefill_forward_args,
                kv_cache=kv_cache,
                enable_trace=warmup_use_trace,
                model_id_warmup=model_id,
                sampling_params=None,
                pixel_values=warmup_pixel_values,
                image_sizes=[(vision_chunk_size, vision_chunk_size)],
            )
            logger.info("Vision encoder warmup completed")

    def _capture_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        kv_cache=None,
        model_id=-1,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        last_token_idx=None,
        **kwargs,
    ):
        if batch_size > 1:
            prefill_kwargs = {**kwargs, "page_table": page_table, "batch_size": batch_size, "user_id": user_id}
            if global_user_id is not None:
                prefill_kwargs["global_user_id"] = global_user_id
            if last_token_idx is not None:
                prefill_kwargs["last_token_idx"] = last_token_idx
            host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
            tt_rot_mats_prefill_global = host_inputs[1]
            tt_rot_mats_prefill_local = host_inputs[2]
            host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            logger.info("Done Compiling Model")

            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
            ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            logger.info("Done Capturing Prefill Trace")
            return trace_id, tt_out_trace, *device_inputs

        prefill_kwargs = {**kwargs, "page_table": page_table}
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id
        if last_token_idx is not None:
            prefill_kwargs["last_token_idx"] = last_token_idx
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        tt_rot_mats_prefill_global = host_inputs[1]
        tt_rot_mats_prefill_local = host_inputs[2]
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        logger.info("Done Compiling Model")

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        logger.info("Done Capturing Prefill Trace")
        return trace_id, tt_out_trace, *device_inputs

    def _easy_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        user_id=0,
        last_token_idx=None,
        kv_cache=None,
        model_id=-1,
        prefill_seq_len=None,
        batch_size=1,
        **kwargs,
    ):
        global_user_id = kwargs.get("global_user_id", None)
        has_vision = kwargs.get("pixel_values", None) is not None
        vision_merged = (
            has_vision
            and batch_size == 1
            and getattr(self.model[model_id], "supports_vision_prefill_host_trace", False)
        )
        trace_key = f"{prefill_seq_len}_{model_id}_{batch_size}{'_vm' if vision_merged else ''}"
        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                prefill_ids,
                page_table=page_table,
                kv_cache=kv_cache,
                model_id=model_id,
                global_user_id=global_user_id,
                batch_size=batch_size,
                user_id=user_id,
                last_token_idx=last_token_idx,
                **kwargs,
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace

        return self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            prefill_ids,
            page_table=page_table,
            model_id=model_id,
            global_user_id=global_user_id,
            batch_size=batch_size,
            user_id=user_id,
            last_token_idx=last_token_idx,
            **kwargs,
        )

    def _prefill_forward_trace(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        prefill_ids,
        user_id=0,
        page_table=None,
        model_id=-1,
        global_user_id=None,
        batch_size=1,
        last_token_idx=None,
        **kwargs,
    ):
        prefill_kwargs = {**kwargs, "page_table": page_table, "batch_size": batch_size, "user_id": user_id}
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id
        if last_token_idx is not None:
            prefill_kwargs["last_token_idx"] = last_token_idx
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        copy_host_to_device(
            host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
        )
        ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)
        return tt_out_trace

    def prefill_forward_text(self, *args, **kwargs):
        if self._has_pixel_values(kwargs) and getattr(self.model[0], "supports_vision_prefill_host_trace", False):
            saved_flags = [getattr(ma, "disable_batched_prefill", False) for ma in self.model_args]
            for ma in self.model_args:
                ma.disable_batched_prefill = True
            try:
                return super().prefill_forward_text(*args, **kwargs)
            finally:
                for ma, prev in zip(self.model_args, saved_flags):
                    ma.disable_batched_prefill = prev
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        **kwargs,
    ):
        if self._decode_trace_unavailable:
            enable_trace = False
        try:
            return super().decode_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                read_from_device=read_from_device,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                **kwargs,
            )
        except RuntimeError as err:
            if not enable_trace or not _is_insufficient_trace_region_error(err):
                raise
            need_b = _trace_buffer_bytes_from_fatal(err)
            if need_b is not None:
                round_mib = ((need_b + 1_048_575) // 1_048_576) * 1_048_576
                min_suggest = max(64 * 1_048_576, round_mib)
                logger.warning(
                    f"Decode trace capture failed (mesh trace region too small): running decode without trace. "
                    f"This run needed ~{need_b} B of trace buffer; set trace_region_size to at least {min_suggest}."
                )
            else:
                logger.warning(
                    "Decode trace capture failed (mesh trace region too small): running decode without trace."
                )
            self._decode_trace_unavailable = True
            return super().decode_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=False,
                read_from_device=read_from_device,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                **kwargs,
            )


class Gemma3Generator(Gemma3GeneratorMixin, Generator):
    pass


if TtTransformersGemma3ForConditionalGeneration is not None:

    class Gemma3ForConditionalGeneration(Gemma3GeneratorMixin, TtTransformersGemma3ForConditionalGeneration):
        def prefill_forward(self, *args, **kwargs):
            images = kwargs.pop("images", None)
            pixel_values_per_user = None
            if images is not None:
                collected = []
                has_any = False
                for img in images:
                    if img is None:
                        collected.append(None)
                        continue
                    if Image is not None and isinstance(img, Image):
                        collected.append(img)
                        has_any = True
                        continue
                    if hasattr(img, "__contains__") and "pixel_values" in img:
                        pv = img["pixel_values"]
                        if not isinstance(pv, torch.Tensor) and hasattr(pv, "data"):
                            pv = pv.data
                        collected.append(pv)
                        has_any = True
                    else:
                        collected.append(None)
                if has_any:
                    pixel_values_per_user = collected

            if pixel_values_per_user is None:
                return super().prefill_forward_text(**kwargs)

            kwargs["pixel_values"] = pixel_values_per_user
            if not getattr(self.model[0], "supports_vision_prefill_host_trace", False):
                kwargs["enable_trace"] = False
            return self.prefill_forward_text(**kwargs)
