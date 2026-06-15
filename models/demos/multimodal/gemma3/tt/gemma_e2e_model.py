# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator
from models.common.utility_functions import is_blackhole
from models.demos.multimodal.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.tt_transformers.tt.common import sample_top_p
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer


class TtGemmaModel(Transformer):
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
        self._configure_on_device_sampling_support()

    def _configure_on_device_sampling_support(self):
        sampling_splits = self.args.num_devices if list(self.mesh_device.shape) != [1, 1] else 2
        total_vocab = getattr(self.args, "padded_vocab_size", None) or self.args.vocab_size
        per_device_vocab = total_vocab // sampling_splits
        max_per_device = getattr(self.args, "device_sampling_max_per_device_vocab", 64 * 1024)
        self._supports_on_device_sampling = per_device_vocab <= max_per_device
        if self._supports_on_device_sampling and self.sampling is None:
            self.sampling = SamplingGenerator(
                args=self.args,
                mesh_device=self.mesh_device,
                tt_ccl=self.tt_ccl,
            )

    def encode_vision_embeddings_from_pixels(self, pixel_values):
        """
        Run only the vision tower and return host patch embeddings for image token positions.
        """
        vision_output = self.compute_vision_token(pixel_values)
        if is_blackhole():
            # BH: vision hidden dim is tensor-parallel sharded; match embd readout (dim=-1) for multi-chip (e.g. P150x4).
            comp_vision_output = ttnn.to_torch(
                vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
            )
            comp_vision_output = comp_vision_output[: int(vision_output.shape[0])]
            if comp_vision_output.shape[-1] > self.args.dim:
                comp_vision_output = comp_vision_output[..., : self.args.dim]
        else:
            comp_vision_output = ttnn.to_torch(
                vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[: vision_output.shape[0], :]
        return comp_vision_output.squeeze(0)

    def _vision_embeddings_to_tensor(self, vision_embeddings, batch_rows: int) -> torch.Tensor | None:
        """
        ``encode_vision_for_prefill`` returns a list (one tensor per batch row / pixel_values slot).
        ``GemmaMultimodalGenerator`` in ``gemma_multimodal_generator.py`` slices to one tensor per user
        before ``prepare_inputs_prefill``; the ``GemmaMultimodalGenerator`` defined in this file
        delegates straight to ``Generator.prefill_forward_text`` and may pass the full list — coalesce
        here so fusion always sees a single tensor.
        """
        if vision_embeddings is None:
            return None
        if isinstance(vision_embeddings, torch.Tensor):
            return vision_embeddings
        if isinstance(vision_embeddings, (list, tuple)):
            parts = [v for v in vision_embeddings if v is not None]
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            if batch_rows == 1:
                first = parts[0]
                if first.dim() == 3 and first.shape[0] == 1:
                    return torch.cat(parts, dim=1)
                if first.dim() == 2:
                    return torch.cat(parts, dim=0)
                return torch.cat(parts, dim=0)
            if len(parts) == batch_rows:
                stacked = torch.stack(parts, dim=0)
                if stacked.dim() == 4 and stacked.shape[1] == 1:
                    stacked = stacked.squeeze(1)
                return stacked
            raise ValueError(
                f"vision_embeddings list length {len(parts)} does not match prompt batch rows {batch_rows}"
            )
        raise TypeError(f"vision_embeddings must be Tensor or sequence of Tensors, got {type(vision_embeddings)}")

    def _fuse_vision_into_text_embeddings(self, pt_tokens, tokens_embd, image_features: torch.Tensor):
        special_image_mask = (pt_tokens == self.args.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(tokens_embd)
        image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
        return tokens_embd.masked_scatter(special_image_mask, image_features)

    def prepare_inputs_prefill(self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.

        For multimodal prompts, pass ``vision_embeddings`` (host tensor or list of tensors from
        :meth:`encode_vision_embeddings_from_pixels` / ``encode_vision_for_prefill``).

        If only ``pixel_values`` is set (e.g. vLLM / ``Generator.warmup_model_prefill`` vision pass),
        embeddings are computed here so callers need not go through ``GemmaMultimodalGenerator``.
        """

        S = pt_tokens.shape[-1]
        batch_rows = pt_tokens.shape[0]
        tokens = ttnn.from_torch(
            pt_tokens.reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # self.embed_scale = self.args.dim**0.5
        tokens_embd = self.embd(tokens)
        tokens_embd = ttnn.to_torch(tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))

        vision_embeddings = kwargs.pop("vision_embeddings", None)
        pixel_values = kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        kwargs.pop("image_sizes", None)

        if vision_embeddings is None and pixel_values is not None:
            pvs = pixel_values if isinstance(pixel_values, (list, tuple)) else [pixel_values]
            vision_embeddings = [
                self.encode_vision_embeddings_from_pixels(pv) if pv is not None else None for pv in pvs
            ]

        if vision_embeddings is not None:
            vision_embeddings = self._vision_embeddings_to_tensor(vision_embeddings, batch_rows)
            if vision_embeddings is not None:
                tokens_embd = self._fuse_vision_into_text_embeddings(pt_tokens, tokens_embd, vision_embeddings)

        tokens_embd = self.args.prepare_residual_tensor_prefill(
            tokens_embd,
        )

        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)
        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix_prefill.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix_prefill.shape[2]}"

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix_prefill[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix_prefill[:, :, start_pos : start_pos + S, :],
        ]

        tt_rot_mats_prefill_local = [
            self.rope_local_setup.cos_matrix_prefill[:, :, start_pos : start_pos + S, :],
            self.rope_local_setup.sin_matrix_prefill[:, :, start_pos : start_pos + S, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
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

    def sample_host(tt_input, temperature=0.6, top_p=0.08, on_host=True):
        vocab_size = tt_input.shape[-1]
        pt_input = tt_input[..., :vocab_size]
        # [B, 1, V] -> [B, V] so softmax / top-p / argmax are batch-correct (avoid blind squeeze())
        if pt_input.dim() == 3 and pt_input.shape[1] == 1:
            pt_input = pt_input.squeeze(1)

        if temperature > 0:
            probs = torch.softmax(pt_input / temperature, dim=-1)
            pt_out = sample_top_p(probs, top_p)
        else:
            pt_out = torch.argmax(pt_input, dim=-1)

        if pt_out.dim() == 0:
            pt_out = pt_out.unsqueeze(0)
        elif pt_out.dim() == 1 and pt_input.dim() >= 2 and pt_input.shape[0] > 1:
            pass  # [B] next tokens
        elif pt_out.dim() == 1:  # single sequence: re-add batch dim
            pt_out = pt_out.unsqueeze(0)
        return None, pt_out


class GemmaMultimodalGenerator(Generator):
    def encode_vision_for_prefill(self, pixel_values: list):
        if not hasattr(self.model[0], "encode_vision_embeddings_from_pixels"):
            raise TypeError(
                "GemmaMultimodalGenerator requires TtGemmaModel (multimodal). "
                "text_demo uses tt_transformers.Generator with a plain Transformer."
            )
        return [
            self.model[0].encode_vision_embeddings_from_pixels(pv) if pv is not None else None for pv in pixel_values
        ]

    def _prepare_multimodal_prefill_kwargs(self, **kwargs):
        if kwargs.get("vision_embeddings") is None and kwargs.get("pixel_values") is not None:
            kwargs = dict(kwargs)
            kwargs["vision_embeddings"] = self.encode_vision_for_prefill(kwargs["pixel_values"])
            kwargs.pop("pixel_values", None)
        return kwargs

    def prefill_forward_multimodal(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos=None,
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        kwargs = self._prepare_multimodal_prefill_kwargs(**kwargs)
        return super().prefill_forward_text(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
            **kwargs,
        )

    def prefill_forward(
        self,
        vision_images,
        vision_masks,
        tokens,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
        **kwargs,
    ):
        del vision_masks, xattn_caches, total_lens, cross_page_table
        return self.prefill_forward_multimodal(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            pixel_values=vision_images,
            **kwargs,
        )

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos=None,
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        return self.prefill_forward_multimodal(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
            **kwargs,
        )
