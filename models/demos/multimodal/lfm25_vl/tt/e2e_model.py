# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""End-to-end LFM2.5-VL model: hybrid (ShortConv/attention) text backbone + SigLIP2 vision
tower + multi-modal projector, with early-fusion image embedding (à la Gemma3).

``TtLfm25VlModel`` subclasses ``models.tt_transformers.tt.model.Transformer`` and reuses
its embedding / RoPE / KV-cache / sampling machinery unchanged. The only thing that
differs from a plain text model is which class builds each decoder layer: for the
duration of ``Transformer.__init__`` (and only then) the module-level ``TransformerBlock``
symbol inside ``models.tt_transformers.tt.model`` is monkeypatched to
``LfmDecoderLayer``, which internally decides -- per layer, from ``args.layer_types`` --
whether to build a real ``TransformerBlock`` (full attention layers) or the custom
``ShortConv`` path (conv layers). This lets ``Transformer.__init__``'s existing
``[TransformerBlock(...) for i in range(n_layers)]`` loop build the whole hybrid stack
completely unmodified.
"""

from __future__ import annotations

import torch

import models.tt_transformers.tt.model as tt_model_module
import ttnn
from models.demos.multimodal.lfm25_vl.tt.decoder import LfmDecoderLayer
from models.demos.multimodal.lfm25_vl.tt.vision_model import TtLfm25VlVisionModel
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer


class TtLfm25VlModel(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        use_host_vision: bool = False,
    ):
        original_transformer_block = tt_model_module.TransformerBlock
        tt_model_module.TransformerBlock = LfmDecoderLayer
        try:
            super().__init__(
                args,
                dtype,
                mesh_device,
                state_dict,
                weight_cache_path,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
            )
        finally:
            tt_model_module.TransformerBlock = original_transformer_block

        self.vision_model = TtLfm25VlVisionModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            tt_ccl=self.tt_ccl,
            dtype=dtype,
            configuration=args,
            weight_cache_path=weight_cache_path,
            use_host_vision=use_host_vision,
        )

    def reset_conv_states(self, batch_indices=None):
        """Clear ShortConv decode-state carried on host between independent generations."""
        for layer in self.layers:
            if not layer.is_attention_layer:
                layer.conv.reset_conv_state(batch_indices)

    def encode_vision_embeddings_from_pixels(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the vision tower + projector and return host-side embeddings for image-token fusion."""
        vision_output = self.vision_model(
            pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
        )
        comp_vision_output = ttnn.to_torch(
            vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        )
        # vision_output is [1, T, dim] (or [T, dim] after squeeze); take full token sequence
        if comp_vision_output.dim() == 3:
            return comp_vision_output[0]
        return comp_vision_output

    def _vision_embeddings_to_tensor(self, vision_embeddings, batch_rows: int) -> torch.Tensor | None:
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
        Inputs are torch tensors or python types. Returns ttnn tensors on device.

        For multimodal prompts, pass ``vision_embeddings`` (host tensor or list of tensors from
        :meth:`encode_vision_embeddings_from_pixels`) or raw ``pixel_values`` (encoded here).
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
        tokens_embd = self.embd(tokens)
        tokens_embd = ttnn.to_torch(tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))

        vision_embeddings = kwargs.pop("vision_embeddings", None)
        pixel_values = kwargs.pop("pixel_values", None)
        spatial_shapes = kwargs.pop("spatial_shapes", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)
        kwargs.pop("image_grid_thw", None)
        kwargs.pop("image_sizes", None)

        if vision_embeddings is None and pixel_values is not None:
            pvs = pixel_values if isinstance(pixel_values, (list, tuple)) else [pixel_values]
            shapes = spatial_shapes if isinstance(spatial_shapes, (list, tuple)) else [spatial_shapes] * len(pvs)
            masks = (
                pixel_attention_mask
                if isinstance(pixel_attention_mask, (list, tuple))
                else [pixel_attention_mask] * len(pvs)
            )
            vision_embeddings = [
                self.encode_vision_embeddings_from_pixels(pv, sh, mk) if pv is not None else None
                for pv, sh, mk in zip(pvs, shapes, masks)
            ]

        if vision_embeddings is not None:
            vision_embeddings = self._vision_embeddings_to_tensor(vision_embeddings, batch_rows)
            if vision_embeddings is not None:
                tokens_embd = self._fuse_vision_into_text_embeddings(pt_tokens, tokens_embd, vision_embeddings)

        tokens_embd = self.args.prepare_residual_tensor_prefill(tokens_embd)
        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        assert (
            self.rope_setup.cos_matrix_prefill.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix_prefill.shape[2]}"

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix_prefill[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix_prefill[:, :, start_pos : start_pos + S, :],
        ]
        tt_rot_mats_prefill_local = (
            [
                self.rope_local_setup.cos_matrix_prefill[:, :, start_pos : start_pos + S, :],
                self.rope_local_setup.sin_matrix_prefill[:, :, start_pos : start_pos + S, :],
            ]
            if getattr(self, "rope_local_setup", None) is not None
            else tt_rot_mats_prefill_global
        )

        tt_page_table = None
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        tt_chunk_page_table = None
        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return tokens_embd, tt_rot_mats_prefill_global, tt_rot_mats_prefill_local, tt_page_table, tt_chunk_page_table


class Lfm25VlMultimodalGenerator(Generator):
    def encode_vision_for_prefill(
        self,
        pixel_values: list,
        spatial_shapes=None,
        pixel_attention_mask=None,
    ):
        if not hasattr(self.model[0], "encode_vision_embeddings_from_pixels"):
            raise TypeError(
                "Lfm25VlMultimodalGenerator requires TtLfm25VlModel (multimodal). "
                "Use tt_transformers.Generator with a plain Transformer for text-only models."
            )
        shapes = spatial_shapes if isinstance(spatial_shapes, (list, tuple)) else [spatial_shapes] * len(pixel_values)
        masks = (
            pixel_attention_mask
            if isinstance(pixel_attention_mask, (list, tuple))
            else [pixel_attention_mask] * len(pixel_values)
        )
        return [
            self.model[0].encode_vision_embeddings_from_pixels(pv, sh, mk) if pv is not None else None
            for pv, sh, mk in zip(pixel_values, shapes, masks)
        ]

    def _prepare_multimodal_prefill_kwargs(self, **kwargs):
        if kwargs.get("vision_embeddings") is None and kwargs.get("pixel_values") is not None:
            kwargs = dict(kwargs)
            kwargs["vision_embeddings"] = self.encode_vision_for_prefill(
                kwargs["pixel_values"],
                spatial_shapes=kwargs.pop("spatial_shapes", None),
                pixel_attention_mask=kwargs.pop("pixel_attention_mask", None),
            )
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
