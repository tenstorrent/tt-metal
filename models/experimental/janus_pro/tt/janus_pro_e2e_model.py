# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Full end-to-end TT model for Janus-Pro-7B.

Composes the TT vision tower (``TtJanusProTransformerVision`` = SigLIP encoder +
aligner) with the LLaMA-style TT text decoder (the shared tt_transformers
``Transformer``, which ``test_lang_decoder.py`` already proves matches HF).

Mirrors ``models/demos/multimodal/gemma3/tt/gemma_e2e_model.py`` but:
  * the vision tower is Janus-owned and sits under state_dict prefix ``"model."``
    (it composes ``model.vision_model.`` + ``model.aligner.`` internally);
  * Janus uses a single (global) LLaMA RoPE, so ``prepare_inputs_prefill`` reuses
    the base ``Transformer`` rope/page-table logic and only injects the host-side
    image-feature fusion (``masked_scatter``) — the same fusion point Gemma uses.

Image tokens only exist during prefill, so decode is inherited unchanged.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.janus_pro.tt.janus_pro_vision_model import TtJanusProTransformerVision
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer


class TtJanusProModel(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        vision_dtype=ttnn.bfloat16,
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

        # Vision tower runs in its own dtype (bf16 gives better vision PCC than the
        # bf8 the 7B decoder needs to fit in memory).
        self.vision_model = TtJanusProTransformerVision(
            mesh_device=mesh_device,
            state_dict=state_dict,
            tt_ccl=self.tt_ccl,
            state_dict_prefix="model.",
            dtype=vision_dtype,
            configuration=args,
            weight_cache_path=args.weight_cache_path(vision_dtype),
        )

        # Janus HF config exposes image_token_id; fall back to the base ModelArgs field.
        self.image_token_id = getattr(args.hf_config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = getattr(args, "image_token_index", None)
        assert self.image_token_id is not None, "Could not resolve Janus image_token_id"

    # ---- vision ----
    def compute_vision_token(self, pixel_values):
        """Run the TT vision tower (encoder + aligner) on pixel_values (B, 3, H, W)."""
        return self.vision_model(pixel_values)

    def encode_vision_embeddings_from_pixels(self, pixel_values):
        """Run the vision tower and return host image features [num_image_tokens, projection_dim]."""
        vision_out = self.compute_vision_token(pixel_values)
        image_features = ttnn.to_torch(
            ttnn.from_device(vision_out),
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )[0]
        return image_features.reshape(-1, self.args.vision_projection_dim)

    # ---- fusion ----
    @staticmethod
    def _coalesce_vision_embeddings(vision_embeddings):
        """The generator path produces a list (one feature tensor per image); the direct
        path a single tensor. Coalesce to one [total_image_tokens, dim] tensor (or None)."""
        if isinstance(vision_embeddings, (list, tuple)):
            parts = [v for v in vision_embeddings if v is not None]
            return torch.cat(parts, dim=0) if parts else None
        return vision_embeddings

    def _fuse_vision_into_text_embeddings(self, pt_tokens, tokens_embd, image_features):
        # tokens_embd: torch [1, S, dim]; pt_tokens: torch [1, S]
        placeholder = pt_tokens == self.image_token_id
        n_positions = int(placeholder.sum())
        assert n_positions == image_features.shape[0], (
            f"image placeholder count ({n_positions}) != vision feature rows "
            f"({image_features.shape[0]}); check the prompt's image tokens / padding"
        )
        mask = placeholder.unsqueeze(-1).expand_as(tokens_embd)
        image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
        return tokens_embd.masked_scatter(mask, image_features)

    def prepare_inputs_prefill(self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """Same as the base, plus optional vision fusion.

        Pass ``pixel_values`` (torch (B,3,H,W), or a list) or precomputed
        ``vision_embeddings`` (host tensor / list from
        :meth:`encode_vision_embeddings_from_pixels`). With neither, this is a plain
        text prefill.

        NOTE: fusion assumes a single (non-chunked) prefill pass — the full prompt,
        including all image placeholders, is embedded at once. Chunked prefill of
        multimodal prompts is not supported here.
        """
        vision_embeddings = kwargs.pop("vision_embeddings", None)
        pixel_values = kwargs.pop("pixel_values", None)

        outs = super().prepare_inputs_prefill(
            pt_tokens, start_pos=start_pos, page_table=page_table, chunk_page_table=chunk_page_table, **kwargs
        )

        if vision_embeddings is None and pixel_values is None:
            return outs

        # Trace prefill returns raw token ids as outs[0], not embeddings — fusing would
        # corrupt them. We don't use traced multimodal prefill; fail loudly if attempted.
        assert not kwargs.get("trace_enabled", False), "Traced multimodal prefill is not supported"

        if vision_embeddings is None:
            pvs = pixel_values if isinstance(pixel_values, (list, tuple)) else [pixel_values]
            vision_embeddings = [
                self.encode_vision_embeddings_from_pixels(pv) if pv is not None else None for pv in pvs
            ]
        vision_embeddings = self._coalesce_vision_embeddings(vision_embeddings)
        if vision_embeddings is None:
            return outs

        # Reassemble the (column-sharded) token embeddings on host, fuse, and send back
        # with prepare_residual_tensor_prefill (the exact layout the decoder expects).
        S = pt_tokens.shape[-1]
        tokens_embd = ttnn.to_torch(outs[0], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        tokens_embd = tokens_embd.reshape(1, S, -1)[..., : self.args.dim]
        tokens_embd = self._fuse_vision_into_text_embeddings(pt_tokens, tokens_embd, vision_embeddings)

        fused = self.args.prepare_residual_tensor_prefill(tokens_embd)
        fused = ttnn.unsqueeze_to_4D(fused)
        return (fused, *outs[1:])


class JanusMultimodalGenerator(Generator):
    """Generator that threads ``pixel_values`` through prefill (Gemma-style)."""

    def encode_vision_for_prefill(self, pixel_values: list):
        if not hasattr(self.model[0], "encode_vision_embeddings_from_pixels"):
            raise TypeError("JanusMultimodalGenerator requires TtJanusProModel (multimodal).")
        return [
            self.model[0].encode_vision_embeddings_from_pixels(pv) if pv is not None else None for pv in pixel_values
        ]

    def _prepare_multimodal_prefill_kwargs(self, **kwargs):
        if kwargs.get("vision_embeddings") is None and kwargs.get("pixel_values") is not None:
            kwargs = dict(kwargs)
            kwargs["vision_embeddings"] = self.encode_vision_for_prefill(kwargs["pixel_values"])
            kwargs.pop("pixel_values", None)
        return kwargs

    def prefill_forward_multimodal(self, tokens, **kwargs):
        kwargs = self._prepare_multimodal_prefill_kwargs(**kwargs)
        return super().prefill_forward_text(tokens, **kwargs)

    def prefill_forward_text(self, tokens, **kwargs):
        return self.prefill_forward_multimodal(tokens, **kwargs)

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
