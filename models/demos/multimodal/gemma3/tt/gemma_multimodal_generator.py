# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gemma3 multimodal wrapper around tt_transformers ``Generator``.

Vision inputs are normalized here before delegating to the shared text-prefill
core, so Gemma3 exposes a multimodal-specific public entrypoint
"""

from __future__ import annotations

import torch

from models.common.sampling import SamplingParams
from models.tt_transformers.tt.generator import Generator


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
        enable_trace: bool = True,
        model_id_warmup=None,
        sampling_params: SamplingParams | None = None,
        start_pos: list[int] | None = None,
        return_hidden_states: bool = False,
        warmup_prefill: bool = True,
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
        enable_trace: bool = True,
        model_id_warmup=None,
        sampling_params: SamplingParams | None = None,
        start_pos: list[int] | None = None,
        return_hidden_states: bool = False,
        warmup_prefill: bool = True,
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
