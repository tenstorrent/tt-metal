# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Canonical **Mistral4 text stack** façade for demos and bring-up.

The real implementation lives in :mod:`model_backbone`, :mod:`causal_lm_prefill`, and
:mod:`causal_lm_incremental` (hybrid = HF weights + ttnn matmuls / SDPA on mesh). This module wraps those
APIs on a single HF ``Mistral4ForCausalLM`` instance so callers do not thread ``mesh_device`` through every call.

**Checkpoint loading:** FP8 hubs often fail ``from_pretrained`` on CPU-only hosts. Use
:func:`load_mistral4_for_causal_lm_bf16`, which retries with ``device_map="cuda"`` when a GPU is available.

**Multimodal (119B):** with env ``TT_METAL_MISTRAL4_HYBRID_TEXT``, ``simple_vision_demo`` builds
``Mistral4HybridMultimodalTransformer`` (TT vision + this backbone) and ``Generator`` runs prefill/decode
via :meth:`Mistral4HybridTextBackbone.incremental_logits` (including optional ``prefused_hidden_states_bsh``
for vision-fused embeddings on the first step).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from models.tt_transformers.tt.mistral_small_4.causal_lm_incremental import (
    causal_lm_greedy_generate_hybrid_bf16,
    causal_lm_incremental_step_logits_hybrid_bf16,
)
from models.tt_transformers.tt.mistral_small_4.causal_lm_prefill import (
    causal_lm_prefill_logits_hybrid_bf16,
    causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16,
)
from models.tt_transformers.tt.mistral_small_4.model_backbone import (
    language_model_backbone_hybrid_forward_bf16,
    language_model_backbone_hybrid_forward_incremental_bf16,
)


def load_mistral4_for_causal_lm_bf16(
    name_or_path: str,
    *,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
    device_map: str | dict | None = None,
):
    """
    Load ``Mistral4ForCausalLM`` in bf16 for the hybrid TT path.

    If ``device_map`` is omitted, tries CPU load first, then (on failure) ``device_map='cuda'`` when CUDA
    is available so HF can dequantize FP8 shards.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    common: dict[str, Any] = dict(
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    # Prefer CPU when available so small / local checkpoints do not occupy GPU; CUDA retry helps FP8 hubs.
    trials: list[dict[str, Any]] = []
    if device_map is not None:
        trials.append({**common, "device_map": device_map})
    trials.append(dict(common))
    if torch.cuda.is_available():
        trials.append({**common, "device_map": "cuda"})

    last_exc: Exception | None = None
    for kwargs in trials:
        try:
            model = Mistral4ForCausalLM.from_pretrained(name_or_path, **kwargs)
            return model.eval()
        except Exception as exc:  # noqa: BLE001 — try next strategy
            last_exc = exc
            continue
    assert last_exc is not None
    raise RuntimeError(
        "Could not load Mistral4ForCausalLM in bf16. FP8 checkpoints often require CUDA for HF dequant; "
        "or use a bf16 export / smaller test model. Last error: "
        f"{last_exc!r}"
    ) from last_exc


class Mistral4HybridTextBackbone:
    """
    HF ``Mistral4ForCausalLM`` + ttnn hybrid forwards (prefill, causal-mask prefill, KV incremental, greedy).

    This is the supported **text backbone** for Mistral4 (standalone or behind ``Mistral4HybridMultimodalTransformer``).
    """

    def __init__(
        self,
        mesh_device: Any,
        hf_model: torch.nn.Module,
        *,
        use_device_sdpa_attention: bool = False,
    ):
        from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

        if not isinstance(hf_model, Mistral4ForCausalLM):
            raise TypeError("Mistral4HybridTextBackbone expects transformers Mistral4ForCausalLM")
        self.mesh_device = mesh_device
        self.hf_model = hf_model
        self.use_device_sdpa_attention = bool(use_device_sdpa_attention)

    def last_hidden_prefill(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B,S,H]`` last hidden states (bf16 host), no LM head."""
        return language_model_backbone_hybrid_forward_bf16(
            self.mesh_device,
            input_ids,
            position_ids,
            self.hf_model.model,
            attention_mask_4d=attention_mask_4d,
            use_device_sdpa_attention=self.use_device_sdpa_attention,
        )

    def logits_prefill(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B,S,V]`` logits, no KV cache."""
        return causal_lm_prefill_logits_hybrid_bf16(
            self.mesh_device,
            input_ids,
            position_ids,
            self.hf_model,
            attention_mask_4d=attention_mask_4d,
            use_device_sdpa_attention=self.use_device_sdpa_attention,
        )

    def logits_prefill_with_hf_causal_mask(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        *,
        attention_mask_2d: torch.Tensor | None = None,
        or_mask_function=None,
        and_mask_function=None,
        block_sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prefill logits with HF-style causal mask (multimodal-ready hooks)."""
        return causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16(
            self.mesh_device,
            input_ids,
            position_ids,
            self.hf_model,
            attention_mask_2d=attention_mask_2d,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
            block_sequence_ids=block_sequence_ids,
            use_device_sdpa_attention=self.use_device_sdpa_attention,
        )

    def last_hidden_incremental(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        *,
        past_key_values=None,
        attention_mask_2d: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, Optional[object]]:
        """Backbone only; returns ``(hidden, past_key_values)`` matching :func:`language_model_backbone_hybrid_forward_incremental_bf16`."""
        return language_model_backbone_hybrid_forward_incremental_bf16(
            self.mesh_device,
            input_ids,
            position_ids,
            self.hf_model.model,
            past_key_values=past_key_values,
            attention_mask_2d=attention_mask_2d,
            use_cache=use_cache,
            use_device_sdpa_attention=self.use_device_sdpa_attention,
        )

    def incremental_logits(
        self,
        input_ids: torch.LongTensor,
        *,
        past_key_values=None,
        position_ids: torch.LongTensor | None = None,
        attention_mask_2d: torch.Tensor | None = None,
        prefused_hidden_states_bsh: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object]:
        """One forward slice with KV cache + LM head (see :func:`causal_lm_incremental_step_logits_hybrid_bf16`)."""
        return causal_lm_incremental_step_logits_hybrid_bf16(
            self.mesh_device,
            input_ids,
            self.hf_model,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask_2d=attention_mask_2d,
            use_device_sdpa_attention=self.use_device_sdpa_attention,
            prefused_hidden_states_bsh=prefused_hidden_states_bsh,
        )

    def greedy_generate(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int,
        *,
        attention_mask_2d: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Greedy decode (batch size 1 recommended when device SDPA is enabled)."""
        return causal_lm_greedy_generate_hybrid_bf16(
            self.mesh_device,
            prompt_ids,
            self.hf_model,
            max_new_tokens,
            attention_mask_2d=attention_mask_2d,
            use_device_sdpa_attention=self.use_device_sdpa_attention,
        )
