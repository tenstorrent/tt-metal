# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Gemma4 Text Model implementation.

Replaces Gemma4TextModel to:
- Handle input_ids → embedding on device (trace-safe)
- Iterate decoder layers without ModuleList slicing
  (HF's self.layers[:N] reconstructs a ModuleList, failing for TTNNModule)
- Keep rotary embeddings and causal masks on host (unchanged)
"""

import logging
import os
from typing import Optional

import torch
import ttnn
from transformers.modeling_outputs import BaseModelOutputWithPast

from models.experimental.tt_symbiote.core.module import TTNNModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefill cross-request sync barrier
#
# Background: when this model is driven by an asynchronous serving stack
# (e.g. vLLM with ``read_from_device=False`` decode), the TTNN command queue
# accumulates work across all decode steps of one request and is only drained
# at the end of the request when its final token is read back to host. If the
# next request's prefill begins before that drain completes, the in-flight
# decode ops can race with the prefill ops issued by the new request.
#
# Empirically, this race manifests as a deterministic deadlock inside the
# ``ttnn.reshape`` after SDPA in
# ``TTNNGemma4Attention._forward_prefill`` (gemma4_attention.py:696). py-spy
# traces show the hang at ttnn/decorators.py:473 with 99% CPU on the
# dispatcher thread and no progress beyond the reshape. The hang is in the
# C++ TTNN runtime, not in Python, and so cannot be unblocked without a
# process kill.
#
# Until the underlying TTNN issue is fixed upstream, we bracket the prefill
# compute path with two ``ttnn.synchronize_device`` calls:
#   1. At the start of ``call()`` when seq_len > 1 (prefill): drains any
#      pending work from the previous request's async-decode pipeline.
#   2. After the decoder layer loop completes: drains any latent ops queued
#      during prefill so they cannot leak into the first decode of this same
#      request.
#
# Each sync costs ~0.7-1.1 ms on T3K (negligible compared to ~1 s prefill or
# ~430 ms decode); validated via 30/30 sequential requests on Gemma4-31B
# (median 60.5 s, p99 66.0 s) where the same workload previously deadlocked
# at request #24 without the barriers.
#
# The behaviour is ON by default and can be disabled via env var for
# benchmarking once the upstream TTNN fix lands:
#   TT_SYMBIOTE_GEMMA4_PREFILL_SYNC=1   enable barriers (default)
#   TT_SYMBIOTE_GEMMA4_PREFILL_SYNC=0   disable
# ---------------------------------------------------------------------------
_PREFILL_SYNC_ENABLED = os.environ.get("TT_SYMBIOTE_GEMMA4_PREFILL_SYNC", "1") == "1"


def _sync_device_safely(device, label: str) -> None:
    """Drain the TTNN command queue, never raising into the model's forward.

    ``ttnn.synchronize_device`` is a thin wrapper around the C++ runtime; if
    it fails for any reason (e.g. mesh teardown during shutdown) we log and
    continue rather than abort the inference call.
    """
    try:
        ttnn.synchronize_device(device)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ttnn.synchronize_device(%s) failed: %s", label, exc)


class TTNNGemma4TextModel(TTNNModule):
    """Replaces Gemma4TextModel (the language_model inside Gemma4Model).

    Follows the same pattern as TTNNBailingMoeV2Model: stores a reference to
    the original HF model, overrides ``call`` to run the forward pass with
    TTNN-replaced children, and handles the embedding input conversion that
    would otherwise break trace capture.
    """

    @staticmethod
    def from_torch(model):
        new_model = TTNNGemma4TextModel()
        new_model.model = model
        new_model._decode_cache_position = None
        return new_model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped HF model for HF compatibility.

        HF code may access config, embed_tokens.weight, etc. on the language_model.
        """
        # Check own __dict__ first (set by TTNNModule.__init__ and from_torch)
        try:
            return self.__dict__[name]
        except KeyError:
            pass
        # Delegate to the wrapped HF model
        return getattr(self.__dict__["model"], name)

    def call(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        ttnn_object = self
        self = self.model
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            # Convert input_ids to UINT32 on device for TTNN embedding lookup.
            # This is the host→device transfer that prevents embed_tokens from
            # being @trace_enabled. By doing it here in the model wrapper
            # (outside any trace boundary) we keep the decoder layer traces clean.
            input_ids_tt = ttnn.from_torch(
                input_ids.cpu().to(torch.int32),
                device=ttnn_object.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
            )
            inputs_embeds = self.embed_tokens(input_ids_tt)

        if use_cache and past_key_values is None:
            from transformers import DynamicCache

            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device="cpu") + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Build causal mask mapping (full_attention + sliding_attention)
        if not isinstance(attention_mask, dict):
            from transformers.models.gemma4.modeling_gemma4 import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds

        # Pre-convert cache_position to an on-device TTNN tensor (trace-safe).
        # This runs OUTSIDE any trace boundary, so ttnn.from_torch is allowed.
        #
        # IMPORTANT: For decode (single-token steps), we use a PERSISTENT device
        # buffer allocated once and updated in-place via ttnn.copy(). This prevents
        # the trace allocator from aliasing the buffer's device address with trace
        # intermediates. Without this, layer 0's trace replay can overwrite the
        # cache_position buffer, corrupting it for layers 1-59.
        # See PLAN_gemma4_traced_mode_root_cause.md for full analysis.
        cache_position = kwargs.get("cache_position")
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=torch.int32
            )
        if not isinstance(cache_position, ttnn.Tensor):
            cp = cache_position
            if hasattr(cp, "cpu"):
                cp = cp.cpu()
            if isinstance(cp, torch.Tensor):
                cp = cp.to(torch.int32)
            else:
                cp = torch.tensor(cp, dtype=torch.int32)
            mesh_mapper = (
                ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
            )
            is_decode = inputs_embeds.shape[1] == 1
            if is_decode and ttnn_object._decode_cache_position is not None:
                # Subsequent decode steps: copy new value into persistent buffer.
                cp_temp = ttnn.from_torch(
                    cp,
                    device=ttnn_object.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(cp_temp, ttnn_object._decode_cache_position)
                cache_position = ttnn_object._decode_cache_position
            else:
                cache_position = ttnn.from_torch(
                    cp,
                    device=ttnn_object.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if is_decode:
                    # First decode step: save as persistent buffer.
                    # This allocation happens BEFORE trace capture, so the
                    # trace allocator knows this address is in use.
                    ttnn_object._decode_cache_position = cache_position
        kwargs["cache_position"] = cache_position

        # Compute position embeddings per layer type (shared across layers)
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Cross-request sync barrier (entry). Drain the TTNN command queue
        # before starting prefill so any in-flight work from the previous
        # request's async-decode pipeline cannot race with this prefill's
        # ttnn.reshape after SDPA. See the module-level docstring on
        # _PREFILL_SYNC_ENABLED for the full rationale.
        is_prefill = inputs_embeds.shape[1] > 1
        if is_prefill and _PREFILL_SYNC_ENABLED:
            _sync_device_safely(ttnn_object.device, "before_prefill")

        # Iterate decoder layers directly — no slicing avoids ModuleList
        # reconstruction (HF's [:N] creates a new ModuleList that rejects TTNNModule).
        num_layers = self.config.num_hidden_layers
        for i, decoder_layer in enumerate(self.layers):
            if i >= num_layers:
                break

            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

            # Update KV cache Python-side counters OUTSIDE the trace boundary.
            # During trace replay, execute_trace only replays device ops;
            # _seq_lengths increments inside paged_update_on_device / paged_fill_on_device
            # do NOT execute. By updating here, the counters advance correctly
            # in all phases (warmup, capture, replay).
            if past_key_values is not None and hasattr(past_key_values, "update_seq_length"):
                seq_len = inputs_embeds.shape[1]  # prefill: SEQ_LEN, decode: 1
                past_key_values.update_seq_length(layer_idx=i, seq_len=seq_len)

        hidden_states = self.norm(hidden_states)

        # Cross-request sync barrier (exit). Drain any latent ops queued
        # during prefill so they cannot leak into the first decode step of
        # this same request (or a downstream caller's host-side post-
        # processing of `last_hidden_state`).
        if is_prefill and _PREFILL_SYNC_ENABLED:
            _sync_device_safely(ttnn_object.device, "after_prefill")

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
