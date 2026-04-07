# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Gemma4 Text Model implementation.

Replaces Gemma4TextModel to:
- Handle input_ids → embedding on device (trace-safe)
- Iterate decoder layers without ModuleList slicing
  (HF's self.layers[:N] reconstructs a ModuleList, failing for TTNNModule)
- Keep rotary embeddings and causal masks on host (unchanged)
"""

from typing import Optional

import torch
import ttnn
from transformers.modeling_outputs import BaseModelOutputWithPast

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor


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

        # Bypass tensor wrapping/unwrapping for decoder layers.
        # These sit under the HF Gemma4TextModel (nn.Module), so
        # set_device() would give them _bypass_tensor_wrapping=False.
        # Bypassing is safe: no PyTorch ops touch hidden_states between
        # layer calls, and each layer's forward already works with raw
        # ttnn.Tensor objects.
        for layer in model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        model.norm._bypass_tensor_wrapping = True
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

        # Pre-convert causal masks to ttnn.Tensor for bypass-enabled decoder layers.
        # With _bypass_tensor_wrapping=True, fast_unwrap_to_device passes torch.Tensor
        # unchanged, but TTNNSDPAAttention needs ttnn.Tensor for on-device SDPA.
        mesh_mapper = (
            ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
        )
        for mask_key, mask_val in causal_mask_mapping.items():
            if isinstance(mask_val, torch.Tensor):
                causal_mask_mapping[mask_key] = ttnn.from_torch(
                    mask_val,
                    device=ttnn_object.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mesh_mapper,
                )

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

        # Skip HF CPU rotary — TTNN attention uses BailingRotarySetup on-device (see rope.py).
        # TTNNGemma4DecoderLayer always passes position_embeddings=None to attention anyway.
        position_embeddings = {layer_type: None for layer_type in self.unique_layer_types}

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

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=TorchTTNNTensor(hidden_states),
            past_key_values=past_key_values,
        )
