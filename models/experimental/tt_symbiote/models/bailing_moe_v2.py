# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN BailingMoeV2 Model implementation."""

from typing import Optional, List

import torch
import ttnn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import MoeModelOutputWithPast

from models.experimental.tt_symbiote.core.module import TTNNModule


class MoeV2ModelOutputWithPast(MoeModelOutputWithPast):
    def __init__(self, mtp_hidden_states=None, **kwargs):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states


class TTNNBailingMoeV2Model(TTNNModule):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`BailingMoeV2DecoderLayer`]

    Args:
        config: BailingMoeV2Config
    """

    @staticmethod
    def from_torch(model):
        new_model = TTNNBailingMoeV2Model()
        new_model.model = model

        # Bypass tensor wrapping/unwrapping for decoder layers.
        # These sit under the HF BailingMoeV2Model (nn.Module), so
        # set_device() would give them _bypass_tensor_wrapping=False.
        # Bypassing is safe: no PyTorch ops touch hidden_states between
        # layer calls, and each layer's forward already works with raw
        # ttnn.Tensor objects.
        for layer in model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        # Also bypass the final norm layer
        if isinstance(model.norm, TTNNModule):
            model.norm._bypass_tensor_wrapping = True

        return new_model

    def call(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        ttnn_object = self
        self = self.model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        cache_position = kwargs.get("cache_position")

        # Convert cache_position to a TTNN int32 device tensor *before* it
        # reaches the traced TTNNBailingMoEDecoderLayer.
        #
        # TTNNBailingMoEDecoderLayer has _bypass_tensor_wrapping=True (child of
        # an untraced TTNNModule), so its module_run uses fast_unwrap_to_device
        # as the kwarg transform.  That function passes plain torch.Tensors
        # through unchanged, so a CPU cache_position would never be allocated as
        # a pre-allocated trace-buffer in _capture_trace.  Instead it would be
        # baked verbatim into trace_func_kwargs, and _get_cur_pos_device_tensor
        # would call ttnn.from_torch *inside* begin_trace_capture — permanently
        # baking the KV-write position from the capture step into the trace.
        # Every subsequent replay would then write to the same KV slot, producing
        # identical logits → repeating / garbled tokens.
        #
        # By converting here (outside any trace boundary) the value is a
        # ttnn.Tensor when it hits _capture_trace, gets properly pre-allocated as
        # a device trace buffer, and _copy_kwargs_to_trace_buffer updates it via
        # ttnn.copy before each execute_trace replay.
        if cache_position is not None and not isinstance(cache_position, ttnn.Tensor):
            if hasattr(cache_position, "ttnn_tensor") and cache_position.ttnn_tensor is not None:
                cache_position = cache_position.ttnn_tensor
            else:
                cp_torch = (
                    cache_position.cpu().to(torch.int32)
                    if isinstance(cache_position, torch.Tensor)
                    else torch.tensor(cache_position, dtype=torch.int32)
                )
                cache_position = ttnn.from_torch(
                    cp_torch.flatten(),
                    device=ttnn_object.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = list(input_ids.shape)[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = list(inputs_embeds.shape)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            input_ids = ttnn.from_torch(
                input_ids.cpu().to(torch.int32),
                device=ttnn_object.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
            )
            inputs_embeds = self.word_embeddings(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = ttnn.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])
            position_ids = ttnn.unsqueeze(position_ids, 0)
        else:
            position_ids = ttnn.from_torch(
                position_ids.cpu().to(torch.int32),
                device=ttnn_object.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
            )

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
            )

        # Pre-convert attention_mask to ttnn.Tensor for bypass-enabled decoder layers.
        # With _bypass_tensor_wrapping=True, fast_unwrap_to_device passes torch.Tensor
        # unchanged, but TTNNSDPAAttention needs ttnn.Tensor for on-device SDPA.
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            mesh_mapper = (
                ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
            )
            attention_mask = ttnn.from_torch(
                attention_mask,
                device=ttnn_object.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        for layer_idx, decoder_layer in enumerate(layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                )
            hidden_states = layer_outputs[0]

            # Update KV cache Python-side counters OUTSIDE the trace boundary.
            # During trace replay, execute_trace only replays device ops;
            # _seq_lengths increments inside paged_update_on_device / paged_fill_on_device
            # do NOT execute. By updating here, the counters advance correctly
            # in all phases (warmup, capture, replay).
            if past_key_values is not None and hasattr(past_key_values, "update_seq_length"):
                seq_len = inputs_embeds.shape[1]  # prefill: SEQ_LEN, decode: 1
                past_key_values.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        main_hidden_states = hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (main_hidden_states,)

        mtp_hidden_states = None

        if mtp_layers:
            for decoder_layer in mtp_layers:
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
                inputs_embeds = self.word_embeddings(input_ids)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        inputs_embeds,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        output_router_logits,
                        use_cache,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                        cache_position=cache_position,
                    )
                if mtp_hidden_states is None:
                    mtp_hidden_states = []
                hidden_states = layer_outputs[0]
                mtp_hidden_states.append(hidden_states)

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [main_hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeV2ModelOutputWithPast(
            last_hidden_state=main_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            mtp_hidden_states=mtp_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
