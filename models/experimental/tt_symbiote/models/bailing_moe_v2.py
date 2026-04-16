# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN BailingMoeV2 Model implementation."""

from typing import Optional, List
import time

import torch
import ttnn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeModelOutputWithPast

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DispatchManager


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
        new_model._bypass_tensor_wrapping = True
        for layer in model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        if isinstance(model.norm, TTNNModule):
            model.norm._bypass_tensor_wrapping = True
        if isinstance(getattr(model, "rotary_emb", None), TTNNModule):
            model.rotary_emb._bypass_tensor_wrapping = True
        if isinstance(getattr(model, "word_embeddings", None), TTNNModule):
            model.word_embeddings._bypass_tensor_wrapping = True
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

        if inputs_embeds is not None:
            batch_size, seq_length = list(inputs_embeds.shape)[:2]
        elif input_ids is not None:
            batch_size, seq_length = list(input_ids.shape)[:2]
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
            if isinstance(input_ids, torch.Tensor):
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
        elif isinstance(position_ids, torch.Tensor):
            position_ids = ttnn.from_torch(
                position_ids.cpu().to(torch.int32),
                device=ttnn_object.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
            )

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
            )

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

        hidden_states = inputs_embeds

        if self.num_nextn_predict_layers > 0:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        t_layers_begin = time.time()
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
                )
            hidden_states = layer_outputs[0]

            if past_key_values is not None and hasattr(past_key_values, "update_seq_length"):
                seq_len = inputs_embeds.shape[1]
                past_key_values.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        if isinstance(hidden_states, ttnn.Tensor):
            ttnn.synchronize_device(ttnn_object.device)
        t_layers_end = time.time()
        DispatchManager.record_timing(
            "TorchModules",
            "TTNNBailingMoeV2Model",
            "decoder_layers_device",
            {},
            t_layers_end - t_layers_begin,
        )

        hidden_states = self.norm(hidden_states)
        if isinstance(hidden_states, ttnn.Tensor):
            ttnn.synchronize_device(ttnn_object.device)
        t_norm_end = time.time()
        DispatchManager.record_timing(
            "TorchModules",
            "TTNNBailingMoeV2Model",
            "final_norm_device",
            {},
            t_norm_end - t_layers_end,
        )
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


class TTNNBailingMoeV2ForCausalLM(TTNNModule):
    @staticmethod
    def from_torch(causal_lm):
        new = TTNNBailingMoeV2ForCausalLM()
        new.causal_lm = causal_lm
        inner_model = causal_lm.model
        new.word_embeddings = inner_model.model.word_embeddings
        if isinstance(causal_lm.lm_head, TTNNModule):
            causal_lm.lm_head._bypass_tensor_wrapping = True
        new._perf_stats = {
            "prefill_count": 0,
            "prefill_total": 0.0,
            "prefill_preprocess": 0.0,
            "prefill_layers": 0.0,
            "prefill_postprocess": 0.0,
            "decode_count": 0,
            "decode_total": 0.0,
            "decode_preprocess": 0.0,
            "decode_layers": 0.0,
            "decode_postprocess": 0.0,
        }
        return new

    def reset_perf_stats(self):
        for k in self._perf_stats:
            self._perf_stats[k] = 0.0 if "count" not in k else 0

    def print_perf_stats(self):
        s = self._perf_stats
        print("\n=== Ling-mini Prefill / Decode Timing ===")
        if s["prefill_count"] > 0:
            print(f"  Prefill  ({s['prefill_count']} call(s)):")
            print(f"    Total          : {s['prefill_total']:.4f}s")
            print(f"    Preprocess     : {s['prefill_preprocess']:.4f}s")
            print(f"    Decoder layers : {s['prefill_layers']:.4f}s")
            print(f"    Postprocess    : {s['prefill_postprocess']:.4f}s")
        if s["decode_count"] > 0:
            avg = s["decode_total"] / s["decode_count"]
            print(f"  Decode   ({s['decode_count']} tokens):")
            print(f"    Total          : {s['decode_total']:.4f}s  ({avg*1000:.1f} ms/token)")
            avg_pre = s["decode_preprocess"] / s["decode_count"]
            avg_lay = s["decode_layers"] / s["decode_count"]
            avg_post = s["decode_postprocess"] / s["decode_count"]
            print(f"    Preprocess     : {s['decode_preprocess']:.4f}s  ({avg_pre*1000:.1f} ms/token)")
            print(f"    Decoder layers : {s['decode_layers']:.4f}s  ({avg_lay*1000:.1f} ms/token)")
            print(f"    Postprocess    : {s['decode_postprocess']:.4f}s  ({avg_post*1000:.1f} ms/token)")
        print("=========================================\n")

    def preprocess_input_tokens(self, input_ids, attention_mask, position_ids):
        device = self.device
        multi_device = device.get_num_devices() > 1
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if multi_device else None

        tt_input_ids = None
        inputs_embeds = None

        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                tt_input_ids = ttnn.from_torch(
                    input_ids.cpu().to(torch.int32),
                    device=device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                )
            else:
                tt_input_ids = input_ids
            inputs_embeds = self.word_embeddings(tt_input_ids)

        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            position_ids = ttnn.from_torch(
                position_ids.cpu().to(torch.int32),
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        return tt_input_ids, inputs_embeds, attention_mask, position_ids

    def postprocess_logits(self, hidden_states):
        if isinstance(hidden_states, ttnn.Tensor):
            shape = list(hidden_states.shape)
            if len(shape) >= 2 and shape[-2] > 1:
                starts = [0] * len(shape)
                ends = list(shape)
                starts[-2] = shape[-2] - 1
                hidden_states = ttnn.slice(hidden_states, starts, ends)

        logits = self.causal_lm.lm_head(hidden_states)

        if isinstance(logits, ttnn.Tensor):
            device = self.device
            num_devices = device.get_num_devices()

            local_max = ttnn.max(logits, dim=-1)
            local_idx = ttnn.argmax(logits, dim=-1)

            if num_devices > 1:
                mesh_composer = ttnn.ConcatMesh2dToTensor(device, device.shape, (0, -1))
                max_vals = ttnn.to_torch(local_max, mesh_composer=mesh_composer).float().flatten()
                idx_vals = ttnn.to_torch(local_idx, mesh_composer=mesh_composer).int().flatten()
            else:
                max_vals = ttnn.to_torch(local_max).float().flatten()
                idx_vals = ttnn.to_torch(local_idx).int().flatten()

            vocab_size = self.causal_lm.config.vocab_size
            if num_devices > 1:
                vocab_per_device = vocab_size // num_devices
                winning_device = max_vals.argmax().item()
                global_idx = winning_device * vocab_per_device + idx_vals[winning_device].item()
            else:
                global_idx = idx_vals[0].item()

            logits = torch.full((1, 1, vocab_size), float("-inf"))
            logits[0, 0, global_idx] = 1e9
            logits = logits.float()
        else:
            logits = logits.float()

        return logits

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=None,
        return_dict=None,
        **kwargs,
    ):
        begin = time.time()
        DispatchManager.set_current_module_name("BailingMoeV2ForCausalLM")

        config = self.causal_lm.config
        return_dict = return_dict if return_dict is not None else config.use_return_dict

        t0 = time.time()
        tt_input_ids = None
        if input_ids is not None and inputs_embeds is None:
            tt_input_ids, inputs_embeds, attention_mask, position_ids = self.preprocess_input_tokens(
                input_ids, attention_mask, position_ids
            )
        t1 = time.time()
        DispatchManager.record_timing(
            "TorchModules",
            "BailingMoeV2ForCausalLM",
            "preprocess_input_tokens",
            {},
            t1 - t0,
        )

        if past_key_values is not None and hasattr(past_key_values, "paged_fill_on_device"):
            attention_mask = None

        outputs = self.causal_lm.model(
            input_ids=tt_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,
            **kwargs,
        )
        if isinstance(outputs[0], ttnn.Tensor):
            ttnn.synchronize_device(self.device)
        t2 = time.time()
        DispatchManager.record_timing(
            "TorchModules",
            "BailingMoeV2ForCausalLM",
            "model_forward",
            {},
            t2 - t1,
        )

        hidden_states = outputs[0]
        logits = self.postprocess_logits(hidden_states)
        t3 = time.time()
        DispatchManager.record_timing(
            "TorchModules",
            "BailingMoeV2ForCausalLM",
            "postprocess_logits",
            {},
            t3 - t2,
        )

        end = time.time()
        DispatchManager.record_timing(
            "TorchModules",
            "BailingMoeV2ForCausalLM",
            "TTNNBailingMoeV2ForCausalLM_forward",
            {},
            end - begin,
        )
        DispatchManager.set_current_module_name(None)

        seq_len = (
            input_ids.shape[-1]
            if input_ids is not None
            else (inputs_embeds.shape[1] if inputs_embeds is not None else 1)
        )
        phase = "prefill" if seq_len > 1 else "decode"
        s = self._perf_stats
        s[f"{phase}_count"] += 1
        s[f"{phase}_total"] += end - begin
        s[f"{phase}_preprocess"] += t1 - t0
        s[f"{phase}_layers"] += t2 - t1
        s[f"{phase}_postprocess"] += t3 - t2

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def patch_forward(causal_lm, mesh_device):
        wrapper = TTNNBailingMoeV2ForCausalLM.from_torch(causal_lm)
        wrapper._device = mesh_device
        causal_lm.forward = wrapper.call
        return wrapper
