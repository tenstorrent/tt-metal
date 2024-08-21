# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import warnings
from torch import nn
import ttnn
import json
from typing import Optional, Tuple

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.experimental.t5.tt.t5_stack import TtT5Stack
from loguru import logger
from transformers import T5ForConditionalGeneration


class Seq2SeqLMOutput:
    def __init__(
        self,
        loss=None,
        logits=None,
        past_key_values=None,
        decoder_hidden_states=None,
        decoder_attentions=None,
        cross_attentions=None,
        encoder_outputs=None,
    ):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.decoder_hidden_states = decoder_hidden_states
        self.decoder_attentions = decoder_attentions
        self.cross_attentions = cross_attentions
        self.encoder_outputs = encoder_outputs


def tuple_to_torch(tt_tuple):
    if tt_tuple is None:
        return None

    result = tuple()

    for tt_tensor in tt_tuple:
        result = result + (tt2torch_tensor(tt_tensor),)

    return result


def tuple_tuple_to_torch(inputs):
    if inputs is None:
        return None

    result = tuple()

    for tt_tuple in inputs:
        result = result + (tuple_to_torch(tt_tuple),)

    return result


class TtT5ForConditionalGeneration(nn.Module):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config, state_dict, device):
        super().__init__()

        self.config = config
        self.device = device
        self.model_dim = config["d_model"]
        self.config_use_cache = config["use_cache"] if "use_cache" in config else False
        self.config_use_return_dict = config["use_return_dict"] if "use_return_dict" in config else False
        self.main_input_name = "input_ids"

        # Re-use embedding layer from reference_module
        self.shared = nn.Embedding(config["vocab_size"], config["d_model"])
        self.shared.weight = nn.Parameter(state_dict["shared.weight"])

        encoder_config = copy.deepcopy(config)
        encoder_config["is_decoder"] = False
        encoder_config["use_cache"] = False
        encoder_config["is_encoder_decoder"] = False
        self.encoder = TtT5Stack(encoder_config, state_dict, "encoder", device, self.shared)

        if "num_decoder_layers" not in config:
            config["num_decoder_layers"] = config["num_layers"]

        decoder_config = copy.deepcopy(config)
        decoder_config["is_decoder"] = True
        decoder_config["is_encoder_decoder"] = False
        decoder_config["num_layers"] = config["num_decoder_layers"]
        self.decoder = TtT5Stack(decoder_config, state_dict, "decoder", device, self.shared)

        self.lm_head_weights = torch2tt_tensor(state_dict[f"lm_head.weight"], device)

        self.lm_head_weights = ttnn.transpose(self.lm_head_weights, -2, -1)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(
        self,
        input_ids=None,  # Optional[torch.LongTensor]
        attention_mask=None,  # Optional[torch.FloatTensor]
        decoder_input_ids=None,  # Optional[torch.LongTensor]
        decoder_attention_mask=None,  # Optional[torch.BoolTensor]
        head_mask=None,  # Optional[torch.FloatTensor]
        decoder_head_mask=None,  # Optional[torch.FloatTensor]
        cross_attn_head_mask=None,  # Optional[torch.Tensor]
        encoder_outputs=None,  # Optional[Tuple[Tuple[torch.Tensor]]]
        past_key_values=None,  # Optional[Tuple[Tuple[torch.Tensor]]]
        inputs_embeds=None,  # Optional[torch.FloatTensor]
        decoder_inputs_embeds=None,  # Optional[torch.FloatTensor]
        labels=None,  # Optional[torch.LongTensor]
        use_cache=None,  # Optional[bool]
        output_attentions=None,  # Optional[bool]
        output_hidden_states=None,
        return_dict=None,
    ):  # Optional[bool]
        use_cache = use_cache if use_cache is not None else self.config_use_cache
        return_dict = return_dict if return_dict is not None else self.config_use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config["num_layers"] == self.config["num_decoder_layers"]:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        if return_dict:
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs[0]

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            logger.debug(f"_shift_right(labels)")
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #     hidden_states = hidden_states.to(self.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(self.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            sequence_output = decoder_outputs.last_hidden_state
        else:
            sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config["tie_word_embeddings"]:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = tt2torch_tensor(sequence_output)
            sequence_output = sequence_output * (self.model_dim**-0.5)
            sequence_output = torch2tt_tensor(sequence_output, self.device)

        lm_logits = ttnn.matmul(sequence_output, self.lm_head_weights)
        loss = None

        # Back to torch
        lm_logits = tt2torch_tensor(lm_logits)
        lm_logits = lm_logits[:, 0, :, :]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_outputs=encoder_outputs,
        )


def _t5_for_conditional_generation(config, state_dict, device) -> TtT5ForConditionalGeneration:
    return TtT5ForConditionalGeneration(config=config, state_dict=state_dict, device=device)


def t5_small_for_conditional_generation(device) -> TtT5ForConditionalGeneration:
    hf_reference_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    hf_reference_model.eval()

    generation_config = hf_reference_model.generation_config
    config = json.loads(hf_reference_model.config.to_json_string())
    config["tie_word_embeddings"] = hf_reference_model.config.tie_word_embeddings

    return (
        _t5_for_conditional_generation(config, hf_reference_model.state_dict(), device),
        hf_reference_model,
    )


def t5_base_for_conditional_generation(device) -> TtT5ForConditionalGeneration:
    hf_reference_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    hf_reference_model.eval()

    generation_config = hf_reference_model.generation_config
    config = json.loads(hf_reference_model.config.to_json_string())
    config["tie_word_embeddings"] = hf_reference_model.config.tie_word_embeddings

    return (
        _t5_for_conditional_generation(config, hf_reference_model.state_dict(), device),
        hf_reference_model,
    )


def flan_t5_small_for_conditional_generation(device) -> TtT5ForConditionalGeneration:
    hf_reference_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    hf_reference_model.eval()

    generation_config = hf_reference_model.generation_config
    config = json.loads(hf_reference_model.config.to_json_string())
    config["tie_word_embeddings"] = hf_reference_model.config.tie_word_embeddings

    return (
        _t5_for_conditional_generation(config, hf_reference_model.state_dict(), device),
        hf_reference_model,
    )
