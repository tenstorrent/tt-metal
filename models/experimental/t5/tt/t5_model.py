# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
import torch
from torch import nn

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.experimental.t5.tt.t5_stack import TtT5Stack


class TtT5Model(nn.Module):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config, state_dict, device):
        super().__init__()

        self.config_use_cache = config["use_cache"] if "use_cache" in config else False
        self.config_use_return_dict = config["use_return_dict"] if "use_return_dict" in config else False

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

        self.config = config
        self.device = device

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,  # Optional[torch.LongTensor]
        attention_mask=None,  # Optional[torch.FloatTensor]
        decoder_input_ids=None,  # Optional[torch.LongTensor]
        decoder_attention_mask=None,  # Optional[torch.BoolTensor]
        head_mask=None,  # Optional[torch.FloatTensor]
        decoder_head_mask=None,  # Optional[torch.FloatTensor]
        cross_attn_head_mask=None,  # Optional[torch.Tensor]
        encoder_outputs=None,  # Optional[Tuple[Tuple[torch.FloatTensor]]]
        past_key_values=None,  # Optional[Tuple[Tuple[torch.FloatTensor]]]
        inputs_embeds=None,  # Optional[torch.Tensor]
        decoder_inputs_embeds=None,  # Optional[torch.Tensor]
        use_cache=None,  # Optional[bool]
        output_attentions=None,  # Optional[bool]
        output_hidden_states=None,  # Optional[bool]
        return_dict=None,
    ):  # Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]
        r"""
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, T5Model
        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)
        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        use_cache = use_cache if use_cache is not None else self.config_use_cache
        return_dict = return_dict if return_dict is not None else self.config_use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config["num_layers"] == self.config["num_decoder_layers"]:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

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

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
