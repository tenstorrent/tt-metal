# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from typing import Optional, Tuple, Union
from loguru import logger
from dataclasses import dataclass

import ttnn

from models.experimental.roberta.tt.roberta_model import TtRobertaModel
from models.experimental.roberta.tt.roberta_lm_head import TtRobertaLMHead


@dataclass
class TtMaskedLMOutput:
    loss: ttnn.Tensor = None
    logits: ttnn.Tensor = None
    hidden_states: ttnn.Tensor = None


class TtRobertaForMaskedLM(nn.Module):
    def __init__(self, config, state_dict, base_address, device, reference_model):
        super().__init__()
        self.device = device
        self.config = config

        if base_address != "":
            base_address = base_address + "."

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.roberta = TtRobertaModel(
            config=config,
            state_dict=state_dict,
            base_address=f"{base_address}" + "roberta",
            device=device,
            reference_model=reference_model.roberta,
            add_pooling_layer=False,
        )

        self.lm_head = TtRobertaLMHead(
            config,
            state_dict,
            f"{base_address}" + "lm_head",
            device,
        )

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ttnn.Tensor], TtMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            """TODO: When implementing Training"""
            raise NotImplementedError
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return TtMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )
