from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F

from libs import tt_lib as ttm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
import numpy as np
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.baddbmm
import python_api_testing.models.bloom.bloom_attention as bloom_attention
import python_api_testing.models.bloom.bloom_mlp as bloom_mlp
import python_api_testing.models.bloom.bloom_block as bloom_block

import python_api_testing.models.bloom.bloom_model as bloom_model

from fused_ops.linear import Linear as TtLinear

from fused_ops.layernorm import Layernorm as TtLayernorm

from fused_ops.softmax import softmax as TtSoftmax
from transformers import BloomForQuestionAnswering

from typing import Optional, Tuple, Union


class PtBloomForQuestionAnswering():

    def __init__(self, hugging_bloom_reference_model, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers):

        state_dict = hugging_bloom_reference_model.state_dict()
        self.transformer = bloom_model.BloomModel(hugging_bloom_reference_model, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers)

        qa_outputs_weight = bloom_utils.pt_load_layer_weights("qa_outputs.weight", state_dict)
        qa_outputs_bias = bloom_utils.pt_load_layer_weights("qa_outputs.bias", state_dict)

        self.qa_outputs = torch.nn.Linear(hidden_size, 2)
        self.qa_outputs.weight = qa_outputs_weight
        self.qa_outputs.weight = qa_outputs_weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        self.use_return_dict = False
        """
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]
        print('PT--------------------SASHAPE1')

        print(sequence_output.shape)

        logits = self.qa_outputs(sequence_output)
        print('PT--------------------SASHAPE')
        print(logits.shape)


        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TtBloomForQuestionAnswering():

    def __init__(self, device, hugging_bloom_reference_model, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers):

        state_dict = hugging_bloom_reference_model.state_dict()
        self.transformer = bloom_model.TtBloomModel(device, hugging_bloom_reference_model, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers)

        qa_outputs_weight = bloom_utils.tt_load_layer_weights("qa_outputs.weight", state_dict)
        qa_outputs_bias = bloom_utils.tt_load_layer_weights("qa_outputs.bias", state_dict)

        self.qa_outputs = TtLinear(hidden_size, 32, qa_outputs_weight, qa_outputs_bias, device)

    def forward(
        self,
        device,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        self.use_return_dict = False
        """
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.transformer.forward(device, input_ids, attention_mask=attention_mask,position_ids=position_ids,head_mask=head_mask,inputs_embeds=inputs_embeds,output_attentions=output_attentions,output_hidden_states=output_hidden_states,return_dict=return_dict)

        sequence_output = outputs[0]
        print('SA SHAPE-----')
        print(sequence_output.shape())


        tt_logits = self.qa_outputs(sequence_output)

        print(tt_logits.shape())

        logits = bloom_utils.tt2torch_tensor(tt_logits)
        print(logits.shape)

        logits = logits.squeeze(0)
        logits = logits.squeeze(0)
        logits = logits[:,0:2]

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def run_bloom_qa_inference(device):
    hugging_bloom_reference_model = BloomForQuestionAnswering.from_pretrained("bigscience/bloom-560m", torchscript=False)

    print(hugging_bloom_reference_model.state_dict())
    #tt_bloom_model = TtBloomModel(device, hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)
    pt_bloom_qa = PtBloomForQuestionAnswering(hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)
    tt_bloom_qa = TtBloomForQuestionAnswering(device, hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)

    # Prepare input
    torch.manual_seed(0)

    input_ids = torch.randint(0, 100, (1, 32))

    pt_out = pt_bloom_qa.forward(input_ids)

    #print(pt_out[0])
    print("PT finished")

    tt_out = tt_bloom_qa.forward(device, input_ids)

    print("TT finished")

    pt_out = pt_out[0]
    tt_out = tt_out[0]
    tt_out = tt_out.squeeze(0)
    tt_out = tt_out.squeeze(0)

    #tt_out = tt_out.reshape(pt_out.shape)

    print(comp_allclose(pt_out, tt_out))
    print(comp_pcc(pt_out, tt_out))

if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_qa_inference(device)
    ttm.device.CloseDevice(device)
