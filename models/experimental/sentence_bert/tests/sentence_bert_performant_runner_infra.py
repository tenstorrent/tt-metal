# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULCMore actions

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import transformers
from loguru import logger
import transformers
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.experimental.sentence_bert.ttnn.ttnn_sentence_bert_model import TtnnSentenceBertModel
from models.experimental.sentence_bert.ttnn.common import custom_preprocessor, preprocess_inputs
from ttnn.model_preprocessing import preprocess_model_parameters


def load_reference_model(model_name, config):
    torch_model = transformers.BertModel.from_pretrained(model_name).eval()
    reference_model = BertModel(config).to(torch.bfloat16)
    reference_model.load_state_dict(torch_model.state_dict())
    return reference_model


def load_ttnn_model(device, torch_model, config):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_model = TtnnSentenceBertModel(parameters=parameters, config=config)
    return ttnn_model


class SentenceBERTPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        sequence_length,
        input_ids=None,
        extended_mask=None,
        token_type_ids=None,
        position_ids=None,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    ):
        torch.manual_seed(0)
        self.device = device
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        config = transformers.BertConfig.from_pretrained(model_name)
        self.torch_model = load_reference_model(model_name, config)
        if input_ids is None:
            print("value are taken")
            self.input_ids = torch.randint(
                low=0, high=config.vocab_size - 1, size=[self.batch_size, self.sequence_length], dtype=torch.int64
            )
            attention_mask = torch.ones(self.batch_size, self.sequence_length)
            self.extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
            self.token_type_ids = torch.zeros([self.batch_size, self.sequence_length], dtype=torch.int64)
            self.position_ids = torch.arange(0, self.sequence_length, dtype=torch.int64).unsqueeze(dim=0)
        else:
            self.input_ids = input_ids
            self.extended_mask = extended_mask
            self.token_type_ids = token_type_ids
            self.position_ids = position_ids

        self.torch_output = self.torch_model(
            self.input_ids,
            attention_mask=self.extended_mask,
            token_type_ids=self.token_type_ids,
            position_ids=self.position_ids,
        )

        self.ttnn_sentencebert_model = load_ttnn_model(self.device, self.torch_model, config)

    def setup_input(self):
        tt_inputs_host, self.ttnn_token_type_ids, self.ttnn_position_ids, self.ttnn_attention_mask = preprocess_inputs(
            self.input_ids, self.token_type_ids, self.position_ids, self.extended_mask, self.device
        )

        return tt_inputs_host

    def run(self):
        # print("before", self.input_ids.shape, self.extended_mask.shape, self.token_type_ids.shape, self.position_ids.shape)
        self.ttnn_output_tensor = self.ttnn_sentencebert_model(
            self.ttnn_input_ids,
            attention_mask=self.ttnn_attention_mask,
            token_type_ids=self.ttnn_token_type_ids,
            position_ids=self.ttnn_position_ids,
            device=self.device,
        )

    def validate(self, output_tensor=None, torch_output_tensor=None):
        ttnn_output_tensor = self.ttnn_output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(ttnn_output_tensor[0]).squeeze(dim=1)
        self.valid_pcc = 0.987
        self.pcc_passed, self.pcc_message = assert_with_pcc(
            torch_output_tensor.last_hidden_state, output_tensor, pcc=self.valid_pcc
        )

        logger.info(f"SentenceBERT - batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.ttnn_output_tensor[0])
