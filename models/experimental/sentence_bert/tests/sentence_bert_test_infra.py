# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import transformers
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0
from models.experimental.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.experimental.sentence_bert.ttnn.ttnn_sentence_bert_model import TtnnSentenceBertModel
from models.experimental.sentence_bert.ttnn.common import custom_preprocessor, preprocess_inputs
from ttnn.model_preprocessing import preprocess_model_parameters


def load_reference_model(config):
    torch_model = transformers.AutoModel.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr").eval()
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


class SentenceBERTTestInfra:
    def __init__(self, device, batch_size, sequence_length):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        config = transformers.BertConfig.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        input_ids = torch.randint(
            low=0, high=config.vocab_size - 1, size=[self.batch_size, self.sequence_length], dtype=torch.int64
        )
        attention_mask = torch.ones(self.batch_size, self.sequence_length)
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = torch.zeros([self.batch_size, self.sequence_length], dtype=torch.int64)
        position_ids = torch.arange(0, self.sequence_length, dtype=torch.int64).unsqueeze(dim=0)
        self.torch_input_tensor = input_ids
        reference_model = load_reference_model(config)
        (
            self.ttnn_input_ids,
            self.ttnn_token_type_ids,
            self.ttnn_position_ids,
            self.ttnn_attention_mask,
        ) = preprocess_inputs(input_ids, token_type_ids, position_ids, extended_mask, device)
        self.ttnn_model = load_ttnn_model(self.device, reference_model, config)
        torch_out = reference_model(
            input_ids, attention_mask=extended_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )
        self.torch_output_tensor_1, self.torch_output_tensor_2 = torch_out.last_hidden_state, torch_out.pooler_output

    def run(self):
        self.output_tensor_1 = self.ttnn_model(
            self.input_tensor,
            self.ttnn_attention_mask,
            self.ttnn_token_type_ids,
            self.ttnn_position_ids,
            device=self.device,
        )

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        torch_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32)
        input_memory_config = ttnn.create_sharded_memory_config(
            torch_input_tensor.shape,
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        return torch_input_tensor, input_memory_config

    def validate(self, output_tensor=None):
        output_tensor_1 = ttnn.to_torch(self.output_tensor_1[0]).squeeze(dim=1)
        valid_pcc = 0.98
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor_1, output_tensor_1, pcc=valid_pcc)

        logger.info(f"sentence bert batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor_1[0])


def create_test_infra(device, batch_size, sequence_length):
    return SentenceBERTTestInfra(device, batch_size, sequence_length)
