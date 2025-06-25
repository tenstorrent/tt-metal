# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.demos.sentence_bert.ttnn.common import custom_preprocessor
from models.demos.sentence_bert.ttnn.ttnn_sentence_bert_model import TtnnSentenceBertModel
from models.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


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
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.sequence_length = sequence_length
        config = transformers.BertConfig.from_pretrained(model_name)
        self.torch_model = load_reference_model(model_name, config)
        if input_ids is None:
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

    def setup_l1_sharded_input(self, input_ids=None, token_type_ids=None, position_ids=None, extended_mask=None):
        if is_wormhole_b0():
            grid_size = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        input_ids = self.input_ids if input_ids is None else input_ids
        token_type_ids = self.token_type_ids if token_type_ids is None else token_type_ids
        position_ids = self.position_ids if position_ids is None else position_ids
        extended_mask = self.extended_mask if extended_mask is None else extended_mask
        input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(
            shard_grid, (position_ids.shape[0], position_ids.shape[1]), ttnn.ShardOrientation.ROW_MAJOR
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        ttnn_token_type_ids = ttnn.from_torch(token_type_ids, dtype=ttnn.uint32)
        ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32)
        ttnn_attention_mask = ttnn.from_torch(extended_mask, dtype=ttnn.bfloat16)
        return input_ids, input_mem_config, ttnn_token_type_ids, ttnn_position_ids, ttnn_attention_mask

    def setup_dram_sharded_input(self, device):
        (
            tt_inputs_host,
            input_mem_config,
            ttnn_token_type_ids,
            ttnn_position_ids,
            ttnn_attention_mask,
        ) = self.setup_l1_sharded_input()
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            (self.position_ids.shape[0], self.position_ids.shape[1]),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )
        return (
            tt_inputs_host,
            sharded_mem_config_DRAM,
            input_mem_config,
            ttnn_token_type_ids,
            ttnn_position_ids,
            ttnn_attention_mask,
        )

    def run(self):
        self.ttnn_output_tensor = self.ttnn_sentencebert_model(
            self.ttnn_input_ids,
            attention_mask=self.ttnn_att_mask,
            token_type_ids=self.ttnn_token_ids,
            position_ids=self.ttnn_pos_ids,
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

        logger.info(
            f"SentenceBERT - batch_size={self.batch_size}, PCC={self.pcc_message}, act_dtype:{self.act_dtype}, weight_dtype:{self.weight_dtype}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.ttnn_output_tensor[0])
