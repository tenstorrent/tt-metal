# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
import torch
import torchvision
import transformers
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.bert_tiny.tt.bert_tiny import bert_for_question_answering
from transformers import BertForQuestionAnswering, BertConfig


class BertTinyTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        model_version,
        config,
        sequence_size,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.math_fidelity = math_fidelity
        self.dealloc_input = dealloc_input
        self.final_output_mem_config = final_output_mem_config
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.config = config

        torch_bert_tiny = BertForQuestionAnswering.from_pretrained(model_version, config=config).eval()
        torch_bert_tiny.eval()

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_bert_tiny,
            device=device,
            convert_to_ttnn=lambda *_: True,
        )

        self.torch_input_tensor = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
        self.torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
        self.torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
        self.torch_attention_mask = torch.zeros(1, sequence_size)

        self.torch_output = torch_bert_tiny(
            self.torch_input_tensor,
            token_type_ids=self.torch_token_type_ids,
            position_ids=self.torch_position_ids,
            attention_mask=self.torch_attention_mask,
        )

        model_config = {
            "MATH_FIDELITY": math_fidelity,
            "WEIGHTS_DTYPE": weight_dtype,
            "ACTIVATIONS_DTYPE": act_dtype,
        }

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

        self.bert_tiny_model = bert_for_question_answering

        self.ops_parallel_config = {}

    def get_mesh_mappers(self, device):
        is_mesh_device = isinstance(device, ttnn.MeshDevice)
        if is_mesh_device:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None  # ttnn.ReplicateTensorToMesh(device) causes unnecessary replication/takes more time on the first pass
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)

        batch_size = output_tensor.shape[0]

        valid_pcc = 1.0

        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(
            f"Bert-Tiny batch_size={batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, math_fidelity={self.math_fidelity}, PCC={self.pcc_message}"
        )

    def setup_inputs(
        self,
        device,
        torch_input_tensor=None,
        torch_token_type_ids=None,
        torch_position_ids=None,
        torch_attention_mask=None,
    ):
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        torch_token_type_ids = self.torch_token_type_ids if torch_token_type_ids is None else torch_token_type_ids
        torch_position_ids = self.torch_position_ids if torch_position_ids is None else torch_position_ids
        torch_attention_mask = self.torch_attention_mask if torch_attention_mask is None else torch_attention_mask

        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        tt_token_type_ids_host = ttnn.from_torch(
            torch_token_type_ids, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        tt_position_ids_host = ttnn.from_torch(
            torch_position_ids, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        tt_attention_mask_host = ttnn.from_torch(
            torch_attention_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )

        return tt_inputs_host, tt_token_type_ids_host, tt_position_ids_host, tt_attention_mask_host

    def run(self, tt_input_tensor=None):
        self.output_tensor = self.bert_tiny_model(
            self.config,
            self.input_tensor,
            self.token_type_ids,
            self.position_ids,
            self.attention_mask,
            parameters=self.parameters,
            device=self.device,
        )
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    config,
    sequence_size,
    use_pretrained_weight=True,
    dealloc_input=True,
    final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    model_location_generator=None,
):
    return BertTinyTestInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        "mrm8488/bert-tiny-finetuned-squadv2",
        config,
        sequence_size,
    )
