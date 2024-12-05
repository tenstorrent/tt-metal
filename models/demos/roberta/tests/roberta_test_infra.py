# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
import pytest
import torchvision
import transformers

from loguru import logger
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.bert.tt import ttnn_optimized_bert as ttnn_roberta
from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return bias


def custom_preprocessor(torch_model, name, mesh_mapper=None):
    import torch

    parameters = {}
    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        qkv_weight = torch.cat(
            [
                torch_model.query.weight,
                torch_model.key.weight,
                torch_model.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.query.bias, torch_model.key.bias, torch_model.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(
            qkv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(
            qkv_bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor


def preprocess_inputs(input_ids, token_type_ids, position_ids, attention_mask, device, mesh_mapper=None):
    import torch

    batch_size, _ = input_ids.shape
    mesh_mapper = None if device is None else mesh_mapper
    memory_config = None if device is None else ttnn.L1_MEMORY_CONFIG
    input_ids = ttnn.from_torch(
        input_ids, dtype=ttnn.uint32, device=device, memory_config=memory_config, mesh_mapper=mesh_mapper
    )
    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=memory_config, mesh_mapper=mesh_mapper
    )
    position_ids = ttnn.from_torch(
        position_ids, dtype=ttnn.uint32, device=device, memory_config=memory_config, mesh_mapper=mesh_mapper
    )
    if attention_mask is not None:
        attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, torch.float32)
        attention_mask = attention_mask.expand((batch_size, -1, -1, -1))
        attention_mask = torch.clamp(attention_mask, min=-100000)
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

    return input_ids, token_type_ids, position_ids, attention_mask


class RobertaTestInfra:
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
        self.sequence_size = sequence_size
        self.config = config

        torch_roberta = transformers.RobertaModel.from_pretrained(model_version)
        torch_roberta.eval()

        model_config = {
            "MATH_FIDELITY": math_fidelity,
            "WEIGHTS_DTYPE": weight_dtype,
            "ACTIVATIONS_DTYPE": act_dtype,
        }

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.sequence_size)).to(torch.int32)
        self.torch_token_type_ids = torch.zeros((self.batch_size, self.sequence_size), dtype=torch.int32)
        self.torch_attention_mask = torch.ones(self.batch_size, self.sequence_size)
        self.torch_position_ids = create_position_ids_from_input_ids(
            input_ids=self.input_ids, padding_idx=self.config.pad_token_id
        )

        # Golden
        torch_output = torch_roberta(
            input_ids=self.input_ids,
            attention_mask=self.torch_attention_mask,
            token_type_ids=self.torch_token_type_ids,
            position_ids=self.torch_position_ids,
        ).last_hidden_state

        # TTNN
        tt_model_name = f"ttnn_roberta_optimized"
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: torch_roberta,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=device,
        )

        ttnn_roberta_inputs = ttnn_roberta.preprocess_inputs(
            self.input_ids,
            self.torch_token_type_ids,
            self.torch_position_ids,
            self.torch_attention_mask,
            device=device,
        )

        self.ttnn_output = ttnn_roberta.bert(
            self.config,
            *ttnn_roberta_inputs,
            parameters=parameters,
        )

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

    def run(self, tt_input_tensor=None):
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.sequence_size)).to(torch.int32)
        self.torch_token_type_ids = torch.zeros((self.batch_size, self.sequence_size), dtype=torch.int32)
        self.torch_attention_mask = torch.ones(self.batch_size, self.sequence_size)
        self.torch_position_ids = create_position_ids_from_input_ids(
            input_ids=self.input_ids, padding_idx=self.config.pad_token_id
        )

        tt_model_name = f"ttnn_roberta_optimized"
        self.parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: transformers.RobertaModel.from_pretrained(
                "deepset/roberta-large-squad2", torchscript=False
            ).eval(),
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=self.device,
        )
        self.ttnn_roberta_inputs = ttnn_roberta.preprocess_inputs(
            self.input_ids,
            self.torch_token_type_ids,
            self.torch_position_ids,
            self.torch_attention_mask,
            device=self.device,
        )
        self.output_tensor = ttnn_roberta.bert(
            self.config,
            *self.ttnn_roberta_inputs,
            parameters=self.parameters,
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        # output_tensor = torch.reshape(output_tensor, (output_tensor.shape[0], 1000))

        batch_size = output_tensor.shape[0]

        valid_pcc = 1.0
        if self.batch_size >= 8:
            valid_pcc = golden_pcc[self.device.arch()][self.batch_size][
                (self.math_fidelity, self.weight_dtype, self.act_dtype)
            ]
        else:
            if self.act_dtype == ttnn.bfloat8_b:
                if self.math_fidelity == ttnn.MathFidelity.LoFi:
                    valid_pcc = 0.87
                else:
                    valid_pcc = 0.94
            else:
                if self.math_fidelity == ttnn.MathFidelity.LoFi:
                    valid_pcc = 0.93
                else:
                    valid_pcc = 0.982

        self.pcc_passed, self.pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(
            f"Roberta batch_size={batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, math_fidelity={self.math_fidelity}, PCC={self.pcc_message}"
        )

    def setup_l1_input(
        self,
        device,
        input_ids=None,
        torch_attention_mask=None,
        torch_token_type_ids=None,
        torch_position_ids=None,
        torch_input_tensor=None,
    ):
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        input_ids = self.input_ids if input_ids is None else input_ids
        torch_attention_mask = self.torch_attention_mask if torch_attention_mask is None else torch_attention_mask
        torch_token_type_ids = self.torch_token_type_ids if torch_token_type_ids is None else torch_token_type_ids
        torch_position_ids = self.torch_position_ids if torch_position_ids is None else torch_position_ids

        grid_size = ttnn.CoreGrid(y=5, x=6)
        num_cores = grid_size.x * grid_size.y
        shard_h = (input_ids.shape[0] + num_cores - 1) // num_cores

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
            [shard_h, input_ids.shape[-1]],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )

        ttnn_roberta_inputs_host = preprocess_inputs(
            input_ids,
            torch_token_type_ids,
            torch_position_ids,
            torch_attention_mask,
            device=None,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        return ttnn_roberta_inputs_host, input_mem_config

    def setup_dram_input(
        self,
        device,
        input_ids=None,
        torch_attention_mask=None,
        torch_token_type_ids=None,
        torch_position_ids=None,
        torch_input_tensor=None,
    ):
        input_ids = self.input_ids if input_ids is None else input_ids
        torch_attention_mask = self.torch_attention_mask if torch_attention_mask is None else torch_attention_mask
        torch_token_type_ids = self.torch_token_type_ids if torch_token_type_ids is None else torch_token_type_ids
        torch_position_ids = self.torch_position_ids if torch_position_ids is None else torch_position_ids

        tt_inputs_host, input_mem_config = self.setup_l1_input(
            device, input_ids, torch_attention_mask, torch_token_type_ids, torch_position_ids, torch_input_tensor
        )
        tt_input_ids_host, tt_token_type_ids_host, tt_position_ids_host, tt_attention_mask_host = tt_inputs_host

        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_input_ids_host.volume() // tt_input_ids_host.shape[-1], dram_grid_size.x),
                tt_input_ids_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, mem_config_DRAM, input_mem_config


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight=True,
    dealloc_input=True,
    final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    model_version=None,
    config=None,
    sequence_size=None,
):
    return RobertaTestInfra(
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
    )
