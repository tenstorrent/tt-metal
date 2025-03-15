# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
import torch
import torchvision
import transformers
import copy
from torch import nn

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, ParameterDict, ParameterList
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.squeezebert.tt import ttnn_functional_squeezebert


def custom_preprocessor(model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(model, nn.Conv1d):
        weight = model.weight
        bias = model.bias
        while bias.dim() < 4:
            bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        parameters["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    elif isinstance(model, nn.Embedding):
        parameters[f"weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)

    elif isinstance(model, nn.LayerNorm):
        weight = model.weight.reshape((1, -1))
        bias = model.bias.reshape((1, -1))
        parameters[f"weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters[f"bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    elif isinstance(model, nn.Linear):
        weight = model.weight.T.contiguous()
        parameters[f"weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if model.bias is not None:
            bias = model.bias.reshape((1, -1))
            parameters[f"bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor


def load_squeezebert_model(model_location_generator):
    model_name = "squeezebert/squeezebert-uncased"
    torch_squeezebert = transformers.SqueezeBertModel.from_pretrained(model_name)
    return torch_squeezebert


def get_mesh_mappers(device):
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    if is_mesh_device:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["query", "key", "value", "conv1d"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


class SqueezeBertTestInfra:
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
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = get_mesh_mappers(device)
        self.sequence_size = 384

        torch_model = load_squeezebert_model(model_location_generator).eval()
        self.config = torch_model.config
        self.state_dict = torch_model.state_dict()
        model_config = {
            "MATH_FIDELITY": math_fidelity,
            "WEIGHTS_DTYPE": weight_dtype,
            "ACTIVATIONS_DTYPE": act_dtype,
        }
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.sequence_size)).to(torch.int32)
        self.torch_token_type_ids = torch.zeros((self.batch_size, self.sequence_size), dtype=torch.int32)
        self.position_ids = torch.zeros((self.batch_size, self.sequence_size), dtype=torch.int32)
        self.torch_attention_mask = torch.ones(1, self.sequence_size)

        # Golden
        self.torch_output = torch_model(
            input_ids=self.input_ids,
            token_type_ids=self.torch_token_type_ids,
            position_ids=self.position_ids,
            attention_mask=self.torch_attention_mask,
        ).last_hidden_state

        # TTNN
        tt_model_name = f"ttnn_squeezebert"
        is_mesh_device = isinstance(device, ttnn.MeshDevice)
        if is_mesh_device:
            with ttnn.distribute(ttnn.ReplicateTensorToMesh(self.device)):
                self.parameters = preprocess_model_parameters(
                    model_name=tt_model_name,
                    initialize_model=lambda: torch_model,
                    custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
                )
        else:
            self.parameters = preprocess_model_parameters(
                model_name=tt_model_name,
                initialize_model=lambda: torch_model,
                custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            )
            self.parameters = move_to_device(self.parameters, self.device)

        (
            self.tt_input_ids,
            self.tt_token_type_ids,
            self.tt_position_ids,
            self.tt_attention_mask,
        ) = ttnn_functional_squeezebert.preprocess_inputs(
            self.input_ids,
            self.torch_token_type_ids,
            self.position_ids,
            self.torch_attention_mask,
            device=self.device,
        )

        self.ttnn_model = ttnn_functional_squeezebert.squeezebert
        self.ops_parallel_config = {}

    def setup_l1_sharded_input(
        self, device, tt_input_ids=None, tt_token_type_ids=None, tt_position_ids=None, tt_attention_mask=None
    ):
        core_grid = ttnn.CoreGrid(y=4, x=3)
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

        input_ids = self.tt_input_ids if tt_input_ids is None else tt_input_ids
        tt_token_type_ids = self.tt_token_type_ids if tt_token_type_ids is None else tt_token_type_ids
        tt_position_ids = self.tt_position_ids if tt_position_ids is None else tt_position_ids
        tt_attention_mask = self.tt_attention_mask if tt_attention_mask is None else tt_attention_mask

        if num_devices > 1:
            h, w = input_ids.shape
            h = h // num_devices
        else:
            h, w = input_ids.shape

        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_w = (w + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )

        return input_ids, tt_token_type_ids, tt_position_ids, tt_attention_mask, input_mem_config

    def setup_dram_sharded_input(
        self, device, input_ids=None, torch_token_type_ids=None, position_ids=None, torch_attention_mask=None
    ):
        input_ids = self.input_ids if input_ids is None else input_ids
        torch_token_type_ids = self.torch_token_type_ids if torch_token_type_ids is None else torch_token_type_ids
        position_ids = self.position_ids if position_ids is None else position_ids
        torch_attention_mask = self.torch_attention_mask if torch_attention_mask is None else torch_attention_mask

        (
            tt_input_ids,
            tt_token_type_ids,
            tt_position_ids,
            tt_attention_mask,
            input_mem_config,
        ) = self.setup_l1_sharded_input(device, input_ids, torch_token_type_ids, position_ids, torch_attention_mask)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_input_ids.volume() // tt_input_ids.shape[-1], dram_grid_size.x),
                tt_input_ids.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )
        return (
            tt_input_ids,
            tt_token_type_ids,
            tt_position_ids,
            tt_attention_mask,
            sharded_mem_config_DRAM,
            input_mem_config,
        )

    def run(self, tt_input_tensor=None):
        self.output_tensor = self.ttnn_model(
            self.config,
            input_ids=self.tt_input_ids,
            token_type_ids=self.tt_token_type_ids,
            position_ids=self.tt_position_ids,
            attention_mask=self.tt_attention_mask,
            state_dict=self.state_dict,
            parameters=self.parameters,
            device=self.device,
            reader_patterns_cache={},
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, mesh_composer=self.output_mesh_composer)

        valid_pcc = 0.78  # 0.93 #0.94 # 0.98
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output, output_tensor, pcc=valid_pcc)

        logger.info(f"SqueezeBERT batch_size={self.batch_size}, PCC={self.pcc_message}")


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight=False,
    dealloc_input=True,
    final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    model_location_generator=None,
):
    return SqueezeBertTestInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        model_location_generator,
    )
