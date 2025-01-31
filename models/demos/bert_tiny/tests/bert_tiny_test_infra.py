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

        self.torch_output_tensor = torch_bert_tiny(
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

    # memory_config = ttnn.create_sharded_memory_config(
    #     x.shape,
    #     core_grid=ttnn.CoreGrid(y=mesh_device.core_grid.y, x=mesh_device.core_grid.x),
    #     strategy=ttnn.ShardStrategy.BLOCK,
    #     # orientation=ttnn.ShardOrientation.TILE,
    # )
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

    def setup_l1_sharded_input(
        self,
        device,
        torch_input_tensor=None,
        torch_token_type_ids=None,
        torch_position_ids=None,
        torch_attention_mask=None,
    ):
        if self.batch_size == 16:
            core_grid = ttnn.CoreGrid(y=2, x=2)
        elif self.batch_size == 20:
            if is_grayskull():
                core_grid = ttnn.CoreGrid(y=8, x=10)
            elif is_wormhole_b0():
                core_grid = ttnn.CoreGrid(y=5, x=6)  # untested due to unsupported batch20 on WH
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        print("Torch tensor in L1 :", torch_input_tensor)
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        torch_token_type_ids = self.torch_token_type_ids if torch_token_type_ids is None else torch_token_type_ids
        torch_position_ids = self.torch_position_ids if torch_position_ids is None else torch_position_ids
        torch_attention_mask = self.torch_attention_mask if torch_attention_mask is None else torch_attention_mask
        if num_devices > 1:
            h, w = torch_input_tensor.shape
            h = h // num_devices
        else:
            h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        core_grid = ttnn.CoreGrid(y=2, x=2)
        num_cores = core_grid.x * core_grid.y
        shard_w = (w + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        print("Shape of tensor :", torch_input_tensor.shape, " ", shard_w)
        shard_spec = ttnn.ShardSpec(shard_grid, (h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        tt_token_type_ids = ttnn.from_torch(
            torch_token_type_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        tt_position_ids = ttnn.from_torch(
            torch_position_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        tt_attention_mask = ttnn.from_torch(
            torch_attention_mask, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        return tt_inputs_host, tt_token_type_ids, tt_position_ids, tt_attention_mask, input_mem_config

    def setup_dram_sharded_input(
        self,
        device,
        torch_input_tensor=None,
        torch_token_type_ids=None,
        torch_position_ids=None,
        torch_attention_mask=None,
    ):
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        torch_token_type_ids = self.torch_token_type_ids if torch_input_tensor is None else torch_token_type_ids
        torch_position_ids = self.torch_position_ids if torch_input_tensor is None else torch_position_ids
        torch_attention_mask = self.torch_attention_mask if torch_input_tensor is None else torch_attention_mask
        (
            tt_inputs_host,
            tt_token_type_ids,
            tt_position_ids,
            tt_attention_mask,
            input_mem_config,
        ) = self.setup_l1_sharded_input(
            device, torch_input_tensor, torch_token_type_ids, torch_position_ids, torch_attention_mask
        )

        dram_grid_size = device.dram_grid_size()
        print("Dram grid size :", dram_grid_size)
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 9, dram_grid_size.y - 1))}
            ),
            [
                tt_inputs_host.shape[0],
                # divup(tt_inputs_host.volume() // tt_inputs_host.shape[0], dram_grid_size.x),
                32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        """
        dram_grid_size = ttnn.CoreGrid(y=2, x=2)
        print("Dram size :", dram_grid_size)
        h, w = torch_input_tensor.shape

        num_cores = dram_grid_size.x * dram_grid_size.y
        shard_w = (w + num_cores - 1) // num_cores
        #grid_size = dram_grid_size
        #grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        #shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        print("Shape of tensor :", torch_input_tensor.shape," ",shard_w)
        #shard_spec = ttnn.ShardSpec(shard_grid, (h, shard_w),ttnn.ShardOrientation.ROW_MAJOR)

        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0),(dram_grid_size.x-1,dram_grid_size.y-1 ))}
            ),
            (h, shard_w),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        """
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return (
            tt_inputs_host,
            tt_token_type_ids,
            tt_position_ids,
            tt_attention_mask,
            sharded_mem_config_DRAM,
            input_mem_config,
        )

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)

        batch_size = output_tensor.shape[0]

        valid_pcc = 0

        start_logits = output_tensor[..., 0]
        end_logits = output_tensor[..., 1]
        self.pcc_passed, self.pcc_message = assert_with_pcc(
            self.torch_output_tensor.start_logits, start_logits, pcc=valid_pcc
        )

        logger.info(
            f"start logits Bert-Tiny batch_size={batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, math_fidelity={self.math_fidelity}, PCC={self.pcc_message}"
        )

        self.pcc_passed, self.pcc_message = assert_with_pcc(
            self.torch_output_tensor.end_logits, end_logits, pcc=valid_pcc
        )
        logger.info(
            f"End logits Bert-Tiny batch_size={batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, math_fidelity={self.math_fidelity}, PCC={self.pcc_message}"
        )

    """
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
    """

    def run(self, tt_input_tensor=None):
        self.output_tensor = self.bert_tiny_model(
            self.config,
            self.input,
            self.token_type,
            self.position,
            self.attention,
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
