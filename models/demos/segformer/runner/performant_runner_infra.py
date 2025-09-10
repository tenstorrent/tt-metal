# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional as F
from loguru import logger
from ttnn.model_preprocessing import ParameterDict, ParameterList, preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
    SegformerForSemanticSegmentationReference,
)
from models.demos.segformer.tests.pcc.test_segformer_decode_head import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_decode_head,
)
from models.demos.segformer.tests.pcc.test_segformer_model import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_model,
)
from models.demos.segformer.tt.ttnn_segformer_for_semantic_segmentation import TtSegformerForSemanticSegmentation
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerForSemanticSegmentationReference):
            parameters["segformer"] = {}
            segformer_preprocess = create_custom_preprocessor_model(mesh_mapper)
            parameters["segformer"] = segformer_preprocess(model.segformer, None, None, None)
            parameters["decode_head"] = {}
            deocde_preprocess = create_custom_preprocessor_decode_head(mesh_mapper)
            parameters["decode_head"] = deocde_preprocess(model.decode_head, None, None, None)

        return parameters

    return custom_mesh_preprocessor


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["sr", "proj", "dwconv", "linear_fuse", "classifier"]:
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


def load_segformer_torch_model(device, model_location_generator=None):
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerForSemanticSegmentationReference(config)
    reference_model = load_torch_model(
        reference_model, f"", module="semantic_sub", model_location_generator=model_location_generator
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    parameters = move_to_device(parameters, device)

    for i in range(4):
        parameters["decode_head"]["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["weight"], device=device
        )
        parameters["decode_head"]["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["bias"], device=device
        )

    return reference_model, config, parameters


class SegformerTestInfra:
    def __init__(self, device, model_location_generator=None, device_batch_size=1, channels=3, resolution=(512, 512)):
        super().__init__()
        torch.manual_seed(0)
        self.device = device
        self.resolution = resolution
        self.num_devices = self.device.get_num_devices()
        self.channels = channels
        self.batch_size_per_device = device_batch_size
        self.batch_size = device_batch_size * self.num_devices
        self.input_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = get_mesh_mappers(self.device)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.model_location_generator = model_location_generator
        self.reference_model, config, self.parameters = load_segformer_torch_model(device, model_location_generator)
        self.ttnn_segformer_model = TtSegformerForSemanticSegmentation(config, self.parameters)

        self.torch_input = torch.randn((self.batch_size, self.channels, self.resolution[0], self.resolution[1]))
        self.torch_output_tensor = self.reference_model(self.torch_input)
        input_pixels_permuted = torch.permute(self.torch_input, (0, 2, 3, 1))

    def run(self):
        self.output_tensor = self.ttnn_segformer_model(
            self.device,
            self.input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            parameters=self.parameters,
        )

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input if torch_input_tensor is None else torch_input_tensor
        n, c, h, w = torch_input_tensor.shape
        n = n // self.num_devices
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = F.pad(torch_input_tensor, (0, 13))
        input_tensor = [torch_input_tensor[i].unsqueeze(0) for i in range(torch_input_tensor.shape[0])]
        tt_inputs_host = ttnn.from_host_shards(
            [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in input_tensor], device.shape
        )
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                16,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )
        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor.logits, mesh_composer=self.output_mesh_composer)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.reshape((self.torch_output_tensor.logits).shape)

        valid_pcc = 0.978
        self.pcc_passed, self.pcc_message = assert_with_pcc(
            self.torch_output_tensor.logits, output_tensor, pcc=valid_pcc
        )
        logger.info(f"Segformer, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)


def create_test_infra(device, model_location_generator=None, device_batch_size=1):
    return SegformerTestInfra(device, model_location_generator, device_batch_size)
