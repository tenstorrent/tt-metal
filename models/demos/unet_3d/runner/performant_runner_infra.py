# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from safetensors.torch import load_file
from ttnn.model_preprocessing import ParameterDict, ParameterList

import ttnn
from models.common.utility_functions import divup
from models.demos.unet_3d.torch_impl.model import UNet3DTch
from models.demos.unet_3d.ttnn_impl.model import UNet3D
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_torch_ttnn_models(device, config, model_path):
    state_dict = load_file(model_path)
    model_ttnn = UNet3D(device, **config)
    model_ttnn.load_state_dict(state_dict)
    model_torch = UNet3DTch(**config)
    model_torch.load_state_dict(state_dict)
    return model_ttnn, model_torch


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
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


class UNet3DTestInfra:
    def __init__(self, device, config):
        super().__init__()
        torch.manual_seed(0)
        self.device = device
        self.num_devices = self.device.get_num_devices()
        self.channels = config["model"]["in_channels"]
        self.batch_size = config["dataset"]["batch_size_per_device"] * self.num_devices
        self.input_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = get_mesh_mappers(self.device)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.ttnn_model, self.reference_model = load_torch_ttnn_models(
            self.device, config["model"], config["model_path"]
        )
        p_s = config["dataset"]["slice_builder"]["patch_shape"]
        h_s = config["dataset"]["slice_builder"]["halo_shape"]
        self.resolution = [p + (2 * h) for p, h in zip(p_s, h_s)]

        self.torch_input = torch.randn(
            (self.batch_size, self.channels, self.resolution[0], self.resolution[1], self.resolution[2])
        )
        self.torch_output_tensor = self.reference_model(self.torch_input)

    def run(self):
        self.output_tensor = self.ttnn_model(
            self.input_tensor,
        )

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=8):
        # torch tensor
        torch_input_tensor = self.torch_input if torch_input_tensor is None else torch_input_tensor
        n, c, d, h, w = torch_input_tensor.shape
        if c < min_channels:
            c = min_channels
        elif c % min_channels != 0:
            c = ((c // min_channels) + 1) * min_channels

        assert n % self.num_devices == 0, "n isn't evenly divided by the available number of devices"
        n = n // self.num_devices if n // self.num_devices != 0 else n
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, d, h, w],
            ttnn.CoreGrid(x=8, y=4),
            ttnn.ShardStrategy.HEIGHT,
        )
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
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )
        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor, mesh_composer=self.output_mesh_composer)
        output_tensor = output_tensor.reshape((self.torch_output_tensor).shape)

        valid_pcc = 0.967
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)
        logger.info(f"UNet3D, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)


def create_test_infra(device, config):
    return UNet3DTestInfra(device, config)
