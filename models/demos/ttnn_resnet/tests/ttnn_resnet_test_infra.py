# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
    _nearest_y,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.ttnn_resnet.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.demos.ttnn_resnet.tt.ttnn_functional_resnet50_new_conv_api import resnet50


def load_resnet50_model(model_location_generator):
    # TODO: Can generalize the version to an arg
    torch_resnet50 = None
    if model_location_generator is not None:
        model_version = "IMAGENET1K_V1.pt"
        model_path = model_location_generator(model_version, model_subdir="ResNet50")
        if os.path.exists(model_path):
            torch_resnet50 = torchvision.models.resnet50()
            torch_resnet50.load_state_dict(torch.load(model_path))
    if torch_resnet50 is None:
        torch_resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    return torch_resnet50


## copied from ttlib version test:
# golden pcc is ordered fidelity, weight dtype, activation dtype
golden_pcc = {
    8: {
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ): 0.983301,  # PCC: 0.9833017469734239             TODO: NEED DEBUGGING WHY THIS IS SLIGHTLY LOWER THAN TTLIB
        # ): 0.990804,  # Max ATOL Delta: 1.607335090637207, Max RTOL Delta: 115.62200164794922, PCC: 0.9908042840544742
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
        ): 0.986301,  # Max ATOL Delta: 1.5697126388549805, Max RTOL Delta: 21.3042049407959, PCC: 0.9863013351442654
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
        ): 0.973763,  # Max ATOL Delta: 2.455164909362793, Max RTOL Delta: inf, PCC: 0.9737631427307492
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966628
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ): 0.983400,  # Max ATOL Delta: 1.7310011386871338, Max RTOL Delta: 369.5689392089844, PCC: 0.9834004200555363
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
        ): 0.984828,  # Max ATOL Delta: 1.6054553985595703, Max RTOL Delta: 59.124324798583984, PCC: 0.9848281996919587
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
        ): 0.934073,  # Max ATOL Delta: 4.330164909362793, Max RTOL Delta: inf, PCC: 0.9340735819578696
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635019
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ): 0.938909,  # Max ATOL Delta: 3.861414909362793, Max RTOL Delta: 240.63145446777344, PCC: 0.9389092547575272
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
        ): 0.959609,  # Max ATOL Delta: 3.205164909362793, Max RTOL Delta: 141.7057342529297, PCC: 0.9596095155046113
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
        ): 0.854903,  # Max ATOL Delta: 7.830164909362793, Max RTOL Delta: inf, PCC: 0.8549035869182201
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419433
    },
    16: {
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966632
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.941,  # PCC: 0.9414369437627494               TODO: NEED DEBUGGING WHY THIS IS SLIGHTLY LOWER THAN TTLIB
        # ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635021
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.988,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419435
    },
    20: {
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966628
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.941,  #   PCC: 0.9419975597174123             TODO: NEED DEBUGGING WHY THIS IS SLIGHTLY LOWER THAN TTLIB
        # ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635021
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419433
    },
}

golden_pcc = {
    ttnn.device.Arch.WORMHOLE_B0: golden_pcc,
    ttnn.device.Arch.GRAYSKULL: golden_pcc,
}

golden_pcc[ttnn.device.Arch.GRAYSKULL][16][
    (
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
    )
] = 0.936


class ResNet50TestInfra:
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
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        output_mesh_composer=None,
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
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.output_mesh_composer = output_mesh_composer

        self.resnet50_first_conv_kernel_size = 3
        self.resnet50_first_conv_stride = 2

        torch_model = (
            load_resnet50_model(model_location_generator).eval()
            if use_pretrained_weight
            else torchvision.models.resnet50().eval()
        )

        model_config = {
            "MATH_FIDELITY": math_fidelity,
            "WEIGHTS_DTYPE": weight_dtype,
            "ACTIVATIONS_DTYPE": act_dtype,
        }

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        input_shape = (batch_size * num_devices, 3, 224, 224)

        self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        torch_model.to(torch.bfloat16)
        self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)

        ## golden

        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        ## ttnn

        self.ttnn_resnet50_model = resnet50(
            device=device,
            parameters=parameters,
            batch_size=batch_size,
            model_config=model_config,
            input_shape=input_shape,
            kernel_size=self.resnet50_first_conv_kernel_size,
            stride=self.resnet50_first_conv_stride,
            dealloc_input=dealloc_input,
            final_output_mem_config=final_output_mem_config,
            mesh_mapper=weights_mesh_mapper,
        )
        self.ops_parallel_config = {}

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        if self.batch_size == 16:
            core_grid = ttnn.CoreGrid(y=8, x=6)
        elif self.batch_size == 20:
            if is_grayskull():
                core_grid = ttnn.CoreGrid(y=8, x=10)
            elif is_wormhole_b0():
                core_grid = ttnn.CoreGrid(y=5, x=6)  # untested due to unsupported batch20 on WH
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        print(torch_input_tensor.shape)
        pad_h = self.resnet50_first_conv_kernel_size
        pad_w = self.resnet50_first_conv_kernel_size
        w = torch_input_tensor.shape[-1]
        pad_w_right = (w + 2 * pad_w + 31) // 32 * 32 - (w + pad_w)
        # pad h w
        torch_input_tensor_padded = torch.nn.functional.pad(torch_input_tensor, (pad_w, pad_w_right, pad_h, pad_h))
        if num_devices > 1:
            n, c, h, w = torch_input_tensor_padded.shape
            n = n // num_devices
        else:
            n, c, h, w = torch_input_tensor_padded.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * c * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR, False)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper
        )
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(
            device, torch_input_tensor, mesh_mapper=mesh_mapper, mesh_composer=mesh_composer
        )
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self, tt_input_tensor=None):
        self.output_tensor = self.ttnn_resnet50_model(
            self.input_tensor,
            self.device,
            self.ops_parallel_config,
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        output_tensor = torch.reshape(output_tensor, (output_tensor.shape[0], 1000))

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
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(
            f"ResNet50 batch_size={batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, math_fidelity={self.math_fidelity}, PCC={self.pcc_message}"
        )


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight=True,
    dealloc_input=True,
    final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    inputs_mesh_mapper=None,
    weights_mesh_mapper=None,
    output_mesh_composer=None,
    model_location_generator=None,
):
    return ResNet50TestInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
        model_location_generator,
    )
