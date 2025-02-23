# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
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
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.ttnn_resnet.tt.custom_preprocessing import create_custom_mesh_preprocessor
from diffusers import StableDiffusion3Pipeline
from tests.ttnn.integration_tests.stable_diffusion3_5.test_ttnn_ada_layernorm_continuous import (
    create_custom_preprocessor,
)
from models.experimental.functional_stable_diffusion3_5.reference.ada_layernorm_continuous import AdaLayerNormContinuous
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_ada_layernorm_continuous import (
    ttnn_AdaLayerNormContinuous,
)


class stable_diffusion3_5_test_infra:
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
        model_name,
        config,
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

        pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        reference_model = AdaLayerNormContinuous(
            embedding_dim=1536,
            conditioning_embedding_dim=1536,
            elementwise_affine=False,
            eps=1e-06,
            bias=True,
            norm_type="layer_norm",
        ).to(dtype=torch.bfloat16)
        reference_model.eval()

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=device,
        )

        self.torch_input_x = torch.randn([2, 1024, 1536], dtype=torch.bfloat16)
        self.torch_input_conditioning_embedding = torch.randn(2, 1536, dtype=torch.bfloat16)
        self.torch_input_x_unsqueezed = self.torch_input_x.unsqueeze(1)

        self.torch_output = reference_model(self.torch_input_x, self.torch_input_conditioning_embedding)

        model_config = {
            "MATH_FIDELITY": math_fidelity,
            "WEIGHTS_DTYPE": weight_dtype,
            "ACTIVATIONS_DTYPE": act_dtype,
        }

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

        ## ttnn
        self.ttnn_model = ttnn_AdaLayerNormContinuous(
            embedding_dim=1536,
            conditioning_embedding_dim=1536,
            elementwise_affine=False,
            eps=1e-06,
            bias=True,
            norm_type="layer_norm",
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

    def setup_inputs(
        self,
        device,
        torch_input_x_unsqueezed=None,
        torch_input_conditioning_embedding=None,
    ):
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_x_unsqueezed = (
            self.torch_input_x_unsqueezed if torch_input_x_unsqueezed is None else torch_input_x_unsqueezed
        )
        torch_input_conditioning_embedding = (
            self.torch_input_conditioning_embedding
            if torch_input_conditioning_embedding is None
            else torch_input_conditioning_embedding
        )

        tt_input_x = ttnn.from_torch(
            torch_input_x_unsqueezed,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn_input_conditioning_embedding = ttnn.from_torch(
            torch_input_conditioning_embedding,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return tt_input_x, ttnn_input_conditioning_embedding

    def run(self, tt_input_tensor=None):
        self.output_tensor = self.ttnn_model(
            self.input_tensor,
            self.device,
            self.ops_parallel_config,
        )
        return self.output_tensor

    # def validate(self, output_tensor=None):
    #     output_tensor = self.output_tensor if output_tensor is None else output_tensor
    #     output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
    #     output_tensor = torch.reshape(output_tensor, (output_tensor.shape[0], 1000))

    #     batch_size = output_tensor.shape[0]

    #     valid_pcc = 1.0
    #     if self.batch_size >= 8:
    #         valid_pcc = golden_pcc[self.device.arch()][self.batch_size][
    #             (self.math_fidelity, self.weight_dtype, self.act_dtype)
    #         ]
    #     else:
    #         if self.act_dtype == ttnn.bfloat8_b:
    #             if self.math_fidelity == ttnn.MathFidelity.LoFi:
    #                 valid_pcc = 0.87
    #             else:
    #                 valid_pcc = 0.94
    #         else:
    #             if self.math_fidelity == ttnn.MathFidelity.LoFi:
    #                 valid_pcc = 0.93
    #             else:
    #                 valid_pcc = 0.982
    #     self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

    #     logger.info(
    #         f"ResNet50 batch_size={batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, math_fidelity={self.math_fidelity}, PCC={self.pcc_message}"
    #     )


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    config,
    use_pretrained_weight=True,
    dealloc_input=True,
    final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
):
    return stable_diffusion3_5_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        "stabilityai/stable-diffusion-3.5-medium",
        config,
    )
