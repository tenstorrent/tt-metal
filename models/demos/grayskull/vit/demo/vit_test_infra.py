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
from models.experimental.functional_vit.tt import ttnn_optimized_sharded_vit
from models.experimental.vit.vit_helper_funcs import get_data_loader, get_batch
import transformers
from transformers import AutoImageProcessor


class VitTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        output_mesh_composer=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.output_mesh_composer = output_mesh_composer

        model_name = "google/vit-base-patch16-224"
        sequence_size = 224

        config = transformers.ViTConfig.from_pretrained(model_name)
        config.num_hidden_layers = 12
        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config)
        self.config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        )

        # cls_token & position embeddings expand to batch_size
        # TODO: pass batch_size to preprocess_model_parameters
        model_state_dict = model.state_dict()
        torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
        torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
        if batch_size > 1:
            torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            torch_cls_token = torch.nn.Parameter(torch_cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        self.cls_token = ttnn.from_torch(
            torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self.position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        torch_attention_mask = torch.ones(self.config.num_hidden_layers, sequence_size, dtype=torch.float32)
        if torch_attention_mask is not None:
            self.head_masks = [
                ttnn.from_torch(
                    torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for index in range(self.config.num_hidden_layers)
            ]
        else:
            self.head_masks = [None for _ in range(self.config.num_hidden_layers)]

        ## IMAGENET INFERENCE
        data_loader = get_data_loader("ImageNet_data", batch_size, 2)
        self.torch_pixel_values, labels = get_batch(data_loader, image_processor)

    def setup_l1_sharded_input(self, device, torch_pixel_values, mesh_mapper=None, mesh_composer=None):
        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
        patch_size = 16
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

        N, H, W, C = torch_pixel_values.shape
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(int(batch_size - 1), 0),
                ),
            }
        )
        n_cores = batch_size
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR, False)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        tt_inputs_host = ttnn.from_torch(torch_pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(
            device, self.torch_pixel_values, mesh_mapper=mesh_mapper, mesh_composer=mesh_composer
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
        self.output_tensor = None
        self.output_tensor = ttnn_optimized_sharded_vit.vit(
            self.config,
            self.input_tensor,
            self.head_masks,
            self.cls_token,
            self.position_embeddings,
            parameters=self.parameters,
        )
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    inputs_mesh_mapper=None,
    weights_mesh_mapper=None,
    output_mesh_composer=None,
):
    return VitTestInfra(
        device,
        batch_size,
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
    )
