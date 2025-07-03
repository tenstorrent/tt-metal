# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
from transformers import AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vit.tt import ttnn_optimized_sharded_vit_wh
from models.demos.wormhole.vit.demo.vit_helper_funcs import get_batch, get_data_loader
from models.utility_functions import divup


class VitTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        output_mesh_composer=None,
        use_random_input_tensor=False,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size

        # Set up mesh mappers if not provided
        if inputs_mesh_mapper is None and weights_mesh_mapper is None and output_mesh_composer is None:
            self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        else:
            self.inputs_mesh_mapper = inputs_mesh_mapper
            self.weights_mesh_mapper = weights_mesh_mapper
            self.output_mesh_composer = output_mesh_composer

        model_name = "google/vit-base-patch16-224"
        sequence_size = 224

        config = transformers.ViTConfig.from_pretrained(model_name)
        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config)
        self.config = ttnn_optimized_sharded_vit_wh.update_model_config(config, batch_size)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            custom_preprocessor=ttnn_optimized_sharded_vit_wh.custom_preprocessor,
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

        if use_random_input_tensor == False:
            ## IMAGENET INFERENCE
            data_loader = get_data_loader("ImageNet_data", batch_size * self.num_devices, 2)
            self.torch_pixel_values, labels = get_batch(data_loader, image_processor)
        else:
            self.torch_pixel_values = torch.randn(
                batch_size * self.num_devices, 3, sequence_size, sequence_size, dtype=torch.bfloat16
            )

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None  # ttnn.ReplicateTensorToMesh(device) causes unnecessary replication/takes more time on the first pass
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def setup_l1_sharded_input(self, device, torch_pixel_values, mesh_mapper=None, mesh_composer=None):
        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
        patch_size = self.config.patch_size
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

        N, H, W, C = torch_pixel_values.shape
        N = N // self.num_devices
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 1),
                ),
            }
        )
        n_cores = 16
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        tt_inputs_host = ttnn.from_torch(
            torch_pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )

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
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self, tt_input_tensor=None):
        self.output_tensor = None
        self.output_tensor = ttnn_optimized_sharded_vit_wh.vit(
            self.config,
            self.input_tensor,
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
    use_random_input_tensor=False,
):
    return VitTestInfra(
        device,
        batch_size,
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
        use_random_input_tensor,
    )
