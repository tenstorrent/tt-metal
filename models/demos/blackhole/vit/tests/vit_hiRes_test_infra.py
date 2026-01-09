# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import divup
from models.demos.blackhole.vit.tt import ttnn_optimized_sharded_vit_hiRes_bh as ttnn_optimized_sharded_vit


class VitHiResTestInfra:
    """Test infrastructure for high resolution ViT encoder layers."""

    def __init__(
        self,
        device,
        batch_size,
        sequence_size,
        hidden_size,
        num_attention_heads=16,
        num_hidden_layers=12,
        use_random_input_tensor=True,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        model_name = "google/vit-base-patch16-224"

        config = transformers.ViTConfig.from_pretrained(model_name)
        config.hidden_size = hidden_size
        config.intermediate_size = hidden_size * 4
        config.num_attention_heads = num_attention_heads
        if config.hidden_size >= 2048:
            config.num_attention_heads = 12
        config.num_hidden_layers = num_hidden_layers

        self.config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size, sequence_size)

        self.torch_layers = []
        self.tt_parameters_list = []

        for i in range(num_hidden_layers):
            torch_layer = transformers.models.vit.modeling_vit.ViTLayer(config).eval()
            self.torch_layers.append(torch_layer)

            layer_parameters = preprocess_model_parameters(
                initialize_model=lambda l=torch_layer: l,
                custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
                device=device,
            )
            self.tt_parameters_list.append(layer_parameters)

        if use_random_input_tensor:
            self.torch_hidden_states = torch.randn(batch_size, 1, sequence_size, hidden_size, dtype=torch.float32)
        else:
            self.torch_hidden_states = torch.randn(batch_size, 1, sequence_size, hidden_size, dtype=torch.float32)

    def setup_l1_sharded_input(self, device):
        hidden_states = ttnn.from_torch(
            self.torch_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        input_mem_config = ttnn.create_sharded_memory_config(
            hidden_states.shape,
            core_grid=self.config.core_grid_BLOCK_SHARDED,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        return hidden_states, input_mem_config

    def setup_dram_sharded_input(self, device):
        tt_inputs_host = ttnn.from_torch(
            self.torch_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
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

        input_mem_config = ttnn.create_sharded_memory_config(
            [self.batch_size, 1, self.sequence_size, self.hidden_size],
            core_grid=self.config.core_grid_BLOCK_SHARDED,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self):
        self.output_tensor = None
        encoder_input = self.input_tensor

        for i, layer_params in enumerate(self.tt_parameters_list):
            encoder_output = ttnn_optimized_sharded_vit.vit_layer(
                self.config,
                encoder_input,
                layer_params,
            )
            encoder_input = encoder_output

        self.output_tensor = encoder_output
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    sequence_size,
    hidden_size,
    num_attention_heads=16,
    num_hidden_layers=12,
    use_random_input_tensor=True,
):
    return VitHiResTestInfra(
        device,
        batch_size,
        sequence_size,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        use_random_input_tensor,
    )
