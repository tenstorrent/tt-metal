"""
This is the implmentation of MultiModalprojector for Gemma-3-4b-it model.
There is no Independent MultiModalprojector support in TT-Transformers.
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma3.tt.gemma_vision_rmsnorm import RMSNorm


class TtGemma3MultiModalProjector(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        image_size,
        patch_size,
        hidden_size,
        mm_tokens_per_image,
        weight_cache_path,
        layer_norm_eps,
        dtype,
        configuration,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.patches_per_image = int(image_size // patch_size)
        self.tokens_per_side = int(mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.hidden_size = hidden_size

        weight_key = state_dict_prefix + ".mm_input_projection_weight"
        weight = state_dict[weight_key]

        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}{name}")

        # Pad dimensions to multiples of 32
        padded_vision_size = ((hidden_size + 31) // 32) * 32

        if padded_vision_size != hidden_size:
            padding = torch.zeros(hidden_size, padded_vision_size - hidden_size, dtype=weight.dtype)
            weight = torch.cat([weight, padding], dim=-1)

        self.mm_input_projection_weight = ttnn.as_tensor(
            weight,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # cache_file_name=cache_name("mm_input_projection_weight"), # pcc drop fix later
        )

        # # Create RMSNorm layer
        weight_key = state_dict_prefix + ".mm_soft_emb_norm"
        self.mm_soft_emb_norm = RMSNorm(
            device=mesh_device,
            dim=1152,
            state_dict=state_dict,
            state_dict_prefix="",
            weight_key=weight_key,
            weight_dtype=dtype,
            is_distributed=False,
            # sharded_program_config=tt_model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
            # sharded_output_config=tt_model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
        )

    def forward(self, vision_outputs: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, _, seq_length = vision_outputs.shape
        mode = "decode" if seq_length <= 32 else "prefill"

        # Reshape: [batch, seq, hidden] -> [batch, hidden, seq]
        reshaped_vision_outputs = ttnn.transpose(vision_outputs, 1, 2)

        ttnn.deallocate(vision_outputs)

        reshaped_vision_outputs = ttnn.reshape(
            reshaped_vision_outputs, (batch_size, seq_length, self.patches_per_image, self.patches_per_image)
        )

        in_n, in_c, in_h, in_w = reshaped_vision_outputs.shape
        reshaped_vision_outputs = ttnn.to_layout(reshaped_vision_outputs, ttnn.ROW_MAJOR_LAYOUT)
        reshaped_vision_outputs = ttnn.permute(reshaped_vision_outputs, (0, 2, 3, 1))
        reshaped_vision_outputs = ttnn.reshape(reshaped_vision_outputs, (1, 1, in_n * in_h * in_w, in_c))
        pooled_vision_outputs = ttnn.avg_pool2d(
            reshaped_vision_outputs,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.kernel_size, self.kernel_size),
            padding=(0, 0),
            ceil_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        # transpose
        HOUT = ((in_h - self.kernel_size) // self.kernel_size) + 1
        WOUT = ((in_w - self.kernel_size) // self.kernel_size) + 1
        pooled_vision_outputs = ttnn.reshape(pooled_vision_outputs, (in_n, HOUT, WOUT, in_c))

        pooled_vision_outputs = ttnn.permute(pooled_vision_outputs, (0, 3, 1, 2))
        pooled_vision_outputs = ttnn.to_layout(pooled_vision_outputs, ttnn.TILE_LAYOUT)

        pooled_vision_outputs = ttnn.reshape(
            pooled_vision_outputs, (pooled_vision_outputs.shape[0], pooled_vision_outputs.shape[1], -1)
        )

        # # Flatten(2)
        pooled_vision_outputs = ttnn.transpose(pooled_vision_outputs, 1, 2)
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs, mode=mode)
        self.mm_input_projection_weight = ttnn.to_layout(self.mm_input_projection_weight, ttnn.TILE_LAYOUT)
        projected_vision_outputs = ttnn.matmul(normed_vision_outputs, self.mm_input_projection_weight)

        ttnn.deallocate(pooled_vision_outputs)
        ttnn.deallocate(normed_vision_outputs)

        return projected_vision_outputs
