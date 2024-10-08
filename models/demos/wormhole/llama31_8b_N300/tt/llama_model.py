# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import ttnn
import torch
import torch.nn as nn
from models.demos.wormhole.llama31_8b_N300.tt.llama_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
import ttnn
from typing import Optional
from models.common.lightweightmodule import LightweightModule


class TtTransformer(LightweightModule):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        state_dict_prefix = args.get_state_dict_prefix("", None)

        self.layers = [
            TtTransformerBlock(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
            )
            for i in range(self.n_layers)
        ]
        self.norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", None),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="norm",
        )

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        get_last_token=-1,
    ):
        for layer in self.layers:
            x = layer(x, current_pos, rot_mat, transformation_mats, user_id, mode, page_table)
        if mode == "prefill" and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, self.args.dim))

        x = self.norm(x)
        output = self.lm_head(x)

        ttnn.deallocate(x)

        return output


class LMHead(nn.Module):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device=128256 // 4,  # larger values per device lead to OOM or hangs
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.num_devices = args.num_devices

        size_per_device = self.vocab_size // self.num_devices

        num_splits = math.ceil(size_per_device / max_columns_per_device)

        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))  # remaining columns
        # TODO remove prints
        print(f"split_sizes: {split_sizes}")

        # Split the output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        self.output_weights = []
        for i, split_size in enumerate(split_sizes):
            cache_file_name = (
                None if args.dummy_weights else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_{i}"
            )

            # Create a list to store the split tensors for each device
            device_splits = []
            for device in range(self.num_devices):
                start = device * size_per_device + sum(split_sizes[:i])
                end = start + split_size
                device_splits.append(torch_output_weights[:, start:end])

            # Concatenate the splits from all devices
            combined_split = torch.cat(device_splits, dim=-1)

            memory_config = args.create_dram_sharded_mem_config(
                k=args.dim, n=combined_split.shape[-1] // self.num_devices
            )
            self.output_weights.append(
                ttnn.as_tensor(
                    combined_split,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    memory_config=memory_config,
                    cache_file_name=cache_file_name,
                )
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.program_configs = [
            args.dram_matmul_config(
                args.tile_padded_batch_rows,
                args.dim,
                split_size,
                (args.lm_head_grid.y, args.lm_head_grid.x),
            )
            for split_size in split_sizes
        ]

    def forward(self, x: ttnn.Tensor):
        x = ttnn.interleaved_to_sharded(x, self.args.get_model_config()["LM_HEAD_INPUT_MEMCFG"])
        outputs = []
        for weight, pc in zip(self.output_weights, self.program_configs):
            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            outputs.append(ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG))

        # Concatenate the outputs
        output = ttnn.concat(outputs, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        return output
