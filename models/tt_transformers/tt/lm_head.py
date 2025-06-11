# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce


class LMHead(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device,  # too many columns per device lead to L1 OOM
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices

        size_per_device = self.vocab_size // self.num_devices

        if args.is_galaxy:
            size_per_device = self.padded_vocab_size // self.num_devices
        num_splits = math.ceil(size_per_device / max_columns_per_device)

        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))  # remaining columns

        # Split the output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        self.output_weights = []
        if args.is_galaxy:
            cache_file_name = (
                None if args.dummy_weights else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_0"
            )
            padded_lm_head = torch.zeros(1, 1, args.dim, self.padded_vocab_size)
            padded_lm_head[:, :, :, : self.vocab_size] = torch_output_weights

            memory_config = (
                ttnn.DRAM_MEMORY_CONFIG
                if args.dim == 2048
                else args.create_dram_sharded_mem_config(k=args.dim // 4, n=self.padded_vocab_size // 8)
            )
            self.output_weights.append(  # (2k, 16k) 128* 1024
                ttnn.as_tensor(
                    padded_lm_head,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=args.cluster_shape),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    memory_config=memory_config,
                    cache_file_name=cache_file_name,
                )
            )
        else:
            for i, split_size in enumerate(split_sizes):
                # Create a list to store the split tensors for each device
                device_splits = []
                for device in range(self.num_devices):
                    start = device * size_per_device + sum(split_sizes[:i])
                    end = start + split_size
                    device_splits.append(torch_output_weights[:, start:end])

                # Concatenate the splits from all devices
                combined_split = torch.cat(device_splits, dim=-1)

                cache_file_name = (
                    None
                    if args.dummy_weights
                    else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_{i}_{combined_split.shape[-1]}"
                )
                memory_config = args.create_dram_sharded_mem_config(
                    k=args.dim, n=math.ceil(combined_split.shape[-1] / self.num_devices)
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
        if args.is_galaxy:
            self.program_configs = [
                (
                    None
                    if args.dim == 2048
                    else args.dram_matmul_config(
                        args.tile_padded_batch_rows,  # (8k, 128k) -> (2k, 16k)
                        args.dim // 4,
                        16 * 1024,
                        args.lm_head_core_grid.num_cores,
                    )
                )
            ]

        else:
            self.program_configs = [
                args.dram_matmul_config(
                    args.tile_padded_batch_rows,
                    args.dim,
                    split_size,
                    args.lm_head_core_grid.num_cores,
                )
                for split_size in split_sizes
            ]

    def forward(self, x: ttnn.Tensor):
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

        output = tt_all_reduce(
            output,
            mesh_device=self.mesh_device,
            cluster_axis=1,
            dim=3 if self.args.is_galaxy else 0,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            sharded=False,
            use_composite=True,
        )

        return output
