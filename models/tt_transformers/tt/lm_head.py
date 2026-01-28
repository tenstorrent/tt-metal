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
        tt_ccl,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device,  # too many columns per device lead to L1 OOM
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices

        # Pad vocab_size to be divisible by 32
        padded_vocab_size = math.ceil(self.vocab_size / 32) * 32

        size_per_device = padded_vocab_size // self.num_devices

        self.model_config = args.get_model_config()

        num_splits = math.ceil(size_per_device / max_columns_per_device)

        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))  # remaining columns

        # Split the output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        # Pad the output weights to the padded vocab size with zeros
        if self.vocab_size < padded_vocab_size:
            padding_size = padded_vocab_size - self.vocab_size
            torch_output_weights = torch.cat(
                [
                    torch_output_weights,
                    torch.zeros(torch_output_weights.shape[0], padding_size, dtype=torch_output_weights.dtype),
                ],
                dim=-1,
            )

        self.output_weights = []
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
                dtype=self.args.lm_head_dtype if hasattr(self.args, "lm_head_dtype") else ttnn.bfloat8_b,
            )
            outputs.append(
                ttnn.sharded_to_interleaved(
                    output, memory_config=self.model_config.get("LM_HEAD_OUTPUT_MEMCFG", ttnn.L1_MEMORY_CONFIG)
                )
            )

        # Concatenate the outputs
        output = ttnn.concat(
            outputs, dim=-1, memory_config=self.model_config.get("LM_HEAD_OUTPUT_MEMCFG", ttnn.L1_MEMORY_CONFIG)
        )

        output = tt_all_reduce(
            output,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            sharded=False,
            use_composite=True,
        )

        return output
