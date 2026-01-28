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
        prefetcher=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices
        self.prefetcher = prefetcher

        # Pad vocab_size to be divisible by (32 * num_devices) so that:
        # 1. vocab_size is tile-aligned (divisible by 32)
        # 2. size_per_device is also tile-aligned after dividing by num_devices
        # This ensures TILE concat doesn't have padding in the middle
        tile_size = 32
        padded_vocab_size = math.ceil(self.vocab_size / (tile_size * self.num_devices)) * (tile_size * self.num_devices)
        size_per_device = padded_vocab_size // self.num_devices

        max_columns_per_device_decode = math.ceil((max_columns_per_device) / tile_size) * tile_size
        max_columns_per_device_prefill = max_columns_per_device

        self.model_config = args.get_model_config()

        num_splits_decode = math.ceil(size_per_device / max_columns_per_device_decode)
        num_splits_prefill = math.ceil(size_per_device / max_columns_per_device_prefill)

        split_sizes_prefill = [min(size_per_device, max_columns_per_device_prefill)] * (num_splits_prefill - 1)
        split_sizes_prefill.append(size_per_device - sum(split_sizes_prefill))  # remaining columns

        split_sizes_decode = [min(size_per_device, max_columns_per_device_decode)] * (num_splits_decode - 1)
        split_sizes_decode.append(size_per_device - sum(split_sizes_decode))  # remaining columns
        # Split the output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        # Pad the output weights to the padded vocab size with zeros
        if self.vocab_size < self.padded_vocab_size:
            padding_size = self.padded_vocab_size - self.vocab_size
            torch_output_weights = torch.cat(
                [
                    torch_output_weights,
                    torch.zeros(torch_output_weights.shape[0], padding_size, dtype=torch_output_weights.dtype),
                ],
                dim=-1,
            )

        self.output_weights_prefill = []
        self.output_weights_decode = []

        for mode, split_sizes in enumerate([split_sizes_prefill, split_sizes_decode]):
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
                    else weight_cache_path
                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}"
                )

                def pad_to_power_of_2(n):
                    if n <= 0:
                        return 1
                    return 1 << (n - 1).bit_length()

                memory_config = args.create_dram_sharded_mem_config(
                    k=args.dim, n=pad_to_power_of_2(math.ceil(combined_split.shape[-1] / self.num_devices))
                )

                if mode == 0:
                    self.output_weights_prefill.append(
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
                else:
                    self.output_weights_decode.append(
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

    def forward(self, x: ttnn.Tensor, debug_input_torch=None, debug_weight_torch=None):
        outputs = []
        use_prefetcher = self.prefetcher is not None and self.prefetcher.mode == "decode"
        split_sizes = split_sizes_decode if use_prefetcher else split_sizes_prefill
        program_configs = [
            self.args.get_lm_head_program_config(split_size, self.prefetcher if use_prefetcher else None)
            for split_size in split_sizes
        ]
        output_weights = self.output_weights_decode if use_prefetcher else self.output_weights_prefill

        self.lm_head_output_memory_config = self.args.get_lm_head_output_mem_config(
            "decode" if use_prefetcher else "prefill", self.prefetcher if use_prefetcher else None
        )

        for i, (weight, pc) in enumerate(zip(output_weights, program_configs)):
            if use_prefetcher:
                lm_head_input_ring_mem_cfg = self.args.get_lm_head_input_ring_mem_config(self.prefetcher)
                x = ttnn.to_memory_config(x, lm_head_input_ring_mem_cfg)

            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc,
                memory_config=self.lm_head_output_memory_config,
                dtype=self.args.lm_head_dtype if hasattr(self.args, "lm_head_dtype") else ttnn.bfloat8_b,
                sub_device_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
            )

            if not use_prefetcher:
                output = ttnn.sharded_to_interleaved(
                    output, memory_config=self.args.get_lm_head_sharded_output_mem_config(None)
                )
            else:
                output = ttnn.to_memory_config(
                    output, memory_config=self.args.get_lm_head_sharded_output_mem_config(self.prefetcher)
                )

            outputs.append(output)

        ttnn.deallocate(x)
        # Number of shards along width 126 must not exceed number of cores 32
        # Concatenate the outputs
        # outputs shape: a list of tensors, each tensor is 1,1,32,size_per_device per device
        output = ttnn.concat(
            outputs,
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG if not use_prefetcher else ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if use_prefetcher else None,
        )

        for output_slice in outputs:
            ttnn.deallocate(output_slice)

        # Only use reshard mem config for decode mode
        # Prefill has different tensor widths (32064 vs 32768) so use L1_MEMORY_CONFIG
        if use_prefetcher:
            output = ttnn.to_memory_config(
                output,
                memory_config=self.args.get_lm_head_reshard_mem_config(self.prefetcher),
            )

        output = tt_all_reduce(
            output,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=3 if self.args.is_galaxy else 0,
            memory_config=output.memory_config(),
            dtype=self.args.ccl_dtype,
            sharded=False,
            use_composite=True,
            subdevice_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
        )

        return output
