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

        # Pad vocab_size to be divisible by 32
        padded_vocab_size = math.ceil(self.vocab_size / 32) * 32

        size_per_device = padded_vocab_size // self.num_devices

        self.model_config = args.get_model_config()

        if args.is_galaxy:
            size_per_device = self.padded_vocab_size // self.num_devices
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
                breakpoint()
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
            # self.program_configs_decode = [self.model_config["LM_HEAD_RING_PROGCFG"] for _ in range(len(split_sizes))]

    def forward(self, x: ttnn.Tensor):
        outputs = []
        use_prefetcher = self.prefetcher is not None and self.prefetcher.mode == "decode"

        if use_prefetcher:
            program_configs = self.program_configs_decode
        else:
            program_configs = self.program_configs_prefill

        self.lm_head_output_memory_config = (
            self.model_config["PREFETCHER_SHARDED_LM_HEAD_OUTPUT_RING_MEMCFG"]
            if use_prefetcher
            else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )

        for weight, pc in zip(self.output_weights, program_configs):
            if use_prefetcher:
                memory_config = self.model_config["PREFETCHER_SHARDED_LM_HEAD_INPUT_RING_MEMCFG"]
                x = ttnn.to_memory_config(x, memory_config)
            breakpoint()
            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc,
                memory_config=self.lm_head_output_memory_config,
                dtype=self.args.lm_head_dtype if hasattr(self.args, "lm_head_dtype") else ttnn.bfloat8_b,
                sub_device_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
            )
            lm_head_outpt_memory_config = self.model_config.get("LM_HEAD_OUTPUT_MEMCFG", ttnn.L1_MEMORY_CONFIG)
            if use_prefetcher:
                lm_head_outpt_memory_config = self.model_config["PREFETCHER_LM_HEAD_OUT_RING_MEMCFG"]
            resharded_output = ttnn.to_memory_config(output, memory_config=lm_head_outpt_memory_config)
            breakpoint()
            outputs.append(resharded_output)

        ttnn.deallocate(x)
        # Number of shards along width 126 must not exceed number of cores 32
        # Concatenate the outputs
        # outputs shape: a list of tenors each tensor is 1,1,32,16032 per device, length of 4
        breakpoint()
        output = ttnn.concat(
            outputs,
            dim=-1,
            memory_config=self.model_config.get("LM_HEAD_OUTPUT_MEMCFG", ttnn.L1_MEMORY_CONFIG)
            if not use_prefetcher
            else self.model_config["PREFETCHER_LM_HEAD_OUT_RING_RESHARD_MEMCFG"],
        )
        # after concat each device has a tensor of shape 1,1,32,64128

        output = tt_all_reduce(
            output,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=3 if self.args.is_galaxy else 0,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            memory_config=output.memory_config(),
            dtype=self.args.ccl_dtype,
            sharded=False,
            use_composite=True,
            subdevice_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
        )

        return output
