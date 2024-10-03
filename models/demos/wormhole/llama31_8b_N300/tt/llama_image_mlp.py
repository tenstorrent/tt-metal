# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtLlamaImageFeedForward(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        model_config,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.model_config = model_config
        torch_weight = lambda name, suffix: torch.transpose(
            self.state_dict[f"{state_dict_prefix}{name}.{suffix}"], -2, -1
        )
        torch_bias = lambda name, suffix: self.state_dict[f"{state_dict_prefix}{name}.{suffix}"]

        if args.dummy_weights:
            cache_name = lambda *_: None
        else:
            cache_name = lambda name, suffix: weight_cache_path / (state_dict_prefix + f".{name}.{suffix}")

        as_interleaved_tensor = lambda name, suffix, type, dim: ttnn.as_tensor(
            torch_weight(name, suffix)
            if suffix == "weight"
            else torch_bias(name, suffix),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim)
            if dim is not None
            else ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name, suffix),
        )

        # Sharded weights
        self.c_fc_weight = as_interleaved_tensor("c_fc", "weight", dtype, dim=-1)
        self.c_fc_bias = as_interleaved_tensor("c_fc", "bias", ttnn.bfloat16, dim=-1)
        self.c_proj_weight = as_interleaved_tensor("c_proj", "weight", dtype, dim=-2)
        self.c_proj_bias = as_interleaved_tensor("c_proj", "bias", ttnn.bfloat16, dim=None)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        compute_kernel_config_hifi2 = self.model_config["MLP_KERNEL_CONFIG_HIFI2"]

        x_in = x
        if seq_len >= 1024:  # Too big to compute. Set different program configs based on seqlen
            # Reshape input to to fit on device and parallelize computation
            x_in = ttnn.reshape(x_in, [1, seq_len // 1024, 1024, -1])
        pc_1 = self.model_config["IMAGE_MLP_FC_PROGCFG"](seq_len)
        pc_2 = self.model_config["IMAGE_MLP_PROJ_PROGCFG"](seq_len)

        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        c_fc_out = ttnn.linear(
            x_in,
            self.c_fc_weight,
            bias=self.c_fc_bias,
            compute_kernel_config=compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activation="gelu",  # NOTE: activation must be passed to linear here, not in program config! Bad output otherwise
        )
        c_proj_out = ttnn.linear(
            c_fc_out,
            self.c_proj_weight,
            compute_kernel_config=compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if seq_len >= 1024:  # Reshape back to intended shape
            c_proj_out = ttnn.reshape(c_proj_out, [1, 1, seq_len, -1])

        # All reduce
        if self.args.num_devices > 1:
            w2_out_gathered = ttnn.all_gather(c_proj_out, dim=1, num_links=1, topology=ttnn.Topology.Linear)
            pre_bias_output = ttnn.experimental.fast_reduce_nc(
                w2_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
        else:
            pre_bias_output = c_proj_out

        output = ttnn.add(pre_bias_output, self.c_proj_bias)
        return output
