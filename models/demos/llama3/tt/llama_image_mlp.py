# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
import os


class TtLlamaImageFeedForward(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.model_config = args.get_model_config()
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

    def forward(self, x):
        return self.forward_tt(x)
        if os.environ.get("MLP") == "tt":
            return self.forward_tt(x)
        else:
            return self.forward_pt(x)

    def forward_pt(self, x):
        x = ttnn.to_torch(
            x, device=self.mesh_device, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        ).float()
        x = x[0]

        c_fc_weight = ttnn.to_torch(
            self.c_fc_weight, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        ).float()
        c_fc_bias = ttnn.to_torch(
            self.c_fc_bias, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        ).float()[:1]
        c_proj_weight = ttnn.to_torch(
            self.c_proj_weight, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-2)
        ).float()
        c_proj_bias = ttnn.to_torch(
            self.c_proj_bias, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        ).float()
        c_proj_bias = c_proj_bias[:1]

        # hidden = (torch.matmul(x, c_fc_weight).bfloat16().float() + c_fc_bias).bfloat16().float()
        # hidden = torch.nn.functional.gelu(hidden).bfloat16().float()
        # hidden = (torch.matmul(hidden, c_proj_weight).bfloat16().float() + c_proj_bias).bfloat16().float()
        # hidden = hidden.view(1, 1, 5120, -1)
        x = x.bfloat16().float()
        hidden = torch.nn.functional.linear(x, c_fc_weight.T, c_fc_bias).bfloat16().float()
        hidden = torch.nn.functional.gelu(hidden).bfloat16().float()
        hidden = torch.nn.functional.linear(hidden, c_proj_weight.T).bfloat16().float()
        hidden += c_proj_bias
        hidden = hidden.view(1, 1, 5120, -1)

        hidden = ttnn.from_torch(
            hidden,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        return hidden

    def forward_tt(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        compute_kernel_config_hifi2 = self.model_config["MLP_KERNEL_CONFIG_HIFI2"]
        compute_kernel_config_hifi4 = self.model_config["MLP_KERNEL_CONFIG_HIFI4"]

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
            compute_kernel_config=compute_kernel_config_hifi4,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # activation="gelu",  # NOTE: activation must be passed to linear here, not in program config! Bad output otherwise
        )
        c_fc_out = ttnn.gelu(c_fc_out, fast_and_approximate_mode=False)
        c_proj_out = ttnn.linear(
            c_fc_out,
            self.c_proj_weight,
            compute_kernel_config=compute_kernel_config_hifi4,
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
