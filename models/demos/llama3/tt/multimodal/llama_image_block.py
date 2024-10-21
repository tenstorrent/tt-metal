# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3.tt.multimodal.llama_layernorm import TtLayerNorm
from models.demos.llama3.tt.multimodal.llama_image_attention import TtLlamaImageAttention
from models.demos.llama3.tt.multimodal.llama_image_mlp import TtLlamaImageFeedForward
import os


class TtLlamaImageTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        gated=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.vision_dim
        self.gated = gated

        self.ln_1 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_1.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.attn = TtLlamaImageAttention(
            mesh_device,
            state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        self.ln_2 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_2.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.mlp = TtLlamaImageFeedForward(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}mlp.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        if gated:
            # Gate tensors must be expanded to hidden dim or we get a PCC error
            self.gate_attn = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}gate_attn"].unsqueeze(0).expand(1, self.hidden_size),
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.gate_ffn = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}gate_ffn"].unsqueeze(0).expand(1, self.hidden_size),
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def forward(self, x, mask):
        return self.forward_tt(x, mask)
        if os.environ.get("BLOCK") == "tt":
            return self.forward_tt(x, mask)
        else:
            return self.forward_pt(x, mask)

    def forward_pt(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        attn_out = self.attn(self.ln_1(x_11SH), mask=mask)
        if self.gated:
            assert False
            attn_out = ttnn.mul(attn_out, ttnn.tanh(self.gate_attn))

        x = ttnn.to_torch(x_11SH, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        attn_out = ttnn.to_torch(attn_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        res = (x + attn_out).bfloat16().float()
        res = ttnn.from_torch(
            res,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )
        mlp_out = self.mlp(self.ln_2(res))
        if self.gated:
            mlp_out = ttnn.mul(mlp_out, ttnn.tanh(self.gate_ffn))
        res = ttnn.to_torch(res, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        mlp_out = ttnn.to_torch(mlp_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        out = (res + mlp_out).bfloat16().float()
        out = ttnn.from_torch(
            out,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )
        return out

    def forward_tt(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        attn_out = self.attn(self.ln_1(x_11SH), mask=mask)
        if self.gated:
            attn_out = ttnn.mul(attn_out, ttnn.tanh(self.gate_attn))

        res = ttnn.add(x_11SH, attn_out)
        mlp_out = self.mlp(self.ln_2(res))
        if self.gated:
            mlp_out = ttnn.mul(mlp_out, ttnn.tanh(self.gate_ffn))
        out = ttnn.add(res, mlp_out)
        return out
