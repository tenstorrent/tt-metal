# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import pad_by_zero
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import (
    vision_activation_memcfg,
    vision_rms_norm_memcfg,
)
from models.experimental.devstarl2_small.tt.tt_patchmerger import TTMistral3PatchMerger


class TTMistral3MultiModalProjector(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix="multi_modal_projector.",
        weight_cache_path=None,
        dtype=ttnn.bfloat16,
        eps=1e-5,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.state_dict_prefix = state_dict_prefix
        self.dtype = dtype

        self.norm_weight, _ = pad_by_zero(
            state_dict[f"{state_dict_prefix}norm.weight"],
            device=mesh_device,
            tt_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tt_dtype=dtype,
        )
        self.norm_eps = eps
        self.patch_merger = TTMistral3PatchMerger(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_merger.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        def as_linear_weight(name, mesh_mapper, cache_suffix=""):
            return ttnn.as_tensor(
                torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=None
                if weight_cache_path is None
                else weight_cache_path / f"{state_dict_prefix}{name}.weight{cache_suffix}",
            )

        self.linear_1_weight = as_linear_weight("linear_1", ttnn.ReplicateTensorToMesh(mesh_device))
        self.linear_2_weight = as_linear_weight(
            "linear_2", ttnn.ShardTensorToMesh(mesh_device, dim=-1), cache_suffix=".sharded_dim_-1"
        )

    def forward(self, image_features: ttnn.Tensor, image_sizes) -> ttnn.Tensor:
        seq_tokens = int(image_features.shape[0])
        norm_mem_cfg = vision_rms_norm_memcfg(seq_tokens, int(image_features.shape[1]))
        x = ttnn.reshape(image_features, (1, 1, seq_tokens, image_features.shape[1]))
        if x.memory_config().buffer_type != norm_mem_cfg.buffer_type:
            x = ttnn.to_memory_config(x, norm_mem_cfg)
        x = ttnn.rms_norm(x, epsilon=self.norm_eps, weight=self.norm_weight, memory_config=norm_mem_cfg)
        x = ttnn.reshape(x, image_features.shape)
        x = self.patch_merger(x, image_sizes)
        x = ttnn.linear(x, self.linear_1_weight, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        m_rows = int(x.shape[0])
        act_mem_cfg = vision_activation_memcfg(m_rows)
        if act_mem_cfg.buffer_type == ttnn.BufferType.L1 and x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, act_mem_cfg)
        x = ttnn.gelu(x, memory_config=act_mem_cfg)
        x = ttnn.typecast(x, ttnn.bfloat8_b, memory_config=act_mem_cfg)
        x = ttnn.linear(
            x,
            self.linear_2_weight,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return x


__all__ = ["TTMistral3MultiModalProjector"]
