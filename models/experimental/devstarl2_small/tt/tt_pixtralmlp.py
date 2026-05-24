# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Vision FeedForward for Mistral-Small / Pixtral-class checkpoints.

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.devstral_utils.vision_ccl import vision_sum_all_reduce
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import (
    pad_seq_to_chunk_multiple,
    pixtral_effective_mm_seq_len,
    trim_seq_dim2,
    vision_seq_memcfg,
)


class MistralTTVisionMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        dtype,
        state_dict_prefix=None,
        tt_ccl=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.state_dict = state_dict
        self.tt_ccl = tt_ccl
        self.num_devices = int(args.num_devices)
        self.vision_dim = int(args.vision_dim)
        self.vision_hidden_dim = int(args.vision_hidden_dim)
        self.hidden_shard = self.vision_hidden_dim // self.num_devices
        self.use_dram_width_shard = self.num_devices > 1
        self.model_config = args.get_model_config()

        def get_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)

        if self.use_dram_width_shard:
            w13_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            w2_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
        else:
            w13_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            w2_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        weight_mem_config = ttnn.DRAM_MEMORY_CONFIG

        prefix = state_dict_prefix or ""

        def as_tensor(name, torch_2d, dtype, mesh_mapper, memory_config):
            cache_name = None
            if weight_cache_path is not None:
                cache_name = weight_cache_path / f"{prefix}{name}.weight"
            return ttnn.as_tensor(
                torch_2d,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                cache_file_name=cache_name,
            )

        self.w1 = as_tensor("w1", get_weight("w1"), dtype, w13_mapper, weight_mem_config)
        self.w3 = as_tensor("w3", get_weight("w3"), dtype, w13_mapper, weight_mem_config)
        self.w2 = as_tensor("w2", get_weight("w2"), dtype, w2_mapper, weight_mem_config)

        self.compute_kernel_config = args.compute_kernel_config_hifi2

    @staticmethod
    def _as_11sh(tensor: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        if (
            len(tensor.shape) == 4
            and int(tensor.shape[0]) == 1
            and int(tensor.shape[1]) == 1
            and int(tensor.shape[2]) == seq_len
        ):
            return tensor
        return ttnn.reshape(tensor, [1, 1, seq_len, -1])

    def _combine_w2_multidevice(self, w2_out: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """Sum partial down-proj outputs (w2 is K-sharded across mesh)."""
        if not self.use_dram_width_shard or self.tt_ccl is None:
            return w2_out

        w2_out = self._as_11sh(w2_out, seq_len)
        return vision_sum_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            seq_len,
            self.vision_dim,
            self.args,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """SwiGLU FFN with mesh width-sharded w1/w3 (N) and w2 (K) when num_devices > 1."""
        if x.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        seq_len = int(x.shape[-2])
        mm_seq_len = pixtral_effective_mm_seq_len(self.args, seq_len)

        def run_chunk(xc: ttnn.Tensor, m_len: int) -> ttnn.Tensor:
            chunk_seq_len = int(xc.shape[-2])
            pc_w13 = self.model_config["IMAGE_MLP_FC_PROGCFG"](m_len, mm_seq_len)
            pc_w2 = self.model_config["IMAGE_MLP_PROJ_PROGCFG"](m_len, mm_seq_len)
            w13_mem_cfg = vision_seq_memcfg(chunk_seq_len, self.hidden_shard)

            w1_out = ttnn.linear(
                xc,
                self.w1,
                dtype=ttnn.bfloat16,
                memory_config=w13_mem_cfg,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc_w13,
            )
            w3_out = ttnn.linear(
                xc,
                self.w3,
                dtype=ttnn.bfloat16,
                memory_config=w13_mem_cfg,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc_w13,
            )
            w2_in = ttnn.mul(
                w1_out,
                w3_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                memory_config=w13_mem_cfg,
            )
            ttnn.deallocate(w1_out)
            ttnn.deallocate(w3_out)

            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc_w2,
            )
            ttnn.deallocate(w2_in)

            if self.use_dram_width_shard:
                return self._combine_w2_multidevice(w2_out, chunk_seq_len)
            return w2_out

        if seq_len <= mm_seq_len:
            return run_chunk(x, seq_len)

        x, seq_len, original_seq_len = pad_seq_to_chunk_multiple(x, seq_len, mm_seq_len)
        x_batched = ttnn.reshape(x, [1, seq_len // mm_seq_len, mm_seq_len, -1])
        out_batched = run_chunk(x_batched, original_seq_len)
        if int(out_batched.shape[1]) == 1:
            out = self._as_11sh(out_batched, seq_len)
        else:
            out = ttnn.reshape(out_batched, [1, 1, seq_len, -1])
        return trim_seq_dim2(out, original_seq_len)


__all__ = ["MistralTTVisionMLP"]
