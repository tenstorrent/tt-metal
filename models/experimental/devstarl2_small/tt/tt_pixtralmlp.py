# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Vision FeedForward for Mistral-Small / Pixtral-class checkpoints.

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import (
    pad_seq_to_chunk_multiple,
    pixtral_effective_mm_seq_len,
    trim_seq_dim2,
    vision_collective_memcfg,
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

        def as_tensor(torch_2d, dtype, mesh_mapper, memory_config):
            return ttnn.as_tensor(
                torch_2d,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
            )

        self.w1 = as_tensor(get_weight("w1"), dtype, w13_mapper, weight_mem_config)
        self.w3 = as_tensor(get_weight("w3"), dtype, w13_mapper, weight_mem_config)
        self.w2 = as_tensor(get_weight("w2"), dtype, w2_mapper, weight_mem_config)

        self.compute_kernel_config = args.compute_kernel_config_hifi2

    def _combine_w2_multidevice(self, w2_out: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """Sum partial down-proj outputs (w2 is K-sharded across mesh)."""
        if not self.use_dram_width_shard or self.tt_ccl is None:
            return w2_out

        w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])
        shard_w = int(w2_out.shape[-1])
        ag_mem_cfg = vision_collective_memcfg(seq_len, shard_w)
        if w2_out.memory_config().buffer_type != ag_mem_cfg.buffer_type:
            w2_out = ttnn.to_memory_config(w2_out, ag_mem_cfg)
        cluster_axis = 1
        gathered = ttnn.experimental.all_gather_async(
            w2_out,
            persistent_output_buffer=None,
            dim=1,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=self.tt_ccl.get_num_links(cluster_axis),
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
            memory_config=ag_mem_cfg,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=40,
            num_workers_per_link=4,
            num_buffers_per_channel=4,
        )
        w2_out.deallocate(True)
        reduced = ttnn.experimental.fast_reduce_nc(
            gathered,
            dims=[1],
            output=None,
            compute_kernel_config=None,
            memory_config=ag_mem_cfg,
        )
        reduced = reduced[:, :, : gathered.shape[-2], :]
        ttnn.deallocate(gathered)
        return reduced

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """SwiGLU FFN with mesh width-sharded w1/w3 (N) and w2 (K) when num_devices > 1."""
        x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        seq_len = int(x.shape[-2])
        mm_seq_len = pixtral_effective_mm_seq_len(self.args, seq_len)

        def run_chunk(xc: ttnn.Tensor, m_len: int) -> ttnn.Tensor:
            chunk_seq_len = int(xc.shape[-2])
            pc_w13 = self.model_config["IMAGE_MLP_FC_PROGCFG"](m_len, mm_seq_len)
            pc_w2 = self.model_config["IMAGE_MLP_PROJ_PROGCFG"](m_len, mm_seq_len)

            w1_out = ttnn.linear(
                xc,
                self.w1,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc_w13,
            )
            w3_out = ttnn.linear(
                xc,
                self.w3,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc_w13,
            )
            w2_in = ttnn.mul(
                w1_out,
                w3_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        out = ttnn.reshape(out_batched, [1, 1, seq_len, -1])
        return trim_seq_dim2(out, original_seq_len)


__all__ = ["MistralTTVisionMLP"]
