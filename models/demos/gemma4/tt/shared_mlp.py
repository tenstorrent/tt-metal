# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has BOTH a shared MLP and routed MoE experts.
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x))
intermediate_size = 2112, no bias.

HF weight shapes:
  gate_proj.weight: [intermediate_size, hidden_size] = [2112, 2816]
  up_proj.weight:   [intermediate_size, hidden_size] = [2112, 2816]
  down_proj.weight: [hidden_size, intermediate_size] = [2816, 2112]
"""

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name


class SharedMLP:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        # Tag the cache filenames with the weight dtype so that flipping a
        # SharedMLP weight's dtype (e.g. bf16 → bfp8 for DRAM-pressure relief)
        # doesn't collide with a previously-cached file that holds the same
        # logical weight at a different dtype. The rest of the model's cache
        # entries are unaffected and stay reusable across runs.
        _dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        dtype_suffix = f"_{_dtype_str}"

        if tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        # Fuse gate+up into one column-parallel matmul. Per TP device we interleave
        # the shards as [up_i | gate_i] so that after column sharding splits the
        # concatenated output dim into ``tp`` contiguous chunks, each device holds
        # its own [up_i | gate_i] pair (see __call__ for the GeGLU eval). One wide
        # matmul replaces the two narrow gate/up matmuls — fewer op launches and
        # better core packing — which is the decode/throughput win.
        self.tp = tp
        if state_dict:
            gate_t = state_dict["gate_proj.weight"].transpose(-2, -1)  # [hidden, inter]
            up_t = state_dict["up_proj.weight"].transpose(-2, -1)  # [hidden, inter]
            if tp > 1:
                gate_shards = torch.chunk(gate_t, tp, dim=-1)
                up_shards = torch.chunk(up_t, tp, dim=-1)
                gate_up_t = torch.cat([torch.cat([up_shards[i], gate_shards[i]], dim=-1) for i in range(tp)], dim=-1)
            else:
                gate_up_t = torch.cat([up_t, gate_t], dim=-1)
            gate_up_weight = gate_up_t.unsqueeze(0).unsqueeze(0)  # [1,1,hidden,2*inter]
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_up_weight = None
            down_proj_weight = None

        # gate_up: column-parallel (shard fused output dim across TP devices)
        self.gate_up_proj = ttnn.as_tensor(
            gate_up_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_up_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # down: row-parallel (shard input dim, allreduce after)
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # Fused gate/up projection: one matmul produces [.., 2*inter/tp] per
        # device laid out as [up_i | gate_i]. A single wide matmul beats two
        # narrow ones (fewer op launches, better core packing), which is the win
        # even though we split the result back out below. ``ttnn.geglu`` would
        # save the split ops but currently breaks the batch-1 decode trace, so we
        # split manually and reuse the original (fast-approx GELU) math — keeping
        # numerics identical to the unfused baseline.
        gate_up = ttnn.linear(hidden_states, self.gate_up_proj)
        shard = gate_up.shape[-1] // 2
        s = gate_up.shape[-2]
        up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, s, shard])
        gate = ttnn.slice(gate_up, [0, 0, 0, shard], [1, 1, s, 2 * shard])
        gate_up.deallocate(True)

        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        output = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
