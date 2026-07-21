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

import os

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# DRAM-width-sharded decode matmuls for the shared MLP. On by default for
# multi-device (tp>1); the single width-sharded weight is the same size as the
# interleaved one, so there is no memory cost. Set GEMMA4_MLP_DRAM_SHARD=0 to
# fall back to plain interleaved matmuls.
_DRAM_SHARD_MLP = os.environ.get("GEMMA4_MLP_DRAM_SHARD", "1") != "0"


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

        gu_n = 2 * self.intermediate_size // tp
        down_k = self.intermediate_size // tp
        # Gate each projection independently — e.g. intermediate=2112 at TP=4/8
        # makes down_k non-tile-aligned, so only gate_up can use the DRAM path.
        dram_shard = _DRAM_SHARD_MLP and tp > 1

        if dram_shard and can_dram_shard(self.hidden_size, gu_n):
            self.gate_up_proj = DramShardedLinear(
                gate_up_weight,
                mesh_device,
                col_mapper,
                k=self.hidden_size,
                n=gu_n,
                dtype=dtype,
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"gate_up_proj.weight.ws{tp_suffix}{dtype_suffix}"
                ),
            )
        else:
            gate_up_proj = ttnn.as_tensor(
                gate_up_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_up_proj.weight{tp_suffix}{dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.gate_up_proj = lambda x: ttnn.linear(x, gate_up_proj)

        if dram_shard and can_dram_shard(down_k, self.hidden_size):
            self.down_proj = DramShardedLinear(
                down_proj_weight,
                mesh_device,
                row_mapper,
                k=down_k,
                n=self.hidden_size,
                dtype=dtype,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight.ws{tp_suffix}{dtype_suffix}"),
            )
        else:
            down_proj = ttnn.as_tensor(
                down_proj_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=row_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.down_proj = lambda x: ttnn.linear(x, down_proj)

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # Fused gate/up projection: one matmul produces [.., 2*inter/tp] per
        # device laid out as [up_i | gate_i]. A single wide matmul beats two
        # narrow ones (fewer op launches, better core packing). When the shard
        # shape allows, the weight is DRAM-width-sharded so decode (M<=32) is
        # weight-read-optimal; prefill uses a 2D program config on the same
        # weight (see DramShardedLinear). We split the result back out and reuse
        # the original (fast-approx GELU) math — numerics identical to baseline.
        gate_up = self.gate_up_proj(hidden_states)
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
        output = self.down_proj(hidden)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
