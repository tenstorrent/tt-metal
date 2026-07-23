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
from models.demos.gemma4.tt.dram_sharded import TILE_SIZE, DramShardedLinear, can_dram_shard
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# DRAM-width-sharded decode matmuls for the shared MLP. On by default for
# multi-device (tp>1); the single width-sharded weight is the same size as the
# interleaved one, so there is no memory cost. Set GEMMA4_MLP_DRAM_SHARD=0 to
# fall back to plain interleaved matmuls.
_DRAM_SHARD_MLP = os.environ.get("GEMMA4_MLP_DRAM_SHARD", "1") != "0"


def resolve_shared_mlp_intermediate_size(hf_config, state_dict=None, layer_idx=None) -> int:
    """Per-layer intermediate width for dense SharedMLP.

    Gemma4-E2B sets ``use_double_wide_mlp=True``: KV-shared layers use
    ``2 * intermediate_size`` (HF gate_proj is [12288, H] vs [6144, H] on early
    layers). Prefer the checkpoint shape when present; otherwise mirror HF's
    double-wide rule from ``layer_idx``.
    """
    if state_dict and state_dict.get("gate_proj.weight") is not None:
        return int(state_dict["gate_proj.weight"].shape[0])
    inter = int(hf_config.intermediate_size)
    if (
        layer_idx is not None
        and bool(getattr(hf_config, "use_double_wide_mlp", False))
        and (getattr(hf_config, "num_kv_shared_layers", 0) or 0) > 0
    ):
        n_layers = int(getattr(hf_config, "num_hidden_layers", 0) or 0)
        first_shared = n_layers - int(hf_config.num_kv_shared_layers)
        if int(layer_idx) >= first_shared:
            inter *= 2
    return inter


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
        layer_idx=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = resolve_shared_mlp_intermediate_size(hf_config, state_dict, layer_idx)

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

        # Pad intermediate to a tile-aligned per-device size (same pattern as
        # experts/weights.py). At TP=8, 2112/8=264 is not tile-aligned; TILE
        # slice rounds the GeGLU half to 288 while an unpadded down_proj stays
        # K=264 → matmul width/height mismatch on WH/BH e2e.
        if tp > 1:
            per_device = self.intermediate_size // tp
            padded_per_device = ((per_device + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
            pad_amount = padded_per_device * tp - self.intermediate_size
        else:
            padded_per_device = self.intermediate_size
            pad_amount = 0
        self._inter_per_device = padded_per_device
        # Invalidate pre-pad cache bins when padding is applied.
        pad_suffix = f"_ipad{padded_per_device}" if pad_amount > 0 else ""

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
            down_t = state_dict["down_proj.weight"].transpose(-2, -1)  # [inter, hidden]
            if gate_t.shape[-1] != self.intermediate_size:
                raise ValueError(
                    f"SharedMLP intermediate mismatch: weights={gate_t.shape[-1]} "
                    f"resolved={self.intermediate_size} (layer_idx={layer_idx})"
                )
            if pad_amount > 0:
                gate_t = torch.nn.functional.pad(gate_t, (0, pad_amount))
                up_t = torch.nn.functional.pad(up_t, (0, pad_amount))
                # down: [I, H] → pad K (dim 0)
                down_t = torch.nn.functional.pad(down_t, (0, 0, 0, pad_amount))
            if tp > 1:
                gate_shards = torch.chunk(gate_t, tp, dim=-1)
                up_shards = torch.chunk(up_t, tp, dim=-1)
                gate_up_t = torch.cat([torch.cat([up_shards[i], gate_shards[i]], dim=-1) for i in range(tp)], dim=-1)
            else:
                gate_up_t = torch.cat([up_t, gate_t], dim=-1)
            gate_up_weight = gate_up_t.unsqueeze(0).unsqueeze(0)  # [1,1,hidden,2*inter_pad]
            down_proj_weight = down_t.unsqueeze(0).unsqueeze(0)
        else:
            gate_up_weight = None
            down_proj_weight = None

        gu_n = 2 * padded_per_device
        down_k = padded_per_device
        dram_shard = _DRAM_SHARD_MLP and tp > 1

        if dram_shard and can_dram_shard(self.hidden_size, gu_n, dtype=dtype):
            self.gate_up_proj = DramShardedLinear(
                gate_up_weight,
                mesh_device,
                col_mapper,
                k=self.hidden_size,
                n=gu_n,
                dtype=dtype,
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"gate_up_proj.weight.ws{tp_suffix}{pad_suffix}{dtype_suffix}"
                ),
            )
        else:
            gate_up_proj = ttnn.as_tensor(
                gate_up_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"gate_up_proj.weight{tp_suffix}{pad_suffix}{dtype_suffix}"
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.gate_up_proj = lambda x: ttnn.linear(x, gate_up_proj)

        if dram_shard and can_dram_shard(down_k, self.hidden_size, dtype=dtype):
            self.down_proj = DramShardedLinear(
                down_proj_weight,
                mesh_device,
                row_mapper,
                k=down_k,
                n=self.hidden_size,
                dtype=dtype,
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"down_proj.weight.ws{tp_suffix}{pad_suffix}{dtype_suffix}"
                ),
            )
        else:
            down_proj = ttnn.as_tensor(
                down_proj_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=row_mapper,
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"down_proj.weight{tp_suffix}{pad_suffix}{dtype_suffix}"
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.down_proj = lambda x: ttnn.linear(x, down_proj)

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # Fused gate/up projection: one matmul produces [.., 2*inter_pad/device]
        # laid out as [up_i | gate_i]. Split with the padded half-width so TILE
        # slice bounds stay aligned (264 would round to 288 and break down_proj).
        gate_up = self.gate_up_proj(hidden_states)
        shard = self._inter_per_device
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
