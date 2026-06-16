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
        self.tp = tp
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

        if state_dict:
            # Fuse gate_proj + up_proj into a single column-parallel weight so the
            # forward runs ONE matmul producing a [gate | up] slab (mirrors the wqkv
            # fusion in attention/weights.py). 
            gate_w = state_dict["gate_proj.weight"]  # [intermediate_size, hidden_size]
            up_w = state_dict["up_proj.weight"]  # [intermediate_size, hidden_size]
            if tp > 1:
                fused_list = []
                for i in range(tp):
                    wg_chunk = torch.chunk(gate_w, tp, dim=0)[i].transpose(-2, -1)
                    wu_chunk = torch.chunk(up_w, tp, dim=0)[i].transpose(-2, -1)
                    fused_list.append(torch.cat([wg_chunk, wu_chunk], dim=-1))  # [hidden, 2*int/tp]
                gate_up_proj_weight = torch.cat(fused_list, dim=-1).unsqueeze(0).unsqueeze(0)
            else:
                gate_up_proj_weight = (
                    torch.cat([gate_w.transpose(-2, -1), up_w.transpose(-2, -1)], dim=-1).unsqueeze(0).unsqueeze(0)
                )
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_up_proj_weight = None
            down_proj_weight = None

        # gate+up fused: column-parallel (shard output dim across TP devices)
        self.gate_up_proj = ttnn.as_tensor(
            gate_up_proj_weight,
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

        gate+up are fused (column-parallel), down is row-parallel + allreduce.
        """
        # Fused gate+up projection: one matmul produces the [gate | up] slab.
        gate_up = ttnn.linear(hidden_states, self.gate_up_proj)

        # Split the slab into gate / up halves (per-device width when column-parallel)
        # and fuse GeGLU into the multiply: fast-approx GELU on the gate half
        # (operand a) only, then elementwise * up. Replaces the separate ttnn.gelu +
        # ttnn.mul, matching the original fast_and_approximate_mode=True semantics.
        #
        # NOTE: when intermediate_size/tp is not a multiple of TILE_WIDTH (32) — e.g.
        # 2112/8 = 264 on T3K — this split is not tile-aligned, so ttnn.slice falls
        # back to a row-major composite path (correct, but a layout round-trip). Making
        # it tile-aligned means padding each per-device half up to a tile multiple in
        # the fused weight AND padding down_proj's per-device input dim to match;
        # deferred to the down_proj mem-config / config-tuning work.
        split = self.intermediate_size // self.tp
        gate = gate_up[..., :split]
        up = gate_up[..., split:]
        hidden = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, True)])
        gate_up.deallocate(True)

        # output = hidden @ down_proj
        output = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
