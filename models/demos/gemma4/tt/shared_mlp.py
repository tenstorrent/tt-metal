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
from models.demos.gemma4.tt.ccl import (
    ccl_allreduce,
    ccl_allreduce_fractured,
    ccl_lane_gather_rows,
    ccl_lane_scatter_rows,
)
from models.demos.gemma4.tt.compute_config import gelu_variant
from models.demos.gemma4.tt.decode_matmul import DecodeLinear, flag, pipeline_row_for
from models.demos.gemma4.tt.dram_sharded import TILE_SIZE, DramShardedLinear, can_dram_shard
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# DRAM-width-sharded decode matmuls for the shared MLP. On by default for
# multi-device (tp>1); the single width-sharded weight is the same size as the
# interleaved one, so there is no memory cost. Set GEMMA4_MLP_DRAM_SHARD=0 to
# fall back to plain interleaved matmuls.
_DRAM_SHARD_MLP = os.environ.get("GEMMA4_MLP_DRAM_SHARD", "1") != "0"
# GEMMA4_MLP_DECODE_1D=1: explicit 1D-multicast decode configs on the interleaved gate_up / down matmuls (the
# MoE path): gate_up streams K in 8-tile blocks, down takes its whole K (<=32 tiles) in one block. Measured on
# a P150 (26B-A4B TP=4, trace replay): gate_up 43 -> 19 us, down 25 -> 16 us, same HiFi2 numerics.
_MLP_GATE_UP_BLOCK_W = int(os.environ.get("GEMMA4_MLP_GATE_UP_BLOCK_W", "8"))  # env overrides for the K-block sweep
_MLP_DOWN_BLOCK_W = int(os.environ.get("GEMMA4_MLP_DOWN_BLOCK_W", "32"))


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


def _geglu_fused_rows(x) -> bool:
    """True for a ``[1, 1, M, N]`` activation of at most four row tiles: the decode rows of the tp shapes (M <= 32)
    and of the lane engine (lanes x 32 gathered rows)."""
    shape = x.shape
    return len(shape) == 4 and int(shape[0]) == 1 and int(shape[1]) == 1 and 1 <= int(shape[2]) <= 4 * TILE_SIZE


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
        # 2D weight fracture: shard the projections over rows*cols chips (one weight copy on the mesh);
        # the residual stays replicated and ccl_allreduce_fractured completes down_proj.
        fractured = bool(mesh_config and getattr(mesh_config, "weight_fracture", False))
        ways = mesh_config.fracture_ways if fractured else tp
        if fractured:
            tp_suffix = f"_f{mesh_config.mesh_shape[0]}x{mesh_config.mesh_shape[1]}"
        else:
            tp_suffix = f"_tp{tp}" if tp > 1 else ""
        self._fractured = fractured

        # Tag the cache filenames with the weight dtype so that flipping a
        # SharedMLP weight's dtype (e.g. bf16 → bfp8 for DRAM-pressure relief)
        # doesn't collide with a previously-cached file that holds the same
        # logical weight at a different dtype. The rest of the model's cache
        # entries are unaffected and stay reusable across runs.
        _dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        dtype_suffix = f"_{_dtype_str}"

        if fractured:
            # Host reshapes expose the mesh-row factor as dim 1 so each mesh axis shards a different dim
            # (the mapper requires it); per-chip shapes match plain TP.
            col_mapper = mesh_config.shard_mapper(mesh_device, mesh_dims=(1, 3))
            row_mapper = mesh_config.shard_mapper(mesh_device, mesh_dims=(1, 2))
        elif tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        # Pad intermediate to a tile-aligned per-device size (same pattern as
        # experts/weights.py). At TP=8, 2112/8=264 is not tile-aligned; TILE
        # slice rounds the GeGLU half to 288 while an unpadded down_proj stays
        # K=264 → matmul width/height mismatch on WH/BH e2e.
        if ways > 1:
            per_device = self.intermediate_size // ways
            padded_per_device = ((per_device + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
            pad_amount = padded_per_device * ways - self.intermediate_size
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
            if ways > 1:
                gate_shards = torch.chunk(gate_t, ways, dim=-1)
                up_shards = torch.chunk(up_t, ways, dim=-1)
                gate_up_t = torch.cat([torch.cat([up_shards[i], gate_shards[i]], dim=-1) for i in range(ways)], dim=-1)
            else:
                gate_up_t = torch.cat([up_t, gate_t], dim=-1)
            if fractured:
                rows, cols = mesh_config.mesh_shape
                k_dim = gate_up_t.shape[0]
                two_n = 2 * padded_per_device
                # N is chunk-major (ways chunks of [up_i|gate_i]): expose rows.
                gate_up_weight = (
                    gate_up_t.reshape(k_dim, rows, cols * two_n).permute(1, 0, 2).contiguous().unsqueeze(0)
                )  # [1, rows, K, cols*2n]
                # down K is chunk-major along dim 0: reshape splits it directly.
                down_proj_weight = (
                    down_t.reshape(rows, cols * padded_per_device, -1).contiguous().unsqueeze(0)
                )  # [1, rows, cols*k, hidden]
            else:
                gate_up_weight = gate_up_t.unsqueeze(0).unsqueeze(0)  # [1,1,hidden,2*inter_pad]
                down_proj_weight = down_t.unsqueeze(0).unsqueeze(0)
        else:
            gate_up_weight = None
            down_proj_weight = None

        gu_n = 2 * padded_per_device
        down_k = padded_per_device
        # MoE (26B-A4B): DRAM-width-sharded decode matmuls drop full-layer decode
        # PCC to ~0.93 vs HF (threshold 0.99) on BH 1x4. Keep interleaved path
        # for MoE; dense 12B/31B retain the sharded opt.
        is_moe = bool(getattr(hf_config, "enable_moe_block", False))
        lab_tp1 = os.environ.get("GEMMA4_DRAM_SHARD_TP1", "0") == "1"  # one TP shard on one chip (the layer lab)
        dram_shard = _DRAM_SHARD_MLP and (tp > 1 or lab_tp1) and not is_moe and not fractured
        decode_1d = flag("GEMMA4_MLP_DECODE_1D")

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
                pipeline_row=pipeline_row_for(mesh_device, self.hidden_size, "gate_up", tp),
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

            self.gate_up_proj = DecodeLinear(
                gate_up_proj,
                mesh_device,
                k=self.hidden_size,
                n=gu_n,
                enabled=decode_1d,
                max_block_w=_MLP_GATE_UP_BLOCK_W,
            )

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
                pipeline_row=pipeline_row_for(mesh_device, self.hidden_size, "down", tp),
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

            self.down_proj = DecodeLinear(
                down_proj, mesh_device, k=down_k, n=self.hidden_size, enabled=decode_1d, max_block_w=_MLP_DOWN_BLOCK_W
            )

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # Fused gate/up projection: one matmul produces [.., 2*inter_pad/device]
        # laid out as [up_i | gate_i]. Split with the padded half-width so TILE
        # slice bounds stay aligned (264 would round to 288 and break down_proj).
        lane_sharded = bool(self.mesh_config is not None and getattr(self.mesh_config, "lane_sharded", False))
        owner_fast = None
        if lane_sharded:
            # Slice-3 choreography: rows are lane-local but the weights are
            # fractured across lanes, so gather all lanes' rows first ...
            hidden_states = ccl_lane_gather_rows(hidden_states, self.mesh_config, ccl_manager=self.ccl_manager)
            owner = getattr(self.mesh_config, "_g4_single_lane_prefill_owner", None)
            if (
                owner is not None
                and self.ccl_manager is not None  # the lane all-reduce needs the manager's topology
                and os.environ.get("GEMMA4_MLP_LANE_FASTPATH", "1") != "0"
                and int(hidden_states.shape[2]) > 32
            ):
                # Single-user prefill: only the owner lane's block of rows is real. The weights' K
                # chunks are spread over the lanes, so every lane still computes its partial, but on
                # the owner's rows only (1x instead of lanes x rows); the lane all-reduce then sums
                # the K partials and every lane holds the owner's result (the other lanes' rows are
                # discarded downstream), the same fast path the routed experts take.
                block = int(hidden_states.shape[2]) // self.mesh_config.lanes
                lo, hi = owner * block, (owner + 1) * block
                owner_rows = ttnn.slice(hidden_states, [0, 0, lo, 0], [1, 1, hi, hidden_states.shape[-1]])
                hidden_states.deallocate(True)
                hidden_states = owner_rows
                owner_fast = owner
        shard = self._inter_per_device
        if (
            flag("GEMMA4_MLP_GEGLU_SHARDED")
            and not lane_sharded
            and isinstance(self.gate_up_proj, DramShardedLinear)
            and isinstance(self.down_proj, DramShardedLinear)
            and self.down_proj.accepts_sharded_rows()
            and len(hidden_states.shape) == 4
            and 1 <= int(hidden_states.shape[2]) <= TILE_SIZE
            and shard % TILE_SIZE == 0
        ):
            # GEMMA4_MLP_GEGLU_SHARDED=1: gate_up's L1 width-sharded output -> up * gelu(gate) straight into
            # down's L1 activation layout (custom_ops/geglu_sharded): no sharded -> interleaved copy after
            # gate_up, no reshard before down, the 2I-wide activation never leaves L1.
            from models.demos.gemma4.custom_ops.geglu_sharded import GegluShardedDecode

            rows = int(hidden_states.shape[2])
            gate_up_sh = self.gate_up_proj(hidden_states, keep_sharded=True)
            hidden_sh = GegluShardedDecode.op(gate_up_sh, self.down_proj.act_memory_config)
            gate_up_sh.deallocate(True)
            output = self.down_proj(hidden_sh, rows=rows)
            if self._fractured:
                output = ccl_allreduce_fractured(output, self.mesh_config, self.ccl_manager)
            elif self.mesh_config is not None and self.mesh_config.tp > 1:
                output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)
            return output
        gate_up = self.gate_up_proj(hidden_states)
        s = gate_up.shape[-2]
        if flag("GEMMA4_MLP_GEGLU_FUSED") and _geglu_fused_rows(gate_up) and shard % TILE_SIZE == 0:
            # GEMMA4_MLP_GEGLU_FUSED=1: up * gelu(gate) over the fused [up | gate] output as one program
            # (custom_ops/geglu_decode) instead of slice, slice, gelu, mul on the decode rows (one row tile on the
            # tp shapes, lanes x 32 rows on the lane engine); prefill rows keep the composite chain (measured null).
            from models.demos.gemma4.custom_ops.geglu_decode import GegluDecode

            hidden = GegluDecode.op(gate_up)
            gate_up.deallocate(True)
        else:
            up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, s, shard])
            gate = ttnn.slice(gate_up, [0, 0, 0, shard], [1, 1, s, 2 * shard])
            gate_up.deallocate(True)

            # Prefer Accurate over FastLut/Tanh for device PCC (see compute_config): the Tanh variant
            # dropped the E4B full-model PCC from 0.9846 to 0.9578 (gate 0.96) on bh_quietbox_2.
            gate = ttnn.gelu(gate, variant=gelu_variant())
            hidden = ttnn.mul(gate, up)
            gate.deallocate(True)
            up.deallocate(True)

        # output = hidden @ down_proj
        output = self.down_proj(hidden)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if lane_sharded and owner_fast is not None:
            # owner rows on every lane: sum the lanes' K partials, then the tp partials
            output = ccl_allreduce(
                output,
                self.mesh_config,
                self.ccl_manager,
                axis=self.mesh_config.ep_axis,
                group_size=self.mesh_config.lanes,
            )
            if self.mesh_config.tp > 1:
                output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)
        elif lane_sharded:
            # ... and reduce_scatter re-shards rows to their lanes while
            # summing the cross-lane K-chunks (plus the tp-axis all-reduce).
            output = ccl_lane_scatter_rows(output, self.mesh_config, ccl_manager=self.ccl_manager)
        elif self._fractured:
            output = ccl_allreduce_fractured(output, self.mesh_config, self.ccl_manager)
        elif self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
