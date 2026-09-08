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
from models.demos.gemma4.tt.compute_config import gelu_variant
from models.demos.gemma4.tt.dram_sharded import (
    TILE_SIZE,
    DramShardedLinear,
    can_dram_shard,
    decode_in0_l1_enabled,
    interleaved_down_proj_prefill_config,
    interleaved_gate_up_prefill_config,
    linear_l1_safe,
    matmul_rows,
    prefill_linear_above_cutoff,
    prefill_lofi_ckc,
    prefill_matmul_lofi_enabled,
    prefill_progcfg_1d_for_width_sharded_in0,
    should_prefill_long_2d,
)
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
        # MoE (26B-A4B): DRAM-width-sharded decode matmuls drop full-layer decode
        # PCC to ~0.93 vs HF (threshold 0.99) on BH 1x4. Keep interleaved path
        # for MoE; dense 12B/31B retain the sharded opt.
        is_moe = bool(getattr(hf_config, "enable_moe_block", False))
        dram_shard = _DRAM_SHARD_MLP and tp > 1 and not is_moe

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
            # Interleaved-weight path (everything off Blackhole). Prefill gets an
            # explicit 1D program config from interleaved_gate_up_prefill_config —
            # see SharedMLP.__call__. Keep the weight as a tensor (not a lambda)
            # so the call site can pass program_config / out memcfg / HiFi4.
            self.gate_up_proj = ttnn.as_tensor(
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
            self.down_proj = ttnn.as_tensor(
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

    def _prepare_prefill_act(self, hidden_states, program_config):
        """Move ``hidden_states`` onto the layout the tuned prefill matmul wants.

        Prefer keeping a width-sharded LN output in place (caller matched a 1D
        progcfg to that shard grid). Only S2I when the auto path has no progcfg.
        Returns ``(act, owned)``.
        """
        if program_config is None:
            if hidden_states.is_sharded():
                # Decode (M <= TILE): the tuned prefill config declines, so we land
                # here with the pre-FF LN's width-sharded L1 island output — and
                # used to un-shard it into *DRAM*, spilling the activation the
                # island had just kept on-chip and making gate_up read in0 from
                # DRAM. The S2I happens either way; only its destination changes,
                # so retargeting it to L1 is bit-exact and costs no extra op.
                # The staged activation is [1, 1, 32, hidden_local] regardless of
                # batch (rows are tile-padded to 32), so the L1 cost does not grow
                # with batch or context length.
                dest = ttnn.DRAM_MEMORY_CONFIG
                if matmul_rows(hidden_states) <= TILE_SIZE and decode_in0_l1_enabled():
                    dest = ttnn.L1_MEMORY_CONFIG
                return ttnn.sharded_to_interleaved(hidden_states, dest), True
            return hidden_states, False
        if hidden_states.is_sharded():
            return hidden_states, False
        if hidden_states.memory_config().buffer_type != ttnn.BufferType.L1:
            return ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG), True
        return hidden_states, False

    def _gate_up_linear(self, hidden_states):
        """Fused gate+up matmul — DramShardedLinear on BH, tuned 1D prefill off BH."""
        if isinstance(self.gate_up_proj, DramShardedLinear):
            return self.gate_up_proj(hidden_states)

        # Interleaved-weight prefill: pin 1D progcfg + L1 interleaved out + HiFi4.
        # Decode (M<=TILE) may consume a width-sharded LN out on that core grid
        # and skip S2I. Prefill-sized 1D CBs on the same shard clash on Wormhole,
        # so M>TILE always S2I's to the tuned interleaved config.
        m = matmul_rows(hidden_states)
        k = int(hidden_states.shape[-1])
        n = int(self.gate_up_proj.shape[-1])
        # 12B long-4k (M=4096, K=N=3840): cutoff-reshape+LoFi beats auto+LoFi.
        # At M=2048 (31B T3K chunks) auto+LoFi still wins — keep that path.
        if should_prefill_long_2d(m) and m >= 4096:
            act, owned = self._prepare_prefill_act(hidden_states, None)
            gate_up = prefill_linear_above_cutoff(act, self.gate_up_proj)
            if owned:
                act.deallocate(True)
            return gate_up
        program_config, out_memcfg, compute_kernel_config = interleaved_gate_up_prefill_config(m, k, n)
        # Above the tuned 1D band, auto DRAM-in0 keeps HiFi2 by default; LoFi is
        # the isolate winner at M=2048.
        if program_config is None and compute_kernel_config is None and prefill_matmul_lofi_enabled(m):
            compute_kernel_config = prefill_lofi_ckc()
        # Decode (M<=TILE): the tuned prefill config declines, so the output would
        # follow the op default (DRAM) and the whole GeGLU group — 2x slice + the
        # gelu*mul — would run against DRAM. The tensor is
        # [1,1,32,2*inter/tp] bf16 = 344 KB, i.e. 5 KB/core, so keep the group on
        # L1. (The prefill-sized rejection of an L1 GeGLU intermediate above is a
        # seq=4096 L1-budget result and does not apply at M=32.)
        if out_memcfg is None and m <= TILE_SIZE:
            out_memcfg = ttnn.L1_MEMORY_CONFIG
        if program_config is not None and hidden_states.is_sharded():
            # Decode (M<=TILE) keep-sharded island is proven. Prefill-sized 1D
            # CBs on the LN width-shard clash with the rest of the layer on
            # Wormhole — S2I to the tuned interleaved config instead.
            matched = (
                prefill_progcfg_1d_for_width_sharded_in0(m, k, n, hidden_states.memory_config())
                if m <= TILE_SIZE
                else None
            )
            if matched is not None:
                program_config = matched
            else:
                act = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
                gate_up = linear_l1_safe(
                    act,
                    self.gate_up_proj,
                    program_config=program_config,
                    memory_config=out_memcfg,
                    compute_kernel_config=compute_kernel_config,
                )
                act.deallocate(True)
                return gate_up
        act, owned = self._prepare_prefill_act(hidden_states, program_config)
        gate_up = linear_l1_safe(
            act,
            self.gate_up_proj,
            program_config=program_config,
            memory_config=out_memcfg,
            compute_kernel_config=compute_kernel_config,
        )
        if owned:
            act.deallocate(True)
        return gate_up

    def _down_proj_linear(self, hidden):
        """Row-parallel down projection — DramShardedLinear on BH, tuned 1D prefill off BH."""
        if isinstance(self.down_proj, DramShardedLinear):
            return self.down_proj(hidden)

        # GeGLU leaves interleaved after gate_up L1-out + slice; keep interleaved path.
        m = matmul_rows(hidden)
        k = int(hidden.shape[-1])
        n = int(self.down_proj.shape[-1])
        # Above the 1D-tuned band, auto streams in0 from DRAM with in0_block_w=1.
        # Cutoff-reshape 2D is the isolate winner (see prefill_linear_above_cutoff).
        if should_prefill_long_2d(m):
            act, owned = self._prepare_prefill_act(hidden, None)
            output = prefill_linear_above_cutoff(act, self.down_proj)
            if owned:
                act.deallocate(True)
            return output
        program_config, out_memcfg, compute_kernel_config = interleaved_down_proj_prefill_config(m, k, n)
        # Decode: L1 writeback, same lever (and same bit-exactness) as o_proj.
        # Consumer is the all-reduce.
        if out_memcfg is None and m <= TILE_SIZE:
            out_memcfg = ttnn.L1_MEMORY_CONFIG
        act, owned = self._prepare_prefill_act(hidden, program_config)
        output = linear_l1_safe(
            act,
            self.down_proj,
            program_config=program_config,
            memory_config=out_memcfg,
            compute_kernel_config=compute_kernel_config,
        )
        if owned:
            act.deallocate(True)
        return output

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # Fused gate/up projection: one matmul produces [.., 2*inter_pad/device]
        # laid out as [up_i | gate_i]. Split with the padded half-width so TILE
        # slice bounds stay aligned (264 would round to 288 and break down_proj).
        gate_up = self._gate_up_linear(hidden_states)
        shard = self._inter_per_device
        s = gate_up.shape[-2]
        # Keep the split halves wherever gate_up landed (L1 at decode, DRAM at
        # prefill) so the group does not bounce.
        geglu_mc = gate_up.memory_config() if not gate_up.is_sharded() else None
        up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, s, shard], memory_config=geglu_mc)
        gate = ttnn.slice(gate_up, [0, 0, 0, shard], [1, 1, s, 2 * shard], memory_config=geglu_mc)
        gate_up.deallocate(True)

        # NOTE: in PREFILL this GeGLU intermediate stays in DRAM. Do NOT move it
        # to L1 to give down_proj an L1 in0: the intermediate is
        # [seq, intermediate_size/tp], and at tp=1/tp=2 that is 4-8x wider than
        # tp=8 and OOMs L1 at seq=4096.
        # DECODE is the opposite case and does use L1 (see _gate_up_linear): seq
        # is one tile, so the whole group is 344 KB, and what L1 buys there is the
        # gate_up WRITE — its output never goes through DRAM — not the down_proj
        # read. ``geglu_mc`` inherits whichever the matmul picked, so neither case
        # is hard-coded here.
        # Preserve main's Accurate GeLU policy and the source tip's memory placement.
        activated = ttnn.gelu(gate, variant=gelu_variant(), memory_config=geglu_mc)
        hidden = ttnn.mul(activated, up, memory_config=geglu_mc)
        activated.deallocate(True)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        output = self._down_proj_linear(hidden)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
