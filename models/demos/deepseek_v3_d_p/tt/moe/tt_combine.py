# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MoE Combine Module (TTNN Implementation)

This module routes expert-processed tokens back to their origin devices and accumulates
weighted contributions at each token's original position. It is the inverse of TtDispatchModule
and sits between TtRoutedExpert (which processes the dispatched tokens) and the MoE aggregation
step (which reduces the num_experts_per_tok contributions per token).

For each expert slot in dispatched_buffer and its corresponding metadata entry, the combine kernel:
  1. Reads metadata fields written by dispatch:
       [0] linearized_mesh_coord  — source device coordinate
       [1] token_idx              — original token index within the source device's sequence
       [2] topk_idx               — which top-k slot this expert contribution corresponds to
       [3] routed_expert          — global expert ID
       [4] weight                 — router weight for this (token, expert) pair
  2. Multiplies the expert output embedding by the router weight.
  3. Writes the weighted embedding to the origin device's output buffer at position
     [token_idx, topk_idx]: locally via NOC if the origin is the same device, or remotely
     via fabric if it is a different device in the dispatch group.

Each destination device accumulates a token-centric output buffer: for each token, up to
num_experts_per_tok expert contributions are written at their respective top-k indices.
Only slots corresponding to experts in this dispatch group are populated; slots for experts
from other dispatch groups contain uninitialized values. The per-device output shape is:
  output: (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim)

TtDispatchModule produces the dispatched_buffer and metadata consumed here.
"""

import os
from pathlib import Path

import torch
from loguru import logger
from tracy import Profiler, signpost

import ttnn
from models.common.lightweightmodule import LightweightModule

_tracy = Profiler()
_tracy.disable()
_PROFILE_OPS = os.getenv("TT_DS_PROFILE_OPS") == "1"

# Capture combine inputs at specific layers to .pt files for off-machine replay.
# TT_DS_CAPTURE_COMBINE_LAYERS supports:
#   - unset / empty: no capture (default)
#   - "all": capture every layer whose forward() runs (i.e., every MoE layer)
#   - comma-separated integers (e.g., "3,30,60"): capture only those layer indices
# Files land at <TT_DS_COMBINE_CAPTURE_DIR>/L<layer>/col<k>.pt, one per Galaxy dispatch group.
#
# By default we save only the *routing* tensors (metadata, counts, offsets) and the
# buffer shape descriptor, NOT the dispatched_buffer itself. The combine kernel's
# per-iteration time is determined entirely by routing volume + pattern; the byte
# values in the FFN output buffer do not affect kernel cycle count. The replay
# test allocates a zero buffer of the right shape on the target device.
# Set TT_DS_CAPTURE_COMBINE_FULL_BUFFER=1 to also save the real buffer (huge:
# ~23 GB per col at 25K isl) — only needed if you want PCC validation on replay.
_CAPTURE_LAYERS_STR = os.getenv("TT_DS_CAPTURE_COMBINE_LAYERS", "").strip()
_CAPTURE_ALL = _CAPTURE_LAYERS_STR.lower() == "all"
_CAPTURE_LAYERS = set() if _CAPTURE_ALL else {int(x) for x in _CAPTURE_LAYERS_STR.split(",") if x.strip()}
_CAPTURE_DIR = Path(
    os.getenv(
        "TT_DS_COMBINE_CAPTURE_DIR",
        str(Path(os.getenv("TT_METAL_HOME", ".")) / "generated" / "combine_capture"),
    )
)
_CAPTURE_FULL_BUFFER = os.getenv("TT_DS_CAPTURE_COMBINE_FULL_BUFFER", "0") == "1"


class TtCombineModule(LightweightModule):
    """TTNN wrapper around the prefill_combine device operation.

    Reads expert-processed token embeddings from dispatched_buffer and routes them back
    to their origin devices using dispatch metadata, accumulating weighted contributions
    at each token's original top-k slot. Produces the combined output consumed by the
    MoE aggregation step.
    See module docstring for full output buffer layout details.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        init_zeros: bool = True,
        layer_idx: int = -1,
    ):
        """
        Initialize combine module with configuration parameters.

        Args:
            mesh_device: TTNN mesh device.
            dispatch_group_size: Number of devices in each dispatch group (mesh rows for cluster_axis=0).
            num_dispatch_groups: Number of independent dispatch groups (mesh columns for cluster_axis=0).
            experts_per_chip: Number of experts hosted on each device.
            num_experts_per_tok: Number of experts each token is routed to (top-k).
            seq_len_per_chip: Number of tokens on each source device (output token dimension size).
            cluster_axis: Mesh axis along which combine communicates (0 = SP/dispatch axis).
            num_links: Number of fabric links for remote token writes.
            topology: Fabric topology for remote token writes.
            memory_config: Output memory configuration. Must be interleaved (L1 or DRAM).
            init_zeros: Whether to zero-initialize the output buffer before writing.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.dispatch_group_size = dispatch_group_size
        self.num_dispatch_groups = num_dispatch_groups
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config
        self.init_zeros = init_zeros
        self.layer_idx = layer_idx

    def _capture_inputs(
        self,
        dispatched_buffer: ttnn.Tensor,
        dispatched_metadata: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
    ):
        # Lazy import to avoid circular dep at module load time.
        from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_ep_mesh_composer

        composer = get_ep_mesh_composer(self.mesh_device)
        # Always capture metadata + counts + offsets (drives routing pattern → ethernet traffic).
        meta_host = ttnn.to_torch(dispatched_metadata, mesh_composer=composer, dtype=torch.int32)
        counts_host = ttnn.to_torch(
            ttnn.unsqueeze_to_4D(expert_token_counts), mesh_composer=composer, dtype=torch.int32
        ).squeeze(2)
        offsets_host = ttnn.to_torch(
            ttnn.unsqueeze_to_4D(expert_region_offsets), mesh_composer=composer, dtype=torch.int32
        ).squeeze(2)
        # Buffer values do not affect combine kernel time; capture only when explicitly requested
        # (e.g., for PCC validation on replay). Otherwise just record the shape.
        if _CAPTURE_FULL_BUFFER:
            buf_host = ttnn.to_torch(dispatched_buffer, mesh_composer=composer, dtype=torch.bfloat16)
        else:
            buf_host = None
        buf_shape = list(dispatched_buffer.shape)  # per-device shape, e.g. (1, 1, 8, max_tokens, emb_dim)

        layer_dir = _CAPTURE_DIR / f"L{self.layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        experts_per_col = self.experts_per_chip * self.dispatch_group_size
        for col in range(self.num_dispatch_groups):
            # Use .clone() (not .contiguous()) — contiguous() returns the same view when the
            # slice is already a contiguous prefix of the source, leaving the full source
            # storage attached. torch.save then pickles the entire storage, blowing the file
            # up by 4× (one per dispatch group). clone() forces a separate allocation.
            meta_col = meta_host[col : col + 1].clone()
            expert_lo = col * experts_per_col
            expert_hi = (col + 1) * experts_per_col
            counts_col = counts_host[col : col + 1, :, expert_lo:expert_hi].clone()
            offsets_col = offsets_host[col : col + 1, :, expert_lo:expert_hi].clone()

            # Per-column buffer shape: replace per-device leading dim (size 1) with
            # dispatch_group_size, since composing across mesh fills that axis. Buffer is
            # 4D per device (1, 1, max_dispatch_buffer_token_size, emb_dim).
            buf_col_shape = [1, self.dispatch_group_size] + list(buf_shape[2:])
            max_dispatch_buffer_token_size = int(buf_col_shape[-2])
            emb_dim_val = int(buf_col_shape[-1])
            max_tokens_per_expert = max_dispatch_buffer_token_size // self.experts_per_chip
            save_dict = {
                "dispatched_metadata": meta_col,
                "expert_token_counts": counts_col,
                "expert_region_offsets": offsets_col,
                "config": {
                    "dispatch_group_size": self.dispatch_group_size,
                    "num_dispatch_groups": 1,
                    "experts_per_chip": self.experts_per_chip,
                    "num_routed_experts": experts_per_col,
                    "num_experts_per_tok": self.num_experts_per_tok,
                    "seq_len_per_chip": self.seq_len_per_chip,
                    "emb_dim": emb_dim_val,
                    "max_dispatched_tokens_per_expert": max_tokens_per_expert,
                    "max_dispatch_buffer_token_size": max_dispatch_buffer_token_size,
                    "metadata_len": int(meta_col.shape[-1]),
                    "buffer_shape": buf_col_shape,  # 4D: (1, dgs, max_dispatch_buffer_token_size, emb_dim)
                    "galaxy_column": col,
                    "layer_idx": self.layer_idx,
                },
            }
            if buf_host is not None:
                save_dict["dispatched_buffer"] = buf_host[col : col + 1].clone()

            save_path = layer_dir / f"col{col}.pt"
            torch.save(save_dict, save_path)

            n_valid = int(counts_col.sum().item())
            meta_mb = meta_col.numel() * 4 / (1024**2)
            buf_note = (
                f"+ buffer {tuple(save_dict['dispatched_buffer'].shape)} "
                f"({save_dict['dispatched_buffer'].numel() * 2 / (1024**3):.2f} GB bf16)"
                if buf_host is not None
                else "(buffer: shape-only, zero-fill on replay)"
            )
            logger.info(
                f"[combine_capture] L{self.layer_idx} col{col} -> {save_path}  "
                f"meta {tuple(meta_col.shape)} ({meta_mb:.1f} MB int32), "
                f"valid_slots_total={n_valid}, {buf_note}"
            )

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        dispatched_metadata: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
    ):
        """
        Route expert-processed tokens back to origin devices and accumulate weighted contributions.

        For each expert slot in dispatched_buffer, the kernel reads the corresponding metadata
        entry to determine the origin device, original token index, top-k slot, and router weight.
        It multiplies the expert output by the weight and writes it to the origin device's output
        buffer: locally via NOC if the origin is the same device, or remotely via fabric if the
        origin is a different device in the dispatch group.

        Args:
            dispatched_buffer: Expert-processed token embeddings produced by TtRoutedExpert.
                Shape per device: (1, 1, max_dispatch_buffer_token_size, emb_dim).
                BFLOAT16 ROW_MAJOR.
            dispatched_metadata: Per-token routing metadata produced by TtDispatchModule.forward().
                Shape per device: (1, 1, max_dispatch_buffer_token_size, metadata_len=5).
                INT32 ROW_MAJOR. Fields per token: [linearized_mesh_coord, token_idx, topk_idx, routed_expert, weight].
            expert_token_counts: Number of tokens dispatched to each expert, used to bound the
                valid range of token slots read per expert in dispatched_buffer.
                Shape per device: (1, 1, num_routed_experts). INT32 ROW_MAJOR.
            expert_region_offsets: Expert region offsets (shared across source devices in a
                dispatch group) giving each expert's region start position in dispatched_buffer.
                Same shape/layout as expert_token_counts. Produced by offset_cumsum.
                Shape per device: (1, 1, num_routed_experts). INT32 or UINT32 ROW_MAJOR.

        Returns:
            output: Combined token embeddings with weighted expert contributions at each token's
                original top-k slot. Produced by ttnn.experimental.deepseek_prefill.combine.
                Shape per device: (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim).
                BFLOAT16 ROW_MAJOR. Token slots for experts outside this dispatch group contain
                uninitialized values.
        """
        if _CAPTURE_ALL or self.layer_idx in _CAPTURE_LAYERS:
            try:
                self._capture_inputs(dispatched_buffer, dispatched_metadata, expert_token_counts, expert_region_offsets)
            except Exception as e:
                logger.error(f"[combine_capture] L{self.layer_idx} failed: {e}")

        if _PROFILE_OPS:
            signpost(header=f"combine_L{self.layer_idx}_start")
            _tracy.enable()
        output = ttnn.experimental.deepseek_prefill.combine(
            dispatched_buffer,
            dispatched_metadata,
            expert_token_counts,
            expert_region_offsets,
            dispatch_group_size=self.dispatch_group_size,
            experts_per_chip=self.experts_per_chip,
            num_experts_per_tok=self.num_experts_per_tok,
            seq_len_per_chip=self.seq_len_per_chip,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=self.memory_config,
            init_zeros=self.init_zeros,
        )
        if _PROFILE_OPS:
            _tracy.disable()
            signpost(header=f"combine_L{self.layer_idx}_end")
        return output
