# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""KV cache and SSM state allocation for NemotronH-30B stateful decode.

KV cache layout (paged): [num_blocks, n_kv_heads, block_size, head_dim]
Page table:               [B, max_blocks_per_seq]
SSM state:                [B, NUM_SSM_HEADS, SSM_HEAD_DIM, SSM_STATE_SIZE]

All tensors are replicated on all 4 TP devices (KV heads are not sharded because
n_kv_heads=2 < TP=4; SSM state is per-device but identical since S=1 decode is
data-parallel across TP within a layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn

from .tp import _R

# Dense-attention constants (from dense_attention.py)
N_KV_HEADS = 2
HEAD_DIM = 128

# Mamba2 constants (from mamba2_layer.py)
NUM_SSM_HEADS = 64
SSM_HEAD_DIM = 64
SSM_STATE_SIZE = 128
CONV_DIM = 6144  # 4096 + 2*8*128 — conv1d input channels

# Layer-type indexing (PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME")
_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
M_LAYER_INDICES: list[int] = [i for i, t in enumerate(_PATTERN) if t == "M"]  # 23 M-layers
D_LAYER_INDICES: list[int] = [i for i, t in enumerate(_PATTERN) if t == "*"]  # 6 D-layers
E_LAYER_INDICES: list[int] = [i for i, t in enumerate(_PATTERN) if t == "E"]  # 23 E-layers
N_M_LAYERS = len(M_LAYER_INDICES)  # 23
N_D_LAYERS = len(D_LAYER_INDICES)  # 6
N_E_LAYERS = len(E_LAYER_INDICES)  # 23
N_ROUTED_EXPERTS = 128  # matches moe_gate.py

# Default paged-cache geometry
DEFAULT_BLOCK_SIZE = 32
DEFAULT_MAX_SEQ_LEN = 4096


@dataclass
class DecoderState:
    """All persistent device-side state for one generation sequence.

    ssm_states:      23 tensors [B, H, D, N] — one per M-layer, persistent state inputs
    ssm_state_outs:  23 pre-allocated output tensors — model writes new state in-place;
                     advance() copies these to ssm_states for the next step
    conv_states:     23 × (h_tm3, h_tm2, h_tm1) each [B,1,6144]
    conv_state_outs: 23 × (Z0, Z1, Z2) pre-allocated outputs — model writes in-place
    kv_caches:       6 (k_tt, v_tt) pairs — updated in-place by paged_update_cache
    page_tables:     6 page tables [B, max_blocks] — static sequential mapping
    current_pos:     [B] device tensor, written via copy_host_to_device_tensor each step
    """

    ssm_states: list = field(default_factory=list)
    ssm_state_outs: list = field(default_factory=list)
    conv_states: list = field(default_factory=list)
    conv_state_outs: list = field(default_factory=list)
    kv_caches: list = field(default_factory=list)
    page_tables: list = field(default_factory=list)
    current_pos: ttnn.Tensor | None = None
    gate_scores_tts: list = field(default_factory=list)  # reserved for predictive routing (unused)
    routing_tts: list = field(default_factory=list)  # reserved for predictive routing (unused)

    def advance(self):
        """Copy new SSM and conv states back to inputs for the next decode step.

        Call this after every forward (traced or eager) before the next forward.
        KV caches are already updated in-place; SSM/conv states require explicit copy.
        """
        for state_out, state_in in zip(self.ssm_state_outs, self.ssm_states):
            ttnn.assign(state_out, state_in)
        for out_tuple, in_tuple in zip(self.conv_state_outs, self.conv_states):
            for out_t, in_t in zip(out_tuple, in_tuple):
                ttnn.assign(out_t, in_t)

    def advance_routing(self, mesh_device) -> None:
        """D2H gate scores → CPU topk → H2D routing tensors for the next decode step.

        Only has effect when gate_scores_tts is populated (predictive routing mode).
        Call AFTER advance() so execute_trace on the next step sees updated routing.
        """
        if not self.gate_scores_tts or not self.routing_tts:
            return

        from .moe_gate import build_routing_from_scores_cpu

        for scores_tt, routing_tt in zip(self.gate_scores_tts, self.routing_tts):
            if scores_tt is None:
                continue

            # D2H: [1,1,1,128] BF16 = 256 bytes (tiny — 23 × 256B = 6KB per token)
            scores_cpu = ttnn.to_torch(
                scores_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
            )[
                0:1
            ]  # [1, 1, 1, 128] BF16 biased scores

            dense_4d = build_routing_from_scores_cpu(scores_cpu)  # [1,1,1,128] BF16 CPU

            # H2D: write into persistent ROW_MAJOR routing_tt via copy_host_to_device_tensor.
            # copy_host_to_device_tensor does NOT hang unlike ttnn.assign(new_device_tt, ...)
            # which deadlocks when the src is freshly allocated outside the trace.
            dense_host = ttnn.from_torch(dense_4d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(dense_host, routing_tt)


def _zeros_on_device(shape, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    cpu = torch.zeros(shape, dtype=torch.bfloat16)
    return ttnn.from_torch(
        cpu,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def allocate_decoder_state(
    mesh_device,
    B: int = 1,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> DecoderState:
    """Allocate all persistent device tensors for one decode sequence.

    Returns a DecoderState with:
      - 23 zero SSM states + 23 output slots
      - 6 paged KV cache pairs (k, v) for the Dense attention layers
      - 6 sequential page tables
      - current_pos = [B] zeros on device
    """
    num_blocks = (max_seq_len + block_size - 1) // block_size

    # SSM states: [B, H, D, N] = [B, 64, 64, 128]
    ssm_states = [
        _zeros_on_device([B, NUM_SSM_HEADS, SSM_HEAD_DIM, SSM_STATE_SIZE], mesh_device) for _ in range(N_M_LAYERS)
    ]
    ssm_state_outs = [
        _zeros_on_device([B, NUM_SSM_HEADS, SSM_HEAD_DIM, SSM_STATE_SIZE], mesh_device) for _ in range(N_M_LAYERS)
    ]

    # Conv states: 3 previous hBC values per M-layer, each [B, 1, CONV_DIM]
    # (h_tm3, h_tm2, h_tm1) — oldest first; zero-initialized.
    def _zero_conv():
        return tuple(_zeros_on_device([B, 1, CONV_DIM], mesh_device) for _ in range(3))

    conv_states = [_zero_conv() for _ in range(N_M_LAYERS)]
    conv_state_outs = [_zero_conv() for _ in range(N_M_LAYERS)]

    # Predictive routing tensors for the 23 E-layers.
    # gate_scores_tts: populated during trace as output tensors (like ssm_state_outs).
    # routing_tts:     persistent [1,1,1,128] BF16 ROW_MAJOR inputs read by the trace;
    #                  updated via copy_host_to_device_tensor (TILE_LAYOUT hangs with assign).
    gate_scores_tts = [None] * N_E_LAYERS
    routing_tts = [
        _zeros_on_device([1, 1, 1, N_ROUTED_EXPERTS], mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        for _ in range(N_E_LAYERS)
    ]

    # Paged KV caches: [num_blocks, n_kv_heads, block_size, head_dim]
    kv_shape = [num_blocks, N_KV_HEADS, block_size, HEAD_DIM]
    kv_caches = [
        (
            _zeros_on_device(kv_shape, mesh_device),
            _zeros_on_device(kv_shape, mesh_device),
        )
        for _ in range(N_D_LAYERS)
    ]

    # Page tables: sequential mapping block_i → physical_block_i
    pt_cpu = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
    page_tables = [
        ttnn.from_torch(
            pt_cpu,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=_R(mesh_device),
        )
        for _ in range(N_D_LAYERS)
    ]

    # Current sequence position: [B] zeros
    current_pos = ttnn.from_torch(
        torch.zeros(B, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )

    return DecoderState(
        ssm_states=ssm_states,
        ssm_state_outs=ssm_state_outs,
        conv_states=conv_states,
        conv_state_outs=conv_state_outs,
        kv_caches=kv_caches,
        page_tables=page_tables,
        current_pos=current_pos,
        gate_scores_tts=gate_scores_tts,
        routing_tts=routing_tts,
    )
