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

# Layer-type indexing (PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME")
_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
M_LAYER_INDICES: list[int] = [i for i, t in enumerate(_PATTERN) if t == "M"]  # 23 M-layers
D_LAYER_INDICES: list[int] = [i for i, t in enumerate(_PATTERN) if t == "*"]  # 6 D-layers
N_M_LAYERS = len(M_LAYER_INDICES)  # 23
N_D_LAYERS = len(D_LAYER_INDICES)  # 6

# Default paged-cache geometry
DEFAULT_BLOCK_SIZE = 32
DEFAULT_MAX_SEQ_LEN = 4096


@dataclass
class DecoderState:
    """All persistent device-side state for one generation sequence.

    ssm_states:    23 tensors [B, H, D, N] — one per M-layer, updated each step
    ssm_state_outs: 23 output tensors from the last forward — copy back to
                    ssm_states between trace executions
    kv_caches:     6 (k_tt, v_tt) pairs — updated in-place by paged_update_cache
    page_tables:   6 page tables [B, max_blocks] — static sequential mapping
    current_pos:   [B] device tensor incremented each decode step
    """

    ssm_states: list = field(default_factory=list)
    ssm_state_outs: list = field(default_factory=list)
    kv_caches: list = field(default_factory=list)
    page_tables: list = field(default_factory=list)
    current_pos: ttnn.Tensor | None = None

    def advance(self):
        """Copy new SSM states back to inputs for the next decode step.

        Call this after every forward (traced or eager) before the next forward.
        KV caches are already updated in-place during the forward; only SSM
        states need an explicit copy because they are not updated in-place.
        """
        for state_out, state_in in zip(self.ssm_state_outs, self.ssm_states):
            ttnn.assign(state_out, state_in)


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
        kv_caches=kv_caches,
        page_tables=page_tables,
        current_pos=current_pos,
    )
