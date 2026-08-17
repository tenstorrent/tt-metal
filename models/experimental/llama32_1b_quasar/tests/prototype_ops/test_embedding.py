# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.embedding`` — two distinct call families the model drives.

Call site 1 — token embedding lookup:
  modules/embedding/embedding_1d.py:139-144  (Embedding1D.forward)
    out = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT,
                         memory_config=self.config.output_memcfg)
  * input ids: uint32, ROW_MAJOR, shape [1, 1, 1, seq] (model) — here [1, seq]
  * weight:    bf16, ROW_MAJOR, table [VOCAB, DIM] (model stores [1, 1, VOCAB, DIM])
  * output:    [1, seq, DIM] in TILE_LAYOUT
  Reference: torch weight[ids].

Call site 2 — RoPE cos/sin table lookup (decode):
  modules/rope/rope_1d.py:172-173  (RotarySetup1D.decode_forward)
    cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
  * rot_idxs:   uint32, ROW_MAJOR, shape [1, batch] (padded to nearest 32,
                see rope_1d.py:489-490 prepare_rot_idxs)
  * cos_matrix: bf16, TILE_LAYOUT, table [1, 1, max_seq_len, head_dim]
  * output:     [1, batch, head_dim]
  Reference: torch table[rot_idxs].

Both compare to a torch gather reference via PCC.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U


# =============================================================================
# Call family 1: token embedding
# =============================================================================


@U.with_default_mesh()
@pytest.mark.parametrize("seq", U.PREFILL_SEQ_LENS, ids=lambda s: f"seq{s}")
def test_embedding_token(ttnn_mesh_device, reset_seeds, seq):
    mesh = ttnn_mesh_device
    vocab, dim = U.VOCAB, U.DIM

    weight_torch = U.torch_rand((vocab, dim))
    ids_torch = torch.randint(0, vocab, (1, seq), dtype=torch.int32)

    # weight: [1, 1, VOCAB, DIM] ROW_MAJOR bf16 (embedding_1d.py resolves ROW_MAJOR).
    w = U.to_tt(weight_torch.reshape(1, 1, vocab, dim), mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
    # ids: [1, 1, 1, seq] uint32 ROW_MAJOR (embedding_1d.py forward docstring).
    ids = U.to_tt(ids_torch.reshape(1, 1, 1, seq), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.embedding(ids, w, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ref = weight_torch[ids_torch[0].long()]  # [seq, DIM]
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# =============================================================================
# Call family 2: RoPE cos/sin table lookup
# =============================================================================


@U.with_default_mesh()
@pytest.mark.parametrize("batch", U.DECODE_BATCHES, ids=lambda b: f"batch{b}")
def test_embedding_rope_lookup(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device
    head_dim = U.HEAD_DIM
    max_seq_len = 128  # small tile-aligned RoPE table for the emulator

    table_torch = U.torch_rand((max_seq_len, head_dim))  # cos/sin table [max_seq, head_dim]

    # rot_idxs are padded to the nearest tile boundary (rope_1d.py:489-490).
    pad = (-batch) % U.TILE
    ids_torch = torch.randint(0, max_seq_len, (1, batch), dtype=torch.int32)
    ids_torch = torch.nn.functional.pad(ids_torch, (0, pad), "constant", 0)  # [1, nearest_32(batch)]

    # cos_matrix: [1, 1, max_seq_len, head_dim] TILE_LAYOUT bf16 (rope_1d.py:397-404).
    cos = U.to_tt(table_torch.reshape(1, 1, max_seq_len, head_dim), mesh, layout=ttnn.TILE_LAYOUT)
    # rot_idxs: [1, nearest_32(batch)] uint32 ROW_MAJOR (prepare_rot_idxs).
    ids = U.to_tt(ids_torch, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.embedding(ids, cos, layout=ttnn.TILE_LAYOUT)

    ref = table_torch[ids_torch[0].long()]  # [nearest_32(batch), head_dim]
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
