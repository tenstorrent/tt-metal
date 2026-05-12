# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""T3.1 — DeltaNet TP across rows (V-heads sharded) on BH GLX mesh.

DeltaNet has NO cross-head communication during the recurrence. Each mesh row
handles a disjoint subset of V-heads:
   48 V-heads / 8 rows = 6 V-heads per row
   16 K-heads / 8 rows = 2 K-heads per row  (after GQA expansion stays per-row local)

Cols replicate the head shard (no further sharding needed for this isolated test).

  RED:    sharding splits heads across the wrong axis → garbled output (PCC <0.5);
          or non-head-divisible sharding fails ShardTensor2dMesh.
  GREEN:  per-row output matches the reference for the corresponding head slice (PCC>0.99);
          full assembled output (host-side concat) matches reference output.
"""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    recurrent_gated_delta_rule as _recurrent_ref,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import recurrent_gated_delta_rule_ttnn


@pytest.fixture(scope="module")
def mesh_2x2():
    """Use a smaller 2x2 mesh for faster iteration; same sharding logic as 8x4."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def mesh_8x4():
    """Full BH Galaxy 8x4."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("seq_len", [1, 4])
def test_deltanet_recurrent_row_sharded_full_galaxy(mesh_8x4, seq_len):
    """DeltaNet TP on full 8x4 BH Galaxy: 6 V-heads per row."""
    torch.manual_seed(0)
    mesh = mesh_8x4
    mesh_rows, mesh_cols = mesh.shape[0], mesh.shape[1]
    B = 1
    n_v = 48
    n_k = 16
    d = 128
    g_ratio = n_v // n_k

    assert n_v % mesh_rows == 0
    assert n_k % mesh_rows == 0

    q = torch.randn(B, seq_len, n_v, d, dtype=torch.float32) * 0.1
    k_pre = torch.randn(B, seq_len, n_k, d, dtype=torch.float32) * 0.1
    v = torch.randn(B, seq_len, n_v, d, dtype=torch.float32) * 0.1
    k = k_pre.repeat_interleave(g_ratio, dim=2)
    g = torch.randn(B, seq_len, n_v, dtype=torch.float32) * 0.05
    beta = torch.sigmoid(torch.randn(B, seq_len, n_v, dtype=torch.float32))

    out_ref, _ = _recurrent_ref(q, k, v, beta, g, output_final_state=True, use_qk_l2norm=True)

    def shard_heads(t, head_dim_idx):
        mapper = ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig(row_dim=head_dim_idx, col_dim=None))
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    q_tt = shard_heads(q, 2)
    k_tt = shard_heads(k, 2)
    v_tt = shard_heads(v, 2)
    beta_tt = shard_heads(beta, 2)
    g_tt = shard_heads(g, 2)
    out_tt, _ = recurrent_gated_delta_rule_ttnn(q_tt, k_tt, v_tt, beta_tt, g_tt, device=mesh)

    out_back_full = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.create_mesh_composer(mesh, ttnn.MeshComposerConfig([2, 0]))
    ).float()
    out_back = out_back_full[:B]

    pcc = _pcc(out_back, out_ref)
    print(f"T={seq_len} mesh=8x4: PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.mark.parametrize("seq_len", [1, 4])
def test_deltanet_recurrent_row_sharded(mesh_2x2, seq_len):
    """Shard V/K-heads across rows; replicate across cols. Run kernel per-shard."""
    torch.manual_seed(0)
    mesh_rows, mesh_cols = mesh_2x2.shape[0], mesh_2x2.shape[1]
    B = 1
    n_v = 48
    n_k = 16
    d = 128
    g_ratio = n_v // n_k  # 3

    # Divisibility: 48 / mesh_rows must be int; 16 / mesh_rows must be int
    assert n_v % mesh_rows == 0, f"n_v={n_v} not divisible by mesh_rows={mesh_rows}"
    assert n_k % mesh_rows == 0, f"n_k={n_k} not divisible by mesh_rows={mesh_rows}"
    n_v_per_row = n_v // mesh_rows
    n_k_per_row = n_k // mesh_rows

    # Build full inputs on host
    q = torch.randn(B, seq_len, n_v, d, dtype=torch.float32) * 0.1
    k_pre = torch.randn(B, seq_len, n_k, d, dtype=torch.float32) * 0.1
    v = torch.randn(B, seq_len, n_v, d, dtype=torch.float32) * 0.1
    k = k_pre.repeat_interleave(g_ratio, dim=2)
    g = torch.randn(B, seq_len, n_v, dtype=torch.float32) * 0.05
    beta = torch.sigmoid(torch.randn(B, seq_len, n_v, dtype=torch.float32))

    # Full reference (single-shard) output
    out_full, _ = _recurrent_ref(q, k, v, beta, g, output_final_state=True, use_qk_l2norm=True)
    print(f"full ref shape: {out_full.shape}")

    # Shard along head dim (dim=2). The mapper sends one head-slice to each row.
    # Cols replicate the same slice.
    # 2D: row_dim=2 (shard heads across rows), col_dim=None (replicate across cols).
    def shard_heads(t, head_dim_idx, mesh):
        cfg = ttnn.MeshMapperConfig(row_dim=head_dim_idx, col_dim=None)
        mapper = ttnn.create_mesh_mapper(mesh, cfg)
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    q_tt = shard_heads(q, 2, mesh_2x2)  # shard n_v across rows
    k_tt = shard_heads(k, 2, mesh_2x2)
    v_tt = shard_heads(v, 2, mesh_2x2)
    beta_tt = shard_heads(beta, 2, mesh_2x2)
    g_tt = shard_heads(g, 2, mesh_2x2)

    print(f"q_tt shape per shard: {q_tt.shape}")

    out_tt, state_tt = recurrent_gated_delta_rule_ttnn(q_tt, k_tt, v_tt, beta_tt, g_tt, device=mesh_2x2)

    # Composer for 2D mesh: dims must be a 2-element sequence. Use distinct dummy dim
    # for col axis (col is replicated; composer will stack identical copies on that dim).
    # We'll take the first stride along that axis after gather.
    out_back_full = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.create_mesh_composer(mesh_2x2, ttnn.MeshComposerConfig([2, 0]))
    ).float()
    # After composing: head dim concatenated (rows), batch dim stacked (cols replicated).
    # Slice the col-axis stack to take 1 copy.
    B_full = out_back_full.shape[0]
    expected_B = B
    if B_full > expected_B:
        out_back = out_back_full[:expected_B]
    else:
        out_back = out_back_full
    print(f"gathered shape: full={out_back_full.shape}, sliced={out_back.shape}")

    if out_back.shape != out_full.shape and out_back.dim() == 4 and out_back.shape[1] == n_v:
        out_back = out_back.transpose(1, 2)

    print(f"final compare shapes: tt={out_back.shape}, ref={out_full.shape}")
    pcc = _pcc(out_back, out_full)
    print(f"T={seq_len} mesh={mesh_2x2.shape}: PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
