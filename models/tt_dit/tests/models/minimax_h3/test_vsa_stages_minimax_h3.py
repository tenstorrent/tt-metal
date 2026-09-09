# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""R3 checks: the device coarse stage against the torch oracle, single device (sp = 1).

Pooled values, scores, O_c, and index sets must match `vsa_oracle` (index sets compared as sets
per row; near-ties across the bf16/fp32 boundary are tolerated when the scores are within bf16
rounding of the k-th score). Also runs coarse + vsa_sdpa end-to-end against the full VSA oracle.
"""

import pytest
import torch

import ttnn

from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSACoarseStage
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry
from ....utils.check import assert_quality
from .vsa_oracle import VSA_INDEX_SENTINEL, coarse_output, coarse_scores, select_index_rows, vsa_attention

_TINY64 = ((70, 5, 130), (9, 10, 13))
HEADS = 2
DIM = 128
SPARSITY = 0.75


def _tiled_inputs(geometry, seed=0):
    torch.manual_seed(seed)
    q, k, v = (torch.randn(geometry.seq_len, HEADS, DIM) for _ in range(3))
    tile = lambda x: geometry.pack_rows(x).permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H, S_pad, D]
    return (tile(q), tile(k), tile(v))


def _upload(mesh_device, x):
    return ttnn.from_torch(
        x.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(mesh_device, x):
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("placement", ["identity", "striped", "interleaved"])
@pytest.mark.parametrize("selection", ["assembled", "raw", "raw_padded"])
def test_coarse_stage_vs_oracle(mesh_device, placement, selection, reset_seeds):
    """``assembled``: host-assembled index rows. ``raw``: top-k rows as vsa_sdpa consumes them
    (exempt ids + dense-row mask applied here the way the kernel does). ``raw_padded``: the same with
    padded pooling (tile-aligned pooled gathers; ids in the padded per-shard numbering)."""
    if selection == "assembled" and placement != "identity":
        pass  # keep the full placement matrix on the assembled path too
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1, placement=placement)
    tq, tk, tv = _tiled_inputs(geometry)

    stage = MiniMaxH3VSACoarseStage(
        geometry,
        sparsity=SPARSITY,
        padded_pooling=(selection == "raw_padded"),
        head_dim=DIM,
        mesh_device=mesh_device,
        sp_axis=1,
        ccl_manager=None,
    )
    tt_q, tt_k, tt_v = (_upload(mesh_device, t) for t in (tq, tk, tv))

    # (a) pooled values
    pooled_tt = _from_device(mesh_device, stage.pool(tt_k, scaled=False)).float()
    pooled_tt = pooled_tt[:, :, : geometry.tiles_per_shard]  # padded pooling pads the tile axis
    from .vsa_oracle import pool_tiles

    pooled_ref = pool_tiles(tk, geometry)
    assert_quality(pooled_ref, pooled_tt, pcc=0.999)

    # (c+e) O_c and index rows
    raw = selection != "assembled"
    tt_oc, tt_idx = stage(tt_q, tt_k, tt_v, raw_selection=raw)
    oc_tt = _from_device(mesh_device, tt_oc).float()
    idx_tt = _from_device(mesh_device, tt_idx).to(torch.int64)
    if raw:
        # decode the rows exactly as the kernel does: first k entries, padded-numbering remap, exempt
        # ids added, dense rows (mask) take every real block
        shift, real_per_shard = stage.coarse_slots_shift, geometry.tiles_per_shard
        pad = (1 << shift) - real_per_shard if shift else 0
        mask = _from_device(mesh_device, stage.dense_row_mask_tensor()).to(torch.int64).reshape(-1)
        real_ids = set(torch.nonzero(geometry.valid_counts > 0).reshape(-1).tolist())
        decoded = torch.full((1, HEADS, geometry.tiles_per_shard, geometry.n_tiles + 1), VSA_INDEX_SENTINEL)
        for h in range(HEADS):
            for row in range(geometry.tiles_per_shard):
                if (int(mask[row // 32]) >> (row % 32)) & 1:
                    ids = sorted(real_ids)
                else:
                    ids = [int(b) - ((int(b) >> shift) * pad if shift else 0) for b in idx_tt[0, h, row, : stage.k]]
                    ids = sorted(set(ids) | set(stage.exempt_ids))
                decoded[0, h, row, : len(ids)] = torch.tensor(ids)
        idx_tt = decoded

    scores_ref = coarse_scores(tq, tk, geometry)
    oc_ref = coarse_output(scores_ref, tv, geometry, torch.float32)
    valid_rows = geometry.gather_index >= 0
    assert_quality(oc_ref[:, :, valid_rows], oc_tt[:, :, valid_rows], pcc=0.99)

    # index sets per row (sets; near-ties across the bf16/fp32 boundary tolerated)
    _, ref_sets = select_index_rows(scores_ref, geometry, SPARSITY)
    k_sel = stage.k
    n_rows = geometry.tiles_per_shard
    mismatched = 0
    for h in range(HEADS):
        for row in range(n_rows):
            got_row = idx_tt[0, h, row]
            got = set(got_row[got_row != VSA_INDEX_SENTINEL].tolist())
            ref = ref_sets[h][row]
            if got == ref:
                continue
            if bool(geometry.is_exempt[row]) or int(geometry.valid_counts[row]) == 0:
                # dense rows must match exactly; pad-q rows are don't-cares
                assert got == ref or int(geometry.valid_counts[row]) == 0, (h, row)
                continue
            # candidate row: mismatched members must be within bf16 rounding of the k-th score
            row_scores = scores_ref[0, h, row]
            cand = torch.nonzero(geometry.is_candidate, as_tuple=False).reshape(-1)
            kth = row_scores[cand].topk(k_sel).values[-1]
            diff = got.symmetric_difference(ref)
            assert all(abs(float(row_scores[i]) - float(kth)) < 0.05 for i in diff), (h, row, diff)
            mismatched += 1
    assert mismatched <= HEADS * n_rows * 0.1, f"{mismatched} near-tie rows out of {HEADS * n_rows}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("m", [1, 4])
def test_coarse_plus_fine_vs_full_oracle(mesh_device, m, reset_seeds):
    """Coarse stage + vsa_sdpa end-to-end (with gate) against the full VSA oracle, sp = 1."""
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
    tq, tk, tv = _tiled_inputs(geometry, seed=1)
    torch.manual_seed(2)
    gate = torch.randn(1, HEADS, geometry.padded_len, DIM) * 0.5

    stage = MiniMaxH3VSACoarseStage(
        geometry, sparsity=SPARSITY, head_dim=DIM, mesh_device=mesh_device, sp_axis=1, ccl_manager=None
    )
    tt_q, tt_k, tt_v, tt_gate = (_upload(mesh_device, t) for t in (tq, tk, tv, gate))

    tt_oc, tt_idx = stage(tt_q, tt_k, tt_v)
    tt_counts = stage.block_counts_tensor()
    tt_fine = ttnn.transformer.vsa_sdpa(tt_q, tt_k, tt_v, tt_idx, tt_counts, k_chunk_blocks=m)
    tt_out = ttnn.add(tt_fine, ttnn.multiply(tt_gate, tt_oc))
    out = _from_device(mesh_device, tt_out).float()

    # oracle on the device's own index rows would hide selection bugs; use the oracle end to end
    ref = vsa_attention(tq, tk, tv, geometry, SPARSITY, gate_tiled=gate)

    valid = geometry.gather_index >= 0
    assert_quality(ref[:, :, valid], out[:, :, valid], pcc=0.99)
