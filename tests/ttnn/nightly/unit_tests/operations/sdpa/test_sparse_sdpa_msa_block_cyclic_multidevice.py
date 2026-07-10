# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sp>1 block-cyclic remap coverage for sparse_sdpa_msa (multi-device).

The post-commit file only covers sp=1, where the invP block remap is the identity. This runs the REAL
permutation at sp=2 and sp=4 via the `mesh_device` fixture (auto-skips when the devices aren't present).
Inputs are replicated across the mesh; K/V are laid out block-cyclic (shard-major, as the AllGather'd
chunked-prefill cache is), block-ids stay NATURAL. The check is remap-transparency: the block-cyclic op
must match the PLAIN op (natural-order K/V, no remap) run with identical inputs and the same per-device
chunk_start. Under causal this is the key case — the diagonal-block mask must stay keyed on the logical
block id, not the remapped physical one; if the remap leaked into the mask, bc would diverge from plain.
Op-vs-golden correctness is covered single-device (post-commit file); here plain is additionally checked
against the layout-agnostic golden in the non-causal case.
"""

import pytest
import torch

import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import (
    BLK_KV,
    make_msa_inputs,
    pcc,
    sparse_attention_ref_msa,
)

DEVICE_PCC = 0.99


def _natural_to_block_cyclic(t, sp, n_chunks, chunk_local):
    """Natural [1,H,T,d] (T = n_chunks*sp*chunk_local, order (chunk, shard, local)) -> block-cyclic
    shard-major (shard, chunk, local), the layout AllGather produces from the per-shard cache."""
    H, T, d = t.shape[1], t.shape[2], t.shape[3]
    t = t.reshape(1, H, n_chunks, sp, chunk_local, d)
    t = t.permute(0, 1, 3, 2, 4, 5)  # (chunk, shard) -> (shard, chunk)
    return t.reshape(1, H, T, d).contiguous()


@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)  # SP along cols; fixture skips if absent
@pytest.mark.parametrize("n_chunks", [8])
@pytest.mark.parametrize("causal", [False, True])  # True: diagonal-block mask must stay on the logical id
def test_msa_native_block_cyclic_sp_gt1_matches_plain(mesh_device, n_chunks, causal):
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp = 1, cols
    if sp < 2:
        pytest.skip(f"needs sp>1 (mesh shape {(rows, cols)})")

    H, n_kv, S, d = 32, 1, 2 * BLK_KV, 128  # S = 2 blocks -> chunk_local spans >1 block (non-trivial invP divide)
    chunk_local = S  # tp=1 (pure-SP mesh) -> guard requires chunk_local == q_isl (= S)
    T = sp * n_chunks * chunk_local
    nblk = T // BLK_KV
    topk = 16  # multiple of 16 (indices row 64B-aligned) and <= nblk
    assert nblk >= topk and chunk_local // BLK_KV > 1 and n_chunks > 1, f"degenerate remap params: nblk={nblk}"
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=topk, d=d, causal=causal, seed=T)
    k_bc = _natural_to_block_cyclic(k, sp, n_chunks, chunk_local)
    v_bc = _natural_to_block_cyclic(v, sp, n_chunks, chunk_local)

    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev_rm(x, dt):
        return ttnn.from_torch(
            x,
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=repl,
        )

    def dev_tile(x):
        return ttnn.from_torch(
            x.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=repl,
        )

    # Both runs get the same per-device chunk_start (compute_chunk_start_local, from the mesh coord); the only
    # difference is K/V layout + the remap, so equal outputs prove the remap is correctness-transparent.
    def run_op(k_in, v_in, bc):
        kw = dict(scale=d**-0.5, block_size=BLK_KV, chunk_start_idx=0 if causal else None)
        if bc:
            kw.update(block_cyclic_sp_axis=sp_axis, block_cyclic_chunk_local=chunk_local)
        out = ttnn.transformer.sparse_sdpa_msa(
            dev_rm(q.to(torch.float32), ttnn.bfloat16),
            dev_tile(k_in),
            dev_tile(v_in),
            dev_rm(indices.to(torch.int32), ttnn.uint32),
            **kw,
        )
        return [ttnn.to_torch(s)[:, :H] for s in ttnn.get_device_tensors(out)]

    plain = run_op(k, v, bc=False)
    blockc = run_op(k_bc, v_bc, bc=True)
    for i, (p_out, b_out) in enumerate(zip(plain, blockc)):
        p = pcc(b_out, p_out)
        assert p >= DEVICE_PCC, f"sp={sp} causal={causal}: block-cyclic != plain on dev {i} (pcc={p:.5f}, T={T})"

    if not causal:  # non-causal is device-uniform -> also anchor plain to the layout-agnostic golden (correctness)
        gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
        p = pcc(plain[0], gold)
        assert p >= DEVICE_PCC, f"sp={sp}: plain op vs golden pcc={p:.5f}"
