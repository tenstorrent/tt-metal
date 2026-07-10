# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sp>1 block-cyclic remap coverage for sparse_sdpa_msa (multi-device).

The post-commit file only covers sp=1, where the invP block remap is the identity. This runs the REAL
permutation at sp=2 and sp=4 via the `mesh_device` fixture (auto-skips when the devices aren't present, so
it's safe off-multi-device): K/V are laid out block-cyclic (shard-major) across the SP axis exactly as the
AllGather'd chunked-prefill cache is, block-ids stay NATURAL, and the op must remap each logical block to
its physical block and reproduce the natural-order golden. Inputs are replicated across the mesh (sp is read
from the mesh shape), so every device computes the same result; each is checked against the golden.
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
    shard-major (shard, chunk, local), the layout AllGather produces from the per-shard cache. Inverse of
    the model's old _blockcyclic_to_natural."""
    H, T, d = t.shape[1], t.shape[2], t.shape[3]
    t = t.reshape(1, H, n_chunks, sp, chunk_local, d)
    t = t.permute(0, 1, 3, 2, 4, 5)  # (chunk, shard) -> (shard, chunk)
    return t.reshape(1, H, T, d).contiguous()


@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)  # SP along cols; fixture skips if absent
@pytest.mark.parametrize("n_chunks", [8])  # sp*n_chunks = nblk >= topk(16): 2*8=16, 4*8=32
def test_msa_native_block_cyclic_sp_gt1_pcc(mesh_device, n_chunks):
    rows, cols = tuple(mesh_device.shape)
    sp_axis, sp = 1, cols
    if sp < 2:
        pytest.skip(f"needs sp>1 (mesh shape {(rows, cols)})")

    H, n_kv, S, d = 32, 1, BLK_KV, 128
    chunk_local = S  # tp=1 (pure-SP mesh) -> guard requires chunk_local == q_isl (= S)
    T = sp * n_chunks * chunk_local
    nblk = T // BLK_KV
    topk = 16  # multiple of 16 (indices row 64B-aligned) and <= nblk
    assert nblk >= topk and (T // sp) // chunk_local > 1, f"non-trivial permutation needs nblk>=topk: nblk={nblk}"
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=topk, d=d, causal=False, seed=T)

    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)  # natural-order, layout-agnostic
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

    def dev_tile(x, dt):
        return ttnn.from_torch(
            x.to(torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=repl,
        )

    out = ttnn.transformer.sparse_sdpa_msa(
        dev_rm(q.to(torch.float32), ttnn.bfloat16),
        dev_tile(k_bc, ttnn.bfloat16),
        dev_tile(v_bc, ttnn.bfloat16),
        dev_rm(indices.to(torch.int32), ttnn.uint32),
        scale=d**-0.5,
        block_size=BLK_KV,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_local,
    )

    # Inputs replicated -> every device must reproduce the natural golden (proves the sp>1 invP permutation).
    for i, shard in enumerate(ttnn.get_device_tensors(out)):
        o = ttnn.to_torch(shard)[:, :H]
        p = pcc(o, gold)
        assert p >= DEVICE_PCC, f"sp={sp} block-cyclic PCC {p:.5f} on device {i} (n_chunks={n_chunks}, T={T})"
