# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa block-cyclic remap — MULTI-DEVICE (sp>1) coverage (Blackhole only).

The single-device suite (test_sparse_sdpa.py) can only smoke the block-cyclic path at sp=1, where the remap
is the identity. The real natural->physical PERMUTATION arithmetic only runs when sp>1 (the cache is striped
over >1 SP shard), which needs a real SP mesh — hence this separate module with a `mesh_device` fixture (the
single-device file pins `use_module_device` and can't host one).

Setup mirrors the DeepSeek chunked-prefill producer: the kv cache is stored block-cyclic across `sp` shards
(see models/.../mla/utils.py::blockcyclic_positions, replicated here so the test has no model dependency); the
queries for one chunk are SP-sharded on seq (chunk_local rows per chip). `indices` are NATURAL token positions;
the kernel remaps each to its physical page. The op must therefore match the natural-order golden.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import make_inputs, golden, pcc

K_DIM = 576  # head dim (q/kv width)
V_DIM = 512  # V width / output width (op arg)


def _blockcyclic_positions(sp, chunk_size_global, seq_len_cache):
    """Natural global position held by each physical block-cyclic row r (the inverse the kernel must apply).
    Mirrors models/demos/deepseek_v3_d_p/tt/mla/utils.py::blockcyclic_positions — kept local so this ttnn op
    test does not depend on the model."""
    seq_len_local = seq_len_cache // sp
    chunk_local = chunk_size_global // sp
    c = torch.arange(sp).repeat_interleave(seq_len_local)
    lr = torch.arange(seq_len_local).repeat(sp)
    slab, off = lr // chunk_local, lr % chunk_local
    return slab * chunk_size_global + c * chunk_local + off


def _to_blockcyclic(kv_natural, sp, chunk_size_global):
    """Reorder a natural [T, D] kv cache into block-cyclic slab-major layout: physical row r holds natural
    token p[r]. Returns [1, 1, T, D]."""
    T = kv_natural.shape[0]
    p = _blockcyclic_positions(sp, chunk_size_global, T)
    return kv_natural[p].reshape(1, 1, T, K_DIM)


# sp is DERIVED from mesh axis 0; tp = mesh_size/sp (== 1 here). chunk_local is fixed and T spans >1 slab per
# shard (slabs = (T/sp)/chunk_local: sp=2 -> 4 slabs, sp=4 -> 2 slabs) so the permutation is non-trivial.
@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(2, 1), (4, 1)], indirect=True, ids=["sp2", "sp4"])
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s * 3) % 20, "boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_block_cyclic_sp_multi(mesh_device, nv_fn, nv_id):
    sp = tuple(mesh_device.shape)[0]
    H, T, TOPK, kc = 32, 256, 64, 32
    chunk_local = 32  # per-chip query rows == one chunk's per-shard length (the op's chunk_local cross-check)
    chunk_size_global = sp * chunk_local
    Sq = sp * chunk_local  # global query count this chunk (chunk_local per chip)
    assert (T // sp) % chunk_local == 0 and (T // sp) // chunk_local > 1, "need >1 slab to exercise the remap"

    # Natural-order reference inputs; golden runs the full Sq queries against the natural cache.
    q, kv_nat, indices = make_inputs(H, Sq, T, TOPK, K_DIM, nv_fn, seed=sp)
    ref = golden(q, kv_nat, indices, K_DIM**-0.5, V_DIM)

    # Device: block-cyclic kv REPLICATED; q + natural indices SP-sharded on seq (dim 2) across mesh axis 0.
    kv_bc = _to_blockcyclic(kv_nat[0, 0], sp, chunk_size_global)
    mesh_shape = tuple(mesh_device.shape)
    shard_seq = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=mesh_shape)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_q = ttnn.from_torch(q.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=shard_seq, **common)
    tt_kv = ttnn.from_torch(
        kv_bc.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device), **common
    )
    tt_idx = ttnn.from_torch(indices.to(torch.int32), dtype=ttnn.uint32, mesh_mapper=shard_seq, **common)

    tt_out = ttnn.transformer.sparse_sdpa(
        tt_q,
        tt_kv,
        tt_idx,
        V_DIM,
        scale=K_DIM**-0.5,
        k_chunk_size=kc,
        block_cyclic_sp_axis=0,  # sp read from mesh axis 0
        block_cyclic_chunk_local=chunk_local,
    )
    # Concat the per-chip seq shards (mesh axis 0 -> tensor dim 2) back to the full [1, H, Sq, V_DIM].
    out = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_shape)
    )
    p = pcc(out, ref)
    assert p >= 0.99, f"PCC {p:.5f} (sp={sp}, {nv_id})"
