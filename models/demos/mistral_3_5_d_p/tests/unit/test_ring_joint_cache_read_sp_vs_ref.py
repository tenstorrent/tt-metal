# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The same ring op reading K/V OUT OF the block-cyclic KV cache — short Q against a longer
accumulated prefix. Pattern: ``minimax_m3/tests/unit/test_ring_joint_cache_read_sp_vs_ref.py``.

This is the mechanism chunked prefill depends on, and it is where the two things validated
separately meet: the cache write (``test_kv_cache_write_vs_ref``) and the ring op
(``test_ring_joint_sp_vs_ref``). Chunks 0..N-2 are written into the cache, then
``dense_sp_attention`` runs for ONLY the last chunk's queries with ``kv_actual`` = the prefix before
it and ``logical_n`` = the full valid prefix, so the op must read the accumulated prefix back out of
the block-cyclic SP cache and softmax over it causally.

Chunked mode requires Q shorter than the cached K/V, which is why the test writes the whole prefix
and attends with the last chunk only — the same shape every non-first chunk of a real prefill has.

Q is presented in block-cyclic order within its chunk (as the production runtime delivers it), and
the output is un-rotated back to natural order before the comparison, so a block-cyclic mistake
cannot hide behind a permutation on both sides.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import allocate_kv_cache
from models.demos.mistral_3_5_d_p.tt.attention.config import ProgramConfig
from models.demos.mistral_3_5_d_p.tt.attention.dense_sp import dense_sp_attention

from ..test_factory import build_mesh_and_ccl, parametrize_mesh, sp_tp_shard_mapper
from .test_ring_joint_sp_vs_ref import torch_gqa_causal

NQ, NKV, HEAD_DIM = C.NUM_ATTENTION_HEADS, C.NUM_KEY_VALUE_HEADS, C.HEAD_DIM


def blockcyclic_chunk_index(kv_actual, sp, chunk_local):
    """Global token positions in the order the per-chip block-cyclic chunk carries them.

    ``rotated_chip_positions`` mirrors the writer kernel exactly: entry ``[c][r]`` is the global
    position chip ``c`` holds at its ``r``-th rotated row. Flattening chip-major gives the row order
    of a chunk tensor that is then SP-sharded contiguously.
    """
    positions = rotated_chip_positions(kv_actual, sp, chunk_local)
    return torch.tensor([positions[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.long)


@parametrize_mesh()
@pytest.mark.parametrize(
    "n_chunks, chunk_local",
    [(2, 32), (2, 1280)],  # 2x128 (quick) and 2x5120 — the spec's real chunk_size at sp=4
    ids=["2x128", "2x5120"],
)
def test_ring_joint_cache_read_sp(mesh_device, device_params, n_chunks, chunk_local, reset_seeds):
    """The last chunk's Q attends the full cached prefix, at SP=4 x TP=8, vs the torch golden."""
    rows, cols = tuple(mesh_device.shape)
    sp, n_kv = rows, cols
    assert (rows, cols) == SPEC.mesh_shape, f"expected the spec's mesh {SPEC.mesh_shape}, got {(rows, cols)}"
    assert n_kv == NKV, "this layout maps KV head c -> TP column c"

    chunk_global = sp * chunk_local
    capacity = n_chunks * chunk_global
    kv_actual_last = (n_chunks - 1) * chunk_global  # prefix length before the last chunk

    q = torch.randn(1, NQ, capacity, HEAD_DIM) * 0.1
    k = torch.randn(1, NKV, capacity, HEAD_DIM) * 0.1
    v = torch.randn(1, NKV, capacity, HEAD_DIM) * 0.1
    ref = torch_gqa_causal(q.float(), k.float(), v.float())[:, :, kv_actual_last:, :]

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    program_config = ProgramConfig()
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=capacity,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
    )
    mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2, head_dim=1)

    def make_chunk(src, kv_actual, dtype):
        """[NKV, capacity, HD] -> device [1, NKV, chunk_global, HD] in block-cyclic row order."""
        idx = blockcyclic_chunk_index(kv_actual, sp, chunk_local)
        return ttnn.from_torch(
            src[:, idx, :].reshape(1, NKV, chunk_global, HEAD_DIM),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    # Write every chunk EXCEPT the last; dense_sp_attention writes that one itself (write_chunk=True),
    # which is also how the runtime's compile path drives it.
    for c in range(n_chunks - 1):
        kv_actual = c * chunk_global
        for cache, src in ((kv_cache.k, k[0]), (kv_cache.v, v[0])):
            chunk = make_chunk(src, kv_actual, SPEC.kv_cache_dtype)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                chunk,
                slot_idx=0,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=kv_actual,
                cluster_axis=SPEC.sp_axis,
            )
            chunk.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    last_idx = blockcyclic_chunk_index(kv_actual_last, sp, chunk_local)
    tt_q = ttnn.from_torch(
        q[:, :, last_idx, :],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    out = dense_sp_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        make_chunk(k[0], kv_actual_last, SPEC.kv_cache_dtype),
        make_chunk(v[0], kv_actual_last, SPEC.kv_cache_dtype),
        kv_actual=kv_actual_last,
        logical_n=capacity,
        n_kv=NKV,
        cache_global=capacity,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=program_config.get_ring_sdpa_config(mesh_device),
        compute_kernel_config=program_config.get_ring_compute_kernel_config(mesh_device),
        scale=HEAD_DIM**-0.5,
        cluster_axis=mesh_config.sp_axis,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
        write_chunk=True,
    )

    # out per chip [1, NQ/tp, chunk_local, HD], block-cyclic over the last chunk. Gather heads (cols)
    # and sequence (rows), then invert the block-cyclic order within the chunk.
    shards = ttnn.get_device_tensors(out)
    per_row = [
        torch.cat([ttnn.to_torch(shards[r * cols + c]).float() for c in range(cols)], dim=1) for r in range(rows)
    ]
    out_bc = torch.cat(per_row, dim=2)  # [1, NQ, chunk_global, HD], block-cyclic
    local_pos = last_idx - kv_actual_last
    inverse = torch.empty(chunk_global, dtype=torch.long)
    inverse[local_pos] = torch.arange(chunk_global)
    full = out_bc[:, :, inverse, :]

    passing, pcc = comp_pcc(ref, full, SPEC.pcc)
    logger.info(f"ring_joint CACHE-READ SP={sp} x TP={cols} chunk_local={chunk_local}: pcc={pcc}")
    assert passing, f"cache-read PCC fail (chunk_local={chunk_local}): {pcc}"
