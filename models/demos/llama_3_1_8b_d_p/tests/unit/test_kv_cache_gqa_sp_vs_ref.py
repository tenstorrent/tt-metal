# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GQA chunked-KV cache at TP=4 x SP=8: write and read back, PCC vs the torch golden.

Structure follows `minimax_m3/tests/unit/test_kv_cache_gqa_sp_vs_ref.py`, but the load-bearing case
is different and it is the reason this file exists at the head of the KV cache work.

## Two KV heads per chip

The donor test asserts "under TP=4 each chip holds 1 KV head (4 heads / 4 cols), so the per-chip
cache is `[1, 1, seq_local, head_dim]` — exactly `init_kvpe_cache`'s shape". Llama-3.1-8B has **8**
KV heads at TP=4, so each chip holds **2**, and the per-chip cache is `[.., 2, seq_local, 128]`.
Every package on this engine has 1, and their allocation code hardcodes the literal.

`update_padded_kv_cache` should be head-agnostic here — it block-cyclic-shards the SEQUENCE on the
SP axis while TP heads are orthogonal, and its only single-head constraint fires when `tp_axis` is
set, which is not this path. `test_two_kv_heads_per_chip` pins that at the boundary rather than
trusting it, because a wrong per-chip head count raises nothing: it writes real numbers to the
wrong rows.

## Layout under test

K cache and V cache, each `[1, NKV, seq_cache, 128]` globally, heads TP-sharded across the cols
(2/chip) and the sequence SP-sharded block-cyclic across the rows, on the DRAM NdShard substrate.
Write chunks -> read back via `ConcatMesh2dToTensor(dims=(seq, heads))` -> invert the block-cyclic
layout -> PCC against natural-order torch K/V.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE
from models.demos.llama_3_1_8b_d_p.tt.attention.kv_cache import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    allocate_kv_caches,
    cache_capacity,
    write_kv_chunk,
)

from ..test_factory import CHUNK_SIZE, KV_DTYPE, MAX_SEQ_LEN, assert_pcc, comp_pcc, parametrize_target_mesh

NKV, HEAD_DIM = 8, 128  # Llama-3.1-8B: 8 KV heads, head_dim 128


@parametrize_target_mesh()
def test_two_kv_heads_per_chip(mesh_device, device_params, mesh_config):
    """The boundary this model sits on and no donor covers: `n_kv_local == 2`.

    Asserted on the allocation itself, so a regression to the donors' hardcoded 1 fails here rather
    than showing up as a KV PCC crater thirty modules later.
    """
    kv = allocate_kv_caches(
        mesh_device,
        num_layers=2,
        max_seq_len=4096,
        chunk_size=1024,
        sp_axis=mesh_config.sp_axis,
        tp_axis=mesh_config.tp_axis,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        cache_dtype=KV_DTYPE,
    )
    assert kv.n_kv_local == 2, f"expected 2 KV heads per chip at TP={mesh_config.tp}, got {kv.n_kv_local}"
    per_chip = ttnn.get_device_tensors(kv.k)[0].shape
    assert tuple(per_chip)[:2] == (1 * 2, 2), f"per-chip cache shape {tuple(per_chip)} should be [slots, 2, seq, hd]"
    assert tuple(per_chip)[2] == 4096 // mesh_config.sp
    assert tuple(per_chip)[3] == HEAD_DIM


def test_capacity_rounds_up_to_whole_chunks():
    """Host-only: capacity is `max_seq_len` rounded UP to a whole number of chunks.

    The spec's 131072 is 25.6 chunks of 5120. Allocating exactly 131072 leaves the last partial
    chunk's block-cyclic addresses pointing past the end of the buffer — and nothing raises.
    """
    assert MAX_SEQ_LEN % CHUNK_SIZE != 0, "this test is only meaningful when the spec is not chunk-aligned"
    cap = cache_capacity(MAX_SEQ_LEN, CHUNK_SIZE)
    assert cap == 26 * CHUNK_SIZE == 133120
    assert cap >= MAX_SEQ_LEN and cap % CHUNK_SIZE == 0
    assert cap % (NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * 8) == 0, "capacity must stay 32*sp aligned after rounding"
    # Already-aligned inputs must not be inflated by a whole chunk.
    assert cache_capacity(10240, 5120) == 10240


@parametrize_target_mesh()
@pytest.mark.parametrize("n_chunks,chunk_local", [(2, 32)], ids=["2x256"])
def test_kv_cache_gqa_sp(mesh_device, device_params, mesh_config, topology_name, n_chunks, chunk_local, reset_seeds):
    """Write GQA K/V chunks into the TP+SP chunked-KV cache, read back, PCC vs natural-order torch."""
    sp, tp = mesh_config.sp, mesh_config.tp
    sp_axis, tp_axis = mesh_config.sp_axis, mesh_config.tp_axis
    n_kv_local = NKV // tp
    assert n_kv_local == 2

    C = chunk_local
    chunk_global = sp * C
    cache_global = n_chunks * chunk_global
    tokens_per_dev = cache_global // sp

    torch.manual_seed(0)
    sent_k = torch.randn(NKV, cache_global, HEAD_DIM, dtype=REF_DTYPE)
    sent_v = torch.randn(NKV, cache_global, HEAD_DIM, dtype=REF_DTYPE)

    kv = allocate_kv_caches(
        mesh_device,
        num_layers=1,
        max_seq_len=cache_global,
        chunk_size=chunk_global,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        cache_dtype=KV_DTYPE,
    )
    assert kv.capacity == cache_global

    # Chunk input sharding: sequence on the SP rows (dim 2), heads on the TP cols (dim 1).
    in_dims = [None, None]
    in_dims[sp_axis] = 2
    in_dims[tp_axis] = 1

    def to_device_chunk(sent, kv_actual):
        """This chunk's global positions in block-cyclic chip-concat order (mirrors the writer)."""
        positions = rotated_chip_positions(kv_actual, sp, C)
        idx = torch.tensor([positions[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)
        chunk = sent[:, idx, :].reshape(1, NKV, chunk_global, HEAD_DIM)
        return ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=KV_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=in_dims),
        )

    for c in range(n_chunks):
        kv_actual = c * chunk_global
        write_kv_chunk(
            kv,
            to_device_chunk(sent_k, kv_actual),
            to_device_chunk(sent_v, kv_actual),
            slot_idx=0,
            layer_idx=0,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )
    ttnn.synchronize_device(mesh_device)

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1

    def readback(cache):
        return ttnn.to_torch(
            cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=tuple(concat_dims), mesh_shape=tuple(mesh_device.shape)
            ),
        ).to(REF_DTYPE)  # [1, NKV, cache_global, HEAD_DIM]

    host_k, host_v = readback(kv.k), readback(kv.v)

    # Invert the block-cyclic layout: natural position p -> (chip, local row) -> dim-2 cache index.
    p = torch.arange(cache_global)
    chip = (p % chunk_global) // C
    local_row = (p // chunk_global) * C + (p % C)
    dim2_idx = chip * tokens_per_dev + local_row

    worst = 1.0
    for h in range(NKV):
        pcc_k = comp_pcc(sent_k[h], host_k[0, h, dim2_idx, :])
        pcc_v = comp_pcc(sent_v[h], host_v[0, h, dim2_idx, :])
        logger.info(f"head {h} (chip {h // n_kv_local}, local head {h % n_kv_local}): K {pcc_k:.6f} V {pcc_v:.6f}")
        worst = min(worst, pcc_k, pcc_v)
    logger.info(f"GQA TP={tp} x SP={sp} chunked-KV, {n_chunks} chunks, {cache_global} tok: worst PCC {worst:.6f}")

    # Assert per head so a failure names which head (and therefore which chip) went wrong.
    for h in range(NKV):
        assert_pcc(f"kv_cache_gqa_sp.k[head{h}]", sent_k[h], host_k[0, h, dim2_idx, :], topology_name)
        assert_pcc(f"kv_cache_gqa_sp.v[head{h}]", sent_v[h], host_v[0, h, dim2_idx, :], topology_name)


@parametrize_target_mesh()
def test_kv_cache_rejects_wrong_local_head_count(mesh_device, device_params, mesh_config):
    """A chunk carrying the wrong number of local KV heads must be refused at the write seam.

    This is the failure mode the whole file guards: with 1 head where the cache has 2, the op would
    write real values into wrong rows and every later read would be quietly corrupt.
    """
    sp = mesh_config.sp
    kv = allocate_kv_caches(
        mesh_device,
        num_layers=1,
        max_seq_len=sp * 32,
        chunk_size=sp * 32,
        sp_axis=mesh_config.sp_axis,
        tp_axis=mesh_config.tp_axis,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        cache_dtype=KV_DTYPE,
    )
    wrong = ttnn.from_torch(
        torch.zeros(1, 1, sp * 32, HEAD_DIM),  # 1 local head, cache wants 2
        device=mesh_device,
        dtype=KV_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with pytest.raises(AssertionError, match="local KV heads"):
        write_kv_chunk(kv, wrong, wrong, slot_idx=0, layer_idx=0, kv_actual=0, sp_axis=mesh_config.sp_axis)
