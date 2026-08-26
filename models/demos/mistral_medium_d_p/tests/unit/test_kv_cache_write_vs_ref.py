# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE: the GQA chunked-KV cache — write, read back, PCC vs a natural-order torch golden.

**This is the test that retires the one genuinely new risk in this bring-up.** Every other GQA model
in the repo lands on exactly ONE KV head per chip (minimax_m3 is 4 heads over TP=4, gpt_oss 8 over
TP=8). Mistral has 8 KV heads at TP=4, so each chip holds **2** — a configuration nothing in the
repo currently exercises, and one that `deepseek_v3_d_p/utils/kv_cache_utils.py::init_kvpe_cache`
cannot express because it hardcodes the head dim to 1.

It is legal: ``update_padded_kv_cache`` only requires ``cache_shape[1] == input_shape[1]`` and
block-cyclic-shards the SEQUENCE on the SP axis, so heads are orthogonal to it. This test proves it
on device, and does so at `1x4` — i.e. on a 4-chip box, before any Galaxy time is spent. (Under
TP=8 the same risk could only have been tested on 8+ chips.)

Layout: K and V caches, each per-chip ``[users*layers, 2, seq_local, 128]``; heads TP-sharded on the
cols, sequence SP-sharded block-cyclic on the rows. Modelled on
``minimax_m3/tests/unit/test_kv_cache_gqa_sp_vs_ref.py``, but driving OUR
``allocate_kv_cache`` / ``write_kv_chunk``.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_kv_cache_write_vs_ref.py -k 1x4
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.mistral_medium_d_p.tt.attention import allocate_kv_cache, write_kv_chunk

from ..test_factory import mesh_setup, parametrize_mesh_with_fabric
from .shapes import HEAD_DIM, N_KV, per_chip


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (2, 4), (8, 4)])
@pytest.mark.parametrize("n_chunks,chunk_local", [(2, 32)], ids=["2chunks"])
def test_kv_cache_write_readback(mesh_device, device_params, n_chunks, chunk_local, reset_seeds):
    """Write GQA K/V chunks into the TP+SP chunked-KV cache, read back, PCC vs natural order."""
    rows, cols = tuple(mesh_device.shape)
    mesh_config, _ = mesh_setup(mesh_device)
    sp, tp, sp_axis, tp_axis = mesh_config.sp, mesh_config.tp, mesh_config.sp_axis, mesh_config.tp_axis
    n_kv_local = per_chip(tp)["n_kv"]
    assert n_kv_local == N_KV // tp

    C = chunk_local
    chunk_global = sp * C
    cache_global = n_chunks * chunk_global
    cache_tokens_per_dev = cache_global // sp

    torch.manual_seed(0)
    sent_k = torch.randn(N_KV, cache_global, HEAD_DIM, dtype=torch.bfloat16)
    sent_v = torch.randn(N_KV, cache_global, HEAD_DIM, dtype=torch.bfloat16)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=cache_global,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=n_kv_local,
    )
    assert kv_cache.k.shape[1] == n_kv_local, f"cache allocated {kv_cache.k.shape[1]} heads/chip, want {n_kv_local}"

    # Chunk input sharding: sequence on the SP rows (dim 2), heads on the TP cols (dim 1).
    in_dims = [None, None]
    in_dims[sp_axis] = 2
    in_dims[tp_axis] = 1

    def to_device(sent, kv_actual):
        """This chunk's global positions in the writer's block-cyclic chip-concat order."""
        positions = rotated_chip_positions(kv_actual, sp, C)
        idx = torch.tensor([positions[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)
        chunk = sent[:, idx, :].reshape(1, N_KV, chunk_global, HEAD_DIM)
        return ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(in_dims)),
        )

    for c in range(n_chunks):
        kv_actual = c * chunk_global
        write_kv_chunk(
            kv_cache,
            to_device(sent_k, kv_actual),
            to_device(sent_v, kv_actual),
            slot_idx=0,
            layer_idx=0,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )
    ttnn.synchronize_device(mesh_device)

    # Read back: concat the sequence over the SP rows (dim 2) and the heads over the TP cols (dim 1).
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1

    def readback(cache):
        return ttnn.to_torch(
            cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=(rows, cols)),
        ).to(
            torch.bfloat16
        )  # -> [1, N_KV, cache_global, HEAD_DIM]

    host_k, host_v = readback(kv_cache.k), readback(kv_cache.v)
    assert host_k.shape[1] == N_KV, f"read back {host_k.shape[1]} heads, expected all {N_KV}"

    # Invert the block-cyclic layout: natural position p -> (chip, local row) -> index on dim 2.
    p = torch.arange(cache_global)
    chip = (p % chunk_global) // C
    local_row = (p // chunk_global) * C + (p % C)
    dim2_idx = chip * cache_tokens_per_dev + local_row

    for h in range(N_KV):
        ok_k, pcc_k = comp_pcc(sent_k[h], host_k[0, h, dim2_idx, :], 0.99)
        ok_v, pcc_v = comp_pcc(sent_v[h], host_v[0, h, dim2_idx, :], 0.99)
        assert ok_k and ok_v, f"head {h} cache mismatch: K={pcc_k} V={pcc_v}"
    logger.info(
        f"GQA chunked-KV cache TP={tp} x SP={sp} ({n_kv_local} KV heads/chip, {n_chunks} chunks, "
        f"{cache_global} tok): all {N_KV} heads PCC>=0.99"
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_write_rejects_head_count_mismatch(mesh_device, device_params, reset_seeds, expect_error):
    """A chunk whose head count disagrees with the cache must fail loud, not corrupt the cache.

    ``update_padded_kv_cache`` enforces ``cache_shape[1] == input_shape[1]`` device-side; we assert
    it host-side first so the message names the cause.
    """
    mesh_config, _ = mesh_setup(mesh_device)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=64 * mesh_config.sp,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=per_chip(mesh_config.tp)["n_kv"],
    )
    wrong = ttnn.from_torch(
        torch.zeros(1, 1, 64, HEAD_DIM),  # 1 head/chip, but the cache wants 2
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with expect_error(AssertionError, "KV heads but the cache was allocated"):
        write_kv_chunk(kv_cache, wrong, wrong, slot_idx=0, layer_idx=0, kv_actual=0, sp_axis=mesh_config.sp_axis)
