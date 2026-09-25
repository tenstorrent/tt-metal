# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar flash-decode SDPA op, seen in llama32_1b decode.

The llama32_1b decode attention ends in a flash-decode SDPA:

    ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        q_heads, keys, values, page_table_tensor=..., cur_pos_tensor=..., scale=..., ...)

The e2e run reaches this op (after QKV matmul -> create_qkv_heads -> RoPE -> paged_update_cache)
and appears to stall/fault inside it on the Quasar simulator. This file exercises JUST that op, with
the model's exact decode shapes and program config, so it can be run under watcher in isolation.

Shapes (llama-3.2-1B, matches the captured graph case):
    q        : [1, 1, n_q_heads=32, head_dim=64]  bf16 TILE, HEIGHT_SHARDED L1 (1 core/batch)
    keys/vals: [max_num_blocks=128, n_kv_heads=8, block_size=32, head_dim=64]  bf16 TILE, DRAM interleaved  (paged)
               [batch, n_kv_heads=8, max_seq_len, head_dim=64]                 bf16 TILE, DRAM interleaved  (non-paged)
    page_table: [batch, max_num_blocks]  int32 ROW_MAJOR DRAM   (paged only)
    cur_pos   : [batch]                  int32 ROW_MAJOR DRAM
    program_config: SDPAProgramConfig(compute_with_storage_grid_size=[8,4], exp_approx_mode=True,
                                      q_chunk_size=0, k_chunk_size=0)

Inputs are built via a bf16 row-major upload + quasar.tilize (NOT from_torch(TILE), which hangs on the
Quasar sim). bf16 throughout (Quasar dropped bf8_b -> MX formats).

NOT marked xfail -- craq-sim tooling drives off a real FAIL/hang. On WH/BH this passes.

Run (Quasar sim, with watcher):
    MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_WATCHER=1 \
        pytest tests/ttnn/unit_tests/operations/test_quasar_sdpa_decode.py -k paged
"""

import pytest
import torch
from loguru import logger

import ttnn

# llama-3.2-1B decode attention dims
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 32  # paged KV cache page size (tile height)
# Keep the KV cache small: craq-sim tilizes every tile in software, so a big cache dominates runtime
# (128 blocks -> 4 MB / 2048 tiles per cache -> minutes just to upload). cur_pos=64 needs only ~3 pages;
# 8 blocks (256 positions) reproduces the SDPA geometry (kv_heads/head_dim/block_size/grid) ~16x faster.
MAX_NUM_BLOCKS = 8
GRID_X, GRID_Y = 8, 4  # model's decode SDPA grid
SCALE = HEAD_DIM**-0.5


def _tile_bf16_dram(t_bf16, mesh_device):
    """bf16 TILE, DRAM-interleaved without from_torch(TILE) (hangs on the Quasar sim): upload row-major,
    then tilize via the Gen2-native quasar op where available."""
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        return ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    except (AttributeError, RuntimeError) as e:
        logger.info(f"[sdpa-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _int32_rm_dram(t_int, mesh_device):
    return ttnn.from_torch(
        t_int,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )


def _q_height_sharded(t_bf16, mesh_device, batch):
    """q -> bf16 TILE, HEIGHT_SHARDED on L1: one core per batch (grid [batch,1]), shard [32, n_q_heads*... ]."""
    qt = _tile_bf16_dram(t_bf16, mesh_device)
    # q logical shape [1, batch, n_q_heads, head_dim] flattens to [batch, n_q_heads*head_dim] worth of a
    # 32-row tile per batch; the captured case shards [32, 64] on a single core for batch=1.
    core_rs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, max(batch - 1, 0)))})
    shard_spec = ttnn.ShardSpec(core_rs, (BLOCK_SIZE, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR)
    memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    qi2s = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    return (qi2s or ttnn.interleaved_to_sharded)(qt, memcfg)


def _prog_cfg():
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(GRID_X, GRID_Y),
        exp_approx_mode=True,
        q_chunk_size=0,
        k_chunk_size=0,
    )


def _compute_cfg():
    # fp32_dest_acc_en=False on Quasar (bf16->Tf32 unpack gap); HiFi2 matches the model.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _skip_if_small(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < GRID_X or grid.y < GRID_Y:
        pytest.skip(f"needs an {GRID_X}x{GRID_Y} grid; device has {grid.x}x{grid.y}")


def test_paged_sdpa_decode(mesh_device):
    """Paged flash-decode SDPA (the exact op the llama decode path stalls on). FAILS/hangs on Quasar."""
    _skip_if_small(mesh_device)
    batch = 1  # page_table.shape[0]; factory derives B from here (captured case is batch=1)
    cur_pos = 64  # current KV position

    torch.manual_seed(0)
    q = torch.randn(1, batch, N_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    keys = torch.randn(MAX_NUM_BLOCKS, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    values = torch.randn(MAX_NUM_BLOCKS, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    # one block per (batch); page_table[b, i] = block id for logical block i of batch b
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).reshape(1, MAX_NUM_BLOCKS).repeat(batch, 1)
    cur_pos_t = torch.full((batch,), cur_pos, dtype=torch.int32)

    q_t = _q_height_sharded(q, mesh_device, batch)
    k_t = _tile_bf16_dram(keys, mesh_device)
    v_t = _tile_bf16_dram(values, mesh_device)
    pt_t = _int32_rm_dram(page_table, mesh_device)
    cp_t = _int32_rm_dram(cur_pos_t, mesh_device)

    logger.info(
        f"[sdpa-repro] paged decode: q[1,{batch},{N_Q_HEADS},{HEAD_DIM}] kv[{MAX_NUM_BLOCKS},{N_KV_HEADS},{BLOCK_SIZE},{HEAD_DIM}] grid {GRID_X}x{GRID_Y}"
    )
    out = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        q_t,
        k_t,
        v_t,
        page_table_tensor=pt_t,
        cur_pos_tensor=cp_t,
        scale=SCALE,
        program_config=_prog_cfg(),
        compute_kernel_config=_compute_cfg(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)
    logger.info(f"[sdpa-repro] paged decode out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()}")
    assert torch.isfinite(o).all(), "SDPA decode produced non-finite output"


def test_non_paged_sdpa_decode(mesh_device):
    """Non-paged flash-decode SDPA (simpler: full [batch,kv_heads,seq,head_dim] cache, no page table)."""
    _skip_if_small(mesh_device)
    batch = 1
    max_seq = MAX_NUM_BLOCKS * BLOCK_SIZE  # 4096
    cur_pos = 64

    torch.manual_seed(0)
    q = torch.randn(1, batch, N_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    keys = torch.randn(batch, N_KV_HEADS, max_seq, HEAD_DIM, dtype=torch.bfloat16)
    values = torch.randn(batch, N_KV_HEADS, max_seq, HEAD_DIM, dtype=torch.bfloat16)
    cur_pos_t = torch.full((batch,), cur_pos, dtype=torch.int32)

    q_t = _q_height_sharded(q, mesh_device, batch)
    k_t = _tile_bf16_dram(keys, mesh_device)
    v_t = _tile_bf16_dram(values, mesh_device)
    cp_t = _int32_rm_dram(cur_pos_t, mesh_device)

    logger.info(
        f"[sdpa-repro] non-paged decode: q[1,{batch},{N_Q_HEADS},{HEAD_DIM}] kv[{batch},{N_KV_HEADS},{max_seq},{HEAD_DIM}] grid {GRID_X}x{GRID_Y}"
    )
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        q_t,
        k_t,
        v_t,
        cur_pos_tensor=cp_t,
        scale=SCALE,
        program_config=_prog_cfg(),
        compute_kernel_config=_compute_cfg(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)
    logger.info(f"[sdpa-repro] non-paged decode out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()}")
    assert torch.isfinite(o).all(), "SDPA decode produced non-finite output"
