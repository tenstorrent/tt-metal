# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro/guard for the Quasar paged_fill_cache DRAM write overrun (llama32_1b prefill KV fill).

The prefill fills the paged KV cache with two back-to-back ``ttnn.experimental.paged_fill_cache`` calls — one
for K, one for V (attention_1d.py:877-878) — with input-prep ops (slice/reshard/tilize) in between. On the
Quasar sim the SECOND fill aborted:
    ERROR: UndefinedBehavior: tile_wr_bytes: DRAM write overrun: tile=D0 addr=0xb0b77c80 size=256

ROOT CAUSE (not the earlier cache-hit partial-args issue — program cache is disabled in the e2e, every launch
is a full-args miss; and not get_tile_size — that fix compiled in and the overrun was unchanged): the writer
reads the page_table into a Quasar SCRATCHPAD via NoC and then reads it back with a RISC load. On Quasar DM
``invalidate_l1_cache()`` is a NO-OP (#48552), so nothing discards a stale L2 line that an intervening op left
in that reused scratchpad L1 slot. The FIRST fill reads a fresh page_table; by the SECOND fill (after ~12 min
of intervening ops) the RISC read-back returns a stale line → garbage ``physical_block`` → out-of-range
``physical_tile_id`` → the ``noc.async_write`` targets a DRAM page past the cache buffer → overrun. FIX:
``invalidate_l2_cache_range(page_table_scratch.get_base_address(), page_table_stick_size)`` (and the batch_idx
/ valid_seq_len scratchpad read-backs) before the RISC read, guarded ARCH_QUASAR, in
writer_fill_cache_interleaved.cpp — the same primitive the quasar/pad reader uses for a reused-scratch
NoC-write→RISC-read. WH/BH read the page_table via a DFB (no L2 hazard). (The get_entry_size migration of the
paged_cache kernels is kept as correctness hygiene per the get_tile_size-stale note, but was NOT this bug.)

Shapes mirror the e2e op (nohup): cache [128, 8, 32, 64] bf16, input [1, 8, 512, 64] bf16, page_table [1, 16]
int32. Two fills (K then V) with the V input built AFTER the K fill, so the intervening upload/tilize ops
perturb the descriptor state (as in the model) — the condition under which the stale get_tile_size bit.

Run (Quasar sim):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_SIMULATOR=~/sim/libttsim.so \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_paged_fill_cache.py
"""

import pytest
import torch
from loguru import logger

import ttnn

NUM_BLOCKS = 128
NUM_KV_HEADS = 8
BLOCK_SIZE = 32  # cache dim 2 (one tile tall)
HEAD_DIM = 64  # 2 tiles wide
SEQ = 512  # input seq -> SEQ/BLOCK_SIZE = 16 virtual blocks
NUM_VBLOCKS = SEQ // BLOCK_SIZE  # 16 (matches page_table width)


def _tile_bf16_dram(t_bf16, mesh_device):
    """bf16 TILE DRAM tensor (plain from_torch(TILE); bf16 narrow, so no fp32/wide tilize issue)."""
    return ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )


def _fill_and_readback(mesh_device, fill_torch, page_table_tt, page_table_perm):
    """Run one paged_fill_cache (cache starts zero) and return the readback cache as torch."""
    cache_tt = _tile_bf16_dram(
        torch.zeros(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16), mesh_device
    )
    fill_tt = _tile_bf16_dram(fill_torch, mesh_device)  # [1, NUM_KV_HEADS, SEQ, HEAD_DIM]
    ttnn.experimental.paged_fill_cache(cache_tt, fill_tt, page_table_tt, batch_idx=0)
    ttnn.synchronize_device(mesh_device)
    return ttnn.to_torch(cache_tt).float()


def _reference(fill_torch, page_table_perm):
    """Expected cache after fill: for virtual block v (0..15), physical page_table_perm[v] gets
    fill[0, :, v*32:(v+1)*32, :]; all other blocks stay zero."""
    exp = torch.zeros(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.float32)
    f = fill_torch.float()
    for v in range(NUM_VBLOCKS):
        pb = int(page_table_perm[v])
        exp[pb] = f[0, :, v * BLOCK_SIZE : (v + 1) * BLOCK_SIZE, :]
    return exp


@pytest.mark.timeout(3600)
def test_quasar_paged_fill_cache_k_then_v(mesh_device):
    """Two back-to-back paged_fill_cache (K then V), as prefill does. The 2nd overran on Quasar before the
    get_entry_size fix. Asserts both filled caches match a torch reference (bf16 round-trip tolerance)."""
    torch.manual_seed(0)

    # Identity-ish page table: virtual block v -> physical block v (distinct, in-range < NUM_BLOCKS).
    page_table_perm = torch.arange(NUM_VBLOCKS, dtype=torch.int32)
    page_table_tt = ttnn.from_torch(
        page_table_perm.reshape(1, NUM_VBLOCKS),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )

    k_torch = torch.randn(1, NUM_KV_HEADS, SEQ, HEAD_DIM, dtype=torch.bfloat16)
    v_torch = torch.randn(1, NUM_KV_HEADS, SEQ, HEAD_DIM, dtype=torch.bfloat16)

    # K fill first; then V fill (its input is built AFTER the K fill, so the intervening upload/tilize ops
    # perturb the descriptor — the exact condition that tripped the stale get_tile_size on the 2nd fill).
    logger.info("[fill-cache] K fill")
    k_cache = _fill_and_readback(mesh_device, k_torch, page_table_tt, page_table_perm)
    logger.info("[fill-cache] V fill (the one that overran pre-fix)")
    v_cache = _fill_and_readback(mesh_device, v_torch, page_table_tt, page_table_perm)

    for name, got, src in (("K", k_cache, k_torch), ("V", v_cache, v_torch)):
        exp = _reference(src, page_table_perm)
        max_abs = (got - exp).abs().max().item()
        logger.info(f"[fill-cache] {name} max|got-exp|={max_abs:.4f}")
        assert torch.allclose(
            got, exp, atol=0.05, rtol=0.05
        ), f"{name} cache mismatch after paged_fill_cache (max {max_abs})"
