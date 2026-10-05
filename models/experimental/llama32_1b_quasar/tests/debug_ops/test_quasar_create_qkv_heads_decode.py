# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar nlp_create_qkv_heads_decode INTERLEAVED-factory self-loop.

The llama32_1b decode splits the fused QKV projection into heads with
``ttnn.experimental.nlp_create_qkv_heads_decode``. Its factory is picked by input layout
(device_operation.cpp:16-24):
  - input SHARDED (WIDTH_SHARDED)  -> NLPCreateQKVHeadsDecodeShardedProgramFactory  (Gen2-clean)
  - input INTERLEAVED              -> NLPCreateQKVHeadsDecodeInterleavedProgramFactory

The Interleaved factory's `reader` data-movement kernel self-loops its `reader_scratch` DataflowBuffer (bound
PRODUCER *and* CONSUMER), which Gen2/Quasar rejects:

    program_spec.cpp:1990: DataflowBuffer 'reader_scratch' is self-looped by data-movement kernel 'reader'.
    Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures. Consider using a
    scratchpad or LocalTensorAccessor instead.

The e2e bring-up feeds this op a WIDTH_SHARDED input (from the decode QKV matmul) so the Gen2-clean Sharded
factory is selected. This file isolates BOTH paths:
  - test_...interleaved: INTERLEAVED input -> self-loop FATAL today. PORT TARGET: convert reader_scratch to a
    Scratchpad / LocalTensorAccessor (recipe section 6) so the Interleaved factory works on Gen2.
  - test_...sharded:     WIDTH_SHARDED input -> Sharded factory -> PASSES (the e2e's workaround).

Dims = llama-3.2-1B: num_heads=32, num_kv_heads=8, head_dim=64 -> qkv = 32*64 + 2*(8*64) = 3072.
num_users=32 keeps the input tile-aligned (logical height 32 = 1 tile) and needs a 32-core (8x4) output grid,
so this skips on a smaller device. Inputs built bf16 via RM upload + quasar.tilize (from_torch(TILE) hangs on
the sim). overlap_qk_coregrid=True (matches the e2e non-fused decode; also skips the partial-head shard check).

Run (Quasar sim, FULL 8x4 grid -- do NOT pass TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so MESH_DEVICE=N150 \
        pytest tests/ttnn/unit_tests/operations/test_quasar_create_qkv_heads_decode.py
"""

import pytest
import torch
from loguru import logger

import ttnn

N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
QKV = N_Q_HEADS * HEAD_DIM + 2 * (N_KV_HEADS * HEAD_DIM)  # 3072
NUM_USERS = 32  # tile-aligned input height; needs a 32-core output grid


def _tile_bf16_dram(t_bf16, mesh_device):
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
        logger.info(f"[create-qkv-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _full_grid_crs(mesh_device, num_cores):
    """CoreRangeSet of exactly num_cores cores laid out row-major over the device grid."""
    g = mesh_device.compute_with_storage_grid_size()
    ranges = []
    remaining = num_cores
    for y in range(int(g.y)):
        if remaining <= 0:
            break
        w = min(int(g.x), remaining)
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(w - 1, y)))
        remaining -= w
    return ttnn.CoreRangeSet(ranges)


def _output_memcfg(mesh_device):
    # HEIGHT_SHARDED output on num_users cores (op requires num_cores >= num_users). Shard shape is the padded
    # q head block [num_q_heads_padded(=32), head_dim]; the op recomputes q/k/v specs internally.
    grid = _full_grid_crs(mesh_device, NUM_USERS)
    shard = ttnn.ShardSpec(grid, (N_Q_HEADS, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)


def _skip_if_small(mesh_device):
    g = mesh_device.compute_with_storage_grid_size()
    if int(g.x) * int(g.y) < NUM_USERS:
        pytest.skip(f"needs >= {NUM_USERS} cores for the decode output grid; device has {g.x}x{g.y}")


def _call(xqkv, mesh_device):
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        overlap_qk_coregrid=True,
        memory_config=_output_memcfg(mesh_device),
    )


def _check(q, k, v):
    qt, kt, vt = ttnn.to_torch(q), ttnn.to_torch(k), ttnn.to_torch(v)
    assert tuple(qt.shape) == (1, NUM_USERS, N_Q_HEADS, HEAD_DIM), f"q shape {tuple(qt.shape)}"
    assert tuple(kt.shape) == (1, NUM_USERS, N_KV_HEADS, HEAD_DIM), f"k shape {tuple(kt.shape)}"
    assert tuple(vt.shape) == (1, NUM_USERS, N_KV_HEADS, HEAD_DIM), f"v shape {tuple(vt.shape)}"
    logger.info(f"[create-qkv-repro] out q{tuple(qt.shape)} k{tuple(kt.shape)} v{tuple(vt.shape)}")


def test_create_qkv_heads_decode_interleaved(mesh_device):
    """INTERLEAVED input -> Interleaved factory -> reader_scratch self-loop FATAL on Quasar today. PORT TARGET:
    scratchpad-convert reader_scratch so the Interleaved factory works on Gen2. On WH/BH this passes now."""
    _skip_if_small(mesh_device)
    torch.manual_seed(0)
    x = torch.randn(1, 1, NUM_USERS, QKV, dtype=torch.bfloat16)
    xqkv = _tile_bf16_dram(x, mesh_device)  # DRAM interleaved TILE
    q, k, v = _call(xqkv, mesh_device)
    ttnn.synchronize_device(mesh_device)
    _check(q, k, v)


def test_create_qkv_heads_decode_sharded(mesh_device):
    """WIDTH_SHARDED input -> Sharded factory (Gen2-clean) -> PASSES. This is the e2e's workaround: the decode
    QKV matmul emits a width-sharded output so this factory is selected instead of the self-looping one."""
    _skip_if_small(mesh_device)
    torch.manual_seed(0)
    x = torch.randn(1, 1, NUM_USERS, QKV, dtype=torch.bfloat16)
    xqkv = _tile_bf16_dram(x, mesh_device)

    # WIDTH_SHARDED: shard height = physical rows (NUM_USERS), width = QKV / num_cores, ROW_MAJOR.
    g = mesh_device.compute_with_storage_grid_size()
    ncols = int(g.x)
    assert QKV % (ncols * 32) == 0, f"QKV={QKV} must split into 32-aligned shards over {ncols} cols"
    shard = ttnn.ShardSpec(
        _full_grid_crs(mesh_device, ncols), (NUM_USERS, QKV // ncols), ttnn.ShardOrientation.ROW_MAJOR
    )
    w_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)
    qi2s = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    xqkv_s = (qi2s or ttnn.interleaved_to_sharded)(xqkv, w_memcfg)

    q, k, v = _call(xqkv_s, mesh_device)
    ttnn.synchronize_device(mesh_device)
    _check(q, k, v)
