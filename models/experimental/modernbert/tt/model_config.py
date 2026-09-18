# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Precision policy and matmul program configs for ModernBERT.

Numbers in this file are b8s256 unless stated. See the README for the full
performance and accuracy tables.

Formats:
    matmul weights          bfloat8_b   -8.9% at b1s256, costs 3.4e-03 MLM PCC
    embedding + norm weights  bfloat16  bfloat8_b cannot represent either
    activations               bfloat16  see ACTIVATIONS_DTYPE

ModernBERT develops channel-localised activation outliers from layer 16 onward
(max|x| ~205 -> ~34000 against a median channel max near 34), so the accumulation
format matters more than the storage format and fp32 accumulation stays on.

Seq 512 is the binding accuracy shape; see the README for the current margin.
"""

from typing import NamedTuple

import ttnn

# bfloat8_b is a tiled block format, so it cannot hold the row-major table
# ttnn.embedding requires, and a 1-D norm weight would share its exponent with 31
# rows of tile padding.
WEIGHTS_DTYPE = ttnn.bfloat16
LINEAR_WEIGHTS_DTYPE = ttnn.bfloat8_b

# bfloat8_b activations are 15% faster but break the MLM head: logit PCC 0.99535
# -> 0.93988 while hidden-state PCC still looks fine. test_ttnn_mlm is the gate.
ACTIVATIONS_DTYPE = ttnn.bfloat16


def compute_kernel_config():
    """MathFidelity.LoFi hangs the device (ETH heartbeat timeout, needs tt-smi -r 0).
    HiFi2 is faster but takes MLM top-1 to 6/8. fp32_dest_acc_en halves the dest
    register file, which caps out_subblock_h at 4 below.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# tanh rather than HF's exact erf: the erf is 11% of the forward pass, and the
# approximation costs 1.66e-03 of MLM PCC with top-1 unchanged. -9.0% at b8s256.
_GELU_APPROX = True

# The attention chain is L1-interleaved, not sharded: SDPA asserts
# `!input_tensor->is_sharded()`, so L1 interleaved is legal and sharded is not.
# Worth -21.7% at b8s256, PCC unchanged. Masks must stay in DRAM
# (sdpa_device_operation.cpp:156).
_L1_ATTENTION = True


def attention_interleaved():
    """Interleaved memory config for the attention chain. See _L1_ATTENTION."""
    return ttnn.L1_MEMORY_CONFIG if _L1_ATTENTION else ttnn.DRAM_MEMORY_CONFIG


def _gelu_activation(fuse_gelu):
    if not fuse_gelu:
        return None
    return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, _GELU_APPROX)


def select_down_projection_grid(grid_x, grid_y):
    """Core grid for the projections that output hidden_size, or None.

    8x8 helps the projections that narrow to hidden_size (attn Wo 25.0 -> 23.7 us,
    mlp Wo 41.9 -> 30.8) and hurts the one that widens (Wqkv 73.8 -> 85.1), which
    is why Wqkv uses a program config instead. Only 8x8 was tuned, so smaller grids
    return None and let ttnn choose.
    """
    if grid_x < 8 or grid_y < 8:
        return None
    return ttnn.CoreGrid(y=8, x=8)


def down_projection_core_grid(device):
    """Core grid for attn Wo and mlp Wo on this device, or None to let ttnn choose."""
    grid = device.compute_with_storage_grid_size()
    return select_down_projection_grid(grid.x, grid.y)


# in0_block_w for Wqkv, keyed by per_core_M (= rows/256, so this must cover every
# batch, not every seq_len - a miss drops the largest matmul in the model onto
# ttnn's choice at 36 of 64 cores). 8 everywhere: 4 is marginally faster at
# per_core_M=1 but costs 2.26e-03 of PCC.
_QKV_IN0_BLOCK_W = {1: 8, 2: 8, 3: 8, 4: 8, 8: 8}

# out_subblock_w for the GeGLU up-projection, keyed by per_core_M. grid 6x8 with
# in0_block_w=8 won at every measured shape; only out_subblock_w moves.
_MLP_OUT_SUBBLOCK_W = {1: 2, 2: 2, 3: 3}


def mlp_up_projection_program_config(device, batch_size, seq_len, hidden_size, intermediate_size, fuse_gelu):
    """Program config for a GeGLU up-projection, or None for unmeasured shapes.

    A 6-wide grid is required, not 8: intermediate_size 1152 is 36 tiles and 36 is
    not divisible by 8, so per_core_N is an integer only for grid_x in {2,3,4,6}.

    fuse_gelu folds the activation into the matmul. Slightly more accurate than the
    separate op, because the intermediate is not rounded to bfloat16 through L1.
    """
    grid = device.compute_with_storage_grid_size()
    if grid.x < 6 or grid.y < 8:
        return None

    rows = batch_size * seq_len
    if rows % 32 or intermediate_size % 32 or hidden_size % 32:
        return None
    m_tiles, n_tiles, k_tiles = rows // 32, intermediate_size // 32, hidden_size // 32
    if m_tiles % 8 or n_tiles % 6:
        return None

    per_core_m, per_core_n = m_tiles // 8, n_tiles // 6
    out_subblock_w = _MLP_OUT_SUBBLOCK_W.get(per_core_m)
    if out_subblock_w is None or per_core_n % out_subblock_w or k_tiles % 8:
        return None

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(6, 8),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=_gelu_activation(fuse_gelu),
    )


# Sequence lengths SDPA chunking has been measured at. The chunk size itself is
# chosen from total rows, not seq_len - see sdpa_program_config.
_SDPA_MEASURED_SEQ_LENS = (256, 512, 768)
_SDPA_ROWS_FOR_LARGE_CHUNK = 768


def sdpa_program_config(device, seq_len, rows):
    """Chunk geometry for SDPA, or None to let ttnn choose (safe - untuned, not wrong).

    Keyed on rows rather than seq_len: work splits as heads x q-chunks, so small
    chunks keep the grid busy at b1s256 (24 units for 64 cores) while per-chunk
    overhead dominates by b8s256 (192).

    Tune this against the full model, not the isolated op. Larger chunks that run
    standalone have deadlocked in-model at the same shape, which is L1 pressure a
    microbenchmark cannot see.
    """
    if seq_len not in _SDPA_MEASURED_SEQ_LENS:
        return None
    chunk = 256 if rows >= _SDPA_ROWS_FOR_LARGE_CHUNK else 128
    q_chunk = k_chunk = min(chunk, seq_len)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


# Sharding the GeGLU block only pays above this much work per core, net of the
# reshard in and out: 4 and 8 tiles/core lose (-58.8%, -17.6%), 12 and up win.
# Forcing it on below the threshold costs 63% at b1s256.
_SHARD_MIN_TILES_PER_CORE = 12

# Run mlp_norm block-sharded so the Wi matmuls take an L1 in0 rather than an
# interleaved one, and the residual add stays in L1. Worth 4-7%.
_SHARD_MLP_NORM = True

# Carry the residual stream in L1 across the whole layer and between layers rather
# than only across the mlp half. The same (8,256,768) add costs 73.1 us interleaved
# and 4.7 us sharded. Worth -4.4% at b8s256.
_SHARD_RESIDUAL_STREAM = True

# 6 wide, not 8: intermediate_size 1152 is 36 tiles, which 6 divides and 8 does
# not, so one grid serves the whole block with no reshard inside it. That leaves
# 16 cores idle; zero-padding to 1280 reaches all 64 and measured a dead heat
# (21.65 ms either way), because the extra weight bytes cancel the extra cores.
_SHARD_GRID_X, _SHARD_GRID_Y = 6, 8


class MlpShardPlan(NamedTuple):
    """Everything the GeGLU block needs to run block-sharded on one 6x8 grid.

    `norm` is None when mlp_norm stays interleaved, in which case the Wi matmuls
    read an interleaved in0 and reshard on the way in.
    """

    hidden_memory: object
    intermediate_memory: object
    act_matmul: object
    gate_matmul: object
    down_matmul: object
    norm: object = None


def _sharded_matmul(m_t, in_width, out_width, fuse_gelu):
    """out_subblock_h is always 1, and that is forced rather than chosen.

    matmul asserts `out_subblock_w == per_core_N || out_subblock_h == 1`, and dest
    holds 4 tiles with fp32_dest_acc_en. Wi's per_core_N of 6 would need w=6 for any
    h>1, which exceeds dest, so 1xW is the only legal shape.
    """
    k_t, n_t = in_width // 32, out_width // 32
    per_core_n = n_t // _SHARD_GRID_X
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(_SHARD_GRID_X, _SHARD_GRID_Y),
        in0_block_w=k_t // _SHARD_GRID_X,
        out_subblock_h=1,
        out_subblock_w=next(v for v in (4, 3, 2, 1) if per_core_n % v == 0),
        per_core_M=m_t // _SHARD_GRID_Y,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=_gelu_activation(fuse_gelu),
    )


def mlp_shard_plan(device, batch_size, seq_len, hidden_size, intermediate_size):
    """Block-shard plan for the GeGLU block, or None to run interleaved.

    With `norm` set the whole mlp half stays in L1 - reshard in once, then norm,
    three matmuls, the gate multiply and the residual add. The point is the Wi
    matmuls: 33.9% utilisation on an interleaved in0 against 78.0% on an L1 one.
    """
    grid = device.compute_with_storage_grid_size()
    if grid.x < _SHARD_GRID_X or grid.y < _SHARD_GRID_Y:
        return None

    rows = batch_size * seq_len
    if rows % 32 or hidden_size % 32 or intermediate_size % 32:
        return None
    m_t, d_t, i_t = rows // 32, hidden_size // 32, intermediate_size // 32
    if m_t % _SHARD_GRID_Y or d_t % _SHARD_GRID_X or i_t % _SHARD_GRID_X:
        return None
    if (m_t * d_t) / (_SHARD_GRID_X * _SHARD_GRID_Y) < _SHARD_MIN_TILES_PER_CORE:
        return None

    def block(width):
        return ttnn.create_sharded_memory_config(
            shape=(1, rows, width),
            core_grid=ttnn.CoreGrid(y=_SHARD_GRID_Y, x=_SHARD_GRID_X),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    block_w = d_t // _SHARD_GRID_X
    norm = (
        ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(_SHARD_GRID_X, _SHARD_GRID_Y),
            subblock_w=next(v for v in (4, 2, 1) if block_w % v == 0),
            block_h=m_t // _SHARD_GRID_Y,
            block_w=block_w,
            inplace=False,
        )
        if _SHARD_MLP_NORM
        else None
    )

    return MlpShardPlan(
        hidden_memory=block(hidden_size),
        intermediate_memory=block(intermediate_size),
        act_matmul=_sharded_matmul(m_t, hidden_size, intermediate_size, fuse_gelu=True),
        gate_matmul=_sharded_matmul(m_t, hidden_size, intermediate_size, fuse_gelu=False),
        down_matmul=_sharded_matmul(m_t, intermediate_size, hidden_size, fuse_gelu=False),
        norm=norm,
    )


def qkv_matmul_program_config(device, batch_size, seq_len, hidden_size):
    """Program config for the fused Wqkv projection, or None to let ttnn choose.

    Keyed on batch_size * seq_len, not seq_len: the matmul flattens batch into M
    and requires num_blocks_y <= grid rows, so a seq_len-only config is rejected at
    batch > 1. Worth -15.2% at b8s256 against ttnn's automatic choice.
    """
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        return None

    rows = batch_size * seq_len
    if rows % 32 or (3 * hidden_size) % 32:
        return None
    m_tiles = rows // 32
    n_tiles = (3 * hidden_size) // 32
    if m_tiles % 8 or n_tiles % 8:
        return None

    per_core_m, per_core_n = m_tiles // 8, n_tiles // 8
    in0_block_w = _QKV_IN0_BLOCK_W.get(per_core_m)
    if in0_block_w is None:
        return None
    if (hidden_size // 32) % in0_block_w:
        return None

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )
