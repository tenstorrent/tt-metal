// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

struct TensorShape2D {
    uint32_t logical_d0;
    uint32_t logical_d1;
    uint32_t padded_d0;
    uint32_t padded_d1;
    // Constructor to initialize with 2D shape
    TensorShape2D(uint32_t _d0, uint32_t _d1, uint32_t _padded_d0, uint32_t _padded_d1) :
        logical_d0(_d0), logical_d1(_d1), padded_d0(_padded_d0), padded_d1(_padded_d1) {
        ASSERT(_d0 > 0);
        ASSERT(_d1 > 0);
        ASSERT(_d0 <= _padded_d0);
        ASSERT(_d1 <= _padded_d1);
    }
};

/**
 * Read a block of in0 from a potentially padded tensor, optionally at an M-row and/or K-col
 * offset into the source tensor (the source is treated as a parent buffer; we read just the
 * sub-range starting at (in0_row_offset_tiles, in0_k_offset_tiles)). Bounds check on
 * matmul-M/K uses the effective `shape`, independent of the parent stride.
 *
 * Iteration order is always M-outer, K-inner (CB layout = [M, K] tile-major). When
 * TransposeA is true, the physical tensor is stored as [K_parent, M_parent] instead of
 * [M_parent, K_parent]; the address formula uses `parent_M_tiles_stride` for the K-row
 * stride. For non-transpose, the row stride is `parent_K_tiles_stride` (= parent K tile
 * count, used only when UseOffset).
 *
 * `shape` carries the matmul-coordinate effective sizes (logical_d0=effective_M for
 * non-transpose / logical_d0=K for transpose, logical_d1=K_tiles or effective_M).
 */
template <
    uint32_t M_block_tiles,
    uint32_t K_block_tiles,
    bool TransposeA,
    bool UseOffset,
    typename TensorAccessorType
#ifdef READ_FROM_LOCAL_INPUT
    ,
    typename LocalTensorAccessorType
#endif
    >
void read_in0_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t tile_size_bytes,
#ifdef READ_FROM_LOCAL_INPUT
    const LocalTensorAccessorType& in3_accessor,
    uint32_t local_k_start,
    uint32_t local_k_end,
    uint32_t input_tensor_Wt,
#endif
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end,
    uint32_t in0_row_offset_tiles,
    uint32_t parent_M_tiles_stride,
    uint32_t in0_k_offset_tiles,
    uint32_t parent_K_tiles_stride) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    // i sweeps M (matmul-outer), j sweeps K (matmul-inner).
    const uint32_t m_bound = TransposeA ? shape.logical_d1 : shape.logical_d0;
    const uint32_t k_bound = TransposeA ? shape.logical_d0 : shape.logical_d1;
    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= m_bound) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j < k_bound) {
#ifdef READ_FROM_LOCAL_INPUT
                if (local_k_start <= j && j <= local_k_end) {
                    // read from self_tensor_accessor
                    uint32_t local_i = UseOffset ? (i + in0_row_offset_tiles) : i;
                    uint32_t tile_id = local_i * input_tensor_Wt + (j - local_k_start);
                    noc_async_read_tile(tile_id, in3_accessor, write_ptr);
                } else {
#endif
                    uint32_t tile_id;
                    if constexpr (TransposeA) {
                        if constexpr (UseOffset) {
                            // [K_parent, M_parent] storage:
                            // (K-row + k_offset) * M_parent_tiles + (M-col + m_offset)
                            tile_id = (j + in0_k_offset_tiles) * parent_M_tiles_stride + (i + in0_row_offset_tiles);
                        } else {
                            tile_id = j * shape.logical_d1 + i;
                        }
                    } else {
                        if constexpr (UseOffset) {
                            // [M_parent, K_parent] storage:
                            // (M-row + m_offset) * K_parent_tiles + (K-col + k_offset)
                            tile_id = (i + in0_row_offset_tiles) * parent_K_tiles_stride + (j + in0_k_offset_tiles);
                        } else {
                            tile_id = i * shape.logical_d1 + j;
                        }
                    }
                    noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
#ifdef READ_FROM_LOCAL_INPUT
                }
#endif
            } else {
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Read a block of in1 from a potentially padded tensor, optionally at a K-axis offset.
 * When UseOffset is true the weight is treated as a parent buffer and the K-axis is
 * sliced [k_offset, k_offset + matmul_K). Bounds checks use the effective (matmul-K)
 * `shape`, independent of the parent K extent.
 *
 * The CB ends up K-outer, N-inner (tile (k, n) at index k * N_block_tiles + n), which
 * is what the matmul compute kernel expects.
 *
 * DRAM access pattern:
 *  - Non-transpose ([K, N] storage): K-outer / N-inner iteration walks storage in
 *    row-major order → sequential DRAM reads.
 *  - Transpose_b ([N, K] storage): K-outer / N-inner iteration jumps K-tiles between
 *    sibling reads → page-thrashing strided pattern. For the transposed case we walk
 *    storage in row-major order (N-outer / K-inner) and compute write_ptr per tile so
 *    the CB layout stays K-outer / N-inner. Sequential DRAM bandwidth recovers ~10x
 *    on large-K weights (e.g. Mixtral H=4096, I=14336).
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, bool TransposeB, bool UseOffset, typename TensorAccessorType>
void read_in1_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr_base,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end,
    uint32_t in1_k_offset_tiles,
    uint32_t parent_K_tiles_stride) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);
    // i sweeps K (matmul-inner), j sweeps N (matmul-outer).
    // shape carries storage-layout sizes: when TransposeB, logical_d0=N, logical_d1=K.
    const uint32_t k_bound = TransposeB ? shape.logical_d1 : shape.logical_d0;
    const uint32_t n_bound = TransposeB ? shape.logical_d0 : shape.logical_d1;
    if constexpr (TransposeB) {
        // Storage-sequential reads: N-outer / K-inner. CB index = (k - d0_start) * N_block + (n - d1_start).
        for (uint32_t j = d1_start; j < d1_end; j++) {
            const uint32_t n_col = j - d1_start;
            if (j >= n_bound) {
                // Out-of-N tiles: zero-fill the whole K-column in this N slot.
                for (uint32_t i = d0_start; i < d0_end; i++) {
                    uint32_t wp = write_ptr_base + ((i - d0_start) * N_block_tiles + n_col) * tile_size_bytes;
                    fill_zeros_async(wp, tile_size_bytes);
                }
                continue;
            }
            for (uint32_t i = d0_start; i < d0_end; i++) {
                uint32_t wp = write_ptr_base + ((i - d0_start) * N_block_tiles + n_col) * tile_size_bytes;
                if (i < k_bound) {
                    uint32_t tile_id;
                    if constexpr (UseOffset) {
                        tile_id = j * parent_K_tiles_stride + (i + in1_k_offset_tiles);
                    } else {
                        tile_id = j * shape.logical_d1 + i;
                    }
                    noc_async_read_tile(tile_id, tensor_accessor, wp);
                } else {
                    fill_zeros_async(wp, tile_size_bytes);
                }
            }
        }
    } else {
        // Non-transpose: storage [K, N], K-outer / N-inner is already storage-sequential.
        uint32_t write_ptr = write_ptr_base;
        for (uint32_t i = d0_start; i < d0_end; i++) {
            for (uint32_t j = d1_start; j < d1_end; j++) {
                if (j >= n_bound) {
                    write_ptr += tile_size_bytes;
                    continue;
                }
                if (i < k_bound) {
                    uint32_t tile_id;
                    if constexpr (UseOffset) {
                        tile_id = (i + in1_k_offset_tiles) * shape.logical_d1 + j;
                    } else {
                        tile_id = i * shape.logical_d1 + j;
                    }
                    noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
                } else {
                    fill_zeros_async(write_ptr, tile_size_bytes);
                }
                write_ptr += tile_size_bytes;
            }
            // finish up incrementing write_ptr if (d1_end - d1_start) < N_block_tiles
            write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
        }
    }
    noc_async_read_barrier();
}

/**
 * Write a block of output to a potentially padded tensor.
 * Skip writing when M >= logical_M or N >= logical_N
 */
// When UseOutOffset is true, the output tensor is a parent buffer; we write into rows
// [out_row_offset_tiles, out_row_offset_tiles + actual_M) of it. matmul-N must equal
// the parent's N (caller-validated), so shape.logical_d1 is still the correct N stride.
template <uint32_t M_block_tiles, uint32_t N_block_tiles, bool UseOutOffset, typename TensorAccessorType>
void write_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end,
    uint32_t out_row_offset_tiles) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                read_ptr += tile_size_bytes;
                continue;
            }
            uint32_t row;
            if constexpr (UseOutOffset) {
                row = i + out_row_offset_tiles;
            } else {
                row = i;
            }
            uint32_t tile_id = row * shape.logical_d1 + j;
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_writes_flushed();
}

/**
 * This write method is more granular, waiting on a row of output tiles
 * in the output CB before writing those out, rather than waiting on the entire block.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, bool UseOutOffset, typename TensorAccessorType>
void write_block_sync_granular(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end,
    uint32_t out_row_offset_tiles) {
    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_wait_front(cb_id_out, N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < shape.logical_d0) {
            uint32_t out_read_ptr = get_read_ptr(cb_id_out);
            uint32_t row;
            if constexpr (UseOutOffset) {
                row = m_tile + out_row_offset_tiles;
            } else {
                row = m_tile;
            }
            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
                if (n_tile_id >= shape.logical_d1) {
                    break;
                }
                uint32_t tile_id = row * shape.logical_d1 + n_tile_id;
                noc_async_write_tile(tile_id, tensor_accessor, out_read_ptr);
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    noc_async_writes_flushed();
}

/**
 * Staggered defer-write index used by both dm_in0_sender and dm_in1_sender_out: each core picks
 * a k-block index for its deferred output write so that the writes spread across the next
 * output block's K-loop (latency hiding). Clamp to the last K iter — without this, K-axis
 * OffsetsRoles that shrink K at runtime can leave the check never firing, deadlocking
 * cb_id_out once M_blocks_per_core * N_blocks_per_core - 1 exceeds the CB depth.
 */
FORCE_INLINE uint32_t compute_defer_write_k_block(uint32_t core_y_index, uint32_t y_axis_cores, uint32_t K_num_blocks) {
    const uint32_t k_blocks_per_axis_core = (K_num_blocks + y_axis_cores - 1U) / y_axis_cores;
    uint32_t defer_write_k_block = core_y_index * k_blocks_per_axis_core;
    if (K_num_blocks > 0U) {
        defer_write_k_block = std::min(defer_write_k_block, K_num_blocks - 1U);
    }
    return defer_write_k_block;
}

/**
 * Wait on the output CB, write the previously-deferred block to DRAM, and pop.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, bool UseOutOffset, typename TensorAccessorType>
FORCE_INLINE void do_deferred_block_write(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& out_shape,
    uint32_t cb_id_out,
    uint32_t out_tile_size,
    uint32_t defer_write_m_tile,
    uint32_t defer_write_m_tile_end,
    uint32_t defer_write_n_tile,
    uint32_t defer_write_n_tile_end,
    uint32_t out_row_offset_tiles) {
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    cb_wait_front(cb_id_out, out_block_num_tiles);
    const uint32_t out_read_ptr = get_read_ptr(cb_id_out);
    write_block_sync<M_block_tiles, N_block_tiles, UseOutOffset>(
        tensor_accessor,
        out_shape,
        out_read_ptr,
        out_tile_size,
        defer_write_m_tile,
        defer_write_m_tile_end,
        defer_write_n_tile,
        defer_write_n_tile_end,
        out_row_offset_tiles);
    cb_pop_front(cb_id_out, out_block_num_tiles);
}

/**
 * Final (non-deferred) output block write at the end of an N-block iteration: granular variant
 * pops the output CB tile-by-tile to overlap compute and write.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, bool UseOutOffset, typename TensorAccessorType>
FORCE_INLINE void do_final_block_write(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& out_shape,
    uint32_t cb_id_out,
    uint32_t out_tile_size,
    uint32_t m_tile,
    uint32_t m_tile_end,
    uint32_t n_tile,
    uint32_t n_tile_end,
    uint32_t out_row_offset_tiles) {
    write_block_sync_granular<M_block_tiles, N_block_tiles, UseOutOffset>(
        tensor_accessor,
        out_shape,
        cb_id_out,
        out_tile_size,
        m_tile,
        m_tile_end,
        n_tile,
        n_tile_end,
        out_row_offset_tiles);
}
