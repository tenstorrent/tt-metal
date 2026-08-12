// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
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
 * Read a block of in0 from a (possibly padded) tensor, optionally sub-ranged at an M-row /
 * K-col offset (UseOffset treats the source as a parent buffer starting at
 * (in0_row_offset_tiles, in0_k_offset_tiles)). Bounds checks use the effective matmul `shape`,
 * not the parent stride.
 *
 * Iteration is M-outer, K-inner (CB layout [M, K] tile-major). TransposeA storage is
 * [K_parent, M_parent] (row stride parent_M_tiles_stride); non-transpose is [M_parent, K_parent]
 * (row stride parent_K_tiles_stride). Both strides used only when UseOffset.
 *
 * `shape` holds matmul-coordinate sizes: non-transpose (logical_d0=M, logical_d1=K),
 * transpose (logical_d0=K, logical_d1=M).
 */
template <uint32_t M_block_tiles, uint32_t K_block_tiles, bool TransposeA, bool UseOffset, typename TensorAccessorType>
void read_in0_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t dst_cb_id,
    uint32_t tile_size_bytes,
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

    // Padded/out-of-range tiles are zero-filled into dst_cb_id; offset within the CB is the
    // current write pointer minus the CB base (write_ptr at entry).
    Noc noc;
    const uint32_t cb_base = write_ptr;

    // i sweeps M (matmul-outer), j sweeps K (matmul-inner).
    const uint32_t m_bound = TransposeA ? shape.logical_d1 : shape.logical_d0;
    const uint32_t k_bound = TransposeA ? shape.logical_d0 : shape.logical_d1;
    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= m_bound) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j < k_bound) {
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
                noc.async_read(
                    tensor_accessor, CoreLocalMem<uint32_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
            } else {
                fill_zeros_async(noc, dst_cb_id, tile_size_bytes, write_ptr - cb_base);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_read_barrier();
}

/**
 * Read a block of in1 from a (possibly padded) tensor, optionally K-sliced: UseOffset treats
 * the weight as a parent buffer and reads K-range [k_offset, k_offset + matmul_K). Bounds
 * checks use the effective matmul `shape`, not the parent K extent.
 *
 * The CB ends up K-outer, N-inner (tile (k, n) at index k * N_block_tiles + n), as the matmul
 * compute kernel expects.
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, bool TransposeB, bool UseOffset, typename TensorAccessorType>
void read_in1_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr_base,
    uint32_t dst_cb_id,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end,
    uint32_t in1_k_offset_tiles,
    uint32_t parent_K_tiles_stride) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);
    // Padded/out-of-range tiles are zero-filled into dst_cb_id; offset within the CB is the
    // target write pointer minus the CB base (write_ptr_base).
    Noc noc;
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
                    fill_zeros_async(noc, dst_cb_id, tile_size_bytes, wp - write_ptr_base);
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
                    noc.async_read(
                        tensor_accessor, CoreLocalMem<uint32_t>(wp), tile_size_bytes, {.page_id = tile_id}, {});
                } else {
                    fill_zeros_async(noc, dst_cb_id, tile_size_bytes, wp - write_ptr_base);
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
                    noc.async_read(
                        tensor_accessor, CoreLocalMem<uint32_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
                } else {
                    fill_zeros_async(noc, dst_cb_id, tile_size_bytes, write_ptr - write_ptr_base);
                }
                write_ptr += tile_size_bytes;
            }
            // finish up incrementing write_ptr if (d1_end - d1_start) < N_block_tiles
            write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
        }
    }
    noc.async_read_barrier();
}

/**
 * Write a block of output to a (possibly padded) tensor; skip tiles where M >= logical_M or
 * N >= logical_N. When UseOutOffset is true the output is a parent buffer and we write into
 * rows [out_row_offset_tiles, out_row_offset_tiles + actual_M); matmul-N equals the parent's N
 * (caller-validated), so shape.logical_d1 stays the correct N stride.
 */
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

    Noc noc;
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
            noc.async_write(
                CoreLocalMem<uint32_t>(read_ptr), tensor_accessor, tile_size_bytes, {}, {.page_id = tile_id});
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_writes_flushed();
}

/**
 * Like write_block_sync, but waits on one row of output tiles in the CB at a time rather than
 * the whole block before writing.
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
    Noc noc;
    CircularBuffer cb_out(cb_id_out);
    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_out.wait_front(N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < shape.logical_d0) {
            uint32_t out_read_ptr = cb_out.get_read_ptr();
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
                noc.async_write(
                    CoreLocalMem<uint32_t>(out_read_ptr), tensor_accessor, tile_size_bytes, {}, {.page_id = tile_id});
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_out.pop_front(N_block_tiles);
    }
    noc.async_writes_flushed();
}

/**
 * Staggered defer-write index for both senders: each core picks a k-block for its deferred
 * output write so writes spread across the next output block's K-loop (latency hiding).
 * Clamp to the last K iter — otherwise K-shrinking OffsetsRoles can leave the trigger never
 * firing, deadlocking cb_id_out once M_blocks_per_core * N_blocks_per_core - 1 exceeds CB depth.
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
    CircularBuffer cb_out(cb_id_out);
    cb_out.wait_front(out_block_num_tiles);
    const uint32_t out_read_ptr = cb_out.get_read_ptr();
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
    cb_out.pop_front(out_block_num_tiles);
}
