// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Dataflow helpers for gram_polynomial Phase 2 (G*G matmul).
// Differs from gram_matmul's matmul_dataflow_common.hpp:
//   - read_in1_block_sync reads G[k, n] directly (no transpose).
//   - read_g_block_sync reads G[i, j] block into c_4 for bG epilogue.
//   - Reuses read_in0_block_sync, write_block_sync, write_block_sync_granular unchanged.
//

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void fill_zeros_async(uint32_t write_addr, uint32_t tile_bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t bytes_left = tile_bytes;
    for (;;) {
        uint32_t read_size = bytes_left > MEM_ZEROS_SIZE ? MEM_ZEROS_SIZE : bytes_left;
        noc_async_read(zeros_noc_addr, write_addr, read_size);
        write_addr += read_size;
        bytes_left -= read_size;
        if (bytes_left == 0) {
            break;
        }
    }
}

struct TensorShape2D {
    uint32_t logical_d0;
    uint32_t logical_d1;
    uint32_t padded_d0;
    uint32_t padded_d1;
    TensorShape2D(uint32_t _d0, uint32_t _d1, uint32_t _padded_d0, uint32_t _padded_d1) :
        logical_d0(_d0), logical_d1(_d1), padded_d0(_padded_d0), padded_d1(_padded_d1) {
        ASSERT(_d0 > 0);
        ASSERT(_d1 > 0);
        ASSERT(_d0 <= _padded_d0);
        ASSERT(_d1 <= _padded_d1);
    }
};

/**
 * Read a block of in0 from a potentially padded tensor.
 * Reads G[m, k] for the matmul LHS.
 */
template <uint32_t M_block_tiles, uint32_t K_block_tiles, typename TensorAccessorType>
void read_in0_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j < shape.logical_d1) {
                uint32_t tile_id = i * shape.logical_d1 + j;
                noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
            } else {
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Read a [K_block, N_block] block for in1 WITHOUT transpose.
 * Reads G[k, n] directly — tile_id = k * logical_N + n.
 * This is the key difference from gram_matmul's read_in1_block_transposed_sync.
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_in1_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t tile_size_bytes,
    uint32_t k_start,
    uint32_t k_end,
    uint32_t n_start,
    uint32_t n_end) {
    ASSERT(k_end > k_start);
    ASSERT(n_end > n_start);
    // CB layout: [K_block, N_block] row-major
    for (uint32_t k = k_start; k < k_end; k++) {
        for (uint32_t n = n_start; n < n_end; n++) {
            if (k < shape.logical_d0 && n < shape.logical_d1) {
                // G is square: shape.logical_d0 == shape.logical_d1 == M_tiles
                // tile_id = k * M_tiles + n
                uint32_t tile_id = k * shape.logical_d1 + n;
                noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
            } else {
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        write_ptr += (N_block_tiles - (n_end - n_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Read a [M_block, N_block] block of G[i, j] into c_4 for the bG epilogue.
 * Same indexing as write_block (G is row-major, tile_id = m * logical_N + n).
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_g_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t tile_size_bytes,
    uint32_t m_start,
    uint32_t m_end,
    uint32_t n_start,
    uint32_t n_end) {
    ASSERT(m_end > m_start);
    ASSERT(n_end > n_start);

    for (uint32_t i = m_start; i < m_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = n_start; j < n_end; j++) {
            if (j < shape.logical_d1) {
                uint32_t tile_id = i * shape.logical_d1 + j;
                noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
            } else {
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        write_ptr += (N_block_tiles - (n_end - n_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Write a block of output to a potentially padded tensor.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
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
            uint32_t tile_id = i * shape.logical_d1 + j;
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);
            read_ptr += tile_size_bytes;
        }
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_writes_flushed();
}

/**
 * Granular write: waits on a row of output tiles at a time.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync_granular(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_wait_front(cb_id_out, N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < shape.logical_d0) {
            uint32_t out_read_ptr = get_read_ptr(cb_id_out);
            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
                if (n_tile_id >= shape.logical_d1) {
                    break;
                }
                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                noc_async_write_tile(tile_id, tensor_accessor, out_read_ptr);
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    noc_async_writes_flushed();
}
