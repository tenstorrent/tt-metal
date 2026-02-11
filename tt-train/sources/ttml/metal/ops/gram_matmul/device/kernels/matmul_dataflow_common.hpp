// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "api/dataflow/dataflow_api.h"

namespace detail {
/**
 * Helper to create tuple of TensorAccessors from TensorAccessorArgs tuple.
 * Unlike make_tensor_accessor_tuple, this takes the actual page_size value,
 * not a Compile Time Argument (CTA) index.
 */
template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple_with_page_size(
    const std::tuple<Args...>& args_tuple,
    uint32_t address_rt_arg_index_start,
    uint32_t page_size,
    std::integer_sequence<uint32_t, Indexes...>) {
    return std::make_tuple(TensorAccessor(
        std::get<Indexes>(args_tuple), get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes), page_size)...);
}
}  // namespace detail

/**
 * Create a tuple of TensorAccessors from a tuple of TensorAccessorArgs.
 * Each tensor gets its address from consecutive RT args starting at address_rt_arg_index_start.
 * All tensors share the same page_size (actual value, not CTA index).
 */
template <typename... Args>
auto make_tensor_accessor_tuple_uniform_page_size(
    const std::tuple<Args...>& args_tuple, uint32_t address_rt_arg_index_start, uint32_t page_size) {
    return detail::make_tensor_accessor_tuple_with_page_size(
        args_tuple, address_rt_arg_index_start, page_size, std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}
void fill_zeros_async(uint32_t write_addr, uint32_t tile_bytes) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    // Fill tile with zeros
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
 * Read a block of in0 from a potentially padded tensor.
 * Since this is for matmul, no need to read when M >= logical_M
 * Otherwise, if K >= logical_K, fill with zeros.
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
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Read a block of in1 from a potentially padded tensor.
 * Since this is for matmul, no need to read when N >= logical_N
 * Otherwise, if K >= logical_K, fill with zeros.
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_in1_block_sync(
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
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                write_ptr += tile_size_bytes;
                continue;
            }
            if (i < shape.logical_d0) {
                uint32_t tile_id = i * shape.logical_d1 + j;
                noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
            } else {
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Read a [K_block, N_block] block for in1 by reading from an [M, K] tensor with transposed indexing.
 * For CB position (k_row, n_col), reads tile X[n, k] = tile_id (n * logical_K + k).
 * Combined with transpose=true in the compute kernel, this produces X^T[k, n] on the fly.
 *
 * shape: the original [M, K] tensor shape (d0 = M_tiles, d1 = K_tiles).
 * k_start..k_end: K tile range (rows of the in1 block).
 * n_start..n_end: N tile range (cols of the in1 block, which correspond to M rows of X).
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_in1_block_transposed_sync(
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
            if (n >= shape.logical_d0) {
                // n indexes M dimension of X; beyond logical M -> zero pad
                write_ptr += tile_size_bytes;
                continue;
            }
            if (k < shape.logical_d1) {
                // k indexes K dimension of X; read tile X[n, k]
                uint32_t tile_id = n * shape.logical_d1 + k;
                noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
            } else {
                // k beyond logical K -> zero pad
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (n_end - n_start) < N_block_tiles
        write_ptr += (N_block_tiles - (n_end - n_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Write a block of output to a potentially padded tensor.
 * Skip writing when M >= logical_M or N >= logical_N
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
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_writes_flushed();
}

/**
 * This write method is more granular, waiting on a row of output tiles
 * in the output CB before writing those out, rather than waiting on the entire block.
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
