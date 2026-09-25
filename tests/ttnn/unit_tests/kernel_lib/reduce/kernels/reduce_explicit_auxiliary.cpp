// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_auxiliary = 1;
constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
constexpr uint32_t words_per_tile = 3;

// Physical pattern of one auxiliary tile; values match TILE_TYPE in reduce_explicit_modes.py.
enum class TileType : uint32_t { FirstRow = 0, FirstColumn = 1, FirstRowPerFaceRow = 2, Zero = 3 };

template <DataFormat data_format, uint32_t face_rows, uint32_t faces_per_row>
FORCE_INLINE void fill_first_column(volatile tt_l1_ptr uint32_t* ptr, uint32_t value, uint32_t valid_rows) {
    constexpr uint32_t face_size_u32 = data_format == DataFormat::Float32 ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t row_size_u32 = data_format == DataFormat::Float32 ? ROW_SIZE_U32_FP32 : ROW_SIZE_U32;
    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        const uint32_t face_row_start = face_row * tt::constants::FACE_HEIGHT;
        const uint32_t remaining = valid_rows > face_row_start ? valid_rows - face_row_start : 0;
        const uint32_t rows_in_face = remaining < tt::constants::FACE_HEIGHT ? remaining : tt::constants::FACE_HEIGHT;
        volatile tt_l1_ptr uint32_t* face_ptr = ptr + face_row * faces_per_row * face_size_u32;
        for (uint32_t row = 0; row < rows_in_face; ++row) {
            fill_face_row0_cols<data_format>(face_ptr + row * row_size_u32, value, 1);
        }
    }
}

template <TileType tile_type, uint32_t valid_elements, uint32_t value_bits>
FORCE_INLINE void prepare_tile() {
    constexpr DataFormat data_format = get_dataformat(cb_auxiliary);
    constexpr uint32_t face_rows = get_tile_r_dim<cb_auxiliary>() / tt::constants::FACE_HEIGHT;
    constexpr uint32_t faces_per_row = get_tile_c_dim<cb_auxiliary>() / tt::constants::FACE_WIDTH;

    DataflowBuffer dfb(cb_auxiliary);
    dfb.reserve_back(1);
    const uint32_t write_addr = dfb.get_write_ptr();
    Noc noc;
    noc.async_write_zeros(dfb, get_tile_size(cb_auxiliary));
    noc.write_zeros_l1_barrier();

    const uint32_t value = float_to_scaler_bits<data_format>(__builtin_bit_cast(float, value_bits));
    auto* ptr = addr_to_l1_ptr(write_addr);
    if constexpr (tile_type == TileType::FirstRow) {
        fill_each_face_row0_partial<data_format, ReduceDim::REDUCE_ROW, face_rows, faces_per_row>(
            ptr, value, valid_elements);
    } else if constexpr (tile_type == TileType::FirstColumn) {
        fill_first_column<data_format, face_rows, faces_per_row>(ptr, value, valid_elements);
    } else if constexpr (tile_type == TileType::FirstRowPerFaceRow) {
        fill_each_face_row0_partial<data_format, ReduceDim::REDUCE_COL, face_rows, faces_per_row>(
            ptr, value, valid_elements);
    }
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    flush_l2_cache_range(write_addr, get_tile_size(cb_auxiliary));
#endif
    dfb.push_back(1);
}

template <uint32_t tile = 0>
FORCE_INLINE void prepare_tiles() {
    if constexpr (tile < num_tiles) {
        constexpr uint32_t base = 1 + tile * words_per_tile;
        prepare_tile<
            static_cast<TileType>(get_compile_time_arg_val(base)),
            get_compile_time_arg_val(base + 1),
            get_compile_time_arg_val(base + 2)>();
        prepare_tiles<tile + 1>();
    }
}

}  // namespace

void kernel_main() {
    prepare_tiles();
#ifdef REDUCE_STREAM_OUTPUT
    // Drain the one-page compute output CB into the resident output tensor. A bulk reservation in
    // compute would deadlock before the first output; per-tile publication permits every ring wrap.
    DataflowBuffer computed(16), output(17);
    const uint32_t output_tiles = get_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(16);
    for (uint32_t tile = 0; tile < output_tiles; ++tile) {
        computed.wait_front(1);
        output.reserve_back(1);
        noc_async_write(computed.get_read_ptr(), get_noc_addr(output.get_write_ptr()), tile_bytes);
        noc_async_write_barrier();
        computed.pop_front(1);
        output.push_back(1);
    }
#endif
}
