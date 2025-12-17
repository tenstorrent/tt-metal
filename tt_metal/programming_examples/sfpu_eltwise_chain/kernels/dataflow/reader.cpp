// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief Converts a 32-bit IEEE 754 float to 16-bit bfloat16 format.
 *
 * This function performs a simple truncation conversion from float32 to bfloat16
 * by extracting the upper 16 bits (sign, exponent, and upper 7 bits of mantissa)
 * of the IEEE 754 float representation. The lower 16 bits of the mantissa are
 * discarded, which may result in precision loss but maintains the same range
 * as float32.
 *
 * @param value The input 32-bit floating point value to convert
 * @return uint16_t The resulting 16-bit bfloat16 value in its binary representation
 *
 * @note This implementation uses simple truncation without rounding, which may
 *       introduce quantization errors for values that cannot be exactly
 *       represented in bfloat16 format - it is sufficient for this example.
 */
inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t src_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(1);

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<2>();
    const auto interleaved_accessor =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    cb_reserve_back(src_cb_index, one_tile);
    const uint32_t l1_write_addr = get_write_ptr(src_cb_index);
    noc_async_read_tile(0, interleaved_accessor, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(src_cb_index, one_tile);

    // Create tile with ones
    cb_reserve_back(ones_cb_index, one_tile);
    const uint32_t ones_l1_write_addr = get_write_ptr(ones_cb_index);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = float_to_bfloat16(1.0f);
    }
    cb_push_back(ones_cb_index, one_tile);
}
