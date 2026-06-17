// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the AXPY example.
//
// Streams x[i] and y[i] tiles in from DRAM and, additionally, materializes
// a single tile filled with the scalar 'a' that the compute kernel will reuse
// for every iteration (broadcast-by-replication).
//
// Producing the scalar tile inside the reader (rather than uploading it from
// the host) keeps the host code shorter and makes this example self-contained.

#include <cstdint>
#include <cstring>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tt-metalium/constants.hpp"

inline uint16_t float_to_bfloat16_bits(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t a_bits = get_arg_val<uint32_t>(2);  // bfloat16 in lower 16 bits
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_y = tt::CBIndex::c_1;
    constexpr uint32_t cb_a = tt::CBIndex::c_2;

    const uint32_t tile_size_bytes = get_tile_size(cb_x);

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, x_addr);
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto y = TensorAccessor(y_args, y_addr);

    // Fill cb_a once with a 32x32 tile where every element is the scalar 'a'.
    // The compute kernel cb_wait_front's this tile a single time and reuses it
    // for every iteration of the loop.
    cb_reserve_back(cb_a, 1);
    {
        const uint32_t l1_addr = get_write_ptr(cb_a);
        volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
        const uint16_t a_bf16 = static_cast<uint16_t>(a_bits & 0xFFFFu);
        for (uint32_t i = 0; i < tt::constants::TILE_HW; ++i) {
            ptr[i] = a_bf16;
        }
    }
    cb_push_back(cb_a, 1);

    Noc noc;
    CircularBuffer cb_x_buf(cb_x);
    CircularBuffer cb_y_buf(cb_y);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_x_buf.reserve_back(1);
        cb_y_buf.reserve_back(1);

        noc.async_read(x, cb_x_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(y, cb_y_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();

        cb_x_buf.push_back(1);
        cb_y_buf.push_back(1);
    }
}
