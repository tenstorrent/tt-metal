// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t face(uint32_t r, uint32_t c) { return (r >> 4) * 512 + (c >> 4) * 256 + (r & 15u) * 16 + (c & 15u); }
uint32_t complete_groups(uint32_t bits, uint32_t capacity) {
    if (bits == 0) {
        return 0;
    }
    const uint32_t exponent = (bits >> 23) & 255u;
    ASSERT((bits >> 31) == 0 && exponent >= 127 && exponent <= 146);
    const uint32_t position = ((bits & 0x7fffffu) | 0x800000u) >> (150u - exponent);
    ASSERT(position < capacity * 4);
    return (position + 1) / 4;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    const auto queries = TensorAccessor(qa, get_arg_val<uint32_t>(0), 2048);
    const auto keys = TensorAccessor(ka, get_arg_val<uint32_t>(1), 2048);
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(2), 4096);
    const uint32_t first = get_arg_val<uint32_t>(3), stride = get_arg_val<uint32_t>(4);
    const uint32_t kt = get_arg_val<uint32_t>(5), rows = get_arg_val<uint32_t>(6);
    // Metadata is retained until all work completes. Its first 32 words are
    // integer live-key counts, transported losslessly rather than via SFPU.
    cb_reserve_back(0, 1);
    const uint32_t meta_address = get_write_ptr(0);
    noc_async_read(positions.get_noc_addr(0), meta_address, 4096);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_address);
    uint32_t live[32];
    for (uint32_t q = 0; q < rows; ++q) {
        live[q] = complete_groups(meta[face(q, 0)], kt * 32);
    }
    for (uint32_t q = 0; q < rows; ++q) {
        meta[q] = live[q];
    }
    cb_push_back(0, 1);
    // Reader-only query scratch, independent of the streamed BF16 broadcasts.
    cb_reserve_back(1, 4);
    const uint32_t query_address = get_write_ptr(1);
    const volatile tt_l1_ptr uint16_t* query = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(query_address);
    for (uint32_t job = first; job < kt * 4; job += stride) {
        const uint32_t head = job / kt, key_tile = job % kt;
        bool active = false;
        for (uint32_t q = 0; q < rows; ++q) {
            active |= key_tile * 32 < live[q];
        }
        if (!active) {
            continue;
        }
        cb_reserve_back(3, 4);
        const uint32_t key_address = get_write_ptr(3);
        for (uint32_t w = 0; w < 4; ++w) {
            noc_async_read(queries.get_noc_addr(head * 4 + w), query_address + w * 2048, 2048);
            noc_async_read(keys.get_noc_addr(key_tile * 4 + w), key_address + w * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(3, 4);
        for (uint32_t q = 0; q < rows; ++q) {
            if (key_tile * 32 >= live[q]) {
                continue;
            }
            for (uint32_t w = 0; w < 4; ++w) {
                cb_reserve_back(2, 1);
                volatile tt_l1_ptr uint16_t* broadcast =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(2));
                // ROW broadcast reads only row 0: face offsets 0..15 and
                // 256..271. Compute expands these 32 BF16 values, avoiding
                // 1024 scalar stores for every query tile.
                for (uint32_t c = 0; c < 32; ++c) {
                    broadcast[face(0, c)] = query[w * 1024 + face(q, c)];
                }
                cb_push_back(2, 1);
            }
        }
    }
}
