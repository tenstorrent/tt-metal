// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t input_tiles = get_compile_time_arg_val(1);
    constexpr bool compact = get_compile_time_arg_val(2) != 0;
    constexpr auto ia = TensorAccessorArgs<3>();
    constexpr auto ha = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto t0a = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto t1a = TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    constexpr auto t2a = TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    constexpr auto t3a = TensorAccessorArgs<t2a.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(ia, get_arg_val<uint32_t>(0), 2048);
    const auto history = TensorAccessor(ha, get_arg_val<uint32_t>(1), 5120);
    const auto tap0 = TensorAccessor(t0a, get_arg_val<uint32_t>(2), 2048);
    const auto tap1 = TensorAccessor(t1a, get_arg_val<uint32_t>(3), 2048);
    const auto tap2 = TensorAccessor(t2a, get_arg_val<uint32_t>(4), 2048);
    const auto tap3 = TensorAccessor(t3a, get_arg_val<uint32_t>(5), 2048);
    const uint32_t first = get_arg_val<uint32_t>(6);
    const uint32_t stride = get_arg_val<uint32_t>(7);
    const uint32_t count = get_arg_val<uint32_t>(8);
    const uint32_t scratch = get_write_ptr(5);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t channel_tile = first + item * stride;
        cb_reserve_back(0, 4);
        cb_reserve_back(2, 4);
        const uint32_t window = get_write_ptr(0);
        // Inactive batch rows must remain zero across every CB wrap.
        auto* zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window);
        for (uint32_t i = 0; i < 2048; ++i) {
            zeros[i] = 0;
        }
        noc_async_read_tile(channel_tile, tap0, get_write_ptr(2));
        noc_async_read_tile(channel_tile, tap1, get_write_ptr(2) + 2048);
        noc_async_read_tile(channel_tile, tap2, get_write_ptr(2) + 4096);
        noc_async_read_tile(channel_tile, tap3, get_write_ptr(2) + 6144);
        for (uint32_t user = 0; user < batch; ++user) {
            for (uint32_t tap = 0; tap < 3; ++tap) {
                noc_async_read(
                    history.get_noc_addr(user * 3 + tap, channel_tile * 64), window + tap * 2048 + user * 64, 64);
            }
            const uint32_t tile = compact ? channel_tile : user * input_tiles + channel_tile;
            const uint32_t row = compact ? user : 0;
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t offset = (row / 16) * 1024 + face * 512 + (row % 16) * 32;
                // A compact odd row starts at +32. Read its aligned pair so
                // source and L1 destination obey Blackhole's 64-byte rule.
                noc_async_read(input.get_noc_addr(tile, offset & ~63u), scratch + user * 128 + face * 64, 64);
            }
        }
        noc_async_read_barrier();
        for (uint32_t user = 0; user < batch; ++user) {
            const uint32_t parity = compact ? user % 2 : 0;
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window + 6144 + user * 64);
            for (uint32_t face = 0; face < 2; ++face) {
                const auto* src =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + user * 128 + face * 64 + parity * 32);
                for (uint32_t word = 0; word < 8; ++word) {
                    dst[face * 8 + word] = src[word];
                }
            }
        }
        // Every old history row is in L1 before the first mutation. Different
        // channel workers own disjoint, aligned 64-byte pieces of each row.
        for (uint32_t user = 0; user < batch; ++user) {
            for (uint32_t tap = 0; tap < 3; ++tap) {
                noc_async_write(
                    window + (tap + 1) * 2048 + user * 64, history.get_noc_addr(user * 3 + tap, channel_tile * 64), 64);
            }
        }
        noc_async_write_barrier();
        cb_push_back(2, 4);
        cb_push_back(0, 4);
    }
}
