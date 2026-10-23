// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stand-in for the MoE reduce launch around BankStreamMatmul.section (kagent_qkv_prefetch_*.py). On one core outside
// the section: raise start_sem of every stream core start_count times by one (the fabric senders, once they no longer
// read DRAM); load the user rows (0-3) of x_src into L1 (with face 0's rows 4-15 between faces 0 and 1, one 640 B
// packet a tile), and once delay_cycles have passed write them into x (the reduce writing its norm; they hold other
// data until then) and raise x_sem of every loader x_count times by one (the reduce cores, x written). QPS marks
// the phases in a device profile.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t start_sem = get_compile_time_arg_val(0);
    constexpr uint32_t start_count = get_compile_time_arg_val(1);
    constexpr uint32_t x_sem = get_compile_time_arg_val(2);
    constexpr uint32_t x_count = get_compile_time_arg_val(3);
    constexpr uint32_t delay_cycles = get_compile_time_arg_val(4);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t chunk_bytes = 640;  // face 0 and rows 0-3 of face 1

    uint32_t arg = 0;
    const uint32_t x_src = get_arg_val<uint32_t>(arg++);
    const uint32_t x_dst = get_arg_val<uint32_t>(arg++);
    const uint32_t n_stream = get_arg_val<uint32_t>(arg++);
    const uint32_t stream_at = arg;
    arg += 2 * n_stream;
    const uint32_t n_loaders = get_arg_val<uint32_t>(arg++);
    const uint32_t loaders_at = arg;

    const uint64_t t0 = get_timestamp();
    DeviceTimestampedData("QPS", 0);
    const uint32_t start = get_semaphore(start_sem);
    for (uint32_t k = 0; k < start_count; k++) {
        for (uint32_t c = 0; c < n_stream; c++) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(stream_at + 2 * c), get_arg_val<uint32_t>(stream_at + 1 + 2 * c), start),
                1);
        }
    }

    const InterleavedAddrGen<true> src = {.bank_base_address = x_src, .page_size = tile_bytes};
    const InterleavedAddrGen<true> dst = {.bank_base_address = x_dst, .page_size = tile_bytes};
    const uint32_t scratch = get_write_ptr(cb_scratch);
    for (uint32_t t = 0; t < x_tiles; t++) {
        noc_async_read(src.get_noc_addr(t, 0), scratch + t * chunk_bytes, chunk_bytes);
    }
    noc_async_read_barrier();
    DeviceTimestampedData("QPS", 1);

    while (get_timestamp() - t0 < delay_cycles) {
    }
    DeviceTimestampedData("QPS", 2);
    for (uint32_t t = 0; t < x_tiles; t++) {
        noc_async_write(scratch + t * chunk_bytes, dst.get_noc_addr(t, 0), chunk_bytes);
    }
    noc_async_write_barrier();
    DeviceTimestampedData("QPS", 3);

    const uint32_t ready = get_semaphore(x_sem);
    for (uint32_t k = 0; k < x_count; k++) {
        for (uint32_t c = 0; c < n_loaders; c++) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(loaders_at + 2 * c), get_arg_val<uint32_t>(loaders_at + 1 + 2 * c), ready),
                1);
        }
    }
    noc_async_atomic_barrier();
    DeviceTimestampedData("QPS", 4);
}
