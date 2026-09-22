// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
constexpr auto packed_args = TensorAccessorArgs<0>();
constexpr auto cos_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
constexpr auto position_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
void kernel_main() {
#ifdef READER
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
    DeviceZoneScopedN("HEADS-AND-ROPE-READ");
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(0), 2048);
    const auto cosine = TensorAccessor(cos_args, get_arg_val<uint32_t>(1), 256);
    const auto sine = TensorAccessor(sin_args, get_arg_val<uint32_t>(2), 256);
    const auto position = TensorAccessor(position_args, get_arg_val<uint32_t>(3), 4);
    const uint32_t meta = get_write_ptr(31);
    noc_async_read(position.get_noc_addr(0), meta, 4);
    noc_async_read_barrier();
    const uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    for (uint32_t cb : {0u, 1u, 2u}) {
        auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
        for (uint32_t i = 0; i < 4 * 2048 / 4; ++i) { p[i] = 0; }
    }
#ifdef QUERY
    constexpr uint32_t heads = 8, first_tile = 0;
#else
    constexpr uint32_t heads = 2, first_tile = 32;
#endif
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t t = 0; t < 4; ++t) {
            const uint64_t source = packed.get_noc_addr(first_tile + h * 4 + t);
            const uint32_t target = get_write_ptr(0) + t * 2048 + h * 32;
            noc_async_read(source, target, 32);
            noc_async_read(source + 512, target + 512, 32);
        }
    }
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(cosine.get_noc_addr(pos) + t * 64, get_write_ptr(1) + t * 2048, 32);
        noc_async_read(cosine.get_noc_addr(pos) + t * 64 + 32, get_write_ptr(1) + t * 2048 + 512, 32);
        noc_async_read(sine.get_noc_addr(pos) + t * 64, get_write_ptr(2) + t * 2048, 32);
        noc_async_read(sine.get_noc_addr(pos) + t * 64 + 32, get_write_ptr(2) + t * 2048 + 512, 32);
    }
    noc_async_read_barrier();
    cb_reserve_back(3, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(3)) = 0xbf80;
    cb_push_back(3, 1);
#else
    cb_wait_front(16, 4);
#ifdef QUERY
    // Query plus the two completed cache writes form the SDPA dependency.
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5), get_semaphore(1)), 1);
    noc_async_atomic_barrier();
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 3);
    for (uint32_t i = 0; i < 32; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(6 + 2 * i), get_arg_val<uint32_t>(7 + 2 * i), get_semaphore(4)), 1);
    }
#else
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5), get_semaphore(0)), 1);
#endif
    noc_async_atomic_barrier();
#endif
}
