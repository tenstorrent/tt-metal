// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "zero_l1.hpp"
#include "read_alignment.hpp"
constexpr auto input_args = TensorAccessorArgs<0>();
constexpr auto cache_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
constexpr auto page_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
void QB2_ENTRY() {
    const auto cache = TensorAccessor(cache_args, get_arg_val<uint32_t>(1), 1088);
#ifdef READER
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
    DeviceZoneScopedN("PAGED-KV-READ");
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0), 2048);
    const auto position = TensorAccessor(pos_args, get_arg_val<uint32_t>(2), 4);
    const auto pages = TensorAccessor(page_args, get_arg_val<uint32_t>(3), PAGE_BYTES);
    cb_reserve_back(31, 1);
    const uint32_t metadata = get_write_ptr(31);
    const uint32_t pos = read_scalar_u32(position.get_noc_addr(0), metadata + 128);
    const uint32_t physical = pos == UINT32_MAX ? 0 :
        read_scalar_u32(pages.get_noc_addr(0) + (pos / 128) * 4, metadata + 128);
    auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata);
    meta[0] = pos;
    meta[1] = physical;
    cb_push_back(31, 1);
    // Compute and writer own separate position CBs. read_tile_value forwards
    // this flag through mailboxes so every TRISC takes the same inactive path.
    cb_reserve_back(30, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(30)) = pos;
    cb_push_back(30, 1);
    if (pos == UINT32_MAX) { return; }
    cb_reserve_back(1, 4);
#ifdef VALUE
    zero_l1<4 * 2048>(get_write_ptr(1));
    for (uint32_t h = 0; h < 2; ++h) {
        for (uint32_t t = 0; t < 4; ++t) {
            const uint64_t source = input.get_noc_addr(40 + h * 4 + t);
            const uint32_t target = get_write_ptr(1) + t * 2048 + h * 32;
            noc_async_read(source, target, 32);
            noc_async_read(source + 512, target + 512, 32);
        }
    }
#else
    for (uint32_t t = 0; t < 4; ++t) { noc_async_read_page(t, input, get_write_ptr(1) + t * 2048); }
#endif
    noc_async_read_barrier();
    cb_push_back(1, 4);
    for (uint32_t h = 0; h < 2; ++h) {
        cb_reserve_back(0, 4);
        const uint32_t first = physical * 32 + h * 16 + ((pos % 128) / 32) * 4;
        for (uint32_t t = 0; t < 4; ++t) { noc_async_read_page(first + t, cache, get_write_ptr(0) + t * 1088); }
        noc_async_read_barrier();
        cb_push_back(0, 4);
    }
#else
    cb_wait_front(31, 1);
    auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(31));
    const uint32_t pos = meta[0], physical = meta[1];
    if (pos == UINT32_MAX) {
        cb_pop_front(31, 1);
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5), get_semaphore(1)), 1);
        noc_async_atomic_barrier();
        return;
    }
    cb_wait_front(16, 4);
    DeviceZoneScopedN("PAGED-KV-UPDATE");
    for (uint32_t h = 0; h < 2; ++h) {
        cb_wait_front(24, 4);
        cb_reserve_back(25, 4);
        noc_async_read(get_noc_addr(get_read_ptr(16) + h * 256), get_read_ptr(24) + (pos % 32) * 256, 256);
        noc_async_read_barrier();
        cb_push_back(25, 4);
        cb_pop_front(24, 4);
        cb_wait_front(8, 4);
        const uint32_t first = physical * 32 + h * 16 + ((pos % 128) / 32) * 4;
        for (uint32_t t = 0; t < 4; ++t) { noc_async_write_page(first + t, cache, get_read_ptr(8) + t * 1088); }
        noc_async_write_barrier();
        cb_pop_front(8, 4);
    }
    cb_pop_front(16, 4);
    cb_pop_front(31, 1);
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5), get_semaphore(1)), 1);
    noc_async_atomic_barrier();
#endif
}
