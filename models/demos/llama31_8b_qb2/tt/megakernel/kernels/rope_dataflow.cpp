// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "zero_l1.hpp"
#include "read_alignment.hpp"
constexpr auto packed_args = TensorAccessorArgs<0>();
constexpr auto cos_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
constexpr auto position_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
void QB2_ENTRY() {
#ifdef READER
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
    DeviceZoneScopedN("HEADS-AND-ROPE-READ");
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(0), 2048);
    const auto cosine = TensorAccessor(cos_args, get_arg_val<uint32_t>(1), 256);
    const auto sine = TensorAccessor(sin_args, get_arg_val<uint32_t>(2), 256);
    const auto position = TensorAccessor(position_args, get_arg_val<uint32_t>(3), 4);
    const uint32_t meta = get_write_ptr(31);
    const uint32_t raw_position = read_scalar_u32(position.get_noc_addr(0), meta);
    const uint32_t pos = raw_position == UINT32_MAX ? 0 : raw_position;
    const uint64_t cosine_row = cosine.get_noc_addr(pos), sine_row = sine.get_noc_addr(pos);
    const uint32_t cosine_scratch = aligned_read_destination(meta + 128, cosine_row);
    const uint32_t sine_scratch = aligned_read_destination(meta + 512, sine_row);
    noc_async_read(cosine_row, cosine_scratch, 256);
    noc_async_read(sine_row, sine_scratch, 256);
    for (uint32_t cb : {0u, 1u, 2u}) {
        zero_l1<4 * 2048>(get_write_ptr(cb));
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
    noc_async_read_barrier();
    // A face's second32B half changes DRAM offset modulo64. Read whole rows
    // into aligned scratch, then place each half in the tiled row locally.
    const auto* cosine_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cosine_scratch);
    const auto* sine_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sine_scratch);
    auto* cosine_tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    auto* sine_tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
    for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t word = 0; word < 8; ++word) {
            cosine_tiles[t * 512 + word] = cosine_words[t * 16 + word];
            cosine_tiles[t * 512 + 128 + word] = cosine_words[t * 16 + 8 + word];
            sine_tiles[t * 512 + word] = sine_words[t * 16 + word];
            sine_tiles[t * 512 + 128 + word] = sine_words[t * 16 + 8 + word];
        }
    }
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
