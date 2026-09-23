// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#if defined(READER) || defined(WRITER)
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef READER
#include "projection_reader.hpp"
#if TINY_PROJECTION_M
#include "compact_rows.hpp"
#endif
#ifdef HEAD_PREFETCH_RECEIVER
#include "prefetch_receiver.hpp"
#endif
#endif
constexpr auto input_args = TensorAccessorArgs<0>();
constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto table_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
uint64_t peer(uint32_t i, uint32_t address) {
    return get_noc_addr(get_arg_val<uint32_t>(4 + 2 * i), get_arg_val<uint32_t>(5 + 2 * i), address);
}
void QB2_ENTRY() {
    const uint32_t bank = get_arg_val<uint32_t>(0);
#ifdef READER
#if TINY_PROJECTION_M
    if (initialize_layer_scratch(1)) { zero_compact_input<6 * 2048>(get_write_ptr(16)); }
#endif
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(2), 128);
    noc_async_read(table.get_noc_addr(get_arg_val<uint32_t>(3)), get_write_ptr(31), 128);
    noc_async_read_barrier();
    const uint32_t weight_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(31))[3];
#if EARLY_WEIGHT_BLOCKS && (EARLY_WEIGHT_PHASES & 1)
    {
        DeviceZoneScopedN("QKV-LOCAL-WEIGHT-PREFETCH");
        const auto weight = TensorAccessor(weight_args, weight_address, 1088);
        prefetch_local_projection_weights<1, 16, 6, 8, 1088>(weight, bank);
    }
#endif
    if (bank == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6)), 8);
        for (uint32_t c = 0; c < 8; ++c) { noc_semaphore_inc(peer(c, get_semaphore(7)), 1); }
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7)), 1);
    DeviceZoneScopedN("QKV-READ");
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(1), 2048);
    const auto weight = TensorAccessor(weight_args, weight_address, 1088);
#if PROJECTION_READER > 0
#ifdef HEAD_PREFETCH_RECEIVER
    const uint64_t staging = wait_head_prefetch(bank);
    tuned_stream_projection<0, 1, 16, 6, 128, 8, 1088>(input, weight, bank, staging, 8);
    finish_head_prefetch(bank);
#else
    tuned_stream_projection<0, 1, 16, 6, 128, 8, 1088>(input, weight, bank, 0, 0,
        (EARLY_WEIGHT_PHASES & 1) ? EARLY_WEIGHT_BLOCKS : 0);
#endif
#else
    for (uint32_t block = 0; block < 128; block += 16) {
        cb_reserve_back(0, 16);
        cb_reserve_back(1, 96);
        for (uint32_t k = 0; k < 16; ++k) {
            noc_async_read_page(block + k, input, get_write_ptr(0) + k * 2048);
            noc_async_read(weight.get_noc_addr((block + k) * 48 + bank * 6), get_write_ptr(1) + k * 6 * 1088, 6 * 1088);
        }
        noc_async_read_barrier();
        cb_push_back(0, 16);
        cb_push_back(1, 96);
    }
#endif
#else
    cb_wait_front(16, 6);
    noc_semaphore_inc(peer(0, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    if (bank == 0) {
        DeviceZoneScopedN("QKV-READY");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 8);
        // Q RoPE, K RoPE and V-cache reader. K-cache starts after K RoPE.
        for (uint32_t i = 8; i < 11; ++i) { noc_semaphore_inc(peer(i, get_semaphore(0)), 1); }
#ifdef HEAD_WEIGHT_TRIGGER_RT
        if (get_arg_val<uint32_t>(3) == get_arg_val<uint32_t>(HEAD_WEIGHT_TRIGGER_RT + 1)) {
            const uint32_t ready = get_arg_val<uint32_t>(HEAD_WEIGHT_TRIGGER_RT);
            for (uint32_t i = 0; i < 16; ++i) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(HEAD_WEIGHT_TRIGGER_RT + 2 + 2*i),
                    get_arg_val<uint32_t>(HEAD_WEIGHT_TRIGGER_RT + 3 + 2*i), ready), 1);
            }
        }
#endif
        noc_async_atomic_barrier();
    }
#endif
}
#else
#include "projection.hpp"
#include "tools/profiler/kernel_profiler.hpp"
void QB2_ENTRY() {
    DeviceZoneScopedN("QKV-MATH");
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 24);
    projection<0, 1, 16, 24, 16, 6, 128, 6>();
}
#endif
