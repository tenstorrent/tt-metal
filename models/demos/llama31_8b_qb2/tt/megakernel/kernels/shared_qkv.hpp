// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(READER) || defined(WRITER)
constexpr auto shared_qkv_weight_args = TensorAccessorArgs<SHARED_QKV_CT>();
constexpr auto shared_qkv_packed_args = TensorAccessorArgs<shared_qkv_weight_args.next_compile_time_args_offset()>();
// Projection-local semaphores8,9,12 are unused by O/GU/down.
// Norm workers use their own local8/9 independently. No new hardware slots.
#ifdef READER
#include "projection_reader.hpp"
void shared_qkv_read(uint32_t bank, uint32_t weight_address) {
    if (bank == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8)), 8);
        release_workers(9, 0, 8);
    }
    wait_phase(9);
    DeviceZoneScopedN("QKV-SHARED-READ");
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(1), 2048);
    const auto weight = TensorAccessor(shared_qkv_weight_args, weight_address, 1088);
    tuned_stream_projection<8, 9, 16, 6, 128, 8, 1088>(input, weight, bank);
}
#else
void shared_qkv_write(uint32_t bank) {
    cb_wait_front(27, 6);
    const auto packed = TensorAccessor(shared_qkv_packed_args, get_arg_val<uint32_t>(SHARED_QKV_RT), 2048);
    for (uint32_t tile = 0; tile < 6; ++tile) {
        noc_async_write_page(bank * 6 + tile, packed, get_read_ptr(27) + tile * 2048);
    }
    noc_async_write_barrier();
    cb_pop_front(27, 6);
    notify_coordinator(12);
    noc_async_atomic_barrier();
    if (bank == 0) {
        DeviceZoneScopedN("QKV-SHARED-READY");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12)), 8);
        // All packed shards are visible before Q/K RoPE and V-cache readers.
        for (uint32_t i = 0; i < 3; ++i) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(SHARED_QKV_RT + 1 + 2 * i),
                get_arg_val<uint32_t>(SHARED_QKV_RT + 2 + 2 * i), get_semaphore(0)), 1);
        }
        noc_async_atomic_barrier();
    }
}
#endif
#else
void shared_qkv_compute() {
    DeviceZoneScopedN("QKV-SHARED-MATH");
    compute_kernel_hw_startup<SrcOrder::Reverse>(8, 9, 26);
    projection<8, 9, 27, 26, 16, 6, 128, 6>();
}
#endif
