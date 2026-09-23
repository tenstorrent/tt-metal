// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void head_worker_prefetch() {
    constexpr auto table_args = TensorAccessorArgs<HEAD_PREFETCH_HELPER_CT>();
    constexpr auto qkv_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    const uint32_t worker = get_arg_val<uint32_t>(0);
#if !HEAD_PREFETCH_QKV
    if (worker < 8) { return; }
#endif
#if !HEAD_PREFETCH_O
    if (worker >= 8) { return; }
#endif
    const uint32_t bank = worker % 8;
    const uint32_t first = get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT + 1);
    const uint32_t count = get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT + 2);
    const uint32_t ready = get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT + 3);
    const uint32_t pointer = get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT + 4);
    const uint32_t consumed = get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT + 5);
    const uint32_t x = get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT + 6 + 2 * worker);
    const uint32_t y = get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT + 7 + 2 * worker);
    auto* epoch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
    auto* acknowledgement = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed);
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(HEAD_PREFETCH_HELPER_RT), 128);
    const uint32_t storage = get_write_ptr(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pointer) = storage;
    noc_async_write(pointer, get_noc_addr(x, y, pointer), 4);
    noc_async_write_barrier();
    for (uint32_t layer = 0; layer < count; ++layer) {
        noc_semaphore_wait(acknowledgement, *epoch);
        {
            DeviceZoneScopedN("HEAD-WORKER-WEIGHT-PREFETCH");
            noc_async_read(table.get_noc_addr(first + layer), get_write_ptr(31), 128);
            noc_async_read_barrier();
            const auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(31));
            if (worker < 8) {
                const auto weight = TensorAccessor(qkv_args, row[3], 1088);
                noc_async_read<128 * 6 * 1088>(weight.get_noc_addr(bank * 6), storage, 128 * 6 * 1088);
            } else {
                const auto weight = TensorAccessor(o_args, row[2], 1088);
                noc_async_read<32 * 16 * 1088>(weight.get_noc_addr(bank * 16), storage, 32 * 16 * 1088);
            }
            noc_async_read_barrier();
        }
        *epoch = *epoch ^ 1u;
        noc_async_write(ready, get_noc_addr(x, y, ready), 4);
        noc_async_write_barrier();
    }
    // The final consumer must release staging before this reader repurposes
    // CB1 for the terminal vocabulary projection in the same program.
    noc_semaphore_wait(acknowledgement, *epoch);
}
