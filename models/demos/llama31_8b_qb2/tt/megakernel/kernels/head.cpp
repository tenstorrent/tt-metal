// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#if defined(READER) || defined(WRITER)
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef READER
#include "projection_reader.hpp"
#if TINY_PROJECTION_M
#include "compact_rows.hpp"
#endif
#ifdef HEAD_PREFETCH_HELPER
#include "head_prefetch.hpp"
#endif
#endif
constexpr auto input_args = TensorAccessorArgs<0>();
constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto output_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(0);
#ifdef READER
#ifdef HEAD_PREFETCH_HELPER
    head_worker_prefetch();
#endif
#if TINY_PROJECTION_M
    zero_compact_input<64 * 2048>(get_write_ptr(16));
#endif
#if HEAD_EARLY_BLOCKS
    {
        // Exactly one trigger per serial resident invocation. Clear locally
        // before refill; all refill bytes are charged to this token.
        auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(36));
        noc_semaphore_wait(ready, 1);
        noc_semaphore_set(ready, 0);
        DeviceZoneScopedN("HEAD-OWN-WEIGHT-PREFETCH");
        const auto weight = TensorAccessor(weight_args, get_arg_val<uint32_t>(2), 1088);
        prefetch_local_projection_weights<1, 4, 64, 16, 1088>(weight, worker);
    }
#endif
    if (worker == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6)), 8);
        for (uint32_t i = 0; i < 16; ++i) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(4 + 2 * i), get_arg_val<uint32_t>(5 + 2 * i), get_semaphore(7)),
                1);
        }
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7)), 1);
    DeviceZoneScopedN("LM-HEAD-READ");
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(1), 2048);
    const auto weight = TensorAccessor(weight_args, get_arg_val<uint32_t>(2), 1088);
#if PROJECTION_READER > 0
    tuned_stream_projection<0, 1, 4, 64, 128, 16, 1088>(input, weight, worker, 0, 0, HEAD_EARLY_BLOCKS);
#else
    for (uint32_t block = 0; block < 128; block += 4) {
        cb_reserve_back(0, 4);
        cb_reserve_back(1, 256);
        for (uint32_t k = 0; k < 4; ++k) {
            noc_async_read_page(block + k, input, get_write_ptr(0) + k * 2048);
            // Each worker reads half of one bank's contiguous width shard.
            // The current arbitrary-length API packetizes this 69,632B read.
            noc_async_read(
                weight.get_noc_addr((block + k) * 1024 + worker * 64),
                get_write_ptr(1) + k * 64 * 1088,
                64 * 1088);
        }
        noc_async_read_barrier();
        cb_push_back(0, 4);
        cb_push_back(1, 256);
    }
#endif
#else
    DeviceZoneScopedN("LM-HEAD-WRITE");
    cb_wait_front(16, 64);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(3), 2048);
    for (uint32_t tile = 0; tile < 64; ++tile) {
        noc_async_write_page(worker * 64 + tile, output, get_read_ptr(16) + tile * 2048);
    }
    noc_async_write_barrier();
    cb_pop_front(16, 64);
#endif
}
#else
#include "projection.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("LM-HEAD-MATH");
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 24);
    projection<0, 1, 16, 24, 4, 64, 128, FULL_DST_HEAD ? 16 : 8>();
}
#endif
