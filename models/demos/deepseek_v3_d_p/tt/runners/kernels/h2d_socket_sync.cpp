// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One-shot worker-side handshake for H2DStreamService.
//
// Implements steps 4 and 5 of the per-iteration protocol from
// h2d_stream_service.md:
//   1. Wait on the local data_ready_sem until > 0 (set by the service core's
//      multicast-inc), then reset it to 0.
//   2. Copy this worker's [start_page, end_page) slice of the backing tensor
//      into the output tensor (page-by-page, via a single-slot scratch CB).
//   3. Atomic-inc the consumed_counter at the service core's physical NoC
//      coords. The service core polls for exactly `num_workers` acks per
//      transfer before reading the next transfer.
//
// Re-enqueued per `forward_to_tensor_bytes` call — no outer loop. Persistent
// variant lives in persistent_h2d_receiver.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

// CT-arg layout (must stay in sync with h2d_socket_sync_op.py).
constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_pages = get_compile_time_arg_val(4);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(5);
// Optional metadata path. When `metadata_size_bytes == 0`, the metadata read
// is compiled out entirely and the trailing metadata TensorAccessorArgs are
// not consumed; the kernel behaves byte-identically to the no-metadata case.
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(6);
constexpr uint32_t metadata_l1_addr = get_compile_time_arg_val(7);
constexpr uint32_t metadata_output_addr = get_compile_time_arg_val(8);

void kernel_main() {
    // RT args (per coord, per worker).
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t start_page = get_arg_val<uint32_t>(3);
    const uint32_t end_page = get_arg_val<uint32_t>(4);

    // TensorAccessorArgs blocks: backing, output, then (when metadata is
    // enabled) metadata output. The host packs them back-to-back starting at
    // CT-arg index 9.
    constexpr auto backing_accessor_args = TensorAccessorArgs<9>();
    constexpr auto output_accessor_args = TensorAccessorArgs<backing_accessor_args.next_compile_time_args_offset()>();

    auto backing = TensorAccessor(backing_accessor_args, backing_tensor_addr);
    auto output = TensorAccessor(output_accessor_args, output_tensor_addr);

    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* data_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);

    // 1. Wait for the service core to signal a fresh transfer, then reset.
    //    The next service-core multicast-inc will set this back to 1. The
    //    service core multicasts metadata into L1 BEFORE flipping this sem,
    //    so by the time we observe data_ready_sem > 0 the metadata bytes at
    //    `metadata_l1_addr` are already valid.
    while (*data_ready == 0) {
        invalidate_l1_cache();
    }
    *data_ready = 0;

    // 2. (Optional) Snapshot metadata from this worker's local L1 into the
    //    metadata output tensor on DRAM. Gated on `start_page == 0` so a
    //    multi-worker run only emits one write per coord — every worker's L1
    //    holds the same metadata (multicast), so picking one is fine.
    if constexpr (metadata_size_bytes > 0) {
        if (start_page == 0) {
            constexpr auto metadata_accessor_args =
                TensorAccessorArgs<output_accessor_args.next_compile_time_args_offset()>();
            auto metadata_out = TensorAccessor(metadata_accessor_args, metadata_output_addr);
            // Single-page write: the metadata tensor is sized to exactly one
            // page of `metadata_size_bytes` bytes (see h2d_socket_sync_op.py).
            noc_async_write(metadata_l1_addr, metadata_out.get_noc_addr(0), metadata_size_bytes);
            noc_async_write_barrier();
        }
    }

    // 3. Copy this worker's slice of the backing tensor into the output. Use
    //    the scratch CB as a per-page staging area. read_barrier before write
    //    so we don't issue a write off an unwritten L1 region; write_barrier
    //    before reusing the slot for the next page.
    for (uint32_t p = start_page; p < end_page; ++p) {
        noc_async_read(backing.get_noc_addr(p), cb_l1, page_size);
        noc_async_read_barrier();

        const uint64_t out_noc = output.get_noc_addr(p);
        noc_async_write(cb_l1, out_noc, page_size);
        noc_async_write_barrier();
    }

    // 4. Ack the service core. Exactly one inc per worker per transfer; the
    //    service core's poll checks (cur - last_consumed) == num_workers.
    const uint64_t consumed_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);
    noc_semaphore_inc(consumed_noc, 1);
    noc_async_atomic_barrier();
}
