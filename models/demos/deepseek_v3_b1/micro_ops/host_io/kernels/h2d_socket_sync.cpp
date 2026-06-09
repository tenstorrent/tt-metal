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

void kernel_main() {
    // RT args (per coord, per worker).
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t start_page = get_arg_val<uint32_t>(3);
    const uint32_t end_page = get_arg_val<uint32_t>(4);

    // TensorAccessorArgs blocks: backing first, then output. The host packs
    // them back-to-back starting at CT-arg index 6.
    constexpr auto backing_accessor_args = TensorAccessorArgs<6>();
    constexpr auto output_accessor_args = TensorAccessorArgs<backing_accessor_args.next_compile_time_args_offset()>();

    auto backing = TensorAccessor(backing_accessor_args, backing_tensor_addr);
    auto output = TensorAccessor(output_accessor_args, output_tensor_addr);

    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* data_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);

    // 1. Wait for the service core to signal a fresh transfer, then reset.
    //    The next service-core multicast-inc will set this back to 1.
    while (*data_ready == 0) {
        invalidate_l1_cache();
    }
    *data_ready = 0;

    // 2. Copy this worker's slice of the backing tensor into the output. Use
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

    // 3. Ack the service core. Exactly one inc per worker per transfer; the
    //    service core's poll checks (cur - last_consumed) == num_workers.
    const uint64_t consumed_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);
    noc_semaphore_inc(consumed_noc, 1);
    noc_async_atomic_barrier();
}
