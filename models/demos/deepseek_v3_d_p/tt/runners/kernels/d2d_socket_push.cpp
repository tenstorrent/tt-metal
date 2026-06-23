// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One-shot worker-side producer for D2DStreamServiceSender — the inverse of
// h2d_socket_sync.cpp. Instead of draining a service-filled backing tensor into an
// output, it fills the sender backing tensor from an input activation and signals the
// persistent sender service that there is data to forward. Re-enqueued per chunk by
// d2d_socket_push_op.py (no outer loop); the producer half mirrors the test relay
// pipeline_relay_worker.cpp.
//
// Per enqueue, each worker:
//   1. Copy its [start_page, end_page) slice of the input tensor into the sender
//      backing tensor (page-by-page via a single-slot scratch CB).
//   2. (Designated worker only, when metadata is enabled) write the metadata words
//      from RT args into the sender service core's L1 metadata buffer, BEFORE acking
//      (the service reads it once it sees num_workers acks).
//   3. Atomic-inc data_ready_counter at the service core (it forwards once it has
//      num_workers acks AND the host grants the fabric lease), then RETURN.
//
// DECOUPLED from the forward on purpose: this models a stage's compute op, which runs
// to completion before the D2D forwards its output. The host drives the cadence
// wait_for_fabric_links -> push -> release_fabric_links, so the sender forwards only
// after this program finishes (release is post-push). Waiting on the sender's
// consumed_sem here would deadlock that order. Not overwriting the backing before the
// prior forward drained is the host's job: it wait_for_fabric_links() on this boundary
// before the next push.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

// CT-arg layout (must stay in sync with d2d_socket_push_op.py).
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(0);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_pages = get_compile_time_arg_val(3);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(4);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(5);  // 0 disables the metadata path

// TensorAccessorArgs: backing, then input, packed back-to-back from CT-arg index 6.
constexpr auto backing_accessor_args = TensorAccessorArgs<6>();
constexpr auto input_accessor_args = TensorAccessorArgs<backing_accessor_args.next_compile_time_args_offset()>();

void kernel_main() {
    // RT args (per coord, per worker).
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t start_page = get_arg_val<uint32_t>(3);
    const uint32_t end_page = get_arg_val<uint32_t>(4);
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(5);       // 1 only on the designated core
    const uint32_t sender_metadata_l1_addr = get_arg_val<uint32_t>(6);  // service-core L1 metadata buffer
    // metadata words (metadata_size_bytes / 4 of them) follow at RT-arg index 7; only read on the writer.

    auto backing = TensorAccessor(backing_accessor_args, backing_tensor_addr);
    auto input = TensorAccessor(input_accessor_args, input_tensor_addr);

    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    // 1. Copy this worker's slice of the input into the sender backing tensor. Use the
    //    scratch CB as a per-page staging area; read_barrier before write so we never
    //    write off an unwritten L1 region, write_barrier before reusing the slot.
    for (uint32_t p = start_page; p < end_page; ++p) {
        noc_async_read(input.get_noc_addr(p), cb_l1, page_size);
        noc_async_read_barrier();
        noc_async_write(cb_l1, backing.get_noc_addr(p), page_size);
        noc_async_write_barrier();
    }

    // 2. (Designated worker) stage the metadata words in the scratch CB (free now — the
    //    data writes are flushed) and write them to the service core's metadata L1.
    if constexpr (metadata_size_bytes > 0) {
        if (is_metadata_writer != 0) {
            volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1);
            for (uint32_t w = 0; w < metadata_size_bytes / 4; ++w) {
                scratch[w] = get_arg_val<uint32_t>(7 + w);
            }
            const uint64_t md_noc = get_noc_addr(service_noc_x, service_noc_y, sender_metadata_l1_addr);
            noc_async_write(cb_l1, md_noc, metadata_size_bytes);
            noc_async_write_barrier();
        }
    }

    // 3. Ack into data_ready_counter — the service forwards the backing tensor (and the
    //    metadata) over fabric once it has num_workers of these AND the host grants the
    //    lease. Return without waiting for the forward (the host drives release/wait).
    const uint64_t data_ready_noc = get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);
    noc_semaphore_inc(data_ready_noc, 1);
    noc_async_atomic_barrier();
}
