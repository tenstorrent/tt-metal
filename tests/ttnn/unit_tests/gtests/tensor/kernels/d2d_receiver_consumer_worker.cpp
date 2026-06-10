// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Real receiver-side CONSUMER worker for D2DStreamService tests. Unlike the
// handshake-only placeholder receiver worker, this one stands in for a true
// downstream consumer op: each iteration it reads its assigned page slice of the
// receiver backing tensor (DRAM, filled by the receiver service kernel) and
// copies it into a SEPARATE output tensor (DRAM, same spec), so the host can
// validate the output tensor end-to-end instead of peeking at the backing tensor.
//
// Per iteration:
//   1. spin on the local data_ready_sem until the service multicast-incs it
//      (transfer landed in the backing tensor), then reset it to 0,
//   2. copy pages [start_page, end_page) backing -> output via a single-slot
//      scratch CB, then barrier so the output DRAM is durable,
//   3. atomic-inc consumed_counter on the receiver service core (the service
//      waits for num_workers of these before draining the next transfer).
//
// Optional metadata is NOT touched here: the receiver service already multicast
// the blob into every worker core's L1 at receiver->get_metadata_addr(); the
// host reads that L1 directly after Finish (single-transfer-per-Finish verify
// flow, so the read is race-free).
//
// Page partitioning across the worker grid is computed host-side (start/end RT
// args); a worker with an empty range (start == end) still runs steps 1 and 3 so
// the service's num_workers ack count is always satisfied (no deadlock).
//
// Runs a fixed num_iters then exits so a test can Finish() the workload. Mirrors
// placeholder_d2d_receiver_worker.cpp's sem protocol (spin / reset, no counter).
//
// CT layout (keep in sync with make_receiver_consumer_workload in
// tests/ttnn/unit_tests/gtests/tensor/test_d2d_stream_service.cpp):
//   [0] data_ready_sem_addr  (local worker-core L1)
//   [1] input_tensor_addr    (receiver backing tensor base)
//   [2] output_tensor_addr   (output tensor base — same spec as input)
//   [3] page_size            (bytes per tensor page)
//   [4] num_iters
//   [5] scratch_cb_index     (single-slot scratch CB)
//   [6..] TensorAccessorArgs (one set; input and output share spec, reused with
//                             the two distinct base addresses)
//
// RT layout (per worker):
//   [0] start_page            (inclusive)
//   [1] end_page              (exclusive)
//   [2] consumed_counter_addr (L1 address on the receiver service core)
//   [3] service_noc_x         (physical NoC x of the service core)
//   [4] service_noc_y         (physical NoC y of the service core)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_iters = get_compile_time_arg_val(4);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(5);
// Input and output share the same spec, so one TensorAccessorArgs set is reused
// below with the two distinct base addresses.
constexpr auto acc_args = TensorAccessorArgs<6>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);

    auto input = TensorAccessor(acc_args, input_tensor_addr);
    auto output = TensorAccessor(acc_args, output_tensor_addr);
    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* data_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    const uint64_t consumed_counter_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // 1. Wait for the service to signal the transfer landed, then reset.
        while (*data_ready_sem == 0) {
            invalidate_l1_cache();
        }
        *data_ready_sem = 0;

        // 2. Copy this worker's assigned page range backing -> output through the
        //    single-slot scratch CB. Empty range => loop body skipped.
        for (uint32_t p = start_page; p < end_page; ++p) {
            noc_async_read(input.get_noc_addr(p), cb_l1, page_size);
            noc_async_read_barrier();
            noc_async_write<page_size>(cb_l1, output.get_noc_addr(p), page_size);
        }
        noc_async_write_barrier();

        // 3. Ack into consumed_counter — the service waits for num_workers.
        noc_semaphore_inc(consumed_counter_noc, 1);
        noc_async_atomic_barrier();
    }
}
