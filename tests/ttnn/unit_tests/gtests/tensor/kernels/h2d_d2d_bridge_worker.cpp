// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bridge worker for the full Host -> H2D -> D2D -> Host pipeline test. Runs on a
// worker grid on the SENDER mesh and fuses the H2D consumer and the D2D producer:
// it drains the H2D backing tensor (fed by the host via forward_to_tensor) into
// the D2D SENDER backing tensor, then triggers the D2D sender service to forward
// it over fabric.
//
// Per iteration (ordering is load-bearing — see notes/d2d_test_flow_vs_realistic_workload.md):
//   1. spin on the local H2D data_ready_sem until the H2D service multicast-incs
//      it (the host push landed in the H2D backing tensor), then reset it,
//   2. copy pages [start_page, end_page) H2D backing -> D2D sender backing via a
//      single-slot scratch CB, then barrier so the D2D backing slice is durable,
//   3. (designated core, metadata on) copy the metadata blob the H2D service
//      multicast into THIS worker core's L1 (h2d_metadata_l1_addr) into the D2D
//      sender SERVICE core's L1 (d2d_sender_metadata_l1_addr) via a unicast NoC
//      write, then barrier — must finish before step 5,
//   4. atomic-inc the H2D consumed_counter on the H2D service core: frees the H2D
//      service to stream the next token. Strictly AFTER step 2's barriers (the
//      H2D backing tensor is step 2's read source) and after step 3's read of
//      h2d_metadata_l1_addr,
//   5. atomic-inc the D2D data_ready_counter on the D2D sender service core:
//      triggers the D2D forward. After steps 2 and 3 (data + metadata in place),
//   6. spin on the local D2D consumed_sem until the D2D sender service
//      multicast-incs it (transfer drained over fabric), then reset it. Gates the
//      NEXT iteration's step-2 write so we don't overwrite the D2D sender backing
//      tensor before the service forwarded it.
//
// No nested cross-service waits: the only blocking wait (step 6) depends on the
// D2D drain (driven by the receiver-side consumer), not on H2D — so no cycle.
//
// An empty page range (num_pages < num_workers) is valid: that worker copies
// nothing but still runs the acks/waits so both services' num_workers ack counts
// are satisfied (no deadlock). The metadata writer is independent of the page
// range, so it may be an empty-range core.
//
// CT layout (keep in sync with make_bridge_workload in
// tests/ttnn/unit_tests/gtests/tensor/test_d2d_stream_service.cpp):
//   [0] h2d_data_ready_sem_addr   (local worker-core L1)
//   [1] h2d_input_addr            (H2D backing tensor base)
//   [2] d2d_sender_backing_addr   (D2D sender backing tensor base — same spec)
//   [3] page_size                 (bytes per tensor page)
//   [4] num_iters
//   [5] scratch_cb_index          (single-slot scratch CB)
//   [6] d2d_consumed_sem_addr     (local worker-core L1)
//   [7] metadata_enabled          (0/1)
//   [8] metadata_size_bytes
//   [9] h2d_metadata_l1_addr      (local worker-core L1 — H2D multicast dest / copy source)
//   [10..] TensorAccessorArgs     (one set; H2D backing and D2D sender backing share spec)
//
// RT layout (per worker):
//   [0] start_page                    (inclusive)
//   [1] end_page                      (exclusive)
//   [2] h2d_consumed_counter_addr     (L1 on the H2D service core)
//   [3] h2d_service_noc_x
//   [4] h2d_service_noc_y
//   [5] d2d_data_ready_counter_addr   (L1 on the D2D sender service core)
//   [6] d2d_service_noc_x
//   [7] d2d_service_noc_y
//   [8] is_metadata_writer            (1 only on the designated core)
//   [9] d2d_sender_metadata_l1_addr   (L1 on the D2D sender service core; 0 if unused)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t h2d_data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t h2d_input_addr = get_compile_time_arg_val(1);
constexpr uint32_t d2d_sender_backing_addr = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_iters = get_compile_time_arg_val(4);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t d2d_consumed_sem_addr = get_compile_time_arg_val(6);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(7);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(8);
constexpr uint32_t h2d_metadata_l1_addr = get_compile_time_arg_val(9);
// H2D backing and D2D sender backing share the same spec, so one TensorAccessorArgs
// set is reused below with the two distinct base addresses.
constexpr auto acc_args = TensorAccessorArgs<10>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t h2d_consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t h2d_service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t h2d_service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t d2d_data_ready_counter_addr = get_arg_val<uint32_t>(5);
    const uint32_t d2d_service_noc_x = get_arg_val<uint32_t>(6);
    const uint32_t d2d_service_noc_y = get_arg_val<uint32_t>(7);
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(8);
    const uint32_t d2d_sender_metadata_l1_addr = get_arg_val<uint32_t>(9);

    auto h2d_input = TensorAccessor(acc_args, h2d_input_addr);
    auto d2d_output = TensorAccessor(acc_args, d2d_sender_backing_addr);
    const uint32_t cb_l1 = get_write_ptr(scratch_cb_index);

    volatile tt_l1_ptr uint32_t* h2d_data_ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h2d_data_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* d2d_consumed_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d2d_consumed_sem_addr);
    const uint64_t h2d_consumed_counter_noc =
        get_noc_addr(h2d_service_noc_x, h2d_service_noc_y, h2d_consumed_counter_addr);
    const uint64_t d2d_data_ready_counter_noc =
        get_noc_addr(d2d_service_noc_x, d2d_service_noc_y, d2d_data_ready_counter_addr);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // 1. Wait for the H2D service to signal the host push landed, then reset.
        while (*h2d_data_ready_sem == 0) {
            invalidate_l1_cache();
        }
        *h2d_data_ready_sem = 0;

        // 2. Copy this worker's page range H2D backing -> D2D sender backing.
        for (uint32_t p = start_page; p < end_page; ++p) {
            noc_async_read(h2d_input.get_noc_addr(p), cb_l1, page_size);
            noc_async_read_barrier();
            noc_async_write<page_size>(cb_l1, d2d_output.get_noc_addr(p), page_size);
        }
        noc_async_write_barrier();

        // 3. (designated core) forward the metadata the H2D service multicast into
        //    this core's L1 to the D2D sender service core. Unicast NoC write from
        //    an allocated L1 buffer source — valid (not a stack-local).
        if constexpr (metadata_enabled) {
            if (is_metadata_writer != 0) {
                const uint64_t md_noc = get_noc_addr(d2d_service_noc_x, d2d_service_noc_y, d2d_sender_metadata_l1_addr);
                noc_async_write(h2d_metadata_l1_addr, md_noc, metadata_size_bytes);
                noc_async_write_barrier();
            }
        }

        // 4. Ack H2D consumption: frees the H2D service to stream the next token.
        noc_semaphore_inc(h2d_consumed_counter_noc, 1);
        noc_async_atomic_barrier();

        // 5. Trigger the D2D forward (data + metadata are now in place).
        noc_semaphore_inc(d2d_data_ready_counter_noc, 1);
        noc_async_atomic_barrier();

        // 6. Wait for the D2D service to confirm the transfer drained, then reset.
        //    Gates the next iteration's step-2 overwrite of the D2D sender backing.
        while (*d2d_consumed_sem == 0) {
            invalidate_l1_cache();
        }
        *d2d_consumed_sem = 0;
    }
}
