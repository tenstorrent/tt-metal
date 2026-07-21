// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bridge worker for the full Host -> H2D -> D2D -> Host pipeline test. Runs on a
// worker grid on the SENDER mesh and fuses the H2D consumer and the D2D producer:
// it drains the H2D backing tensor (fed by the host via forward_to_tensor) into
// the D2D SENDER backing tensor, then triggers the D2D sender service to forward
// it over fabric.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

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

    // 2.0 NoC interface for the bulk DRAM<->L1 copy and the unicast metadata forward.
    // The relay reuses a single CB staging slot's address for both read-dest and
    // write-src, as the legacy code did.
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    CoreLocalMem<uint32_t> scratch(scratch_cb.get_write_ptr());

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
            noc.async_read(h2d_input, scratch, page_size, {.page_id = p}, {});
            noc.async_read_barrier();
            noc.async_write<NocOptions::DEFAULT, page_size>(scratch, d2d_output, page_size, {}, {.page_id = p});
        }
        noc.async_write_barrier();

        // 3. (designated core) forward the metadata the H2D service multicast into
        //    this core's L1 to the D2D sender service core. Unicast NoC write from
        //    an allocated L1 buffer source — valid (not a stack-local).
        if constexpr (metadata_enabled) {
            if (is_metadata_writer != 0) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(h2d_metadata_l1_addr),
                    UnicastEndpoint{},
                    metadata_size_bytes,
                    {},
                    {.noc_x = d2d_service_noc_x, .noc_y = d2d_service_noc_y, .addr = d2d_sender_metadata_l1_addr});
                noc.async_write_barrier();
            }
        }

        // 4. Ack H2D consumption: frees the H2D service to stream the next token.
        noc_semaphore_inc(h2d_consumed_counter_noc, 1);
        noc.async_atomic_barrier();

        // 5. Trigger the D2D forward (data + metadata are now in place).
        noc_semaphore_inc(d2d_data_ready_counter_noc, 1);
        noc.async_atomic_barrier();

        // 6. Wait for the D2D service to confirm the transfer drained, then reset.
        //    Gates the next iteration's step-2 overwrite of the D2D sender backing.
        while (*d2d_consumed_sem == 0) {
            invalidate_l1_cache();
        }
        *d2d_consumed_sem = 0;
    }
}
