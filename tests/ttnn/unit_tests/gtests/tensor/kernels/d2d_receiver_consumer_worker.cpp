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

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

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

    // 2.0 NoC interface for the bulk DRAM<->L1 copy. The relay reuses a single CB
    // staging slot's address as both read-dest and write-src, as the legacy code did.
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    CoreLocalMem<uint32_t> scratch(scratch_cb.get_write_ptr());

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
            noc.async_read(input, scratch, page_size, {.page_id = p}, {});
            noc.async_read_barrier();
            noc.async_write<NocOptions::DEFAULT, page_size>(scratch, output, page_size, {}, {.page_id = p});
        }
        noc.async_write_barrier();

        // 3. Ack into consumed_counter — the service waits for num_workers.
        noc_semaphore_inc(consumed_counter_noc, 1);
        noc.async_atomic_barrier();
    }
}
