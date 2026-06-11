// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Gate-free producer op for the cross-process D2D chain (pipeline head). Stands
// in for the model graph's compute op at stage 0: it writes one iteration's iota
// into the outbound D2DStreamServiceSender backing tensor and incs the sender
// service's data_ready_counter — i.e. the op OWNS the data_ready signal.
//
// Crucially it does NOT wait the sender's consumed_sem: that overwrite-gate is a
// SEPARATE op (d2d_sync.cpp), the unbundled split that mirrors a real graph. So
// the host must launch the d2d_sync gate before this op on every iter except the
// first. One iteration per launch; the host passes a fresh fill_base each iter.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(0);
constexpr uint32_t tensor_page_size = get_compile_time_arg_val(1);
constexpr uint32_t fill_base = get_compile_time_arg_val(2);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(3);
constexpr auto backing_tensor_accessor_args = TensorAccessorArgs<4>();

void kernel_main() {
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t start_page = get_arg_val<uint32_t>(3);
    const uint32_t end_page = get_arg_val<uint32_t>(4);

    auto backing = TensorAccessor(backing_tensor_accessor_args, backing_tensor_addr);
    const uint32_t cb_l1_addr = get_write_ptr(scratch_cb_index);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1_addr);
    const uint32_t page_elems = tensor_page_size / sizeof(uint32_t);

    // Produce this worker's page range as a global iota: row-major element i holds
    // fill_base + i (distinct per element; shifted by 1 each iter so a stale /
    // unforwarded transfer reads back the wrong base).
    for (uint32_t p = start_page; p < end_page; ++p) {
        const uint32_t page_base = fill_base + p * page_elems;
        for (uint32_t e = 0; e < page_elems; ++e) {
            scratch[e] = page_base + e;
        }
        noc_async_write(cb_l1_addr, backing.get_noc_addr(p), tensor_page_size);
        noc_async_write_barrier();
    }

    // Signal: the op owns the data_ready_counter inc. The sender service forwards
    // once it sees num_workers acks AND it has been granted its lease turn.
    const uint64_t data_ready_counter_noc = get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);
    noc_semaphore_inc(data_ready_counter_noc, 1);
    noc_async_atomic_barrier();
}
