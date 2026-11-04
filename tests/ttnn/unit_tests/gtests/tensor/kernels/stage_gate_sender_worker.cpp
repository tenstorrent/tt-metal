// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Single-core sender worker for the stage gate tests. The host drives it through
// a small control block in this core's L1:
//   word 0             : go count; chunk i is sent once go > i
//   word 1             : cancel; nonzero makes the worker exit from either wait
//   words 2 + 2i, 3 + 2i: slot id and gate mode for chunk i
//
// Per chunk i:
//   1. wait for go > i,
//   2. fill the sender backing tensor with an iota starting at first_value + i * chunk_elems,
//      so every chunk carries unique values,
//   3. write the metadata record {slot, i, base, 0, mode} (slot and mode at the
//      configured word offsets) to the sender service core,
//   4. bump data_ready on the service core and wait for consumed_sem.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t consumed_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
constexpr uint32_t num_pages = get_compile_time_arg_val(2);
constexpr uint32_t tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t first_value = get_compile_time_arg_val(6);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(7);
constexpr uint32_t slot_word = get_compile_time_arg_val(8);
constexpr uint32_t mode_word = get_compile_time_arg_val(9);
constexpr uint32_t control_addr = get_compile_time_arg_val(10);
constexpr auto backing_tensor_accessor_args = TensorAccessorArgs<11>();

constexpr uint32_t page_elems = tensor_page_size / sizeof(uint32_t);
constexpr uint32_t chunk_elems = num_pages * page_elems;
constexpr uint32_t metadata_words = metadata_size_bytes / sizeof(uint32_t);

void kernel_main() {
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t sender_metadata_l1_addr = get_arg_val<uint32_t>(3);

    auto backing = TensorAccessor(backing_tensor_accessor_args, backing_tensor_addr);
    volatile tt_l1_ptr uint32_t* consumed_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem_addr);
    volatile tt_l1_ptr uint32_t* control = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control_addr);
    const uint64_t data_ready_counter_noc = get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);

    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    const uint32_t cb_l1_addr = scratch_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1_addr);
    CoreLocalMem<uint32_t> scratch_src(cb_l1_addr);

    for (uint32_t i = 0; i < num_chunks; ++i) {
        do {
            invalidate_l1_cache();
            if (control[1] != 0) {
                return;
            }
        } while (control[0] <= i);
        const uint32_t slot = control[2 + 2 * i];
        const uint32_t mode = control[3 + 2 * i];

        const uint32_t base = first_value + i * chunk_elems;
        for (uint32_t p = 0; p < num_pages; ++p) {
            const uint32_t page_base = base + p * page_elems;
            for (uint32_t e = 0; e < page_elems; ++e) {
                scratch[e] = page_base + e;
            }
            noc.async_write(scratch_src, backing, tensor_page_size, {}, {.page_id = p});
            noc.async_write_barrier();
        }

        for (uint32_t w = 0; w < metadata_words; ++w) {
            scratch[w] = 0;
        }
        scratch[1] = i;
        scratch[2] = base;
        scratch[slot_word] = slot;
        scratch[mode_word] = mode;
        noc.async_write(
            scratch_src,
            UnicastEndpoint{},
            metadata_size_bytes,
            {},
            {.noc_x = service_noc_x, .noc_y = service_noc_y, .addr = sender_metadata_l1_addr});
        noc.async_write_barrier();

        noc_semaphore_inc(data_ready_counter_noc, 1);
        noc.async_atomic_barrier();

        while (*consumed_sem == 0) {
            invalidate_l1_cache();
            if (control[1] != 0) {
                return;
            }
        }
        *consumed_sem = 0;
    }
}
