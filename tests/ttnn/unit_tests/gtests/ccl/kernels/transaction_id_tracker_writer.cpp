
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/transaction_id_tracker.hpp"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t num_transaction_ids = get_compile_time_arg_val(0);
constexpr bool use_transaction_ids = num_transaction_ids != 0;

constexpr bool is_dram = get_compile_time_arg_val(1) == 1;
constexpr bool do_write_barrier = get_compile_time_arg_val(2) == 1;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    uint32_t cb0_id = get_arg_val<uint32_t>(arg_idx++);
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    auto page_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    auto num_pages = get_arg_val<uint32_t>(arg_idx++);
    auto num_per_push_page_counts = get_arg_val<uint32_t>(arg_idx++);
    uint32_t* pages_per_push_array = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    for (size_t i = 0; i < num_per_push_page_counts; i++) {
        DPRINT << "pages_per_push_array[" << (uint32_t)i << "] = " << (uint32_t)pages_per_push_array[i] << ENDL();
    }
    arg_idx += num_per_push_page_counts;

    auto trid_tracker = TransactionIdTracker<num_transaction_ids>(cb0_id);
    auto addrgen = InterleavedAddrGen<is_dram>{.bank_base_address = tensor_address0, .page_size = page_size_bytes};

    DPRINT << "Waiting for initial pages before starting and profiling main loop\n";
    while (!cb_pages_available_at_front(cb0_id, pages_per_push_array[0])) {
    };

    DPRINT << "Initial pages available\n";

    {
        DeviceZoneScopedN("MAIN-LOOP");
        size_t pages_written = 0;
        size_t pages_per_push_idx = 0;
        while (pages_written < num_pages) {
            size_t pages_to_write = pages_per_push_array[pages_per_push_idx];
            if constexpr (use_transaction_ids) {
                if constexpr (do_write_barrier) {
                    if (trid_tracker.has_unflushed_trid() && trid_tracker.oldest_write_trid_flushed()) {
                        DPRINT << "Popping pages for oldest trid\n";
                        trid_tracker.pop_pages_for_oldest_trid();
                    }
                } else {
                    if (trid_tracker.has_unflushed_trid() && trid_tracker.oldest_write_trid_sent()) {
                        DPRINT << "Popping pages for oldest trid\n";
                        trid_tracker.pop_pages_for_oldest_trid();
                    }
                }

                if (trid_tracker.next_cb_read_slot_is_available(pages_to_write)) {
                    DPRINT << "Getting next cb read slot\n";
                    pages_per_push_idx = wrap_increment<size_t>(pages_per_push_idx, num_per_push_page_counts);

                    auto [l1_read_addr, trid] = trid_tracker.get_next_cb_read_slot(pages_to_write);

                    for (size_t i = 0; i < pages_to_write; i++) {
                        uint64_t write_addr = addrgen.get_noc_addr(pages_written + i);
                        noc_async_write_one_packet(l1_read_addr, write_addr, page_size_bytes);
                        l1_read_addr += page_size_bytes;
                    }

                    pages_written += pages_to_write;
                }
            } else {
                pages_per_push_idx = wrap_increment<size_t>(pages_per_push_idx, num_per_push_page_counts);
                cb_wait_front(cb0_id, pages_to_write);

                auto l1_read_addr = get_read_ptr(cb0_id);
                for (size_t i = 0; i < pages_to_write; i++) {
                    uint64_t write_addr = addrgen.get_noc_addr(pages_written + i);
                    noc_async_write_one_packet(l1_read_addr, write_addr, page_size_bytes);
                    l1_read_addr += page_size_bytes;
                }

                if constexpr (do_write_barrier) {
                    noc_async_write_barrier();
                } else {
                    noc_async_writes_flushed();
                }
                cb_pop_front(cb0_id, pages_to_write);
                pages_written += pages_to_write;
            }
        }
        if constexpr (use_transaction_ids) {
            trid_tracker.write_barrier();
        } else {
            noc_async_write_barrier();
        }
    }
}
