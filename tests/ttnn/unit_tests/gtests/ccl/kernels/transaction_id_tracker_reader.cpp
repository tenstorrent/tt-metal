
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
static_assert(!use_transaction_ids, "Transaction IDs are not supported on reader side");
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
    bool perf_mode = get_arg_val<uint32_t>(arg_idx++) != 0;
    uint32_t* pages_per_push_array = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_per_push_page_counts;

    auto trid_tracker = TransactionIdTracker<num_transaction_ids>(cb0_id);
    auto addrgen = InterleavedAddrGen<is_dram>{.bank_base_address = tensor_address0, .page_size = page_size_bytes};

    size_t pages_read = 0;
    size_t pages_per_push_idx = 0;

    while (pages_read < num_pages) {
        size_t pages_to_read = pages_per_push_array[pages_per_push_idx];
        if constexpr (use_transaction_ids) {
            if (trid_tracker.has_unflushed_trid() && trid_tracker.oldest_read_trid_flushed()) {
                DPRINT << "Pushing pages for oldest trid" << ENDL();
                trid_tracker.push_pages_for_oldest_trid();
            }
            if (!trid_tracker.backpressured() && trid_tracker.next_cb_write_slot_is_available(pages_to_read)) {
                DPRINT << "Getting next cb write slot" << ENDL();
                auto [l1_write_addr, trid] = trid_tracker.get_next_cb_write_slot(pages_to_read);
                ncrisc_noc_set_transaction_id(noc_index, read_cmd_buf, trid);
                pages_per_push_idx = wrap_increment<size_t>(pages_per_push_idx, num_per_push_page_counts);

                for (size_t i = 0; i < pages_to_read; i++) {
                    uint64_t read_addr = addrgen.get_noc_addr(pages_read);
                    ncrisc_noc_fast_read_with_transaction_id<noc_mode>(
                        noc_index, read_cmd_buf, tensor_address0, read_addr, l1_write_addr, trid);
                    l1_write_addr += page_size_bytes;
                }

                pages_read += pages_to_read;
            }
        } else {
            pages_per_push_idx = wrap_increment<size_t>(pages_per_push_idx, num_per_push_page_counts);
            cb_reserve_back(cb0_id, pages_to_read);

            auto l1_write_addr = get_write_ptr(cb0_id);
            for (size_t i = 0; i < pages_to_read; i++) {
                uint64_t read_addr = addrgen.get_noc_addr(pages_read + i);
                // For now if running with perf mode we skip the read so that we can keep up with the write side
                // (which properly supports trids)
                if (!perf_mode) {
                    noc_async_read_one_packet(read_addr, l1_write_addr, page_size_bytes);
                }
                l1_write_addr += page_size_bytes;
            }

            noc_async_read_barrier();
            cb_push_back(cb0_id, pages_to_read);
            pages_read += pages_to_read;
        }
    }
    noc_async_read_barrier();
}
