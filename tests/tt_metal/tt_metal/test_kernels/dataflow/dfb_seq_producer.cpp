// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential cooperative DM producer for multi-DFB TC-exhaustion tests (Gap 7).
//
// All N producer threads cooperate on DFB_0 (each handling its own strided slice),
// then cooperate on DFB_1, and so on.  Every DFB has its own independent DRAM
// source buffer whose base address is supplied as a runtime arg.
//
// After all producers call dfb.finish() for DFB_i, the barrier ensures they have
// all completed before any of them advances to DFB_i+1.  The consumer kernel uses
// the symmetric dfb_seq_consumer.cpp pattern so both sides stay in lock-step.
//
// Compile-time args:
//   [0]: num_entries_per_producer  � per DFB, same for all DFBs
//   [1]: implicit_sync             � 0 or 1
//   [2..]: TensorAccessorArgs      � shared DRAM layout (same page_size for all bufs)
//
// Runtime args (per-core):
//   [0]: producer_mask             � bitmask of DM producer threads
//   [1]: num_dfbs                  � number of DFBs to loop through
//   [2 .. 2+num_dfbs-1]: src_addr[i] � DRAM base address of in_buffer_i

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(0);
    constexpr uint32_t implicit_sync        = get_compile_time_arg_val(1);
    constexpr auto src_args                 = TensorAccessorArgs<2>();

    const uint32_t producer_mask = get_arg_val<uint32_t>(0);
    const uint32_t num_dfbs      = get_arg_val<uint32_t>(1);

#ifdef ARCH_QUASAR
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    const uint32_t producer_idx =
        static_cast<uint32_t>(__builtin_popcount(producer_mask & ((1u << hartid) - 1u)));
#else
    const uint32_t producer_idx = 0;
#endif
    const uint32_t num_producers = static_cast<uint32_t>(__builtin_popcount(producer_mask));

    Noc noc;

    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        // Each DFB has its own source buffer; address comes from runtime args.
        const uint32_t src_addr = get_arg_val<uint32_t>(2 + dfb_id);

        DataflowBuffer dfb(dfb_id);
        const uint32_t entry_size    = dfb.get_entry_size();
        const auto tensor_accessor   = TensorAccessor(src_args, src_addr, entry_size);

        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
            // Strided: producer i owns pages i, i+P, i+2P, ...
            const uint32_t page_id = tile_id * num_producers + producer_idx;
            if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
                noc.async_read<Noc::TxnIdMode::ENABLED>(
                    tensor_accessor, dfb, {.page_id = page_id}, {});
#endif
            } else {
                dfb.reserve_back(1);
                noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = page_id}, {});
                noc.async_read_barrier();
                dfb.push_back(1);
            }
        }
        // All producers of this DFB must finish before moving to the next DFB.
        dfb.finish();
    }
}
