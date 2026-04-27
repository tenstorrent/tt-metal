// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential cooperative DM consumer for multi-DFB TC-exhaustion tests (Gap 7).
//
// All N consumer threads cooperate on DFB_0, then DFB_1, and so on.  Each DFB
// has its own independent DRAM output buffer.  Both STRIDED and BLOCKED access
// patterns are supported on a per-DFB basis (controlled by the is_blocked runtime
// arg for each DFB).
//
// Compile-time args:
//   [0]: implicit_sync             � 0 or 1
//   [1..]: TensorAccessorArgs      � shared DRAM layout (same page_size for all bufs)
//
// Runtime args (per-core):
//   [0]: consumer_mask             � bitmask of DM consumer threads
//   [1]: num_dfbs                  � number of DFBs to loop through
//   [2 .. 2+M-1]:     dst_addr[i]             � DRAM base address of out_buffer_i
//   [2+M .. 2+2M-1]:  entries_per_consumer[i] � loop count per thread for DFB_i
//   [2+2M .. 2+3M-1]: is_blocked[i]           � 1 for BLOCKED, 0 for STRIDED

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t implicit_sync = get_compile_time_arg_val(0);
    constexpr auto dst_args          = TensorAccessorArgs<1>();

    const uint32_t consumer_mask = get_arg_val<uint32_t>(0);
    const uint32_t num_dfbs      = get_arg_val<uint32_t>(1);

#ifdef ARCH_QUASAR
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    const uint32_t consumer_idx =
        static_cast<uint32_t>(__builtin_popcount(consumer_mask & ((1u << hartid) - 1u)));
#else
    const uint32_t consumer_idx = 0;
#endif
    const uint32_t num_consumers = static_cast<uint32_t>(__builtin_popcount(consumer_mask));

    experimental::Noc noc;

    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        const uint32_t dst_addr            = get_arg_val<uint32_t>(2 + dfb_id);
        const uint32_t entries_per_consumer = get_arg_val<uint32_t>(2 + num_dfbs + dfb_id);
        const uint32_t is_blocked          = get_arg_val<uint32_t>(2 + 2 * num_dfbs + dfb_id);

        experimental::DataflowBuffer dfb(dfb_id);
        const uint32_t entry_size  = dfb.get_entry_size();
        const auto tensor_accessor = TensorAccessor(dst_args, dst_addr, entry_size);

        for (uint32_t tile_id = 0; tile_id < entries_per_consumer; tile_id++) {
            // BLOCKED: each consumer reads all entries sequentially (broadcast).
            // STRIDED: interleaved, consumer i reads pages i, i+C, i+2C, ...
            const uint32_t page_id = is_blocked
                ? tile_id
                : tile_id * num_consumers + consumer_idx;

            if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
                noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
                    dfb, tensor_accessor, {}, {.page_id = page_id});
#endif
            } else {
                dfb.wait_front(1);
                noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
                noc.async_write_barrier();
                dfb.pop_front(1);
            }
        }
        // All consumers of this DFB must finish before moving to the next DFB.
        dfb.finish();
        dfb.write_barrier(noc);
    }
}
