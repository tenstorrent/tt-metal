// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-DFB consumer kernel for concurrent DFB stress tests (Gap 7).
//
// Each instance handles exactly one DFB (identified by dfb_id in the CTA) and
// drains entries to its chunk of a shared DRAM out_buffer (offset by chunk_offset).
// The harness creates one instance per DFB so each DFB gets exactly one consumer
// thread – matching num_consumers=1 in the DFB config.
//
// Compile-time args:
//   [0]: num_entries_per_consumer  – entries this instance consumes
//   [1]: implicit_sync             – 0 or 1
//   [2]: dst_addr_base             – shared DRAM out_buffer base address
//   [3]: dfb_id                    – logical DFB ID this instance handles
//   [4]: chunk_offset              – page offset into the shared DRAM buffer
//   [5..]: TensorAccessorArgs      – shared DRAM layout descriptor

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_entries_per_consumer = get_compile_time_arg_val(0);
    constexpr uint32_t implicit_sync        = get_compile_time_arg_val(1);
    const uint32_t dst_addr_base            = get_compile_time_arg_val(2);
    const uint32_t dfb_id                   = get_compile_time_arg_val(3);
    const uint32_t chunk_offset             = get_compile_time_arg_val(4);
    constexpr auto dst_args                 = TensorAccessorArgs<5>();

    DataflowBuffer dfb(dfb_id);
    Noc noc;
    const uint32_t entry_size  = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base, entry_size);


    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        const uint32_t page_id = chunk_offset + tile_id;
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_write<Noc::TxnIdMode::ENABLED>(
                dfb, tensor_accessor, {}, {.page_id = page_id});
#endif
        } else {
            DPRINT << "consumer wait page id: " << page_id << ENDL();
            dfb.wait_front(1);
            noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    DPRINT << "consumer before finish" << ENDL();
    dfb.finish();
    DPRINT << "at end of kernel_main b4 write barrier" << ENDL();
    dfb.write_barrier(noc);
    DPRINT << "finished write barrier" << ENDL();
}
