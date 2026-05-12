// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-DFB producer kernel for concurrent DFB stress tests (Gap 7).
//
// Each instance of this kernel runs as a single DM thread and handles exactly one
// DFB, identified by the dfb_id compile-time arg.  The harness creates one kernel
// instance per DFB (with a unique dfb_id and chunk_offset in the CTA), so each DFB
// gets precisely one producer thread – matching num_producers=1 in the DFB config.
//
// Encoding dfb_id and chunk_offset in the CTA (rather than deriving them from
// mhartid at runtime) avoids any dependency on the DM thread assignment order and
// makes each kernel instance distinct so the runtime allocates a separate DM thread
// to each one.
//
// Compile-time args:
//   [0]: num_entries_per_producer  – entries this instance produces
//   [1]: implicit_sync             – 0 or 1
//   [2]: src_addr_base             – shared DRAM in_buffer base address
//   [3]: dfb_id                    – logical DFB ID this instance handles
//   [4]: chunk_offset              – page offset into the shared DRAM buffer
//   [5..]: TensorAccessorArgs      – shared DRAM layout descriptor

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(0);
    constexpr uint32_t implicit_sync        = get_compile_time_arg_val(1);
    const uint32_t src_addr_base            = get_compile_time_arg_val(2);
    const uint32_t dfb_id                   = get_compile_time_arg_val(3);
    const uint32_t chunk_offset             = get_compile_time_arg_val(4);
    constexpr auto src_args                 = TensorAccessorArgs<5>();

    experimental::DataflowBuffer dfb(dfb_id);
    experimental::Noc noc;
    const uint32_t entry_size  = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(src_args, src_addr_base, entry_size);


    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        const uint32_t page_id = chunk_offset + tile_id;
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(
                tensor_accessor, dfb, {.page_id = page_id}, {});
#endif
        } else {
            dfb.reserve_back(1);
            noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = page_id}, {});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    }
    DPRINT << "producer before finish" << ENDL();
    dfb.finish();
    DPRINT << "producer after finish" << ENDL();
}
