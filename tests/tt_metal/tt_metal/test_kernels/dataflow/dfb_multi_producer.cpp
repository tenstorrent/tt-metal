// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-DFB producer kernel for concurrent DFB stress tests (Gap 7).
//
// Each instance of this kernel runs as a single DM thread and handles exactly one
// DFB.  Under the legacy quasar API the DFB is identified by the dfb_id CTA; under
// the Metal 2.0 API the DFB is identified by the kernel's `dfb::out` binding (one
// binding per kernel instance).  In both cases the harness creates one kernel
// instance per DFB so each DFB gets precisely one producer thread - matching
// num_producers=1 in the DFB config.
//
// Compile-time args (legacy):
//   [0]: num_entries_per_producer  - entries this instance produces
//   [1]: implicit_sync             - 0 or 1
//   [2]: src_addr_base             - shared DRAM in_buffer base address
//   [3]: dfb_id                    - logical DFB ID this instance handles
//   [4]: chunk_offset              - page offset into the shared DRAM buffer
//   [5..]: TensorAccessorArgs      - shared DRAM layout descriptor
//
// QUASAR named args (CTAs):
//   args::num_entries_per_producer
//   args::implicit_sync
//   args::chunk_offset
// QUASAR bindings:
//   dfb::out               - producer endpoint of this instance's DFB
//   ta::src_tensor         - shared DRAM in_tensor accessor

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#ifdef ARCH_QUASAR
#include "experimental/kernel_args.h"
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t chunk_offset = get_arg(args::chunk_offset);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(ta::src_tensor);
#else
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(0);
    constexpr uint32_t implicit_sync        = get_compile_time_arg_val(1);
    const uint32_t src_addr_base            = get_compile_time_arg_val(2);
    const uint32_t dfb_id                   = get_compile_time_arg_val(3);
    const uint32_t chunk_offset             = get_compile_time_arg_val(4);
    constexpr auto src_args                 = TensorAccessorArgs<5>();

    DataflowBuffer dfb(dfb_id);
    Noc noc;
    const uint32_t entry_size  = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(src_args, src_addr_base, entry_size);
#endif

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        const uint32_t page_id = chunk_offset + tile_id;
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
    DPRINT << "producer before finish" << ENDL();
    dfb.finish();
    DPRINT << "producer after finish" << ENDL();
}
