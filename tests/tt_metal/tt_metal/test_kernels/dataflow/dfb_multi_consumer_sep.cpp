// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-DFB consumer kernel with a separate DRAM output buffer per DFB (Gap 7).
//
// Like dfb_multi_consumer.cpp but each instance writes to its own independent DRAM
// output buffer (dst_addr_base is per-instance) rather than a shared buffer + offset.
// Used for TensixDMTest4xDFB_1Sx1S where the host verifies each out_buffer separately.
//
// Compile-time args:
//   [0]: num_entries_per_consumer  – entries this instance consumes
//   [1]: implicit_sync             – 0 or 1
//   [2]: dst_addr_base             – this DFB's own DRAM out_buffer base address
//   [3]: dfb_id                    – logical DFB ID this instance handles
//   [4..]: TensorAccessorArgs      – shared DRAM layout descriptor

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_entries_per_consumer = get_compile_time_arg_val(0);
    constexpr uint32_t implicit_sync        = get_compile_time_arg_val(1);
    const uint32_t dst_addr_base            = get_compile_time_arg_val(2);
    const uint32_t dfb_id                   = get_compile_time_arg_val(3);
    constexpr auto dst_args                 = TensorAccessorArgs<4>();

    DataflowBuffer dfb(dfb_id);
    Noc noc;
    const uint32_t entry_size  = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base, entry_size);

    // 1Sx1S sole consumer: pages are sequential starting from 0.
    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_write<Noc::TxnIdMode::ENABLED>(
                dfb, tensor_accessor, {}, {.page_id = tile_id});
#endif
        } else {
            dfb.wait_front(1);
            noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = tile_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    dfb.finish();
    dfb.write_barrier(noc);
}
