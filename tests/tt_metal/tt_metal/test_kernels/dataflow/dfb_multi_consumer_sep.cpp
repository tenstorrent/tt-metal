// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-DFB consumer kernel with a separate DRAM output buffer per DFB (Gap 7).
//
// Like dfb_multi_consumer.cpp but each instance writes to its own independent DRAM
// output buffer (per-instance tensor::dst_tensor binding pointing at a per-DFB
// OUT_TENSOR_i parameter). Used for TensixDMTest4xDFB_1Sx1S where the host
// verifies each out_buffer separately.
//
// Named args (CTAs):
//   args::num_entries_per_consumer
//   args::implicit_sync
// Bindings:
//   dfb::in                - consumer endpoint of this instance's DFB
//   tensor::dst_tensor     - this instance's per-DFB DRAM out_tensor accessor

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    DataflowBuffer dfb(dfb::in);
    Noc noc;
    const uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(tensor::dst_tensor);

    // 1Sx1S sole consumer: pages are sequential starting from 0.
    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_write<NocOptions::TXN_ID>(
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
