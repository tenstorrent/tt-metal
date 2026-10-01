// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    // The dataflow buffer we are going to read from and write to DRAM. The host binds this
    // kernel as its consumer; the compute kernel is the producer.
    DataflowBuffer dfb_out(dfb::out);

    // Address of the output tensor, supplied as a ProgramRunArgs tensor argument.
    const auto out = TensorAccessor(tensor::out);

    Noc noc;

#ifdef ARCH_QUASAR
    // Quasar: implicit-sync write. The dataflow buffer credit advances via the per-trid
    // completion ISR, so no wait_front / barrier / pop_front is required.
    noc.async_write<NocOptions::TXN_ID>(dfb_out, out, {}, {.page_id = 0});
#else
    // Make sure there is a tile in the dataflow buffer
    dfb_out.wait_front(onetile);
    // write the tile to DRAM
    noc.async_write(dfb_out, out, dfb_out.get_entry_size(), {}, {.page_id = 0});
    noc.async_write_barrier();  // This will wait until the write is done. As an alternative,
                                // a flush can be faster because it waits until the write
                                // request is sent. In that case, you have to use a write
                                // barrier at least once at the end of the data movement
                                // kernel to make sure all writes are done.
    // Mark the tile as consumed
    dfb_out.pop_front(onetile);
#endif

    dfb_out.finish();

#ifdef ARCH_QUASAR
    // Required on Quasar to flush transactions enqueued via NocOptions::TXN_ID before the
    // kernel exits.
    dfb_out.write_barrier(noc);
#endif
}
