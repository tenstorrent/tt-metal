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

    // The dataflow buffers to read the tiles into. A DataflowBuffer is backed by a circular
    // buffer on Gen1 (Wormhole, Blackhole) and by a hardware dataflow buffer on Gen2 (Quasar);
    // the host declares them in the ProgramSpec and binds this kernel as their producer.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // Address generators for the input tensors. Consider these the pointers for interleaved
    // buffers. The tensor addresses are supplied as ProgramRunArgs tensor arguments, so the
    // kernel needs no runtime args of its own.
    const auto in0 = TensorAccessor(tensor::in0);
    const auto in1 = TensorAccessor(tensor::in1);

    Noc noc;

#ifdef ARCH_QUASAR
    // Quasar: implicit-sync read. The dataflow buffer credit advances via the per-trid
    // completion ISR, so no reserve_back / barrier / push_back is required.
    noc.async_read<NocOptions::TXN_ID>(in0, dfb_in0, {.page_id = 0}, {});
    noc.async_read<NocOptions::TXN_ID>(in1, dfb_in1, {.page_id = 0}, {});
#else
    // read the tile from the first input tensor into its dataflow buffer
    dfb_in0.reserve_back(onetile);
    noc.async_read(in0, dfb_in0, dfb_in0.get_entry_size(), {.page_id = 0}, {});
    noc.async_read_barrier();    // wait until the read is done
    dfb_in0.push_back(onetile);  // mark the tile as ready

    // same process for the second input (different dataflow buffer and input tensor)
    dfb_in1.reserve_back(onetile);
    noc.async_read(in1, dfb_in1, dfb_in1.get_entry_size(), {.page_id = 0}, {});
    noc.async_read_barrier();
    dfb_in1.push_back(onetile);
#endif

    dfb_in0.finish();
    dfb_in1.finish();
}
