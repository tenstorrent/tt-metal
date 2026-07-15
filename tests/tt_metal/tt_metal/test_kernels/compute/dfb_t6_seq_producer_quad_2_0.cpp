// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) sequential Tensix producer for 4 concurrent DFBs.
// Parallel to ../dfb_t6_seq_producer.cpp; M2 uses 4 named DFB bindings instead
// of a runtime-determined DFB count.
//
// A single Neo thread loops dfb::buf_0..buf_3, calling reserve_back/push_back
// for each entry. The host pre-fills each DFB's L1 ring directly via
// uniform_alloc_addr (NOT via borrowed_from / tensor bindings: TRISC compute
// kernels can't include tensor_accessor.h transitively, so the producer cannot
// carry tensor bindings — only DFB bindings).

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <typename Dfb>
static inline void signal_one_dfb(Dfb& dfb, uint32_t num_entries_per_producer) {
    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    dfb.finish();
}

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);

    {
        DataflowBuffer dfb(dfb::buf_0);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
    {
        DataflowBuffer dfb(dfb::buf_1);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
    {
        DataflowBuffer dfb(dfb::buf_2);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
    {
        DataflowBuffer dfb(dfb::buf_3);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
}
