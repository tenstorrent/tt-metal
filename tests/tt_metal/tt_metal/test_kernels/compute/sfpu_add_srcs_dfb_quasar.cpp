// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Two-input SFPU op on SrcS with both inputs from L1, all on trisc3. out = in + in1.
// Bindings: dfb::in and dfb::in1 CONSUMER, dfb::out PRODUCER, all handed to the SFPU pipeline.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    constexpr uint32_t in_id = dfb::in;  // the DFB id is a compile-time constant: it names the tiles' format
    using AddOp = SfpuOp<op::Add<>, SfpuReg::SrcS, in_id>;
    AddOp::init();

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb_in.wait_front(1);
        dfb_in1.wait_front(1);
        dfb_out.reserve_back(1);
        AddOp::unpack(dfb_in, dfb_in1);
        AddOp::pack(dfb_out);
        AddOp::tile();
        dfb_in.pop_front(1);  // must not free the entries before unpacker 2 has read them
        dfb_in1.pop_front(1);
        dfb_out.push_back(1);  // must not post the entry before packer 1 has written it
    }
}
