// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SFPU op on SrcS between two DataflowBuffers, all on trisc3. out = exp(in).
// Bindings: dfb::in CONSUMER and dfb::out PRODUCER, both handed to the SFPU pipeline.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    // L1 format of both DFBs, also the SrcS register format.
    constexpr DataFormat format = static_cast<DataFormat>(get_arg(args::data_format));

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    using ExpOp = SfpuOp<op::Exp<>, SfpuReg::SrcS, format>;
    ExpOp::init();

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        ExpOp::unpack(dfb_in);
        ExpOp::pack(dfb_out);
        ExpOp::tile();
        dfb_in.pop_front(1);   // must not free the entry before unpacker 2 has read it
        dfb_out.push_back(1);  // must not post the entry before packer 1 has written it
    }
}
