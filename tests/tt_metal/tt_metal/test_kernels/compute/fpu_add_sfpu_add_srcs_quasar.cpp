// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Two-input SFPU op on the SrcS pipeline: input 0 is the tile the FPU left in Dest, input 1 comes from L1.
// out = (in + in) + in1.
// Bindings: dfb::in CONSUMER; dfb::in1 CONSUMER; dfb::out PRODUCER.
// The in1 CONSUMER end and the out PRODUCER end are handed to the SFPU pipeline.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/experimental/sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    constexpr uint32_t in_id = dfb::in;
    constexpr uint32_t in1_id = dfb::in1;
    // in1_id: the DFB of the L1 input. in_id, in_id: input 0 is the Dest tile the FPU op on them left.
    using AddOp = SfpuOp<op::Add<>, SfpuReg::SrcS, in1_id, in_id, in_id>;

    // Packer 0 packs nothing here; it is set up for the output format only.
    compute_kernel_hw_startup(in_id, in_id, dfb_out.get_id());
    add_init(in_id, in_id);
    AddOp::init();

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb_in.wait_front(1);
        dfb_in1.wait_front(1);
        dfb_out.reserve_back(1);

        // FPU: Dest[0] = in + in.
        tile_regs_acquire();
        add_tiles(in_id, in_id, /*itile0=*/0, /*itile1=*/0, /*idst=*/0);

        // SFPU on trisc3: Dest[0] + in1 -> out. MATH lends it the Dest section.
        tile_regs_borrow<DstClient::MATH, DstClient::SFPU>();
        AddOp::unpack(dfb_in1);
        AddOp::pack(dfb_out);
        AddOp::tile(DstIndex{0});
        tile_regs_return<DstClient::MATH, DstClient::SFPU>();

        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_in1.pop_front(1);  // must not free the entry before unpacker 2 has read it
        dfb_out.push_back(1);  // must not post the entry before packer 1 has written it
    }
}
