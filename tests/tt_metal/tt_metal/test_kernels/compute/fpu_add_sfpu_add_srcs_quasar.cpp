// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Two-input SFPU op on SrcS: one input is a tile the FPU left in Dest, the other comes from L1.
// out = (in + in) + in1.
// Bindings: dfb::in CONSUMER; dfb::mid PRODUCER, and CONSUMER with use_trisc3; dfb::in1 CONSUMER and
// dfb::out PRODUCER, both with use_trisc3.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/experimental/sfpu.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    // L1 format of mid, in1 and out, also the SrcS register format.
    constexpr DataFormat format = static_cast<DataFormat>(get_arg(args::data_format));

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_mid(dfb::mid);
    DataflowBuffer dfb_out(dfb::out);
    constexpr uint32_t in_id = dfb::in;
    const uint32_t mid_id = dfb_mid.get_id();
    using AddOp = SfpuOp<op::Add<>, SfpuReg::SrcS, format>;

    compute_kernel_hw_startup(in_id, in_id, mid_id);
    add_init(in_id, in_id);
    AddOp::init();

    for (uint32_t i = 0; i < num_entries; i++) {
        // Stage A, unpack / math / pack: in -> in + in on the FPU -> Dest -> mid.
        dfb_in.wait_front(1);
        dfb_mid.reserve_back(1);
        tile_regs_acquire();
        add_tiles(in_id, in_id, /*itile0=*/0, /*itile1=*/0, /*idst=*/0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/*dst_index=*/0, mid_id);
        tile_regs_release();
        dfb_in.pop_front(1);
        dfb_mid.push_back(1);

        // Stage B, trisc3: mid (from Dest) and in1 (from L1) -> add on SrcS -> out.
        dfb_mid.wait_front(1);
        dfb_in1.wait_front(1);
        dfb_out.reserve_back(1);
        AddOp::tile(unpack_from(dfb_mid), unpack_from(dfb_in1), pack_to(dfb_out));
        dfb_mid.pop_front(1);  // must not free an input entry before unpacker 2 has read it
        dfb_in1.pop_front(1);
        dfb_out.push_back(1);  // must not post the entry before packer 1 has written it
    }
}
