// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Mixed: an FPU op on Dest feeds an SFPU op on SrcS through a DataflowBuffer. out = exp(in + in).
// Bindings: dfb::in CONSUMER; dfb::mid PRODUCER and CONSUMER; dfb::out PRODUCER.
// The mid CONSUMER end and the out PRODUCER end are handed to the SFPU pipeline.

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
    // L1 format of mid and out, also the SrcS register format.
    constexpr DataFormat format = static_cast<DataFormat>(get_arg(args::data_format));

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_mid(dfb::mid);
    DataflowBuffer dfb_out(dfb::out);
    constexpr uint32_t in_id = dfb::in;
    const uint32_t mid_id = dfb_mid.get_id();
    using ExpOp = SfpuOp<op::Exp<>, SfpuReg::SrcS, format>;

    compute_kernel_hw_startup(in_id, in_id, mid_id);
    add_init(in_id, in_id);
    ExpOp::init();

    for (uint32_t i = 0; i < num_entries; i++) {
        // Stage A, unpack / math / pack: in -> in + in on the FPU -> mid.
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

        // Stage B, trisc3: mid -> exp on SrcS -> out.
        dfb_mid.wait_front(1);
        dfb_out.reserve_back(1);
        ExpOp::tile(unpack_from(dfb_mid), pack_to(dfb_out));
        dfb_mid.pop_front(1);  // must not free the entry before unpacker 2 has read it
        dfb_out.push_back(1);  // must not post the entry before packer 1 has written it
    }
}
