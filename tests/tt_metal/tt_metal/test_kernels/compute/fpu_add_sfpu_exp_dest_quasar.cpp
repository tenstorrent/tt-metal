// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// FPU op, then SFPU op on Dest. out = exp(in + in).
// The SfpuOp alias names the two DFBs the FPU op read (SrcA and SrcB); here both are dfb::in.
// Bindings: dfb::in CONSUMER and dfb::out PRODUCER.

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

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    constexpr uint32_t in_id = dfb::in;  // the DFB id is a compile-time constant: it names the tiles' format
    const uint32_t out_id = dfb_out.get_id();
    using ExpOp = SfpuOp<op::Exp<>, SfpuReg::Dest, in_id, in_id>;

    compute_kernel_hw_startup(in_id, in_id, out_id);
    add_init(in_id, in_id);
    ExpOp::init();

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        add_tiles(in_id, in_id, /*itile0=*/0, /*itile1=*/0, /*idst=*/0);  // FPU: Dest[0] = in + in
        ExpOp::tile(DstIndex{0});                                         // SFPU: exp(Dest[0]) in place
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/*dst_index=*/0, out_id);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in.pop_front(1);
    }
}
