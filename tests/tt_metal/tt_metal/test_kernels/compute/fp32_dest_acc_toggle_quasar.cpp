// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mid-kernel FP32 dest-acc toggling on Quasar (set_fp32_dest_acc / restore_fp32_dest_acc, which wrap
// enable_fp32_dest_acc / disable_fp32_dest_acc).
//
// Built either for 16-bit or 32-bit dest (DST_ACCUM_MODE) and single-buffered dest (SyncFull). It runs
// NUM_PHASES phases that alternate between the compiled width W and !W:
//   phase 0: W     -> set_fp32_dest_acc<!W>
//   phase 1: !W    -> restore_fp32_dest_acc<!W>
//   phase 2: W     -> set_fp32_dest_acc<!W>
//   phase 3: !W    -> restore_fp32_dest_acc<!W>
// Each phase unpacks TILES_PER_PHASE (in0 + in1) tile pairs, accumulates them into dest tile 0 with
// add_tiles(acc_to_dest), and packs the result to its own output DFB (out<phase>): Float32 for 32-bit
// phases, Float16_b for 16-bit phases. See test_fp32_dest_acc_toggle.cpp for the stimulus that makes the
// two widths give different results.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {

constexpr std::uint32_t TILES_PER_PHASE = 9;
constexpr bool COMPILED_FP32 = DST_ACCUM_MODE;

// Every width-dependent call takes the phase's dest width explicitly: the packer's IN_DATA_FORMAT (Quasar
// has no Read_32b_data bit), the dest-section handling in tile_regs_commit / tile_regs_release, and the pack
// ZEROACC width. The defaults would be the compiled DST_ACCUM_MODE. add_init is not repeated: its ALU config
// is width-independent apart from the Fp32 bits the toggle writes.
template <bool fp32_dest>
void run_phase(DataflowBuffer& in0, DataflowBuffer& in1, DataflowBuffer& out) {
    pack_init(out.get_id());
    pack_reconfig_data_format<false /*is_tile_dim_reconfig_en*/, fp32_dest>(out.get_id());

    tile_regs_acquire();
    for (std::uint32_t i = 0; i < TILES_PER_PHASE; ++i) {
        in0.wait_front(1);
        in1.wait_front(1);
        add_tiles<fp32_dest>(in0.get_id(), in1.get_id(), 0, 0, 0);
        in0.pop_front(1);
        in1.pop_front(1);
    }
    tile_regs_commit<fp32_dest>();

    tile_regs_wait();
    out.reserve_back(1);
    pack_tile(0, out.get_id());
    out.push_back(1);
    tile_regs_release<fp32_dest>();
}

}  // namespace

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer out1(dfb::out1);
    DataflowBuffer out2(dfb::out2);
    DataflowBuffer out3(dfb::out3);

    compute_kernel_hw_startup(in0.get_id(), in1.get_id(), out0.get_id());
    add_init(in0.get_id(), in1.get_id(), true /*acc_to_dest*/);

    run_phase<COMPILED_FP32>(in0, in1, out0);
    set_fp32_dest_acc<!COMPILED_FP32>();
    run_phase<!COMPILED_FP32>(in0, in1, out1);
    restore_fp32_dest_acc<!COMPILED_FP32>();
    run_phase<COMPILED_FP32>(in0, in1, out2);
    set_fp32_dest_acc<!COMPILED_FP32>();
    run_phase<!COMPILED_FP32>(in0, in1, out3);
    restore_fp32_dest_acc<!COMPILED_FP32>();
}
