// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/generic_moe_gate.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t selected = get_compile_time_arg_val(1);
    constexpr bool normalize = get_compile_time_arg_val(2) != 0;
    constexpr bool scores_include_bias = get_compile_time_arg_val(3) != 0;
// The compile-contract test defines this to pass an explicit false for is_fp32_dest_acc_en under an
// fp32 Dest build, so the gate's DST_ACCUM_MODE clause is exercised on its own.
#ifdef GENERIC_MOE_GATE_EXPLICIT_FP32_FALSE
    constexpr bool gate_fp32_dest_acc = false;
#else
    constexpr bool gate_fp32_dest_acc = DST_ACCUM_MODE;
#endif
    CircularBuffer scores(tt::CBIndex::c_0);
    CircularBuffer bias(tt::CBIndex::c_1);
    CircularBuffer output(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    generic_moe_gate_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    scores.wait_front(1);
    bias.wait_front(1);
    output.reserve_back(1);
    tile_regs_acquire();
    if constexpr (scores_include_bias) {
        generic_moe_gate<normalize, selected, 256, true, true, true, false, gate_fp32_dest_acc, true>(
            tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0x3f800000);
    } else {
        // Intentionally omit scores_include_bias to exercise the existing caller syntax
        // and verify that the default still returns the unbiased payload.
        generic_moe_gate<normalize, selected, 256, true, true, true, false, gate_fp32_dest_acc>(
            tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0x3f800000);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_16);
    tile_regs_release();
    scores.pop_front(1);
    bias.pop_front(1);
    output.push_back(1);
}
