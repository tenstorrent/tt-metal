// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/compressed_custom_mm.h"
#include "api/compute/experimental/2_0/custom_mm.h"

// Runs only the CB-id init of custom_mm (mode 0) or compressed_custom_mm (mode 1), on one thread
// only (thread 0: unpack init, thread 1: math init), and exits: the in0 tile-height assert in that
// init is what the guard test observes through the watcher, and one asserting thread per run is
// what lets the test attribute the report to a RISC (the watcher keeps the first assert per core).
// Thread 2 runs the 2.0 mixed-overload init block instead (CB-id in0, typed in1), as the product
// test's mixed mode calls it; its unpack and math parts both read the in0 CB geometry.
// Nothing here waits on another thread, so a tripped assert parks only the thread that hit it and
// the core can be re-initialized afterwards; a thread parked in a tile-register semaphore wait
// cannot be, which is why the product run is not used for the asserting case.
void kernel_main() {
    constexpr std::uint32_t mode = get_compile_time_arg_val(0);    // plain, compressed
    constexpr std::uint32_t thread = get_compile_time_arg_val(1);  // unpack init, math init, 2.0 mixed init
    constexpr auto in0 = tt::CBIndex::c_0;
    constexpr auto in1 = tt::CBIndex::c_1;
    if constexpr (thread == 2) {
        constexpr auto weight_format = mode == 1 ? DataFormat::Bfp8_b : DataFormat::Float16_b;
        using Weight = experimental::LLKOperand<weight_format, TensorShape{16, 16, 2, 2}>;
        UNPACK((llk_unpack_hw_configure<false /*fp32_dest_acc_en*/>(in1, in0)));
        MATH((llk_math_hw_configure<false /*fp32_dest_acc_en*/>(in0, in1)));
        if constexpr (mode == 1) {
            compressed_custom_mm_block_init_short<false /*transpose*/, false /*split_acc*/, true /*dense_packing*/>(
                in0, Weight(0));
        } else {
            custom_mm_block_init_short<false /*transpose*/, false /*split_acc*/, true /*dense_packing*/>(
                in0, Weight(0), 1 /*ct_dim*/);
        }
    } else if constexpr (thread == 0) {
        UNPACK((llk_unpack_hw_configure<false /*fp32_dest_acc_en*/>(in1, in0)));
        if constexpr (mode == 1) {
            UNPACK((llk_unpack_AB_compressed_custom_mm_init<false /*transpose*/, true /*clear_src*/>(in0, in1)));
        } else {
            UNPACK((llk_unpack_AB_custom_mm_init<false /*transpose*/, true /*clear_src*/>(in0, in1, 1 /*ct_dim*/)));
        }
    } else {
        MATH((llk_math_hw_configure<false /*fp32_dest_acc_en*/>(in0, in1)));
        if constexpr (mode == 1) {
            MATH((llk_math_compressed_custom_mm_init<false /*transpose*/, false /*split_acc*/, true /*dense_packing*/>(
                in0, in1)));
        } else {
            MATH((llk_math_custom_mm_init<false /*transpose*/, false /*split_acc*/, true /*dense_packing*/>(
                in0, in1, 1 /*ct_dim*/)));
        }
    }
}
