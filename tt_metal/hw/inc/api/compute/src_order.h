// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel {

// clang-format off
/**
 * Describes how the two input circular buffers map onto the source registers (SrcA/SrcB) of the
 * compute engine. This matters because both compute_kernel_hw_startup() and reconfig_data_format()
 * program per-source-register state (data formats, tile/face dimensions, tile sizes) and that state
 * must match the operand ordering of the operation that follows.
 *
 *  - SrcOrder::Regular : icb0 -> SrcA, icb1 -> SrcB. This is the natural ordering used by virtually
 *                        every operation (e.g. eltwise binary, where in0 -> SrcA and in1 -> SrcB).
 *  - SrcOrder::Reverse : icb0 -> SrcB, icb1 -> SrcA. The operands are mapped onto the source registers
 *                        in reverse. Matmul is the operation that needs this today: in0 (the "A" /
 *                        activations operand) is loaded into SrcB, while in1 (the "B" / weights operand)
 *                        is loaded into SrcA. Passing this tag lets the kernel keep calling with the
 *                        natural (in0, in1) argument order while the source registers are programmed in
 *                        the reversed order such an operation requires.
 */
// clang-format on
enum class SrcOrder : uint8_t {
    Regular = 0,  // icb0 -> SrcA, icb1 -> SrcB
    Reverse = 1,  // icb0 -> SrcB, icb1 -> SrcA
};

}  // namespace ckernel
