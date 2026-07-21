// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#endif

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Describes how the two input circular buffers map onto the source registers (SrcA/SrcB) of the
 * compute engine. This matters because compute_kernel_hw_startup() programs per-source-register
 * state (data formats, tile/face dimensions, tile sizes) and that state must match the operand
 * ordering of the operation that follows.
 *
 *  - SrcOrder::Regular : icb0 -> SrcA, icb1 -> SrcB. This is the natural ordering used by virtually
 *                        every operation (e.g. eltwise binary, where in0 -> SrcA and in1 -> SrcB).
 *  - SrcOrder::Reverse : icb0 -> SrcB, icb1 -> SrcA. The operands are mapped onto the source registers
 *                        in reverse. Matmul is the operation that needs this today: in0 (the "A" /
 *                        activations operand) is loaded into SrcB, while in1 (the "B" / weights operand)
 *                        is loaded into SrcA. Passing this tag lets the kernel keep calling startup with
 *                        the natural (in0, in1) argument order while startup programs the source
 *                        registers in the reversed order such an operation requires.
 */
// clang-format on
enum class SrcOrder : uint8_t {
    Regular = 0,  // icb0 -> SrcA, icb1 -> SrcB
    Reverse = 1,  // icb0 -> SrcB, icb1 -> SrcA
};

// clang-format off
/**
 * Performs the required hardware initialization for all subsequent operations in the compute kernel. This function should be
 * called exactly once at the very beginning of the kernel, before any operation-specific initialization functions (such as
 * reduce_init, tilize_init, etc.). The circular buffer (CB) IDs provided to this function must match those used in the next
 * operation-specific initialization function. If the operands for the next operation require a different data format than
 * what was configured here, you must call one of the reconfig_data_format functions before proceeding with the next
 * initialization. Similarly, if the next operation requires different properties (such as tile or face dimensions), you must
 * ensure that the same CB IDs are used as in this function.
 *
 * The src_order template parameter selects how (icb0, icb1) map onto SrcA/SrcB; this is the single piece of
 * operation-specific knowledge startup needs, because the per-source-register state it programs (formats, tile/face
 * dimensions, tile sizes) depends on that mapping. Use SrcOrder::Regular for all operations except matmul, which must use
 * SrcOrder::Reverse (see the SrcOrder documentation). The (icb0, icb1) arguments are always passed in natural operand order
 * (in0, in1) regardless of the tag.
 *
 * NOTE: This function performs MMIO writes, which are slow and almost exclusively require the idle state of the execution
 * units that should be configured (PACK, MATH, UNPACK, CFG, etc.). This is why it is unsafe to call this function in the
 * middle of a kernel execution. This function should be called only once at the beginning of the kernel, before any other
 * calls to Compute API are made (either init or other). Calling this function after other API calls may lead cause race
 * conditions and undefined behavior which can be hard to debug.
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                                                     | Type     | Valid Range | Required |
 * |------------|-----------|-----------------------------------------------------------------|----------|-------------|----------|
 * | Template   | src_order | How icb0/icb1 map onto SrcA/SrcB (Regular or Reverse)          | SrcOrder | N/A         | False    |
 * | Function   | icb0      | The identifier of the circular buffer (CB) containing operand A | uint32_t | 0 to 31     | True     |
 * | Function   | icb1      | The identifier of the circular buffer (CB) containing operand B | uint32_t | 0 to 31     | True     |
 * | Function   | ocb       | The identifier of the output circular buffer (CB)               | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    // Map the operands onto the physical source registers. For SrcOrder::Reverse (matmul) in0 (icb0)
    // lands in SrcB and in1 (icb1) lands in SrcA, so the per-source state below is programmed with the
    // operands swapped. src_order is a template parameter, so reverse (and the selection below) is
    // resolved at compile time. Both UNPACK and MATH hw_configure are programmed with the same
    // (src_a_cb, src_b_cb) ordering so the unpacker tile descriptors and the math ALU format registers agree.
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    const uint32_t src_a_cb = reverse ? icb1 : icb0;
    const uint32_t src_b_cb = reverse ? icb0 : icb1;
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(src_a_cb, src_b_cb)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(src_a_cb, src_b_cb)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init<PackMode::Default>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>(ocb)));

    ComputeKernelSentinel::instance().set_srca(src_a_cb).set_srcb(src_b_cb).set_pack(ocb);
#else
    UNPACK((llk_unpack_hw_configure(src_a_cb, src_b_cb)));

    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(src_a_cb, src_b_cb)));

    PACK((llk_pack_hw_configure(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));
#endif
}

// clang-format off
/**
 * Convenience overload for hardware initialization when only one input circular buffer is used.
 * Both input operands (srcA and srcB) will be programmed using the same circular buffer identifier (`icb0`).
 * Internally, this calls the three-parameter version with `icb0` passed for both input operands.
 *
 * | Param Type | Name  | Description                                                        | Type     | Valid Range | Required |
 * |------------|-------|--------------------------------------------------------------------|----------|-------------|----------|
 * | Function   | icb0  | The identifier of the circular buffer (CB) used for both input ops | uint32_t | 0 to 31     | True     |
 * | Function   | ocb   | The identifier of the output circular buffer (CB)                  | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void compute_kernel_hw_startup(uint32_t icb0, uint32_t ocb) { compute_kernel_hw_startup(icb0, icb0, ocb); }

// clang-format off
/**
 * Enables FP32 accumulation in the destination register.
 *
 * Configures both the math pipeline (ALU_ACC_CTRL Fp32_enabled and
 * SFPU_Fp32_enabled) and the packer (PCK_DEST_RD_CTRL Read_32b_data)
 * for 32-bit destination reads. This is a lightweight, standalone
 * reconfiguration that is safe to call mid-kernel without re-running
 * compute_kernel_hw_startup.
 *
 * Must be paired with disable_fp32_dest_acc() when switching back to
 * BF16 accumulation mode within the same kernel.
 *
 * Only available on Wormhole and Blackhole. Not supported on Quasar (compile error)
 *
 * Return value: None
 */
// clang-format on
#ifndef ARCH_QUASAR
ALWI void enable_fp32_dest_acc() {
    MATH((llk_math_set_fp32_dest_acc(true)));
    PACK((llk_pack_set_fp32_dest_acc(true)));
}
#endif

// clang-format off
/**
 * Disables FP32 accumulation in the destination register, reverting to
 * BF16 accumulation mode.
 *
 * Configures both the math pipeline (ALU_ACC_CTRL Fp32_enabled and
 * SFPU_Fp32_enabled) and the packer (PCK_DEST_RD_CTRL Read_32b_data)
 * to disable 32-bit destination reads. This is a lightweight, standalone
 * reconfiguration that is safe to call mid-kernel without re-running
 * compute_kernel_hw_startup.
 *
 * Only available on Wormhole and Blackhole. Not supported on Quasar (compile error)
 *
 * Return value: None
 */
// clang-format on
#ifndef ARCH_QUASAR
ALWI void disable_fp32_dest_acc() {
    MATH((llk_math_set_fp32_dest_acc(false)));
    PACK((llk_pack_set_fp32_dest_acc(false)));
}
#endif

}  // namespace ckernel
