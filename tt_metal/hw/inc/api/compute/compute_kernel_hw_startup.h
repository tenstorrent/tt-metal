// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "sanitizer/api.h"
#include "api/compute/src_order.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#endif

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif

#ifdef TRISC_PACK
#include "llk_pack_common_api.h"
#endif

namespace ckernel {

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
    LLK_SAN_FUNCTION();

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

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
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
ALWI void compute_kernel_hw_startup(uint32_t icb0, uint32_t ocb) {
    LLK_SAN_FUNCTION();

    compute_kernel_hw_startup(icb0, icb0, ocb);
}

// clang-format off
/**
 * Resets (re-arms) the MATH<->PACK Dest synchronization mechanism for the output circular buffer (ocb)
 *
 * Background: the MATH<->PACK Dest handshake.
 *
 * During standard kernel operation the MATH thread writes its results into the Dest register and
 * the PACK thread moves those results to L1. How the two threads share Dest is determined by DstSync
 * mode (a compile-time flag passed into the kernel). DstSync::SyncFull lets only one thread (either
 * MATH or PACK) use Dest at a time, while DstSync::SyncHalf lets MATH and PACK work concurrently.
 *
 * Under DstSync::SyncHalf, Dest is split into two halves, such that MATH computes into one
 * half while PACK drains the other. MATH and PACK are coordinated by a single bounded counting
 * semaphore, semaphore::MATH_PACK, whose current value is the number of Dest sections MATH has
 * committed that PACK has not yet released: a half of the Dest under SyncHalf, the whole Dest under
 * SyncFull. The semaphore::MATH_PACK is initialized with a max value (2 under SyncHalf, 1 under SyncFull)
 * and a starting value of 0. Additionally, each thread tracks the section of Dest it should be
 * working on in private software state. Both MATH and PACK start at section 0.
 *
 * A compute kernel loops over the following handshake, one Dest section per iteration, using the
 * tile_regs_* API. Call this the handshake loop:
 *
 *   tile_regs_acquire()   MATH   blocks while the semaphore value is at the max value (no free half)
 *   tile_regs_commit()    MATH   posts semaphore (+1), "a section is full", then updates its own tracker
 *   tile_regs_wait()      PACK   blocks while the value is zero (nothing to drain)
 *   tile_regs_release()   PACK   gets semaphore (-1), "I drained one", then flips its own tracker
 *
 * Quiescent: semaphore::MATH_PACK is zero and both trackers point at the same Dest section.
 *
 * Note that in a correct kernel the second clause (both trackers point at the same Dest section) follows
 * from the first, since trackers advance only on commit and release; it is stated separately to
 * exclude a tracker reset from outside the handshake, described below.
 *
 * Quiescence is the handshake loop's precondition. A simple balanced loop body preserves it: each
 * complete iteration posts once, gets once, and advances both trackers once, so the loop exits
 * quiescent and another handshake loop can follow immediately.
 *
 * Note that "after the loop" is thread-relative. MATH and PACK are pipelined, so MATH leaves the
 * loop up to max sections ahead of PACK, and at that instant the state is not yet quiescent; it is
 * PACK's exit, the last one, that is quiescent.
 *
 * Please also note that quiescent does not mean "at section 0". A balanced loop with an odd iteration
 * count exits with both trackers on section 1, which is a valid entry state for the next loop.
 *
 * This function can be used as a in-kernel workaround for an compute-API operation that leaks state
 * that is not quiescence.
 * This function re-establishes the canonical quiescent state in precisely this case: it drains any outstanding
 * packs, re-seeds MATH_PACK, and returns both trackers to section 0 with `ocb` as the packer's destination.
 * To restore the pack MOP one needs to also add the llk_pack_init<PackMode::Default>.
 *
 * Caution: calling this function on a (buggy) kernel whose commits and releases do not balance will hang.
 *
 * Return value: None
 *
 * | Param Type | Name | Description                                       | Type     | Valid Range | Required |
 * |------------|------|---------------------------------------------------|----------|-------------|----------|
 * | Function   | ocb  | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void rearm_dest_sync([[maybe_unused]] uint32_t ocb) {
    LLK_SAN_FUNCTION();

#ifndef ARCH_QUASAR
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>(ocb)));
#else
    MATH((llk_math_pack_sync_init()));
    PACK((llk_pack_dest_init()));
#endif
}

// clang-format off
/**
 * Enables FP32 accumulation in the destination register.
 *
 * Configures both the math pipeline (ALU_ACC_CTRL Fp32_enabled and
 * SFPU_Fp32_enabled) and the packer (PCK_DEST_RD_CTRL Read_32b_data)
 * for 32-bit destination reads. UNPACK/PACK tensix_sync then notify MATH
 * and wait; MATH writes dest-acc CFG and releases them. Every thread
 * STALLWAITs on TRISC_CFG, blocking unpacker / packer / FPU / SFPU until
 * those writes are visible. Safe to call mid-kernel without re-running
 * compute_kernel_hw_startup.
 *
 * All three TRISC threads must call this together. TRISC mailboxes must not
 * be in use.
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
    UNPACK((llk_unpack_wait_fp32_dest_acc()));
    MATH((llk_math_set_fp32_dest_acc(true)));
    PACK((llk_pack_wait_fp32_dest_acc()));
}
#endif

// clang-format off
/**
 * Disables FP32 accumulation in the destination register, reverting to
 * BF16 accumulation mode.
 *
 * Configures both the math pipeline (ALU_ACC_CTRL Fp32_enabled and
 * SFPU_Fp32_enabled) and the packer (PCK_DEST_RD_CTRL Read_32b_data)
 * to disable 32-bit destination reads. UNPACK/PACK tensix_sync then notify
 * MATH and wait; MATH writes dest-acc CFG and releases them. Every thread
 * STALLWAITs on TRISC_CFG, blocking unpacker / packer / FPU / SFPU until
 * those writes are visible. Safe to call mid-kernel without re-running
 * compute_kernel_hw_startup.
 *
 * All three TRISC threads must call this together. TRISC mailboxes must not
 * be in use.
 *
 * Only available on Wormhole and Blackhole. Not supported on Quasar (compile error)
 *
 * Return value: None
 */
// clang-format on
#ifndef ARCH_QUASAR
ALWI void disable_fp32_dest_acc() {
    UNPACK((llk_unpack_wait_fp32_dest_acc()));
    MATH((llk_math_set_fp32_dest_acc(false)));
    PACK((llk_pack_wait_fp32_dest_acc()));
}
#endif

// clang-format off
/**
 * Sets FP32 destination accumulation to `enable` for a subsequent section
 * of the kernel by calling enable_fp32_dest_acc() or disable_fp32_dest_acc().
 *
 * Configures both the math pipeline (ALU_ACC_CTRL Fp32_enabled and
 * SFPU_Fp32_enabled) and the packer (PCK_DEST_RD_CTRL Read_32b_data).
 * This is a lightweight, standalone reconfiguration that is safe to call
 * mid-kernel without re-running compute_kernel_hw_startup.
 *
 * No-op when `enable` already matches the kernel's DST_ACCUM_MODE
 * (compute_kernel_hw_startup already programmed the requested mode).
 * Must be paired with restore_fp32_dest_acc<enable>() using the same flag.
 *
 * Only available on Wormhole and Blackhole. Not supported on Quasar (compile error)
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                                         | Type | Valid Range | Required |
 * |------------|--------|-----------------------------------------------------|------|-------------|----------|
 * | Template   | enable | Dest-acc mode the enclosed section needs            | bool | true, false | True     |
 */
// clang-format on
#ifndef ARCH_QUASAR
template <bool enable>
ALWI void set_fp32_dest_acc() {
    if constexpr (enable != static_cast<bool>(DST_ACCUM_MODE)) {
        if constexpr (enable) {
            enable_fp32_dest_acc();
        } else {
            disable_fp32_dest_acc();
        }
    }
}
#endif

// clang-format off
/**
 * Restores destination accumulation to the kernel's DST_ACCUM_MODE after a
 * matching set_fp32_dest_acc<enable>(), via enable_fp32_dest_acc() or
 * disable_fp32_dest_acc().
 *
 * Configures both the math pipeline (ALU_ACC_CTRL Fp32_enabled and
 * SFPU_Fp32_enabled) and the packer (PCK_DEST_RD_CTRL Read_32b_data).
 * This is a lightweight, standalone reconfiguration that is safe to call
 * mid-kernel without re-running compute_kernel_hw_startup.
 *
 * No-op when `enable` already matches DST_ACCUM_MODE (the matching set
 * was also a no-op). Pass the same `enable` used at the set.
 *
 * Only available on Wormhole and Blackhole. Not supported on Quasar (compile error)
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                                         | Type | Valid Range | Required |
 * |------------|--------|-----------------------------------------------------|------|-------------|----------|
 * | Template   | enable | Same flag passed to the matching set_fp32_dest_acc  | bool | true, false | True     |
 */
// clang-format on
#ifndef ARCH_QUASAR
template <bool enable>
ALWI void restore_fp32_dest_acc() {
    if constexpr (enable != static_cast<bool>(DST_ACCUM_MODE)) {
        if constexpr (DST_ACCUM_MODE) {
            enable_fp32_dest_acc();
        } else {
            disable_fp32_dest_acc();
        }
    }
}
#endif

}  // namespace ckernel
