// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Block-based compute kernel for mul-relu-overlap-block: y = relu(a * b).
//
// Sequential baseline (no FPU/SFPU overlap): for each block of BS tiles,
// run mul_tiles (FPU) for all BS slots, then relu_tile (SFPU) for all BS
// slots, all on TRISC1 (MATH). A future iteration will introduce overlap.
//
// The reader always pushes BS tiles per CB transaction (filling the tail
// block's unused slots with uninitialized data); the writer drops the unused
// tail slots. So this kernel only ever sees full BS-tile blocks.
//
// Compile-time args:
//   [0] BS (block size, tiles)
//
// Runtime args:
//   [0] num_blocks

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/relu.h"

#ifdef TRISC_PACK
#include "api/compute/eltwise_unary/relu.h"
#endif  // TRISC_PACK

ALWI void relu_packthread_tile_init() { PACK(SFPU_UNARY_KERNEL_INIT(relu_min, APPROX)); }

ALWI void relu_packthread_tile(uint32_t idst) {
    PACK(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0));
}

#define OVERLAP_MATH_PACK 1

void kernel_main() {
    constexpr uint32_t BS = get_compile_time_arg_val(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr auto c_a = tt::CBIndex::c_0;
    constexpr auto c_b = tt::CBIndex::c_1;
    constexpr auto c_out = tt::CBIndex::c_2;

    binary_op_init_common(c_a, c_b, c_out);

#ifdef OVERLAP_MATH_PACK
    relu_packthread_tile_init();

    // Initialize semaphores.
    //
    // NOTE: TTI_SEMINIT / TTI_SEMPOST / TTI_SEMGET / SemaphoreMask in TTI_SEMWAIT all take a
    // u8 *bitmask* (bit i selects semaphore i), not a semaphore index. semaphore::FPU_SFPU = 0
    // and semaphore::MATH_PACK = 1 are *indices*, so passing them raw means:
    //   - "TTI_SEMINIT(..., semaphore::FPU_SFPU)"  -> mask = 0  -> no-op
    //   - "TTI_SEMINIT(..., semaphore::MATH_PACK)" -> mask = 1  -> initializes sem 0 (FPU_SFPU)
    // The t6_semaphore_init helper applies the (1 << idx) conversion for us.
    //
    // FPU_SFPU: "PACK has freed a DST slot for MATH". Start with 1 free slot so MATH can begin
    // immediately; max=BS allows up to BS in-flight tiles between MATH and PACK.
    // MATH_PACK: "MATH has produced a DST slot for PACK". Start empty; PACK blocks on first
    // iteration until MATH posts.
    MATH(ckernel::t6_semaphore_init(semaphore::FPU_SFPU, /*value*/ 1, /*max*/ BS));
    PACK(ckernel::t6_semaphore_init(semaphore::MATH_PACK, /*value*/ 0, /*max*/ BS));
#endif

    DEVICE_PRINT("Starting kernel main\n");

    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_wait_front(c_a, BS);
        cb_wait_front(c_b, BS);
        cb_reserve_back(c_out, BS);

#ifdef OVERLAP_MATH_PACK

        // tile_regs_acquire() equivalent: wait for PACK to free a DST slot, then decrement.
        // STALL_MATH | STALL_SYNC blocks the MATH thread from starting new FPU and Sync Unit
        // instructions until the wait clears (B1=STALL_SYNC is required by the ISA when any
        // SEMWAIT is in flight; otherwise back-to-back SEMWAITs can supersede each other).
        MATH(ckernel::t6_semaphore_wait_on_zero < p_stall::STALL_MATH | p_stall::STALL_SYNC > (semaphore::FPU_SFPU));
        MATH(ckernel::t6_semaphore_get(semaphore::FPU_SFPU));

        mul_tiles_init(c_a, c_b);
        for (uint32_t i = 0; i < BS; i++) {
            mul_tiles(c_a, c_b, i, i, i);
            // tile_regs_commit() equivalent: wait for the FPU pipeline to drain, then signal PACK.
            // t6_semaphore_post<MATH> inserts the STALLWAIT(SYNC, MATH) for us before SEMPOST.
            MATH(ckernel::t6_semaphore_post<p_stall::MATH>(semaphore::MATH_PACK));
        }

        for (uint32_t i = 0; i < BS; i++) {
            // tile_regs_wait() equivalent on PACK: wait for MATH to produce, then decrement.
            // STALL_TDMA | STALL_CFG | STALL_SYNC blocks PACK from issuing further Sync Unit /
            // packer / cfg instructions until the wait clears.
            PACK(
                ckernel::t6_semaphore_wait_on_zero < p_stall::STALL_TDMA | p_stall::STALL_CFG |
                p_stall::STALL_SYNC > (semaphore::MATH_PACK));
            PACK(ckernel::t6_semaphore_get(semaphore::MATH_PACK));

            relu_packthread_tile(i);
        }

        // Wait for SFPU to drain before packing.
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);)
        for (uint32_t i = 0; i < BS; i++) {
            pack_tile(i, c_out);
        }

        // tile_regs_release() equivalent: wait for the packer to drain, zero the DST slots we
        // just consumed, then signal MATH that DST is free again.
        // ZEROACC is an FPU op issued from the PACK thread (same pattern as SDPA). Using
        // t6_semaphore_post<MATH> ensures we wait for the FPU pipeline (i.e. ZEROACC) to drain
        // before posting to FPU_SFPU, so MATH does not see a free slot before it actually is.
        PACK(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);)
        PACK(TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_1, 0););
        PACK(ckernel::t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU));

#else

        tile_regs_acquire();

        // FPU: A[i] * B[i] -> DST[i]
        mul_tiles_init(c_a, c_b);
        for (uint32_t i = 0; i < BS; ++i) {
            mul_tiles(c_a, c_b, i, i, i);
        }

        // SFPU: relu(DST[i]) in place
        relu_tile_init();
        for (uint32_t i = 0; i < BS; ++i) {
            relu_tile(i);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t i = 0; i < BS; ++i) {
            pack_tile(i, c_out);
        }
        tile_regs_release();
#endif

        cb_push_back(c_out, BS);
        cb_pop_front(c_a, BS);
        cb_pop_front(c_b, BS);
    }
}
