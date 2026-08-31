// SPDX-License-Identifier: Apache-2.0
//
// Heavy-compute resident kernel for validating host-read SFPU/FPU perf counters.
//
// WHY: every Blaze / deepseek single-device fused op is batch-1 decode (M=1 GEMV), so
// the FPU is busy ~1us per ~5300us launch — ~0.02% duty, which correctly reads 0.0%.
// Useless for answering "do the counters register compute inside a resident kernel?".
// This kernel does one launch then a long inner compute loop with no host round-trip.
//
// ---------------------------------------------------------------------------------
// HISTORY — READ BEFORE EDITING. The first version of this kernel HUNG THE BOARD
// TWICE (needed `tt-smi -r`). It called custom_mm_block() with no cb_wait_front, on
// the assumption that would read stale garbage. It does not: the unpacker BLOCKS
// waiting for tiles that are never pushed. A bounded max_iterations does NOT protect
// you — it only limits loop trips, not blocking inside the body.
//
// SAFE DESIGN RULES for a resident compute kernel measured this way:
//   1. No cb_wait_front / cb_reserve_back anywhere — every CB primitive can block.
//   2. No per-iteration tile_regs_acquire/commit — that syncs with the PACK thread,
//      which will stall if nothing ever packs.
//   3. Use SFPU ops that operate IN PLACE on a dest register (exp_tile(idst)) — they
//      take no CB input, so there is nothing to wait for.
//   4. Acquire dest ONCE outside the loop; only commit/release after the loop ends.
// Result: no synchronisation on the hot path, so the kernel cannot deadlock.
//
// This measures SFPU, not FPU. That is sufficient for the mechanism question: SFPU
// and FPU are the SAME counter block (FPU group, counter_sel 0 = FPU_COUNTER,
// 1 = SFPU_COUNTER), so if a resident kernel's SFPU work registers, the block is
// working inside a resident kernel. Adding real FPU matmul needs proper producer
// plumbing (a dataflow kernel pushing tiles) — do that separately, and test the
// dataflow half with a bounded non-resident launch FIRST.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/persistent_loop.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
// reg_api.h is NOT pulled in by compute_kernel_api.h — tile_regs_* live here. They are
// MATH()-wrapped so they compile to a no-op on the unpack/pack TUs.
#include "api/compute/reg_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#endif

void kernel_main() {
    constexpr uint32_t persistent_mode = get_named_compile_time_arg_val("persistent_mode");
    constexpr uint32_t termination_semaphore_addr = get_named_compile_time_arg_val("termination_semaphore_addr");
    constexpr uint32_t max_iterations = get_named_compile_time_arg_val("max_iterations");
    constexpr uint32_t iteration_count_addr = get_named_compile_time_arg_val("iteration_count_addr");
    constexpr uint32_t inner_iters = get_named_compile_time_arg_val("inner_iters");
    // burn_mode: 0 = SFPU (exp_tile, proven safe), 1 = FPU (matmul_tiles)
    constexpr uint32_t burn_mode = get_named_compile_time_arg_val("burn_mode");

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_out = 2;

    deepseek_b1_ops::PersistentLoop<persistent_mode == 1> loop(termination_semaphore_addr, max_iterations);

#if defined(COMPILE_FOR_TRISC)
    // Configure once, outside the loop.
    if constexpr (burn_mode == 1) {
        // FPU path. matmul_tiles() contains NO cb_wait_front — it is just
        // UNPACK(llk_unpack_AB_matmul) + MATH(llk_math_matmul) — so calling it without
        // CB flow control unpacks whatever is at the CB read pointer and issues real
        // FPU tile ops. Nothing waits on a producer, so it cannot deadlock.
        compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
        matmul_init(cb_in0, cb_in1, 0);
    } else {
        exp_tile_init();
    }
    // Acquire dest ONCE. No per-iteration regs handshake -> nothing to stall on.
    tile_regs_acquire();
#endif

    while (loop.next()) {
#if defined(COMPILE_FOR_BRISC)
        // Liveness proof, readable from the host via `ttact --peek <addr>`.
        volatile tt_l1_ptr uint32_t* count_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iteration_count_addr);
        *count_ptr = loop.iteration();
#endif
#if defined(COMPILE_FOR_TRISC)
        if constexpr (burn_mode == 1) {
            // FPU: back-to-back matmul tile ops, no CB waits, no pops.
            for (uint32_t i = 0; i < inner_iters; ++i) {
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
            }
        } else {
            // SFPU: in-place on dest reg 0. Takes no CB input, so it cannot wait.
            for (uint32_t i = 0; i < inner_iters; ++i) {
                exp_tile(0);
            }
        }
#endif
    }

#if defined(COMPILE_FOR_TRISC)
    tile_regs_commit();
    tile_regs_release();
#endif
}
