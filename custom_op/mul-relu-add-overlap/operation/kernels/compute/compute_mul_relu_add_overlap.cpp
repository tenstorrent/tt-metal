// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Block-based compute kernel for mul-relu-add-overlap: y = relu(a * b) + c.
//
// Sequential baseline (no FPU/SFPU overlap yet): for each block of BS tiles,
// run mul_tiles (FPU) for all BS slots, then relu_tile (SFPU) for all BS
// slots, then DEST->SrcA reuse add for all BS slots, all on TRISC1 (MATH).
// A future iteration will move relu to PACK (TRISC2) to overlap with the
// surrounding FPU ops.
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

void kernel_main() {
    constexpr uint32_t BS = get_compile_time_arg_val(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr auto c_a = tt::CBIndex::c_0;
    constexpr auto c_b = tt::CBIndex::c_1;
    constexpr auto c_c = tt::CBIndex::c_2;
    constexpr auto c_out = tt::CBIndex::c_3;

    binary_op_init_common(c_a, c_b, c_out);

#define OVERLAP_MATH_PACK 1

#ifdef OVERLAP_MATH_PACK
    relu_packthread_tile_init();
#else
    relu_tile_init();
#endif

    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_wait_front(c_a, BS);
        cb_wait_front(c_b, BS);
        cb_wait_front(c_c, BS);
        cb_reserve_back(c_out, BS);

#ifdef OVERLAP_MATH_PACK
        // Unpacker: wait for FPU to complete
        // Similar to tile_regs_acquire()
        // but
        // TTI_SEMWAIT(STALL_MATH | STALL_SYNC, MATH_PACK, STALL_ON_MAX)
        MATH(TTI_SEMWAIT(p_stall::STALL_MATH, semaphore::t6_sem(semaphore::FPU_SFPU), p_stall::STALL_ON_ZERO););

        // Why does t6_semaphore_get() do STALLWAIT on STALL_SYNC ?

        // FPU: A[i] * B[i] -> DST[i]
        mul_tiles_init(c_a, c_b);
        for (uint32_t i = 0; i < BS; ++i) {
            mul_tiles(c_a, c_b, i, i, i);

            // Increment semaphore to SFPU by 1 ?
            // ~ set 1 tile as being done processed
            MATH(TTI_SEMPOST(semaphore::t6_sem(semaphore::MATH_PACK)););
        }

        // SFPU: relu(DST[i]) in place
        // SFPU: Flip Dst
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()););

        // ReLU block
        for (uint32_t i = 0; i < BS; ++i) {
            // Wait for FPU semaphore ~ wait for 1 tile

            PACK(TTI_SEMWAIT(p_stall::STALL_PACK, semaphore::t6_sem(semaphore::FPU_SFPU), p_stall::STALL_ON_ZERO););

            PACK(relu_packthread_tile(i);)

            // Notify FPU that tile is done
            PACK(TTI_SEMPOST(semaphore::t6_sem(semaphore::FPU_SFPU)););

            // Decrement STALL_PACK semaphore by 1
            PACK(TTI_SEMGET(semaphore::t6_sem(semaphore::FPU_SFPU)););
        }

        // FPU: DST[i] + C[i] -> DST[i] via DST->SrcA reuse
        // For each tile, wait for SFPU to complete - SEMWAIT.
        // Here, we stall the MATH Unit (?) for each tile
        // We do not need to wait for FPU to end; because this is transitively implied
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_c);
        for (uint32_t i = 0; i < BS; ++i) {
            // Wait for tile from SFPU
            // Question: How is unpacker synced ?
            MATH(TTI_SEMWAIT(p_stall::STALL_MATH, semaphore::t6_sem(semaphore::FPU_SFPU), p_stall::STALL_ON_ZERO);)

            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_c, i, i);

            // Stall Sync unit until FPU is done with this tile
            // Increment MATH_PACK semaphore by 1 once tile is done
            MATH(TTI_SEMPOST(semaphore::t6_sem(semaphore::MATH_PACK));)

            // FPU is done with this tile: decrement semaphore by 1
            MATH(TTI_SEMGET(semaphore::t6_sem(semaphore::FPU_SFPU));)
        }

        // Only pack once all SFPU operations are done
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);)

        // For each tile, wait for FPU semaphore
        // In this case, FPU increment semaphore by 1 for each tile; Packer thread decrement by 1 for each tile
        for (uint32_t i = 0; i < BS; ++i) {
            PACK(TTI_SEMWAIT(p_stall::STALL_PACK, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO););

            pack_tile(i, c_out);

            PACK(TTI_SEMGET(semaphore::t6_sem(semaphore::MATH_PACK)););  // Decrement semaphore by 1
        }

#else
        tile_regs_acquire();

        // FPU: A[i] * B[i] -> DST[i]
        mul_tiles_init(c_a, c_b);
        for (uint32_t i = 0; i < BS; ++i) {
            mul_tiles(c_a, c_b, i, i, i);
        }

        for (uint32_t i = 0; i < BS; ++i) {
            relu_tile(i);
        }

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_c);
        for (uint32_t i = 0; i < BS; ++i) {
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_c, i, i);
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
        cb_pop_front(c_c, BS);
    }
}
