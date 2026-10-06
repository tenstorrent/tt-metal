// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/compute_kernel_api.h"

using std::uint32_t;

// Output block handled per iteration, in tiles. BLOCK_M x BLOCK_N output tiles are accumulated
// in the destination registers simultaneously, so one row of A and one column of B serve
// BLOCK_M x BLOCK_N multiplies instead of one. 1x1 is the original tile-at-a-time behaviour.
#ifndef BLOCK_M
#define BLOCK_M 1
#endif
#ifndef BLOCK_N
#define BLOCK_N 1
#endif

// Which per-tile instruction the compute kernel runs, selected by the host from LONG_MATMUL_OP.
// Exactly one LONG_MATMUL_OP_* is defined; matmul is the default when the host passes none.
//
// The point of the switch is a controlled comparison: the reader and writer kernels are
// untouched, so every op streams byte-identical DRAM traffic through the same circular buffers
// at the same rate, and the only thing that varies is what the math unit does with each tile
// pair. The difference in dynamic energy between two ops is therefore attributable to the
// instruction rather than to data movement.
//
// To keep that true, every variant issues exactly kBlockTiles math operations per step of the
// shared dimension and pops both input CBs, matching matmul's rate. The binary ops (matmul,
// add) consume a tile from each CB the way matmul does. The unary SFPU ops have no second
// operand, so they copy from cb_in0 into the destination register and operate in place; cb_in1
// is still waited on and popped, so the reader's traffic and the pipeline's back-pressure are
// unchanged. Unary variants overwrite the destination register each step rather than
// accumulating into it, which matmul does -- that is a real difference, but it is a property of
// the instruction, not of this harness.
#if !defined(LONG_MATMUL_OP_MATMUL) && !defined(LONG_MATMUL_OP_ADD) && !defined(LONG_MATMUL_OP_SILU) && \
    !defined(LONG_MATMUL_OP_EXP) && !defined(LONG_MATMUL_OP_SIGMOID) && !defined(LONG_MATMUL_OP_GELU) && \
    !defined(LONG_MATMUL_OP_RECIP)
#define LONG_MATMUL_OP_MATMUL 1
#endif

constexpr uint32_t kBlockM = BLOCK_M;
constexpr uint32_t kBlockN = BLOCK_N;
constexpr uint32_t kBlockTiles = kBlockM * kBlockN;

void kernel_main() {
    uint32_t num_output_blocks = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);
    uint32_t num_iterations = get_arg_val<uint32_t>(2);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Each op has its own hardware configuration. compute_kernel_hw_startup configures the
    // unpacker/packer for the CBs involved; the op-specific *_init then sets up the math unit.
    // Without the startup call pack_tile() produces nothing, the writer blocks forever and the
    // device hangs, so it must stay in every branch.
#if defined(LONG_MATMUL_OP_MATMUL)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);
#elif defined(LONG_MATMUL_OP_ADD)
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    add_init(cb_in0, cb_in1);
#else
    compute_kernel_hw_startup(cb_in0, cb_out);
    copy_init(cb_in0);
#if defined(LONG_MATMUL_OP_SILU)
    silu_tile_init();
#elif defined(LONG_MATMUL_OP_EXP)
    exp_tile_init();
#elif defined(LONG_MATMUL_OP_SIGMOID)
    sigmoid_tile_init();
#elif defined(LONG_MATMUL_OP_GELU)
    gelu_tile_init();
#elif defined(LONG_MATMUL_OP_RECIP)
    recip_tile_init();
#endif
#endif

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        if (iter % 10 == 0) {
            // Only print from TRISC0, otherwise we'll get three identical prints
            DPRINT("Iteration {} of {}\n", iter, num_iterations);
        }
        for (uint32_t blk = 0; blk < num_output_blocks; blk++) {
            // TRISC1 acquires the tile registers -- kBlockTiles of them, one accumulator per
            // output tile in the block.
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // The reader delivers a column slice of A (kBlockM tiles) and a row slice of B
                // (kBlockN tiles) for this step of the shared dimension.
                cb_wait_front(cb_in0, kBlockM);
                cb_wait_front(cb_in1, kBlockN);
                // Every A tile is combined with every B tile in the slice, so the two slices
                // are reused kBlockN and kBlockM times respectively before being popped. This
                // is where the DRAM traffic reduction comes from.
                // LONG_MATMUL_DISABLE_COMPUTE skips the real math while keeping every CB and
                // tile-register handshake intact, so reader and writer are stimulated exactly as
                // before and neither deadlocks. Output tiles then contain garbage, which is
                // harmless: the app never verifies its output.
#ifndef LONG_MATMUL_DISABLE_COMPUTE
                for (uint32_t m = 0; m < kBlockM; m++) {
                    for (uint32_t n = 0; n < kBlockN; n++) {
                        const uint32_t idst = m * kBlockN + n;
#if defined(LONG_MATMUL_OP_MATMUL)
                        matmul_tiles(cb_in0, cb_in1, m, n, idst);
#elif defined(LONG_MATMUL_OP_ADD)
                        add_tiles(cb_in0, cb_in1, m, n, idst);
#else
                        // No second operand: bring a tile into the destination register and
                        // transform it in place.
                        copy_tile(cb_in0, m, idst);
#if defined(LONG_MATMUL_OP_SILU)
                        silu_tile(idst);
#elif defined(LONG_MATMUL_OP_EXP)
                        exp_tile(idst);
#elif defined(LONG_MATMUL_OP_SIGMOID)
                        sigmoid_tile(idst);
#elif defined(LONG_MATMUL_OP_GELU)
                        gelu_tile(idst);
#elif defined(LONG_MATMUL_OP_RECIP)
                        recip_tile(idst);
#endif
#endif
                    }
                }
#endif
                cb_pop_front(cb_in0, kBlockM);
                cb_pop_front(cb_in1, kBlockN);
            }
            // TRISC1 commits the results, transferring ownership of the tile registers to the packer (TRISC2)
            tile_regs_commit();
            // TRISC2 waits for the FPU (TRISC1) to be ready
            tile_regs_wait();
            // TRISC2 reserves space in the output circular buffer for the whole block
            cb_reserve_back(cb_out, kBlockTiles);
#ifndef LONG_MATMUL_DISABLE_COMPUTE
            for (uint32_t i = 0; i < kBlockTiles; i++) {
                pack_tile(i, cb_out);
            }
#endif
            // TRISC2 marks the result tiles as used by pushing them to the back of the output
            // circular buffer (writer kernel can read them now)
            cb_push_back(cb_out, kBlockTiles);
            // TRISC1 releases the tile registers, allowing TRISC0 to acquire them again
            tile_regs_release();
        }
    }
}
