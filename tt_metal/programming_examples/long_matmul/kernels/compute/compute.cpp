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
#include "api/debug/dprint.h"
#include "../profile_markers.h"
#include "../config.h"

using std::uint32_t;

// Which per-tile instruction the compute kernel runs, selected by the host from LONG_MATMUL_OP.
// Exactly one LONG_MATMUL_OP_* is defined; matmul is the default when the host passes none.
//
// The point of the switch is a controlled comparison: the reader and writer kernels are
// untouched, so every op streams byte-identical DRAM traffic through the same circular buffers,
// and the only thing that varies is what the math unit does with each tile pair. Every variant
// issues kBlockTiles math operations per step of the shared dimension and pops both input CBs.
// The rate is the same only while the reader sets the pace, which it does for the shapes tt-ember
// uses (on n150 at 1024x2048x2048, 1x1 and 2x4, matmul/add/exp have the same reader time and
// per-iteration time). A compute-bound shape would let the cheaper ops (add runs at LoFi, the
// unary ones are a copy plus one SFPU op) drain the CBs faster, and then the energy difference
// is no longer only the instruction.
//
// The binary ops (matmul, add) consume a tile from each CB. The unary SFPU ops have no second
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

// The whole output block lives in the destination registers at once.
static_assert(
    kBlockTiles <= ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>(),
    "BLOCK_M * BLOCK_N exceeds the destination register capacity");

#if defined(LONG_MATMUL_DISABLE_COMPUTE)
constexpr bool kComputeIdle = true;
#else
constexpr bool kComputeIdle = false;
#endif

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

    // All three TRISCs run this code, so every marker below is recorded three times per core, once
    // per TRISC. Each TRISC times only the calls it actually executes: wait is cb_wait_front on
    // TRISC0 (unpack), tile_regs_acquire on TRISC1 (math), and tile_regs_wait plus
    // cb_reserve_back(cb_out) on TRISC2 (pack); compute is the math loop on TRISC1 and pack_tile on
    // TRISC2. TRISC0's compute time is the unpacker blocking inside matmul_tiles / copy_tile.
    namespace prof = long_matmul_profile;
    DeviceTimestampedData("COMPUTE_KERNEL_START", /*data=*/0);
    uint64_t wait_cycles = 0;
    uint64_t compute_cycles = 0;
    [[maybe_unused]] uint32_t real_blocks_left = kRealStepsWhenIdle;  // only used when compute is idle

    constexpr uint32_t kProgressPrintInterval = 10;
    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        if (iter % kProgressPrintInterval == 0) {
            // Only print from TRISC0, otherwise we'll get three identical prints
            DPRINT_UNPACK("Iteration {} of {}\n", iter, num_iterations);
        }
        for (uint32_t blk = 0; blk < num_output_blocks; blk++) {
            // With compute idle, updated mode still computes and packs the first
            // kRealStepsWhenIdle blocks, so every slot of the output buffer the writer keeps
            // sending holds real results; legacy mode never does.
            bool compute = true;
            if constexpr (kComputeIdle) {
                compute = real_blocks_left > 0;
                real_blocks_left -= compute ? 1 : 0;
            }
            // TRISC1 acquires the tile registers -- kBlockTiles of them, one accumulator per
            // output tile in the block.
            uint32_t t_block = prof::now();
            tile_regs_acquire();
            prof::add_since(wait_cycles, t_block);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // The reader delivers a column slice of A (kBlockM tiles) and a row slice of B
                // (kBlockN tiles) for this step of the shared dimension.
                uint32_t t = prof::now();
                cb_wait_front(cb_in0, kBlockM);
                cb_wait_front(cb_in1, kBlockN);
                prof::add_since(wait_cycles, t);
                // For matmul and add every A tile is combined with every B tile in the slice, so
                // the two slices are reused kBlockN and kBlockM times respectively before being
                // popped; this is where the DRAM traffic reduction comes from. The unary ops only
                // read A and pop B unread.
                // LONG_MATMUL_DISABLE_COMPUTE skips the real math while keeping every CB and
                // tile-register handshake, so the reader and writer move the same data and
                // neither deadlocks. Without the math, unpack no longer feeds math, so the reader
                // and writer pace each other a little differently (case 2 vs case 1 on n150: a
                // few percent per iteration; see the README).
                if (compute) {
                    t = prof::now();
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
                prof::add_since(compute_cycles, t);
                }
                cb_pop_front(cb_in0, kBlockM);
                cb_pop_front(cb_in1, kBlockN);
            }
            // TRISC1 commits the results, transferring ownership of the tile registers to the packer (TRISC2)
            tile_regs_commit();
            // TRISC2 waits for the FPU (TRISC1) to be ready, then reserves space in the output
            // circular buffer for the whole block. Both are where the writer's back-pressure shows.
            t_block = prof::now();
            tile_regs_wait();
            cb_reserve_back(cb_out, kBlockTiles);
            prof::add_since(wait_cycles, t_block);
            if (compute) {
                const uint32_t t = prof::now();
                for (uint32_t i = 0; i < kBlockTiles; i++) {
                    pack_tile(i, cb_out);
                }
                prof::add_since(compute_cycles, t);
            }
            // TRISC2 marks the result tiles as used by pushing them to the back of the output
            // circular buffer (writer kernel can read them now)
            cb_push_back(cb_out, kBlockTiles);
            // TRISC2 releases the tile registers, allowing TRISC1 to acquire them again
            tile_regs_release();
        }
    }

    // The totals go out before KERNEL_END so they fall inside this run's marker window.
    DeviceTimestampedData("COMPUTE_WAIT_CYCLES", wait_cycles);
    DeviceTimestampedData("COMPUTE_COMPUTE_CYCLES", compute_cycles);
    DeviceTimestampedData("COMPUTE_KERNEL_END", /*data=*/0);
}
