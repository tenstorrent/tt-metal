// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for the Gemma 4 GeGLU routed-expert custom op (models/demos/gemma4/custom_ops/
// geglu_routed_expert): down(gelu_tanh(x @ gate) * (x @ up)) for every local expert, PACKER_L1_ACC variant.
//
// Copied from unified_routed_expert_ffn's fused_swiglu.cpp and adapted:
//   * the gate activation is gelu_tanh (gelu_tanh_tile on the MATH SFPU, between the copy out of the
//     gate accumulator and the pack into cb_gate_intermed) instead of silu;
//   * the other activation variants (SwiGLU-OAI, SiTU-GLU, clamped SiLU-GLU) and the expert-bias
//     (FUSE_BIAS) branch are gone: Gemma 4 has one activation and no expert biases.
// The matmul phases, the per-K-block packer-L1 accumulation, the constant CB block sizes and the
// runtime chunk picker (adaptive_chunk.hpp) are unchanged, so the reader and writer beside this file
// keep the row mapping the shared kernels agree on.
//
// Pattern (modelled on bmm_large_block_zm_fused_bias_activation.cpp), with one deviation shared by
// every matmul phase here: the partials CB is reserved ONCE for the whole per-core output block
// instead of per subblock.
//   * Each matmul phase has num_blocks K-blocks. dst is acquired/released once per (sb_m, sb_n)
//     subblock pair within each K-block, and the K-loop accumulates that subblock's per-output-tile
//     dot products into it.
//   * The packer is configured with L1_ACC=0 on block 0 (overwrite) and L1_ACC=1 on block 1+ (add
//     to existing L1). Each subblock packs to an ABSOLUTE partials slot (pack_tile<true>); WrPtr
//     does not advance until the single push_back after the K-loop, so every K-block's packs land
//     in the SAME L1 addresses and accumulate physically in place.
//   * Reserving once means no cb_push_back/wait_front round trip separates the K-blocks, so each
//     block ends with an explicit packer drain (drain_packer_before_reaccumulate).
//   * After the K-loop the partials CB is pushed once, holding the final accumulated sum, and a
//     second pass copies each subblock back into dst (applying gelu_tanh for the gate) and packs
//     into the final CB.
//
// Thread-private `mailbox_write`/`mailbox_read` broadcast the device-side count value read on the
// UNPACK thread to MATH and PACK so all three agree on `effective_chunks` without re-reading L1.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "adaptive_chunk.hpp"

namespace {

// Packer-completion barrier for the K-block boundary of a PACKER_L1_ACC phase.
//
// Every K-block re-packs the same absolute partials slots, and the accumulate is a packer-side L1
// read-modify-write. Pack issue must therefore stop until the packer has drained, or the next
// block's read can sample a slot the previous block's write has not reached yet -- a lost
// contribution, nondeterministic and worst on small per-core blocks.
FORCE_INLINE void drain_packer_before_reaccumulate() {
#ifdef PACKER_L1_ACC
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK));
#endif
}

// Single-matmul phase accumulating across K-blocks via PACKER_L1_ACC. Used for the down matmul.
template <
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_block_num_tiles,
    // Width of the LAST N-subblock. Equals out_subblock_w for an exact subblock grid; smaller when
    // the grid is ragged (op.py allows that only when no divisor of per_core_N clears the width where
    // this matmul stops being math-bound).
    uint32_t out_subblock_w_tail = out_subblock_w,
    // Real (unpadded) K-tiles of the reduction dim: no padded K position is ever reduced.
    uint32_t real_k_tiles = num_blocks * in0_block_w>
FORCE_INLINE void matmul_phase(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t partials_cb_id,
    uint32_t final_cb_id,
    uint32_t m_subblocks,
    uint32_t n_subblocks) {
    static_assert(out_subblock_h == 1, "m_subblocks bounds sb_m only when out_subblock_h == 1");
    static_assert(out_subblock_w_tail >= 1 && out_subblock_w_tail <= out_subblock_w, "bad tail width");
    static_assert(
        in0_num_subblocks * ((in1_num_subblocks - 1) * out_subblock_w + out_subblock_w_tail) * out_subblock_h ==
            out_block_num_tiles,
        "subblock grid must tile the output block exactly");
    static_assert(real_k_tiles <= num_blocks * in0_block_w, "real_k_tiles must not exceed the padded K extent");
    // Adaptive per_core_M: this core's down output is m_subblocks tile-rows this chunk. The partials
    // accumulator shrinks with it; the down WEIGHT block stays full-width. Rows past m_subblocks are
    // never written by the reader's (bounded) activated mcast, so they are simply not computed.
    //
    // n_subblocks bounds the OTHER free dim the same way: subblocks past it are wholly phantom output
    // columns (col >= N_down_tiles_full) whose weights the reader never fetched. Unlike the M bound
    // this does NOT shrink EFF_OUT -- the partials ring and the final_cb push stay full width, since
    // the writer drains a fixed subblock count.
    const uint32_t EFF_M = m_subblocks;
    constexpr uint32_t N_PER_ROW = (in1_num_subblocks - 1) * out_subblock_w + out_subblock_w_tail;
    const uint32_t EFF_OUT = m_subblocks * N_PER_ROW * out_subblock_h;
    pack_reconfig_data_format(partials_cb_id);
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // block 0 must overwrite, not accumulate
#endif

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer partials_cb(partials_cb_id);
    CircularBuffer final_cb(final_cb_id);

    // Reserve the partials block ONCE, and always the FULL block (never the runtime EFF_OUT):
    // partials_cb is single-buffered at exactly one max block, and the packs below use an absolute
    // output_tile_index off get_write_ptr(). Only [0, EFF_OUT) is ever packed; the unpacked tail is
    // drained by the pointer-only pop after the copy loop.
    partials_cb.reserve_back(out_block_num_tiles);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // The reader pushes the FULL compile-time-max activated block (filling only its first
        // per_core_M tile-rows), so this wait/pop stays compile-time sized.
        in0_cb.wait_front(in0_block_num_tiles);
        in1_cb.wait_front(in1_block_num_tiles);

        // Reduce only the REAL K-tiles this block covers: neither operand is zero-filled past
        // real_k_tiles, and a reduction position contaminates EVERY output column. kt_dim stays
        // in0_block_w (the in0 row stride), not the step count.
        const uint32_t k_done = block * in0_block_w;
        const uint32_t k_left = k_done < real_k_tiles ? real_k_tiles - k_done : 0;
        const uint32_t k_steps = k_left < in0_block_w ? k_left : in0_block_w;

        // A block entirely past real_k_tiles contributes nothing: skip its MAC AND its pack (with
        // L1_ACC on, packing an untouched dst would ADD garbage). Its CBs are still drained below.
        if (k_steps > 0) {
            int in0_index_subblock_offset = 0;
            uint32_t partials_slot_idx = 0;
            for (uint32_t sb_m = 0; sb_m < EFF_M; ++sb_m) {
                int in1_index_subblock_offset = 0;
                for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
                    // Ragged grid: the last N-subblock is narrower. Phantom-column subblocks: no MAC,
                    // no pack; the slot counters still advance.
                    const uint32_t this_w = (sb_n + 1 == in1_num_subblocks) ? out_subblock_w_tail : out_subblock_w;
                    if (sb_n < n_subblocks) {
                        tile_regs_acquire();
                        {
                            uint32_t in0_index = in0_index_subblock_offset;
                            uint32_t in1_index = in1_index_subblock_offset;
                            for (uint32_t inner_dim = 0; inner_dim < k_steps; ++inner_dim) {
                                matmul_block(
                                    in0_cb_id,
                                    in1_cb_id,
                                    in0_index,
                                    in1_index,
                                    /*dst_index=*/0,
                                    /*transpose=*/0,
                                    this_w,
                                    out_subblock_h,
                                    in0_block_w);
                                in0_index += 1;
                                in1_index += in1_per_core_w;
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < this_w * out_subblock_h; ++i) {
                            pack_tile<true>(i, partials_cb_id, partials_slot_idx + i);
                        }
                        tile_regs_release();
                    }
                    partials_slot_idx += this_w * out_subblock_h;

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }
        }

#ifdef PACKER_L1_ACC
        if (block == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
#endif
        drain_packer_before_reaccumulate();

        in0_cb.pop_front(in0_block_num_tiles);
        in1_cb.pop_front(in1_block_num_tiles);
    }
    partials_cb.push_back(out_block_num_tiles);

    // After the K-loop: move the accumulated partials through dst into final_cb.
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // future packs (to final_cb) must overwrite
#endif
    pack_reconfig_data_format(final_cb_id);
    // matmul puts in1 -> SrcA, in0 -> SrcB. Reconfigure SrcA from in1 to partials so copy_tile
    // reads partials.
    reconfig_data_format_srca(in1_cb_id, partials_cb_id);
    copy_init(partials_cb_id);

    // Walk (sb_m, sb_n) rather than a flat subblock count: on a ragged grid the last N-subblock is
    // narrower, so the drain must pop and push exactly what the MAC packed.
    for (uint32_t sb_m = 0; sb_m < EFF_M; ++sb_m) {
        for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
            const uint32_t this_w = (sb_n + 1 == in1_num_subblocks) ? out_subblock_w_tail : out_subblock_w;
            const uint32_t this_tiles = this_w * out_subblock_h;
            tile_regs_acquire();
            partials_cb.wait_front(this_tiles);
            for (uint32_t i = 0; i < this_tiles; ++i) {
                copy_tile(partials_cb_id, i, i);
            }
            partials_cb.pop_front(this_tiles);
            tile_regs_commit();
            tile_regs_wait();
            final_cb.reserve_back(this_tiles);
            for (uint32_t i = 0; i < this_tiles; ++i) {
                pack_tile(i, final_cb_id);
            }
            final_cb.push_back(this_tiles);
            tile_regs_release();
        }
    }

    // Pointer-only drain of the partials tail: slots [EFF_OUT, out_block_num_tiles) were never
    // packed this chunk. Popping them keeps the single-buffered partials ring block-aligned.
    if (EFF_OUT < out_block_num_tiles) {
        const uint32_t partials_pad = out_block_num_tiles - EFF_OUT;
        partials_cb.wait_front(partials_pad);
        partials_cb.pop_front(partials_pad);
    }

    // The writer drains final_cb at the compile-time-MAX subblock count and drops the rows past the
    // runtime per_core_M via its own row guards. Hand it the leftover slots WITHOUT packing, in the
    // SAME ragged width sequence it waits on.
    for (uint32_t sb_m = EFF_M; sb_m < in0_num_subblocks; ++sb_m) {
        for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
            const uint32_t pad_tiles =
                ((sb_n + 1 == in1_num_subblocks) ? out_subblock_w_tail : out_subblock_w) * out_subblock_h;
            final_cb.reserve_back(pad_tiles);
            final_cb.push_back(pad_tiles);
        }
    }
}

// Fused gate+up matmul phase. Per K-block: wait on x, gate, up; for each (sb_m, sb_n) subblock do
// TWO matmul_block sequences on the SAME x K-block (x*gate -> partials_gu, x*up -> partials_up);
// pop x, gate, up once per K-block. After the K-loop, partials_gu -> gate_intermed through dst with
// gelu_tanh applied on the MATH SFPU; partials_up stays in its CB (multiply_phase reads it directly).
template <
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_block_num_tiles,
    // x_cb_id / x_rm_cb_id are compile-time so the tilize helper can consume them.
    uint32_t x_cb_id,
    uint32_t x_rm_cb_id,
    bool tilize_x = false>
FORCE_INLINE void matmul_phase_fused_gu(
    uint32_t gate_cb_id,
    uint32_t up_cb_id,
    uint32_t partials_gu_cb_id,
    uint32_t partials_up_cb_id,
    uint32_t gate_intermed_cb_id,
    uint32_t m_subblocks,
    uint32_t n_subblocks) {
    // No real_k_tiles bound here: the host asserts K_gate_tiles % in0_block_w == 0, so the gate/up
    // K-loop covers the reduction dim exactly. This phase's padding is all in N (the hidden dim),
    // where a matmul is column-independent -- the unwritten weight columns past N_gate_tiles_full
    // corrupt only their OWN output columns, which the down phase excludes from its K reduction.
    //
    // Adaptive per_core_M bounds the WORK (MACs, tilize strips, copies, packs). The CB BLOCK SIZE is
    // the compile-time MAX and never varies (see adaptive_chunk.hpp): every reserve/push/wait/pop
    // below moves a full max block, and the runtime remainder is padded with pointer-only bumps.
    const uint32_t EFF_M = m_subblocks;
    const uint32_t x_block_tiles = m_subblocks * in0_block_w;
    const uint32_t EFF_OUT = m_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    constexpr uint32_t X_BLOCK_TILES_MAX = in0_block_num_tiles;
    constexpr uint32_t EFF_OUT_MAX = out_block_num_tiles;
    pack_reconfig_data_format(partials_gu_cb_id);
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

    CircularBuffer x_cb(x_cb_id);
    CircularBuffer gate_cb(gate_cb_id);
    CircularBuffer up_cb(up_cb_id);
    CircularBuffer partials_gu_cb(partials_gu_cb_id);
    CircularBuffer partials_up_cb(partials_up_cb_id);
    CircularBuffer gate_intermed_cb(gate_intermed_cb_id);

    // Reserve both partials CBs once for the FULL max block (absolute-slot packs, see above).
    partials_gu_cb.reserve_back(EFF_OUT_MAX);
    partials_up_cb.reserve_back(EFF_OUT_MAX);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        if constexpr (tilize_x) {
            // Row-major x: tilize this K-block's cb_x_rm strips (bf16) -> x_cb (bf8_b) before the
            // matmul consumes it. L1_ACC off so the tilize packs OVERWRITE; the shared tilize helper
            // reconfigures unpack SrcA + pack format, drives the per-strip wait/reserve/tilize/push/pop
            // over the runtime rows, and restores init on exit. Then restore SrcA to the gate weight
            // format and the partials packer + L1_ACC state for this block.
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(0)));
#endif
            const uint32_t n_strips = x_block_tiles / in0_block_w;
            compute_kernel_lib::tilize<
                in0_block_w,
                x_rm_cb_id,
                x_cb_id,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(n_strips);
            // The helper consumed/produced only the runtime rows, but the reader pushed a full max
            // cb_x_rm block and the matmul below pops a full max cb_in0_x block. Settle both with
            // pointer-only bumps so every CB moves a constant-size block.
            {
                const uint32_t x_pad = X_BLOCK_TILES_MAX - x_block_tiles;
                if (x_pad > 0) {
                    CircularBuffer x_rm_cb(x_rm_cb_id);
                    x_rm_cb.wait_front(x_pad);
                    x_rm_cb.pop_front(x_pad);
                    x_cb.reserve_back(x_pad);
                    x_cb.push_back(x_pad);
                }
            }
            reconfig_data_format_srca(gate_cb_id);
            matmul_block_init(x_cb_id, gate_cb_id, 0, out_subblock_w, out_subblock_h, in0_block_w);
            pack_reconfig_data_format(x_cb_id, partials_gu_cb_id);
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
#endif
        }
        x_cb.wait_front(X_BLOCK_TILES_MAX);
        gate_cb.wait_front(in1_block_num_tiles);
        up_cb.wait_front(in1_block_num_tiles);

        int in0_index_subblock_offset = 0;
        uint32_t partials_slot_idx = 0;
        for (uint32_t sb_m = 0; sb_m < EFF_M; ++sb_m) {
            int in1_index_subblock_offset = 0;
            for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
                // Phantom-column subblocks (col >= N_gate_tiles_full): skip the MAC and the pack; the
                // slot counters still advance so the surviving subblocks keep their absolute slots.
                if (sb_n < n_subblocks) {
                    // --- Gate matmul: x * gate -> partials_gu ---
                    tile_regs_acquire();
                    {
                        uint32_t in0_index = in0_index_subblock_offset;
                        uint32_t in1_index = in1_index_subblock_offset;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                            matmul_block(
                                x_cb_id,
                                gate_cb_id,
                                in0_index,
                                in1_index,
                                /*dst_index=*/0,
                                /*transpose=*/0,
                                out_subblock_w,
                                out_subblock_h,
                                in0_block_w);
                            in0_index += 1;
                            in1_index += in1_per_core_w;
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                        pack_tile<true>(i, partials_gu_cb_id, partials_slot_idx + i);
                    }
                    tile_regs_release();

                    // --- Up matmul: x * up -> partials_up (same x, different in1) ---
                    tile_regs_acquire();
                    {
                        uint32_t in0_index = in0_index_subblock_offset;
                        uint32_t in1_index = in1_index_subblock_offset;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                            matmul_block(
                                x_cb_id,
                                up_cb_id,
                                in0_index,
                                in1_index,
                                /*dst_index=*/0,
                                /*transpose=*/0,
                                out_subblock_w,
                                out_subblock_h,
                                in0_block_w);
                            in0_index += 1;
                            in1_index += in1_per_core_w;
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                        pack_tile<true>(i, partials_up_cb_id, partials_slot_idx + i);
                    }
                    tile_regs_release();
                }
                partials_slot_idx += out_subblock_num_tiles;

                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

#ifdef PACKER_L1_ACC
        if (block == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
#endif
        drain_packer_before_reaccumulate();

        x_cb.pop_front(X_BLOCK_TILES_MAX);
        gate_cb.pop_front(in1_block_num_tiles);
        up_cb.pop_front(in1_block_num_tiles);
    }
    partials_gu_cb.push_back(EFF_OUT_MAX);
    partials_up_cb.push_back(EFF_OUT_MAX);

#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

    // Gate partials -> gate_intermed with gelu_tanh applied on the MATH-thread SFPU on dst (NOT
    // packer-fused): copy partials_gu -> dst, gelu_tanh_tile on each dst tile (overlapping the next
    // subblock's UNPACK rather than gating the pack pipeline), pack dst -> gate_intermed.
    pack_reconfig_data_format(gate_intermed_cb_id);
    // SrcA was last configured for the up matmul's in1 (up_cb_id). Switch to partials_gu so copy_tile
    // reads the accumulator.
    reconfig_data_format_srca(up_cb_id, partials_gu_cb_id);
    copy_init(partials_gu_cb_id);
    for (uint32_t sb = 0; sb < (EFF_OUT / out_subblock_num_tiles); ++sb) {
        tile_regs_acquire();
        partials_gu_cb.wait_front(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            copy_tile(partials_gu_cb_id, i, i);
        }
        partials_gu_cb.pop_front(out_subblock_num_tiles);
        // GeGLU: gelu_tanh(gate) on each dst tile before the pack (SiLU in the shared kernel).
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            gelu_tanh_tile(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        gate_intermed_cb.reserve_back(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, gate_intermed_cb_id);
        }
        gate_intermed_cb.push_back(out_subblock_num_tiles);
        tile_regs_release();
    }
    // Settle the rest of the constant-size block with pointer-only bumps: drop the stale tail of
    // partials_gu and pad gate_intermed so multiply_phase sees a full max block.
    {
        const uint32_t pad = EFF_OUT_MAX - EFF_OUT;
        if (pad > 0) {
            partials_gu_cb.wait_front(pad);
            partials_gu_cb.pop_front(pad);
            gate_intermed_cb.reserve_back(pad);
            gate_intermed_cb.push_back(pad);
        }
    }
    // Up partials are NOT copied to a separate CB: multiply_phase reads cb_partials_up directly
    // (bf16) and pairs each tile with cb_gate_intermed (bf8 after gelu_tanh + pack).
}

// activated = gate_intermed * partials_up, elementwise. eff_out_tiles = per_core_M * per_core_N_gu
// (runtime-adaptive), precomputed by the caller.
template <uint32_t out_block_num_tiles, uint32_t out_subblock_num_tiles>
FORCE_INLINE void multiply_phase(
    uint32_t gate_cb_id, uint32_t up_cb_id, uint32_t activated_cb_id, uint32_t eff_out_tiles) {
    CircularBuffer gate_cb(gate_cb_id);
    CircularBuffer up_cb(up_cb_id);
    CircularBuffer activated_cb(activated_cb_id);

    const uint32_t EFF_OUT = eff_out_tiles;
    constexpr uint32_t EFF_OUT_MAX = out_block_num_tiles;

    gate_cb.wait_front(EFF_OUT_MAX);
    up_cb.wait_front(EFF_OUT_MAX);

    // Reconfigure the packer for the activated format and the unpacker for both gate_cb (SrcA) and
    // up_cb (SrcB): mul_init only reprograms the unpack MOP, not the data formats, and SrcB still
    // points at cb_in0_x (bf8) from the matmul.
    pack_reconfig_data_format(activated_cb_id);
    reconfig_data_format(gate_cb_id, up_cb_id);
    mul_init(gate_cb_id, up_cb_id);

    const uint32_t num_subblocks = EFF_OUT / out_subblock_num_tiles;
    uint32_t base = 0;
    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            mul_tiles(gate_cb_id, up_cb_id, base + i, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        activated_cb.reserve_back(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, activated_cb_id);
        }
        activated_cb.push_back(out_subblock_num_tiles);
        tile_regs_release();
        base += out_subblock_num_tiles;
    }
    // Pad cb_activated up to the constant block (pointer-only); the reader drains the full max
    // block and mcasts only the runtime rows.
    {
        const uint32_t pad = EFF_OUT_MAX - EFF_OUT;
        if (pad > 0) {
            activated_cb.reserve_back(pad);
            activated_cb.push_back(pad);
        }
    }
    gate_cb.pop_front(EFF_OUT_MAX);
    up_cb.pop_front(EFF_OUT_MAX);
}

}  // namespace

void kernel_main() {
    // Per-core valid N-subblock counts: per_core_N is the GRID-ceil'd width, so the highest-gx cores
    // own phantom output columns whose weights were never fetched.
    const uint32_t gu_valid_n_subblocks = get_arg_val<uint32_t>(0);
    const uint32_t d_valid_n_subblocks = get_arg_val<uint32_t>(1);

    // Phase 1 (gate)
    constexpr uint32_t g_in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t g_in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t g_in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t g_in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t g_in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t g_in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t g_in1_per_core_w = get_compile_time_arg_val(6);
    constexpr uint32_t g_num_blocks = get_compile_time_arg_val(7);
    // Phase 2 (up): args 8..15 mirror the gate block (same x, same N split) and are read for parity
    // with the host arg list; the fused phase uses the gate values for both matmuls.
    // Phase 4 (down)
    constexpr uint32_t d_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t d_in0_num_subblocks = get_compile_time_arg_val(17);
    constexpr uint32_t d_in0_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t d_in0_subblock_num_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t d_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t d_in1_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t d_in1_per_core_w = get_compile_time_arg_val(22);
    constexpr uint32_t d_num_blocks = get_compile_time_arg_val(23);
    // Subblock dims
    constexpr uint32_t gu_out_subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t gu_out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t gu_out_block_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(27);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(28);
    constexpr uint32_t d_out_subblock_w_tail = get_compile_time_arg_val(29);
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_out_block_num_tiles = get_compile_time_arg_val(30);
    // Chunk loop control: num_chunks_max is the compile-time upper bound (clamp); chunk_M_max the
    // CB-sized maximum chunk (per_core_M_max * GRID_Y) the runtime picker never exceeds.
    constexpr uint32_t num_chunks_max = get_compile_time_arg_val(31);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(32);
    constexpr uint32_t chunk_M_max = get_compile_time_arg_val(33);
    // x_is_row_major: tilize cb_x_rm -> cb_in0_x before the gate/up matmul; 0 => x already TILE.
    constexpr uint32_t x_is_row_major = get_compile_time_arg_val(34);
    // Real (unpadded) down-K tiles: the down phase skips every K position at or past this count.
    constexpr uint32_t d_K_down_tiles = get_compile_time_arg_val(35);
    constexpr uint32_t min_active_tokens = get_compile_time_arg_val(36);
    constexpr uint32_t max_active_tokens = get_compile_time_arg_val(37);

    // CBs
    constexpr uint32_t cb_in0_x = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_in1_up = get_named_compile_time_arg_val("cb_in1_up");
    constexpr uint32_t cb_in1_down = get_named_compile_time_arg_val("cb_in1_down");
    constexpr uint32_t cb_gate_intermed = get_named_compile_time_arg_val("cb_gate_intermed");
    constexpr uint32_t cb_activated = get_named_compile_time_arg_val("cb_activated");
    constexpr uint32_t cb_in0_down_full = get_named_compile_time_arg_val("cb_in0_down_full");
    constexpr uint32_t cb_partials_gu = get_named_compile_time_arg_val("cb_mm_partials_gu");
    constexpr uint32_t cb_partials_up = get_named_compile_time_arg_val("cb_mm_partials_up");
    constexpr uint32_t cb_partials_d = get_named_compile_time_arg_val("cb_mm_partials_d");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_counts_scratch = get_named_compile_time_arg_val("cb_counts_scratch");
    constexpr uint32_t cb_idx_scratch = get_named_compile_time_arg_val("cb_idx_scratch");
    // This expert's region size in tile-rows; caps the device-provided count.
    constexpr uint32_t m_tiles_full = get_named_compile_time_arg_val("m_tiles_full");
    // Row-major bf16 x staging (x_is_row_major only); tilize input CB.
    constexpr uint32_t cb_x_rm = get_named_compile_time_arg_val("cb_x_rm");

    // First Compute API call in the kernel (MMIO config of MATH/PACK/UNPACK; needs those units idle).
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0_x, cb_in1_gate, cb_partials_gu);

    CircularBuffer counts_scratch_cb(cb_counts_scratch);
    CircularBuffer idx_scratch_cb(cb_idx_scratch);

    // Wait for the reader (NCRISC) to push the counts/idx into shared L1: pushed ONCE, resident.
    counts_scratch_cb.wait_front(1);
    idx_scratch_cb.wait_front(1);

    // gelu_tanh is applied as a MATH-thread SFPU pass on dst (gelu_tanh_tile) between copy_tile and
    // pack_tile, exactly where the shared kernel applies silu. Init once, shared across all experts.
    gelu_tanh_tile_init();

    // ======================= per-local-expert loop =======================
    for (uint32_t local_expert_id = 0; local_expert_id < experts_per_chip; ++local_expert_id) {
        // This expert's token count via the UNPACK->{MATH,PACK} mailbox (MATH/PACK cannot read the
        // counts/idx L1 via the CB interface). count=0 => the chunk loop is skipped entirely.
        uint32_t count_value = 0;
        UNPACK(({
            const uint32_t counts_l1_addr = get_local_cb_interface(cb_counts_scratch).fifo_rd_ptr << 4;
            const uint32_t idx_l1_addr = get_local_cb_interface(cb_idx_scratch).fifo_rd_ptr << 4;
            const volatile tt_l1_ptr uint32_t* counts_ptr =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counts_l1_addr);
            const volatile tt_l1_ptr uint32_t* idx_ptr =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1_addr);
            const uint32_t global_expert_id = idx_ptr[local_expert_id];
            count_value =
                adaptive_chunk::count_in_band(counts_ptr[global_expert_id], min_active_tokens, max_active_tokens);
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, count_value);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, count_value);
        }));
        MATH(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        // Token rows -> tile rows (ceil), clamped to this program's capacity (arithmetic, so the
        // bound holds in Release where ASSERT is a no-op); reader and writer apply the same clamp.
        const uint32_t count_tiles_raw = (count_value + 31) / 32;
        const uint32_t count_tiles =
            adaptive_chunk::clamp_count_tiles(count_tiles_raw, chunk_M_max, num_chunks_max, m_tiles_full);
        ASSERT(count_tiles == count_tiles_raw);
        const uint32_t effective_chunks = adaptive_chunk::num_chunks(count_tiles, chunk_M_max);

        for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
            // Per-chunk per_core_M (per_core_M_max for full chunks, a smaller divisor for the tail).
            const uint32_t re_m_valid = adaptive_chunk::per_core_M_for_chunk(chunk, count_tiles, chunk_M_max);
            const uint32_t re_eff_out_gu = re_m_valid * g_in1_num_subblocks * gu_out_subblock_num_tiles;
            // matmul_block_init only re-programs addressing, not SrcA/SrcB formats: on chunk >= 1
            // the unpacker is left on multiply_phase's operands, so reset it to the gate/up inputs.
            reconfig_data_format(cb_in1_gate, cb_in0_x);
            matmul_block_init(
                cb_in0_x,
                cb_in1_gate,
                /*transpose=*/0,
                gu_out_subblock_w,
                gu_out_subblock_h,
                g_in0_block_w);

            // Phases 1 & 2 fused: gate and up matmuls share the per-K-block x push.
            matmul_phase_fused_gu<
                g_in0_block_w,
                g_in0_num_subblocks,
                g_in0_block_num_tiles,
                g_in0_subblock_num_tiles,
                g_in1_num_subblocks,
                g_in1_block_num_tiles,
                g_in1_per_core_w,
                g_num_blocks,
                gu_out_subblock_h,
                gu_out_subblock_w,
                gu_out_subblock_num_tiles,
                gu_out_block_num_tiles,
                /*x_cb_id=*/cb_in0_x,
                /*x_rm_cb_id=*/cb_x_rm,
                /*tilize_x=*/(x_is_row_major != 0)>(
                cb_in1_gate,
                cb_in1_up,
                cb_partials_gu,
                cb_partials_up,
                cb_gate_intermed,
                /*m_subblocks=*/re_m_valid,
                /*n_subblocks=*/gu_valid_n_subblocks);

            // Phase 3: activated = gelu_tanh(gate) (cb_gate_intermed, bf8) * up (cb_partials_up, bf16).
            multiply_phase<gu_out_block_num_tiles, gu_out_subblock_num_tiles>(
                cb_gate_intermed, cb_partials_up, cb_activated, re_eff_out_gu);

            // Phase 4: down matmul, output to cb_out. Reset the down operands (in1 -> SrcA, in0 ->
            // SrcB) before the matmul: multiply_phase left the unpacker on its own operands.
            reconfig_data_format(cb_in1_down, cb_in0_down_full);
            matmul_block_init(
                cb_in0_down_full,
                cb_in1_down,
                /*transpose=*/0,
                d_out_subblock_w,
                d_out_subblock_h,
                d_in0_block_w);
            matmul_phase<
                d_in0_block_w,
                d_in0_num_subblocks,
                d_in0_block_num_tiles,
                d_in0_subblock_num_tiles,
                d_in1_num_subblocks,
                d_in1_block_num_tiles,
                d_in1_per_core_w,
                d_num_blocks,
                d_out_subblock_h,
                d_out_subblock_w,
                d_out_subblock_num_tiles,
                d_out_block_num_tiles,
                /*out_subblock_w_tail=*/d_out_subblock_w_tail,
                /*real_k_tiles=*/d_K_down_tiles>(
                cb_in0_down_full,
                cb_in1_down,
                cb_partials_d,
                cb_out,
                /*m_subblocks=*/re_m_valid,
                /*n_subblocks=*/d_valid_n_subblocks);
        }  // end chunk loop
    }  // end per-local-expert loop
}
