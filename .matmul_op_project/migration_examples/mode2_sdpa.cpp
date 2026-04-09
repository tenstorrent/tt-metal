// Migration Examples: Mode 2 (Semi-Automatic) -- SDPA kernels
// Call sites: B4, B5, B6, B7
//
// SDPA kernels use Mode 2 because they interleave custom operations
// (mask addition, reduction, custom pack patterns) between matmul accumulation
// and pack. They also use pack_tile<true> (out-of-order packing), which
// precludes Mode 3's sequential pack.
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul_op.h"
#include "api/compute/eltwise_binary.h"
#include "experimental/circular_buffer.h"

// ============================================================================
// B4: SDPA streaming -- blocked_matmul_and_pack
// Source: ttnn/.../transformer/sdpa/.../compute_streaming.hpp (lines 80-134)
//
// ORIGINAL CODE:
//   tile_regs_acquire();
//   for (inner) {
//       #ifdef ARCH_BLACKHOLE
//           matmul_block_no_mop(in0_cb, in1_cb, in0_index, in1_index, dst_index,
//                               transpose, subblock_w, subblock_h, matmul_stride);
//       #else
//           matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index,
//                        transpose, subblock_w, subblock_h, matmul_stride);
//       #endif
//       in0_index++; in1_index += in1_stride;
//   }
//   tile_regs_commit();
//   tile_regs_wait();
//   // pack_tile<true> at computed row-major positions
//   tile_regs_release();
//
// The key feature here is the arch-conditional no_mop path: on Blackhole,
// matmul_block_no_mop is used instead of matmul_block. MatmulOp handles this
// transparently via the use_no_mop config flag.
//
// NOTE: This call site uses pack_tile<true> (out-of-order pack), so we cannot
// use end_to_output(). We use Mode 1 matmul() within Mode 2 DST management.
// ============================================================================
namespace b4_sdpa_streaming {

template <bool transpose, uint32_t in1_stride, uint32_t out_num_cols, bool blocked_pack = false>
void blocked_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t row_subblock_idx,
    uint32_t out_col_offset,
    uint32_t subblock_w,
    uint32_t subblock_h,
    uint32_t inner_dim,
    uint32_t matmul_stride,
    bool trigger_reduce = false) {
    // --- NEW: MatmulOp configuration ---
    // Note: use_no_mop=true is set; on Blackhole it routes to matmul_block_no_mop,
    // on other architectures it falls through to matmul_block.
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = in0_cb,
        .in1_cb_id = in1_cb,
        .out_cb_id = out_cb,
        .ct_dim = subblock_w,
        .rt_dim = subblock_h,
        .kt_dim = matmul_stride,
        .transpose = transpose,
        .use_no_mop = true,  // SDPA streaming uses no_mop on Blackhole
    };
    ckernel::BlockMatmulOp mm(cfg);
    // NOTE: init_short() is called by the caller before this function, not here.
    // --- END NEW ---

    tile_regs_acquire();

    // --- NEW: accumulate replaces the inner dim loop ---
    // The original loop: for (inner) { matmul_block[_no_mop](...); in0_index++; in1_index += in1_stride; }
    // This maps directly to accumulate() with in1_stride as the stride parameter.
    uint32_t in0_index = in0_index_start;
    uint32_t in1_index = in1_index_start;
    for (uint32_t inner = 0; inner < inner_dim; ++inner) {
        mm.matmul(in0_index, in1_index, 0);
        in0_index++;
        in1_index += in1_stride;
    }
    // --- END NEW ---

    tile_regs_commit();

    // UNCHANGED: Custom pack with pack_tile<true> at computed positions
    tile_regs_wait();
    uint32_t dst_idx = 0;
#ifdef ARCH_BLACKHOLE
    if constexpr (blocked_pack) {
        for (uint32_t r = 0; r < subblock_h; r++) {
            uint32_t out_row_offset = (r + row_subblock_idx * subblock_h) * out_num_cols;
            pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset);
            dst_idx += subblock_w;
        }
    } else
#endif
    {
        for (uint32_t r = 0; r < subblock_h; r++) {
            uint32_t out_row_offset = (r + row_subblock_idx * subblock_h) * out_num_cols;
            for (uint32_t c = 0; c < subblock_w; c++) {
                pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                dst_idx++;
            }
        }
    }
    tile_regs_release();
}

}  // namespace b4_sdpa_streaming

// ============================================================================
// B5: SDPA matmul_blocks helper (with optional mask fusion)
// Source: ttnn/.../transformer/sdpa/.../compute_common.hpp (lines 1180-1262)
//
// ORIGINAL CODE:
//   mm_block_init_short(in0_cb, in1_cb, transpose, subblock_w, subblock_h, in0_block_w);
//   for (in0_subblock) for (in1_subblock) {
//       tile_regs_acquire();
//       for (inner_dim) { matmul_block(...); in0_index++; in1_index += N; }
//       if (add_mask) { add_tiles(zero_cb, mask_cb, ...) }
//       tile_regs_commit(); tile_regs_wait();
//       pack_tile<true>(...);  // out-of-order at row-major positions
//       tile_regs_release();
//   }
//
// MIGRATED: Uses MatmulOp Mode 1 matmul() + Mode 2 begin_subblock() for
// the DST management, but manual pack since it needs pack_tile<true>.
// The mask fusion is inserted between accumulate and commit.
// ============================================================================
namespace b5_sdpa_matmul_blocks {

void matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& num_blocks,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose,
    const bool& add_mask = false,
    const uint32_t& mask_cb = 0,
    const uint32_t& zero_cb = 0) {
    const uint32_t output_num_tiles = M * N;
    const uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    // --- NEW: MatmulOp configuration ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = in0_cb,
        .in1_cb_id = in1_cb,
        .out_cb_id = out_cb,
        .ct_dim = subblock_w,
        .rt_dim = subblock_h,
        .kt_dim = in0_block_w,
        .transpose = transpose,
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init_short();  // replaces mm_block_init_short(...)
    // --- END NEW ---

    uint32_t in0_index_offset = 0;
    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;

    reconfig_data_format(in1_cb, in0_cb);
    cb_wait_front(in1_cb, K * N);
    cb_reserve_back(out_cb, output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_wait_front(in0_cb, in0_wait_tiles);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            // --- NEW: accumulate replaces inner dim matmul_block loop ---
            mm.accumulate(
                in0_index_offset,
                in1_index_offset,
                /*dst_index_start=*/0,
                in0_block_w,  // inner_dim
                N);           // in1_stride = N (full output width)
            // --- END NEW ---

            // UNCHANGED: Mask fusion (inserted between accumulate and pack)
            if (add_mask) {
                cb_wait_front(mask_cb, out_subblock_num_tiles);
                cb_wait_front(zero_cb, 1);
                add_tiles_init(zero_cb, mask_cb, true);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    add_tiles(zero_cb, mask_cb, 0, i, i);
                }
            }

            // UNCHANGED: Custom pack with pack_tile<true> at computed row-major positions
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = r * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        cb_push_back(out_cb, subblock_h * N);
    }
    cb_pop_front(in1_cb, K * N);
}

}  // namespace b5_sdpa_matmul_blocks

// ============================================================================
// B6: SDPA matmul_reduce -- Mx1 reduction via matmul_block
// Source: ttnn/.../transformer/sdpa/.../compute_common.hpp (lines 1264-1316)
//
// ORIGINAL CODE:
//   mm_block_init_short(out_cb, in1_cb, 0, subblock_w, subblock_h, in0_block_w);
//   for (in0_subblock) {
//       tile_regs_acquire();
//       matmul_block(out_cb, in1_cb, 0, 0, 0, 0, subblock_w, subblock_h, in0_block_w);
//       tile_regs_commit();
//       cb_pop_front(out_cb, subblock_h);
//       tile_regs_wait();
//       pack_tile(i, out_cb);
//       tile_regs_release();
//       cb_push_back(out_cb, subblock_h);
//   }
//
// NOTE: The in0 CB is the same as out_cb (reuses the output CB as input).
// This is a single matmul_block call per subblock (not a loop), so Mode 1
// matmul() is the natural fit.
// ============================================================================
namespace b6_sdpa_matmul_reduce {

template <uint32_t M>
void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
    constexpr uint32_t N = 1;
    constexpr uint32_t in0_block_w = N;
    constexpr uint32_t subblock_w = N;
#ifdef STATS_GRANULARITY
    constexpr uint32_t subblock_h = STATS_GRANULARITY;
    constexpr uint32_t in0_num_subblocks = M / STATS_GRANULARITY;
#else
    constexpr uint32_t subblock_h = 1;
    constexpr uint32_t in0_num_subblocks = M;
#endif

    // --- NEW: MatmulOp for the Mx1 reduction ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = out_cb,  // NOTE: in0 = out_cb (reuse)
        .in1_cb_id = in1_cb,
        .out_cb_id = out_cb,
        .ct_dim = subblock_w,  // 1
        .rt_dim = subblock_h,
        .kt_dim = in0_block_w,  // 1
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init_short();  // replaces mm_block_init_short(...)
    // --- END NEW ---

    constexpr uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    reconfig_data_format(in1_cb, out_cb);
    cb_wait_front(in1_cb, N);
    cb_wait_front(out_cb, M);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        tile_regs_acquire();

        // --- NEW: single matmul call replaces matmul_block ---
        mm.matmul(/*in0_tile_index=*/0, /*in1_tile_index=*/0, /*dst_tile_index=*/0);
        // --- END NEW ---

        tile_regs_commit();
        cb_pop_front(out_cb, subblock_h);

        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_h; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, subblock_h);
    }
}

}  // namespace b6_sdpa_matmul_reduce

// ============================================================================
// B7: SDPA decode -- uses matmul_blocks wrapper
// Source: ttnn/.../transformer/sdpa_decode/.../sdpa_flash_decode.cpp
//
// This kernel calls the matmul_blocks() helper function from compute_common.hpp
// (same as B5) at two call sites (line 347 and 438). The migration is identical
// to B5 -- the matmul_blocks helper is migrated once, and both call sites
// benefit automatically.
//
// MIGRATION: Identical to B5 above. The SDPA decode kernel's flash attention
// flow (max, sum_exp, rescale) around the matmul calls is UNCHANGED.
// ============================================================================
namespace b7_sdpa_decode {
// Uses the same migrated matmul_blocks() helper as B5.
// Both call sites (QK^T matmul and PV matmul) pass through the same function.
}  // namespace b7_sdpa_decode
