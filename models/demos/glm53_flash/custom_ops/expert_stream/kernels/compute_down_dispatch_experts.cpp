// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down projection per (unit, pass) + gathered token sums (TRISC) of the per-expert dispatch op in its
// EXPERT-PARALLEL split (custom_ops/expert_stream/op.py, ExpertDownDispatch(down_split="experts")).
//
// Phase 1 (this core's units and their pass counts from the writer's metadata page; a unit's W2 block of
// `unit_cols` columns stays resident over the expert's passes):
//   h_k[:, unit cols] = act[k] @ W2[e][:, q unit_cols ..]   (seg_tiles products per column into DST, fp32)
// packed BF16 into cb_hs (unit_cols tiles per pass), from where the writer scatters the pass rows into the owning
// cores' gather slots. The per-column product order (K tiles ascending into one fp32 DST tile, HiFi4) is the one of
// compute_down_dispatch.cpp, so every h tile is bit-identical to the column-split form's. The unit's columns go
// through DST in batches of `dst_batch` tiles (op.py DST_FP32_HALF_TILES = 4): in the default half-sync mode the
// math thread owns HALF of the 8-fp32-tile Blackhole DEST per tile_regs_acquire while the packer drains the other
// half, so 8 fp32 tiles per acquire overwrite the half being packed (the 2026-11-03 exactness failure: 8-column
// units wrong, 4-column units bit-exact). A batch = acquire, fill, the batch's columns, commit, pack, release.
// Phase 2 (per token row tile, used slot count from the mailbox; this core's `cols_own` output columns):
//   out[r] = sum_j G[j]   as identity-tile matmuls accumulated in fp32 DST (exact), packed once -- unchanged.
// REGIONS_SRC: the unit block arrives k-major (tile k * unit_cols + c: reader_down_dispatch_experts.cpp).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ckernel.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_hs = get_compile_time_arg_val(3);
    constexpr uint32_t cb_g = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t unit_cols = get_compile_time_arg_val(7);
    constexpr uint32_t cols_own = get_compile_time_arg_val(8);
    constexpr uint32_t J = get_compile_time_arg_val(9);
    constexpr uint32_t RT = get_compile_time_arg_val(10);
    constexpr uint32_t cb_m = get_compile_time_arg_val(11);
    constexpr uint32_t dst_batch = get_compile_time_arg_val(12);  // fp32 DST tiles per acquire (half-sync: 4)

    constexpr uint32_t unit_tiles = unit_cols * seg_tiles;
    constexpr uint32_t slot_tiles = J * cols_own;
    static_assert(dst_batch >= 1 && dst_batch <= 4, "half of the 8-fp32-tile DEST per acquire (half-sync mode)");
    static_assert(unit_cols % dst_batch == 0, "the DST batches must tile the unit's columns");
    static_assert(cols_own <= 4, "phase 2 sums cols_own fp32 tiles per acquire: half of the DEST");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_hs);
    matmul_init(cb_a, cb_w);

    // ---- phase 1: per unit its W2 block stays resident over the expert's passes; the unit list and pass counts
    // come from the metadata page (read by the unpack thread and forwarded to the math / pack threads), already
    // clamped to the pass budget by the writer (dispatch_common.hpp clamp_passes)
    CircularBuffer meta(cb_m);
    meta.wait_front(1);
    const uint32_t n_units = meta.read_tile_value(0, 0);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t passes = meta.read_tile_value(0, 1 + u);
        cb_wait_front(cb_w, unit_tiles);
        for (uint32_t p = 0; p < passes; ++p) {
            cb_wait_front(cb_a, seg_tiles);
            for (uint32_t c0 = 0; c0 < unit_cols; c0 += dst_batch) {  // the unit's columns, dst_batch per acquire
                tile_regs_acquire();
                fill_tile_init();
                for (uint32_t d = 0; d < dst_batch; ++d) {
                    fill_tile(d, 0.0f);
                }
                reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_w);
                matmul_init(cb_a, cb_w);
                for (uint32_t d = 0; d < dst_batch; ++d) {
                    const uint32_t c = c0 + d;
                    for (uint32_t k = 0; k < seg_tiles; ++k) {
#ifdef REGIONS_SRC
                        matmul_tiles(cb_a, cb_w, k, k * unit_cols + c, d);
#else
                        matmul_tiles(cb_a, cb_w, k, c * seg_tiles + k, d);
#endif
                    }
                }
                tile_regs_commit();
                cb_reserve_back(cb_hs, dst_batch);
                tile_regs_wait();
                pack_reconfig_data_format(cb_hs);
                for (uint32_t d = 0; d < dst_batch; ++d) {
                    pack_tile(d, cb_hs);
                }
                tile_regs_release();
                cb_push_back(cb_hs, dst_batch);
            }
            cb_pop_front(cb_a, seg_tiles);
        }
        cb_pop_front(cb_w, unit_tiles);
    }

    // ---- phase 2 (compute_down_dispatch.cpp's, over this core's cols_own columns)
    cb_wait_front(cb_id, 1);
    for (uint32_t r = 0; r < RT; ++r) {
        uint32_t j_max = 0;
        UNPACK(j_max = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(j_max = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(j_max = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        cb_wait_front(cb_g, slot_tiles);
        tile_regs_acquire();
        fill_tile_init();
        for (uint32_t c = 0; c < cols_own; ++c) {
            fill_tile(c, 0.0f);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_id, cb_g);
        matmul_init(cb_id, cb_g);
        for (uint32_t j = 0; j < j_max; ++j) {
            for (uint32_t c = 0; c < cols_own; ++c) {
                matmul_tiles(cb_id, cb_g, 0, j * cols_own + c, c);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, cols_own);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        for (uint32_t c = 0; c < cols_own; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols_own);
        cb_pop_front(cb_g, slot_tiles);
    }
}
