// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE combine program, compute.  Per work unit (one row tile by `cols` column tiles) the two replaced ops'
// instruction sequences, so the values are the composed chain's:
//   ttnn/cpp/ttnn/kernel/compute/tilize.cpp (the to_layout op on a bf16 block): compute_kernel_lib::tilize with
//     Fp32Mode::Fast -- on Blackhole the fast tilize path for Float16_b -- one 32-row block of `cols` tiles per slot,
//     slot order 0..top_k-1                                                                          -> cb_tiled
//   deepseek_moe_fast_reduce_nc_fused_compute.cpp: FPU ELWMUL COL-broadcast MAC into the 32-bit dest (acc_to_dest,
//     the zeroed DEST tile 0 seeds the sum), slot order 0..top_k-1 per output tile, one bf16 pack   -> cb_out
// The kernel runs with the 32-bit dest and HiFi4 the reduce's compute config asks for (fp32_dest_acc_en = True,
// MathFidelity.HiFi4: moe.py's compute_config).  The fast tilize path clears the dest's fp32 mode for its copy and
// restores it in its uninit (llk_math_fast_tilize.h), so the tilize is the chain's 16-bit copy here too; only the
// cols = 1 fallback (tilize_block, an A2D copy) goes through the 32-bit dest -- bf16 bits either way (the device
// test's special-value and column-group cases pin this).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);
    if (unit_count == 0) {
        return;
    }

    constexpr uint32_t cb_rm = get_named_compile_time_arg_val("cb_rm");
    constexpr uint32_t cb_tiled = get_named_compile_time_arg_val("cb_tiled");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cols = get_named_compile_time_arg_val("cols");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t groups = get_named_compile_time_arg_val("groups");

    DataflowBuffer tiled(cb_tiled);
    DataflowBuffer scores(cb_scores);
    DataflowBuffer out(cb_out);

    compute_kernel_hw_startup(cb_tiled, cb_scores, cb_out);

    uint32_t last_r = 0xffffffffu;
    bool have_scores = false;
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t r = u / groups;
        if (r != last_r) {
            // the row tile's score tiles, resident for every unit of the row tile (the reduce's prologue)
            if (have_scores) {
                scores.pop_front(top_k);
            }
            scores.wait_front(top_k);
            have_scores = true;
            last_r = r;
        }
        {
            FUSED_ZONE("fz_mc_c_tilize");
            // ---- ttnn/cpp/ttnn/kernel/compute/tilize.cpp: top_k blocks of `cols` tiles (one per slot) ----
            compute_kernel_lib::tilize<
                cols,
                cb_rm,
                cb_tiled,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
                compute_kernel_lib::tilize_config::Fp32Mode::Fast>(top_k);
        }
        {
            FUSED_ZONE("fz_mc_c_mac");
            // ---- deepseek_moe_fast_reduce_nc_fused_compute.cpp ----
            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_tiled, cb_scores);
            // Override MATH init to enable acc_to_dest=1 (hardware accumulate mode): dst0 += act * score
            MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
                cb_tiled, cb_scores, 1 /*acc_to_dest*/)));
            reconfig_data_format(cb_tiled, cb_scores);
            pack_reconfig_data_format(cb_out);
            tiled.wait_front(top_k * cols);
            for (uint32_t c = 0; c < cols; ++c) {
                out.reserve_back(1);
                tile_regs_acquire();
                for (uint32_t e = 0; e < top_k; ++e) {
                    // dst0 += tile(slot e, column c) * score_col[e]  (single MAC, the chain's slot order)
                    mul_tiles_bcast_cols(cb_tiled, cb_scores, e * cols + c, e, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                out.push_back(1);
            }
            tiled.pop_front(top_k * cols);
        }
    }
    if (have_scores) {
        scores.pop_front(top_k);
    }
}
