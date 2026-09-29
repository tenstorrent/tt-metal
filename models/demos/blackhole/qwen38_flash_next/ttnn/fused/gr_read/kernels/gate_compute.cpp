// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR read stage 3b, one hidden column tile per core: the low-rank row (Kt tiles) times the B branch weight columns
// accumulated in the 32-bit dest in K order (packed bf16 by the packer as the up matmul does), the bf16-dest
// sigmoid of each packed tile (unary sigmoid, vector mode RC), the SFPU product with the normalized tile of the same
// branch (bf16-dest rounding and zero rule), and the B branch products summed with the zero tile in branch order
// (reduce_nc.cpp), packed bf16.
// CBs: 0 low rank (bf16, Kt), 1 weight (bf16, B x Kt, K-minor), 2 normalized (bf16, B), 3 zero (bf16, 1),
// 4 up (bf16, B), 5 gate (bf16, B), 6 gated (bf16, B), 16 block (bf16, 1).  DEBUG_KEEP leaves up/gate/gated for the
// writer.  Compile-time args: 0 Kt, 1 B (even), 2-9 the cbs (low rank, weight, normalized, zero, up, gate, gated,
// block).

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t c_lr = get_compile_time_arg_val(2);
    constexpr uint32_t c_w = get_compile_time_arg_val(3);
    constexpr uint32_t c_nws = get_compile_time_arg_val(4);
    constexpr uint32_t c_zero = get_compile_time_arg_val(5);
    constexpr uint32_t c_up = get_compile_time_arg_val(6);
    constexpr uint32_t c_gate = get_compile_time_arg_val(7);
    constexpr uint32_t c_gated = get_compile_time_arg_val(8);
    constexpr uint32_t c_out = get_compile_time_arg_val(9);
    constexpr uint32_t spill = get_compile_time_arg_val(10);
    constexpr uint32_t c_interm = get_compile_time_arg_val(11);
    constexpr uint32_t block = spill > 0 ? spill : Kt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_lr, c_w, c_interm);
    matmul_block_init(c_lr, c_w, 0, 1, 1, 1);
    DataflowBuffer lr(c_lr);
    DataflowBuffer w(c_w);
    DataflowBuffer nws(c_nws);
    DataflowBuffer zero(c_zero);
    DataflowBuffer interm(c_interm);
    DataflowBuffer up(c_up);
    DataflowBuffer gate(c_gate);
    DataflowBuffer gated(c_gated);
    DataflowBuffer out(c_out);

    // K blocks of `block` tiles over the B output tiles; between blocks the partials spill to the fp32 intermediate
    // CB and reload through SrcA (the DRAM-sharded up matmul's path; spill = 0 accumulates in the dest throughout).
    {
        FUSED_ZONE("fz_gr_gate_c_matmul");
        lr.wait_front(Kt);
        w.wait_front(B * Kt);
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < Kt; kb += block) {
            for (uint32_t b = 0; b < B; ++b) {
                for (uint32_t kt = kb; kt < kb + block; ++kt) {
                    matmul_block(c_lr, c_w, kt, b * Kt + kt, b, 0, 1, 1, 1);
                }
            }
            if (kb + block < Kt) {
                tile_regs_commit();
                interm.reserve_back(B);
                tile_regs_wait();
                for (uint32_t b = 0; b < B; ++b) {
                    pack_tile(b, c_interm);
                }
                tile_regs_release();
                interm.push_back(B);
                tile_regs_acquire();
                reconfig_data_format_srca(c_w, c_interm);
                copy_init(c_interm);
                interm.wait_front(B);
                for (uint32_t b = 0; b < B; ++b) {
                    copy_tile(c_interm, b, b);
                }
                interm.pop_front(B);
                reconfig_data_format_srca(c_interm, c_w);
                matmul_block_init(c_lr, c_w, 0, 1, 1, 1);
            }
        }
        tile_regs_commit();
        up.reserve_back(B);
        tile_regs_wait();
        pack_reconfig_data_format(c_up);
        for (uint32_t b = 0; b < B; ++b) {
            pack_tile(b, c_up);
        }
        tile_regs_release();
        up.push_back(B);
        w.pop_front(B * Kt);
        lr.pop_front(Kt);
    }

    {
        FUSED_ZONE("fz_gr_gate_c_sigmoid");
        up.wait_front(B);
        reconfig_data_format_srca(c_w, c_up);
        copy_init(c_up);
        pack_reconfig_data_format(c_gate);
        sigmoid_tile_init<false>();
        tile_regs_acquire();
        for (uint32_t b = 0; b < B; ++b) {
            copy_tile(c_up, b, b);
            sigmoid_tile<VectorMode::RC, false, false>(b);
        }
        tile_regs_commit();
        gate.reserve_back(B);
        tile_regs_wait();
        for (uint32_t b = 0; b < B; ++b) {
            pack_tile(b, c_gate);
        }
        tile_regs_release();
        gate.push_back(B);
#ifndef DEBUG_KEEP
        up.pop_front(B);
#endif
    }

    {
        FUSED_ZONE("fz_gr_gate_c_gated");
        gate.wait_front(B);
        nws.wait_front(B);
        pack_reconfig_data_format(c_gated);
        mul_binary_tile_init();
        for (uint32_t b = 0; b < B; b += 2) {
            gated.reserve_back(2);
            tile_regs_acquire();
            reconfig_data_format_srca(c_gate, c_nws);
            copy_init(c_nws);
            copy_tile(c_nws, b, 0);
            copy_tile(c_nws, b + 1, 2);
            reconfig_data_format_srca(c_nws, c_gate);
            copy_init(c_gate);
            copy_tile(c_gate, b, 1);
            mul_binary_tile<false>(0, 1, 0);
            copy_tile(c_gate, b + 1, 3);
            mul_binary_tile<false>(2, 3, 2);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, c_gated);
            pack_tile(2, c_gated);
            tile_regs_release();
            gated.push_back(2);
        }
#ifndef DEBUG_KEEP
        gate.pop_front(B);
#endif
        nws.pop_front(B);
    }

    {
        FUSED_ZONE("fz_gr_gate_c_sum");
        gated.wait_front(B);
        zero.wait_front(1);
        add_init(c_gated, c_zero, true);
        reconfig_data_format(c_gated, c_zero);
        pack_reconfig_data_format(c_out);
        tile_regs_acquire();
        for (uint32_t b = 0; b < B; ++b) {
            add_tiles(c_gated, c_zero, b, 0, 0);
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_out);
        tile_regs_release();
        out.push_back(1);
#ifndef DEBUG_KEEP
        gated.pop_front(B);
#endif
        zero.pop_front(1);
    }
}
