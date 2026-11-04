// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR read stage 3b, one hidden column tile per core: the low-rank row (k_tiles tiles) times the branch weight columns
// (one per branch) in the 32-bit dest in K order, packed bf16; the bf16-dest sigmoid of each tile; the SFPU product
// with the branch's normalized tile; the branches' products summed with the zero tile in branch order as reduce_nc.cpp
// does; packed bf16.  CBs: 0 low rank (bf16, k_tiles), 1 weight (bf16, branches x k_tiles, K-minor), 2 normalized
// (bf16, branches), 3 zero (bf16, 1), 4 up (bf16, branches), 5 gate (bf16, branches), 6 gated (bf16, branches), 16
// block (bf16, 1).  Compile-time args: 0 k_tiles, 1 branches (even), 2-9 the cbs (low rank, weight, normalized, zero,
// up, gate, gated, block), 10 spill, 11 the intermediate cb.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t branches = get_compile_time_arg_val(1);
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
    static_assert(spill > 0 && k_tiles % spill == 0, "the partials spill every `spill` K tiles");

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_lr, c_w, c_interm);
    matmul_block_init(c_lr, c_w, 0, 1, 1, 1);
    DataflowBuffer lr(c_lr);
    DataflowBuffer weight(c_w);
    DataflowBuffer nws(c_nws);
    DataflowBuffer zero(c_zero);
    DataflowBuffer interm(c_interm);
    DataflowBuffer up(c_up);
    DataflowBuffer gate(c_gate);
    DataflowBuffer gated(c_gated);
    DataflowBuffer out(c_out);

    // K blocks of `spill` tiles over the `branches` output tiles; between blocks the partials spill to the fp32
    // intermediate CB and reload through SrcA (the DRAM-sharded up matmul's path).
    {
        lr.wait_front(k_tiles);
        weight.wait_front(branches * k_tiles);
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < k_tiles; kb += spill) {
            for (uint32_t b = 0; b < branches; ++b) {
                for (uint32_t kt = kb; kt < kb + spill; ++kt) {
                    matmul_block(c_lr, c_w, kt, b * k_tiles + kt, b, 0, 1, 1, 1);
                }
            }
            if (kb + spill < k_tiles) {
                tile_regs_commit();
                interm.reserve_back(branches);
                tile_regs_wait();
                for (uint32_t b = 0; b < branches; ++b) {
                    pack_tile(b, c_interm);
                }
                tile_regs_release();
                interm.push_back(branches);
                tile_regs_acquire();
                reconfig_data_format_srca(c_w, c_interm);
                copy_init(c_interm);
                interm.wait_front(branches);
                for (uint32_t b = 0; b < branches; ++b) {
                    copy_tile(c_interm, b, b);
                }
                interm.pop_front(branches);
                reconfig_data_format_srca(c_interm, c_w);
                matmul_block_init(c_lr, c_w, 0, 1, 1, 1);
            }
        }
        tile_regs_commit();
        up.reserve_back(branches);
        tile_regs_wait();
        pack_reconfig_data_format(c_up);
        for (uint32_t b = 0; b < branches; ++b) {
            pack_tile(b, c_up);
        }
        tile_regs_release();
        up.push_back(branches);
        weight.pop_front(branches * k_tiles);
        lr.pop_front(k_tiles);
    }

    {
        up.wait_front(branches);
        reconfig_data_format_srca(c_w, c_up);
        copy_init(c_up);
        pack_reconfig_data_format(c_gate);
        sigmoid_tile_init<false>();
        tile_regs_acquire();
        for (uint32_t b = 0; b < branches; ++b) {
            copy_tile(c_up, b, b);
            sigmoid_tile<VectorMode::RC, false, false>(b);
        }
        tile_regs_commit();
        gate.reserve_back(branches);
        tile_regs_wait();
        for (uint32_t b = 0; b < branches; ++b) {
            pack_tile(b, c_gate);
        }
        tile_regs_release();
        gate.push_back(branches);
        up.pop_front(branches);
    }

    {
        gate.wait_front(branches);
        nws.wait_front(branches);
        pack_reconfig_data_format(c_gated);
        mul_binary_tile_init();
        for (uint32_t b = 0; b < branches; b += 2) {
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
        gate.pop_front(branches);
        nws.pop_front(branches);
    }

    {
        gated.wait_front(branches);
        zero.wait_front(1);
        add_init(c_gated, c_zero, true);
        reconfig_data_format(c_gated, c_zero);
        pack_reconfig_data_format(c_out);
        tile_regs_acquire();
        for (uint32_t b = 0; b < branches; ++b) {
            add_tiles(c_gated, c_zero, b, 0, 0);
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_out);
        tile_regs_release();
        out.push_back(1);
        gated.pop_front(branches);
        zero.pop_front(1);
    }
}
