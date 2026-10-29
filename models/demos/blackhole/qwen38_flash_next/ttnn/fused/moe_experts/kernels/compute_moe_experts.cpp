// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Routed-expert program, compute (one per core): BIT FOR BIT moe_compute's ring kernel (compute.cpp), on units instead
// of a ring.  A core may own (1) a gate/up unit G: a pair (columns c, c + 1: W0 c, W1 c, W0 c + 1, W1 c + 1, 4 tiles
// wide, K = 80 as 8 blocks of 10 rows) or a half (1 column, 2 wide, K = 80 as 4 blocks of 20 stored rows), accumulated
// in ONE DEST in K order 0..79, then SiLU(W0 x) * (W1 x) on the pack thread (moe_activation.h) packed to Float16_b: the
// in2 tiles, handed to the writer, which sends them to every W core; and (2) one down unit W: K = 20 rows in the ring
// order of the cached W2 (the runtime LUT gives the natural in2 column of each K row), 4 output tiles wide (2 blocks)
// or 2 wide (1 block), one DEST, pack_untilize_dest to rows.
// Per active expert i: G(0) first, then G(i + 1) before W(i) (the in2 buffer is double buffered by the expert's
// parity), the reader streams the weight blocks in that order.  Every init call sequence is compute.cpp's.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_activation.h"
#include "moe_experts_common.h"
#include "../../kernels/zones.h"

using namespace moe_experts;

namespace detail {

// compute.cpp's compute_w0_w1_block_column (no bias): in @ {W0, W1} for one gate/up unit, the activation, packed to
// `gout` tiles [out_tile, out_tile + kPairs).  kPairs = 2: ct 4 (a pair), 1: ct 2 (a half column).
template <ttnn::experimental::prim::detail::MoEActivationFunction activation, uint32_t kPairs, uint32_t num_blocks>
FORCE_INLINE void gate_up_unit(uint32_t in_cb_id, uint32_t w_cb_id, uint32_t out_cb_id, uint32_t out_tile) {
    constexpr uint32_t ct_dim = 2 * kPairs;
    constexpr uint32_t tiles_per_block = BLOCK_TILES;
    CircularBuffer cb_w(w_cb_id);

    tile_regs_acquire();
    uint32_t in0_index = 0;
    for (uint32_t block_id = 0; block_id < num_blocks; ++block_id) {
        cb_w.wait_front(tiles_per_block);
        for (uint32_t k = 0; k < tiles_per_block; k += ct_dim) {
            matmul_block(
                in_cb_id,
                w_cb_id,
                in0_index++,
                /*in1_index=*/k,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/ct_dim,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }
        cb_w.pop_front(tiles_per_block);
    }

    tile_regs_commit();

    // The below is equivalent to tile_regs_wait(), but we stall CFG as well, so that the succeeding
    // TT_SETC16 instruction is also stalled until math thread is done with these dest registers.
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));

    // Make SFPU access the appropriate half of the destination registers
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

    ::moe_activation::PackActivation<activation, kPairs>::compute();

    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

    pack_tile</*out_of_order_output=*/true>(0, out_cb_id, /*output_tile_index=*/out_tile);
    if constexpr (kPairs == 2) {
        pack_tile</*out_of_order_output=*/true>(2, out_cb_id, /*output_tile_index=*/out_tile + 1);
    }
    tile_regs_release();
}

}  // namespace detail

void kernel_main() {
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t cb_gout = get_named_compile_time_arg_val("cb_gout");
    constexpr uint32_t cb_tok = get_named_compile_time_arg_val("cb_tok");
    constexpr uint32_t cb_in2 = get_named_compile_time_arg_val("cb_in2");
    constexpr uint32_t cb_md = get_named_compile_time_arg_val("cb_md");
    constexpr uint32_t stage = get_named_compile_time_arg_val("stage");  // 2: gate/up only (the in2 dump), 3: all
    constexpr auto activation = ttnn::experimental::prim::detail::MoEActivationFunction::SILU;

    uint32_t arg = 0;
    const uint32_t g_kind = get_arg_val<uint32_t>(arg++);   // 0 none, 1 pair, 2 half
    const uint32_t w_units = get_arg_val<uint32_t>(arg++);  // down units of this core: 0 (a pair core), 1 or 2
    uint32_t w_kind[MAX_W_UNITS];                           // 1: 4 wide (2 blocks), 2: 2 wide (1 block)
    for (uint32_t u = 0; u < MAX_W_UNITS; ++u) {
        w_kind[u] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t order[NT];  // natural in2 column of the k-th K row of the down matmul
    for (uint32_t k = 0; k < NT; ++k) {
        order[k] = get_arg_val<uint32_t>(arg++);
    }

    CircularBuffer x(cb_x);
    CircularBuffer out(cb_out);
    CircularBuffer wcb(cb_w);
    CircularBuffer gout(cb_gout);
    CircularBuffer tok(cb_tok);
    CircularBuffer md(cb_md);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_in2);
    pack_reconfig_data_format(cb_in2);
    reconfig_data_format_srcb(cb_x);
    reconfig_data_format_srca(cb_w);

    md.wait_front(1);
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_md, 0));
    const uint32_t n = page[0];

    MATH((ckernel::zeroacc()));

    // in2: the double buffer, raw L1 (the writer fills it; the matmul reads its tiles by index)
    auto gate_up = [&]() {
        FUSED_ZONE("fz_me_c_g");
        ::moe_activation::pack_init_activation<activation>();
        gout.reserve_back(2);
        if (g_kind == 1) {
            matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
            ::detail::gate_up_unit<activation, 2, 8>(cb_x, cb_w, cb_gout, 0);
        } else {
            matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/2, /*rt_dim=*/1, /*kt_dim=*/1);
            ::detail::gate_up_unit<activation, 1, 4>(cb_x, cb_w, cb_gout, 0);
        }
        gout.push_back(2);
    };

    auto down = [&](uint32_t i) {
        FUSED_ZONE("fz_me_c_w");
        tok.wait_front(1);  // expert i's in2 tiles have all arrived (the writer)
        const uint32_t in2_base = (i & 1) * NT;
        // pack_untilize_dest init / uninit ONCE around the units (compute.cpp: cycling them per iteration races with
        // the packer on Blackhole); a 2-wide unit packs 4 tiles as the ring kernel's half iteration does (tiles 2 and 3
        // of its DEST half are zero), so every unit's rows are 4 tiles = 256 B apart
        pack_untilize_dest_init</*block_ct_dim=*/4, /*full_ct_dim=*/4>(cb_out);
        for (uint32_t u = 0; u < w_units; ++u) {
            out.reserve_back(4);
            uint32_t kidx = 0;
            if (w_kind[u] == 1) {
                matmul_block_init(cb_in2, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
                tile_regs_acquire();
                for (uint32_t block_id = 0; block_id < 2; ++block_id) {
                    wcb.wait_front(BLOCK_TILES);
                    for (uint32_t k = 0; k < BLOCK_TILES; k += 4) {
                        matmul_block(
                            cb_in2,
                            cb_w,
                            in2_base + order[kidx++],
                            /*in1_index=*/k,
                            /*idst=*/0,
                            /*transpose=*/false,
                            /*ct_dim=*/4,
                            /*rt_dim=*/1,
                            /*kt_dim=*/1);
                    }
                    wcb.pop_front(BLOCK_TILES);
                }
            } else {
                matmul_block_init(cb_in2, cb_w, /*transpose=*/false, /*ct_dim=*/2, /*rt_dim=*/1, /*kt_dim=*/1);
                tile_regs_acquire();
                wcb.wait_front(BLOCK_TILES);
                for (uint32_t k = 0; k < BLOCK_TILES; k += 2) {
                    matmul_block(
                        cb_in2,
                        cb_w,
                        in2_base + order[kidx++],
                        /*in1_index=*/k,
                        /*idst=*/0,
                        /*transpose=*/false,
                        /*ct_dim=*/2,
                        /*rt_dim=*/1,
                        /*kt_dim=*/1);
                }
                wcb.pop_front(BLOCK_TILES);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest</*block_ct_dim=*/4, /*full_ct_dim=*/4>(cb_out, /*block_rt_dim=*/1, /*block_c_index=*/0);
            tile_regs_release();
            out.push_back(4);
        }
        pack_untilize_uninit(cb_out);
        pack_reconfig_data_format(cb_in2);
        tok.pop_front(1);
    };

    if (g_kind != 0 && n > 0) {
        x.wait_front(HT);
    }
    if (g_kind != 0 && n > 0) {
        gate_up();
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (g_kind != 0 && i + 1 < n) {
            gate_up();
        }
        if constexpr (stage >= 3) {
            if (w_units != 0) {
                down(i);
            }
        }
    }

    // drain: the reader's last (unused) slot
    wcb.wait_front(BLOCK_TILES);
    wcb.pop_front(BLOCK_TILES);
}
