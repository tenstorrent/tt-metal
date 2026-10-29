// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// row_reduce + gr_write compute (QWEN38_RR_WRITE=1): the summation tree of compute.cpp (identical add order and pack
// per add), then gr_write's math on the DENSE tiles of the slice with gr_write/kernels/compute.cpp's exact op sequence:
//   per branch b:  upd = bf16(block * coef_b)  (copy_tile, COL-broadcast of the coefficient tile's column 0, SFPU
//                  mul_binary_tile, pack bf16),  new_b = residual_b + upd  (FPU add_tiles, bf16 pack).
// Every op is elementwise, so the dense layout (the slice's valid-row chunks of 10 tiles in one tile) gives the same
// bits per element as the 40 separate tiles of gr_write; the coefficient tile's rows follow the dense layout's rows
// (transport3.cpp builds them).  With an fp32 partial the sum is first narrowed to bf16 (pack of the fp32 dest).
// CBs: transport3's CB_IN/FS/BS/F/B/G/OUT as compute.cpp, 7 CB_SBF (bf16 narrowed sum, fp32 only), CB_A the block tile
// (BRISC re-laid), CB_COEF 4 coefficient tiles, CB_RES 4 dense residual tiles, CB_UPD scratch, CB_NEW 4 new residual
// tiles. Compile-time args: 0 CB_IN, 1 CB_FS, 2 CB_BS, 3 CB_F, 4 CB_B, 5 CB_G, 6 CB_OUT, 7 unused, 8 RING, 9 CB_SBF, 10
// CB_A,
//   11 CB_COEF, 12 CB_RES, 13 CB_UPD, 14 CB_NEW, 15 IS_FP32, 16 HELPER.  Runtime arg 0: L.
// The four branches run on four helper cores (helper.cpp feeds them), one tile each, in the 16-bit dest mode of
// gr_write whatever the partial's dtype; the reduce core only sums (and narrows an fp32 sum).

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "../../kernels/zones.h"

using namespace ckernel;

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t CB_FS = get_compile_time_arg_val(1);
constexpr uint32_t CB_BS = get_compile_time_arg_val(2);
constexpr uint32_t CB_F = get_compile_time_arg_val(3);
constexpr uint32_t CB_B = get_compile_time_arg_val(4);
constexpr uint32_t CB_G = get_compile_time_arg_val(5);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(6);
constexpr uint32_t RING = get_compile_time_arg_val(8);
constexpr uint32_t CB_SBF = get_compile_time_arg_val(9);
constexpr uint32_t CB_A = get_compile_time_arg_val(10);
constexpr uint32_t CB_COEF = get_compile_time_arg_val(11);
constexpr uint32_t CB_RES = get_compile_time_arg_val(12);
constexpr uint32_t CB_UPD = get_compile_time_arg_val(13);
constexpr uint32_t CB_NEW = get_compile_time_arg_val(14);
constexpr uint32_t IS_FP32 = get_compile_time_arg_val(15);
constexpr uint32_t HELPER = get_compile_time_arg_val(16);  // a helper core: only the write math of its one branch
constexpr uint32_t BRANCHES = 1;  // branches per core (branch 0 on the reduce core, 1..3 on helper cores)

// cb_out <- front(a) + front(b); pops both fronts after the pack
FORCE_INLINE void add_pop(uint32_t a, uint32_t b, uint32_t out) {
    cb_wait_front(a, 1);
    cb_wait_front(b, 1);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    add_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    cb_pop_front(a, 1);
    cb_pop_front(b, 1);
}

void kernel_main() {
    const uint32_t L = get_arg_val<uint32_t>(0);
    if constexpr (HELPER) {
        compute_kernel_hw_startup(CB_A, CB_COEF, CB_NEW);  // bf16 only: the formats of the gr_write math
    } else {
        compute_kernel_hw_startup(CB_IN, CB_FS, CB_OUT);
        add_init(CB_IN, CB_FS, false);
    }
    if constexpr (!HELPER) {
        FUSED_ZONE("fz_rr_wc_sum");
        uint32_t own_sum = CB_FS;  // G = p_L + F (L >= 1), the own term itself for L == 0
        if (L >= 1) {
            uint32_t acc = CB_FS;  // F, seeded with p_0
            for (uint32_t k = 1; k < L; ++k) {
                add_pop(CB_IN, acc, CB_F);
                acc = CB_F;
            }
            add_pop(CB_IN, acc, L == RING - 1 ? CB_OUT : CB_G);  // p_L + F
            own_sum = CB_G;
        }
        if (L < RING - 1) {
            uint32_t acc = CB_BS;  // B, seeded with p_{R-1}
            for (uint32_t k = RING - 2; k > L; --k) {
                add_pop(CB_IN, acc, CB_B);
                acc = CB_B;
            }
            add_pop(own_sum, acc, CB_OUT);
        }
    }
    if constexpr (IS_FP32 && !HELPER) {  // the narrowing of the fp32 sum to the bf16 block (the QSA typecast), BRISC
                                         // reads CB_SBF
        FUSED_ZONE("fz_rr_wc_narrow");
        cb_wait_front(CB_OUT, 1);
        cb_reserve_back(CB_SBF, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_OUT);
        copy_init(CB_OUT);
        copy_tile(CB_OUT, 0, 0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_SBF);
        pack_tile(0, CB_SBF);
        tile_regs_release();
        cb_push_back(CB_SBF, 1);
    }
    if constexpr (HELPER) {
        FUSED_ZONE("fz_rr_wc_write");
        constexpr uint32_t dst_upd = 0, dst_coef = 1;
        mul_binary_tile_init();
        cb_wait_front(CB_A, 1);
        for (uint32_t b = 0; b < BRANCHES; ++b) {
            cb_wait_front(CB_COEF, 1);
            cb_reserve_back(CB_UPD, 1);
            tile_regs_acquire();
            reconfig_data_format_srca(CB_A);
            copy_init(CB_A);
            copy_tile(CB_A, 0, dst_upd);
            reconfig_data_format_srca(CB_COEF);
            unary_bcast_init<BroadcastType::COL>(CB_COEF);
            unary_bcast<BroadcastType::COL>(CB_COEF, 0, dst_coef);
            mul_binary_tile(dst_upd, dst_coef, dst_upd);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_UPD);
            pack_tile(dst_upd, CB_UPD);
            tile_regs_release();
            cb_push_back(CB_UPD, 1);
            cb_pop_front(CB_COEF, 1);

            cb_wait_front(CB_RES, 1);
            cb_wait_front(CB_UPD, 1);
            cb_reserve_back(CB_NEW, 1);
            reconfig_data_format(CB_RES, CB_UPD);
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(CB_RES, CB_UPD);
            tile_regs_acquire();
            add_tiles(CB_RES, CB_UPD, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_NEW);
            pack_tile(0, CB_NEW);
            tile_regs_release();
            cb_push_back(CB_NEW, 1);
            cb_pop_front(CB_RES, 1);
            cb_pop_front(CB_UPD, 1);
        }
        cb_pop_front(CB_A, 1);
    }
}
