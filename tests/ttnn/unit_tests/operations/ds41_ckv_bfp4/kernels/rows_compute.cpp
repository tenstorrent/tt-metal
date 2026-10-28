// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Probe: K tiles [Skt x DHt] from a gathered row block, every pass.
// BFP4 mode: the block is a row-major bfloat4_b tensor [n_rows x 512] (exponent section, then the nibbles). The
// unpacker reads it with the tile descriptor x = 16, y = 32 column groups, z = n_rows rows, so datum
// ((z * 32) + y) * 16 is group y of row z and its exponent is byte (z * 32 + y) of the section: one UNPACR per face
// row (16 values, 1 exponent), output row +1 / input z +1 per UNPACR, 16 per face, 64 per tile into SrcA; then the
// standard datacopy to DEST and the pack (the values are on the bfp4 grid: the pack copies them).
// BF16 mode (baseline): the standard tilize of 1024 B rows.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#endif

#ifdef TRISC_UNPACK
using namespace ckernel;
using namespace ckernel::unpacker;

template <uint32_t n_rows>
inline void rows_unpack_init() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xFFFFFFFF>(16 | (16 << 16));
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xFFFFFFFF>(32 | (n_rows << 16));
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(16);  // one SrcA row (16 datums)
    TTI_SETADCXX(p_setadc::UNP_A, 16 - 1, 0x0);
}

// tile (tr, c): faces (rows 32 tr + 16 (f / 2) .. + 15, group 2 c + f % 2)
inline void rows_unpack_tile(uint32_t base16, uint32_t tr, uint32_t c) {
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_single_address_(base16, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    // per face (as the plain copy's MOP: one SrcA face + valid, a SrcB valid): 16 one-row unpacks into SrcA rows
    // 0..15, the last with the valid flag
    for (uint32_t f = 0; f < 4; ++f) {
        TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000);  // SrcA row (ch1 Y) = 0
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, tr * 32 + (f >> 1) * 16);
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, 2 * c + (f & 1));
        for (uint32_t r = 0; r < 15; ++r) {
            TTI_UNPACR(
                SrcA, 0b01000001 /*CH1_Y+=1, CH0_Z+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        }
        TTI_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR_NOP(
            SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

void kernel_main() {
    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t n_rows = get_compile_time_arg_val(2);
    constexpr uint32_t passes = get_compile_time_arg_val(3);
    constexpr uint32_t ppp = get_compile_time_arg_val(4);  // CB pages a pass
    constexpr uint32_t DHt = 16;
    constexpr uint32_t Skt = n_rows / 32;

    compute_kernel_hw_startup(cb_rows, cb_out);
#ifdef PROBE_BFP4
    copy_tile_init(cb_rows);
    UNPACK((rows_unpack_init<n_rows>()));
    for (uint32_t pass = 0; pass < passes; ++pass) {
        cb_wait_front(cb_rows, ppp);
#ifdef PROBE_NO_CONVERT
        for (uint32_t t = 0; t < Skt * DHt; ++t) {  // timing only: no conversion
            cb_reserve_back(cb_out, 1);
            cb_push_back(cb_out, 1);
        }
#else
        for (uint32_t tr = 0; tr < Skt; ++tr) {
            for (uint32_t c = 0; c < DHt; ++c) {
                cb_reserve_back(cb_out, 1);
                tile_regs_acquire();
                UNPACK((rows_unpack_tile(get_local_cb_interface(cb_rows).fifo_rd_ptr - 1, tr, c)));
                MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, false>(
                    0, cb_rows)));
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }
        }
#endif
        cb_pop_front(cb_rows, ppp);
    }
#else
    // baseline: the gathered bf16 rows (one CB page = the whole block) tilized in DHt-wide row bands
    tilize_init(cb_rows, DHt, cb_out);
    for (uint32_t pass = 0; pass < passes; ++pass) {
        cb_wait_front(cb_rows, ppp);
        for (uint32_t tr = 0; tr < Skt; ++tr) {
            cb_reserve_back(cb_out, DHt);
#ifndef PROBE_NO_CONVERT
            tilize_block(cb_rows, DHt, cb_out, tr * DHt, 0);
#endif
            cb_push_back(cb_out, DHt);
        }
        cb_pop_front(cb_rows, ppp);
    }
#endif
}
