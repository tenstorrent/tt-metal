// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4 iHC fused PRE kernel, compute.  Per core (column tiles [k*GPC, (k+1)*GPC) of all four streams, G token groups of
// <= 8):
//   SQ_{g,j} = X_{g,j}^2;  Z_g = sum_j X_{g,j} @ Wp_j   (Wp_j[c, 8i+m] = fn[m, i*D + col(c)], bf16)  ZM_g = Z_g * MASK
//   (stream rows x own fn block) U = sum_g SELALL_g @ ZM_g;   Q_g = sum_j SQ_{g,j} @ ONES8;   PART = U @ FOLD + sum_g
//   SELALL_g @ Q_g     row = token, columns 0..7 mixes, 8 = mean x^2 (exchange 1 in the reader: RECV = 32 faces of
//   PARTs)  RAW = (sum_u FL @ RECV_u) @ FR  = sum of the 32 PARTs MIX = RAW * rsqrt(RAW[:,8] + eps_n);  pp =
//   (sigmoid(MIX @ SD + BASE) + OFF) * SCALE      (SD = diag(hc_scale); columns 0..3 pre, 4..7 post) (reader builds A_g
//   from pp)  HS_j = bf16(sum_g A_g @ X_{g,j});  P2 = sum_j HS_j^2 @ ONES  (exchange 2 in the reader) rs = rsqrt(R @
//   ONES + D*eps);  out_j = bf16(HS_j * rs * w'_j)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_a = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p2 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(8);
    constexpr uint32_t cb_w4 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_cst = get_compile_time_arg_val(10);
    [[maybe_unused]] constexpr uint32_t cb_ones = get_compile_time_arg_val(11);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(12);
    constexpr uint32_t cb_z = get_compile_time_arg_val(13);
    constexpr uint32_t cb_q = get_compile_time_arg_val(14);
    constexpr uint32_t cb_e = get_compile_time_arg_val(15);
    constexpr uint32_t cb_raw = get_compile_time_arg_val(16);
    constexpr uint32_t cb_mix = get_compile_time_arg_val(17);
    constexpr uint32_t cb_hs = get_compile_time_arg_val(18);
    constexpr uint32_t cb_sqh = get_compile_time_arg_val(19);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(20);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(21);
    constexpr uint32_t cb_out = get_compile_time_arg_val(22);
    constexpr uint32_t G = get_compile_time_arg_val(23);
    constexpr uint32_t GPC = get_compile_time_arg_val(24);
    constexpr uint32_t EPS_N_BITS = get_compile_time_arg_val(25);  // eps of the iHC mixes rsqrt
    constexpr uint32_t EPS_2_BITS = get_compile_time_arg_val(26);  // D * eps of the final rms norm
    constexpr uint32_t cb_sd = get_compile_time_arg_val(27);
    constexpr uint32_t cb_zm = get_compile_time_arg_val(28);
    constexpr uint32_t cb_u = get_compile_time_arg_val(29);
    constexpr uint32_t cb_dbg = get_compile_time_arg_val(30);
    constexpr uint32_t DBG = get_compile_time_arg_val(31);
    constexpr uint32_t CABL = get_compile_time_arg_val(
        32);  // timing ablation bits (results wrong when set): 1 skip rsqrt (mix), 2 skip sigmoid, 4 skip rsqrt (final)
    constexpr uint32_t EXPAND = get_compile_time_arg_val(33);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(34);
    constexpr uint32_t cb_y = get_compile_time_arg_val(35);
    constexpr uint32_t cb_m = get_compile_time_arg_val(36);
    constexpr uint32_t cb_inc = get_compile_time_arg_val(37);
    constexpr uint32_t NCST = 10 + G;
    constexpr uint32_t C_SELSS = 1, C_FL = 2, C_FR = 3, C_SCALE = 4, C_OFF = 5, C_MASK = 6, C_FOLD = 7, C_SEL = 8,
                       C_ONES = 8 + G, C_IDENT = 9 + G;

    compute_kernel_hw_startup(cb_x, cb_w, cb_part);
    CircularBuffer x(cb_x), w(cb_w), base(cb_base), recv(cb_recv), part(cb_part), pp(cb_pp), a(cb_a), p2(cb_p2),
        tot(cb_tot), w4(cb_w4), cst(cb_cst), sq(cb_sq), z(cb_z), q(cb_q), e(cb_e), raw(cb_raw), mix(cb_mix), hs(cb_hs),
        sqh(cb_sqh), rs(cb_rs), tmp(cb_tmp), out(cb_out), sd(cb_sd), zm(cb_zm), u(cb_u), xr(cb_xr), yy(cb_y), mm(cb_m),
        inc(cb_inc);

#define STAMP(i)                                                                                             \
    if constexpr (DBG) {                                                                                     \
        PACK(                                                                                                \
            (reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(cb_dbg).fifo_wr_ptr << 4)[16 + i] = \
                 *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)));                      \
    }
    STAMP(0)
    cst.wait_front(NCST);
    if constexpr (EXPAND) {
        // x_new_{g,j} = I @ XR_{g,j} + M_g @ Y_j   (fp32 HiFi4: the stream keeps its precision)
        xr.wait_front(G * GPC);
        yy.wait_front(GPC);
        mm.wait_front(G);
        // pass 1: INC_{g,j} = M_g @ Y_j  (tf32-class products of the exact bf16 y and the post weights)
        for (uint32_t g = 0; g < G; ++g) {
            for (uint32_t j = 0; j < GPC; ++j) {
                reconfig_data_format(cb_y, cb_m);  // matmul: in1 -> srcA, in0 -> srcB (M fp32, Y bf16)
                matmul_init(cb_m, cb_y);
                tile_regs_acquire();
                matmul_tiles(cb_m, cb_y, g, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                inc.reserve_back(1);
                pack_tile(0, cb_inc);
                inc.push_back(1);
                tile_regs_release();
            }
        }
        // pass 2: x_new = XR + INC, EXACT fp32 (both copied unpack-to-dest, added on the SFPU)
        inc.wait_front(G * GPC);
        reconfig_data_format_srca(cb_xr);
        for (uint32_t n = 0; n < G * GPC; ++n) {
            copy_tile_to_dst_init_short(cb_xr);
            tile_regs_acquire();
            copy_tile(cb_xr, n, 0);
            copy_tile_to_dst_init_short(cb_inc);
            copy_tile(cb_inc, n, 1);
            if constexpr (!(CABL & 8)) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            x.reserve_back(1);
            pack_tile(0, cb_x);
            x.push_back(1);
            tile_regs_release();
        }
    }
    x.wait_front(G * GPC);
    w.wait_front(GPC);

    STAMP(1)
    // ---- squares ----
    for (uint32_t n = 0; n < G * GPC; ++n) {
        reconfig_data_format(cb_x, cb_x);
        mul_init(cb_x, cb_x);
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, n, n, 0);
        tile_regs_commit();
        tile_regs_wait();
        sq.reserve_back(1);
        pack_tile(0, cb_sq);
        sq.push_back(1);
        tile_regs_release();
    }
    STAMP(2)
    // ---- Z_g = sum_j X_{g,j} @ Wp_j ----
    for (uint32_t g = 0; g < G; ++g) {
        reconfig_data_format(cb_w, cb_x);  // matmul: in1 -> srcA, in0 -> srcB (formats differ here: x fp32, W bf16)
        matmul_init(cb_x, cb_w);
        tile_regs_acquire();
        for (uint32_t j = 0; j < GPC; ++j) {
            matmul_tiles(cb_x, cb_w, g * GPC + j, j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        z.reserve_back(1);
        pack_tile(0, cb_z);
        z.push_back(1);
        tile_regs_release();
    }
    STAMP(3)
    // ---- ZM_g = Z_g * MASK ----
    z.wait_front(G);
    for (uint32_t g = 0; g < G; ++g) {
        reconfig_data_format(cb_z, cb_cst);
        mul_init(cb_z, cb_cst);
        tile_regs_acquire();
        mul_tiles(cb_z, cb_cst, g, C_MASK, 0);
        tile_regs_commit();
        tile_regs_wait();
        zm.reserve_back(1);
        pack_tile(0, cb_zm);
        zm.push_back(1);
        tile_regs_release();
    }
    STAMP(4)
    // ---- U = sum_g SELALL_g @ ZM_g ----
    zm.wait_front(G);
    reconfig_data_format(cb_cst, cb_zm);
    matmul_init(cb_cst, cb_zm);
    tile_regs_acquire();
    for (uint32_t g = 0; g < G; ++g) {
        matmul_tiles(cb_cst, cb_zm, C_SEL + g, g, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    u.reserve_back(1);
    pack_tile(0, cb_u);
    u.push_back(1);
    tile_regs_release();
    STAMP(5)
    // ---- Q_g = sum_j SQ_{g,j} @ ONES8 ----
    sq.wait_front(G * GPC);
    for (uint32_t g = 0; g < G; ++g) {
        reconfig_data_format(cb_sq, cb_cst);
        matmul_init(cb_sq, cb_cst);
        tile_regs_acquire();
        for (uint32_t j = 0; j < GPC; ++j) {
            matmul_tiles(cb_sq, cb_cst, g * GPC + j, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        q.reserve_back(1);
        pack_tile(0, cb_q);
        q.push_back(1);
        tile_regs_release();
    }
    STAMP(6)
    // ---- PART = U @ FOLD + sum_g SELALL_g @ Q_g ----
    u.wait_front(1);
    q.wait_front(G);
    reconfig_data_format(cb_u, cb_cst);
    matmul_init(cb_u, cb_cst);
    tile_regs_acquire();
    matmul_tiles(cb_u, cb_cst, 0, C_FOLD, 0);
    reconfig_data_format(cb_cst, cb_q);
    matmul_init(cb_cst, cb_q);
    for (uint32_t g = 0; g < G; ++g) {
        matmul_tiles(cb_cst, cb_q, C_SEL + g, g, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    part.reserve_back(1);
    pack_tile(0, cb_part);
    part.push_back(1);
    tile_regs_release();

    STAMP(7)
    // ---- RAW = (sum_u FL @ RECV_u) @ FR ----
    recv.wait_front(8);
    reconfig_data_format(cb_cst, cb_recv);
    matmul_init(cb_cst, cb_recv);
    tile_regs_acquire();
    for (uint32_t u = 0; u < 8; ++u) {
        matmul_tiles(cb_cst, cb_recv, C_FL, u, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    e.reserve_back(1);
    pack_tile(0, cb_e);
    e.push_back(1);
    tile_regs_release();
    STAMP(8)
    e.wait_front(1);
    reconfig_data_format(cb_e, cb_cst);
    matmul_init(cb_e, cb_cst);
    tile_regs_acquire();
    matmul_tiles(cb_e, cb_cst, 0, C_FR, 0);
    tile_regs_commit();
    tile_regs_wait();
    raw.reserve_back(1);
    pack_tile(0, cb_raw);
    raw.push_back(1);
    tile_regs_release();

    STAMP(9)
    // ---- MIX = RAW * rsqrt(mean x^2 + eps) ----
    raw.wait_front(1);
    reconfig_data_format(cb_raw, cb_cst);
    matmul_init(cb_raw, cb_cst);
    tile_regs_acquire();
    matmul_tiles(cb_raw, cb_cst, 0, C_SELSS, 0);  // every column = mean of squares
    binop_with_scalar_tile_init();
    add_unary_tile(0, EPS_N_BITS);
    if constexpr (!(CABL & 1)) {
        rsqrt_tile_init();
        rsqrt_tile(0);
    }
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_raw);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_raw, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    mix.reserve_back(1);
    pack_tile(0, cb_mix);
    mix.push_back(1);
    tile_regs_release();
    mix.wait_front(1);

    STAMP(10)
    // ---- pp = (sigmoid(MIX @ SD + BASE) + OFF) * SCALE ----
    base.wait_front(1);
    sd.wait_front(1);
    reconfig_data_format(cb_mix, cb_sd);
    matmul_init(cb_mix, cb_sd);
    tile_regs_acquire();
    matmul_tiles(cb_mix, cb_sd, 0, 0, 0);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_base);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_base, 0, 0);
    if constexpr (!(CABL & 2)) {
        sigmoid_tile_init();
        sigmoid_tile<VectorMode::R>(0);  // rows 0..15 only (T <= 16)
    }
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cst);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cst, C_OFF, 0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cst);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cst, C_SCALE, 0);
    tile_regs_commit();
    tile_regs_wait();
    pp.reserve_back(1);
    pack_tile(0, cb_pp);
    pp.push_back(1);
    tile_regs_release();

    STAMP(11)
    // ---- collapse: HS_j = bf16(sum_g A_g @ X_{g,j}) ----
    a.wait_front(G);
    pack_reconfig_data_format(cb_hs);
    for (uint32_t j = 0; j < GPC; ++j) {
        reconfig_data_format(cb_a, cb_x);
        matmul_init(cb_a, cb_x);
        tile_regs_acquire();
        for (uint32_t g = 0; g < G; ++g) {
            matmul_tiles(cb_a, cb_x, g, g * GPC + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        hs.reserve_back(1);
        pack_tile(0, cb_hs);
        hs.push_back(1);
        tile_regs_release();
    }
    STAMP(12)
    hs.wait_front(GPC);
    pack_reconfig_data_format(cb_sqh);
    for (uint32_t j = 0; j < GPC; ++j) {
        reconfig_data_format(cb_hs, cb_hs);
        mul_init(cb_hs, cb_hs);
        tile_regs_acquire();
        mul_tiles(cb_hs, cb_hs, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        sqh.reserve_back(1);
        pack_tile(0, cb_sqh);
        sqh.push_back(1);
        tile_regs_release();
    }
    STAMP(13)
    sqh.wait_front(GPC);
    reconfig_data_format(cb_sqh, cb_cst);
    matmul_init(cb_sqh, cb_cst);
    tile_regs_acquire();
    for (uint32_t j = 0; j < GPC; ++j) {
        matmul_tiles(cb_sqh, cb_cst, j, C_ONES, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    p2.reserve_back(1);
    pack_tile(0, cb_p2);
    p2.push_back(1);
    tile_regs_release();

    STAMP(14)
    // ---- rs = rsqrt(R @ ONES + D*eps) ----
    tot.wait_front(1);
    reconfig_data_format(cb_tot, cb_cst);
    matmul_init(cb_tot, cb_cst);
    tile_regs_acquire();
    matmul_tiles(cb_tot, cb_cst, 0, C_ONES, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, EPS_2_BITS);
    if constexpr (!(CABL & 4)) {
        rsqrt_tile_init();
        rsqrt_tile(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();
    STAMP(15)
    rs.wait_front(1);
    w4.wait_front(GPC);

    for (uint32_t j = 0; j < GPC; ++j) {
        reconfig_data_format(cb_hs, cb_rs);
        mul_init(cb_hs, cb_rs);
        tile_regs_acquire();
        mul_tiles(cb_hs, cb_rs, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        tmp.reserve_back(1);
        pack_tile(0, cb_tmp);
        tmp.push_back(1);
        tile_regs_release();
    }
    STAMP(16)
    tmp.wait_front(GPC);
    pack_reconfig_data_format(cb_out);
    for (uint32_t j = 0; j < GPC; ++j) {
        reconfig_data_format(cb_tmp, cb_w4);
        mul_init(cb_tmp, cb_w4);
        tile_regs_acquire();
        mul_tiles(cb_tmp, cb_w4, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    }
    STAMP(17)
}
