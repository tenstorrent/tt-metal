// G-b (E1_GB): the in-role GDN prefix, compute side (see gb_dm.hpp). Per layer, in job order (the writer pops in the
// same order): conv jobs (tp_compute kind 0: DST_k = SH_k * tap row k, the qkv tile as SH_{K-1}, DST_0 += DST_1..K-1,
// SiLU -> CV; P @ CV one-hot placement -> OUT), silu(z) (QIN page 2 -> OUT), the gate pair of the a|b tile (QIN page 3,
// tg_compute's SFPU sequence, both face halves -> OUT [dec, beta]). Raw cb API; formats reconfigured per op.
#pragma once
#ifndef INP_FLOAT32
#define INP_FLOAT32
#endif
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"

#pragma GCC push_options
#pragma GCC optimize("Os")
#ifndef GB_CV_PAGES
#define GB_CV_PAGES 4
#endif
namespace gb {

template <typename F, typename... Args>
ALWI void gb_sfpu_face(F f, uint32_t base_tile, bool bottom, Args... args) {
#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(base_tile);
    if (bottom) {
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    f(args...);
    _llk_math_eltwise_sfpu_done_();
#else
    (void)f;
    (void)base_tile;
    (void)bottom;
    ((void)args, ...);
#endif
}

template <uint32_t K>
__attribute__((noinline)) void conv_place(uint32_t cb_sht, uint32_t cb_qpx, uint32_t qpage, uint32_t cb_cv, uint32_t cb_out) {
    // SHT front = [SH_0 .. SH_{K-2} | tap] (the qkv tile in QPX page qpage stands in for SH_{K-1}); P = QPX tile 4; CV = the
    // step's tmp (idle here; reserved / popped whole so its pointer stays at the base for the step's l2)
    constexpr uint32_t kTap = K - 1;
    cb_wait_front(cb_sht, K);
    cb_reserve_back(cb_cv, GB_CV_PAGES);
    pack_reconfig_data_format(cb_cv);
    reconfig_data_format(cb_sht, cb_sht);
    mul_bcast_rows_init(cb_sht, cb_sht);
    tile_regs_acquire();
    for (uint32_t k = 0; k < K - 1; ++k) mul_tiles_bcast_rows(cb_sht, cb_sht, k, kTap, k, k);
    reconfig_data_format(cb_qpx, cb_sht);
    mul_bcast_rows_init(cb_qpx, cb_sht);
    mul_tiles_bcast_rows(cb_qpx, cb_sht, qpage, kTap, K - 1, K - 1);
    add_binary_tile_init();
    for (uint32_t k = 1; k < K; ++k) add_binary_tile(0, k, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_cv, 0);
    tile_regs_release();
    cb_push_back(cb_cv, GB_CV_PAGES);
    cb_pop_front(cb_sht, K);
    // placement: the op's one-hot P @ CV (tp_compute; the matmul's rounding of CV is part of the op's bits)
    cb_wait_front(cb_cv, GB_CV_PAGES);
    cb_reserve_back(cb_out, 1);
    pack_reconfig_data_format(cb_out);
    reconfig_data_format<SrcOrder::Reverse>(cb_qpx, cb_cv);
    matmul_init(cb_qpx, cb_cv);
    tile_regs_acquire();
    matmul_tiles(cb_qpx, cb_cv, 4, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_cv, GB_CV_PAGES);
}

__attribute__((noinline)) void silu_page(uint32_t cb_in, uint32_t page, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_reconfig_data_format(cb_out);
    reconfig_data_format_srca(cb_in);
    copy_init(cb_in);
    silu_tile_init();
    tile_regs_acquire();
    copy_tile(cb_in, page, 0);
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

__attribute__((noinline)) void gates_page(uint32_t cb_in, uint32_t page, uint32_t cb_out, uint32_t dtb_bits, uint32_t nea_bits) {
    constexpr uint32_t kOneBits = 0x3F800000u, kTwentyBits = 0x41A00000u;
    cb_reserve_back(cb_out, 2);
    pack_reconfig_data_format(cb_out);
    reconfig_data_format_srca(cb_in);
    copy_init(cb_in);
    tile_regs_acquire();
    copy_tile(cb_in, page, 0);
    copy_tile(cb_in, page, 1);
#pragma GCC unroll 1
    for (uint32_t hf = 0; hf < 2; ++hf) {
        const bool hb = hf != 0;
        sigmoid_tile_init();
#ifdef TRISC_MATH
        gb_sfpu_face(ckernel::sfpu::calculate_sigmoid<false, DST_ACCUM_MODE, 8>, 0, hb);
#endif
        fill_tile_init();
#ifdef TRISC_MATH
        gb_sfpu_face(ckernel::sfpu::_calculate_fill_bitcast_<APPROX, 8>, 2, hb, dtb_bits);
#endif
        add_binary_tile_init();
#ifdef TRISC_MATH
        gb_sfpu_face(ckernel::sfpu::calculate_sfpu_binary<APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE,
                                                          ckernel::DstRoundingMode::Default>,
                     0, hb, 1u, 2u, 1u);
#endif
        softplus_tile_init();
#ifdef TRISC_MATH
        gb_sfpu_face(ckernel::sfpu::calculate_softplus<APPROX, DST_ACCUM_MODE, 8>, 1, hb, kOneBits, kOneBits, kTwentyBits);
#endif
        fill_tile_init();
#ifdef TRISC_MATH
        gb_sfpu_face(ckernel::sfpu::_calculate_fill_bitcast_<APPROX, 8>, 3, hb, nea_bits);
#endif
        mul_binary_tile_init();
#ifdef TRISC_MATH
        gb_sfpu_face(ckernel::sfpu::calculate_sfpu_binary_mul<APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE>, 0, hb,
                     1u, 3u, 1u);
#endif
        exp_tile_init<false>();
#ifdef TRISC_MATH
        gb_sfpu_face(ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 8, true>, 1, hb,
                     p_sfpu::kCONST_1_FP16B);
#endif
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, cb_out, 0);
    pack_tile(0, cb_out, 1);
    tile_regs_release();
    cb_push_back(cb_out, 2);
}

// pb: this core's G-b payload block (gb_dm.hpp offsets)
template <uint32_t K>
ALWI void compute_layer(volatile tt_l1_ptr uint32_t* pb, uint32_t L, uint32_t cb_qpx, uint32_t npx, uint32_t cb_sht,
                        uint32_t cb_cv, uint32_t cb_out) {
    constexpr uint32_t kPerLayer = 132;  // gb_dm.hpp
    const uint32_t nconv = pb[3], silu = pb[22];
    cb_wait_front(cb_qpx, npx);  // QIN 0..3, P 4..5 (resident data; the pointer returns to the base every layer)
    if (silu) silu_page(cb_qpx, 2, cb_out);  // needs only the pushed z tile (the convs wait the staging)
    {
        DeviceZoneScopedN("GB-CCONV");
        conv_place<K>(cb_sht, cb_qpx, 0, cb_cv, cb_out);  // q / k first on the q / k cores (their consumers' l2)
    }
    {
        DeviceZoneScopedN("GB-CGATE");
        gates_page(cb_qpx, 3, cb_out, pb[kPerLayer + 2 * L], pb[kPerLayer + 2 * L + 1]);  // every core: its head's pair
    }
    if (nconv == 2) {
        DeviceZoneScopedN("GB-CCONV1");
        conv_place<K>(cb_sht, cb_qpx, 1, cb_cv, cb_out);
        cb_reserve_back(cb_out, 1);  // pad: per-layer OUT count even (the gate pair stays contiguous)
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_qpx, npx);
}

}  // namespace gb
#pragma GCC pop_options
