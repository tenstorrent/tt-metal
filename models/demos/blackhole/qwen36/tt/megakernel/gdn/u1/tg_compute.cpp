// lever 1 producer compute: gdn_spec_step's lz_sfpu_gates_face (the prod SFPU_GATE_FACE path, user 0) on the raw a|b tile:
// gt[0] = exp(nea * softplus(ab + dtb)) (dec), gt[1] = sigmoid(ab) (beta), same ops and order.
// softplus_tile picks its algorithm on INP_FLOAT32: gdn_step_helpers.hpp defines it for every GDN compute kernel (the
// fp32 branch); without it the BF16 polynomial runs and the decays differ (the first lever-1 run: dec off by ~3e-4)
#ifndef INP_FLOAT32
#define INP_FLOAT32
#endif
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tg_common.hpp"

// gdn_spec_step's sfpu_face: one face routine on the face holding the read elements (SFPU_GATE_FACE, the prod path)
template <typename F, typename... Args>
ALWI void sfpu_face(F f, uint32_t base_tile, bool bottom, Args... args) {
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

void kernel_main() {
    const uint32_t dtb_bits = get_arg_val<uint32_t>(0), nea_bits = get_arg_val<uint32_t>(1);
    constexpr uint32_t kOneBits = 0x3F800000u, kTwentyBits = 0x41A00000u;
    compute_kernel_hw_startup(TG_AB, TG_AB, TG_GT);
    DeviceZoneScopedN("TG-C");
    const uint32_t halves = get_arg_val<uint32_t>(2);  // 1: rows 0..15 (users 0..3), 2: also rows 16..31
    cb_wait_front(TG_AB, 1);
    cb_reserve_back(TG_GT, 2);
    pack_reconfig_data_format(TG_GT);
    reconfig_data_format_srca(TG_AB);
    copy_init(TG_AB);
    tile_regs_acquire();
    copy_tile(TG_AB, 0, 0);
    copy_tile(TG_AB, 0, 1);
    for (uint32_t hf = 0; hf < halves; ++hf) {
        const bool hb = hf != 0;
        sigmoid_tile_init();
    #ifdef TRISC_MATH
        sfpu_face(ckernel::sfpu::calculate_sigmoid<false, DST_ACCUM_MODE, 8>, 0, hb);
    #endif
        fill_tile_init();
    #ifdef TRISC_MATH
        sfpu_face(ckernel::sfpu::_calculate_fill_bitcast_<APPROX, 8>, 2, hb, dtb_bits);
    #endif
        add_binary_tile_init();
    #ifdef TRISC_MATH
        sfpu_face(
            ckernel::sfpu::
                calculate_sfpu_binary<APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default>,
            0, hb, 1u, 2u, 1u);
    #endif
        softplus_tile_init();
    #ifdef TRISC_MATH
        sfpu_face(ckernel::sfpu::calculate_softplus<APPROX, DST_ACCUM_MODE, 8>, 1, hb, kOneBits, kOneBits, kTwentyBits);
    #endif
        fill_tile_init();
    #ifdef TRISC_MATH
        sfpu_face(ckernel::sfpu::_calculate_fill_bitcast_<APPROX, 8>, 3, hb, nea_bits);
    #endif
        mul_binary_tile_init();
    #ifdef TRISC_MATH
        sfpu_face(ckernel::sfpu::calculate_sfpu_binary_mul<APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE>, 0, hb, 1u, 3u, 1u);
    #endif
        exp_tile_init<false>();
    #ifdef TRISC_MATH
        sfpu_face(ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 8, true>, 1, hb, p_sfpu::kCONST_1_FP16B);
    #endif

    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, TG_GT, 0);
    pack_tile(0, TG_GT, 1);
    tile_regs_release();
    cb_push_back(TG_GT, 2);
    cb_pop_front(TG_AB, 1);
}
