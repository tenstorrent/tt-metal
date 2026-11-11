// Includable wo unit, compute side (see wo_unit_dm.hpp). The weight / activation formats come from the CB configs.
#pragma once
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"

#ifndef WOU_NOUNROLL
#define WOU_NOUNROLL
#endif

namespace wou {
using namespace ckernel;   // MathFidelity, state_configure, throttled_mop_status

// A (.) S in place over S (cb_as aliases cb_s): wk tiles, 8 per DST round (fp32 DST, full sync). Pops A and S.
ALWI void gate_mul(uint32_t cb_a, uint32_t cb_s, uint32_t cb_as, uint32_t wk) {
    cb_wait_front(cb_a, wk);
    cb_wait_front(cb_s, wk);
    reconfig_data_format(cb_a, cb_s);
    mul_init(cb_a, cb_s);
    pack_reconfig_data_format(cb_as);
    for (uint32_t t = 0; t < wk; t += 8) {
        const uint32_t n = (wk - t) < 8 ? (wk - t) : 8;
        tile_regs_acquire();
        WOU_NOUNROLL
        for (uint32_t i = 0; i < n; ++i) mul_tiles(cb_a, cb_s, t + i, t + i, i);
        tile_regs_commit();
        cb_reserve_back(cb_as, n);
        tile_regs_wait();
        WOU_NOUNROLL
        for (uint32_t i = 0; i < n; ++i) pack_tile(i, cb_as);   // S slots t.., already unpacked this round
        tile_regs_release();
        cb_push_back(cb_as, n);
    }
    cb_pop_front(cb_a, wk);
    cb_pop_front(cb_s, wk);
}

// matmul_block{_init} at LoFi when kLoFi (the kernel's MATH_FIDELITY otherwise; MATH_FIDELITY exists only on the math
// TRISC). On BFP4 weights (srcA) LoFi == HiFi2 (the weight mantissa fits the first phase), i.e. the prod BFP4 out-proj
// numerics under QWEN36_BFP4_LAYOUT_LOFI; lets a HiFi4 kernel (the GDN role) run gdn_out at LoFi speed.
template <bool kLoFi>
ALWI void mm_block_init_f(uint32_t cb_a, uint32_t cb_w, uint32_t ct, uint32_t kt) {
    if constexpr (!kLoFi) {
        matmul_block_init(cb_a, cb_w, 0, ct, 1, kt);
    } else {
        state_configure(cb_w, cb_a, __builtin_LINE());
        UNPACK((llk_unpack_AB_matmul_init(cb_a, cb_w, 0, ct, 1, kt)));
        MATH((llk_math_matmul_init<MathFidelity::LoFi, MM_THROTTLE>(cb_a, cb_w, 0, ct, 1)));
#ifdef ARCH_BLACKHOLE
        MATH((throttled_mop_status = 0));
#endif
    }
}
template <bool kLoFi>
ALWI void mm_block_f(uint32_t cb_a, uint32_t cb_w, uint32_t ia, uint32_t iw, uint32_t ct, uint32_t kt) {
    if constexpr (!kLoFi) {
        matmul_block(cb_a, cb_w, ia, iw, 0, 0, ct, 1, kt);
    } else {
        state_configure(cb_w, cb_a, __builtin_LINE());
        UNPACK((llk_unpack_AB_matmul(cb_a, cb_w, ia, iw, ct, 1, kt)));
        MATH((llk_math_matmul<MathFidelity::LoFi, MM_THROTTLE>(0, ct, 1)));
    }
}

// out = A @ W slice: A = wk tiles at the front of cb_a, W = the ring's k-major nt-wide slice (ring_pages slot);
// <= max_nw output tiles per DST pass (5: fp32 DST in full-sync mode, 8 tiles; 4: half-sync, e.g. the GDN role);
// output pushed as a full `out_slot` slot of cb_out. Pops A and the ring slot.
template <bool kLoFi = false>
ALWI void matmul_nt(uint32_t cb_a, uint32_t cb_w, uint32_t ring_pages, uint32_t cb_out, uint32_t out_slot, uint32_t wk, uint32_t nt,
                    uint32_t max_nw = 5) {
    cb_wait_front(cb_a, wk);
    cb_wait_front(cb_w, ring_pages);
    reconfig_data_format(cb_w, cb_a);
    cb_reserve_back(cb_out, out_slot);
    pack_reconfig_data_format(cb_out);
    for (uint32_t j0 = 0; j0 < nt; j0 += max_nw) {
        const uint32_t nw = (nt - j0) < max_nw ? (nt - j0) : max_nw;
        mm_block_init_f<kLoFi>(cb_a, cb_w, nw, wk);
        tile_regs_acquire();
        WOU_NOUNROLL
        for (uint32_t k = 0; k < wk; ++k) mm_block_f<kLoFi>(cb_a, cb_w, k, k * nt + j0, nw, wk);
        tile_regs_commit();
        tile_regs_wait();
        WOU_NOUNROLL
        for (uint32_t j = 0; j < nw; ++j) pack_tile<true>(j, cb_out, j0 + j);
        tile_regs_release();
    }
    cb_push_back(cb_out, out_slot);
    cb_pop_front(cb_a, wk);
    cb_pop_front(cb_w, ring_pages);
}

// GOTAIL: matmul_nt with (kKs, E1_GO_KSTREAM) the first DST pass waiting for A tile by tile (the reader pushes each A tile
// as its source flag lands; every output tile still accumulates k = 0 .. wk - 1 in the same order -> same result) and
// (kPushPass, E1_GO_SCPASS) each pass's output tiles pushed when packed (packed relative to the advanced write pointer),
// the slot's padding (out_slot - nt) pushed at the end.
template <bool kLoFi, bool kKs, bool kPushPass>
ALWI void matmul_nt_ks(uint32_t cb_a, uint32_t cb_w, uint32_t ring_pages, uint32_t cb_out, uint32_t out_slot, uint32_t wk, uint32_t nt,
                       uint32_t max_nw) {
    if constexpr (!kKs) cb_wait_front(cb_a, wk);
    cb_wait_front(cb_w, ring_pages);
    reconfig_data_format(cb_w, cb_a);
    cb_reserve_back(cb_out, out_slot);
    pack_reconfig_data_format(cb_out);
    for (uint32_t j0 = 0; j0 < nt; j0 += max_nw) {
        const uint32_t nw = (nt - j0) < max_nw ? (nt - j0) : max_nw;
        mm_block_init_f<kLoFi>(cb_a, cb_w, nw, wk);
        tile_regs_acquire();
        WOU_NOUNROLL
        for (uint32_t k = 0; k < wk; ++k) {
            if constexpr (kKs) {
                if (j0 == 0) cb_wait_front(cb_a, k + 1);
            }
            mm_block_f<kLoFi>(cb_a, cb_w, k, k * nt + j0, nw, wk);
        }
        tile_regs_commit();
        tile_regs_wait();
        WOU_NOUNROLL
        for (uint32_t j = 0; j < nw; ++j) pack_tile<true>(j, cb_out, kPushPass ? j : j0 + j);
        tile_regs_release();
        if constexpr (kPushPass) cb_push_back(cb_out, nw);
    }
    if constexpr (kPushPass) {
        if (out_slot > nt) cb_push_back(cb_out, out_slot - nt);
    } else {
        cb_push_back(cb_out, out_slot);
    }
    cb_pop_front(cb_a, wk);
    cb_pop_front(cb_w, ring_pages);
}

}  // namespace wou
