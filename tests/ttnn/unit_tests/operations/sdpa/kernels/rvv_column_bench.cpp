// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf research (not for merge): cost of SDPA column math on the TRISC2 RISC-V vector unit (Zve32f).
// Scratch layout (runtime arg 0, an L1 buffer on this core), in bytes:
//   0     m_old, a BF16 32x32 tile in face order (only column 0 is read)
//   2048  m_new, same
//   4096  c = exp(scale * (m_old - m_new)) for the 32 rows, FP32
//   4224  c written back as column 0 of a BF16 tile (CB 14 layout), 2 KB
//   6272  stats: [vlmax_e32m1, vlmax_e32m4, ticks_c_fp32, ticks_c_bf16tile, ticks_fold, reps, 0, 0]
//   6304  fold test: hi[32], lo[32], chunk[32] FP32 in, hi'/lo' out in place; 6688 saved copy
// Column 0 of a face-ordered tile: rows 0-15 at element 16 * r (face 0), rows 16-31 at 512 + 16 * r (face 2).

#include <cstdint>

#include "api/compute/common.h"

#if defined(TRISC_PACK) && defined(__riscv_vector)
#include <riscv_vector.h>

namespace {
inline uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// exp(x) for x <= 0 in FP32: n = round(x * log2e), r = x - n * ln2, degree-5 Taylor, scale by 2^n.
inline vfloat32m1_t vexp(vfloat32m1_t x, size_t vl) {
    x = __riscv_vfmax_vf_f32m1(x, -87.0f, vl);
    vint32m1_t n = __riscv_vfcvt_x_f_v_i32m1(__riscv_vfmul_vf_f32m1(x, 1.4426950408889634f, vl), vl);
    vfloat32m1_t nf = __riscv_vfcvt_f_x_v_f32m1(n, vl);
    vfloat32m1_t r = __riscv_vfnmsac_vf_f32m1(x, 0.693145752f, nf, vl);
    r = __riscv_vfnmsac_vf_f32m1(r, 1.42860677e-6f, nf, vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.0f / 120, vl);
    p = __riscv_vfmadd_vv_f32m1(p, r, __riscv_vfmv_v_f_f32m1(1.0f / 24, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, r, __riscv_vfmv_v_f_f32m1(1.0f / 6, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, r, __riscv_vfmv_v_f_f32m1(0.5f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, r, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, r, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vint32m1_t bits = __riscv_vadd_vv_i32m1(
        __riscv_vreinterpret_v_f32m1_i32m1(p), __riscv_vsll_vx_i32m1(n, 23, vl), vl);
    return __riscv_vreinterpret_v_i32m1_f32m1(bits);
}

// Load 16 BF16 column values of one face (stride 32 bytes) as FP32.
inline vfloat32m1_t load_col(const uint16_t* face, size_t vl) {
    vuint16mf2_t h = __riscv_vlse16_v_u16mf2(face, 32, vl);
    vuint32m1_t w = __riscv_vsll_vx_u32m1(__riscv_vzext_vf2_u32m1(h, vl), 16, vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(w);
}

// c for 32 rows. to_tile: also round to BF16 and store as column 0 of a face-ordered tile.
template <bool to_tile>
inline void correction(uint8_t* s, float scale) {
    const uint16_t* m_old = reinterpret_cast<const uint16_t*>(s);
    const uint16_t* m_new = reinterpret_cast<const uint16_t*>(s + 2048);
    float* c = reinterpret_cast<float*>(s + 4096);
    uint16_t* c_tile = reinterpret_cast<uint16_t*>(s + 4224);
    for (uint32_t face = 0; face < 2; ++face) {
        size_t vl = __riscv_vsetvl_e32m1(16);
        for (uint32_t r0 = 0; r0 < 16; r0 += vl) {
            vl = __riscv_vsetvl_e32m1(16 - r0);
            const uint32_t e = face * 512 + r0 * 16;
            vfloat32m1_t d = __riscv_vfsub_vv_f32m1(load_col(m_old + e, vl), load_col(m_new + e, vl), vl);
            vfloat32m1_t y = vexp(__riscv_vfmul_vf_f32m1(d, scale, vl), vl);
            if constexpr (to_tile) {
                // Round to nearest even BF16 and store with the tile's column stride.
                vuint32m1_t b = __riscv_vreinterpret_v_f32m1_u32m1(y);
                vuint32m1_t lsb = __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(b, 16, vl), 1, vl);
                b = __riscv_vadd_vv_u32m1(b, __riscv_vadd_vx_u32m1(lsb, 0x7fff, vl), vl);
                __riscv_vsse16_v_u16mf2(c_tile + e, 32, __riscv_vnsrl_wx_u16mf2(b, 16, vl), vl);
            } else {
                __riscv_vse32_v_f32m1(c + face * 16 + r0, y, vl);
            }
        }
    }
}

// Compensated hi/lo fold of one column (32 rows): total = (hi + lo) * c + chunk, hi' = bf16(total),
// lo' = bf16(total - hi'). FP32 lanes; BF16 rounding by RNE on the bit pattern.
inline vfloat32m1_t round_bf16(vfloat32m1_t x, size_t vl) {
    vuint32m1_t b = __riscv_vreinterpret_v_f32m1_u32m1(x);
    vuint32m1_t lsb = __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(b, 16, vl), 1, vl);
    b = __riscv_vadd_vv_u32m1(b, __riscv_vadd_vx_u32m1(lsb, 0x7fff, vl), vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vx_u32m1(b, 0xffff0000u, vl));
}
inline void fold(uint8_t* s) {
    float* hi = reinterpret_cast<float*>(s + 6304);
    float* lo = hi + 32;
    float* chunk = lo + 32;
    const float* c = reinterpret_cast<const float*>(s + 4096);
    size_t vl;
    for (uint32_t i = 0; i < 32; i += vl) {
        vl = __riscv_vsetvl_e32m1(32 - i);
        vfloat32m1_t t = __riscv_vfadd_vv_f32m1(__riscv_vle32_v_f32m1(hi + i, vl), __riscv_vle32_v_f32m1(lo + i, vl), vl);
        t = __riscv_vfmadd_vv_f32m1(t, __riscv_vle32_v_f32m1(c + i, vl), __riscv_vle32_v_f32m1(chunk + i, vl), vl);
        vfloat32m1_t h = round_bf16(t, vl);
        __riscv_vse32_v_f32m1(hi + i, h, vl);
        __riscv_vse32_v_f32m1(lo + i, round_bf16(__riscv_vfsub_vv_f32m1(t, h, vl), vl), vl);
    }
}
}  // namespace
#endif

void kernel_main() {
#if defined(TRISC_PACK) && defined(__riscv_vector)
    uint8_t* s = reinterpret_cast<uint8_t*>(get_arg_val<uint32_t>(0));
    const float scale = __builtin_bit_cast(float, get_arg_val<uint32_t>(1));
    constexpr uint32_t reps = 64;
    volatile uint32_t* stats = reinterpret_cast<volatile uint32_t*>(s + 6272);
    stats[0] = __riscv_vsetvlmax_e32m1();
    stats[1] = __riscv_vsetvlmax_e32m4();
    // Save the fold inputs (at 6688) so the final checked fold starts from the original state.
    float* saved = reinterpret_cast<float*>(s + 6688);
    const float* f = reinterpret_cast<const float*>(s + 6304);
    for (uint32_t i = 0; i < 96; ++i) {
        saved[i] = f[i];
    }
    uint32_t t0 = wall();
    for (uint32_t r = 0; r < reps; ++r) {
        correction<false>(s, scale);
        asm volatile("" ::: "memory");
    }
    uint32_t t1 = wall();
    for (uint32_t r = 0; r < reps; ++r) {
        correction<true>(s, scale);
        asm volatile("" ::: "memory");
    }
    uint32_t t2 = wall();
    for (uint32_t r = 0; r < reps; ++r) {
        fold(s);
        asm volatile("" ::: "memory");
    }
    uint32_t t3 = wall();
    float* g = reinterpret_cast<float*>(s + 6304);
    for (uint32_t i = 0; i < 96; ++i) {
        g[i] = saved[i];
    }
    fold(s);
    stats[2] = t1 - t0;
    stats[3] = t2 - t1;
    stats[4] = t3 - t2;
    stats[5] = reps;
#endif
}
