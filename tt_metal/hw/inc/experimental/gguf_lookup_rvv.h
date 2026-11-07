// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "experimental/gguf_page.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>

namespace experimental::gguf {

// lookup_grid (gguf_page.h) on the RISC-V vector unit of Blackhole's TRISC2 (ComputeConfig::enable_trisc2_rvv), with
// the same arguments and results: 32 level codes per indexed load (vluxei16) from the same table. A page's level codes
// in order are the table entries of its index bytes, high bits added: code 4m + k from byte k of index word m.
// IQ2_XXS / IQ3_XXS widen the bytes to table offsets; the others build the scalar path's two 16-bit index pairs per
// index word on 32-bit lanes and interleave them as 16-bit lanes (a + 0x10000 b by a widening add and multiply-add).
template <Format FMT, typename Out>
inline __attribute__((always_inline)) void lookup_grid_rvv(
    const std::uint32_t* indices,
    const std::uint16_t* table,
    Out* levels,
    const std::uint32_t part = 0,
    const std::uint32_t parts = 1) {
    static_assert(is_codebook<FMT>(), "lookup_grid_rvv: not a codebook format");
    constexpr std::uint32_t words = FMT == Format::IQ3_S || FMT == Format::IQ3_XXS ? 64 : 32;
    const std::uint32_t m_begin = part * words / parts;
    const std::uint32_t m_end = (part + 1) * words / parts;
    auto* out = reinterpret_cast<std::uint16_t*>(const_cast<std::uint32_t*>(levels));
    constexpr std::size_t VL8 = 8, VL32 = 32;
    if constexpr (FMT == Format::IQ2_XXS || FMT == Format::IQ3_XXS) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(indices);
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8) {
            const vuint8m2_t b = __riscv_vle8_v_u8m2(bytes + 4 * m0, VL32);
            const vuint16m4_t off = __riscv_vwaddu_vv_u16m4(b, b, VL32);  // 2 * index: byte offsets
            __riscv_vse16_v_u16m4(out + 4 * m0, __riscv_vluxei16_v_u16m4(table, off, VL32), VL32);
        }
    } else {
        constexpr bool iq3 = FMT == Format::IQ3_S;
        constexpr bool xs = FMT == Format::IQ2_XS;
        constexpr std::uint32_t lo = 0x00ff00ffu;
        constexpr std::uint32_t hi_mask = FMT == Format::IQ2_S ? 0x03000300u : xs || iq3 ? 0x01000100u : 0x07000700u;
        const vuint32m2_t lane = __riscv_vid_v_u32m2(VL8);  // b, the index word of the batch
        vuint32m2_t sh_ac = lane, sh_bd_r = lane, sh_bd_l = lane, h_at = lane;
        if constexpr (iq3) {  // h_ac = hw << (8 - b), h_bd = hw >> b
            sh_ac = __riscv_vrsub_vx_u32m2(lane, 8, VL8);
        } else if constexpr (!xs) {
            // word b's high bits in word b / 2; odd b: h_ac = h << 8, h_bd = h << (8 - w_bits); even: h, h >> w_bits
            constexpr std::uint32_t w_bits = FMT == Format::IQ2_S ? 2 : 3;
            const vuint32m2_t odd = __riscv_vand_vx_u32m2(lane, 1, VL8);
            sh_ac = __riscv_vsll_vx_u32m2(odd, 3, VL8);
            sh_bd_l = __riscv_vmul_vx_u32m2(odd, 8 - w_bits, VL8);
            sh_bd_r = __riscv_vmul_vx_u32m2(__riscv_vxor_vx_u32m2(odd, 1, VL8), w_bits, VL8);
            h_at = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(lane, 1, VL8), 2, VL8);  // byte offsets of word b / 2
        }
        const vuint16m4_t e = __riscv_vid_v_u16m4(VL32);
        vuint16m4_t sc_at = e, sc_sr = e, sc_keep = e;
        if constexpr (iq3) {  // scale byte b's low / high nibble into bits 12..15 of codes 4b, 4b + 1
            const vuint16m4_t k = __riscv_vand_vx_u16m4(e, 3, VL32);
            sc_at = __riscv_vsrl_vx_u16m4(e, 2, VL32);
            sc_sr = __riscv_vsll_vx_u16m4(k, 2, VL32);
            sc_keep =
                __riscv_vmul_vx_u16m4(__riscv_vxor_vx_u16m4(__riscv_vsrl_vx_u16m4(k, 1, VL32), 1, VL32), 0xf000, VL32);
        }
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8) {
            const vuint32m2_t x = __riscv_vle32_v_u32m2(indices + m0, VL8);
            vuint32m2_t h_ac, h_bd;
            if constexpr (iq3) {
                const vuint32m2_t hw = __riscv_vmv_v_x_u32m2(indices[64 + (m0 >> 3)], VL8);
                h_ac = __riscv_vsll_vv_u32m2(hw, sh_ac, VL8);
                h_bd = __riscv_vsrl_vv_u32m2(hw, sh_bd_r, VL8);
            } else if constexpr (xs) {  // word b's ninth bits in sign word 2b of the batch's row
                const vuint32m2_t h = __riscv_vlse32_v_u32m2(indices - 32 + 16 * (m0 >> 4) + ((m0 >> 3) & 1), 8, VL8);
                h_ac = __riscv_vsll_vx_u32m2(h, 1, VL8);
                h_bd = __riscv_vsrl_vx_u32m2(h, 7, VL8);
            } else {
                const vuint32m2_t h = __riscv_vluxei32_v_u32m2(indices + 32 + m0 / 2, h_at, VL8);
                h_ac = __riscv_vsll_vv_u32m2(h, sh_ac, VL8);
                h_bd = __riscv_vsll_vv_u32m2(__riscv_vsrl_vv_u32m2(h, sh_bd_r, VL8), sh_bd_l, VL8);
            }
            const vuint32m2_t ac =
                __riscv_vor_vv_u32m2(__riscv_vand_vx_u32m2(x, lo, VL8), __riscv_vand_vx_u32m2(h_ac, hi_mask, VL8), VL8);
            const vuint32m2_t bd = __riscv_vor_vv_u32m2(
                __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(x, 8, VL8), lo, VL8),
                __riscv_vand_vx_u32m2(h_bd, hi_mask, VL8),
                VL8);
            // Byte offsets (2 * index; an index has at most 11 bits, so no carry crosses the 16-bit halves).
            const vuint16m2_t a = __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vsll_vx_u32m2(ac, 1, VL8));
            const vuint16m2_t d = __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vsll_vx_u32m2(bd, 1, VL8));
            const vuint32m4_t z = __riscv_vwmaccu_vx_u32m4(__riscv_vwaddu_vv_u32m4(a, d, 2 * VL8), 0xffff, d, 2 * VL8);
            vuint16m4_t g = __riscv_vluxei16_v_u16m4(table, __riscv_vreinterpret_v_u32m4_u16m4(z), VL32);
            if constexpr (iq3) {
                if (((m0 >> 3) & 3) == 0) {
                    const auto* sc8 = reinterpret_cast<const std::uint8_t*>(indices + 72 + 2 * (m0 >> 5));
                    const vuint16m4_t s = __riscv_vwaddu_vx_u16m4(__riscv_vluxei16_v_u8m2(sc8, sc_at, VL32), 0, VL32);
                    const vuint16m4_t nib = __riscv_vsll_vx_u16m4(__riscv_vsrl_vv_u16m4(s, sc_sr, VL32), 12, VL32);
                    g = __riscv_vor_vv_u16m4(g, __riscv_vand_vv_u16m4(nib, sc_keep, VL32), VL32);
                }
            }
            __riscv_vse16_v_u16m4(out + 4 * m0, g, VL32);
        }
    }
}

}  // namespace experimental::gguf

#endif  // __riscv_vector
