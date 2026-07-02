// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"

using namespace ckernel;
using namespace ckernel::math;

static constexpr char _llk_math_face_compressed_mm_code_sequence_[] = "niNinINIncNcnCNC";
static_assert(
    sizeof(_llk_math_face_compressed_mm_code_sequence_) - 1 <= 16, "Code sequence length must not be greater than 16");

inline constexpr std::uint32_t _llk_math_face_compressed_mm_find_(const char* needle) {
    constexpr std::uint32_t hay_len = sizeof(_llk_math_face_compressed_mm_code_sequence_) - 1;
    std::uint32_t len = 0;
    while (needle[len] != '\0') {
        ++len;
    }
    for (std::uint32_t i = 0; i + len <= hay_len; ++i) {
        std::uint32_t k = 0;
        for (; k < len; ++k) {
            if (_llk_math_face_compressed_mm_code_sequence_[i + k] != needle[k]) {
                break;
            }
        }
        if (k == len) {
            return lltt::replay_insn(ckernel::math::replay_buf_offset + i, len);
        }
    }
    return 0;  // not found
}

inline constexpr std::array<char, 17> _llk_math_face_compressed_mm_encode_(const char* tmpl, std::uint32_t mask) {
    std::array<char, 17> out{};
    std::uint32_t n = 0;
    for (; tmpl[n] != '\0'; ++n) {
        out[n] = ((mask >> n) & 1u) ? static_cast<char>(tmpl[n] - ('a' - 'A')) : tmpl[n];
    }
    out[n] = '\0';
    return out;
}

inline constexpr bool _llk_math_face_compressed_mm_all_found_(const std::array<std::uint32_t, 64>& table) {
    for (std::uint32_t entry : table) {
        if (entry == 0) {
            return false;
        }
    }
    return true;
}

inline constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_build_even_l1_() {
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        const std::uint32_t faces = (m >> 2) & 0b111;
        table[m] = TT_OP_MOP(p_mop::MASK_LOOP, 2, faces);
    }
    return table;
}

inline constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_build_even_l2_() {
    constexpr auto find = [](const char* s) constexpr { return _llk_math_face_compressed_mm_find_(s); };
    const std::array<std::uint32_t, 8> lut = {{
        find("n"),  // 000 zero face3, nopB (CONV3S1 @ ADDR_MOD_0)
        find("i"),  // 001 zero face3, incB (CONV3S1 @ ADDR_MOD_1)
        find("c"),  // 010 zero face3, clrB (CONV3S1 @ ADDR_MOD_2)
        TT_OP_NOP,  // 011 zero face3, invalid hdr
        find("N"),  // 100 data face3, nopB (MVMUL   @ ADDR_MOD_0)
        find("I"),  // 101 data face3, incB (MVMUL   @ ADDR_MOD_1)
        find("C"),  // 110 data face3, clrB (MVMUL   @ ADDR_MOD_2)
        TT_OP_NOP,  // 111 data face3, invalid hdr
    }};
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        const std::uint32_t hdr = m & 0b11;
        const std::uint32_t face3 = (m >> 5) & 0b1;
        table[m] = lut[(face3 << 2) | hdr];
    }
    return table;
}

inline constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_build_odd_l1_() {
    auto entry = [](std::uint32_t hdr, std::uint32_t faces) -> std::uint32_t {
        constexpr auto find = [](const char* s) constexpr { return _llk_math_face_compressed_mm_find_(s); };
        constexpr auto encode = [](const char* t, std::uint32_t m) constexpr {
            return _llk_math_face_compressed_mm_encode_(t, m);
        };
        switch (hdr) {
            case 0b00: return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces);    // nopB
            case 0b01: return find(encode("ni", faces & 0b11).data());  // midInc
            case 0b10: return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces);    // endInc
            case 0b11: return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces);    // endClr
            default: return 0;
        }
    };
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        const std::uint32_t hdr = m & 0b11;
        const std::uint32_t faces = (m >> 2) & 0b111;
        table[m] = entry(hdr, faces);
    }
    return table;
}

inline constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_build_odd_l2_() {
    auto entry = [](std::uint32_t hdr, std::uint32_t faces23) -> std::uint32_t {
        constexpr auto find = [](const char* s) constexpr { return _llk_math_face_compressed_mm_find_(s); };
        switch (hdr) {
            case 0b00: return faces23 & 0b10 ? find("N") : find("n");   // nopB
            case 0b01: return TT_OP_MOP(p_mop::MASK_LOOP, 1, faces23);  // midInc
            case 0b10: return faces23 & 0b10 ? find("I") : find("i");   // endInc
            case 0b11: return faces23 & 0b10 ? find("C") : find("c");   // endClr
            default: return 0;
        }
    };
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        const std::uint32_t hdr = m & 0b11;
        const std::uint32_t faces23 = (m >> 4) & 0b11;
        table[m] = entry(hdr, faces23);
    }
    return table;
}

inline constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_build_one_l1_() {
    constexpr auto find = [](const char* s) constexpr { return _llk_math_face_compressed_mm_find_(s); };
    constexpr auto encode = [](const char* t, std::uint32_t m) constexpr {
        return _llk_math_face_compressed_mm_encode_(t, m);
    };
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        const std::uint32_t faces01 = (m >> 2) & 0b11;
        table[m] = find(encode("ni", faces01).data());  // always midInc
    }
    return table;
}

inline constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_build_one_l2_() {
    auto entry = [](std::uint32_t hdr, std::uint32_t faces23) -> std::uint32_t {
        constexpr auto find = [](const char* s) constexpr { return _llk_math_face_compressed_mm_find_(s); };
        constexpr auto encode = [](const char* t, std::uint32_t m) constexpr {
            return _llk_math_face_compressed_mm_encode_(t, m);
        };
        switch (hdr) {
            case 0b00: return find(encode("ni", faces23).data());  // endInc
            case 0b01: return find(encode("nc", faces23).data());  // endClr
            case 0b10: return TT_OP_NOP;                           // invalid
            case 0b11: return TT_OP_NOP;                           // invalid
            default: return 0;                                     // only hdr 00 and 01 are valid for CT==1
        }
    };
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        const std::uint32_t hdr = m & 0b11;
        const std::uint32_t faces23 = (m >> 4) & 0b11;
        table[m] = entry(hdr, faces23);
    }
    return table;
}

static constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_even_l1_table_ =
    _llk_math_face_compressed_mm_build_even_l1_();
static constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_even_l2_table_ =
    _llk_math_face_compressed_mm_build_even_l2_();
static constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_odd_l1_table_ =
    _llk_math_face_compressed_mm_build_odd_l1_();
static constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_odd_l2_table_ =
    _llk_math_face_compressed_mm_build_odd_l2_();
static constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_one_l1_table_ =
    _llk_math_face_compressed_mm_build_one_l1_();
static constexpr std::array<std::uint32_t, 64> _llk_math_face_compressed_mm_one_l2_table_ =
    _llk_math_face_compressed_mm_build_one_l2_();
static_assert(
    _llk_math_face_compressed_mm_all_found_(_llk_math_face_compressed_mm_even_l1_table_),
    "face_compressed_mm (math): an even-CT L1 fragment is not a substring of the code sequence");
static_assert(
    _llk_math_face_compressed_mm_all_found_(_llk_math_face_compressed_mm_even_l2_table_),
    "face_compressed_mm (math): an even-CT L2 fragment is not a substring of the code sequence");
static_assert(
    _llk_math_face_compressed_mm_all_found_(_llk_math_face_compressed_mm_odd_l1_table_),
    "face_compressed_mm (math): an odd-CT L1 fragment is not a substring of the code sequence");
static_assert(
    _llk_math_face_compressed_mm_all_found_(_llk_math_face_compressed_mm_odd_l2_table_),
    "face_compressed_mm (math): an odd-CT L2 fragment is not a substring of the code sequence");
static_assert(
    _llk_math_face_compressed_mm_all_found_(_llk_math_face_compressed_mm_one_l1_table_),
    "face_compressed_mm (math): a CT==1 L1 fragment is not a substring of the code sequence");
static_assert(
    _llk_math_face_compressed_mm_all_found_(_llk_math_face_compressed_mm_one_l2_table_),
    "face_compressed_mm (math): a CT==1 L2 fragment is not a substring of the code sequence");

inline void _llk_math_face_compressed_mm_addrmod_config_() {
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);  // nopB

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_1);  // incB

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_2);  // clrB
}

inline void _llk_math_face_compressed_mm_mop_config_() {
    constexpr std::size_t code_len = sizeof(_llk_math_face_compressed_mm_code_sequence_) - 1;

    auto instr_for_code = [](char code) {
        switch (code) {
            case 'n': TTI_ZEROACC(p_zeroacc::CLR_16, 0, 0, ADDR_MOD_0, 0xff); break;
            case 'N': TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_0, 0); break;
            case 'i': TTI_ZEROACC(p_zeroacc::CLR_16, 0, 0, ADDR_MOD_1, 0xff); break;
            case 'I': TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_1, 0); break;
            case 'c': TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); break;
            case 'C': TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_2, 0); break;
            default: LLK_ASSERT(false, "Invalid code for math instruction"); break;
        }
    };

    load_replay_buf(ckernel::math::replay_buf_offset, code_len, [&instr_for_code] {
        auto emit = [&](auto self, auto idx) -> void {
            constexpr std::size_t i = decltype(idx)::value;
            if constexpr (i < code_len) {
                instr_for_code(_llk_math_face_compressed_mm_code_sequence_[i]);
                self(self, std::integral_constant<std::size_t, i + 1>{});
            }
        };
        emit(emit, std::integral_constant<std::size_t, 0>{});
    });

    constexpr std::uint32_t op0 = _llk_math_face_compressed_mm_find_("n");
    constexpr std::uint32_t op1 = _llk_math_face_compressed_mm_find_("N");

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,      // unpackB    = false
        false,      // unpackHalo = false
        op0,        // A
        TT_OP_NOP,  // A1    (unused)
        TT_OP_NOP,  // A2    (unused)
        TT_OP_NOP,  // A3    (unused)
        op1,        // skipA
        TT_OP_NOP,  // B     (unused)
        TT_OP_NOP   // skipB (unused)
    );
    tmp.program();
}

inline void _llk_math_face_compressed_mm_init_() {
    _llk_math_face_compressed_mm_addrmod_config_();
    _llk_math_face_compressed_mm_mop_config_();

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <std::uint32_t ct_dim>
inline constexpr std::array<const std::uint32_t*, 2> _llk_math_face_compressed_mm_tables_() {
    if constexpr (ct_dim == 1) {
        return {{_llk_math_face_compressed_mm_one_l1_table_.data(), _llk_math_face_compressed_mm_one_l2_table_.data()}};
    } else if constexpr (ct_dim % 2 == 0) {
        return {
            {_llk_math_face_compressed_mm_even_l1_table_.data(), _llk_math_face_compressed_mm_even_l2_table_.data()}};
    } else {
        return {{_llk_math_face_compressed_mm_odd_l1_table_.data(), _llk_math_face_compressed_mm_odd_l2_table_.data()}};
    }
}

template <std::uint32_t ct_dim = 1>
inline void _llk_math_face_compressed_mm_(
    const std::uint32_t base_address_meta, const std::uint32_t dst_index, const std::uint32_t kt_dim) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    const std::uint32_t iters = kt_dim * ct_dim;
    const std::uint32_t full_iters = iters / 5;
    const std::uint32_t rem_iters = iters % 5;
    const std::uint32_t* meta_ptr = reinterpret_cast<const std::uint32_t*>(base_address_meta);

    constexpr auto tables = _llk_math_face_compressed_mm_tables_<ct_dim>();
    constexpr const std::uint32_t* l1_table = tables[0];
    constexpr const std::uint32_t* l2_table = tables[1];

    for (std::uint32_t i = 0; i < full_iters; ++i) {
        std::uint32_t meta = meta_ptr[i];

        std::uint32_t idx0 = (meta >> 0) & 0b111111;
        std::uint32_t idx1 = (meta >> 6) & 0b111111;
        std::uint32_t idx2 = (meta >> 12) & 0b111111;
        std::uint32_t idx3 = (meta >> 18) & 0b111111;
        std::uint32_t idx4 = (meta >> 24) & 0b111111;

        std::uint32_t data0 = l1_table[idx0];
        std::uint32_t data1 = l2_table[idx0];
        std::uint32_t data2 = l1_table[idx1];
        std::uint32_t data3 = l2_table[idx1];
        std::uint32_t data4 = l1_table[idx2];
        std::uint32_t data5 = l2_table[idx2];
        std::uint32_t data6 = l1_table[idx3];
        std::uint32_t data7 = l2_table[idx3];
        std::uint32_t data8 = l1_table[idx4];
        std::uint32_t data9 = l2_table[idx4];

        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
        ckernel::instrn_buffer[0] = data2;
        ckernel::instrn_buffer[0] = data3;
        ckernel::instrn_buffer[0] = data4;
        ckernel::instrn_buffer[0] = data5;
        ckernel::instrn_buffer[0] = data6;
        ckernel::instrn_buffer[0] = data7;
        ckernel::instrn_buffer[0] = data8;
        ckernel::instrn_buffer[0] = data9;
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t i = 0; i < rem_iters; ++i) {
        std::uint32_t idx0 = meta & 0b111111;
        std::uint32_t data0 = l1_table[idx0];
        std::uint32_t data1 = l2_table[idx0];
        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
        meta >>= 6;
    }
}
