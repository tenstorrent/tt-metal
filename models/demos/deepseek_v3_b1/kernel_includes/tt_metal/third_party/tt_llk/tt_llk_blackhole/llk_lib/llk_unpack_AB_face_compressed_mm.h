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
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

static constexpr char _llk_unpack_AB_face_compressed_mm_code_sequence_[] = "SB0B1S0S1S0S111S00S1S000S11S0S11";
static_assert(
    sizeof(_llk_unpack_AB_face_compressed_mm_code_sequence_) - 1 <= 32,
    "Code sequence length must not be greater than 32");

inline constexpr std::uint32_t _llk_unpack_AB_face_compressed_mm_find_(const char* needle) {
    constexpr std::uint32_t hlen = sizeof(_llk_unpack_AB_face_compressed_mm_code_sequence_) - 1;
    std::uint32_t len = 0;
    while (needle[len] != '\0') {
        ++len;
    }
    for (std::uint32_t i = 0; i + len <= hlen; ++i) {
        std::uint32_t k = 0;
        for (; k < len; ++k) {
            if (_llk_unpack_AB_face_compressed_mm_code_sequence_[i + k] != needle[k]) {
                break;
            }
        }
        if (k == len) {
            return lltt::replay_insn(i, len);
        }
    }
    return 0;  // not found
}

inline constexpr bool _llk_unpack_AB_face_compressed_mm_all_found_(const std::array<std::uint32_t, 64>& table) {
    for (std::uint32_t entry : table) {
        if (entry == 0) {
            return false;
        }
    }
    return true;
}

inline constexpr std::array<std::uint32_t, 64> _llk_unpack_AB_face_compressed_mm_build_header_table_() {
    constexpr auto find = [](const char* s) constexpr { return _llk_unpack_AB_face_compressed_mm_find_(s); };
    const std::array<std::uint32_t, 8> lut = {{
        find("0"),                            // 000  bfp2->bfp2 no B
        find("S0"),                           // 001  bfp4->bfp2 no B
        find("B0"),                           // 010  bfp2->bfp2 with B
        TT_OP_MOP(p_mop::MASK_LOOP, 0, 0x0),  // 011  bfp4->bfp2 with B (SB0)
        find("S1"),                           // 100  bfp2->bfp4 no B
        find("1"),                            // 101  bfp4->bfp4 no B
        TT_OP_MOP(p_mop::MASK_LOOP, 0, 0x1),  // 110  bfp2->bfp4 with B (SB1)
        find("B1"),                           // 111  bfp4->bfp4 with B
    }};
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        table[m] = lut[m & 0x7];
    }
    return table;
}

inline constexpr std::array<std::uint32_t, 64> _llk_unpack_AB_face_compressed_mm_build_pairs_table_() {
    constexpr auto pair_frag = [](std::uint32_t idx) -> const char* {
        switch (idx & 0x3) {
            case 0b00: return "0";   // 00 bfp2->bfp2
            case 0b01: return "S0";  // 01 bfp4->bfp2
            case 0b10: return "S1";  // 10 bfp2->bfp4
            case 0b11: return "1";   // 11 bfp4->bfp4
            default: return "";
        }
    };
    std::array<std::uint32_t, 64> table{};
    for (std::uint32_t m = 0; m < 64; ++m) {
        std::uint32_t v = m >> 2;
        char needle[8] = {};
        std::uint32_t len = 0;
        for (std::uint32_t i = 0; i < 3; ++i) {
            for (const char* p = pair_frag(v & 0x3); *p != '\0'; ++p) {
                needle[len++] = *p;
            }
            v >>= 1;
        }
        table[m] = _llk_unpack_AB_face_compressed_mm_find_(needle);
    }
    return table;
}

static constexpr std::array<std::uint32_t, 64> _llk_unpack_AB_face_compressed_mm_header_table_ =
    _llk_unpack_AB_face_compressed_mm_build_header_table_();
static constexpr std::array<std::uint32_t, 64> _llk_unpack_AB_face_compressed_mm_pairs_table_ =
    _llk_unpack_AB_face_compressed_mm_build_pairs_table_();
static_assert(
    _llk_unpack_AB_face_compressed_mm_all_found_(_llk_unpack_AB_face_compressed_mm_header_table_),
    "face_compressed_mm: a header fragment is not a substring of the code sequence");
static_assert(
    _llk_unpack_AB_face_compressed_mm_all_found_(_llk_unpack_AB_face_compressed_mm_pairs_table_),
    "face_compressed_mm: a pairs fragment is not a substring of the code sequence");

inline void _llk_unpack_AB_face_compressed_mm_mop_config_() {
    constexpr std::size_t code_len = sizeof(_llk_unpack_AB_face_compressed_mm_code_sequence_) - 1;

    auto instr_for_code = [](char code) {
        switch (code) {
            case '0': TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 0, 1, 1); break;  // Ch0Y += 1
            case '1': TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 1, 2, 1); break;  // Ch0Y += 1
            case 'S': TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC); break;
            case 'B': TTI_UNPACR_COMMON(SrcB, 0b00'00'10'10, 1); break;  // Ch0Y += 2 Ch0Z += 2
            default: LLK_ASSERT(false, "Invalid code for unpack instruction"); break;
        }
    };

    load_replay_buf(0, code_len, [&instr_for_code] {
        auto emit = [&](auto self, auto idx) -> void {
            constexpr std::size_t i = decltype(idx)::value;
            if constexpr (i < code_len) {
                instr_for_code(_llk_unpack_AB_face_compressed_mm_code_sequence_[i]);
                self(self, std::integral_constant<std::size_t, i + 1>{});
            }
        };
        emit(emit, std::integral_constant<std::size_t, 0>{});
    });

    constexpr std::uint32_t SB = _llk_unpack_AB_face_compressed_mm_find_("SB");
    constexpr std::uint32_t op0 = _llk_unpack_AB_face_compressed_mm_find_("0");
    constexpr std::uint32_t op1 = _llk_unpack_AB_face_compressed_mm_find_("1");

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,       // unpackB    = true
        false,      // unpackHalo = false
        SB,         // A
        TT_OP_NOP,  // A1 (unused)
        TT_OP_NOP,  // A2 (unused)
        TT_OP_NOP,  // A3 (unused)
        SB,         // skipA
        op0,        // B
        op1         // skipB
    );
    tmp.program();
}

template <bool transpose = false>
inline void _llk_unpack_AB_face_compressed_mm_init_(const std::uint32_t unpB_face_r_dim) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Ovrd_data_format_RMW>(1);  // read dataformat from per cntx registers
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx0_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx0_RMW>(
        static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx1_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx1_RMW>(
        static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx2_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx2_RMW>(
        static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx3_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx3_RMW>(
        static_cast<std::uint32_t>(DataFormat::Bfp4_b));

    // replicate standard settings hw configure sets for cntx0/1 onto cntx2/3 since cntx 2/3 were never used before
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx2_ADDR32, 0, 0xffffffff>((256 << 16) | 256);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx2_address_ADDR32, 0, 0xffffffff>(((4 * 16) << 16) | (4 * 16));

    // override z dim
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xFF0000>(1);

    _llk_unpack_AB_face_compressed_mm_mop_config_();

    // counter override overrides CH0 XY and CH1 X counters, CH0 ZW counters are still shared
    TTI_SETADCXY(0b011, 0b10'000, 0, 0, 0, 0b1111);  // reset overridden counters for bfp2
    TTI_SETADCXY(0b011, 0b11'000, 0, 0, 0, 0b1111);  // reset overridden counters for bfp4
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1111);         // reset CH0 XY counters for unp1 (and unp0 while at it)
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);         // reset shared counters for both unpackers

    constexpr std::uint32_t unpA_x_end = FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end = 4 * unpB_face_r_dim * FACE_C_DIM - 1;

    // set CH1 X counters for both unpackers, bfp4 and bfp2 use different counter overrides so we need to set both
    TTI_SETADC(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, (0b10 << 16) | unpA_x_end);
    TTI_SETADC(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, (0b11 << 16) | unpA_x_end);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);
}

inline void _llk_unpack_AB_face_compressed_mm_uninit_(const std::uint32_t unpA_num_faces) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Ovrd_data_format_RMW>(0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xFF0000>(unpA_num_faces);

    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1111);  // reset CH0 XY counters for both unpackers to leave them in a clean state
}

template <std::uint32_t ct_dim = 1, bool clear_src = true, bool finalize = true>
inline void _llk_unpack_AB_face_compressed_mm_(
    const std::uint32_t base_address_b, const std::uint32_t base_address_meta, const std::uint32_t kt_dim) {
    volatile std::uint32_t* cfg = get_cfg_pointer();

    const std::uint32_t math_meta_size =
        (kt_dim * ct_dim + 4) / 5;  // per tile math meta is 6 bits, 5 entries fit into 32bits, round up
    const std::uint32_t* pre_meta_ptr = reinterpret_cast<std::uint32_t*>(base_address_meta) + math_meta_size;
    const std::uint32_t iters = pre_meta_ptr[0];
    const std::uint32_t full_iters = iters / 6;
    const std::uint32_t rem_iters = iters % 6;
    const std::uint32_t full_blocks = full_iters / 16;
    const bool odd_block = (full_iters % 16) >= 8;
    const std::uint32_t* meta_ptr = pre_meta_ptr + 1 + 2 * (full_iters / 8) + 2;

    wait_for_next_context(1);
    reset_config_context();

    if constexpr (clear_src) {
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    }
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = pre_meta_ptr[1] & 0x00FFFFFF;
    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = pre_meta_ptr[2] & 0x00FFFFFF;
    TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b10 << 16) | (pre_meta_ptr[1] >> 24));
    TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b11 << 16) | (pre_meta_ptr[2] >> 24));
    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = base_address_b;

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    std::uint32_t c = 0;

    auto emit_word = [](std::uint32_t meta) {
        std::uint32_t idx0 = (meta >> 0) & 0b111111;
        std::uint32_t idx1 = (meta >> 5) & 0b111111;
        std::uint32_t idx2 = (meta >> 10) & 0b111111;
        std::uint32_t idx3 = (meta >> 15) & 0b111111;
        std::uint32_t idx4 = (meta >> 20) & 0b111111;
        std::uint32_t idx5 = (meta >> 25) & 0b111111;

        std::uint32_t data0 = _llk_unpack_AB_face_compressed_mm_header_table_[idx0];
        std::uint32_t data1 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx0];
        std::uint32_t data2 = _llk_unpack_AB_face_compressed_mm_header_table_[idx1];
        std::uint32_t data3 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx1];
        std::uint32_t data4 = _llk_unpack_AB_face_compressed_mm_header_table_[idx2];
        std::uint32_t data5 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx2];
        std::uint32_t data6 = _llk_unpack_AB_face_compressed_mm_header_table_[idx3];
        std::uint32_t data7 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx3];
        std::uint32_t data8 = _llk_unpack_AB_face_compressed_mm_header_table_[idx4];
        std::uint32_t data9 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx4];
        std::uint32_t data10 = _llk_unpack_AB_face_compressed_mm_header_table_[idx5];
        std::uint32_t data11 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx5];

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
        ckernel::instrn_buffer[0] = data10;
        ckernel::instrn_buffer[0] = data11;
    };

    for (std::uint32_t b = 0; b < full_blocks; ++b) {
        for (std::uint32_t i = 0; i < 4; ++i, ++c) {
            emit_word(meta_ptr[c]);
        }
        cfg[THCON_SEC0_REG3_Base_cntx2_address_ADDR32] = pre_meta_ptr[3 + 4 * b] & 0x00FFFFFF;
        cfg[THCON_SEC0_REG3_Base_cntx3_address_ADDR32] = pre_meta_ptr[4 + 4 * b] & 0x00FFFFFF;
        for (std::uint32_t i = 0; i < 4; ++i, ++c) {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0002);
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b10 << 16) | (pre_meta_ptr[3 + 4 * b] >> 24));
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b11 << 16) | (pre_meta_ptr[4 + 4 * b] >> 24));
        for (std::uint32_t i = 0; i < 4; ++i, ++c) {
            emit_word(meta_ptr[c]);
        }
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = pre_meta_ptr[5 + 4 * b] & 0x00FFFFFF;
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = pre_meta_ptr[6 + 4 * b] & 0x00FFFFFF;
        for (std::uint32_t i = 0; i < 4; ++i, ++c) {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b10 << 16) | (pre_meta_ptr[5 + 4 * b] >> 24));
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b11 << 16) | (pre_meta_ptr[6 + 4 * b] >> 24));
    }

    if (odd_block) {
        for (std::uint32_t i = 0; i < 4; ++i, ++c) {
            emit_word(meta_ptr[c]);
        }
        cfg[THCON_SEC0_REG3_Base_cntx2_address_ADDR32] = pre_meta_ptr[3 + 4 * full_blocks] & 0x00FFFFFF;
        cfg[THCON_SEC0_REG3_Base_cntx3_address_ADDR32] = pre_meta_ptr[4 + 4 * full_blocks] & 0x00FFFFFF;
        for (std::uint32_t i = 0; i < 4; ++i, ++c) {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0002);
        TT_SETADC(
            p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b10 << 16) | (pre_meta_ptr[3 + 4 * full_blocks] >> 24));
        TT_SETADC(
            p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, (0b11 << 16) | (pre_meta_ptr[4 + 4 * full_blocks] >> 24));
    }

    for (; c < full_iters; ++c) {
        emit_word(meta_ptr[c]);
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t j = 0; j < rem_iters; ++j) {
        std::uint32_t idx0 = meta & 0b111111;
        std::uint32_t data0 = _llk_unpack_AB_face_compressed_mm_header_table_[idx0];
        std::uint32_t data1 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx0];
        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
        meta >>= 5;
    }

    if constexpr (ct_dim == 1 && finalize) {
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    wait_for_next_context(1);
    reset_config_context();

    TTI_SETADCXY(0b011, 0b10'000, 0, 0, 0, 0b1010);
    TTI_SETADCXY(0b011, 0b11'000, 0, 0, 0, 0b1010);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
}
