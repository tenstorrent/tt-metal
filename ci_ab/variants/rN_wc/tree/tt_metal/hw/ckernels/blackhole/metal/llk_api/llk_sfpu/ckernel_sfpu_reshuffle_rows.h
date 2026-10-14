// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Implements gradient accumulation with row reshuffling for embedding backward pass
 *
 * This function performs a scatter-add operation: output[mask[i]] += input[i] for each row i (0-31).
 * It's the core SFPU implementation for embedding gradient accumulation, where gradients from
 * different input positions are accumulated into their corresponding embedding rows based on
 * a mask that specifies the destination mapping.
 *
 * Algorithm:
 * - Input:  Gradient tile (tile 0) + destination row mask (idx_addr)
 * - Output: Accumulated gradients in reshuffled pattern (tile 1, offset 64)
 * - For each input row i: if mask[i] < 32, then output[mask[i]] += input[i]
 * - Mask value 255 indicates "skip this row" (no accumulation)
 *
 * SFPU Implementation Details:
 * - Leverages vector register parallelism for efficient row processing
 * - Uses face-aware addressing to handle tile memory layout (faces 0/1 for rows 0-15, faces 2/3 for rows 16-31)
 * - Employs transpose operations to work around SFPLOAD/SFPSTORE 4-row granularity constraints
 * - Processes both even/odd columns simultaneously using +2 offset addressing
 *
 * @param idx_addr: L1 address of the mask tile with the destination rows (uint8_t[32] at idx_addr + 16, 4-byte aligned)
 */
inline void reshuffle_rows_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

namespace reshuffle_rows_detail {

constexpr std::uint32_t output_tile_offset = 64;
static_assert(output_tile_offset % 32 == 0, "the +2, +16 and +18 offsets are ORed into the output row address");
constexpr std::uint32_t rows_per_group = 4;
constexpr std::uint32_t num_groups = 32 / rows_per_group;

// Bits 0, 1 and 4 of a group address are clear, so the +2, +16 and +18 column and face offsets can be ORed in.
constexpr std::uint32_t row_group_addr(const std::uint32_t row) { return (row & ~0x3u) + (row & 0x10u); }

constexpr std::uint32_t word(const int encoding) { return static_cast<std::uint32_t>(encoding); }

template <std::uint32_t ROW_IN_GROUP>
inline __attribute__((always_inline)) void reshuffle_row(const std::uint32_t input_row_addr, const std::uint32_t idx_word) {
    const std::uint32_t dst_row = (idx_word >> (8 * ROW_IN_GROUP)) & 0xFFu;
    if (dst_row >= 32) {
        return;
    }
    const std::uint32_t output_row_addr = output_tile_offset + row_group_addr(dst_row);
    constexpr std::uint32_t input_row_lreg = p_sfpu::LREG0 + ROW_IN_GROUP;
    // The SFPADD's VC and VD fields: the register holding the destination row after the transpose
    const std::uint32_t output_row_lreg_fields = (dst_row & 0x3u) * ((1u << 8) | (1u << 4));

    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0)) | input_row_addr);
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2)) | input_row_addr);
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 16)) | input_row_addr);
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 18)) | input_row_addr);
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, 0)) | output_row_addr);
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, 2)) | output_row_addr);
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, 16)) | output_row_addr);
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_7, 18)) | output_row_addr);
    // SFPLOAD brings in 4 consecutive rows; the transpose makes the wanted input and output rows addressable.
    TTI_SFPTRANSP(0, 0, 0, 0);

    TT_INSN(word(TT_OP_SFPADD(input_row_lreg, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0)) | output_row_lreg_fields);

    TTI_SFPTRANSP(0, 0, 0, 0);
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, 0)) | output_row_addr);
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 2)) | output_row_addr);
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_7, 16)) | output_row_addr);
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_7, 18)) | output_row_addr);
}


// Replay slots: the input group's loads, the output group's loads and transpose, and the transpose and stores back.
constexpr std::uint32_t replay_input = 0;
constexpr std::uint32_t replay_output_load = 4;
constexpr std::uint32_t replay_output_store = 9;
constexpr std::uint32_t replay_len = 14;

// The replayed loads and stores carry only the column and face offsets; the group is addressed through the math Dest
// offset, written once the SFPU has finished the loads and stores that use the previous one.
inline void set_group_offset(const std::uint32_t offset) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::SFPU1);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, offset);
}

inline void record_group_programs() {
    load_replay_buf(replay_input, replay_len, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 16);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 18);
        TTI_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, 2);
        TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, 16);
        TTI_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_7, 18);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 2);
        TTI_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_7, 16);
        TTI_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_7, 18);
    });
}

// The input group stays in LREG0-3 across its rows: the two transposes around each row's add cancel.
template <std::uint32_t ROW_IN_GROUP>
inline __attribute__((always_inline)) void reshuffle_row_replayed(const std::uint32_t tile_base, const std::uint32_t idx_word) {
    const std::uint32_t dst_row = (idx_word >> (8 * ROW_IN_GROUP)) & 0xFFu;
    if (dst_row >= 32) {
        return;
    }
    constexpr std::uint32_t input_row_lreg = p_sfpu::LREG0 + ROW_IN_GROUP;
    const std::uint32_t output_row_lreg_fields = (dst_row & 0x3u) * ((1u << 8) | (1u << 4));

    set_group_offset(tile_base + output_tile_offset + row_group_addr(dst_row));
    lltt::replay(replay_output_load, 5);
    TT_INSN(word(TT_OP_SFPADD(input_row_lreg, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0)) | output_row_lreg_fields);
    lltt::replay(replay_output_store, 5);
}

// True when any of the word's four index bytes is below 32: bits 5 to 7 of that byte are clear, and adding 0x7F to
// the three-bit lanes cannot carry into the next byte.
constexpr bool group_has_live_row(const std::uint32_t idx_word) {
    const std::uint32_t high = (idx_word >> 5) & 0x07070707u;
    return (~(high + 0x7F7F7F7Fu) & 0x80808080u) != 0;
}

}  // namespace reshuffle_rows_detail

template <bool APPROXIMATION_MODE>
inline void calculate_reshuffle_rows(uint idx_addr) {
    // clr DEST tile 1
    // TODO (Radomir): Add optional clear that is more optimal using tile copy

    // Skip tile header, hence + 16:
    const volatile tt_l1_ptr std::uint32_t* idx_words =
        reinterpret_cast<const volatile tt_l1_ptr std::uint32_t*>(idx_addr + 16);

    // TODO: Add dynamic assert for idx_words being within L1 memory bounds
    // using hardware memory map constants: MEM_L1_BASE and MEM_L1_SIZE

#pragma GCC unroll 0
    for (std::uint32_t group = 0; group < reshuffle_rows_detail::num_groups; group++) {
        const std::uint32_t input_row_addr =
            reshuffle_rows_detail::row_group_addr(group * reshuffle_rows_detail::rows_per_group);
        const std::uint32_t idx_word = idx_words[group];
        reshuffle_rows_detail::reshuffle_row<0>(input_row_addr, idx_word);
        reshuffle_rows_detail::reshuffle_row<1>(input_row_addr, idx_word);
        reshuffle_rows_detail::reshuffle_row<2>(input_row_addr, idx_word);
        reshuffle_rows_detail::reshuffle_row<3>(input_row_addr, idx_word);
    }
}

/**
 * The same scatter-add with the row loads and stores replayed at a per-row Dest offset: about half the cycles of
 * calculate_reshuffle_rows when a call has many live rows, but its replay record and offset writes run after the
 * tile's data lands, so a call with one or two live rows that waits on its inputs takes longer.
 *
 * @param dst_index: the Dest tile the call runs on, whose offset is restored at the end
 */
template <bool APPROXIMATION_MODE>
inline void calculate_reshuffle_rows_replayed(uint idx_addr, uint dst_index) {
    const volatile tt_l1_ptr std::uint32_t* idx_words =
        reinterpret_cast<const volatile tt_l1_ptr std::uint32_t*>(idx_addr + 16);

    // The Dest offset _llk_math_eltwise_sfpu_start_ programmed for this tile
    const std::uint32_t tile_base = (dst_index << 6) + get_dest_buffer_base();
    reshuffle_rows_detail::record_group_programs();

#pragma GCC unroll 0
    for (std::uint32_t group = 0; group < reshuffle_rows_detail::num_groups; group++) {
        const std::uint32_t idx_word = idx_words[group];
        if (!reshuffle_rows_detail::group_has_live_row(idx_word)) {
            continue;
        }
        reshuffle_rows_detail::set_group_offset(
            tile_base + reshuffle_rows_detail::row_group_addr(group * reshuffle_rows_detail::rows_per_group));
        lltt::replay(reshuffle_rows_detail::replay_input, 4);
        reshuffle_rows_detail::reshuffle_row_replayed<0>(tile_base, idx_word);
        reshuffle_rows_detail::reshuffle_row_replayed<1>(tile_base, idx_word);
        reshuffle_rows_detail::reshuffle_row_replayed<2>(tile_base, idx_word);
        reshuffle_rows_detail::reshuffle_row_replayed<3>(tile_base, idx_word);
    }
    reshuffle_rows_detail::set_group_offset(tile_base);
}

}  // namespace sfpu
}  // namespace ckernel
