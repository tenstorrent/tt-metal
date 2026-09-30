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
 * Scalar path: the loop runs over the eight 4-row input groups and handles the four rows of a group with
 * compile-time row-in-group constants, so the per-row RISC work is the index byte, the destination group address
 * and thirteen instruction words (four input loads, four output loads, the add, four stores) built with one OR
 * from register-resident constants; the two transposes are immediates. The instruction stream the SFPU executes
 * is the same as before; the RISC work between the instructions is what shrank (about 46 scalar cycles per live
 * row and 27 per skipped row before, measured on Blackhole). The loop body stays small on purpose: a version
 * expanded over all 32 rows was slower, its 8 KB of straight-line code being fetch bound.
 *
 * @param idx_addr: L1 address of the mask tile containing destination row mappings (uint8_t[32])
 */
inline void reshuffle_rows_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

namespace reshuffle_rows_detail {

constexpr std::uint32_t output_tile_offset = 64;

// The 4-row group a row belongs to: rows 0-15 live in faces 0/1 at 4-row steps, rows 16-31 in faces 2/3 at
// +16 (the face pair offset). Bits 0, 1 and 4 of the result are always clear, so the +2, +16 and +18 column and
// face offsets below can be ORed in.
constexpr std::uint32_t row_group_addr(const std::uint32_t row) { return (row & ~0x3u) + (row & 0x10u); }

constexpr std::uint32_t word(const int encoding) { return static_cast<std::uint32_t>(encoding); }

// One input row: the group of the row and its neighbours is loaded into LREG0-3 and the destination group into
// LREG4-7, a transpose isolates the two rows, the add accumulates, a transpose restores the layout and the
// destination group is stored. ROW_IN_GROUP (0-3) selects the input register at compile time; the input group
// address and the destination row come at run time.
template <std::uint32_t ROW_IN_GROUP>
inline __attribute__((always_inline)) void reshuffle_row(const std::uint32_t input_row_addr, const std::uint32_t idx_word) {
    const std::uint32_t dst_row = (idx_word >> (8 * ROW_IN_GROUP)) & 0xFFu;
    // Skip if dst_row is 255, i.e. mask is invalid and we don't want to process the current row
    if (dst_row >= 32) {
        return;
    }
    const std::uint32_t output_row_addr = output_tile_offset + row_group_addr(dst_row);
    constexpr std::uint32_t input_row_lreg = p_sfpu::LREG0 + ROW_IN_GROUP;
    // dst = 1.0 * src + dst, on the output register that holds the destination row after the transpose
    const std::uint32_t output_row_lreg_fields = (dst_row & 0x3u) * ((1u << 8) | (1u << 4));

    // load in the input row and output row
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0)) | input_row_addr);         // Face 0/2, even columns
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2)) | input_row_addr);         // Face 0/2, odd columns
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 16)) | input_row_addr);        // Face 1/3, even columns
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 18)) | input_row_addr);        // Face 1/3, odd columns
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, 0)) | output_row_addr);        // Face 0/2, even columns
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, 2)) | output_row_addr);        // Face 0/2, odd columns
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, 16)) | output_row_addr);       // Face 1/3, even columns
    TT_INSN(word(TT_OP_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_7, 18)) | output_row_addr);       // Face 1/3, odd columns
    // TRANSPOSE #1: Rearrange loaded 4-row blocks to isolate target rows
    // SFPLOAD loads 4 consecutive rows (e.g., rows 4-7) into LREG0-3, but we only want one specific row (e.g., row
    // 5) This transpose shuffles the register contents so row 5 data becomes accessible via input_row_lreg[1]
    TTI_SFPTRANSP(0, 0, 0, 0);  // Puts desired input row into LREG "input_row_lreg" and output row into "output_row_lreg"

    // ACCUMULATION: Perform gradient accumulation for embedding backward pass
    // Implements: output[dst_row] += input[row] (scatter-add operation)
    // Uses LCONST_1 (value 1.0) as multiplier: dst = 1.0 * src + dst
    TT_INSN(word(TT_OP_SFPADD(input_row_lreg, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0)) | output_row_lreg_fields);

    // TRANSPOSE #2: Rearrange accumulated results back to 4-row storage format
    // Prepares the computed result for SFPSTORE, which expects data in LREG4-7 positions
    // This undoes the first transpose to match the expected storage layout
    TTI_SFPTRANSP(0, 0, 0, 0);  // Puts desired output row back into LREG4-7 for storage
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, 0)) | output_row_addr);       // Face 0/2, even columns
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 2)) | output_row_addr);       // Face 0/2, odd columns
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_7, 16)) | output_row_addr);      // Face 1/3, even columns
    TT_INSN(word(TT_OP_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_7, 18)) | output_row_addr);      // Face 1/3, odd columns
}

}  // namespace reshuffle_rows_detail

template <bool APPROXIMATION_MODE>
inline void calculate_reshuffle_rows(uint idx_addr) {
    // clr DEST tile 1
    // TODO (Radomir): Add optional clear that is more optimal using tile copy
    // for (uint row=0; row < 32; row+=4) {
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row + 2);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row + 32);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, output_tile_offset + row + 34);
    // }

    // Skip tile header, hence + 16. The 32 index bytes are read as eight words, one per 4-row group.
    const volatile tt_l1_ptr std::uint32_t* idx_words =
        reinterpret_cast<const volatile tt_l1_ptr std::uint32_t*>(idx_addr + 16);

    // TODO: Add dynamic assert for idx_ptr being within L1 memory bounds
    // using hardware memory map constants: MEM_L1_BASE and MEM_L1_SIZE

#pragma GCC unroll 0
    for (std::uint32_t group = 0; group < 8; group++) {
        const std::uint32_t input_row_addr = reshuffle_rows_detail::row_group_addr(group * 4);
        const std::uint32_t idx_word = idx_words[group];
        reshuffle_rows_detail::reshuffle_row<0>(input_row_addr, idx_word);
        reshuffle_rows_detail::reshuffle_row<1>(input_row_addr, idx_word);
        reshuffle_rows_detail::reshuffle_row<2>(input_row_addr, idx_word);
        reshuffle_rows_detail::reshuffle_row<3>(input_row_addr, idx_word);
    }
}

}  // namespace sfpu
}  // namespace ckernel
