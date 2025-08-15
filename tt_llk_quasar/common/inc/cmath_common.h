// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_trisc_common.h"

namespace ckernel::math
{

// Number of rows for MATH functions
constexpr static uint ELTWISE_MATH_ROWS = MATH_ROWS; // 8 for quasar, 4 for trinity
constexpr static uint MOVE_MATH_ROWS[3] = {8, 4, 1};
constexpr static unsigned int SFP_ROWS  = 2;
constexpr static uint TRISC_ID          = 1;

// Struct for the ALU addresses
constexpr uint32_t NUM_WORDS_ALU_FORMAT = 3;

typedef struct
{
    // word 0
    uint32_t ALU_FORMAT_SPEC_REG_SrcA_val        : 8;
    uint32_t ALU_FORMAT_SPEC_REG_SrcA_override   : 1;
    uint32_t ALU_FORMAT_SPEC_REG_SrcB_val        : 8;
    uint32_t ALU_FORMAT_SPEC_REG_SrcB_override   : 1;
    uint32_t ALU_FORMAT_SPEC_REG_Dstacc_val      : 8;
    uint32_t ALU_FORMAT_SPEC_REG_Dstacc_override : 1;
    uint32_t EMPTY0                              : 5;
    // word 1
    uint32_t ALU_ROUNDING_MODE_Fpu_srnd_en : 1;
    uint32_t UNUSED0                       : 2;
    uint32_t ALU_ROUNDING_MODE_Padding     : 10;
    uint32_t ALU_ROUNDING_MODE_GS_LF       : 1;
    uint32_t ALU_ROUNDING_MODE_Bfp8_HF     : 1;
    uint32_t ALU_FORMAT_SPEC_REG0_SrcA     : 8;
    uint32_t ALU_FORMAT_SPEC_REG1_SrcB     : 8;
    uint32_t EMPTY1                        : 1;
    // word 2
    uint32_t ALU_FORMAT_SPEC_REG2_Dstacc    : 8;
    uint32_t ALU_ACC_CTRL_Fp32_enabled      : 1;
    uint32_t ALU_ACC_CTRL_SFPU_Fp32_enabled : 1;
    uint32_t ALU_ACC_CTRL_INT8_math_enabled : 1;
    uint32_t UNUSED1                        : 21;
} alu_config_t;

static_assert(sizeof(alu_config_t) == NUM_WORDS_ALU_FORMAT * sizeof(uint32_t));

typedef union
{
    uint32_t val[NUM_WORDS_ALU_FORMAT];
    alu_config_t f;
} alu_config_u;

// /**
// * @brief Helper function to calculate log2,
// * only works for 32 bit unsigned inputs
// * @param val: Input value to log2 operation
// */
// inline uint32_t trisc_log2(const uint32_t val) {
//     return 31 - __builtin_clz(val);
// }

/**
 * @brief Helper function to calculate log2 for FPU rows
 * since FPU rows are <=16, and are power of 2, can use
 * simplified higher perf method
 * @param val: Input value to log2 operation
 */
inline uint32_t math_rows_log2(const uint32_t math_rows)
{
    switch (math_rows)
    {
        case 16:
            return 4;
        case 8:
            return 3;
        case 4:
            return 2;
        case 2:
            return 1;
        default:
            return 0;
    }
}

/**
 * @brief Increments given counters
 * @tparam: SRCA_INCR: SrcA increment values = 0 - 15
 * @tparam: SRCB_INCR: SrcA increment values = 0 - 15
 * @tparam: SRCD_INCR: SrcA increment values = 0 - 15
 * @tparam: CR_INCR: SrcA increment values = 0 - 63
 */
template <uint SRCA_INCR, uint SRCB_INCR, uint DEST_INCR, uint CR_INCR>
inline void _incr_counters_()
{
    static_assert(SRCA_INCR < 32, "Value exceeds RWC_A width of 5 bits");
    static_assert(SRCB_INCR < 32, "Value exceeds RWC_B width of 5 bits");
    static_assert(DEST_INCR < 256, "Value exceeds RWC_D width of 8 bits");
    static_assert(CR_INCR < 64, "Value exceeds RWC_CR width of 6 bits");
    TTI_INCRWC(CR_INCR, SRCA_INCR, SRCB_INCR, DEST_INCR);
}

// TODO (RT): Is there now an alternative to this?
inline void _sfpu_load_config32_(const uint dest, const uint upper16, const uint lower16)
{
    // registers 11 through 14 are programmable "constants" which are shared across all 4 rows
    // They are updated only through the CONFIG path, which uses LREG[0] first and then copies it to the desired register location
    TTI_SFPLOADI(p_sfpu::LREG0, 10, lower16); // insmod == A will write the lower bits, and not affect the upper bits;
    TTI_SFPLOADI(p_sfpu::LREG0, 8, upper16);  // insmod == 8 will write the upper bits, and not affect the lower bits;
    TTI_SFPCONFIG(0, dest, 0);
}

/**
 * @brief Initializes the programmable registers for the SFPU
 */
inline void _init_sfpu_config_reg_()
{
    _sfpu_load_config32_(0xF, 0x0, 0x0);
}

/**
 * @brief Reset given counters to 0
 * @tparam: SETRWC: which counter to reset, values = p_setrwc::[SET_A, SET_B, SET_D, SET_F]
 */
template <uint SETRWC>
inline void _reset_counters_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, SETRWC);
}

/**
 * @brief Inc dest counter using carriage return (why use the CR?)
 * @tparam NUM_ROWS: number of 16 datum rows to increment dest by, value must be <=255
 */
template <uint NUM_ROWS>
inline void _inc_dst_addr_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, NUM_ROWS, p_setrwc::SET_D);
}

/**
 * @brief Zeroes out all dest banks, should only be done at kernel start
 * WARNING: Uses an addrmod, make sure it does not conflict with other addrmods
 */
inline void _zero_dest_reg_()
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0}}.set(ADDR_MOD_5);

    TTI_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, ADDR_MOD_5, 0);
}

/**
 * @brief Sets destination register base address depending on tile idx
 * @param tile_idx: Tile index in the dest reg
 * 16bit dest reg data format -> tile_idx = 0 - 7
 * 32bit dest reg data format -> tile_idx = 0 - 3
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE>
inline void _set_dst_write_addr_(const uint32_t tile_index)
{
    const uint tile_shape_idx = (TILE_SHAPE == ckernel::trisc::DstTileShape::Tile32x32) ? 6 : ((TILE_SHAPE == ckernel::trisc::DstTileShape::Tile32x16) ? 5 : 4);
    const uint dst_index      = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

template <uint32_t num_rows_per_tile>
inline void _set_dst_write_addr_by_rows_(const uint32_t tile_index)
{
    const uint tile_shape_idx =
        (num_rows_per_tile == 64)
            ? 6
            : ((num_rows_per_tile == 32) ? 5 : ((num_rows_per_tile == 16) ? 4 : ((num_rows_per_tile == 8) ? 3 : ((num_rows_per_tile == 4) ? 2 : 1))));
    const uint dst_index = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

} // namespace ckernel::math
