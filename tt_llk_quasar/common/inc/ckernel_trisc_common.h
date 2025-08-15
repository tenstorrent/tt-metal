// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cfg_defines.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_proj_params.h"
#include "ckernel_template.h"
#include "tensix_types.h"

namespace ckernel::trisc
{
// Num of words in buffer descriptor struct
constexpr static uint32_t BD_NUM_WORDS = 3;

// Default face dimensions
constexpr static uint32_t FACE_R_DIM = 16;
constexpr static uint32_t FACE_C_DIM = 16;

// Default tile dimensions
constexpr static uint32_t TILE_R_DIM = 32;
constexpr static uint32_t TILE_C_DIM = 32;

// Default number of faces
constexpr static uint32_t NUM_FACES = 4;

// Number of rows that can fit into Dest
static constexpr std::uint32_t DEST_REGISTER_FULL_SIZE = 64 * FACE_R_DIM;
static constexpr std::uint32_t DEST_REGISTER_HALF_SIZE = DEST_REGISTER_FULL_SIZE >> 1;

// Points to the config space
uint32_t volatile* const cfg = (uint32_t volatile*)TENSIX_CFG_BASE;
// Points to the buffer table
buffer_descriptor_u volatile* const bd_table = (buffer_descriptor_u volatile* const)(cfg + BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_ADDR32);

constexpr uint32_t NUM_WORDS_TILE_CNT = 8;

typedef struct
{
    std::uint32_t reserved0                    : 32;
    std::uint32_t reset                        : 32;
    std::uint32_t posted                       : 32;
    std::uint32_t acked                        : 32;
    std::uint32_t buf_capacity                 : 32;
    std::uint32_t reserved1                    : 32;
    std::uint32_t tiles_avail_interrupt_thresh : 32;
    std::uint32_t space_avail_interrupt_thresh : 32;
} tile_counter_t;

static_assert(sizeof(tile_counter_t) == NUM_WORDS_TILE_CNT * sizeof(uint32_t), "tile_counter_t must be 96b!");

typedef union
{
    uint32_t words[NUM_WORDS_TILE_CNT];
    tile_counter_t f;
} tile_counter_u;

// Points to the tile counters
tile_counter_u volatile* const tile_counters = (tile_counter_u volatile* const)TILE_COUNTERS_BASE;

// Destination register bank id, id = 0 -> dest rows 0 to 511, id = 1 -> dest rows 512 - 1023
static uint32_t dest_bank_id = 0;

/**
* @brief Check divisibility by power of 2
* @param value: input value to check divisibility
* @param power_of_two_divisor: divisor that must be a power of 2, will check if
    value % power_of_two_divisor == 0
*/
inline bool _divisible_by_pow_two_(const uint32_t value, const uint32_t power_of_two_divisor)
{
    return ((value & (power_of_two_divisor - 1)) == 0);
}

/**
 * @brief Populates buffer table entry for TDMA engines
 * @param buf_desc_id: Buffer descriptor id into the buffer descriptor table
 * @param buf_desc: Contains L1 buffer descriptor information
 */
inline void _configure_buf_desc_table_(const uint buf_desc_id, const buffer_descriptor_u& buf_desc)
{
    for (uint i = 0; i < BD_NUM_WORDS; i++)
    {
        bd_table[buf_desc_id].words[i] = buf_desc.words[i];
    }
}

/**
 * @brief Zero source registers A&B, usually done at unpack start
 */
inline void _zerosrc_()
{
    TTI_ZEROSRC(0, 0, 0, 0, 0, p_zerosrc::ALL_BANKS, p_zerosrc::CLR_AB);
}

enum class DstTileShape : uint8_t
{
    Tile32x1  = 1,
    Tile32x2  = 2,
    Tile32x4  = 3,
    Tile32x8  = 4,
    Tile32x16 = 5,
    Tile32x32 = 6
};

/**
 * @brief Sets the destination register base address, each Trisc0/1/2/3 has separate
 * registers for setting dest base address.
 * @tparam TRISC_ID: Trisc core which is executing this function, values = [0, 1, 2, 3]
 * @param base_addr: Base address for destination register
 * Bank 0 -> base_addr = 0 - 511
 * Bank 1 -> base_addr = 512 - 1023
 */
template <uint8_t TRISC_ID>
inline void _set_dest_section_base_(const uint32_t base_addr)
{
    if constexpr (TRISC_ID == 0)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32] = base_addr;
    }
    else if constexpr (TRISC_ID == 1)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC1_Offset_ADDR32] = base_addr;
    }
    else if constexpr (TRISC_ID == 2)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC2_Offset_ADDR32] = base_addr;
    }
    else
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC3_Offset_ADDR32] = base_addr;
    }
}

/**
 * @brief Returns dest buffer base addr according to dest_bank_id
 * Bank 0 -> addr = 0
 * Bank 1 -> addr = 512
 */
inline uint32_t _get_dest_buffer_base_()
{
    return (dest_bank_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
}

constexpr static std::uint32_t SCALE_DATUM_SIZE(uint format, uint datum_count)
{
    switch (format & 0xF)
    {
        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::Float32):
            return (datum_count << 2);
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b):
            return (datum_count << 1);
        default:
            return datum_count;
    };
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 *
 * Another Issue:
 * Some of the following enums/functions are needed to increment to arbitrary addresses in dest, the functions can be removed
 * if a Tensix instruction is added to address the full dest
 *
 * The following functions should be removed once the dvalid scheme is completely used for all LLK operations
 */

/**
 * @brief Set destination register bank id variable to 0
 */
inline void _reset_dest_bank_id_()
{
    dest_bank_id = 0;
}

/**
 * @brief Update destination register bank id, bank id can only toggle between 0 & 1;
 */
inline void _update_dest_bank_id_()
{
    dest_bank_id = 1 - dest_bank_id;
}

// Semaphores mapping and trisc space -> tensix space conversion
struct semaphore
{
    constexpr static uint32_t MATH_PACK = 1; // math <-> pack sync on dest register

    constexpr static uint16_t t6_sem(const uint8_t sem_index)
    {
        return (1 << sem_index);
    }
};

// Tensix thread semaphore post optionally stalled
// Can stall on up to 3 resources at a time
template <uint WaitRes0 = p_stall::NOTHING, uint WaitRes1 = p_stall::NOTHING, uint WaitRes2 = p_stall::NOTHING>
inline void t6_semaphore_post(const uint8_t index)
{
    if constexpr (WaitRes0 != p_stall::NOTHING)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes2, WaitRes1, WaitRes0);
    }

    TTI_SEMPOST(0, semaphore::t6_sem(index));
}

// Tensix thread semaphore get optionally stalled
// Can stall on up to 3 resources at a time
template <uint WaitRes0 = p_stall::NOTHING, uint WaitRes1 = p_stall::NOTHING, uint WaitRes2 = p_stall::NOTHING>
inline void t6_semaphore_get(const uint8_t index)
{
    if constexpr (WaitRes0 != p_stall::NOTHING)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes2, WaitRes1, WaitRes0);
    }

    TTI_SEMGET(0, semaphore::t6_sem(index));
}

/**
 * @brief Flip packer dest register offset to 0 or DEST_REGISTER_HALF_SIZE, flip-flopping between two halves
 */
template <uint32_t PACK_SEL, DstSync DST>
inline void _set_packer_dest_registers_()
{
    static_assert(DST == DstSync::SyncHalf || DST == DstSync::SyncFull);
    uint32_t dest_buffer_base_offset = (DST == DstSync::SyncFull) ? 0 : _get_dest_buffer_base_();

    if constexpr (PACK_SEL == p_pacr::PACK0)
    {
        cfg[THCON_PACKER0_REG0_SRC_ADDR_OFFSET_ADDR32] = dest_buffer_base_offset;
    }
    else
    {
        cfg[THCON_PACKER1_REG0_SRC_ADDR_OFFSET_ADDR32] = dest_buffer_base_offset;
    }
}

} // namespace ckernel::trisc
