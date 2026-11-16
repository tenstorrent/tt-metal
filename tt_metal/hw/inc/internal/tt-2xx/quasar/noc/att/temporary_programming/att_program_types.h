// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-2xx/quasar/noc/att/att.h"
#include "internal/tt-2xx/quasar/noc/registers/noc_address_translation_table_a_reg.h"

/**
 * @file
 * @brief Data shapes and register-address arithmetic for temporary bring-up
 * ATT programming. Everything here is constexpr-evaluable and host-safe so the
 * image test can decode a register image without device code; the replay
 * functions that touch hardware live in att_program.h.
 */

namespace noc_att {

/// @brief One raw register write of a generated image.
struct RegisterWrite {
    std::uint32_t address;
    std::uint32_t data;
};

/**
 * @brief A raw generated register image: ordered writes replayed verbatim,
 * plus the addresses the replay must treat specially (the enable register is
 * written last, and the per-initiator self endpoint is patched to this NIU's
 * own coordinate).
 */
struct ProgramImage {
    std::uint32_t enable_tables_address;
    std::uint32_t debug_misc_address;
    std::uint32_t local_endpoint_address;
    const RegisterWrite* writes;
    std::uint32_t write_count;
    const RegisterWrite* overrides;
    std::uint32_t override_count;
};

/**
 * @brief Compact form for simple emulation topologies: the same ATT
 * programming operation for every configuration - only mask slots/windows and
 * endpoint data vary.
 */
struct MaskEntry {
    std::uint8_t slot;
    Window window;
    std::uint64_t bar;
};

struct EndpointEntry {
    std::uint16_t index;
    std::uint8_t x;
    std::uint8_t y;
};

struct Program {
    const MaskEntry* masks;
    std::uint32_t mask_count;
    const EndpointEntry* endpoints;
    std::uint32_t endpoint_count;
    std::uint16_t local_endpoint_index;
};

constexpr std::uint32_t MASK_TABLE_ENTRY_COUNT = 16;
constexpr std::uint32_t MASK_TABLE_STRIDE = NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
                                            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET +
                                            sizeof(std::uint32_t);

/// @brief Register address of one field of a mask-table slot.
constexpr std::uint32_t mask_register_address(std::uint32_t slot, std::uint32_t register_offset) {
    return NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR + register_offset + slot * MASK_TABLE_STRIDE;
}

/// @brief Register address of one endpoint-table row.
constexpr std::uint32_t endpoint_register_address(std::uint32_t endpoint_index) {
    return NOC_ADDRESS_TRANSLATION_TABLE_A_ENDPOINT_TABLE_ENTRY_0__REG_ADDR + endpoint_index * sizeof(std::uint32_t);
}

/// @brief The mask-table control word for a window, in the hardware's layout
/// (mask:6, ep_id_idx:6, ep_id_size:6, table_offset:10, translate_addr:1).
constexpr std::uint32_t mask_entry_control_word(const Window& window) {
    return static_cast<std::uint32_t>(window.mask_bits) | (static_cast<std::uint32_t>(window.endpoint_shift) << 6) |
           (static_cast<std::uint32_t>(window.endpoint_size) << 12) |
           (static_cast<std::uint32_t>(window.endpoint_table_offset) << 18) |
           (static_cast<std::uint32_t>(window.translate_address) << 28);
}

}  // namespace noc_att
