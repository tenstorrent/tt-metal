// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-2xx/quasar/noc/att/temporary_programming/att_program_types.h"
#include "noc_parameters.h"

/**
 * @file
 * @brief Temporary bring-up ATT programming replay. Production boot
 * firmware/UMD programs and enables ATT before any DM core starts; keeping the
 * algorithm data-driven here prevents a test image from becoming part of the
 * kernel-facing address ABI.
 *
 * The replay runs from device firmware during noc_init, gated by the
 * ATT_PROGRAM_FOR_TEST define; nothing sets that define yet - the ATT
 * enablement wires it to TT_METAL_ATT_PROGRAM_FOR_TEST.
 */

#if defined(ATT_PROGRAM_FOR_TEST) && !defined(FW_BUILD)
#error "ATT_PROGRAM_FOR_TEST is a firmware-only bring-up hook; kernels must never program the ATT"
#endif

namespace noc_att {

inline std::uint32_t read_register(std::uint32_t address) {
    return *reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(address));
}

inline void write_register(std::uint32_t address, std::uint32_t data) {
    *reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(address)) = data;
}

/// @brief Program one mask-table slot from a window + BAR.
inline void program_mask_entry(const MaskEntry& entry) {
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET),
        mask_entry_control_word(entry.window));
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET),
        static_cast<std::uint32_t>(entry.window.compare));
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET),
        static_cast<std::uint32_t>(entry.window.compare >> 32));
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_LO_REG_OFFSET),
        static_cast<std::uint32_t>(entry.bar));
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET),
        static_cast<std::uint32_t>(entry.bar >> 32));
}

/// @brief Park every mask-table slot on an unmatchable compare value.
inline void disable_all_mask_entries() {
    for (std::uint32_t slot = 0; slot < MASK_TABLE_ENTRY_COUNT; ++slot) {
        // Raw mask 0 compares all 64 bits. UINT64_MAX is outside every valid
        // device address configuration, so this entry cannot overlap an active one.
        write_register(mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET), 0);
        write_register(
            mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET), UINT32_MAX);
        write_register(
            mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET), UINT32_MAX);
        write_register(mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_LO_REG_OFFSET), 0);
        write_register(mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET), 0);
    }
}

/// @brief Replay a raw generated register image: translation disabled until
/// the complete image is present, the per-initiator self endpoint patched to
/// this NIU's own coordinate, enable written last behind fences.
inline void program_for_test(const ProgramImage& image) {
    // A running firmware must disable translation until the complete final
    // image is present. Boot-owned production programming happens before DM
    // startup and does not use this function.
    write_register(image.enable_tables_address, 0u);

    for (std::uint32_t i = 0; i < image.write_count; ++i) {
        const auto& write = image.writes[i];
        if (write.address != image.enable_tables_address) {
            write_register(write.address, write.data);
        }
    }
    for (std::uint32_t i = 0; i < image.override_count; ++i) {
        const auto& write = image.overrides[i];
        write_register(write.address, write.data);
    }

    // A nonzero local_endpoint_address identifies the per-initiator endpoint that
    // must be patched to the NIU's own package-NoC coordinate.
    if (image.local_endpoint_address != 0) {
        const std::uint32_t node_id = read_register(NOC_NODE_ID);
        const std::uint32_t local_x = node_id & NOC_NODE_ID_MASK;
        const std::uint32_t local_y = (node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        write_register(image.local_endpoint_address, NOC_XY_COORD(local_x, local_y));
    }

    asm volatile("fence iorw, iorw" ::: "memory");
    write_register(image.enable_tables_address, 1u);
    asm volatile("fence iorw, iorw" ::: "memory");
}

/// @brief Program the tables from a compact Program description (simple
/// emulation topologies); same disable/patch/enable contract as the raw form.
inline void program_for_test(const Program& program) {
    write_register(NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_ADDR, 0);
    disable_all_mask_entries();

    for (std::uint32_t i = 0; i < program.mask_count; ++i) {
        program_mask_entry(program.masks[i]);
    }
    for (std::uint32_t i = 0; i < program.endpoint_count; ++i) {
        const auto& endpoint = program.endpoints[i];
        write_register(endpoint_register_address(endpoint.index), NOC_XY_COORD(endpoint.x, endpoint.y));
    }

    const std::uint32_t node_id = read_register(NOC_NODE_ID);
    const std::uint32_t local_x = node_id & NOC_NODE_ID_MASK;
    const std::uint32_t local_y = (node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    write_register(endpoint_register_address(program.local_endpoint_index), NOC_XY_COORD(local_x, local_y));

    asm volatile("fence iorw, iorw" ::: "memory");
    write_register(NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_ADDR, 1);
    asm volatile("fence iorw, iorw" ::: "memory");
}

/// @brief True while translation is enabled and the ATT has recorded no match
/// faults. Read-only: no fault-clear register exists, so a fault latches until
/// reset. Pass a ProgramImage's enable/debug addresses when they differ from
/// this NIU's defaults.
inline bool check_no_faults(
    std::uint32_t enable_tables_address = NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_ADDR,
    std::uint32_t debug_misc_address = NOC_ADDRESS_TRANSLATION_TABLE_A_DEBUG_MISC_REG_ADDR) {
    if ((read_register(enable_tables_address) & 1u) == 0) {
        return false;
    }
    NOC_ADDRESS_TRANSLATION_TABLE_DEBUG_MISC_reg_u debug_misc;
    debug_misc.val = read_register(debug_misc_address);
    return debug_misc.f.no_match == 0 && debug_misc.f.more_than_one_match == 0;
}

}  // namespace noc_att
