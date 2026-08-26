// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "noc/att/att.h"
#include "noc/noc_parameters.h"
#include "noc/registers/noc_address_translation_table_a_reg.h"

// Temporary bring-up support only. Production boot firmware/UMD programs and
// enables ATT before any DM core starts. Keeping the algorithm data-driven here
// prevents a test image from becoming part of the kernel-facing address ABI.

namespace noc_att {

struct RegisterWrite {
    uint32_t address;
    uint32_t data;
};

struct ProgramImage {
    uint32_t enable_tables_address;
    uint32_t debug_misc_address;
    uint32_t local_endpoint_address;
    const RegisterWrite* writes;
    uint32_t write_count;
    const RegisterWrite* overrides;
    uint32_t override_count;
};

// Compact form for simple emulation topologies. It is the same ATT programming
// operation for every configuration: only mask slots/windows and endpoint data vary.
struct MaskEntry {
    uint8_t slot;
    Window window;
    uint64_t bar;
};

struct EndpointEntry {
    uint16_t index;
    uint8_t x;
    uint8_t y;
};

struct Program {
    const MaskEntry* masks;
    uint32_t mask_count;
    const EndpointEntry* endpoints;
    uint32_t endpoint_count;
    uint16_t local_endpoint_index;
};

inline uint32_t read_register(uint32_t address) {
    return *reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(address));
}

inline void write_register(uint32_t address, uint32_t data) {
    *reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(address)) = data;
}

constexpr uint32_t MASK_TABLE_ENTRY_COUNT = 16;
constexpr uint32_t MASK_TABLE_STRIDE = NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
                                       NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + sizeof(uint32_t);

inline uint32_t mask_register_address(uint32_t slot, uint32_t register_offset) {
    return NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR + register_offset + slot * MASK_TABLE_STRIDE;
}

inline void program_mask_entry(const MaskEntry& entry) {
    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u control{};
    control.f.mask = entry.window.mask_bits;
    control.f.ep_id_idx = entry.window.endpoint_shift;
    control.f.ep_id_size = entry.window.endpoint_size;
    control.f.table_offset = entry.window.endpoint_table_offset;
    control.f.translate_addr = entry.window.translate_address;

    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET), control.val);
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET),
        static_cast<uint32_t>(entry.window.compare));
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET),
        static_cast<uint32_t>(entry.window.compare >> 32));
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_LO_REG_OFFSET),
        static_cast<uint32_t>(entry.bar));
    write_register(
        mask_register_address(entry.slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET),
        static_cast<uint32_t>(entry.bar >> 32));
}

inline void disable_all_mask_entries() {
    for (uint32_t slot = 0; slot < MASK_TABLE_ENTRY_COUNT; ++slot) {
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

inline void program_for_test(const ProgramImage& image) {
    // A running firmware must disable translation until the complete final
    // image is present. Boot-owned production programming happens before DM
    // startup and does not use this function.
    write_register(image.enable_tables_address, 0u);

    for (uint32_t i = 0; i < image.write_count; ++i) {
        const auto& write = image.writes[i];
        if (write.address != image.enable_tables_address) {
            write_register(write.address, write.data);
        }
    }
    for (uint32_t i = 0; i < image.override_count; ++i) {
        const auto& write = image.overrides[i];
        write_register(write.address, write.data);
    }

    // A nonzero local_endpoint_address identifies the per-initiator endpoint that
    // must be patched to the NIU's own package-NoC coordinate.
    if (image.local_endpoint_address != 0) {
        const uint32_t node_id = read_register(NOC_NODE_ID);
        const uint32_t local_x = node_id & NOC_NODE_ID_MASK;
        const uint32_t local_y = (node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        write_register(image.local_endpoint_address, NOC_XY_COORD(local_x, local_y));
    }

    asm volatile("fence iorw, iorw" ::: "memory");
    write_register(image.enable_tables_address, 1u);
    asm volatile("fence iorw, iorw" ::: "memory");
}

inline void program_for_test(const Program& program) {
    write_register(NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_ADDR, 0);
    disable_all_mask_entries();

    for (uint32_t i = 0; i < program.mask_count; ++i) {
        program_mask_entry(program.masks[i]);
    }
    for (uint32_t i = 0; i < program.endpoint_count; ++i) {
        const auto& endpoint = program.endpoints[i];
        write_register(
            NOC_ADDRESS_TRANSLATION_TABLE_A_ENDPOINT_TABLE_ENTRY_0__REG_ADDR + endpoint.index * sizeof(uint32_t),
            NOC_XY_COORD(endpoint.x, endpoint.y));
    }

    const uint32_t node_id = read_register(NOC_NODE_ID);
    const uint32_t local_x = node_id & NOC_NODE_ID_MASK;
    const uint32_t local_y = (node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    write_register(
        NOC_ADDRESS_TRANSLATION_TABLE_A_ENDPOINT_TABLE_ENTRY_0__REG_ADDR +
            program.local_endpoint_index * sizeof(uint32_t),
        NOC_XY_COORD(local_x, local_y));

    asm volatile("fence iorw, iorw" ::: "memory");
    write_register(NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_ADDR, 1);
    asm volatile("fence iorw, iorw" ::: "memory");
}

inline bool check_no_faults(const ProgramImage& image) {
    if ((read_register(image.enable_tables_address) & 1u) == 0) {
        return false;
    }
    NOC_ADDRESS_TRANSLATION_TABLE_DEBUG_MISC_reg_u debug_misc;
    debug_misc.val = read_register(image.debug_misc_address);
    return debug_misc.f.no_match == 0 && debug_misc.f.more_than_one_match == 0;
}

inline bool check_no_faults(const Program&) {
    if ((read_register(NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_ADDR) & 1u) == 0) {
        return false;
    }
    NOC_ADDRESS_TRANSLATION_TABLE_DEBUG_MISC_reg_u debug_misc;
    debug_misc.val = read_register(NOC_ADDRESS_TRANSLATION_TABLE_A_DEBUG_MISC_REG_ADDR);
    return debug_misc.f.no_match == 0 && debug_misc.f.more_than_one_match == 0;
}

}  // namespace noc_att
