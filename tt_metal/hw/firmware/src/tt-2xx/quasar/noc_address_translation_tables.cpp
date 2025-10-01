// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_address_translation_tables.hpp"
#include "noc_address_translation_table_a_reg.h"

#ifdef TB_NOC

#include "noc_api_dpi.h"

#else

#define NOC_WRITE_REG(addr, val) ((*((volatile uint32_t*)(addr))) = (val))
#define NOC_READ_REG(addr) (*((volatile uint32_t*)(addr)))

#endif

void noc_address_translation_table_en(bool en_address_translation, bool en_dynamic_routing) {
    NOC_ADDRESS_TRANSLATION_TABLE_ENABLE_TABLES_reg_u ctrl;
    ctrl.f.en_address_translation = en_address_translation;
    ctrl.f.en_dynamic_routing = en_dynamic_routing;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR + NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_OFFSET,
        ctrl.val);
}

void noc_address_translation_table_en_read(bool& en_address_translation, bool& en_dynamic_routing) {
    NOC_ADDRESS_TRANSLATION_TABLE_ENABLE_TABLES_reg_u ctrl;
    ctrl.val = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR + NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_OFFSET);
    en_address_translation = ctrl.f.en_address_translation;
    en_dynamic_routing = ctrl.f.en_dynamic_routing;
}

void noc_address_translation_table_clk_gating(bool clk_gating_enable, uint32_t clk_gating_hysteresis) {
    NOC_ADDRESS_TRANSLATION_TABLE_CLK_GATING_reg_u ctrl;
    ctrl.f.clk_gating_enable = clk_gating_enable;
    ctrl.f.clk_gating_hysteresis = clk_gating_hysteresis;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR + NOC_ADDRESS_TRANSLATION_TABLE_A_CLK_GATING_REG_OFFSET,
        ctrl.val);
}

void noc_address_translation_table_clk_gating_read(bool& clk_gating_enable, uint32_t& clk_gating_hysteresis) {
    NOC_ADDRESS_TRANSLATION_TABLE_CLK_GATING_reg_u ctrl;
    ctrl.val = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR + NOC_ADDRESS_TRANSLATION_TABLE_A_CLK_GATING_REG_OFFSET);
    clk_gating_enable = ctrl.f.clk_gating_enable;
    clk_gating_hysteresis = ctrl.f.clk_gating_hysteresis;
}

void noc_address_translation_table_mask_table_entry(
    uint32_t entry, uint32_t mask, uint64_t compare, uint32_t ep_id_idx, uint32_t ep_id_size, uint32_t table_offset) {
    mask = 64 - mask;

    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u ctrl;
    ctrl.f.mask = mask;
    ctrl.f.ep_id_idx = ep_id_idx;
    ctrl.f.ep_id_size = ep_id_size;
    ctrl.f.table_offset = table_offset;
    ctrl.f.translate_addr = 0;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        ctrl.val);

    uint32_t val = compare & 0xFFFFFFFF;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        val);

    val = (compare >> 32) & 0xFFFFFFFF;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        val);
}

void noc_address_translation_table_mask_table_entry_read(
    uint32_t entry,
    uint32_t& mask,
    uint64_t& compare,
    uint32_t& ep_id_idx,
    uint32_t& ep_id_size,
    uint32_t& table_offset) {
    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u ctrl;
    ctrl.val = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);
    mask = 64 - ctrl.f.mask;
    ep_id_idx = ctrl.f.ep_id_idx;
    ep_id_size = ctrl.f.ep_id_size;
    table_offset = ctrl.f.table_offset;

    uint32_t val_low = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);

    uint32_t val_high = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);

    compare = ((uint64_t)val_high << 32) | val_low;
}

void noc_address_translation_table_mask_table_entry_rebase(
    uint32_t entry,
    uint32_t mask,
    uint64_t compare,
    uint32_t ep_id_idx,
    uint32_t ep_id_size,
    uint64_t base,
    uint32_t table_offset) {
    mask = 64 - mask;

    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u ctrl;
    ctrl.f.mask = mask;
    ctrl.f.ep_id_idx = ep_id_idx;
    ctrl.f.ep_id_size = ep_id_size;
    ctrl.f.table_offset = table_offset;
    ctrl.f.translate_addr = 1;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        ctrl.val);

    uint32_t val = compare & 0xFFFFFFFF;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        val);

    val = (compare >> 32) & 0xFFFFFFFF;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        val);

    val = base & 0xFFFFFFFF;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_LO_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        val);

    val = (base >> 32) & 0xFFFFFFFF;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
                entry,
        val);
}

void noc_address_translation_table_mask_table_entry_rebase_read(
    uint32_t entry,
    uint32_t& mask,
    uint64_t& compare,
    uint32_t& ep_id_idx,
    uint32_t& ep_id_size,
    uint64_t& base,
    uint32_t& table_offset) {
    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u ctrl;
    ctrl.val = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);
    mask = 64 - ctrl.f.mask;
    ep_id_idx = ctrl.f.ep_id_idx;
    ep_id_size = ctrl.f.ep_id_size;
    table_offset = ctrl.f.table_offset;

    uint32_t val_low = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);

    uint32_t val_high = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);

    compare = ((uint64_t)val_high << 32) | val_low;

    val_low = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_LO_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);

    val_high = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET +
        (NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
         NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4) *
            entry);

    base = ((uint64_t)val_high << 32) | val_low;
}

void noc_address_translation_table_routing_table_entry(uint32_t entry, uint32_t compare, uint32_t routing[]) {
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_MATCH_REG_OFFSET + 4 * entry,
        compare);

    for (int k = 0; k < 32; k++) {
        NOC_WRITE_REG(
            NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
                NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_PART_ENTRY_0__REG_OFFSET + k * 4 +
                (NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_PART_ENTRY_31__REG_OFFSET -
                 NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_PART_ENTRY_0__REG_OFFSET + 4) *
                    entry,
            routing[k]);
    }
}

void noc_address_translation_table_routing_table_entry_read(uint32_t entry, uint32_t& compare, uint32_t routing[]) {
    compare = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_MATCH_REG_OFFSET + 4 * entry);

    for (int k = 0; k < 32; k++) {
        routing[k] = NOC_READ_REG(
            NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_PART_ENTRY_0__REG_OFFSET + k * 4 +
            (NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_PART_ENTRY_31__REG_OFFSET -
             NOC_ADDRESS_TRANSLATION_TABLE_A_ROUTING_TABLE_PART_ENTRY_0__REG_OFFSET + 4) *
                entry);
    }
}

void noc_address_translation_table_endpoint_table_entry(uint32_t entry, uint32_t x, uint32_t y) {
    NOC_ADDRESS_TRANSLATION_TABLE_ENDPOINT_TABLE_ENTRY_reg_u ctrl;
    ctrl.f.x = x;
    ctrl.f.y = y;
    NOC_WRITE_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_ENDPOINT_TABLE_ENTRY_0__REG_OFFSET + 4 * entry,
        ctrl.val);
}

void noc_address_translation_table_endpoint_table_entry_read(uint32_t entry, uint32_t& x, uint32_t& y) {
    NOC_ADDRESS_TRANSLATION_TABLE_ENDPOINT_TABLE_ENTRY_reg_u ctrl;
    ctrl.val = NOC_READ_REG(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
        NOC_ADDRESS_TRANSLATION_TABLE_A_ENDPOINT_TABLE_ENTRY_0__REG_OFFSET + 4 * entry);
    x = ctrl.f.x;
    y = ctrl.f.y;
}
