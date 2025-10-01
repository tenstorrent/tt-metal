// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NOC_ADDRESS_TRANSLATION_TABLES_HPP
#define NOC_ADDRESS_TRANSLATION_TABLES_HPP

#include <cstdint>
#ifdef TB_NOC
extern "C" {
#include "noc.h"
#include "noc_api_dpi.h"
}
#endif

// The address translation is initially disabled and will simply propagate the coordinates from the initiator
// en_address_translation - enable tables
// en_dynamic_routing - enable dynamic routing, if this is disabled all routing will be dimension order
void noc_address_translation_table_en(bool en_address_translation, bool en_dynamic_routing);

// Read back the address translation table enable register
void noc_address_translation_table_en_read(bool& en_address_translation, bool& en_dynamic_routing);

// clock gating control, by default it's disabled
void noc_address_translation_table_clk_gating(bool clk_gating_enable, uint32_t clk_gating_hysteresis);

// Read back clock gating register
void noc_address_translation_table_clk_gating_read(bool& clk_gating_enable, uint32_t& clk_gating_hysteresis);

// Write an entry to the mask table
// entry - a value between 0 to 15
// mask - The number of MSB bits that will be compared with the address (e.g. 2 will create a mask
// 0xC000_0000_0000_0000) compare - The value that the masked address should be compared to ep_id_idx - The bit offset
// into the address that contains the endpoint id ep_id_size - The size of the endpoint id (i.e. endpoint id =
// address[ep_id_idx, ep_id_idx + ep_id_size] table_offset - The value that is used to index into the endpoint table
// along with the endpoint id
void noc_address_translation_table_mask_table_entry(
    uint32_t entry, uint32_t mask, uint64_t compare, uint32_t ep_id_idx, uint32_t ep_id_size, uint32_t table_offset);

// Read back Mask Table registers for the specified entry
void noc_address_translation_table_mask_table_entry_read(
    uint32_t entry,
    uint32_t& mask,
    uint64_t& compare,
    uint32_t& ep_id_idx,
    uint32_t& ep_id_size,
    uint32_t& table_offset);

// Write an entry to the mask table with address rebase enabled
// entry - a value between 0 to 15
// mask - The number of MSB bits that will be compared with the address (e.g. 2 will create a mask
// 0xC000_0000_0000_0000) compare - The value that the masked address should be compared to ep_id_idx - The bit offset
// into the address that contains the endpoint id ep_id_size - The size of the endpoint id (i.e. endpoint id =
// address[ep_id_idx, ep_id_idx + ep_id_size] base - The new base that should be added to the offset of the address.
// table_offset - The value that is used to index into the endpoint table along with the endpoint id
void noc_address_translation_table_mask_table_entry_rebase(
    uint32_t entry,
    uint32_t mask,
    uint64_t compare,
    uint32_t ep_id_idx,
    uint32_t ep_id_size,
    uint64_t base,
    uint32_t table_offset);

// Read back Mask Table registers for the specified entry
void noc_address_translation_table_mask_table_entry_rebase_read(
    uint32_t entry,
    uint32_t& mask,
    uint64_t& compare,
    uint32_t& ep_id_idx,
    uint32_t& ep_id_size,
    uint64_t& base,
    uint32_t& table_offset);

// Write an entry into the routing table.
// entry - a value between 0 to 31
// compare - The value that {table_offset + endpoint_id, virtual_chnnel} should be compared to
// routing - An array with 16 elements that corresponds to the routing list
void noc_address_translation_table_routing_table_entry(uint32_t entry, uint32_t compare, uint32_t routing[]);

// Read back routing registers for the specified entry
void noc_address_translation_table_routing_table_entry_read(uint32_t entry, uint32_t& compare, uint32_t routing[]);

// Wrtie and entry into the endpoint table.
// entry - a value between 0 to 1023
// x - The translated (physical) x coodinate
// y - The translated (physical) y coordinate
void noc_address_translation_table_endpoint_table_entry(uint32_t entry, uint32_t x, uint32_t y);

// Read back endpoint table registers for the specified entry
void noc_address_translation_table_endpoint_table_entry_read(uint32_t entry, uint32_t& x, uint32_t& y);

#endif
