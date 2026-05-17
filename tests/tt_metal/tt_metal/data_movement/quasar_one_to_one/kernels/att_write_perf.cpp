// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Measures per-issue DM stall for back-to-back posted NOC writes, with and
// without ATT translation. Runs N issues in a tight loop on every user DM of
// the master tile (6 DMs) and divides the cycle-counter delta by N. Each DM
// DPRINTs its own number.
//
// Cross-DM coordination: DM at the lowest user hartid (2) configures ATT and
// drives an L1 barrier between the two phases so the other DMs don't observe
// an inconsistent ATT enable state mid-loop.
//
// NOC-address encoding follows the aether tb/dm_lib convention:
//   bits [25:0]   local L1 address (64 MB max)
//   bits [35:26]  endpoint id (10 bits) - meaningful when ATT translates
//   bit  [36]    "MASK_SHIFT" - flips between mask rules (low/high address)
//   bits [47:36]  physical (y,x) when ATT is off (NOC_XY_ENCODING)
// Our ATT mask rule fires for any address with the top 28 bits == 0 (bit 36
// clear, single rule), extracts ep_id from [35:26], and looks up the endpoint
// table at index ep_id+0.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "internal/tt-2xx/quasar/noc/noc_parameters.h"
#include "internal/tt-2xx/quasar/noc/registers/noc_address_translation_table_a_reg.h"
#include "internal/tt-2xx/quasar/dev_mem_map.h"
#include <cstdint>

namespace {

inline uint32_t rdcycle() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

inline uint32_t get_mhartid() {
    uint32_t h;
    asm volatile("csrr %0, mhartid" : "=r"(h));
    return h;
}

inline void att_reg_write(uint32_t addr, uint32_t val) { *reinterpret_cast<volatile uint32_t*>(addr) = val; }

inline void att_en(bool en_address_translation, bool en_dynamic_routing) {
    NOC_ADDRESS_TRANSLATION_TABLE_ENABLE_TABLES_reg_u ctrl;
    ctrl.val = 0;
    ctrl.f.en_address_translation = en_address_translation;
    ctrl.f.en_dynamic_routing = en_dynamic_routing;
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR + NOC_ADDRESS_TRANSLATION_TABLE_A_ENABLE_TABLES_REG_OFFSET,
        ctrl.val);
}

// Stride between consecutive mask-table entry register blocks.
constexpr uint32_t kMaskStride = NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
                                 NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4;

// `mask` is the number of MSB address bits to compare (matches the canonical
// API in tt_metal/hw/firmware/src/tt-2xx/quasar/noc_address_translation_tables.cpp).
// The hardware register encoding is (64 - mask).
inline void att_mask_table_entry(
    uint32_t entry, uint32_t mask, uint64_t compare, uint32_t ep_id_idx, uint32_t ep_id_size, uint32_t table_offset) {
    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u ctrl;
    ctrl.val = 0;
    ctrl.f.mask = 64 - mask;
    ctrl.f.ep_id_idx = ep_id_idx;
    ctrl.f.ep_id_size = ep_id_size;
    ctrl.f.table_offset = table_offset;
    ctrl.f.translate_addr = 0;
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + kMaskStride * entry,
        ctrl.val);
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET + kMaskStride * entry,
        static_cast<uint32_t>(compare));
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET + kMaskStride * entry,
        static_cast<uint32_t>(compare >> 32));
}

// Same as att_mask_table_entry but with translate_addr=1 (rebase variant) and
// a separate base-address pair written to BAR_LO/BAR_HI. Matches the aether
// configuration where remote-tile addresses (bit 36 set) use this rule.
inline void att_mask_table_entry_rebase(
    uint32_t entry,
    uint32_t mask,
    uint64_t compare,
    uint32_t ep_id_idx,
    uint32_t ep_id_size,
    uint64_t base,
    uint32_t table_offset) {
    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u ctrl;
    ctrl.val = 0;
    ctrl.f.mask = 64 - mask;
    ctrl.f.ep_id_idx = ep_id_idx;
    ctrl.f.ep_id_size = ep_id_size;
    ctrl.f.table_offset = table_offset;
    ctrl.f.translate_addr = 1;
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + kMaskStride * entry,
        ctrl.val);
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET + kMaskStride * entry,
        static_cast<uint32_t>(compare));
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET + kMaskStride * entry,
        static_cast<uint32_t>(compare >> 32));
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_LO_REG_OFFSET + kMaskStride * entry,
        static_cast<uint32_t>(base));
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET + kMaskStride * entry,
        static_cast<uint32_t>(base >> 32));
}

inline void att_endpoint_table_entry(uint32_t entry, uint32_t x, uint32_t y) {
    NOC_ADDRESS_TRANSLATION_TABLE_ENDPOINT_TABLE_ENTRY_reg_u ctrl;
    ctrl.val = 0;
    ctrl.f.x = x;
    ctrl.f.y = y;
    att_reg_write(
        NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR +
            NOC_ADDRESS_TRANSLATION_TABLE_A_ENDPOINT_TABLE_ENTRY_0__REG_OFFSET + 4 * entry,
        ctrl.val);
}

// ATT address encoding constants - match aether tb/dm_lib convention.
constexpr uint32_t kEpShift = 26;                               // ep_id at bits [35:26]
constexpr uint32_t kShift64gb = 36;                             // bit that distinguishes self vs remote
constexpr uint32_t kEpIdSize = kShift64gb - kEpShift;           // 10
constexpr uint32_t kMaskBitsAbove = 64 - kShift64gb;            // 28 MSBs compared
constexpr uint64_t kRemoteCompare = uint64_t{1} << kShift64gb;  // 0x1000000000
constexpr uint32_t kEndpointSelf = 0;                           // table[0] = self (rule 0)
constexpr uint32_t kEndpointDestIdx = 0;                        // ep_id 0 + table_offset 1 -> table[1] = dst

// L1 sync flags on the master tile. User DMs on Quasar are hartids 2..7 (DM0,
// DM1 are reserved). Lead is the lowest user hartid; done slots are packed by
// (hartid - lead).
constexpr uint32_t kNumDms = 6;
constexpr uint32_t kLeadHartid = 2;
constexpr uint32_t kSyncBase = 0x40000;
constexpr uint32_t kPhase1Go = kSyncBase + 0x000;
constexpr uint32_t kPhase2Go = kSyncBase + 0x004;
constexpr uint32_t kPhase1Done = kSyncBase + 0x100;
constexpr uint32_t kPhase2Done = kSyncBase + 0x200;

// Route sync-flag accesses through the L1 uncached alias so writes from one
// DM are immediately visible to spinning readers on other DMs.
inline volatile uint32_t* sync_ptr(uint32_t addr) {
    return reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + addr);
}

inline void wait_flag(uint32_t addr) {
    while (*sync_ptr(addr) != 1) {
    }
}

inline void wait_all_done(uint32_t base_addr) {
    for (uint32_t i = 1; i < kNumDms; i++) {
        wait_flag(base_addr + 4 * i);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t src_addr = get_arg(args::src_addr);
    constexpr uint32_t dst_addr = get_arg(args::dst_addr);
    constexpr uint32_t dst_x = get_arg(args::dst_x);
    constexpr uint32_t dst_y = get_arg(args::dst_y);
    constexpr uint32_t master_x = get_arg(args::master_x);
    constexpr uint32_t master_y = get_arg(args::master_y);
    constexpr uint32_t payload_bytes = get_arg(args::payload_bytes);
    constexpr uint32_t num_iters = get_arg(args::num_iters);

    const uint32_t hartid = get_mhartid();
    const uint32_t dm_idx = hartid - kLeadHartid;
    const bool is_lead = (hartid == kLeadHartid);

    DPRINT << "ENTER hartid=" << hartid << " dm_idx=" << dm_idx << " is_lead=" << (uint32_t)is_lead << ENDL();

    // ATT-on destination: set the MASK_SHIFT bit (bit 36) so rule 1 (rebase)
    // fires; rule 1 has table_offset=1, so endpoint id 0 at bits [35:26] looks
    // up table[0+1] = (dst_x, dst_y). Local addr in bits [25:0].
    const uint64_t dst_att_addr = (uint64_t{1} << kShift64gb) | (static_cast<uint64_t>(kEndpointDestIdx) << kEpShift) |
                                  static_cast<uint64_t>(dst_addr);
    // ATT-off destination: physical (x,y) at bits [47:36], local addr in
    // [35:0]. With ATT disabled this is the standard physical NOC encoding.
    const uint64_t dst_phys_addr = (static_cast<uint64_t>(dst_y) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
                                   (static_cast<uint64_t>(dst_x) << NOC_ADDR_LOCAL_BITS) |
                                   static_cast<uint64_t>(dst_addr);

    // ---------------------------------------------------------------------
    // Phase 1 setup: lead clears sync slots and configures ATT, others wait.
    // ---------------------------------------------------------------------
    if (is_lead) {
        *sync_ptr(kPhase1Go) = 0;
        *sync_ptr(kPhase2Go) = 0;
        for (uint32_t i = 0; i < kNumDms; i++) {
            *sync_ptr(kPhase1Done + 4 * i) = 0;
            *sync_ptr(kPhase2Done + 4 * i) = 0;
        }

        // Two mask-table rules, matching the aether tb/dm_lib convention:
        //   Rule 0: addresses with top 28 bits == 0 (bit 36 clear) -> self
        //   Rule 1: addresses with top 28 bits == 0x1 (bit 36 set) -> remote,
        //           ep_id offset by table_offset=1
        att_mask_table_entry(
            /*entry=*/0, /*mask=*/kMaskBitsAbove, /*compare=*/0, kEpShift, kEpIdSize, /*table_offset=*/0);
        att_mask_table_entry_rebase(
            /*entry=*/1,
            /*mask=*/kMaskBitsAbove,
            /*compare=*/kRemoteCompare,
            kEpShift,
            kEpIdSize,
            /*base=*/0,
            /*table_offset=*/1);
        att_endpoint_table_entry(/*entry=*/0, master_x, master_y);  // self (rule 0)
        att_endpoint_table_entry(/*entry=*/1, dst_x, dst_y);        // first remote (rule 1)
        att_en(/*en=*/true, /*dyn_routing=*/false);

        *sync_ptr(kPhase1Go) = 1;
    } else {
        wait_flag(kPhase1Go);
    }

    // ---------------------------------------------------------------------
    // Phase 1 measurement: every DM issues to ATT-translated address.
    // ---------------------------------------------------------------------
    reset_cmdbuf_0();
    setup_as_copy_cmdbuf_0(/*wr=*/true, /*mcast=*/false, /*mcast_exclude=*/{0}, /*wrapping_en=*/false, /*posted=*/true);
    setup_vcs_cmdbuf_0(/*wr=*/true);
    setup_trids_cmdbuf_0(/*trid_offset=*/0);
    setup_packet_tags_cmdbuf_0(/*snoop_bit=*/false, /*flush_bit=*/false);

    set_src_cmdbuf_0(static_cast<uint64_t>(src_addr));
    set_dest_cmdbuf_0(dst_att_addr);
    set_len_cmdbuf_0(payload_bytes);

    uint32_t t0_on = rdcycle();
    for (uint32_t i = 0; i < num_iters; i++) {
        issue_cmdbuf_0();
    }
    uint32_t t1_on = rdcycle();
    uint32_t cycles_on = t1_on - t0_on;

    // Drain after the measurement window so the per-issue number stays clean
    // but writes have actually landed before the phase boundary.
    while (!noc_writes_sent_cmdbuf_0()) {
    }

    *sync_ptr(kPhase1Done + 4 * dm_idx) = 1;

    // ---------------------------------------------------------------------
    // Phase 2 setup: lead waits for all DMs, disables ATT, releases others.
    // ---------------------------------------------------------------------
    if (is_lead) {
        wait_all_done(kPhase1Done);
        att_en(/*en=*/false, /*dyn_routing=*/false);
        *sync_ptr(kPhase2Go) = 1;
    } else {
        wait_flag(kPhase2Go);
    }

    // ---------------------------------------------------------------------
    // Phase 2 measurement: every DM issues to physical (dst_x, dst_y).
    // ---------------------------------------------------------------------
    reset_cmdbuf_0();
    setup_as_copy_cmdbuf_0(/*wr=*/true, /*mcast=*/false, /*mcast_exclude=*/{0}, /*wrapping_en=*/false, /*posted=*/true);
    setup_vcs_cmdbuf_0(/*wr=*/true);
    setup_trids_cmdbuf_0(/*trid_offset=*/0);
    setup_packet_tags_cmdbuf_0(/*snoop_bit=*/false, /*flush_bit=*/false);

    set_src_cmdbuf_0(static_cast<uint64_t>(src_addr));
    set_dest_cmdbuf_0(dst_phys_addr);
    set_len_cmdbuf_0(payload_bytes);

    uint32_t t0_off = rdcycle();
    for (uint32_t i = 0; i < num_iters; i++) {
        issue_cmdbuf_0();
    }
    uint32_t t1_off = rdcycle();
    uint32_t cycles_off = t1_off - t0_off;

    while (!noc_writes_sent_cmdbuf_0()) {
    }

    *sync_ptr(kPhase2Done + 4 * dm_idx) = 1;

    if (is_lead) {
        wait_all_done(kPhase2Done);
    }

    DPRINT << "DM" << hartid << " ATT_WRITE_PERF iters=" << num_iters << " bytes=" << payload_bytes
           << " att_on_per_issue=" << (cycles_on / num_iters) << " att_off_per_issue=" << (cycles_off / num_iters)
           << ENDL();
}
