// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Companion to att_write_perf.cpp: same structure, but uses non-posted writes
// so each issue waits for the destination ack. The measured per-issue cycles
// therefore include end-to-end round-trip latency (issue -> NOC fabric ->
// dst NOC ingress -> ack returned). Comparing ATT-on vs ATT-off here pins
// down whether ATT translation adds latency to the round-trip, not just to
// the local issue path.
//
// Kernel structure (barriers, ATT setup, address encoding, cmdbuf setup
// requirements) is identical to att_write_perf.cpp - only the cmdbuf MISC
// posted flag and the drain probe differ.

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

constexpr uint32_t kMaskStride = NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET -
                                 NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET + 4;

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
constexpr uint32_t kEpShift = 26;
constexpr uint32_t kShift64gb = 36;
constexpr uint32_t kEpIdSize = kShift64gb - kEpShift;
constexpr uint32_t kMaskBitsAbove = 64 - kShift64gb;
constexpr uint64_t kRemoteCompare = uint64_t{1} << kShift64gb;
constexpr uint32_t kEndpointDestIdx = 0;  // ep_id 0 + table_offset 1 -> table[1] = dst

// Cross-DM L1 barrier.
constexpr uint32_t kNumDms = 6;
constexpr uint32_t kLeadHartid = 2;
constexpr uint32_t kSyncBase = 0x40000;
constexpr uint32_t kPhase1Go = kSyncBase + 0x000;
constexpr uint32_t kPhase2Go = kSyncBase + 0x004;
constexpr uint32_t kPhase1Done = kSyncBase + 0x100;
constexpr uint32_t kPhase2Done = kSyncBase + 0x200;

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

    const uint64_t dst_att_addr = (uint64_t{1} << kShift64gb) | (static_cast<uint64_t>(kEndpointDestIdx) << kEpShift) |
                                  static_cast<uint64_t>(dst_addr);
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
        att_endpoint_table_entry(/*entry=*/0, master_x, master_y);
        att_endpoint_table_entry(/*entry=*/1, dst_x, dst_y);
        att_en(/*en=*/true, /*dyn_routing=*/false);

        *sync_ptr(kPhase1Go) = 1;
    } else {
        wait_flag(kPhase1Go);
    }

    // ---------------------------------------------------------------------
    // Phase 1 measurement: non-posted writes through ATT.
    // ---------------------------------------------------------------------
    reset_cmdbuf_0();
    setup_as_copy_cmdbuf_0(
        /*wr=*/true, /*mcast=*/false, /*mcast_exclude=*/{0}, /*wrapping_en=*/false, /*posted=*/false);
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
    // For non-posted writes, the round-trip ack is part of the per-issue cost
    // - the cmdbuf stalls subsequent issues once max-outstanding is reached.
    // Include the tail ack-wait inside the measurement window so the average
    // reflects round-trip latency rather than only issue-rate.
    while (!noc_nonposted_writes_acked_cmdbuf_0()) {
    }
    uint32_t t1_on = rdcycle();
    uint32_t cycles_on = t1_on - t0_on;

    *sync_ptr(kPhase1Done + 4 * dm_idx) = 1;

    // ---------------------------------------------------------------------
    // Phase 2 setup.
    // ---------------------------------------------------------------------
    if (is_lead) {
        wait_all_done(kPhase1Done);
        att_en(/*en=*/false, /*dyn_routing=*/false);
        *sync_ptr(kPhase2Go) = 1;
    } else {
        wait_flag(kPhase2Go);
    }

    // ---------------------------------------------------------------------
    // Phase 2 measurement: non-posted writes to physical (dst_x, dst_y).
    // ---------------------------------------------------------------------
    reset_cmdbuf_0();
    setup_as_copy_cmdbuf_0(
        /*wr=*/true, /*mcast=*/false, /*mcast_exclude=*/{0}, /*wrapping_en=*/false, /*posted=*/false);
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
    while (!noc_nonposted_writes_acked_cmdbuf_0()) {
    }
    uint32_t t1_off = rdcycle();
    uint32_t cycles_off = t1_off - t0_off;

    *sync_ptr(kPhase2Done + 4 * dm_idx) = 1;

    if (is_lead) {
        wait_all_done(kPhase2Done);
    }

    DPRINT << "DM" << hartid << " ATT_WRITE_NONPOSTED_PERF iters=" << num_iters << " bytes=" << payload_bytes
           << " att_on_per_issue=" << (cycles_on / num_iters) << " att_off_per_issue=" << (cycles_off / num_iters)
           << ENDL();
}
