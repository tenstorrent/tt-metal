// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_isr.h"
#include "internal/tt-2xx/quasar/overlay/remapper_api.hpp"
#include "internal/circular_buffer_interface.h"  // for cb_addr_shift
#include "internal/tt-2xx/quasar/dev_mem_map.h"
#include "internal/tt-2xx/risc_common.h"
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#else
#include "ckernel_trisc_common.h"
#endif

FORCE_INLINE volatile uint32_t* dfb_init_timing_slot_words(uint8_t slot) {
    // Device writes via uncached L1 alias; host reads the mirrored cached offset (TL1 @ 0x3ffc00).
    return reinterpret_cast<volatile uint32_t*>(
        static_cast<uintptr_t>(MEM_L1_UNCACHED_BASE) +
        static_cast<uintptr_t>(dfb::DFB_INIT_TIMING_L1_BYTE_OFFSET) +
        static_cast<uint32_t>(slot) * dfb::DFB_INIT_TIMING_WORDS_PER_SLOT * sizeof(uint32_t));
}

FORCE_INLINE void dfb_init_timing_write_slot(
    uint8_t slot,
    uint8_t role,
    uint32_t e2e,
    uint32_t metric_a,
    uint32_t metric_b,
    uint32_t metric_c,
    uint32_t metric_d,
    uint32_t metric_e,
    uint32_t metric_f,
    uint32_t start_time,
    uint32_t end_time,
    uint32_t metric_g = 0,
    uint32_t metric_h = 0,
    uint32_t metric_i = 0,
    uint32_t metric_j = 0) {
    volatile uint32_t* words = dfb_init_timing_slot_words(slot);
    // Publish pattern: write payload first, VALID last. Uncached stores land in TL1 directly.
    words[dfb::DFB_INIT_TIMING_W_MAGIC] = dfb::DFB_INIT_TIMING_MAGIC;
    words[dfb::DFB_INIT_TIMING_W_ROLE] = role;
    words[dfb::DFB_INIT_TIMING_W_E2E] = e2e;
    words[dfb::DFB_INIT_TIMING_W_METRIC_A] = metric_a;
    words[dfb::DFB_INIT_TIMING_W_METRIC_B] = metric_b;
    words[dfb::DFB_INIT_TIMING_W_METRIC_C] = metric_c;
    words[dfb::DFB_INIT_TIMING_W_METRIC_D] = metric_d;
    words[dfb::DFB_INIT_TIMING_W_METRIC_E] = metric_e;
    words[dfb::DFB_INIT_TIMING_W_METRIC_F] = metric_f;
    words[dfb::DFB_INIT_TIMING_W_START] = start_time;
    words[dfb::DFB_INIT_TIMING_W_END] = end_time;
    words[dfb::DFB_INIT_TIMING_W_METRIC_G] = metric_g;
    words[dfb::DFB_INIT_TIMING_W_METRIC_H] = metric_h;
    words[dfb::DFB_INIT_TIMING_W_METRIC_I] = metric_i;
    words[dfb::DFB_INIT_TIMING_W_METRIC_J] = metric_j;
    asm volatile("fence w, w" ::: "memory");
    words[dfb::DFB_INIT_TIMING_W_VALID] = 1u;
    asm volatile("fence w, w" ::: "memory");
}

#ifdef COMPILE_FOR_TRISC
FORCE_INLINE uint8_t dfb_init_timing_trisc_slot_index() {
    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
#if defined(UCK_CHLKC_PACK)
    return static_cast<uint8_t>(8u + neo_id * 2u + 1u);
#else
    return static_cast<uint8_t>(8u + neo_id * 2u);
#endif
}
#endif

// Participant mask (bits 0-11) only; storage u16 also includes tensix_trisc_mask in bits 12-15.
FORCE_INLINE uint16_t load_dfb_risc_mask(const volatile dfb_initializer_t* init) {
    const volatile uint8_t* bp =
        reinterpret_cast<const volatile uint8_t*>(init) + offsetof(dfb_initializer_t, risc_mask_bits);
    return static_cast<uint16_t>(bp[0]) | (static_cast<uint16_t>(bp[1] & 0x0Fu) << 8);
}

FORCE_INLINE void copy_txn_descriptor_32(
    volatile TxnDFBDescriptor* dst, const volatile dfb_dm0_txn_descriptor_image_t* src) {
    const volatile uint32_t* s = reinterpret_cast<const volatile uint32_t*>(src);
    volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(dst);
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
    d[3] = s[3];
    d[4] = s[4];
    d[5] = s[5];
    d[6] = s[6];
    d[7] = s[7];
}

// tc_init_done is published/consumed via the uncached TL1 alias so DM L2 flush is not
// required for Neo TRISC waiters (TRISC has no Quasar L2 invalidate API).
static constexpr uint32_t dfb_per_risc_tc_init_done_byte_off =
    offsetof(dfb_initializer_per_risc_t, num_tcs_and_init);
static constexpr uint8_t dfb_per_risc_tc_init_done_bit = 4;

FORCE_INLINE volatile uint8_t* dfb_per_risc_tc_init_done_byte_uncached(
    volatile dfb_initializer_per_risc_t* per_risc_ptr) {
    return reinterpret_cast<volatile uint8_t*>(
        reinterpret_cast<uintptr_t>(per_risc_ptr) + MEM_L1_UNCACHED_BASE + dfb_per_risc_tc_init_done_byte_off);
}

FORCE_INLINE void dfb_publish_per_risc_tc_init_done(volatile dfb_initializer_per_risc_t* per_risc_ptr) {
    per_risc_ptr->num_tcs_and_init.tc_init_done = 1;
    volatile uint8_t* tc_init_byte = dfb_per_risc_tc_init_done_byte_uncached(per_risc_ptr);
    *tc_init_byte = static_cast<uint8_t>(*tc_init_byte | (1u << dfb_per_risc_tc_init_done_bit));
    asm volatile("fence w, w" ::: "memory");
}

FORCE_INLINE bool dfb_load_per_risc_tc_init_done(volatile dfb_initializer_per_risc_t* per_risc_ptr) {
    const volatile uint8_t* tc_init_byte = dfb_per_risc_tc_init_done_byte_uncached(per_risc_ptr);
    return ((*tc_init_byte) & static_cast<uint8_t>(1u << dfb_per_risc_tc_init_done_bit)) != 0;
}

// Returns true when every local producer RISC on this DFB has tc_init_done set.
// Only producer entries are checked; consumers never set tc_init_done. Cores with no
// local producers (e.g. Tensix consumer binding) vacuously pass.
FORCE_INLINE bool dfb_producers_tc_init_done(
    volatile tt_l1_ptr uint8_t* config_base,
    volatile uint16_t* per_risc_byte_offset_table,
    volatile dfb_initializer_t* init_ptr,
    uint8_t logical_dfb_id) {
    const uint16_t risc_mask = load_dfb_risc_mask(init_ptr);
    uint16_t pending = risc_mask;
    while (pending) {
        const uint8_t hartid = static_cast<uint8_t>(__builtin_ctz(pending));
        pending = static_cast<uint16_t>(pending & (pending - 1u));

        const uint32_t table_idx =
            dfb_per_risc_byte_offset_table_index(logical_dfb_id, hartid);
        const uint16_t per_risc_off = per_risc_byte_offset_table[table_idx];

        volatile dfb_initializer_per_risc_t* per_risc_ptr =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(config_base + per_risc_off);
        if (!per_risc_ptr->flags.is_producer) {
            continue;
        }
        if (!dfb_load_per_risc_tc_init_done(per_risc_ptr)) {
            return false;
        }
    }
    return true;
}

// Poll until local producers on each participating DFB have tc_init_done set.
// Tracks a per-DFB ready mask so already-complete DFBs are not re-walked on later poll iterations.
// TC init and DM0 ISR setup are independent; both are checked each iteration and required to exit.
// DMs on cores with implicit sync also require ghdr->dm0_isr_ready. TRISC skips the ISR requirement.
FORCE_INLINE void wait_all_tcs_initialized(uint32_t tt_l1_ptr* dfb_config_base, uint8_t hart_u8) {
    WAYPOINT("TCIW");
    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    const uint32_t num_dfbs = ghdr->num_dfbs;
    volatile uint16_t* dfb_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_byte_offset_table_byte_offset());
    volatile uint16_t* per_risc_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_per_risc_byte_offset_table_byte_offset(static_cast<uint8_t>(num_dfbs)));

    const bool need_isr_gate =
#ifndef COMPILE_FOR_TRISC
        ghdr->per_dfb_layout_offset > ghdr->dm0_isr_blob_offset;
#else
        false;
#endif

    uint32_t producers_ready_mask = 0;
    const uint32_t wait_dfbs_mask = ghdr->participation_mask[hart_u8];

    while (true) {
        uint32_t pending_dfbs = wait_dfbs_mask & ~producers_ready_mask;
        while (pending_dfbs) {
            const uint32_t logical_dfb_id = static_cast<uint32_t>(__builtin_ctz(pending_dfbs));
            pending_dfbs &= pending_dfbs - 1u;

            const uint32_t layout_byte_off = dfb_byte_offset_table[logical_dfb_id];
            volatile dfb_initializer_t* init_ptr =
                reinterpret_cast<volatile dfb_initializer_t*>(config_base + layout_byte_off);

            if (dfb_producers_tc_init_done(
                    config_base, per_risc_byte_offset_table, init_ptr, static_cast<uint8_t>(logical_dfb_id))) {
                producers_ready_mask |= (1u << logical_dfb_id);
                WAYPOINT("PDI");
            } else {
                WAYPOINT("PND");
            }
        }

        const bool all_producers_ready = producers_ready_mask == wait_dfbs_mask;
        const bool isr_ready = !need_isr_gate || ghdr->dm0_isr_ready == 1;
        if (all_producers_ready && isr_ready) {
            break;
        }
    }
    WAYPOINT("TCID");
}

inline uint32_t rdcycle() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

#ifndef COMPILE_FOR_TRISC

FORCE_INLINE void setup_dfb_implicit_sync(uint32_t tt_l1_ptr* dfb_config_base, uint32_t /*num_dfbs*/) {
    uint32_t start_time = rdcycle();

    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    uint32_t dm0_isr_blob_offset = ghdr->dm0_isr_blob_offset;

    volatile tt_l1_ptr uint8_t* dm0_blob_ptr = config_base + dm0_isr_blob_offset;

    uint32_t producer_txn_id_mask = 0;
    uint32_t consumer_txn_id_mask = 0;

    // Core-wide masks precomputed on host (Phase A).
    const volatile tt_l1_ptr uint32_t* core_hdr_src =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(dm0_blob_ptr);
    producer_txn_id_mask = core_hdr_src[0];
    consumer_txn_id_mask = core_hdr_src[1];
    dm0_blob_ptr += sizeof(dfb_dm0_isr_blob_core_header_t);

    WAYPOINT("IS1");

    const uint32_t txn_hw_bytes =
        dm0_isr_txn_hw_pool_byte_size(producer_txn_id_mask, consumer_txn_id_mask);
    const uint32_t pool_bytes = dm0_isr_txn_desc_pool_byte_size(producer_txn_id_mask, consumer_txn_id_mask);
    const volatile tt_l1_ptr uint8_t* pool_base = dm0_blob_ptr + txn_hw_bytes;

    const volatile dfb_dm0_isr_txn_threshold_t* txn_threshold_table =
        reinterpret_cast<const volatile dfb_dm0_isr_txn_threshold_t*>(dm0_blob_ptr);

    const uint32_t t_before_desc_copy = rdcycle();
    if (pool_bytes != 0) {
        const uint32_t all_mask = producer_txn_id_mask | consumer_txn_id_mask;
        uint32_t pending_desc = all_mask;
        while (pending_desc) {
            const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending_desc));
            copy_txn_descriptor_32(
                &g_txn_dfb_descriptor[txn_id],
                reinterpret_cast<const volatile dfb_dm0_txn_descriptor_image_t*>(pool_base + txn_id * 32u));
            pending_desc &= (pending_desc - 1u);
        }
    }
    const uint32_t t_after_desc_copy = rdcycle();

    uint32_t total_l1_read = 0;
    uint32_t total_rocc_issue = 0;
    const uint32_t t_before_cmdbuf = t_after_desc_copy;

    uint32_t pending = producer_txn_id_mask;
    while (pending) {
        const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending));
        const uint32_t t_slot_start = rdcycle();
        const uint32_t threshold = txn_threshold_table[txn_id].threshold;
        const uint32_t t_after_l1 = rdcycle();
        total_l1_read += t_after_l1 - t_slot_start;
        CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, txn_id);
        asm volatile("nop");
        SET_TILES_TO_PROCESS_THRES_TR_ACK(txn_id, threshold);
        const uint32_t t_after_rocc = rdcycle();
        total_rocc_issue += t_after_rocc - t_after_l1;
        pending &= (pending - 1u);
    }

    pending = consumer_txn_id_mask;
    while (pending) {
        const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending));
        const uint32_t t_slot_start = rdcycle();
        const uint32_t threshold = txn_threshold_table[txn_id].threshold;
        const uint32_t t_after_l1 = rdcycle();
        total_l1_read += t_after_l1 - t_slot_start;
        CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, txn_id);
        asm volatile("nop");
        SET_TILES_TO_PROCESS_THRES_WR_SENT(txn_id, threshold);
        const uint32_t t_after_rocc = rdcycle();
        total_rocc_issue += t_after_rocc - t_after_l1;
        pending &= (pending - 1u);
    }
    const uint32_t t_after_cmdbuf = rdcycle();

    const uint32_t t_before_ie = rdcycle();
    uint64_t reg_val = CMDBUF_RD_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(producer_txn_id_mask & 0xFFFFFFFFULL) << 32);
    CMDBUF_WR_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);
    const uint32_t t_after_first_ie_rmw = rdcycle();

    reg_val = CMDBUF_RD_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (consumer_txn_id_mask & 0xFFFFFFFFULL);
    CMDBUF_WR_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);
    const uint32_t t_after_isr_ie_writes = rdcycle();

    if ((producer_txn_id_mask | consumer_txn_id_mask) != 0) {
        enable_dfb_tile_isr();
    } else {
        disable_dfb_tile_isr();
    }
    const uint32_t end_isr_enable_time = rdcycle();

    ghdr->dm0_isr_ready = 1;
    asm volatile("fence w, w" ::: "memory");

    WAYPOINT("ISD");

    const uint32_t end_time = rdcycle();

    const uint32_t pre_loop_sw = t_before_desc_copy - start_time;
    const uint32_t subpassB_desc = t_after_desc_copy - t_before_desc_copy;
    const uint32_t between_dfb_sw = 0;
    const uint32_t subpassB_hw = t_after_cmdbuf - t_before_cmdbuf;
    const uint32_t first_ie_rmw = t_after_first_ie_rmw - t_before_ie;
    const uint32_t second_ie_rmw = t_after_isr_ie_writes - t_after_first_ie_rmw;
    const uint32_t isr_enable = (end_isr_enable_time > t_after_isr_ie_writes)
                                    ? end_isr_enable_time - t_after_isr_ie_writes
                                    : 0;
    dfb_init_timing_write_slot(
        0,
        dfb::DFB_INIT_TIMING_ROLE_DM0_ISR,
        end_time - start_time,
        pre_loop_sw,
        subpassB_desc,
        between_dfb_sw,
        total_l1_read,
        total_rocc_issue,
        first_ie_rmw,
        start_time,
        end_time,
        second_ie_rmw,
        isr_enable,
        0,
        subpassB_hw);
}

FORCE_INLINE void setup_dfb_remapper(uint32_t tt_l1_ptr* dfb_config_base, uint32_t num_dfbs) {
    const uint32_t start_time = rdcycle();

    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    num_dfbs = ghdr->num_dfbs;
    const uint32_t dm1_remapper_blob_offset = ghdr->dm1_remapper_blob_offset;
    volatile tt_l1_ptr uint8_t* dm1_blob_ptr = config_base + dm1_remapper_blob_offset;

    bool enable_remapper = false;
    uint32_t end_remapper_config_time = 0;

    WAYPOINT("RS");

    uint32_t blob_l1_read_sw = 0;
    uint32_t pairs_reg_hw = 0;
    uint32_t blob_loop_ovhd = 0;
    uint32_t pairs_slots_written = 0;
    uint32_t first_pair_clientR_hw = 0;
    uint32_t first_pair_clientL_hw = 0;
    uint32_t last_pair_hw = 0;
    bool first_slot_written = false;

    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        const uint32_t t_pass_start = rdcycle();
        uint32_t pass_l1_read = 0;
        uint32_t pass_pairs_reg = 0;

        const volatile dfb_dm1_remapper_entry_header_t* entry_hdr =
            reinterpret_cast<const volatile dfb_dm1_remapper_entry_header_t*>(dm1_blob_ptr);
        const int num_rmp = entry_hdr->num_remapper_slots;
        WAYPOINT("RS2");

        const uint32_t entry_bytes = sizeof(dfb_dm1_remapper_entry_header_t)
                                     + static_cast<uint32_t>(num_rmp) * sizeof(dfb_dm0_remapper_slot_t);
        const volatile dfb_dm0_remapper_slot_t* slots =
            reinterpret_cast<const volatile dfb_dm0_remapper_slot_t*>(
                dm1_blob_ptr + sizeof(dfb_dm1_remapper_entry_header_t));
        const uint32_t t_after_hdr = rdcycle();
        pass_l1_read += t_after_hdr - t_pass_start;

        for (int s = 0; s < num_rmp; s++) {
            const uint32_t t_slot_read_start = rdcycle();
            const uint32_t pair_idx = slots[s].pair_index;
            const uint32_t clientR_val = slots[s].clientR_val;
            const uint32_t clientL_val = slots[s].clientL_val;
            const uint32_t t_after_slot_read = rdcycle();
            pass_l1_read += t_after_slot_read - t_slot_read_start;

            const uint32_t t_clientR_start = rdcycle();
            WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pair_idx), clientR_val);
            const uint32_t t_after_clientR = rdcycle();
            if (!first_slot_written) {
                first_pair_clientR_hw = t_after_clientR - t_clientR_start;
            }

            const uint32_t t_clientL_start = rdcycle();
            WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx), clientL_val);
            const uint32_t t_after_clientL = rdcycle();
            if (!first_slot_written) {
                first_pair_clientL_hw = t_after_clientL - t_clientL_start;
                first_slot_written = true;
            }

            const uint32_t pair_reg_hw = (t_after_clientR - t_clientR_start) + (t_after_clientL - t_clientL_start);
            pass_pairs_reg += pair_reg_hw;
            last_pair_hw = pair_reg_hw;
            pairs_slots_written++;
            enable_remapper = true;
        }

        dm1_blob_ptr += entry_bytes;
        const uint32_t t_pass_end = rdcycle();
        blob_l1_read_sw += pass_l1_read;
        pairs_reg_hw += pass_pairs_reg;
        blob_loop_ovhd += (t_pass_end - t_pass_start) - pass_l1_read - pass_pairs_reg;
    }

    uint32_t enable_remapper_hw = 0;
    if (enable_remapper) {
        const uint32_t t_before_enable = rdcycle();
        g_remapper_configurator.enable_remapper();
        end_remapper_config_time = rdcycle();
        enable_remapper_hw = end_remapper_config_time - t_before_enable;
    }

    WAYPOINT("RSD");
    const uint32_t end_time = rdcycle();

    dfb_init_timing_write_slot(
        1,
        dfb::DFB_INIT_TIMING_ROLE_DM1_RMP,
        end_time - start_time,
        blob_l1_read_sw,
        blob_loop_ovhd,
        pairs_reg_hw,
        enable_remapper_hw,
        first_pair_clientR_hw,
        first_pair_clientL_hw,
        start_time,
        end_time,
        last_pair_hw,
        0,
        0,
        pairs_slots_written);
}

#endif  // !COMPILE_FOR_TRISC

// DM0/DM1 coordinators run setup_dfb_implicit_sync / setup_dfb_remapper from DM firmware (no TC wait).
// DM2-7 + TRISC: shared per-DFB layout loop (dfb_initializer_t + per-risc entries), then wait_all_tcs_initialized.
FORCE_INLINE void setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t local_dfb_mask) {
    uint32_t start_time = rdcycle();

    uint64_t hartid;
#ifdef COMPILE_FOR_TRISC
    std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    // Building up g_dfb_interface is not at granularity of trisc in a Neo so only need Neo ID here
    // The initialization structs track producers/consumers for a given DFB and they would only be used by one of the
    // unpacker or packer
    hartid = 8 + neo_id;
#else
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
#endif
    const uint8_t hart_u8 = static_cast<uint8_t>(hartid);

    // Read the global header: region offsets, dfb_byte_offset[], per_risc_byte_offset[][], participation_mask[].
    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    uint32_t num_dfbs = ghdr->num_dfbs;
    (void)local_dfb_mask;  // launch local_cb_mask is a legacy carrier; ghdr->num_dfbs is authoritative
    volatile uint16_t* dfb_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_byte_offset_table_byte_offset());
    volatile uint16_t* per_risc_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_per_risc_byte_offset_table_byte_offset(static_cast<uint8_t>(num_dfbs)));

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    uint8_t compact_dfb_count = 0;
#endif

    // Timing probes for DM2-7 + TRISC merged loop.
    //   merged_sw     = t_after_merged_loop - start_time
    //   remapper_spin = accumulated producer spin before llk_intf (DM2-7 / TRISC prod)
    //   tc_hw         = accumulated llk_intf_reset/set_capacity or tile_counter HW
    //   wait_all      = t_after_wait_all - t_before_wait_all (barrier poll in wait_all_tcs_initialized)
    uint32_t t_after_merged_loop = 0;
    uint32_t t_before_wait_all = 0;
    uint32_t t_after_wait_all = 0;
    uint32_t total_remapper_spin = 0;
    uint32_t total_tc_hw = 0;
    uint32_t total_tc_reset_hw = 0;
    uint32_t total_tc_capacity_hw = 0;
    uint32_t t_before_tc_writes = 0;

    // -----------------------------------------------------------------------
    // DM2-7 + TRISC: shared per-DFB layout loop.
    // Populates g_dfb_interface (TC base/limit addrs, txn IDs, entry sizes) for
    // each RISC that participates as a producer or consumer.
    // DM0 and DM1 are pure coordinators and skip this loop entirely.
    // Host guarantees participation_mask[hartid] is well-formed (no bits >= num_dfbs).
    // -----------------------------------------------------------------------
    {
        uint32_t participating = ghdr->participation_mask[hart_u8];
        while (participating) {
            const uint32_t logical_dfb_id = __builtin_ctz(participating);
            // DPRINT("participating: {}\n", participating);
            participating &= participating - 1u;
            // DPRINT("logical_dfb_id: {}\n", logical_dfb_id);
            WAYPOINT("L1");

            const uint32_t layout_byte_off = dfb_byte_offset_table[logical_dfb_id];
            volatile dfb_initializer_t* init_ptr =
                reinterpret_cast<volatile dfb_initializer_t*>(config_base + layout_byte_off);

            WAYPOINT("L2");

            const uint32_t per_risc_table_idx =
                dfb_per_risc_byte_offset_table_index(static_cast<uint8_t>(logical_dfb_id), hart_u8);
            const uint32_t per_risc_byte_off = per_risc_byte_offset_table[per_risc_table_idx];
            volatile dfb_initializer_per_risc_t* per_risc_ptr =
                reinterpret_cast<volatile dfb_initializer_per_risc_t*>(config_base + per_risc_byte_off);

            WAYPOINT("L3");

            // --- Sub-pass A: populate g_dfb_interface ---
            {

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
                ASSERT(compact_dfb_count < dfb::MAX_ACTIVE_DFBS_PACK);
                const uint8_t compact_dfb_id = compact_dfb_count++;
                g_dfb_logical_to_compact[logical_dfb_id] = compact_dfb_id;
                LocalDFBInterface& dfb_interface = g_dfb_interface[compact_dfb_id];
#else
                LocalDFBInterface& dfb_interface = g_dfb_interface[logical_dfb_id];
#endif

                WAYPOINT("L4");
                const uint8_t num_tcs = per_risc_ptr->num_tcs_and_init.num_tcs_to_rr;
                dfb_interface.num_tcs_to_rr = num_tcs;
                // DPRINT("dfb_interface.num_tcs_to_rr: {}\n", dfb_interface.num_tcs_to_rr);
#ifdef COMPILE_FOR_TRISC
                dfb_interface.entry_size = static_cast<uint16_t>(init_ptr->entry_size >> cb_addr_shift);
                dfb_interface.stride_size = static_cast<uint16_t>(
                    static_cast<uint32_t>(dfb_interface.entry_size) * static_cast<uint32_t>(init_ptr->stride_in_entries));
                dfb_interface.stride_size_tiles = static_cast<uint8_t>(init_ptr->stride_in_entries);
#if defined(UCK_CHLKC_PACK)
                dfb_interface.wr_entry_ptr = 0;
#else
                dfb_interface.tensix_trisc_mask = static_cast<uint8_t>(init_ptr->risc_mask_bits.tensix_trisc_mask);
#endif
#else
                dfb_interface.entry_size = init_ptr->entry_size >> cb_addr_shift;
                dfb_interface.stride_size = dfb_interface.entry_size * init_ptr->stride_in_entries;
#endif

                // DPRINT("got here");
                WAYPOINT("L5");

                for (int i = 0; i < num_tcs; i++) {
                    uint32_t base = per_risc_ptr->tc_addrs[i].base_addr >> cb_addr_shift;
                    uint32_t limit_s = per_risc_ptr->tc_addrs[i].limit >> cb_addr_shift;
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
                    dfb_interface.tc_slots[i].base_addr = base;
                    dfb_interface.tc_slots[i].wr_offset = 0;
                    dfb_interface.tc_slots[i].ring_size = static_cast<uint16_t>(limit_s - base);
                    dfb_interface.tc_slots[i].packed_tile_counter = per_risc_ptr->packed_tile_counter[i];
                    dfb_interface.tc_slots[i].base_entry_idx = static_cast<uint16_t>(
                        (base - dfb_interface.tc_slots[0].base_addr) / dfb_interface.entry_size);
                    dfb_interface.tc_slots[i].wr_entry_idx = dfb_interface.tc_slots[i].base_entry_idx;
#elif defined(COMPILE_FOR_TRISC)
                    dfb_interface.tc_slots[i].base_addr = base;
                    dfb_interface.tc_slots[i].rd_offset = 0;
                    dfb_interface.tc_slots[i].ring_size = static_cast<uint16_t>(limit_s - base);
                    dfb_interface.tc_slots[i].packed_tile_counter = per_risc_ptr->packed_tile_counter[i];
                    dfb_interface.tc_slots[i].base_entry_idx = static_cast<uint16_t>(
                        (base - dfb_interface.tc_slots[0].base_addr) / dfb_interface.entry_size);
                    dfb_interface.tc_slots[i].rd_entry_idx = dfb_interface.tc_slots[i].base_entry_idx;
#else
                    dfb_interface.tc_slots[i].base_addr = base;
                    dfb_interface.tc_slots[i].limit = limit_s;
                    dfb_interface.tc_slots[i].rd_ptr = base;
                    dfb_interface.tc_slots[i].wr_ptr = base;
                    dfb_interface.tc_slots[i].packed_tile_counter = per_risc_ptr->packed_tile_counter[i];
                    // DPRINT("dfb_interface.tc_slots[i].base_addr: {}\n", dfb_interface.tc_slots[i].base_addr);
#endif
                }

                WAYPOINT("L6");

                dfb_interface.tc_idx = 0;
#ifndef COMPILE_FOR_TRISC
                dfb_interface.broadcast_tc = per_risc_ptr->num_tcs_and_init.broadcast_tc;

                if (per_risc_ptr->flags.is_producer) {
                    dfb_interface.num_txn_ids = init_ptr->producer_txn_descriptor.num_txn_ids;
                    dfb_interface.threshold = init_ptr->producer_txn_descriptor.num_entries_to_process_threshold;
                    dfb_interface.num_entries_per_txn_id = init_ptr->producer_txn_descriptor.num_entries_per_txn_id;
                    dfb_interface.num_entries_per_txn_id_per_tc =
                        init_ptr->producer_txn_descriptor.num_entries_per_txn_id_per_tc;
                    for (int i = 0; i < dfb_interface.num_txn_ids; i++) {
                        dfb_interface.txn_ids[i] = init_ptr->producer_txn_descriptor.txn_ids[i];
                    }
                } else {
                    dfb_interface.num_txn_ids = init_ptr->consumer_txn_descriptor.num_txn_ids;
                    dfb_interface.threshold = init_ptr->consumer_txn_descriptor.num_entries_to_process_threshold;
                    dfb_interface.num_entries_per_txn_id = init_ptr->consumer_txn_descriptor.num_entries_per_txn_id;
                    dfb_interface.num_entries_per_txn_id_per_tc =
                        init_ptr->consumer_txn_descriptor.num_entries_per_txn_id_per_tc;
                    for (int i = 0; i < dfb_interface.num_txn_ids; i++) {
                        dfb_interface.txn_ids[i] = init_ptr->consumer_txn_descriptor.txn_ids[i];
                    }
                }
#endif

                WAYPOINT("L7");

                // --- TC hardware init merged into this loop (producers only) ---
                if (per_risc_ptr->flags.is_producer) {
                    if (per_risc_ptr->flags.remapper_en) {
                        if (!t_before_tc_writes) {
                            t_before_tc_writes = rdcycle();
                        }
                        const uint32_t spin_start = rdcycle();
                        while (per_risc_ptr->flags.remapper_en && !overlay::RemapperAPI::is_remapper_enabled());
                        total_remapper_spin += rdcycle() - spin_start;
                    }
                    const uint32_t tc_hw_start = rdcycle();
                    for (int tc = 0; tc < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; tc++) {
                        dfb::PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                        uint8_t tc_id = dfb::get_counter_id(ptc);
#ifndef COMPILE_FOR_TRISC
                        uint8_t tensix_id = dfb::get_tensix_id(ptc);
                        const uint32_t t_reset_start = rdcycle();
                        overlay::fast_llk_intf_reset(tensix_id, tc_id);
                        total_tc_reset_hw += rdcycle() - t_reset_start;
                        const uint32_t t_capacity_start = rdcycle();
                        overlay::fast_llk_intf_set_capacity(tensix_id, tc_id, init_ptr->capacity);
                        total_tc_capacity_hw += rdcycle() - t_capacity_start;
#elif defined(UCK_CHLKC_PACK)
                        const uint32_t t_reset_start = rdcycle();
                        ckernel::trisc::tile_counters[tc_id].f.reset = 1;
                        total_tc_reset_hw += rdcycle() - t_reset_start;
                        const uint32_t t_capacity_start = rdcycle();
                        ckernel::trisc::tile_counters[tc_id].f.buf_capacity = init_ptr->capacity;
                        total_tc_capacity_hw += rdcycle() - t_capacity_start;
#endif
                    }
                    total_tc_hw += rdcycle() - tc_hw_start;
                    dfb_publish_per_risc_tc_init_done(per_risc_ptr);
                }
            }
            WAYPOINT("L8");
        }
    }

    // -----------------------------------------------------------------------
    // End merged loop (DM2-7 + TRISC)
    // -----------------------------------------------------------------------

    t_after_merged_loop = rdcycle();

    t_before_wait_all = rdcycle();
    wait_all_tcs_initialized(dfb_config_base, hart_u8);
    t_after_wait_all = rdcycle();
    WAYPOINT("L12");
    const uint32_t end_time = rdcycle();

    const uint32_t merged_sw = t_after_merged_loop - start_time;
    const uint32_t wait_all_cycles = t_after_wait_all - t_before_wait_all;

#ifdef COMPILE_FOR_TRISC
    const uint8_t timing_slot = dfb_init_timing_trisc_slot_index();
    const uint8_t timing_role = dfb::DFB_INIT_TIMING_ROLE_TRISC_LOCAL;
#else
    const uint8_t timing_slot = hart_u8;
    const uint8_t timing_role = dfb::DFB_INIT_TIMING_ROLE_DM_LOCAL;
#endif
    dfb_init_timing_write_slot(
        timing_slot,
        timing_role,
        end_time - start_time,
        merged_sw,
        total_remapper_spin,
        total_tc_hw,
        wait_all_cycles,
        total_tc_reset_hw,
        total_tc_capacity_hw,
        start_time,
        end_time);
}



// 1k cycles to program isrs in the worst case
// programming remapper is 37 cycles / remapper pair (how many in worst case??)
// enabling remapper is 4-100 cycles (why the range?)
// tc reset + set cap is 45 cycles / tc (worst case how many???)
