// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "overlay/interrupt_defines.h"
#include "overlay/meta/registers/overlay_reg_defines_debug.h"

namespace overlay::quasar {

// PLIC source 0 means "no interrupt", so the FDS threshold sources sit one above their interrupt ids.
constexpr uint32_t plic_source_base = DM_CORE_INT_ID_FDS_THRESHOLD_INTERRUPTS_0 + 1;

// A source is delivered only while its priority strictly exceeds the context threshold, so this
// pairing delivers every FDS source: the minimum priority that beats the allow-all threshold.
constexpr uint32_t plic_threshold_allow_all = 0;
constexpr uint32_t plic_fds_priority = 1;

// FDS group g arrives at the PLIC as source plic_source_base + g.
constexpr uint32_t plic_source_for_go_group(uint32_t group_id) { return plic_source_base + group_id; }
constexpr uint32_t go_group_from_plic_source(uint32_t source) { return source - plic_source_base; }

// Only DM0 takes FDS interrupts, so every access below targets PLIC context 0.
constexpr uint32_t plic_priority_register_address(uint32_t source) {
    return TT_CLUSTER_PLIC_REG_MAP_BASE_ADDR + source * static_cast<uint32_t>(sizeof(uint32_t));
}

constexpr uint32_t plic_enable_register_address(uint32_t enable_word) {
    return TT_CLUSTER_PLIC_CORE0_IE_0__REG_ADDR + enable_word * static_cast<uint32_t>(sizeof(uint32_t));
}

static_assert(plic_enable_register_address(1) == TT_CLUSTER_PLIC_CORE0_IE_1__REG_ADDR);
static_assert(plic_enable_register_address(2) == TT_CLUSTER_PLIC_CORE0_IE_2__REG_ADDR);

inline uint32_t plic_read32(uint32_t address) { return *reinterpret_cast<volatile uint32_t*>(address); }

inline void plic_write32(uint32_t address, uint32_t value) { *reinterpret_cast<volatile uint32_t*>(address) = value; }

inline uint32_t plic_claim() { return plic_read32(TT_CLUSTER_PLIC_CORE0_CLAIM_COMPLETE_REG_ADDR); }

inline void plic_complete(uint32_t source) { plic_write32(TT_CLUSTER_PLIC_CORE0_CLAIM_COMPLETE_REG_ADDR, source); }

inline void plic_enable_source(uint32_t source, bool enable) {
    const uint32_t enable_bit = uint32_t{1} << (source % 32);
    const uint32_t address = plic_enable_register_address(source / 32);
    const uint32_t current_value = plic_read32(address);
    plic_write32(address, enable ? (current_value | enable_bit) : (current_value & ~enable_bit));
}

inline void plic_set_priority(uint32_t source, uint32_t priority) {
    plic_write32(plic_priority_register_address(source), priority);
}

inline void plic_set_threshold(uint32_t threshold) {
    plic_write32(TT_CLUSTER_PLIC_CORE0_THRESHOLD_REG_ADDR, threshold);
}

inline void plic_drain_pendings() {
    while (const uint32_t source = plic_claim()) {
        plic_complete(source);
    }
}

}  // namespace overlay::quasar
