// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Accessors for the cluster's RISC-V platform-level interrupt controller (PLIC). It gates many
// interrupt sources onto one machine-external line per core context. A context enables the sources
// it wants and sets its threshold below their priority; the handler claims the highest-priority
// pending source and writes that id back to complete it.

#pragma once

#include <cstdint>

#include "overlay/meta/registers/overlay_reg_defines_debug.h"

namespace overlay::quasar {

// A source is delivered only while its priority strictly exceeds the context threshold, so a zero
// threshold delivers every source with nonzero priority.
constexpr uint32_t plic_threshold_allow_all = 0;

// Source enables are packed one bit per source across a run of 32-bit words.
constexpr uint32_t plic_bits_per_enable_word = 32;
constexpr uint32_t plic_num_enable_words = 3;

// Only DM0 takes PLIC interrupts, so every access below targets PLIC context 0.
constexpr uint32_t plic_priority_register_address(uint32_t source) {
    return TT_CLUSTER_PLIC_REG_MAP_BASE_ADDR + source * static_cast<uint32_t>(sizeof(uint32_t));
}

constexpr uint32_t plic_enable_register_address(uint32_t enable_word) {
    return TT_CLUSTER_PLIC_CORE0_IE_0__REG_ADDR + enable_word * static_cast<uint32_t>(sizeof(uint32_t));
}

inline uint32_t plic_read32(uint32_t address) { return *reinterpret_cast<volatile uint32_t*>(address); }

inline void plic_write32(uint32_t address, uint32_t value) { *reinterpret_cast<volatile uint32_t*>(address) = value; }

inline uint32_t plic_claim() { return plic_read32(TT_CLUSTER_PLIC_CORE0_CLAIM_COMPLETE_REG_ADDR); }

inline void plic_complete(uint32_t source) { plic_write32(TT_CLUSTER_PLIC_CORE0_CLAIM_COMPLETE_REG_ADDR, source); }

// The enable registers have no reset, so every word is written to leave no other source enabled.
inline void plic_enable_only_sources(uint32_t first_source, uint32_t last_source) {
    for (uint32_t enable_word = 0; enable_word < plic_num_enable_words; ++enable_word) {
        uint32_t enable_bits = 0;
        for (uint32_t bit = 0; bit < plic_bits_per_enable_word; ++bit) {
            const uint32_t source = enable_word * plic_bits_per_enable_word + bit;
            if (source >= first_source && source <= last_source) {
                enable_bits |= uint32_t{1} << bit;
            }
        }
        plic_write32(plic_enable_register_address(enable_word), enable_bits);
    }
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
