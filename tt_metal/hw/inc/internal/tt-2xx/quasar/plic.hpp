// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "overlay/meta/registers/overlay_reg_defines_debug.h"

namespace overlay::quasar {

constexpr uint32_t plic_source_base = 16;

constexpr uint32_t plic_priority_register_address(uint32_t source) {
    return TT_CLUSTER_PLIC_REG_MAP_BASE_ADDR + source * static_cast<uint32_t>(sizeof(uint32_t));
}

constexpr uint32_t plic_enable_register_addresses[8][3] = {
    {TT_CLUSTER_PLIC_CORE0_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE0_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE0_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE1_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE1_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE1_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE2_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE2_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE2_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE3_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE3_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE3_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE4_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE4_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE4_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE5_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE5_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE5_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE6_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE6_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE6_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE7_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE7_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE7_IE_2__REG_ADDR},
};

constexpr uint32_t plic_threshold_register_addresses[8] = {
    TT_CLUSTER_PLIC_CORE0_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE1_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE2_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE3_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE4_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE5_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE6_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE7_THRESHOLD_REG_ADDR,
};

constexpr uint32_t plic_claim_complete_register_addresses[8] = {
    TT_CLUSTER_PLIC_CORE0_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE1_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE2_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE3_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE4_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE5_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE6_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE7_CLAIM_COMPLETE_REG_ADDR,
};

inline uint32_t plic_read32(uint32_t address) { return *reinterpret_cast<volatile uint32_t*>(address); }

inline void plic_write32(uint32_t address, uint32_t value) { *reinterpret_cast<volatile uint32_t*>(address) = value; }

inline uint32_t plic_claim(uint32_t context) { return plic_read32(plic_claim_complete_register_addresses[context]); }

inline void plic_complete(uint32_t context, uint32_t source) {
    plic_write32(plic_claim_complete_register_addresses[context], source);
}

inline void plic_enable_source(uint32_t context, uint32_t source, bool enable) {
    const uint32_t enable_word = source / 32;
    const uint32_t enable_bit = uint32_t{1} << (source % 32);
    const uint32_t address = plic_enable_register_addresses[context][enable_word];
    const uint32_t current_value = plic_read32(address);
    plic_write32(address, enable ? (current_value | enable_bit) : (current_value & ~enable_bit));
}

inline void plic_set_priority(uint32_t source, uint32_t priority) {
    plic_write32(plic_priority_register_address(source), priority);
}

inline void plic_set_threshold(uint32_t context, uint32_t threshold) {
    plic_write32(plic_threshold_register_addresses[context], threshold);
}

inline void plic_drain_pendings(uint32_t context) {
    while (const uint32_t source = plic_claim(context)) {
        plic_complete(context, source);
    }
}

}  // namespace overlay::quasar
