// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "overlay/meta/registers/overlay_reg_defines_debug.h"

namespace overlay::quasar {

constexpr std::uint32_t plic_source_base = 16;

constexpr std::uint32_t plic_priority_register_address(std::uint32_t source) {
    return TT_CLUSTER_PLIC_REG_MAP_BASE_ADDR + source * static_cast<std::uint32_t>(sizeof(std::uint32_t));
}

constexpr std::uint32_t plic_enable_register_addresses[8][3] = {
    {TT_CLUSTER_PLIC_CORE0_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE0_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE0_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE1_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE1_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE1_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE2_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE2_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE2_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE3_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE3_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE3_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE4_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE4_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE4_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE5_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE5_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE5_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE6_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE6_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE6_IE_2__REG_ADDR},
    {TT_CLUSTER_PLIC_CORE7_IE_0__REG_ADDR, TT_CLUSTER_PLIC_CORE7_IE_1__REG_ADDR, TT_CLUSTER_PLIC_CORE7_IE_2__REG_ADDR},
};

constexpr std::uint32_t plic_threshold_register_addresses[8] = {
    TT_CLUSTER_PLIC_CORE0_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE1_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE2_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE3_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE4_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE5_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE6_THRESHOLD_REG_ADDR,
    TT_CLUSTER_PLIC_CORE7_THRESHOLD_REG_ADDR,
};

constexpr std::uint32_t plic_claim_complete_register_addresses[8] = {
    TT_CLUSTER_PLIC_CORE0_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE1_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE2_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE3_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE4_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE5_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE6_CLAIM_COMPLETE_REG_ADDR,
    TT_CLUSTER_PLIC_CORE7_CLAIM_COMPLETE_REG_ADDR,
};

inline std::uint32_t plic_read32(std::uint32_t address) { return *reinterpret_cast<volatile std::uint32_t*>(address); }

inline void plic_write32(std::uint32_t address, std::uint32_t value) {
    *reinterpret_cast<volatile std::uint32_t*>(address) = value;
}

inline std::uint32_t plic_claim(std::uint32_t context) {
    return plic_read32(plic_claim_complete_register_addresses[context]);
}

inline void plic_complete(std::uint32_t context, std::uint32_t source) {
    plic_write32(plic_claim_complete_register_addresses[context], source);
}

inline void plic_enable_source(std::uint32_t context, std::uint32_t source, bool enable) {
    const std::uint32_t enable_word = source / 32;
    const std::uint32_t enable_bit = std::uint32_t{1} << (source % 32);
    const std::uint32_t address = plic_enable_register_addresses[context][enable_word];
    const std::uint32_t current_value = plic_read32(address);
    plic_write32(address, enable ? (current_value | enable_bit) : (current_value & ~enable_bit));
}

inline void plic_set_priority(std::uint32_t source, std::uint32_t priority) {
    plic_write32(plic_priority_register_address(source), priority);
}

inline void plic_set_threshold(std::uint32_t context, std::uint32_t threshold) {
    plic_write32(plic_threshold_register_addresses[context], threshold);
}

inline void plic_drain_pendings(std::uint32_t context) {
    while (const std::uint32_t source = plic_claim(context)) {
        plic_complete(context, source);
    }
}

}  // namespace overlay::quasar
