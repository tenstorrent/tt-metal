// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_config.h"
#include "noc/noc_parameters.h"
#include "llrt/hal.hpp"
#include "llrt/blackhole/bh_hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"

// Reserved DRAM addresses
// Host writes (4B value) to and reads from DRAM_BARRIER_BASE across all channels to ensure previous writes have been committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_BARRIER_SIZE = ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

namespace tt {

namespace tt_metal {

static inline int hv (enum HalL1MemAddrType v) {
    return static_cast<int>(v);
}

void Hal::initialize_bh() {
#if defined(ARCH_BLACKHOLE)
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));
    static_assert(static_cast<int>(HalProgrammableCoreType::ACTIVE_ETH) == static_cast<int>(ProgrammableCoreType::ACTIVE_ETH));
    static_assert(static_cast<int>(HalProgrammableCoreType::IDLE_ETH) == static_cast<int>(ProgrammableCoreType::IDLE_ETH));

    HalCoreInfoType tensix_mem_map = create_tensix_mem_map();
    this->core_info_.push_back(tensix_mem_map);

    HalCoreInfoType active_eth_mem_map = create_active_eth_mem_map();
    this->core_info_.push_back(active_eth_mem_map);

    HalCoreInfoType idle_eth_mem_map = create_idle_eth_mem_map();
    this->core_info_.push_back(idle_eth_mem_map);

    this->dram_bases_.resize(utils::underlying_type<HalDramMemAddrType>(HalDramMemAddrType::COUNT));
    this->dram_sizes_.resize(utils::underlying_type<HalDramMemAddrType>(HalDramMemAddrType::COUNT));
    this->dram_bases_[utils::underlying_type<HalDramMemAddrType>(HalDramMemAddrType::DRAM_BARRIER)] = DRAM_BARRIER_BASE;
    this->dram_sizes_[utils::underlying_type<HalDramMemAddrType>(HalDramMemAddrType::DRAM_BARRIER)] = DRAM_BARRIER_SIZE;

    this->mem_alignments_.resize(utils::underlying_type<HalMemType>(HalMemType::COUNT));
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::L1)] = L1_ALIGNMENT;
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::DRAM)] = DRAM_ALIGNMENT;
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::HOST)] = PCIE_ALIGNMENT;
#endif
}

}  // namespace tt_metal
}  // namespace tt
