// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/dram/mimir_snoop_ctl.hpp"

#include <cstddef>
#include <exception>

#include <gtest/gtest.h>

#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

// dev_mem_map.h defines CCE_RESET_VECTOR_BASE as a macro; read UMD's constant of the same name with the macro hidden.
#pragma push_macro("CCE_RESET_VECTOR_BASE")
#undef CCE_RESET_VECTOR_BASE
#include "umd/device/arch/grendel_implementation.hpp"
#include "umd/device/coordinates/grendel_noc_address_resolver.hpp"
namespace tt::tt_metal::mimir_test::snoop_ctl_impl {
// SMC view of CCE0's tt_cluster_ctrl register plane: its first register is RESET_VECTOR_0 (CCE-local 0x03000000).
inline constexpr uint64_t kClusterCtrlBaseAddress = tt::umd::grendel::CCE_RESET_VECTOR_BASE;
}  // namespace tt::tt_metal::mimir_test::snoop_ctl_impl
#pragma pop_macro("CCE_RESET_VECTOR_BASE")

namespace tt::tt_metal::mimir_test {

namespace snoop_ctl_impl {

inline constexpr uint64_t kRemapBaseAddress = tt::umd::grendel::CCE_DMRISC_REMAP_BASE;
inline constexpr uint64_t kRemapEntryStride = tt::umd::grendel::CCE_DMRISC_REMAP_ENTRY_STRIDE;
// Entry 1 is the local Mimir's SRAM window (UMD cce_dmrisc_remap_entries).
inline constexpr uint64_t kSramRemapEntry = 1;
inline constexpr uint64_t kRemapAttrsOffset = 0x18;
// NOC_SNOOP_TL_MASTER_CFG offset in tt_cluster_ctrl (tt-2.0.0 tt_cluster_ctrl_reg.h, CCE-local 0x03000124).
inline constexpr uint64_t kSnoopMasterConfigOffset = 0x124;

// Channel 0: the SMC core of the CCE's Mimir already selects the Mimir, so chiplet-local windows.
uint64_t cce_register_address(uint32_t cce_index, uint64_t register_address) {
    return tt::umd::mimir_local_address_windows(configured_soc_desc())
        .cce_control_address(0, location_for_cce(cce_index), register_address);
}

uint64_t sram_remap_attrs_address(uint32_t cce_index) {
    return cce_register_address(cce_index, kRemapBaseAddress + kSramRemapEntry * kRemapEntryStride + kRemapAttrsOffset);
}

uint64_t snoop_master_config_address(uint32_t cce_index) {
    return cce_register_address(cce_index, kClusterCtrlBaseAddress + kSnoopMasterConfigOffset);
}

tt_cxy_pair smc_target(const Cluster& cluster, ChipId chip, uint32_t cce_index) {
    const CoreCoord smc_core = mimir_smc_core(cluster.get_soc_desc(chip), mimir_index_for_cce(cce_index));
    return tt_cxy_pair(static_cast<std::size_t>(chip), smc_core.x, smc_core.y);
}

uint32_t read_word(const Cluster& cluster, ChipId chip, uint32_t cce_index, uint64_t address) {
    uint32_t word = 0;
    cluster.read_reg(&word, smc_target(cluster, chip, cce_index), address);
    return word;
}

void write_word(const Cluster& cluster, ChipId chip, uint32_t cce_index, uint64_t address, uint32_t word) {
    cluster.write_reg(&word, smc_target(cluster, chip, cce_index), address);
}

// Low word of RegionAttrs of `cce_index`'s DMRISC remap entry 1.
uint32_t read_sram_remap_attrs(const Cluster& cluster, ChipId chip, uint32_t cce_index) {
    return read_word(cluster, chip, cce_index, sram_remap_attrs_address(cce_index));
}

}  // namespace snoop_ctl_impl

uint32_t read_snoop_master_config(const Cluster& cluster, ChipId chip, uint32_t cce_index) {
    return snoop_ctl_impl::read_word(cluster, chip, cce_index, snoop_ctl_impl::snoop_master_config_address(cce_index));
}

ScopedSramRemapSnoop::ScopedSramRemapSnoop(const Cluster& cluster, ChipId chip, uint32_t cce_index, bool snoop) :
    cluster_(cluster), chip_(chip), cce_index_(cce_index) {
    original_ = snoop_ctl_impl::read_sram_remap_attrs(cluster_, chip_, cce_index_);
    const uint32_t word = snoop ? (original_ | kRemapCmdSnoopBit) : (original_ & ~kRemapCmdSnoopBit);
    snoop_ctl_impl::write_word(cluster_, chip_, cce_index_, snoop_ctl_impl::sram_remap_attrs_address(cce_index_), word);
    applied_ = snoop_ctl_impl::read_sram_remap_attrs(cluster_, chip_, cce_index_);
}

ScopedSramRemapSnoop::~ScopedSramRemapSnoop() {
    try {
        snoop_ctl_impl::write_word(
            cluster_, chip_, cce_index_, snoop_ctl_impl::sram_remap_attrs_address(cce_index_), original_);
    } catch (const std::exception& error) {
        ADD_FAILURE() << "restoring CCE " << cce_index_ << " remap entry 1 attrs threw: " << error.what();
    }
}

ScopedSnoopMasterConfig::ScopedSnoopMasterConfig(
    const Cluster& cluster, ChipId chip, uint32_t cce_index, uint32_t config) :
    cluster_(cluster), chip_(chip), cce_index_(cce_index) {
    original_ = read_snoop_master_config(cluster_, chip_, cce_index_);
    snoop_ctl_impl::write_word(
        cluster_, chip_, cce_index_, snoop_ctl_impl::snoop_master_config_address(cce_index_), config);
    applied_ = read_snoop_master_config(cluster_, chip_, cce_index_);
}

ScopedSnoopMasterConfig::~ScopedSnoopMasterConfig() {
    try {
        snoop_ctl_impl::write_word(
            cluster_, chip_, cce_index_, snoop_ctl_impl::snoop_master_config_address(cce_index_), original_);
    } catch (const std::exception& error) {
        ADD_FAILURE() << "restoring CCE " << cce_index_ << " NOC_SNOOP_TL_MASTER_CFG threw: " << error.what();
    }
}

}  // namespace tt::tt_metal::mimir_test
