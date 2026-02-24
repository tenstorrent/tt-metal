// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_device_config.hpp"

#include <cmath>
#include <climits>
#include <map>
#include <string>

#include "common/stable_hash.hpp"
#include "core_descriptor.hpp"
#include "dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"

#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

JitDeviceConfig create_jit_device_config(ChipId device_id, uint8_t num_hw_cqs) {
    auto& ctx = MetalContext::instance();
    const auto& hal = ctx.hal();
    const auto& cluster = ctx.get_cluster();
    const auto& dispatch_core_config = ctx.get_dispatch_core_manager().get_dispatch_core_config();
    const metal_SocDescriptor& soc_d = cluster.get_soc_desc(device_id);

    const size_t num_dram_banks = static_cast<size_t>(soc_d.get_num_dram_views());
    // # of L1 banks needs to match allocator. For L1BankingAllocator this is the # of storage cores. TODO: when
    // allocator is pulled out of device, use it to get that info here.
    const size_t num_l1_banks = get_logical_compute_cores(device_id, num_hw_cqs, dispatch_core_config).size();

    auto pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    CoreCoord pcie_core = pcie_cores.empty() ? soc_d.grid_size : pcie_cores[0];

    JitDeviceConfig config;
    config.hal = &hal;
    config.arch = cluster.arch();
    config.num_dram_banks = num_dram_banks;
    config.num_l1_banks = num_l1_banks;
    config.pcie_core = pcie_core;
    config.harvesting_mask = cluster.get_harvesting_mask(device_id);
    config.dispatch_core_type = dispatch_core_config.get_dispatch_core_type();
    config.dispatch_core_axis = dispatch_core_config.get_dispatch_core_axis();
    config.coordinate_virtualization_enabled = hal.is_coordinate_virtualization_enabled();
    config.dispatch_message_addr = ctx.dispatch_mem_map().get_dispatch_message_addr_start();
    config.max_cbs = hal.get_arch_num_circular_buffers();
    config.num_hw_cqs = num_hw_cqs;
    config.routing_fw_enabled = cluster.is_base_routing_fw_enabled();
    config.profiler_dram_bank_size_per_risc_bytes = get_profiler_dram_bank_size_per_risc_bytes(ctx.rtoptions());
    return config;
}

std::map<std::string, std::string> initialize_device_kernel_defines(const JitDeviceConfig& config) {
    std::map<std::string, std::string> device_kernel_defines;

    bool is_dram_pow2 = ceil(log2(config.num_dram_banks)) == log2(config.num_dram_banks);
    bool is_l1_pow2 = ceil(log2(config.num_l1_banks)) == log2(config.num_l1_banks);

    device_kernel_defines.emplace("NUM_DRAM_BANKS", std::to_string(config.num_dram_banks));
    device_kernel_defines.emplace("NUM_L1_BANKS", std::to_string(config.num_l1_banks));

    if (is_dram_pow2) {
        device_kernel_defines.emplace(
            "LOG_BASE_2_OF_NUM_DRAM_BANKS", std::to_string(static_cast<size_t>(log2(config.num_dram_banks))));
    } else {
        device_kernel_defines.emplace("IS_NOT_POW2_NUM_DRAM_BANKS", "1");
    }
    if (is_l1_pow2) {
        device_kernel_defines.emplace(
            "LOG_BASE_2_OF_NUM_L1_BANKS", std::to_string(static_cast<size_t>(log2(config.num_l1_banks))));
    } else {
        device_kernel_defines.emplace("IS_NOT_POW2_NUM_L1_BANKS", "1");
    }

    device_kernel_defines.emplace("PCIE_NOC_X", std::to_string(config.pcie_core.x));
    device_kernel_defines.emplace("PCIE_NOC_Y", std::to_string(config.pcie_core.y));

    return device_kernel_defines;
}

uint64_t compute_build_key(const JitDeviceConfig& config, const llrt::RunTimeOptions& rtoptions) {
    // Collect all the parameters that affect the build configuration
    FNV1a hasher;

    hasher.update(static_cast<uint32_t>(config.dispatch_core_type));
    hasher.update(static_cast<uint32_t>(config.dispatch_core_axis));

    // Hash the number of hardware command queues
    hasher.update(static_cast<uint32_t>(config.num_hw_cqs));

    // Hash the harvesting configuration based on whether coordinate virtualization is enabled
    if (!config.coordinate_virtualization_enabled) {
        // Coordinate virtualization is not enabled. For a single program, its associated binaries will vary across
        // devices with different cores harvested.
        hasher.update(config.harvesting_mask);
    }

    hasher.update(rtoptions.get_compile_hash_string());

    return hasher.digest();
}

}  // namespace tt::tt_metal
