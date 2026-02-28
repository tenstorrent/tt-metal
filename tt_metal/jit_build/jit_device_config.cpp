// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_device_config.hpp"

#include "core_descriptor.hpp"
#include "dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
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

    return {
        .hal = &hal,
        .arch = cluster.arch(),
        .num_dram_banks = num_dram_banks,
        .num_l1_banks = num_l1_banks,
        .pcie_core = pcie_core,
        .harvesting_mask = cluster.get_harvesting_mask(device_id),
        .dispatch_core_type = dispatch_core_config.get_dispatch_core_type(),
        .dispatch_core_axis = dispatch_core_config.get_dispatch_core_axis(),
        .coordinate_virtualization_enabled = hal.is_coordinate_virtualization_enabled(),
        .dispatch_message_addr = ctx.dispatch_mem_map().get_dispatch_message_addr_start(),
        .max_cbs = hal.get_arch_num_circular_buffers(),
        .num_hw_cqs = num_hw_cqs,
        .routing_fw_enabled = cluster.is_base_routing_fw_enabled(),
        .profiler_dram_bank_size_per_risc_bytes = get_profiler_dram_bank_size_per_risc_bytes(ctx.rtoptions())};
}

}  // namespace tt::tt_metal
