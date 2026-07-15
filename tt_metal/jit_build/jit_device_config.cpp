// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_device_config.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tt_stl/assert.hpp>

#include "context/metal_env_accessor.hpp"
#include "core_descriptor.hpp"
#include "dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"

#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

JitDeviceConfig create_jit_device_config(ChipId device_id, uint8_t num_hw_cqs, ContextId context_id) {
    // Need both runtime state and hardware query
    auto& ctx = MetalContext::instance(context_id);
    auto& env = MetalEnvAccessor(ctx.get_env()).impl();
    const auto& hal = env.get_hal();
    const auto& cluster = env.get_cluster();
    const auto& dispatch_core_config = ctx.get_dispatch_core_manager().get_dispatch_core_config();
    const metal_SocDescriptor& soc_d = cluster.get_soc_desc(device_id);

    const size_t num_dram_banks = static_cast<size_t>(soc_d.get_num_dram_views());
    // # of L1 banks needs to match allocator. For L1BankingAllocator this is the # of storage cores. TODO: when
    // allocator is pulled out of device, use it to get that info here.
    const size_t num_l1_banks = get_logical_compute_cores(env, device_id, num_hw_cqs, dispatch_core_config).size();

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

namespace {

std::vector<bool> routing_fw_configs_for_product(tt::ARCH arch, std::string_view product_name) {
    // TODO(#39462): Hand-curated precompile coverage. A principled derivation would require a single source of truth
    // mapping (cluster_type, board_type, tensix_harvest) tuples to (product_name, routing_fw_enabled);
    // product_name is a lossy projection of that space and cannot be inverted cleanly today.
    if (arch == tt::ARCH::WORMHOLE_B0) {
        if (product_name == "galaxy" || product_name == "nebula_x1") {
            return {false, true};
        }
        return {true};
    }
    return {false};
}

std::vector<uint32_t> dram_harvesting_configs_for_product(tt::ARCH arch, std::string_view /*product_name*/) {
    // TODO(#39462): Hand-curated precompile coverage. DRAM bank count is a compile input (it feeds the
    // kernel/firmware build key), so a config must be precompiled for every DRAM-harvest state a device may
    // report at runtime. On Blackhole DRAM harvesting is independent of tensix (column) harvesting, so any
    // tensix-harvest product can appear with 0 or 1 DRAM banks harvested; emit both variants for every product.
    // Wormhole has no DRAM harvesting, so only the unharvested variant is needed.
    if (arch == tt::ARCH::BLACKHOLE) {
        return {0, 1};
    }
    return {0};
}

CoreCoord pcie_core_for_arch(tt::ARCH arch) {
    // FIXME: Replace with an offline query equivalent to
    // metal_SocDescriptor::get_cores(CoreType::PCIE, CoordSystem::TRANSLATED).
    static const std::unordered_map<tt::ARCH, CoreCoord> pcie_core_map = {
        {tt::ARCH::WORMHOLE_B0, {0, 3}},
        {tt::ARCH::BLACKHOLE, {19, 24}},
    };
    return pcie_core_map.at(arch);
}

// TODO: use DispatchMemMap::get_dispatch_message_addr_start() as the single
// source of truth. Blocked today because constructing a DispatchMemMap for one
// address requires the full DispatchSettings + L1 layout machinery.
uint32_t dispatch_message_addr(const Hal& hal, DispatchCoreType dispatch_core_type) {
    uint32_t stream_index = 0;
    if (dispatch_core_type == DispatchCoreType::WORKER) {
        // There are 64 streams. CBs use entries 8-39.
        stream_index = 48u;
    } else if (dispatch_core_type == DispatchCoreType::ETH) {
        // There are 32 streams.
        stream_index = 16u;
    } else {
        TT_THROW("dispatch_message_addr not implemented for core type");
    }

    return hal.get_noc_overlay_start_addr() + (hal.get_noc_stream_reg_space_size() * stream_index) +
           (hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index() * sizeof(uint32_t));
}

}  // namespace

void enumerate_jit_device_configs(
    tt::ARCH arch,
    const std::string& core_descriptor_path,
    const std::string& soc_descriptor_path,
    const std::function<void(const JitDeviceConfig&)>& callback) {
    // NOTE: precompile policy, not missing wiring. Precompile intentionally targets:
    //   - profiler disabled (profiler_dram_bank_size_per_risc_bytes = 0),
    //   - 2-ERISC mode enabled on Blackhole only,
    //   - DRAM-backed command queue disabled.
    // These choices define the set of build keys the firmware precompile step produces.
    constexpr uint32_t profiler_dram_bank_size_per_risc_bytes = 0;
    const bool enable_2_erisc_mode = (arch == tt::ARCH::BLACKHOLE);
    const bool enable_dram_backed_cq = false;

    const CoreCoord pcie_core = pcie_core_for_arch(arch);
    const size_t base_num_dram_banks = YAML::LoadFile(soc_descriptor_path)["dram_views"].size();

    YAML::Node core_descriptor_yaml = YAML::LoadFile(core_descriptor_path);
    for (const auto& product : core_descriptor_yaml) {
        const std::string product_name = product.first.as<std::string>();
        const auto routing_fw_enabled_cfgs = routing_fw_configs_for_product(arch, product_name);
        const auto dram_harvesting_cfgs = dram_harvesting_configs_for_product(arch, product_name);
        for (const auto& axis_config : product.second) {
            std::string dispatch_core_axis_str = axis_config.first.as<std::string>();
            tt_metal::DispatchCoreAxis dispatch_core_axis;
            if (dispatch_core_axis_str == "row") {
                dispatch_core_axis = tt_metal::DispatchCoreAxis::ROW;
            } else if (dispatch_core_axis_str == "col") {
                dispatch_core_axis = tt_metal::DispatchCoreAxis::COL;
            } else {
                TT_THROW("Invalid dispatch core axis: {}", dispatch_core_axis_str);
            }
            for (const auto& config_node : axis_config.second) {
                auto num_hw_cqs = config_node.first.as<uint8_t>();
                const auto& config = config_node.second;

                // NOTE: precompile deliberately does not produce TG-specific binaries, so we
                // read only `compute_with_storage_grid_range` and ignore `tg_compute_with_storage_grid_range`
                // even though the runtime path in core_descriptor.cpp honors it for nebula_x1 on galaxy clusters.
                auto compute_with_storage_start = config["compute_with_storage_grid_range"]["start"];
                auto compute_with_storage_end = config["compute_with_storage_grid_range"]["end"];
                size_t end_x = compute_with_storage_end[0].as<size_t>();
                size_t end_y = compute_with_storage_end[1].as<size_t>();
                size_t start_x = compute_with_storage_start[0].as<size_t>();
                size_t start_y = compute_with_storage_start[1].as<size_t>();
                auto compute_grid_size = CoreCoord((end_x - start_x) + 1, (end_y - start_y) + 1);
                auto num_l1_banks = compute_grid_size.x * compute_grid_size.y;

                auto dispatch_core_type_str = config["dispatch_core_type"].as<std::string>();
                tt_metal::DispatchCoreType dispatch_core_type;
                if (dispatch_core_type_str == "tensix") {
                    dispatch_core_type = tt_metal::DispatchCoreType::WORKER;
                } else if (dispatch_core_type_str == "ethernet") {
                    dispatch_core_type = tt_metal::DispatchCoreType::ETH;
                } else {
                    TT_THROW("Invalid dispatch core type: {}", dispatch_core_type_str);
                }

                // now enumerate all possible combinations of routing_fw_enabled and dram_harvesting_cfg
                for (const auto& routing_fw_enabled : routing_fw_enabled_cfgs) {
                    for (const auto& dram_harvesting_count : dram_harvesting_cfgs) {
                        TT_FATAL(
                            dram_harvesting_count <= base_num_dram_banks,
                            "DRAM harvesting count {} exceeds base DRAM bank count {}",
                            dram_harvesting_count,
                            base_num_dram_banks);
                        const size_t num_dram_banks = base_num_dram_banks - dram_harvesting_count;
                        tt::tt_metal::Hal hal(
                            arch,
                            routing_fw_enabled,
                            enable_2_erisc_mode,
                            profiler_dram_bank_size_per_risc_bytes,
                            enable_dram_backed_cq,
                            /*is_simulator=*/false,
                            /*enable_blackhole_dram_programmable_cores=*/true);
                        JitDeviceConfig jit_device_config = {
                            .hal = &hal,
                            .arch = arch,
                            .num_dram_banks = num_dram_banks,
                            .num_l1_banks = num_l1_banks,
                            .pcie_core = pcie_core,
                            // We only precompile with coordinate_virtualization_enabled = true, so harvesting_mask
                            // has no effect on compilation
                            .harvesting_mask = 0,
                            .dispatch_core_type = dispatch_core_type,
                            .dispatch_core_axis = dispatch_core_axis,
                            .coordinate_virtualization_enabled = true,
                            .dispatch_message_addr = dispatch_message_addr(hal, dispatch_core_type),
                            .max_cbs = hal.get_arch_num_circular_buffers(),
                            .num_hw_cqs = num_hw_cqs,
                            .routing_fw_enabled = routing_fw_enabled,
                            // We only precompile with profiler disabled, so profiler_dram_bank_size_per_risc_bytes
                            // has no effect on compilation
                            .profiler_dram_bank_size_per_risc_bytes = profiler_dram_bank_size_per_risc_bytes,
                        };
                        callback(jit_device_config);
                    }
                }
            }
        }
    }
}

namespace {

// Supported (arch, core_descriptor, soc_descriptor) tuples for ahead-of-time
// (offline) compilation. Used by both firmware precompile and offline kernel
// compile to enumerate JitDeviceConfig values without a live device.
struct OfflineCompileDeviceConfig {
    tt::ARCH arch;
    std::string_view core_descriptor_name;
    std::string_view soc_descriptor_name;
};

constexpr auto offline_compile_device_configs = std::to_array<OfflineCompileDeviceConfig>({
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch.yaml", "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_eth_dispatch.yaml", "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_fabric_mux.yaml", "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch.yaml", "blackhole_140_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_eth_dispatch.yaml", "blackhole_140_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_fabric_mux.yaml", "blackhole_140_arch.yaml"},
});

}  // namespace

void enumerate_offline_compile_device_configs(
    const tt::llrt::RunTimeOptions& rtoptions, const std::function<void(const JitDeviceConfig&)>& callback) {
    const std::string core_descriptors_dir = rtoptions.get_root_dir() + "tt_metal/core_descriptors/";
    const std::string soc_descriptors_dir = rtoptions.get_root_dir() + "tt_metal/soc_descriptors/";
    for (const auto& [arch, core_descriptor_name, soc_descriptor_name] : offline_compile_device_configs) {
        enumerate_jit_device_configs(
            arch,
            core_descriptors_dir + std::string(core_descriptor_name),
            soc_descriptors_dir + std::string(soc_descriptor_name),
            callback);
    }
}

}  // namespace tt::tt_metal
