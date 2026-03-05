// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_device_config.hpp"
#include "jit_build/build.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {

namespace {

// FIXME: use DispatchMemMap::get_dispatch_message_addr_start() once it is decoupled from global MetalContext
uint32_t dispatch_message_addr(const Hal& hal, DispatchCoreType dispatch_core_type) {
    uint32_t stream_index = 0;
    if (dispatch_core_type == DispatchCoreType::WORKER) {
        // There are 64 streams. CBs use entries 8-39.
        stream_index = 48u;
    } else if (dispatch_core_type == DispatchCoreType::ETH) {
        // There are 32 streams.
        stream_index = 16u;
    } else {
        TT_THROW("get_dispatch_starting_stream_index not implemented for core type");
    }

    return hal.get_noc_overlay_start_addr() + (hal.get_noc_stream_reg_space_size() * stream_index) +
           (hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index() * sizeof(uint32_t));
}

void copy_firmware_to_precompiled_dir(
    const std::string& firmware_out_path, const std::string& precompiled_firmware_dir) {
    namespace fs = std::filesystem;
    for (const auto& entry : fs::recursive_directory_iterator(firmware_out_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".elf") {
            auto relative = fs::relative(entry.path(), firmware_out_path);
            auto dest = fs::path(precompiled_firmware_dir) / relative;
            fs::create_directories(dest.parent_path());
            fs::copy_file(entry.path(), dest, fs::copy_options::overwrite_existing);
        }
    }
}

}  // namespace

void enumerate_jit_device_configs(
    tt::ARCH arch, std::string& core_descriptor_path, const std::function<void(const JitDeviceConfig&)>& callback) {
    /* FIXME: need to figure out these values for each product type */
    constexpr bool is_base_routing_fw_enabled = false;
    constexpr bool enable_2_erisc_mode = false;
    constexpr uint32_t profiler_dram_bank_size_per_risc_bytes = 0;

    /* FIXME: need to account for dram harvesting */
    static const std::unordered_map<tt::ARCH, uint32_t> num_dram_banks_map = {
        {tt::ARCH::WORMHOLE_B0, 12},
        {tt::ARCH::BLACKHOLE, 8},
    };
    const uint32_t num_dram_banks = num_dram_banks_map.at(arch);

    static const std::unordered_map<tt::ARCH, CoreCoord> pcie_core_map = {
        {tt::ARCH::WORMHOLE_B0, {0, 3}},
        {tt::ARCH::BLACKHOLE, {19, 24}},
    };
    const CoreCoord pcie_core = pcie_core_map.at(arch);

    tt::tt_metal::Hal hal(
        arch, is_base_routing_fw_enabled, enable_2_erisc_mode, profiler_dram_bank_size_per_risc_bytes);
    YAML::Node core_descriptor_yaml = YAML::LoadFile(core_descriptor_path);
    for (const auto& product : core_descriptor_yaml) {
        const std::string product_name = product.first.as<std::string>();
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

                // TODO: handle tg_compute_with_storage_grid_range

                // Excerpt from core_descriptor.cpp
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

                JitDeviceConfig jit_device_config = {
                    .hal = &hal,
                    .arch = arch,
                    .num_dram_banks = num_dram_banks,
                    .num_l1_banks = num_l1_banks,
                    .pcie_core = pcie_core,
                    // We only precompile for coordinate_virtualization_enabled = true, so harvesting_mask has no effect
                    // on compilation
                    .harvesting_mask = 0,
                    .dispatch_core_type = dispatch_core_type,
                    .dispatch_core_axis = dispatch_core_axis,
                    .coordinate_virtualization_enabled = true,
                    .dispatch_message_addr = dispatch_message_addr(hal, dispatch_core_type),
                    .max_cbs = hal.get_arch_num_circular_buffers(),
                    .num_hw_cqs = num_hw_cqs,
                    .routing_fw_enabled = false,  // will enumerate
                    // We only precompile for profiler disabled, so profiler_dram_bank_size_per_risc_bytes has no effect
                    // on compilation
                    .profiler_dram_bank_size_per_risc_bytes = profiler_dram_bank_size_per_risc_bytes};
                callback(jit_device_config);
                // routing_fw_enabled is set when is_galaxy_cluster() is true, but this
                // symbol has effect only for Wormhole.
                // See get_core_descriptor_config in core_descriptor.cpp: galaxy_cluster could
                // also end up with product name "nebula_x1".
                if (product_name == "galaxy" || product_name == "nebula_x1") {
                    jit_device_config.routing_fw_enabled = true;
                    callback(jit_device_config);
                }
            }
        }
    }
}

void precompile_for_config(
    const tt::tt_metal::JitDeviceConfig& jit_device_config, const tt::llrt::RunTimeOptions& rtoptions) {
    BuildEnvManager build_env_manager(*jit_device_config.hal);
    build_env_manager.add_build_env(0, jit_device_config, rtoptions, false);
    build_env_manager.build_firmware(0);

    std::cout << jit_device_config << std::endl;

    auto dev_build_env = build_env_manager.get_device_build_env(0);
    auto build_key = dev_build_env.build_key();
    auto firmware_out_path = dev_build_env.build_env.get_out_firmware_root_path();
    auto precompiled_firmware_dir = rtoptions.get_root_dir() + "pre-compiled/" + std::to_string(build_key) + "/";

    std::cout << "cp -r " << firmware_out_path << " " << precompiled_firmware_dir << std::endl;
    copy_firmware_to_precompiled_dir(firmware_out_path, precompiled_firmware_dir);
}

}  // namespace tt::tt_metal

namespace {

struct PrecompileConfig {
    tt::ARCH arch;
    std::string core_descriptor_name;
};

const auto supported_configs = std::to_array<PrecompileConfig>({
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_eth_dispatch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_fabric_mux.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_eth_dispatch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_fabric_mux.yaml"},
});

}  // namespace

int main() {
    tt::llrt::RunTimeOptions rtoptions;
    const std::string core_descriptors_dir = rtoptions.get_root_dir() + "tt_metal/core_descriptors/";
    for (const auto& [arch, core_descriptor_name] : supported_configs) {
        std::string full_core_descriptor_path = core_descriptors_dir + core_descriptor_name;
        tt::tt_metal::enumerate_jit_device_configs(
            arch, full_core_descriptor_path, [&rtoptions](const tt::tt_metal::JitDeviceConfig& jit_device_config) {
                tt::tt_metal::precompile_for_config(jit_device_config, rtoptions);
            });
    }
    return 0;
}
