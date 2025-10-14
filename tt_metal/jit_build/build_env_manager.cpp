// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_env_manager.hpp"

#include <limits.h>
#include <enchantum/enchantum.hpp>
#include <math.h>
#include <tracy/Tracy.hpp>
#include <bitset>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <variant>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "core_descriptor.hpp"
#include "dispatch_core_common.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/build.hpp"
#include "metal_soc_descriptor.h"
#include "dispatch/system_memory_manager.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

BuildEnvManager& BuildEnvManager::get_instance() {
    static BuildEnvManager instance;
    return instance;
}

BuildEnvManager::BuildEnvManager() {
    // Initialize build_state_indices_
    uint32_t index = 0;
    const auto& hal = MetalContext::instance().hal();
    uint32_t programmable_core_type_count = hal.get_programmable_core_type_count();
    build_state_indices_.resize(programmable_core_type_count);
    for (uint32_t programmable_core = 0; programmable_core < programmable_core_type_count; programmable_core++) {
        uint32_t processor_class_count =
            hal.get_processor_classes_count(hal.get_programmable_core_type(programmable_core));
        build_state_indices_[programmable_core].resize(processor_class_count);
        for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
            uint32_t processor_types_count = hal.get_processor_types_count(programmable_core, processor_class);
            build_state_indices_[programmable_core][processor_class] = {index, processor_types_count};
            index += processor_types_count;
        }
    }
}

namespace {

std::map<std::string, std::string> initialize_device_kernel_defines(chip_id_t device_id, uint8_t num_hw_cqs) {
    std::map<std::string, std::string> device_kernel_defines;

    const metal_SocDescriptor& soc_d = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
    const size_t num_dram_banks = static_cast<size_t>(soc_d.get_num_dram_views());
    // # of L1 banks needs to match allocator. For L1BankingAllocator this is the # of storage cores. TODO: when
    // allocator is pulled out of device, use it to get that info here.
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    const size_t num_compute_and_storage_cores =
        tt::get_logical_compute_cores(device_id, num_hw_cqs, dispatch_core_config).size();
    const size_t num_storage_only_cores =
        tt::get_logical_storage_cores(device_id, num_hw_cqs, dispatch_core_config).size();
    size_t num_banks_per_storage_core = 0;
    if (num_storage_only_cores > 0) {
        num_banks_per_storage_core =
            static_cast<size_t>(soc_d.worker_l1_size) /
            tt::get_storage_core_bank_size(device_id, num_hw_cqs, dispatch_core_config).value();
    }
    const size_t num_l1_banks = num_compute_and_storage_cores + (num_storage_only_cores * num_banks_per_storage_core);

    bool is_dram_pow2 = ceil(log2(num_dram_banks)) == log2(num_dram_banks);
    bool is_l1_pow2 = ceil(log2(num_l1_banks)) == log2(num_l1_banks);

    device_kernel_defines.emplace("NUM_DRAM_BANKS", std::to_string(num_dram_banks));
    device_kernel_defines.emplace("NUM_L1_BANKS", std::to_string(num_l1_banks));

    if (is_dram_pow2) {
        device_kernel_defines.emplace(
            "LOG_BASE_2_OF_NUM_DRAM_BANKS", std::to_string(static_cast<size_t>(log2(num_dram_banks))));
    } else {
        device_kernel_defines.emplace("IS_NOT_POW2_NUM_DRAM_BANKS", "1");
    }
    if (is_l1_pow2) {
        device_kernel_defines.emplace(
            "LOG_BASE_2_OF_NUM_L1_BANKS", std::to_string(static_cast<size_t>(log2(num_l1_banks))));
    } else {
        device_kernel_defines.emplace("IS_NOT_POW2_NUM_L1_BANKS", "1");
    }

    auto pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    CoreCoord pcie_core = pcie_cores.empty() ? soc_d.grid_size : pcie_cores[0];

    device_kernel_defines.emplace("PCIE_NOC_X", std::to_string(pcie_core.x));
    device_kernel_defines.emplace("PCIE_NOC_Y", std::to_string(pcie_core.y));

    return device_kernel_defines;
}

uint32_t compute_build_key(chip_id_t device_id, uint8_t num_hw_cqs) {
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();

    // Collect all the parameters that affect the build configuration
    std::size_t hash = 0;

    // Hash the dispatch core configuration
    std::hash<uint32_t> uint32_hasher;
    hash ^= uint32_hasher(static_cast<uint32_t>(dispatch_core_config.get_dispatch_core_type()));
    hash ^= uint32_hasher(static_cast<uint32_t>(dispatch_core_config.get_dispatch_core_axis())) << 1;

    // Hash the number of hardware command queues
    hash ^= uint32_hasher(static_cast<uint32_t>(num_hw_cqs)) << 2;

    // Hash the harvesting configuration based on whether coordinate virtualization is enabled
    if (not MetalContext::instance().hal().is_coordinate_virtualization_enabled()) {
        // Coordinate virtualization is not enabled. For a single program, its associated binaries will vary across
        // devices with different cores harvested.
        hash ^= uint32_hasher(tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id)) << 3;
    } else {
        // Coordinate Virtualization is enabled. Track only the number of harvested cores, instead of the exact
        // harvesting configuration (this is not needed).
        uint32_t harvested_core_count = std::bitset<32>(
            tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id)
        ).count();
        hash ^= uint32_hasher(harvested_core_count) << 4;
    }

    hash ^= std::hash<std::string>{}(MetalContext::instance().rtoptions().get_compile_hash_string());

    // Convert the hash to a 32-bit value
    return static_cast<uint32_t>(hash);
}

std::vector<JitBuildState> create_build_state(
    JitBuildEnv& build_env, chip_id_t /*device_id*/, uint8_t num_hw_cqs, bool is_fw) {
    // Get the dispatch message address for this device
    uint32_t dispatch_message_addr = MetalContext::instance().dispatch_mem_map().get_dispatch_message_addr_start();

    // Prepare the container for build states
    const auto& hal = MetalContext::instance().hal();
    uint32_t num_build_states = hal.get_total_num_risc_processors();
    std::vector<JitBuildState> build_states;
    build_states.reserve(num_build_states);

    // Loop through programmable core types and their processor classes/types.
    uint32_t programmable_core_type_count = hal.get_programmable_core_type_count();
    for (uint32_t programmable_core = 0; programmable_core < programmable_core_type_count; programmable_core++) {
        uint32_t processor_class_count =
            hal.get_processor_classes_count(hal.get_programmable_core_type(programmable_core));
        for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
            JitBuiltStateConfig config{
                .core_type = static_cast<HalProgrammableCoreType>(programmable_core),
                .processor_class = static_cast<HalProcessorClassType>(processor_class),
                .is_fw = is_fw,
                .dispatch_message_addr = dispatch_message_addr,
                .is_cooperative = hal.get_eth_fw_is_cooperative(),
            };
            uint32_t processor_types_count = hal.get_processor_types_count(programmable_core, processor_class);
            for (uint32_t processor_type = 0; processor_type < processor_types_count; processor_type++) {
                config.processor_id = processor_type;
                build_states.emplace_back(build_env, config);
            }
        }
    }
    TT_ASSERT(build_states.size() == num_build_states);
    return build_states;
}

}  // namespace

void BuildEnvManager::add_build_env(chip_id_t device_id, uint8_t num_hw_cqs) {
    const std::lock_guard<std::mutex> lock(this->lock);
    uint32_t build_key = compute_build_key(device_id, num_hw_cqs);
    auto device_kernel_defines = initialize_device_kernel_defines(device_id, num_hw_cqs);

    device_id_to_build_env_[device_id].build_key = build_key;
    device_id_to_build_env_[device_id].build_env.init(
        build_key, tt::tt_metal::MetalContext::instance().get_cluster().arch(), device_kernel_defines);
    device_id_to_build_env_[device_id].firmware_build_states =
        create_build_state(device_id_to_build_env_[device_id].build_env, device_id, num_hw_cqs, true);
    device_id_to_build_env_[device_id].kernel_build_states =
        create_build_state(device_id_to_build_env_[device_id].build_env, device_id, num_hw_cqs, false);
}

const DeviceBuildEnv& BuildEnvManager::get_device_build_env(chip_id_t device_id) {
    const std::lock_guard<std::mutex> lock(this->lock);
    TT_ASSERT(device_id_to_build_env_.count(device_id) != 0, "Couldn't find build env for device {}.", device_id);
    return device_id_to_build_env_[device_id];
}

const JitBuildState& BuildEnvManager::get_firmware_build_state(
    chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id) {
    uint32_t state_idx = get_build_index_and_state_count(programmable_core, processor_class).first + processor_id;
    return get_device_build_env(device_id).firmware_build_states[state_idx];
}

const JitBuildState& BuildEnvManager::get_kernel_build_state(
    chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id) {
    uint32_t state_idx = get_build_index_and_state_count(programmable_core, processor_class).first + processor_id;
    return get_device_build_env(device_id).kernel_build_states[state_idx];
}

JitBuildStateSubset BuildEnvManager::get_kernel_build_states(
    chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class) {
    auto [b_id, count] = get_build_index_and_state_count(programmable_core, processor_class);
    auto& kernel_build_states = get_device_build_env(device_id).kernel_build_states;
    return {kernel_build_states.begin() + b_id, count};
}

BuildIndexAndTypeCount BuildEnvManager::get_build_index_and_state_count(
    uint32_t programmable_core, uint32_t processor_class) {
    const std::lock_guard<std::mutex> lock(this->lock);
    TT_ASSERT(
        programmable_core < build_state_indices_.size(),
        "Programmable core type {} is not included in the FW or Kernel build state",
        programmable_core);
    TT_ASSERT(
        processor_class < build_state_indices_[programmable_core].size(),
        "Processor class type {} is not included in the FW or Kernel build state",
        processor_class);
    return build_state_indices_[programmable_core][processor_class];
}

void BuildEnvManager::build_firmware(chip_id_t device_id) {
    ZoneScoped;
    jit_build_subset(get_device_build_env(device_id).firmware_build_states, nullptr);
}

}  // namespace tt::tt_metal
