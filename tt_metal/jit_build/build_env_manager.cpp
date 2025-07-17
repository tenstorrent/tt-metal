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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "assert.hpp"
#include "core_coord.hpp"
#include "core_descriptor.hpp"
#include "dispatch_core_common.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/build.hpp"
#include "metal_soc_descriptor.h"
#include "dispatch/system_memory_manager.hpp"
#include <umd/device/tt_core_coordinates.h>

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
        uint32_t processor_class_count = hal.get_processor_classes_count(programmable_core);
        build_state_indices_[programmable_core].resize(processor_class_count);
        for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
            uint32_t processor_types_count = hal.get_processor_types_count(programmable_core, processor_class);
            build_state_indices_[programmable_core][processor_class] = {index, processor_types_count};
            index += processor_types_count;
        }
    }
}

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
    const size_t num_l1_banks = num_compute_and_storage_cores + num_storage_only_cores * num_banks_per_storage_core;

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
    uint32_t build_key = 0;
    constexpr uint32_t harvesting_map_bits = 12;
    constexpr uint32_t num_hw_cq_bits = 8;
    constexpr uint32_t dispatch_core_axis_bits = 1;
    constexpr uint32_t dispatch_core_type_bits = 1;
    static_assert(dispatch_core_manager::MAX_NUM_HW_CQS <= (1 << num_hw_cq_bits));
    static_assert(static_cast<uint32_t>(DispatchCoreAxis::COUNT) <= (1 << dispatch_core_axis_bits));
    static_assert(static_cast<uint32_t>(DispatchCoreType::COUNT) <= (1 << dispatch_core_type_bits));
    static_assert(
        harvesting_map_bits + num_hw_cq_bits + dispatch_core_axis_bits + dispatch_core_type_bits <=
        sizeof(build_key) * CHAR_BIT);

    // num_hw_cqs, dispatch_core_axis, dispatch_core_type all change the number of banks, so need to be part of the
    // build key since we have defines based on number of banks.
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    build_key = (static_cast<uint32_t>(dispatch_core_config.get_dispatch_core_type())
                 << (harvesting_map_bits + num_hw_cq_bits + dispatch_core_axis_bits)) |
                (static_cast<uint32_t>(dispatch_core_config.get_dispatch_core_axis())
                 << (harvesting_map_bits + num_hw_cq_bits)) |
                (static_cast<uint32_t>(num_hw_cqs) << harvesting_map_bits);
    if (not MetalContext::instance().hal().is_coordinate_virtualization_enabled()) {
        // Coordinate virtualization is not enabled. For a single program, its associated binaries will vary across
        // devices with different cores harvested.
        build_key |= tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id);
    } else {
        // Coordinate Virtualization is enabled. Track only the number of harvested cores, instead of the exact
        // harvesting configuration (this is not needed).
        build_key |= (std::bitset<harvesting_map_bits>(
                          tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id))
                          .count());
    }
    return build_key;
}

JitBuildStateSet create_build_state(JitBuildEnv& build_env, chip_id_t /*device_id*/, uint8_t num_hw_cqs, bool is_fw) {
    // Get the dispatch message address for this device
    uint32_t dispatch_message_addr = MetalContext::instance().dispatch_mem_map().get_dispatch_message_addr_start();

    // Prepare the container for build states
    const auto& hal = MetalContext::instance().hal();
    uint32_t num_build_states = hal.get_total_num_risc_processors();
    std::vector<std::shared_ptr<JitBuildState>> build_states(num_build_states);

    // Helper lambda to create a build state based on the core type and processor info.
    auto create_jit_build_state = [&](HalProgrammableCoreType core_type,
                                      uint32_t processor_class,
                                      uint32_t processor_type,
                                      bool is_compute_processor) -> std::shared_ptr<JitBuildState> {
        switch (core_type) {
            case HalProgrammableCoreType::TENSIX: {
                if (is_compute_processor) {
                    return std::make_shared<JitBuildCompute>(
                        build_env,
                        JitBuiltStateConfig{
                            .processor_id = processor_type,
                            .is_fw = is_fw,
                            .dispatch_message_addr = dispatch_message_addr});
                } else {
                    // TODO: Make .processor_id = processor_type when brisc and ncrisc are considered one
                    // processor class
                    return std::make_shared<JitBuildDataMovement>(
                        build_env,
                        JitBuiltStateConfig{
                            .processor_id = processor_class,
                            .is_fw = is_fw,
                            .dispatch_message_addr = dispatch_message_addr});
                }
                break;
            }
            case HalProgrammableCoreType::ACTIVE_ETH: {
                // Cooperative means active erisc FW needs to context switch to base FW
                return std::make_shared<JitBuildActiveEthernet>(
                    build_env,
                    JitBuiltStateConfig{
                        .processor_id = processor_class,
                        .is_fw = is_fw,
                        .dispatch_message_addr = dispatch_message_addr,
                        .is_cooperative = hal.get_eth_fw_is_cooperative()});
                break;
            }
            case HalProgrammableCoreType::IDLE_ETH: {
                return std::make_shared<JitBuildIdleEthernet>(
                    build_env,
                    JitBuiltStateConfig{
                        .processor_id = processor_class,
                        .is_fw = is_fw,
                        .dispatch_message_addr = dispatch_message_addr});
                break;
            }
            default:
                TT_THROW(
                    "Unsupported programable core type {} to initialize build states",
                    enchantum::to_string(core_type));
        }
    };

    // Loop through programmable core types and their processor classes/types.
    uint32_t index = 0;
    uint32_t programmable_core_type_count = hal.get_programmable_core_type_count();
    for (uint32_t programmable_core = 0; programmable_core < programmable_core_type_count; programmable_core++) {
        HalProgrammableCoreType core_type = *enchantum::index_to_enum<HalProgrammableCoreType>(programmable_core);
        uint32_t processor_class_count = hal.get_processor_classes_count(programmable_core);
        for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
            auto compute_proc_class = enchantum::cast<HalProcessorClassType>(processor_class);
            bool is_compute_processor =
                compute_proc_class.has_value() and compute_proc_class.value() == HalProcessorClassType::COMPUTE;
            uint32_t processor_types_count = hal.get_processor_types_count(programmable_core, processor_class);
            for (uint32_t processor_type = 0; processor_type < processor_types_count; processor_type++) {
                build_states[index++] =
                    create_jit_build_state(core_type, processor_class, processor_type, is_compute_processor);
            }
        }
    }

    return build_states;
}

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
    return *get_device_build_env(device_id).firmware_build_states[state_idx];
}

const JitBuildState& BuildEnvManager::get_kernel_build_state(
    chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id) {
    uint32_t state_idx = get_build_index_and_state_count(programmable_core, processor_class).first + processor_id;
    return *get_device_build_env(device_id).kernel_build_states[state_idx];
}

JitBuildStateSubset BuildEnvManager::get_kernel_build_states(
    chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class) {
    std::pair<int, int> b_id_and_count = get_build_index_and_state_count(programmable_core, processor_class);
    JitBuildStateSubset subset = {
        &get_device_build_env(device_id).kernel_build_states[b_id_and_count.first], b_id_and_count.second};
    return subset;
}

std::pair<int, int> BuildEnvManager::get_build_index_and_state_count(
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
    jit_build_set(get_device_build_env(device_id).firmware_build_states, nullptr);
}

}  // namespace tt::tt_metal
