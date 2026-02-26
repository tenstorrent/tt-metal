// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_env_manager.hpp"

#include <tracy/Tracy.hpp>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>

#include <tt_stl/assert.hpp>
#include "common/stable_hash.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/build.hpp"
#include "jit_build/precompiled.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {

BuildEnvManager& BuildEnvManager::get_instance() {
    static BuildEnvManager instance(MetalContext::instance().hal());
    return instance;
}

BuildEnvManager::BuildEnvManager(const Hal& hal) {
    uint32_t kernel_index = 0;
    uint32_t firmware_index = 0;
    uint32_t programmable_core_type_count = hal.get_programmable_core_type_count();
    kernel_build_state_indices_.resize(programmable_core_type_count);
    firmware_build_state_indices_.resize(programmable_core_type_count);
    for (uint32_t programmable_core = 0; programmable_core < programmable_core_type_count; programmable_core++) {
        uint32_t processor_class_count =
            hal.get_processor_classes_count(hal.get_programmable_core_type(programmable_core));
        kernel_build_state_indices_[programmable_core].resize(processor_class_count);
        firmware_build_state_indices_[programmable_core].resize(processor_class_count);
        for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
            const uint32_t processor_types_count = hal.get_processor_types_count(programmable_core, processor_class);
            kernel_build_state_indices_[programmable_core][processor_class] = {kernel_index, processor_types_count};
            kernel_index += processor_types_count;

            const uint32_t processor_class_num_fw_binaries =
                hal.get_processor_class_num_fw_binaries(programmable_core, processor_class);
            firmware_build_state_indices_[programmable_core][processor_class] = {
                firmware_index, processor_class_num_fw_binaries};
            firmware_index += processor_class_num_fw_binaries;
        }
    }
}

namespace {

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

std::vector<JitBuildState> create_build_state(JitBuildEnv& build_env, const JitDeviceConfig& dev_config, bool is_fw) {
    TT_ASSERT(dev_config.hal != nullptr);
    const auto& hal = *dev_config.hal;
    uint32_t total_num_build_states = 0;
    if (is_fw) {
        for (uint32_t programmable_core = 0; programmable_core < hal.get_programmable_core_type_count();
             programmable_core++) {
            const uint32_t processor_class_count =
                hal.get_processor_classes_count(hal.get_programmable_core_type(programmable_core));
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                total_num_build_states += hal.get_processor_class_num_fw_binaries(programmable_core, processor_class);
            }
        }
    } else {
        total_num_build_states = hal.get_total_num_risc_processors();
    }
    std::vector<JitBuildState> build_states;
    build_states.reserve(total_num_build_states);

    uint32_t programmable_core_type_count = hal.get_programmable_core_type_count();
    for (uint32_t programmable_core = 0; programmable_core < programmable_core_type_count; programmable_core++) {
        uint32_t processor_class_count =
            hal.get_processor_classes_count(hal.get_programmable_core_type(programmable_core));
        for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
            JitBuiltStateConfig config{
                .core_type = static_cast<HalProgrammableCoreType>(programmable_core),
                .processor_class = static_cast<HalProcessorClassType>(processor_class),
                .is_fw = is_fw,
                .dispatch_message_addr = dev_config.dispatch_message_addr,
                .is_cooperative = hal.get_eth_fw_is_cooperative(),
            };

            const uint32_t num_build_states =
                is_fw ? hal.get_processor_class_num_fw_binaries(programmable_core, processor_class)
                      : hal.get_processor_types_count(programmable_core, processor_class);
            for (uint32_t build_state_idx = 0; build_state_idx < num_build_states; build_state_idx++) {
                config.processor_id = build_state_idx;
                build_states.emplace_back(build_env, config, hal);
            }
        }
    }

    TT_ASSERT(build_states.size() == total_num_build_states);
    return build_states;
}

}  // namespace

void BuildEnvManager::add_build_env(ChipId device_id, uint8_t num_hw_cqs) {
    const std::lock_guard<std::mutex> lock(this->lock);

    auto& dev_build_env = device_id_to_build_env_[device_id];
    auto dev_config = create_jit_device_config(device_id, num_hw_cqs);
    const auto& rtoptions = MetalContext::instance().rtoptions();

    uint64_t build_key = compute_build_key(dev_config, rtoptions);
    auto device_kernel_defines = initialize_device_kernel_defines(dev_config);

    dev_build_env.build_env.init(build_key, dev_config, rtoptions, device_kernel_defines);
    auto precompiled_dir = precompiled::find_precompiled_dir(
        MetalContext::instance().rtoptions().get_root_dir(), dev_build_env.build_env.get_build_key());
    if (precompiled_dir.has_value()) {
        dev_build_env.build_env.set_firmware_binary_root(*precompiled_dir);
        dev_build_env.firmware_precompiled = true;
        log_info(tt::LogBuildKernels, "Using pre-compiled firmware from: {}", *precompiled_dir);
    }

    dev_build_env.firmware_build_states = create_build_state(dev_build_env.build_env, dev_config, true);
    dev_build_env.kernel_build_states = create_build_state(dev_build_env.build_env, dev_config, false);
}

const DeviceBuildEnv& BuildEnvManager::get_device_build_env(ChipId device_id) {
    const std::lock_guard<std::mutex> lock(this->lock);
    TT_ASSERT(device_id_to_build_env_.contains(device_id), "Couldn't find build env for device {}.", device_id);
    return device_id_to_build_env_[device_id];
}

const JitBuildState& BuildEnvManager::get_firmware_build_state(
    ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id) {
    const uint32_t state_idx =
        get_firmware_build_index_and_state_count(programmable_core, processor_class).first + processor_id;
    return get_device_build_env(device_id).firmware_build_states[state_idx];
}

const JitBuildState& BuildEnvManager::get_kernel_build_state(
    ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id) {
    const uint32_t state_idx =
        get_kernel_build_index_and_state_count(programmable_core, processor_class).first + processor_id;
    return get_device_build_env(device_id).kernel_build_states[state_idx];
}

JitBuildStateSubset BuildEnvManager::get_kernel_build_states(
    ChipId device_id, uint32_t programmable_core, uint32_t processor_class) {
    auto [b_id, count] = get_kernel_build_index_and_state_count(programmable_core, processor_class);
    const auto& kernel_build_states = get_device_build_env(device_id).kernel_build_states;
    return {kernel_build_states.begin() + b_id, static_cast<size_t>(count)};
}

BuildIndexAndTypeCount BuildEnvManager::get_kernel_build_index_and_state_count(
    uint32_t programmable_core, uint32_t processor_class) const {
    TT_ASSERT(
        programmable_core < kernel_build_state_indices_.size(),
        "Programmable core type {} is not included in the Kernel build state",
        programmable_core);
    TT_ASSERT(
        processor_class < kernel_build_state_indices_[programmable_core].size(),
        "Processor class type {} is not included in the Kernel build state",
        processor_class);
    return kernel_build_state_indices_[programmable_core][processor_class];
}

BuildIndexAndTypeCount BuildEnvManager::get_firmware_build_index_and_state_count(
    uint32_t programmable_core, uint32_t processor_class) const {
    TT_ASSERT(
        programmable_core < firmware_build_state_indices_.size(),
        "Programmable core type {} is not included in the Firmware build state",
        programmable_core);
    TT_ASSERT(
        processor_class < firmware_build_state_indices_[programmable_core].size(),
        "Processor class type {} is not included in the Firmware build state",
        processor_class);
    return firmware_build_state_indices_[programmable_core][processor_class];
}

BuildIndexAndTypeCount BuildEnvManager::get_build_index_and_state_count(
    uint32_t programmable_core, uint32_t processor_class, bool is_fw) {
    const std::lock_guard<std::mutex> lock(this->lock);
    if (is_fw) {
        return get_firmware_build_index_and_state_count(programmable_core, processor_class);
    }
    return get_kernel_build_index_and_state_count(programmable_core, processor_class);
}

void BuildEnvManager::build_firmware(ChipId device_id) {
    ZoneScoped;
    const auto& build_env = get_device_build_env(device_id);
    if (build_env.firmware_precompiled) {
        return;
    }
    jit_build_once(build_env.build_key(), [&build_env] { jit_build_subset(build_env.firmware_build_states, nullptr); });
}

std::string BuildEnvManager::get_firmware_binary_path(
    ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id) {
    const auto& env = get_device_build_env(device_id).build_env;
    const auto& state = get_firmware_build_state(device_id, programmable_core, processor_class, processor_id);
    return env.get_firmware_binary_root() + state.get_target_full_path();
}

// Get build environment info for all devices
std::vector<BuildEnvInfo> BuildEnvManager::get_all_build_envs_info() {
    const std::lock_guard<std::mutex> lock(this->lock);
    std::vector<BuildEnvInfo> build_env_info;
    build_env_info.reserve(device_id_to_build_env_.size());
    for (const auto& [device_id, build_env] : device_id_to_build_env_) {
        build_env_info.emplace_back(device_id, build_env.build_key(), build_env.build_env.get_out_firmware_root_path());
    }
    return build_env_info;
}
}  // namespace tt::tt_metal
