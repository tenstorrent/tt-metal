// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_env_manager.hpp"

#include <tracy/Tracy.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
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

namespace {

// Per-ContextId BuildEnvManager instances. Each MetalContext (silicon, mock, future) gets its
// own slot so that contexts with different dispatch configurations do not share
// device_id_to_build_env_ entries. Seeded lazily by seed_if_unseeded_with_hal(ContextId, Hal).
// Slots are read-only for the lifetime of the process once seeded.
//
// PRE-EXISTING LIMITATION (carried forward from the original singleton design, now scoped
// per-context): HAL layout within a single context can be modulated by rtoptions --
// simulator mode, Blackhole DRAM programmable cores, 2-erisc mode -- so the HAL that first
// seeds a context's slot wins and shapes the kernel/firmware build_state_indices_ tables
// for the lifetime of that context. Cross-context, the slots are independent and unaffected.
std::array<std::atomic<BuildEnvManager*>, MAX_CONTEXT_COUNT> s_instances{};
std::mutex s_seed_mutex;

}  // namespace

void BuildEnvManager::seed_if_unseeded_with_hal(ContextId context_id, const Hal& hal) {
    const int slot = context_id.get();
    TT_FATAL(
        slot >= 0 && static_cast<size_t>(slot) < MAX_CONTEXT_COUNT,
        "BuildEnvManager: context_id {} out of range [0, {})",
        slot,
        MAX_CONTEXT_COUNT);
    // Fast path: this context's slot is already seeded. No lock, no allocation.
    if (s_instances[slot].load(std::memory_order_acquire) != nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_seed_mutex);
    if (s_instances[slot].load(std::memory_order_acquire) != nullptr) {
        return;
    }
    // Seed this context's slot with the supplied HAL. The constructor sizes
    // kernel_build_state_indices_ / firmware_build_state_indices_ from that HAL's
    // programmable-core-type, processor-class, and fw-binary counts.
    s_instances[slot].store(new BuildEnvManager(hal), std::memory_order_release);
}

void BuildEnvManager::destroy_for_context(ContextId context_id) {
    const int slot = context_id.get();
    TT_FATAL(
        slot >= 0 && static_cast<size_t>(slot) < MAX_CONTEXT_COUNT,
        "BuildEnvManager: context_id {} out of range [0, {})",
        slot,
        MAX_CONTEXT_COUNT);
    // Lock pairs with seed_if_unseeded_with_hal()'s double-check to avoid a concurrent reseed.
    std::lock_guard<std::mutex> lock(s_seed_mutex);
    auto* existing = s_instances[slot].exchange(nullptr, std::memory_order_acq_rel);
    delete existing;
}

BuildEnvManager& BuildEnvManager::get_instance(ContextId context_id) {
    const int slot = context_id.get();
    TT_FATAL(
        slot >= 0 && static_cast<size_t>(slot) < MAX_CONTEXT_COUNT,
        "BuildEnvManager: context_id {} out of range [0, {})",
        slot,
        MAX_CONTEXT_COUNT);
    auto* existing = s_instances[slot].load(std::memory_order_acquire);
    if (existing != nullptr) {
        return *existing;
    }
    // Lazy seeding path: resolve the ContextId to a HAL via MetalContext::instance(context_id).
    // Callers that already hold g_instance_mutex (i.e. MetalContext::create_*) must use
    // seed_if_unseeded_with_hal() directly with the in-scope HAL to avoid re-entering
    // MetalContext::instance().
    seed_if_unseeded_with_hal(context_id, MetalContext::instance(context_id).hal());
    return *s_instances[slot].load(std::memory_order_acquire);
}

BuildEnvManager& BuildEnvManager::get_instance() {
    auto* existing = s_instances[DEFAULT_CONTEXT_ID.get()].load(std::memory_order_acquire);
    TT_FATAL(
        existing != nullptr,
        "BuildEnvManager::get_instance() called before the default-context MetalContext was "
        "created. The default-context slot is seeded by MetalContext::create_instance(); "
        "ensure at least one default-context MetalEnv has been constructed first, or call "
        "get_instance(ContextId) explicitly for a non-default context.");
    return *existing;
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

    if (config.quasar_dm_only) {
        device_kernel_defines.emplace("QUASAR_DM_ONLY", "1");
    }

    return device_kernel_defines;
}

uint64_t compute_build_key(const JitDeviceConfig& config, const llrt::RunTimeOptions& rtoptions) {
    // Collect all the parameters that affect the build configuration
    StableHasher hasher;

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
    hasher.update(static_cast<uint32_t>(config.quasar_dm_only));

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

void BuildEnvManager::add_build_env(ChipId device_id, uint8_t num_hw_cqs, ContextId context_id) {
    const std::lock_guard<std::mutex> lock(this->lock);
    auto dev_config = create_jit_device_config(device_id, num_hw_cqs, context_id);
    add_build_env_locked(device_id, dev_config, MetalContext::instance(context_id).rtoptions());
}

void BuildEnvManager::add_build_env(
    ChipId device_id, const JitDeviceConfig& dev_config, const llrt::RunTimeOptions& rtoptions) {
    const std::lock_guard<std::mutex> lock(this->lock);
    add_build_env_locked(device_id, dev_config, rtoptions);
}

void BuildEnvManager::add_build_env_locked(
    ChipId device_id, const JitDeviceConfig& dev_config, const llrt::RunTimeOptions& rtoptions) {
    auto& dev_build_env = device_id_to_build_env_[device_id];
    uint64_t build_key = compute_build_key(dev_config, rtoptions);
    auto device_kernel_defines = initialize_device_kernel_defines(dev_config);
    dev_build_env.build_env.init(build_key, dev_config, rtoptions, device_kernel_defines);
    if (rtoptions.get_disable_precompiled_fw()) {
        dev_build_env.firmware_precompiled = false;
    } else {
        auto precompiled_dir =
            precompiled::find_precompiled_dir(rtoptions.get_root_dir(), dev_build_env.build_env.get_build_key());
        if (precompiled_dir.has_value()) {
            dev_build_env.build_env.set_firmware_binary_root(*precompiled_dir);
            dev_build_env.firmware_precompiled = true;
        } else {
            dev_build_env.firmware_precompiled = false;
            log_debug(tt::LogBuildKernels, "No pre-compiled firmware found for build key: {}", build_key);
        }
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
    // `processor_id` is indexed in the per-processor-type space (0..get_processor_types_count-1),
    // which on Quasar can span replicated NEOs (e.g. COMPUTE has 16 entries: 4 NEOs x 4 TRISCs).
    // Firmware binaries are built per TRISC type only (num_fw_binaries == 4 for COMPUTE), and the
    // processor layout is {NEO0 TR0..3, NEO1 TR0..3, ...}, so the type index is processor_id % num_fw_binaries.
    const auto [base, num_fw_binaries] = get_firmware_build_index_and_state_count(programmable_core, processor_class);
    TT_ASSERT(
        num_fw_binaries > 0,
        "No firmware binaries for programmable_core={} processor_class={}",
        programmable_core,
        processor_class);
    const uint32_t state_idx = base + (static_cast<uint32_t>(processor_id) % num_fw_binaries);
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

void BuildEnvManager::build_firmware(ChipId device_id, bool ignore_precompiled) {
    ZoneScoped;
    const auto& build_env = get_device_build_env(device_id);
    if (!ignore_precompiled && build_env.firmware_precompiled) {
        log_info(
            tt::LogBuildKernels,
            "Using pre-compiled firmware from: {}",
            build_env.build_env.get_firmware_binary_root());
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

std::string BuildEnvManager::get_kernel_binary_path(
    ChipId device_id,
    uint32_t programmable_core,
    uint32_t processor_class,
    int processor_id,
    const std::string& binary_root,
    const std::string& kernel_full_name) {
    const auto& state = get_kernel_build_state(device_id, programmable_core, processor_class, processor_id);
    auto path = std::filesystem::path(binary_root) / kernel_full_name;
    path += state.get_target_full_path();
    return path.string();
}

// Get build environment info for all devices
std::vector<BuildEnvInfo> BuildEnvManager::get_all_build_envs_info() {
    const std::lock_guard<std::mutex> lock(this->lock);
    std::vector<BuildEnvInfo> build_env_info;
    build_env_info.reserve(device_id_to_build_env_.size());
    for (const auto& [device_id, build_env] : device_id_to_build_env_) {
        build_env_info.emplace_back(device_id, build_env.build_key(), build_env.build_env.get_firmware_binary_root());
    }
    return build_env_info;
}
}  // namespace tt::tt_metal
