// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "build.hpp"
#include "impl/context/context_types.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

using BuildIndexAndTypeCount = std::pair<int, int>;            // Build index and processor type count
using ProcClassMapping = std::vector<BuildIndexAndTypeCount>;  // Processor class to BuildIndexAndTypeCount
using ProgCoreMapping =
    std::vector<ProcClassMapping>;  // Programmable core and processor class to BuildIndexAndTypeCount

// A struct to hold device-specific build environment
struct DeviceBuildEnv {
    uint64_t build_key() const { return build_env.get_build_key(); }
    JitBuildEnv build_env;
    std::vector<JitBuildState> firmware_build_states;
    std::vector<JitBuildState> kernel_build_states;
    bool firmware_precompiled = false;
};

// A struct to hold device-specific build environment info (lightweight version of DeviceBuildEnv)
struct BuildEnvInfo {
    ChipId device_id;
    uint64_t build_key;
    std::string firmware_root_path;
};

// Singleton class to generate and hold build environments, build keys, and build states.
class BuildEnvManager {
public:
    explicit BuildEnvManager(const Hal& hal);
    ~BuildEnvManager() = default;
    BuildEnvManager(const BuildEnvManager&) = delete;
    BuildEnvManager& operator=(const BuildEnvManager&) = delete;
    BuildEnvManager(BuildEnvManager&&) = delete;
    BuildEnvManager& operator=(BuildEnvManager&&) = delete;

    // Returns the per-context BuildEnvManager for the supplied ContextId, seeding its HAL on
    // first call from that ContextId's MetalContext. Each ContextId gets its own
    // BuildEnvManager instance so that contexts with different dispatch configurations
    // (e.g. silicon vs mock) do not share device_id_to_build_env_ slots and overwrite each
    // other's build_keys. Required to fix the mock/silicon coexistence regression
    // (#38445 follow-up).
    static BuildEnvManager& get_instance(ContextId context_id);

    // Legacy no-arg accessor that returns the DEFAULT_CONTEXT_ID instance. Asserts that the
    // default context's BuildEnvManager has been seeded (driven by MetalContext::create_*),
    // which is true for every reader that runs after the default-context MetalEnv has been
    // constructed. Callers that may run under a non-default ContextId must use the
    // get_instance(ContextId) overload instead.
    static BuildEnvManager& get_instance();

    // Internal seeding entry: seeds the BuildEnvManager slot for the given ContextId with the
    // supplied HAL on first call, no-op thereafter. Takes a HAL by reference so callers that
    // already hold g_instance_mutex (e.g. MetalContext::create_*) can seed without re-entering
    // MetalContext::instance(ContextId). Idempotent and thread-safe.
    static void seed_if_unseeded_with_hal(ContextId context_id, const Hal& hal);

    // Frees the per-context slot so a recycled context_id seeds fresh on next create.
    // No-op on an unseeded slot.
    static void destroy_for_context(ContextId context_id);

    // Add a new build environment for the corresponding device id and num_hw_cqs. Also generates the build key and
    // build states.  This requires a live device to be available at device_id.
    // `context_id` selects which MetalContext owns the device; required so the right context's runtime state and
    // hardware query layer is consulted instead of implicitly initializing the silicon default.
    void add_build_env(ChipId device_id, uint8_t num_hw_cqs, ContextId context_id);

    // Add a new build environment for the corresponding device id and device configuration. Also generates the build
    // key and build states.
    void add_build_env(ChipId device_id, const JitDeviceConfig& dev_config, const llrt::RunTimeOptions& rtoptions);

    // Getter functions for build envs/keys/states
    const DeviceBuildEnv& get_device_build_env(ChipId device_id);

    // Helper functions to extract build states from the build env.
    const JitBuildState& get_firmware_build_state(
        ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id);
    const JitBuildState& get_kernel_build_state(
        ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id);
    JitBuildStateSubset get_kernel_build_states(ChipId device_id, uint32_t programmable_core, uint32_t processor_class);

    void build_firmware(ChipId device_id, bool ignore_precompiled = false);

    // Get the path to a firmware binary for loading/linking. Uses pre-compiled path if available.
    std::string get_firmware_binary_path(
        ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id);

    // Get the path to a kernel binary for loading/linking from the provided binary root.
    std::string get_kernel_binary_path(
        ChipId device_id,
        uint32_t programmable_core,
        uint32_t processor_class,
        int processor_id,
        const std::string& binary_root,
        const std::string& kernel_full_name);

    // Helper function to get the unique build id and number of states for a given programmable_core and
    // processor_class.
    BuildIndexAndTypeCount get_build_index_and_state_count(
        uint32_t programmable_core, uint32_t processor_class, bool is_fw);

    // Method to get the build environment info for all devices
    std::vector<BuildEnvInfo> get_all_build_envs_info();

private:
    void add_build_env_locked(
        ChipId device_id, const JitDeviceConfig& dev_config, const llrt::RunTimeOptions& rtoptions);

    std::unordered_map<ChipId, DeviceBuildEnv> device_id_to_build_env_;

    // A device-agnostic mapping from programmable_core_type and processor_class to unique index + processor_type_count.
    // TODO: processor_type_count can be looked up in the hal, do we need this in here?
    ProgCoreMapping kernel_build_state_indices_;
    ProgCoreMapping firmware_build_state_indices_;
    std::mutex lock;

    BuildIndexAndTypeCount get_kernel_build_index_and_state_count(
        uint32_t programmable_core, uint32_t processor_class) const;
    BuildIndexAndTypeCount get_firmware_build_index_and_state_count(
        uint32_t programmable_core, uint32_t processor_class) const;
};

}  // namespace tt::tt_metal
