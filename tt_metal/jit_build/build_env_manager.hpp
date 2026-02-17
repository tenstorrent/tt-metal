// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "build.hpp"
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
    BuildEnvManager(const BuildEnvManager&) = delete;
    BuildEnvManager& operator=(const BuildEnvManager&) = delete;
    static BuildEnvManager& get_instance();

    // Add a new build environment for the corresponding device id and num_hw_cqs. Also generates the build key and
    // build states.
    void add_build_env(ChipId device_id, uint8_t num_hw_cqs);

    // Getter functions for build envs/keys/states
    const DeviceBuildEnv& get_device_build_env(ChipId device_id);

    // Helper functions to extract build states from the build env.
    const JitBuildState& get_firmware_build_state(
        ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id);
    const JitBuildState& get_kernel_build_state(
        ChipId device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id);
    JitBuildStateSubset get_kernel_build_states(ChipId device_id, uint32_t programmable_core, uint32_t processor_class);

    void build_firmware(ChipId device_id);

    // Helper function to get the unique build id and number of states for a given programmable_core and
    // processor_class.
    BuildIndexAndTypeCount get_build_index_and_state_count(
        uint32_t programmable_core, uint32_t processor_class, bool is_fw);

    // Method to get the build environment info for all devices
    std::vector<BuildEnvInfo> get_all_build_envs_info();

private:
    BuildEnvManager();
    ~BuildEnvManager() = default;

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
