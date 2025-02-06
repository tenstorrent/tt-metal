// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "build.hpp"

namespace tt::tt_metal {

// Singleton class to generate and hold build environments, build keys, and build states.
class BuildEnvManager {
public:
    BuildEnvManager(const BuildEnvManager&) = delete;
    BuildEnvManager& operator=(const BuildEnvManager&) = delete;
    static BuildEnvManager& get_instance() {
        static BuildEnvManager instance;
        return instance;
    }

    // Add a new build environment for the corresponding device id and num_hw_cqs. Also generates the build key and
    // build states.
    void add_build_env(chip_id_t device_id, uint8_t num_hw_cqs);

    // Getter functions for build envs/keys/states
    const JitBuildEnv& get_build_env(chip_id_t device_id);
    uint32_t get_build_key(chip_id_t device_id);
    const JitBuildState& get_firmware_build_state(
        chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id);
    const JitBuildState& get_kernel_build_state(
        chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class, int processor_id);
    const JitBuildStateSubset get_kernel_build_states(
        chip_id_t device_id, uint32_t programmable_core, uint32_t processor_class);

    void build_firmware(chip_id_t device_id);

    // Helper function to get the unique build id and number of states for a given programmable_core and
    // processor_class.
    std::pair<int, int> get_build_index_and_state_count(uint32_t programmable_core, uint32_t processor_class);

private:
    BuildEnvManager();
    ~BuildEnvManager();

    std::unordered_map<chip_id_t, JitBuildEnv> device_id_to_build_env_;
    std::unordered_map<chip_id_t, uint32_t> device_id_to_build_key_;
    std::unordered_map<chip_id_t, JitBuildStateSet> device_id_to_firmware_build_states_;
    std::unordered_map<chip_id_t, JitBuildStateSet> device_id_to_kernel_build_states_;

    // A device-agnostic mapping from programmable_core_type and processor_class to unique index + processor_type_count.
    // TODO: processor_type_count can be looked up in the hal, do we need this in here?
    std::vector<std::vector<std::pair<int, int>>> build_state_indices_;
};

}  // namespace tt::tt_metal
