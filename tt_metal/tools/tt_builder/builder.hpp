// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <filesystem>
#include "tt_metal/jit_build/build.hpp"
#include "tt_metal/host_api.hpp"

namespace tt::tt_metal {

class BuilderTool {
public:
    BuilderTool();
    ~BuilderTool();

    void set_built_path(const std::string& new_built_path);
    std::string get_built_path() { return this->output_dir_.string(); }

    // Returns the path to cache of latest build, valid only after susscessful build.
    std::string get_firmware_root_path() { return this->firmware_output_dir_.string(); }

    void build_firmware();
    void build_dispatch();

private:
    std::filesystem::path output_dir_;
    std::filesystem::path firmware_output_dir_;

    std::vector<std::shared_ptr<JitBuildState>> get_build_states(tt_metal::IDevice* device, int id, bool is_fw);
    std::vector<std::vector<std::pair<int, int>>> build_state_indices_;
};

}  // namespace tt::tt_metal
