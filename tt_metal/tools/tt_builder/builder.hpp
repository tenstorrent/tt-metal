// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <filesystem>

namespace tt::tt_metal {

class tt_builder {
public:
    tt_builder();
    ~tt_builder();

    void set_built_path(const std::string& new_built_path);
    std::string get_built_path() { return this->output_dir_.string(); }

    void build_firmware();
    void build_kernel();

private:
    std::filesystem::path output_dir_;
};

}  // namespace tt::tt_metal
