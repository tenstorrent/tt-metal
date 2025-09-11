// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include "tt_cluster.hpp"

namespace lite_fabric {

int CompileFabricLite(
    tt::Cluster& cluster,
    const std::filesystem::path& root_dir,
    const std::filesystem::path& out_dir,
    const std::vector<std::string>& defines = {});

int LinkFabricLite(
    const std::filesystem::path& root_dir, const std::filesystem::path& out_dir, const std::filesystem::path& elf_out);

}  // namespace lite_fabric
