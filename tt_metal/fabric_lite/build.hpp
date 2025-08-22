// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include "tt_cluster.hpp"

namespace fabric_lite {

int CompileFabricLite(
    tt::Cluster& cluster,
    const std::filesystem::path& root_dir,
    const std::filesystem::path& out_dir,
    const std::vector<std::string>& defines = {});

int LinkFabricLite(
    const std::filesystem::path& root_dir, const std::filesystem::path& out_dir, const std::filesystem::path& elf_out);

}  // namespace fabric_lite
