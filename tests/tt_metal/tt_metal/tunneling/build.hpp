// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "tt_cluster.hpp"

namespace lite_fabric {

int CompileLiteFabric(
    std::shared_ptr<tt::Cluster> cluster,
    const std::filesystem::path& root_dir,
    const std::filesystem::path& out_dir,
    const std::vector<std::string>& defines = {});

int LinkLiteFabric(
    const std::filesystem::path& root_dir, const std::filesystem::path& out_dir, const std::filesystem::path& elf_out);

}  // namespace lite_fabric
