// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "tt_cluster.hpp"

namespace lite_fabric {

void CompileLiteFabric(
    std::shared_ptr<tt::Cluster> cluster,
    const std::string& root_dir,
    const std::string& out_dir,
    const std::vector<std::string>& defines = {});

void LinkLiteFabric(const std::string& root_dir, const std::string& out_dir);

}  // namespace lite_fabric
