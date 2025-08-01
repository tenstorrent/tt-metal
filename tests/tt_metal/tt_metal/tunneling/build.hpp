// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace lite_fabric {

void CompileLiteFabric(const std::string& root_dir, const std::string& out_dir);

void LinkLiteFabric(const std::string& root_dir, const std::string& out_dir);

}  // namespace lite_fabric
