// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include "hal/lite_fabric_hal.hpp"
#include <memory>

namespace lite_fabric {

int CompileFabricLite(
    const std::shared_ptr<lite_fabric::LiteFabricHal>& lite_fabric_hal,
    const std::filesystem::path& root_dir,
    const std::filesystem::path& out_dir,
    const std::vector<std::string>& defines = {});

int LinkFabricLite(
    const std::filesystem::path& root_dir, const std::filesystem::path& out_dir, const std::filesystem::path& elf_out);

}  // namespace lite_fabric
