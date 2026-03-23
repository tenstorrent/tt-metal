// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal::experimental {

std::optional<std::string> get_mock_cluster_desc_name(tt::ARCH arch, uint32_t num_chips);

}  // namespace tt::tt_metal::experimental
