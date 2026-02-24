// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <umd/device/types/arch.hpp>
#include <cstdint>

namespace tt::tt_metal::experimental {

std::optional<std::string> get_mock_cluster_desc_for_config(tt::ARCH arch, uint32_t num_chips);

}
