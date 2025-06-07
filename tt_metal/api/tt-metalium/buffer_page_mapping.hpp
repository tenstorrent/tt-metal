// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

#include <vector>
#include <optional>
#include <unordered_map>
#include <array>

namespace tt::tt_metal {

struct BufferPageMapping {
    std::vector<CoreCoord> all_cores;
    std::vector<std::vector<uint32_t>> core_host_page_indices;
    std::vector<uint32_t> dev_page_to_core_mapping;

    // some dev pages don't have mapping to host (in case of padding)
    std::vector<std::optional<uint32_t>> dev_page_to_host_page_mapping;
    std::vector<uint32_t> host_page_to_dev_page_mapping;
    std::unordered_map<CoreCoord, uint32_t> core_to_core_id;
    std::vector<uint32_t> host_page_to_local_shard_page_mapping;
    std::vector<std::array<uint32_t, 2>> core_shard_shape;
};

}  // namespace tt::tt_metal
