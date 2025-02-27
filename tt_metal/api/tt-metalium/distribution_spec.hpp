// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

namespace tt::tt_metal {

struct DistributionSpec {
    std::vector<uint32_t> tensor_shape;
    std::vector<uint32_t> shard_shape;
};

}  // namespace tt::tt_metal
