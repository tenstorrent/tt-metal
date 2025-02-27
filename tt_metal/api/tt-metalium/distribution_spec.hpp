// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <variant>

#include "shape_base.hpp"

namespace tt::tt_metal {

using TargetData = std::vector<std::tuple<size_t, size_t, size_t>>;  // src, dst, size

class DistributionSpec {
public:
    static DistributionSpec from_shard_shape(
        const std::vector<uint32_t>& tensor_shape, const std::vector<uint32_t>& shard_shape, size_t num_targets);

    size_t get_num_targets() const { return num_targets_; }
    std::vector<TargetData> compute_mapping() const;

private:
    struct Shard {
        size_t size = 0;
    };

    struct Replicate {
        size_t multiple = 0;
    };
    using DistributionType = std::variant<Shard, Replicate>;

    DistributionSpec(
        const std::vector<uint32_t>& tensor_shape, const std::vector<DistributionType>& spec, size_t num_targets);

    std::vector<uint32_t> tensor_shape_;
    std::vector<DistributionType> spec_;
    size_t num_targets_;
};

}  // namespace tt::tt_metal
