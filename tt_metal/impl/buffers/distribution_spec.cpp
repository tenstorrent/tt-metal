// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distribution_spec.hpp"
#include "math.hpp"

namespace tt::tt_metal {

DistributionSpec::DistributionSpec(
    const std::vector<uint32_t>& tensor_shape, const std::vector<DistributionType>& spec, size_t num_targets) :
    tensor_shape_(tensor_shape), spec_(spec), num_targets_(num_targets) {}

DistributionSpec DistributionSpec::from_shard_shape(
    const std::vector<uint32_t>& tensor_shape, const std::vector<uint32_t>& shard_shape, size_t num_targets) {
    // Create spec with Shard only
    // Set num_targets to max number of used targets
    std::vector<DistributionType> spec(shard_shape.size());
    size_t max_targets_used = 1;
    for (size_t i = 0; i < spec.size(); i++) {
        spec[i] = Shard{shard_shape[i]};

        max_targets_used *= tt::div_up(tensor_shape[i], shard_shape[i]);
    }

    return DistributionSpec(tensor_shape, spec, std::min(num_targets, max_targets_used));
}

std::vector<TargetData> DistributionSpec::compute_mapping() const {
    std::vector<TargetData> shard_mapping(num_targets_);
    for (size_t target = 0; target < num_targets_; target++) {
        shard_mapping[target] = {{0, 0, 0}};
    }
    return shard_mapping;
}

}  // namespace tt::tt_metal
