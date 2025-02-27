// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/distribution_spec.hpp>

namespace {
struct DistributionSpecInputs {
    std::vector<uint32_t> tensor_shape;
    std::vector<uint32_t> shard_shape;
    size_t num_targets;
};

struct DistributionSpecExpected {};

struct DistributionSpecParams {
    DistributionSpecInputs inputs;
    DistributionSpecExpected expected;
};
}  // namespace
// namespace

class DistributionSpecTests : public ::testing::TestWithParam<DistributionSpecParams> {};

TEST_P(DistributionSpecTests, Generic) {
    const auto& params = GetParam();
    auto tensor_shape = params.inputs.tensor_shape;
    auto shard_shape = params.inputs.shard_shape;
    auto num_targets = params.inputs.num_targets;

    auto distribution_spec = DistributionSpec::from_shard_shape(tensor_shape, shard_shape, num_targets);
    auto num_targets_used = distribution_spec.get_num_targets();
    auto shard_mapping = distribution_spec.compute_mapping();

    std::cout << num_targets_used << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    DistributionSpec,
    DistributionSpecTests,
    ::testing::Values(DistributionSpecParams{
        DistributionSpecInputs{
            .tensor_shape = {2, 3, 4},
            .shard_shape = {2, 3, 2},
            .num_targets = 4,
        },
        DistributionSpecExpected{},
    })  // Values
);
