// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/distribution_spec.hpp>

namespace {
struct DistributionSpecInputs {};

struct DistributionSpecExpected {};

struct DistributionSpecParams {
    DistributionSpecInputs inputs;
    DistributionSpecExpected expected;
};
}  // namespace
// namespace

class DistributionSpecTests : public ::testing::TestWithParam<DistributionSpecParams> {};

TEST_P(DistributionSpecTests, Generic) { const auto& params = GetParam(); }

INSTANTIATE_TEST_SUITE_P(
    DistributionSpec,
    DistributionSpecTests,
    ::testing::Values(DistributionSpecParams{
        DistributionSpecInputs{},
        DistributionSpecExpected{},
    })  // Values
);
