// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <stddef.h>
#include <tt-metalium/distribution_spec.hpp>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>

namespace distribution_spec_tests {
using tt::tt_metal::DistributionSpec;

struct DistributionSpecInputs {
    tt::tt_metal::Shape tensor_shape;
    tt::tt_metal::Shape shard_shape;
    size_t num_targets;
};

struct DistributionSpecExpected {
    std::vector<DistributionSpec::TargetData> shard_mapping;
    std::vector<DistributionSpec::TargetData> coalesced_shard_mapping;
    tt::tt_metal::Shape shard_shape;
    size_t num_targets;
};

struct DistributionSpecParams {
    DistributionSpecInputs inputs;
    DistributionSpecExpected expected;
};
}  // namespace distribution_spec_tests
// namespace

using namespace distribution_spec_tests;

TEST(IllegalDistributionSpecCreationTests, RankUnequal) {
    EXPECT_THAT(
        std::function<void()>([]() {
            const auto tensor_shape = tt::tt_metal::Shape{2, 3, 4};
            const auto shard_shape = tt::tt_metal::Shape{1, 1};
            const size_t num_targets = 4;
            auto distribution_spec =
                tt::tt_metal::DistributionSpec::from_shard_shape(tensor_shape, shard_shape, num_targets);
        }),
        ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr(std::string("Shard shape rank (2) must be same as tensor shape rank (3)!"))));
}

class DistributionSpecTests : public ::testing::TestWithParam<DistributionSpecParams> {};

TEST_P(DistributionSpecTests, Sharding) {
    const auto& params = GetParam();
    auto tensor_shape = params.inputs.tensor_shape;
    auto shard_shape = params.inputs.shard_shape;
    auto num_targets = params.inputs.num_targets;

    // DistributionSpec sets shard_shape and num_targets internally, so test them explicitly here
    // For sharding, the shard shape should be the same as the one passed in to from_shard_shape
    // num_targets should be set to num_shards if num_targets > num_shards
    auto distribution_spec = tt::tt_metal::DistributionSpec::from_shard_shape(tensor_shape, shard_shape, num_targets);
    auto shard_shape_stored = distribution_spec.get_shard_shape();
    ASSERT_EQ(shard_shape_stored, params.expected.shard_shape);
    auto num_targets_stored = distribution_spec.get_num_targets();
    ASSERT_EQ(num_targets_stored, params.expected.num_targets);

    auto shard_mapping = distribution_spec.compute_metadata_for_targets(DistributionSpec::MappingMode::NONCOALESCED);
    auto coalesced_shard_mapping =
        distribution_spec.compute_metadata_for_targets(DistributionSpec::MappingMode::COALESCED);

    auto validate_target_data = [](const auto& target_data, const auto& expected_target_data) {
        ASSERT_EQ(target_data.size(), expected_target_data.size());
        for (size_t chunk_id = 0; chunk_id < target_data.size(); chunk_id++) {
            EXPECT_EQ(target_data[chunk_id], expected_target_data[chunk_id]);
        }
    };

    for (size_t target_id = 0; target_id < num_targets_stored; target_id++) {
        validate_target_data(shard_mapping[target_id], params.expected.shard_mapping[target_id]);
        validate_target_data(coalesced_shard_mapping[target_id], params.expected.coalesced_shard_mapping[target_id]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DistributionSpec,
    DistributionSpecTests,
    ::testing::Values(
        // 3D spec with 1 shard per core
        DistributionSpecParams{
            DistributionSpecInputs{
                .tensor_shape = tt::tt_metal::Shape{5, 3, 5},
                .shard_shape = tt::tt_metal::Shape{2, 1, 3},
                .num_targets = 20,
            },
            DistributionSpecExpected{
                .shard_mapping =
                    {{{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {15, 3, 1}, {16, 4, 1}, {17, 5, 1}},
                     {{3, 0, 1}, {4, 1, 1}, {18, 3, 1}, {19, 4, 1}},
                     {{5, 0, 1}, {6, 1, 1}, {7, 2, 1}, {20, 3, 1}, {21, 4, 1}, {22, 5, 1}},
                     {{8, 0, 1}, {9, 1, 1}, {23, 3, 1}, {24, 4, 1}},
                     {{10, 0, 1}, {11, 1, 1}, {12, 2, 1}, {25, 3, 1}, {26, 4, 1}, {27, 5, 1}},
                     {{13, 0, 1}, {14, 1, 1}, {28, 3, 1}, {29, 4, 1}},
                     {{30, 0, 1}, {31, 1, 1}, {32, 2, 1}, {45, 3, 1}, {46, 4, 1}, {47, 5, 1}},
                     {{33, 0, 1}, {34, 1, 1}, {48, 3, 1}, {49, 4, 1}},
                     {{35, 0, 1}, {36, 1, 1}, {37, 2, 1}, {50, 3, 1}, {51, 4, 1}, {52, 5, 1}},
                     {{38, 0, 1}, {39, 1, 1}, {53, 3, 1}, {54, 4, 1}},
                     {{40, 0, 1}, {41, 1, 1}, {42, 2, 1}, {55, 3, 1}, {56, 4, 1}, {57, 5, 1}},
                     {{43, 0, 1}, {44, 1, 1}, {58, 3, 1}, {59, 4, 1}},
                     {{60, 0, 1}, {61, 1, 1}, {62, 2, 1}},
                     {{63, 0, 1}, {64, 1, 1}},
                     {{65, 0, 1}, {66, 1, 1}, {67, 2, 1}},
                     {{68, 0, 1}, {69, 1, 1}},
                     {{70, 0, 1}, {71, 1, 1}, {72, 2, 1}},
                     {{73, 0, 1}, {74, 1, 1}}},
                .coalesced_shard_mapping =
                    {{{0, 0, 3}, {15, 3, 3}},
                     {{3, 0, 2}, {18, 3, 2}},
                     {{5, 0, 3}, {20, 3, 3}},
                     {{8, 0, 2}, {23, 3, 2}},
                     {{10, 0, 3}, {25, 3, 3}},
                     {{13, 0, 2}, {28, 3, 2}},
                     {{30, 0, 3}, {45, 3, 3}},
                     {{33, 0, 2}, {48, 3, 2}},
                     {{35, 0, 3}, {50, 3, 3}},
                     {{38, 0, 2}, {53, 3, 2}},
                     {{40, 0, 3}, {55, 3, 3}},
                     {{43, 0, 2}, {58, 3, 2}},
                     {{60, 0, 3}},
                     {{63, 0, 2}},
                     {{65, 0, 3}},
                     {{68, 0, 2}},
                     {{70, 0, 3}},
                     {{73, 0, 2}}},
                .shard_shape = tt::tt_metal::Shape{2, 1, 3},
                .num_targets = 18,
            },
        },
        // Same 3D spec but with multiple shards per core
        DistributionSpecParams{
            DistributionSpecInputs{
                .tensor_shape = tt::tt_metal::Shape{5, 3, 5},
                .shard_shape = tt::tt_metal::Shape{2, 1, 3},
                .num_targets = 4,
            },
            DistributionSpecExpected{
                .shard_mapping =
                    {{{0, 0, 1},   {1, 1, 1},   {2, 2, 1},   {15, 3, 1},  {16, 4, 1},  {17, 5, 1},
                      {10, 6, 1},  {11, 7, 1},  {12, 8, 1},  {25, 9, 1},  {26, 10, 1}, {27, 11, 1},
                      {35, 12, 1}, {36, 13, 1}, {37, 14, 1}, {50, 15, 1}, {51, 16, 1}, {52, 17, 1},
                      {60, 18, 1}, {61, 19, 1}, {62, 20, 1}, {70, 24, 1}, {71, 25, 1}, {72, 26, 1}},
                     {{3, 0, 1},
                      {4, 1, 1},
                      {18, 3, 1},
                      {19, 4, 1},
                      {13, 6, 1},
                      {14, 7, 1},
                      {28, 9, 1},
                      {29, 10, 1},
                      {38, 12, 1},
                      {39, 13, 1},
                      {53, 15, 1},
                      {54, 16, 1},
                      {63, 18, 1},
                      {64, 19, 1},
                      {73, 24, 1},
                      {74, 25, 1}},
                     {{5, 0, 1},   {6, 1, 1},   {7, 2, 1},   {20, 3, 1},  {21, 4, 1},  {22, 5, 1},  {30, 6, 1},
                      {31, 7, 1},  {32, 8, 1},  {45, 9, 1},  {46, 10, 1}, {47, 11, 1}, {40, 12, 1}, {41, 13, 1},
                      {42, 14, 1}, {55, 15, 1}, {56, 16, 1}, {57, 17, 1}, {65, 18, 1}, {66, 19, 1}, {67, 20, 1}},
                     {{8, 0, 1},
                      {9, 1, 1},
                      {23, 3, 1},
                      {24, 4, 1},
                      {33, 6, 1},
                      {34, 7, 1},
                      {48, 9, 1},
                      {49, 10, 1},
                      {43, 12, 1},
                      {44, 13, 1},
                      {58, 15, 1},
                      {59, 16, 1},
                      {68, 18, 1},
                      {69, 19, 1}}},
                .coalesced_shard_mapping =
                    {{{0, 0, 3},
                      {15, 3, 3},
                      {10, 6, 3},
                      {25, 9, 3},
                      {35, 12, 3},
                      {50, 15, 3},
                      {60, 18, 3},
                      {70, 24, 3}},
                     {{3, 0, 2},
                      {18, 3, 2},
                      {13, 6, 2},
                      {28, 9, 2},
                      {38, 12, 2},
                      {53, 15, 2},
                      {63, 18, 2},
                      {73, 24, 2}},
                     {{5, 0, 3}, {20, 3, 3}, {30, 6, 3}, {45, 9, 3}, {40, 12, 3}, {55, 15, 3}, {65, 18, 3}},
                     {{8, 0, 2}, {23, 3, 2}, {33, 6, 2}, {48, 9, 2}, {43, 12, 2}, {58, 15, 2}, {68, 18, 2}}},
                .shard_shape = tt::tt_metal::Shape{2, 1, 3},
                .num_targets = 4,
            },
        },
        // 5D spec with no cuts along last two dims (ie. can coalesce last 3 dims)
        DistributionSpecParams{
            DistributionSpecInputs{
                .tensor_shape = tt::tt_metal::Shape{2, 3, 4, 2, 3},
                .shard_shape = tt::tt_metal::Shape{1, 2, 3, 2, 3},
                .num_targets = 5,
            },
            DistributionSpecExpected{
                .shard_mapping =
                    {{{0, 0, 1},    {1, 1, 1},    {2, 2, 1},    {3, 3, 1},    {4, 4, 1},    {5, 5, 1},   {6, 6, 1},
                      {7, 7, 1},    {8, 8, 1},    {9, 9, 1},    {10, 10, 1},  {11, 11, 1},  {12, 12, 1}, {13, 13, 1},
                      {14, 14, 1},  {15, 15, 1},  {16, 16, 1},  {17, 17, 1},  {24, 18, 1},  {25, 19, 1}, {26, 20, 1},
                      {27, 21, 1},  {28, 22, 1},  {29, 23, 1},  {30, 24, 1},  {31, 25, 1},  {32, 26, 1}, {33, 27, 1},
                      {34, 28, 1},  {35, 29, 1},  {36, 30, 1},  {37, 31, 1},  {38, 32, 1},  {39, 33, 1}, {40, 34, 1},
                      {41, 35, 1},  {90, 36, 1},  {91, 37, 1},  {92, 38, 1},  {93, 39, 1},  {94, 40, 1}, {95, 41, 1},
                      {114, 54, 1}, {115, 55, 1}, {116, 56, 1}, {117, 57, 1}, {118, 58, 1}, {119, 59, 1}},
                     {{18, 0, 1},   {19, 1, 1},   {20, 2, 1},   {21, 3, 1},   {22, 4, 1},   {23, 5, 1},
                      {42, 18, 1},  {43, 19, 1},  {44, 20, 1},  {45, 21, 1},  {46, 22, 1},  {47, 23, 1},
                      {120, 36, 1}, {121, 37, 1}, {122, 38, 1}, {123, 39, 1}, {124, 40, 1}, {125, 41, 1},
                      {126, 42, 1}, {127, 43, 1}, {128, 44, 1}, {129, 45, 1}, {130, 46, 1}, {131, 47, 1},
                      {132, 48, 1}, {133, 49, 1}, {134, 50, 1}, {135, 51, 1}, {136, 52, 1}, {137, 53, 1}},
                     {{48, 0, 1},   {49, 1, 1},   {50, 2, 1},   {51, 3, 1},   {52, 4, 1},   {53, 5, 1},
                      {54, 6, 1},   {55, 7, 1},   {56, 8, 1},   {57, 9, 1},   {58, 10, 1},  {59, 11, 1},
                      {60, 12, 1},  {61, 13, 1},  {62, 14, 1},  {63, 15, 1},  {64, 16, 1},  {65, 17, 1},
                      {138, 36, 1}, {139, 37, 1}, {140, 38, 1}, {141, 39, 1}, {142, 40, 1}, {143, 41, 1}},
                     {{66, 0, 1}, {67, 1, 1}, {68, 2, 1}, {69, 3, 1}, {70, 4, 1}, {71, 5, 1}},
                     {{72, 0, 1},   {73, 1, 1},   {74, 2, 1},   {75, 3, 1},   {76, 4, 1},   {77, 5, 1},
                      {78, 6, 1},   {79, 7, 1},   {80, 8, 1},   {81, 9, 1},   {82, 10, 1},  {83, 11, 1},
                      {84, 12, 1},  {85, 13, 1},  {86, 14, 1},  {87, 15, 1},  {88, 16, 1},  {89, 17, 1},
                      {96, 18, 1},  {97, 19, 1},  {98, 20, 1},  {99, 21, 1},  {100, 22, 1}, {101, 23, 1},
                      {102, 24, 1}, {103, 25, 1}, {104, 26, 1}, {105, 27, 1}, {106, 28, 1}, {107, 29, 1},
                      {108, 30, 1}, {109, 31, 1}, {110, 32, 1}, {111, 33, 1}, {112, 34, 1}, {113, 35, 1}}},
                .coalesced_shard_mapping =
                    {{{0, 0, 18}, {24, 18, 18}, {90, 36, 6}, {114, 54, 6}},
                     {{18, 0, 6}, {42, 18, 6}, {120, 36, 18}},
                     {{48, 0, 18}, {138, 36, 6}},
                     {{66, 0, 6}},
                     {{72, 0, 18}, {96, 18, 18}}},
                .shard_shape = tt::tt_metal::Shape{1, 2, 3, 2, 3},
                .num_targets = 5,
            },
        },
        // 4D spec with cut along first dim (ie. can coalesce all dims because last dim is coalesced by default)
        DistributionSpecParams{
            DistributionSpecInputs{
                .tensor_shape = tt::tt_metal::Shape{3, 2, 1, 3},
                .shard_shape = tt::tt_metal::Shape{2, 2, 1, 3},
                .num_targets = 5,
            },
            DistributionSpecExpected{
                .shard_mapping =
                    {{{0, 0, 1},
                      {1, 1, 1},
                      {2, 2, 1},
                      {3, 3, 1},
                      {4, 4, 1},
                      {5, 5, 1},
                      {6, 6, 1},
                      {7, 7, 1},
                      {8, 8, 1},
                      {9, 9, 1},
                      {10, 10, 1},
                      {11, 11, 1}},
                     {{12, 0, 1}, {13, 1, 1}, {14, 2, 1}, {15, 3, 1}, {16, 4, 1}, {17, 5, 1}}},
                .coalesced_shard_mapping = {{{0, 0, 12}}, {{12, 0, 6}}},
                .shard_shape = tt::tt_metal::Shape{2, 2, 1, 3},
                .num_targets = 2,
            },
        },
        // 4D spec with no cuts (ie. can coalesce all dims)
        DistributionSpecParams{
            DistributionSpecInputs{
                .tensor_shape = tt::tt_metal::Shape{2, 2, 3, 1},
                .shard_shape = tt::tt_metal::Shape{2, 2, 3, 1},
                .num_targets = 5,
            },
            DistributionSpecExpected{
                .shard_mapping =
                    {{{0, 0, 1},
                      {1, 1, 1},
                      {2, 2, 1},
                      {3, 3, 1},
                      {4, 4, 1},
                      {5, 5, 1},
                      {6, 6, 1},
                      {7, 7, 1},
                      {8, 8, 1},
                      {9, 9, 1},
                      {10, 10, 1},
                      {11, 11, 1}},
                     {{0, 0, 12}}},
                .coalesced_shard_mapping = {{{0, 0, 12}}},
                .shard_shape = tt::tt_metal::Shape{2, 2, 3, 1},
                .num_targets = 1,
            },
        },
        // 1D spec for edge case
        DistributionSpecParams{
            DistributionSpecInputs{
                .tensor_shape = tt::tt_metal::Shape{7},
                .shard_shape = tt::tt_metal::Shape{3},
                .num_targets = 5,
            },
            DistributionSpecExpected{
                .shard_mapping = {{{0, 0, 1}, {1, 1, 1}, {2, 2, 1}}, {{3, 0, 1}, {4, 1, 1}, {5, 2, 1}}, {{6, 0, 1}}},
                .coalesced_shard_mapping = {{{0, 0, 3}}, {{3, 0, 3}}, {{6, 0, 1}}},
                .shard_shape = tt::tt_metal::Shape{3},
                .num_targets = 3,
            },
        })  // Values
);
