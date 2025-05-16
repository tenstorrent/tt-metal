// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "tt_metal/hw/inc/dataflow_api_generic_addrgen.h"

namespace addr_gen_tests {

template <typename DSpecT>
struct AddrGenInputs {
    using dspec = DSpecT;
};

struct ExpectedDSpec {
    std::array<uint32_t, 2> tensor_strides;
    size_t tensor_volume;

    std::array<uint32_t, 2> shard_strides;
    size_t shard_volume;

    std::array<uint32_t, 2> shard_grid;
    std::array<uint32_t, 2> shard_grid_strides;
};

struct ExpectedBankAndOffset {
    size_t page_id;
    size_t bank_id;
    size_t bank_offset;
};

template <ExpectedDSpec ExpectedDSpecVal, ExpectedBankAndOffset... ExpectedBankAndOffsetVals>
struct AddrGenExpected {
    static constexpr auto dspec = ExpectedDSpecVal;
    static constexpr auto bank_and_offset =
        std::array<ExpectedBankAndOffset, sizeof...(ExpectedBankAndOffsetVals)>{ExpectedBankAndOffsetVals...};
};

template <typename Inputs, typename Expected>
struct AddrGenParams {
    using inputs = Inputs;
    using expected = Expected;
};

namespace addr_gen_test_params {

constexpr std::array<uint32_t, 2> tensor_shape_array = {2, 3};
constexpr std::array<uint32_t, 2> shard_shape_array = {1, 2};
USING_SHAPE_WRAPPER(tensor_shape_1, tensor_shape_array);
USING_SHAPE_WRAPPER(shard_shape_1, shard_shape_array);

using test_params_1 = AddrGenParams<
    AddrGenInputs<KernelDistributionSpec<tensor_shape_1, shard_shape_1, 4>>,
    AddrGenExpected<
        ExpectedDSpec{{3, 1}, 6, {2, 1}, 2, {2, 2}, {2, 1}},
        ExpectedBankAndOffset{0, 0, 0},
        ExpectedBankAndOffset{1, 0, 1},
        ExpectedBankAndOffset{2, 1, 0},
        ExpectedBankAndOffset{3, 2, 0},
        ExpectedBankAndOffset{4, 2, 1},
        ExpectedBankAndOffset{5, 3, 0}>>;

}  // namespace addr_gen_test_params

}  // namespace addr_gen_tests

using namespace addr_gen_tests;

template <typename T>
class AddrGenTests : public ::testing::Test {};

TYPED_TEST_SUITE(AddrGenTests, ::testing::Types<addr_gen_test_params::test_params_1>);

TYPED_TEST(AddrGenTests, LookUp) {
    using dspec_t = TypeParam::inputs::dspec;
    constexpr auto dspec_val = dspec_t{};
    using expected = TypeParam::expected;

    // Create sharded accessor
    auto sharded_accessor = ShardedAccessor<dspec_t>{};

    // Check that the computed values in DSpec match the expected values
    ASSERT_EQ(dspec_val.tensor_strides, expected::dspec.tensor_strides);
    ASSERT_EQ(dspec_val.tensor_volume, expected::dspec.tensor_volume);
    ASSERT_EQ(dspec_val.shard_strides, expected::dspec.shard_strides);
    ASSERT_EQ(dspec_val.shard_volume, expected::dspec.shard_volume);
    ASSERT_EQ(dspec_val.shard_grid, expected::dspec.shard_grid);
    ASSERT_EQ(dspec_val.shard_grid_strides, expected::dspec.shard_grid_strides);

    // Check that the computed bank and offset values match the expected values
    for (const auto& expected_bank_and_offset : expected::bank_and_offset) {
        auto [bank_id, bank_offset] = sharded_accessor.get_bank_and_offset(expected_bank_and_offset.page_id);
        std::cout << "page_id: " << expected_bank_and_offset.page_id << std::endl;
        std::cout << "bank_id: " << bank_id << std::endl;
        std::cout << "bank_offset: " << bank_offset << std::endl;
        EXPECT_EQ(bank_id, expected_bank_and_offset.bank_id);
        EXPECT_EQ(bank_offset, expected_bank_and_offset.bank_offset);
    }
}
