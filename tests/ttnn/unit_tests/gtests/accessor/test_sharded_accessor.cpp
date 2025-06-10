// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <fmt/format.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include "ttnn/cpp/ttnn/operations/sharding_utilities.hpp"

// Defines to include tt_metal/hw/inc/accessor/sharded_accessor.h but won't need these
#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

template <int N>
constexpr auto get_ct_arg();
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

#define noc_index 0
#define ASSERT(condition, ...)
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#include "tt_metal/hw/inc/accessor/sharded_accessor.h"

#undef get_compile_time_arg_val
#undef noc_index
#undef ASSERT
#undef FORCE_INLINE

// If inputs are passed as constexpr arrays, we can use this style to directly create the structs
// Example:
//    constexpr std::array<uint32_t, 3> tensor_shape_array_1 = {1, 2, 3};
//    USING_STRUCT_FROM_ARRAY_WRAPPER(ShapeWrapper, tensor_shape_1, tensor_shape_array);
//    static_assert(std::is_same_v<tensor_shape_1, ShapeWrapper<1, 2, 3>>);
template <template <size_t...> class Wrapper, typename F, size_t... Is>
constexpr auto make_struct_from_array_wrapper(F, std::index_sequence<Is...>) -> Wrapper<F{}()[Is]...>;

#define USING_STRUCT_FROM_ARRAY_WRAPPER(Wrapper, name, arr) \
    struct name##_fn {                                      \
        constexpr auto operator()() const { return (arr); } \
    };                                                      \
    using name =                                            \
        decltype(make_struct_from_array_wrapper<Wrapper>(name##_fn{}, std::make_index_sequence<(arr).size()>{}))

namespace sharded_accessor_tests {

template <typename DSpecT>
struct ShardedAccessorInputs {
    using dspec = DSpecT;
};

template <size_t Rank>
struct ExpectedDSpec {
    std::array<uint32_t, Rank> tensor_strides;
    size_t tensor_volume;

    std::array<uint32_t, Rank> shard_strides;
    size_t shard_volume;

    std::array<uint32_t, Rank> shard_grid;
    std::array<uint32_t, Rank> shard_grid_strides;
};

struct ExpectedBankAndOffset {
    size_t page_id;
    size_t bank_id;
    size_t bank_offset;
};

template <ExpectedDSpec ExpectedDSpecVal, ExpectedBankAndOffset... ExpectedBankAndOffsetVals>
struct ShardedAccessorExpected {
    static constexpr auto dspec = ExpectedDSpecVal;
    static constexpr auto bank_and_offset =
        std::array<ExpectedBankAndOffset, sizeof...(ExpectedBankAndOffsetVals)>{ExpectedBankAndOffsetVals...};
};

template <typename Inputs, typename Expected>
struct ShardedAccessorParams {
    using inputs = Inputs;
    using expected = Expected;
};

namespace params {

constexpr size_t rank_1 = 2;
constexpr size_t num_banks_1 = 4;
constexpr std::array<uint32_t, rank_1> tensor_shape_array_1 = {2, 3};
constexpr std::array<uint32_t, rank_1> shard_shape_array_1 = {1, 2};
constexpr std::array<uint32_t, num_banks_1> bank_coord_array_1{};
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::ShapeWrapper, tensor_shape_1, tensor_shape_array_1);
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::ShapeWrapper, shard_shape_1, shard_shape_array_1);
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::BankCoordWrapper, bank_coords_1, bank_coord_array_1);

using test_params_1 = ShardedAccessorParams<
    ShardedAccessorInputs<::detail::DistributionSpec<tensor_shape_1, shard_shape_1, bank_coords_1>>,
    ShardedAccessorExpected<
        ExpectedDSpec<rank_1>{
            .tensor_strides = {3, 1},
            .tensor_volume = 6,
            .shard_strides = {2, 1},
            .shard_volume = 2,
            .shard_grid = {2, 2},
            .shard_grid_strides = {2, 1}},
        ExpectedBankAndOffset{.page_id = 0, .bank_id = 0, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 1, .bank_id = 0, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 2, .bank_id = 1, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 3, .bank_id = 2, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 4, .bank_id = 2, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 5, .bank_id = 3, .bank_offset = 0}>>;

constexpr size_t rank_2 = 4;
constexpr size_t num_banks_2 = 6;
constexpr std::array<uint32_t, rank_2> tensor_shape_array_2 = {2, 1, 3, 4};
constexpr std::array<uint32_t, rank_2> shard_shape_array_2 = {2, 2, 1, 2};
constexpr std::array<uint32_t, num_banks_2> bank_coord_array_2{};
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::ShapeWrapper, tensor_shape_2, tensor_shape_array_2);
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::ShapeWrapper, shard_shape_2, shard_shape_array_2);
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::BankCoordWrapper, bank_coords_2, bank_coord_array_2);

using test_params_2 = ShardedAccessorParams<
    ShardedAccessorInputs<::detail::DistributionSpec<tensor_shape_2, shard_shape_2, bank_coords_2>>,
    ShardedAccessorExpected<
        ExpectedDSpec<rank_2>{
            .tensor_strides = {12, 12, 4, 1},
            .tensor_volume = 24,
            .shard_strides = {4, 2, 2, 1},
            .shard_volume = 8,
            .shard_grid = {1, 1, 3, 2},
            .shard_grid_strides = {6, 6, 2, 1}},
        ExpectedBankAndOffset{.page_id = 0, .bank_id = 0, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 1, .bank_id = 0, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 2, .bank_id = 1, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 3, .bank_id = 1, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 4, .bank_id = 2, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 5, .bank_id = 2, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 6, .bank_id = 3, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 7, .bank_id = 3, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 8, .bank_id = 4, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 9, .bank_id = 4, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 10, .bank_id = 5, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 11, .bank_id = 5, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 12, .bank_id = 0, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 13, .bank_id = 0, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 14, .bank_id = 1, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 15, .bank_id = 1, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 16, .bank_id = 2, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 17, .bank_id = 2, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 18, .bank_id = 3, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 19, .bank_id = 3, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 20, .bank_id = 4, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 21, .bank_id = 4, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 22, .bank_id = 5, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 23, .bank_id = 5, .bank_offset = 5}>>;

constexpr size_t rank_3 = 4;
constexpr size_t num_banks_3 = 5;
constexpr std::array<uint32_t, rank_3> tensor_shape_array_3 = {2, 1, 3, 4};
constexpr std::array<uint32_t, rank_3> shard_shape_array_3 = {2, 2, 1, 2};
constexpr std::array<uint32_t, num_banks_3> bank_coord_array_3{};
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::ShapeWrapper, tensor_shape_3, tensor_shape_array_3);
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::ShapeWrapper, shard_shape_3, shard_shape_array_3);
USING_STRUCT_FROM_ARRAY_WRAPPER(::detail::BankCoordWrapper, bank_coords_3, bank_coord_array_3);

using test_params_3 = ShardedAccessorParams<
    ShardedAccessorInputs<::detail::DistributionSpec<tensor_shape_3, shard_shape_3, bank_coords_3>>,
    ShardedAccessorExpected<
        ExpectedDSpec<rank_3>{
            .tensor_strides = {12, 12, 4, 1},
            .tensor_volume = 24,
            .shard_strides = {4, 2, 2, 1},
            .shard_volume = 8,
            .shard_grid = {1, 1, 3, 2},
            .shard_grid_strides = {6, 6, 2, 1}},
        ExpectedBankAndOffset{.page_id = 0, .bank_id = 0, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 1, .bank_id = 0, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 2, .bank_id = 1, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 3, .bank_id = 1, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 4, .bank_id = 2, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 5, .bank_id = 2, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 6, .bank_id = 3, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 7, .bank_id = 3, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 8, .bank_id = 4, .bank_offset = 0},
        ExpectedBankAndOffset{.page_id = 9, .bank_id = 4, .bank_offset = 1},
        ExpectedBankAndOffset{.page_id = 10, .bank_id = 0, .bank_offset = 8},
        ExpectedBankAndOffset{.page_id = 11, .bank_id = 0, .bank_offset = 9},
        ExpectedBankAndOffset{.page_id = 12, .bank_id = 0, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 13, .bank_id = 0, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 14, .bank_id = 1, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 15, .bank_id = 1, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 16, .bank_id = 2, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 17, .bank_id = 2, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 18, .bank_id = 3, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 19, .bank_id = 3, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 20, .bank_id = 4, .bank_offset = 4},
        ExpectedBankAndOffset{.page_id = 21, .bank_id = 4, .bank_offset = 5},
        ExpectedBankAndOffset{.page_id = 22, .bank_id = 0, .bank_offset = 12},
        ExpectedBankAndOffset{.page_id = 23, .bank_id = 0, .bank_offset = 13}>>;

}  // namespace params

}  // namespace sharded_accessor_tests

using namespace sharded_accessor_tests;
using namespace tt::tt_metal;

template <typename T>
class ShardedAccessorTests : public ::testing::Test {};

using test_params_t = ::testing::Types<params::test_params_1, params::test_params_2, params::test_params_3>;
TYPED_TEST_SUITE(ShardedAccessorTests, test_params_t);

TYPED_TEST(ShardedAccessorTests, PageLookUp) {
    using dspec_t = TypeParam::inputs::dspec;
    constexpr auto dspec_val = dspec_t{};
    using expected = TypeParam::expected;

    // Create sharded accessor
    auto sharded_accessor = ShardedAccessor<dspec_t, 0>{.bank_base_address = 0};

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
        EXPECT_EQ(bank_id, expected_bank_and_offset.bank_id);
        EXPECT_EQ(bank_offset, expected_bank_and_offset.bank_offset);
    }
}
