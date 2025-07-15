// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <fmt/format.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <cstddef>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include <ttnn/tensor/tensor_accessor_args.hpp>

// Defines to include tt_metal/hw/inc/accessor/tensor_accessor.h but won't need these
#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

template <int N>
constexpr auto get_ct_arg();
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

template <typename T>
constexpr T get_arg_val(size_t idx);

template <typename T>
constexpr T get_common_arg_val(size_t idx);

constexpr uint32_t get_arg_val(int arg_idx);
namespace tensor_accessor {
uint64_t get_dram_bank_base_offset(uint32_t base_address, uint32_t bank_id, uint8_t noc);
}

#define noc_index 0
#define ASSERT(condition, ...)
#define FORCE_INLINE inline __attribute__((always_inline))
#define DPRINT std::cout
#define ENDL() std::endl
#define DPRINT_DATA0(x) x
#define DPRINT_DATA1(x) x
#define DPRINT_MATH(x) x
#endif

#include "tt_metal/hw/inc/accessor/tensor_accessor.h"

#undef get_compile_time_arg_val
#undef noc_index
#undef DPRINT
#undef END
#undef DPRINT_DATA0
#undef DPRINT_DATA1
#undef DPRINT_MATH
#undef FORCE_INLINE

template <size_t... Dims>
using ArrayWrapperU32 = tensor_accessor::ArrayStaticWrapperU32<Dims...>;

template <size_t... Dims>
using ArrayWrapperU16 = tensor_accessor::ArrayStaticWrapperU16<Dims...>;

// GCC does not support proper passing of variadic templates in template aliases, so we define a struct to be used with
// USING_STRUCT_FROM_ARRAY_WRAPPER
template <size_t... Dims>
struct ArrayWrapperU32Class {
    using type = tensor_accessor::ArrayStaticWrapperU32<Dims...>;
};

template <size_t... Dims>
struct ArrayWrapperU16Class {
    using type = tensor_accessor::ArrayStaticWrapperU16<Dims...>;
};

using ArrayWrapperDynamic = tensor_accessor::ArrayDynamicWrapper;

template <uint32_t Size>
// using DevSpan = tensor_accessor::Span<uint32_t, Size>;
using DevSpan = std::array<uint32_t, Size>;

// If inputs are passed as constexpr arrays, we can use this style to directly create the structs
// Example:
//    constexpr std::array<uint32_t, 3> tensor_shape_array_1 = {1, 2, 3};
//    USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU32, tensor_shape_1, tensor_shape_array);
//    static_assert(std::is_same_v<tensor_shape_1, ArrayWrapperU32<1, 2, 3>>);
template <template <size_t...> class Wrapper, typename F, size_t... Is>
constexpr auto make_struct_from_array_wrapper(F, std::index_sequence<Is...>) -> Wrapper<F{}()[Is]...>::type;

#define USING_STRUCT_FROM_ARRAY_WRAPPER(Wrapper, name, arr) \
    struct name##_fn {                                      \
        constexpr auto operator()() const { return (arr); } \
    };                                                      \
    using name =                                            \
        decltype(make_struct_from_array_wrapper<Wrapper>(name##_fn{}, std::make_index_sequence<(arr).size()>{}))

namespace sharded_accessor_tests {

template <typename DSpecT>
struct TensorAccessorInputs {
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
struct TensorAccessorExpected {
    static constexpr auto dspec = ExpectedDSpecVal;
    static constexpr auto bank_and_offset =
        std::array<ExpectedBankAndOffset, sizeof...(ExpectedBankAndOffsetVals)>{ExpectedBankAndOffsetVals...};
};

template <typename Inputs, typename Expected>
struct TensorAccessorParams {
    using inputs = Inputs;
    using expected = Expected;
};

// Completely compile time case test //

namespace params {

constexpr size_t rank_1 = 2;
constexpr size_t num_banks_1 = 4;
constexpr std::array<uint32_t, rank_1> tensor_shape_array_1 = {2, 3};
constexpr std::array<uint32_t, rank_1> shard_shape_array_1 = {1, 2};
constexpr std::array<uint32_t, num_banks_1> bank_coord_array_1{};
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU32Class, tensor_shape_1, tensor_shape_array_1);
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU32Class, shard_shape_1, shard_shape_array_1);
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU16Class, bank_coords_1, bank_coord_array_1);

using test_params_1 = TensorAccessorParams<
    TensorAccessorInputs<
        tensor_accessor::DistributionSpec<rank_1, num_banks_1, tensor_shape_1, shard_shape_1, bank_coords_1>>,
    TensorAccessorExpected<
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
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU32Class, tensor_shape_2, tensor_shape_array_2);
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU32Class, shard_shape_2, shard_shape_array_2);
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU16Class, bank_coords_2, bank_coord_array_2);

using test_params_2 = TensorAccessorParams<
    TensorAccessorInputs<
        tensor_accessor::DistributionSpec<rank_2, num_banks_2, tensor_shape_2, shard_shape_2, bank_coords_2>>,
    TensorAccessorExpected<
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
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU32Class, tensor_shape_3, tensor_shape_array_3);
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU32Class, shard_shape_3, shard_shape_array_3);
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU16Class, bank_coords_3, bank_coord_array_3);

using test_params_3 = TensorAccessorParams<
    TensorAccessorInputs<
        tensor_accessor::DistributionSpec<rank_3, num_banks_3, tensor_shape_3, shard_shape_3, bank_coords_3>>,
    TensorAccessorExpected<
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
class TensorAccessorTests : public ::testing::Test {};

using test_params_t = ::testing::Types<params::test_params_1, params::test_params_2, params::test_params_3>;
TYPED_TEST_SUITE(TensorAccessorTests, test_params_t);

TYPED_TEST(TensorAccessorTests, PageLookUp) {
    using dspec_t = TypeParam::inputs::dspec;
    constexpr auto dspec_val = dspec_t{};
    using expected = TypeParam::expected;

    // Create sharded accessor
    auto sharded_accessor = TensorAccessor<dspec_t>(dspec_val, 0);

    // Check that the computed values in DSpec match the expected values
    ASSERT_EQ(dspec_val.tensor_strides(), expected::dspec.tensor_strides);
    ASSERT_EQ(dspec_val.tensor_volume(), expected::dspec.tensor_volume);
    ASSERT_EQ(dspec_val.shard_strides(), expected::dspec.shard_strides);
    ASSERT_EQ(dspec_val.shard_volume(), expected::dspec.shard_volume);
    ASSERT_EQ(dspec_val.shard_grid(), expected::dspec.shard_grid);
    ASSERT_EQ(dspec_val.shard_grid_strides(), expected::dspec.shard_grid_strides);

    // Check that the computed bank and offset values match the expected values
    for (const auto& expected_bank_and_offset : expected::bank_and_offset) {
        auto [bank_id, bank_offset] = sharded_accessor.get_bank_and_offset(expected_bank_and_offset.page_id);
        EXPECT_EQ(bank_id, expected_bank_and_offset.bank_id);
        EXPECT_EQ(bank_offset, expected_bank_and_offset.bank_offset);
    }
}

namespace crta_params {
constexpr size_t rank = 2;
constexpr size_t num_banks = 4;
constexpr std::array<uint32_t, num_banks> bank_coord_array{};
USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU16Class, bank_coords, crta_params::bank_coord_array);
using expected = TensorAccessorExpected<
    ExpectedDSpec<rank>{
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
    ExpectedBankAndOffset{.page_id = 5, .bank_id = 3, .bank_offset = 0}>;

template <typename DSpecT>
void assert_dspec(const DSpecT& dspec_val) {
    auto cmpare_array = [](const auto& a, const auto& b) { return std::equal(a.begin(), a.end(), b.begin()); };
    ASSERT_TRUE(cmpare_array(dspec_val.tensor_strides(), expected::dspec.tensor_strides));
    ASSERT_TRUE(cmpare_array(dspec_val.shard_strides(), expected::dspec.shard_strides));
    ASSERT_TRUE(cmpare_array(dspec_val.shard_grid(), expected::dspec.shard_grid));
    ASSERT_TRUE(cmpare_array(dspec_val.shard_grid_strides(), expected::dspec.shard_grid_strides));
    ASSERT_EQ(dspec_val.tensor_volume(), expected::dspec.tensor_volume);
    ASSERT_EQ(dspec_val.shard_volume(), expected::dspec.shard_volume);
}

template <typename ShardAccessorT>
void assert_sharded_accessor(const ShardAccessorT& sharded_accessor) {
    // Check that the computed values in DSpec match the expected values
    assert_dspec(sharded_accessor.dspec());

    // Check that the computed bank and offset values match the expected values
    for (const auto& expected_bank_and_offset : expected::bank_and_offset) {
        auto [bank_id, bank_offset] = sharded_accessor.get_bank_and_offset(expected_bank_and_offset.page_id);
        EXPECT_EQ(bank_id, expected_bank_and_offset.bank_id);
        EXPECT_EQ(bank_offset, expected_bank_and_offset.bank_offset);
    }
}
}  // namespace crta_params

TEST(TensorAccessorTestsCRTA, RuntimeTensorRuntimeShardShapeCompileTimeBanks) {
    using TensorShapeT = ArrayWrapperDynamic;
    using ShardShapeT = ArrayWrapperDynamic;
    USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU16Class, bank_coords, crta_params::bank_coord_array);
    using dspec_t = tensor_accessor::
        DistributionSpec<crta_params::rank, crta_params::num_banks, TensorShapeT, ShardShapeT, bank_coords>;

    std::array<uint32_t, crta_params::rank> tensor_shape_array = {2, 3};
    std::array<uint32_t, crta_params::rank> shard_shape_array = {1, 2};

    auto dspec_val = dspec_t(tensor_shape_array, shard_shape_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}

TEST(TensorAccessorTestsCRTA, RuntimeTensorCompiletimeShardShapeCompileTimeBanks) {
    using TensorShapeT = ArrayWrapperDynamic;
    using ShardShapeT = ArrayWrapperU32<1, 2>;
    USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU16Class, bank_coords, crta_params::bank_coord_array);
    using dspec_t = tensor_accessor::
        DistributionSpec<crta_params::rank, crta_params::num_banks, TensorShapeT, ShardShapeT, bank_coords>;

    std::array<uint32_t, crta_params::rank> tensor_shape_array = {2, 3};

    auto dspec_val = dspec_t(tensor_shape_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}

TEST(TensorAccessorTestsCRTA, CompiletimeTensorRuntimeShardShapeCompileTimeBanks) {
    using TensorShapeT = ArrayWrapperU32<2, 3>;
    using ShardShapeT = ArrayWrapperDynamic;
    USING_STRUCT_FROM_ARRAY_WRAPPER(ArrayWrapperU16Class, bank_coords, crta_params::bank_coord_array);

    using dspec_t = tensor_accessor::
        DistributionSpec<crta_params::rank, crta_params::num_banks, TensorShapeT, ShardShapeT, bank_coords>;
    std::array<uint32_t, crta_params::rank> shard_shape_array = {1, 2};

    auto dspec_val = dspec_t({}, shard_shape_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}

TEST(TensorAccessorTestsCRTA, RuntimeTensorRuntimeShardShapeRuntimeBanks) {
    using TensorShapeT = ArrayWrapperDynamic;
    using ShardShapeT = ArrayWrapperDynamic;
    using bank_coords = ArrayWrapperDynamic;

    using dspec_t = tensor_accessor::
        DistributionSpec<crta_params::rank, crta_params::num_banks, TensorShapeT, ShardShapeT, bank_coords>;

    std::array<uint32_t, crta_params::rank> tensor_shape_array = {2, 3};
    std::array<uint32_t, crta_params::rank> shard_shape_array = {1, 2};
    std::array<uint16_t, crta_params::num_banks> bank_coord_array{0, 1, 2, 3};

    auto dspec_val = dspec_t(tensor_shape_array, shard_shape_array, bank_coord_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}

TEST(TensorAccessorTestsCRTA, RuntimeTensorCompiletimeShardShapeRuntimeBanks) {
    using TensorShapeT = ArrayWrapperDynamic;
    using ShardShapeT = ArrayWrapperU32<1, 2>;
    using bank_coords = ArrayWrapperDynamic;

    using dspec_t = tensor_accessor::
        DistributionSpec<crta_params::rank, crta_params::num_banks, TensorShapeT, ShardShapeT, bank_coords>;

    std::array<uint32_t, crta_params::rank> tensor_shape_array = {2, 3};
    std::array<uint16_t, crta_params::num_banks> bank_coord_array{0, 1, 2, 3};

    auto dspec_val = dspec_t(tensor_shape_array, {}, bank_coord_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}

TEST(TensorAccessorTestsCRTA, CompiletimeTensorRuntimeShardShapeRuntimeBanks) {
    using TensorShapeT = ArrayWrapperU32<2, 3>;
    using ShardShapeT = ArrayWrapperDynamic;
    using bank_coords = ArrayWrapperDynamic;

    using dspec_t = tensor_accessor::
        DistributionSpec<crta_params::rank, crta_params::num_banks, TensorShapeT, ShardShapeT, bank_coords>;

    std::array<uint32_t, crta_params::rank> shard_shape_array = {1, 2};
    std::array<uint16_t, crta_params::num_banks> bank_coord_array{0, 1, 2, 3};

    auto dspec_val = dspec_t({}, shard_shape_array, bank_coord_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}

TEST(TensorAccessorTestsCRTA, CompiletimeTensorCompileTimeShardShapeRuntimeBanks) {
    using TensorShapeT = ArrayWrapperU32<2, 3>;
    using ShardShapeT = ArrayWrapperU32<1, 2>;
    using bank_coords = ArrayWrapperDynamic;

    using dspec_t = tensor_accessor::
        DistributionSpec<crta_params::rank, crta_params::num_banks, TensorShapeT, ShardShapeT, bank_coords>;

    std::array<uint32_t, crta_params::rank> shard_shape_array = {1, 2};
    std::array<uint16_t, crta_params::num_banks> bank_coord_array{0, 1, 2, 3};

    auto dspec_val = dspec_t({}, {}, bank_coord_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}
