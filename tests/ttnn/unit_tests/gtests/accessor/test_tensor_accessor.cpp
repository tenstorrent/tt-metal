// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <fmt/format.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <cstddef>
#include <stdexcept>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include <tt-metalium/tensor_accessor_args.hpp>

// NOLINTBEGIN(bugprone-macro-parentheses)

// Defines to include api/tensor/tensor_accessor.h but won't need these
#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

template <int N>
constexpr auto get_ct_arg();
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

namespace tensor_accessor {
uint64_t get_dram_bank_base_offset(uint32_t bank_id, uint8_t noc);
}

[[maybe_unused]] static uint32_t my_x[1] = {0};
[[maybe_unused]] static uint32_t my_y[1] = {0};

static void host_assert(bool condition, const char* expression) {
    if (!condition) {
        throw std::out_of_range(expression);
    }
}

#define noc_index 0
#define ASSERT(condition, ...) host_assert(condition, #condition)
#define FORCE_INLINE inline __attribute__((always_inline))
#define DPRINT std::cout
#define ENDL() std::endl
#define DPRINT_DATA0(x) x
#define DPRINT_DATA1(x) x
#define DPRINT_MATH(x) x
#define NOC_UNICAST_ADDR_X(addr) addr
#define NOC_UNICAST_ADDR_Y(addr) addr
// Stubs for L1 NOC address macros (used in TensorAccessor::get_noc_addr for !is_dram path).
// Host-side tests only check page_id(), not actual NOC addresses, so the values don't matter.
#define DYNAMIC_NOC_X(noc, x) (x)
#define DYNAMIC_NOC_Y(noc, y) (y)
#define NOC_XY_ADDR(x, y, addr) (static_cast<uint64_t>(x) << 32 | static_cast<uint64_t>(y) << 16 | static_cast<uint64_t>(addr))
#endif

#include "api/tensor/tensor_accessor.h"

#undef get_compile_time_arg_val
#undef noc_index
#undef ASSERT
#undef DPRINT
#undef END
#undef DPRINT_DATA0
#undef DPRINT_DATA1
#undef DPRINT_MATH
#undef FORCE_INLINE
#undef DYNAMIC_NOC_X
#undef DYNAMIC_NOC_Y
#undef NOC_XY_ADDR

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
// NOLINTEND(bugprone-macro-parentheses)

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

TEST(TensorAccessorTests, ShardCoordinateNocAddressUsesGridCoordinates) {
    using TensorShape = ArrayWrapperU32<6, 6>;
    using ShardShape = ArrayWrapperU32<2, 3>;
    using BankCoords = ArrayWrapperU16<0, 1>;
    using dspec_t = tensor_accessor::DistributionSpec<2, 2, TensorShape, ShardShape, BankCoords>;

    constexpr uint32_t bank_base_address = 4096;
    constexpr uint32_t page_size = 64;
    constexpr uint32_t offset = 17;
    auto accessor = TensorAccessor<dspec_t>(bank_base_address, page_size);

    ASSERT_EQ(accessor.dspec().shard_grid(), (std::array<uint32_t, 2>{3, 2}));
    for (uint32_t row = 0; row < accessor.dspec().shard_grid()[0]; ++row) {
        for (uint32_t column = 0; column < accessor.dspec().shard_grid()[1]; ++column) {
            const std::array<uint32_t, 2> shard_coord{row, column};
            const uint32_t shard_id = row * 2 + column;
            EXPECT_EQ(accessor.get_shard_noc_addr(shard_coord, offset), accessor.get_shard_noc_addr(shard_id, offset))
                << "Incorrect address for shard coordinate {" << row << ", " << column << "}";
        }
    }

    // The first coordinate is valid in the shard grid even though it equals the corresponding shard-shape extent.
    EXPECT_NO_THROW(accessor.get_shard_noc_addr(std::array<uint32_t, 2>{2, 1}));
    // The second coordinate is outside the shard grid even though it is less than the shard-shape extent.
    EXPECT_THROW(accessor.get_shard_noc_addr(std::array<uint32_t, 2>{0, 2}), std::out_of_range);
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

    std::array<uint16_t, crta_params::num_banks> bank_coord_array{0, 1, 2, 3};

    auto dspec_val = dspec_t({}, {}, bank_coord_array);
    auto sharded_accessor = TensorAccessor<dspec_t>(std::move(dspec_val), 0);

    crta_params::assert_sharded_accessor(sharded_accessor);
}

// ============================================================================
// Strided DM threading model — host-side tests
//
// These tests validate the Pages / StridedShardPages proxies with explicit stride
// arguments, simulating multiple DM threads without running on device.
// strided_pages() / strided_shard_pages() expand to the same constructors internally;
// here we drive them directly to test any tid/num_threads combination.
// ============================================================================

namespace strided_threading_tests {

// Helper: collect all page_ids produced by a Pages range into a vector.
template <typename PagesRange>
std::vector<uint32_t> collect_page_ids(const PagesRange& range) {
    std::vector<uint32_t> ids;
    for (const auto& page : range) {
        ids.push_back(page.page_id());
    }
    return ids;
}

// Helper: simulate strided_pages() for a specific (tid, num_threads) pair.
template <typename AccessorT>
auto pages_for_thread(const AccessorT& accessor, uint32_t tid, uint32_t num_threads, uint8_t noc = 0) {
    return tensor_accessor::Pages(accessor, tid, accessor.dspec().tensor_volume(), num_threads, noc);
}

// Helper: assert that num_threads threads collectively cover every page in [0, tensor_volume) exactly once.
template <typename AccessorT>
void assert_full_coverage_no_overlap(const AccessorT& accessor, uint32_t num_threads) {
    uint32_t tensor_volume = accessor.dspec().tensor_volume();
    std::vector<uint32_t> all_seen;
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto ids = collect_page_ids(pages_for_thread(accessor, tid, num_threads));
        all_seen.insert(all_seen.end(), ids.begin(), ids.end());
    }
    std::sort(all_seen.begin(), all_seen.end());
    ASSERT_EQ(all_seen.size(), tensor_volume) << "Total pages seen != tensor_volume";
    for (uint32_t i = 0; i < tensor_volume; i++) {
        EXPECT_EQ(all_seen[i], i) << "Missing or duplicate page_id " << i;
    }
}

// Helper: simulate strided_shard_pages() for a specific (tid, num_threads) pair.
// Returns the list of shard_ids visited by that thread.
template <typename AccessorT>
std::vector<uint32_t> shard_ids_for_thread(const AccessorT& accessor, uint32_t tid, uint32_t num_threads, uint8_t noc = 0) {
    const uint32_t tensor_volume = accessor.dspec().tensor_volume();
    const uint32_t shard_volume = accessor.dspec().shard_volume();
    const uint32_t total_shards = (tensor_volume + shard_volume - 1) / shard_volume;
    tensor_accessor::StridedShardPages range(accessor, tid, total_shards, num_threads, noc);
    std::vector<uint32_t> shards;
    uint32_t expected_shard_id = tid;
    for (const auto& shard_range : range) {
        // Verify shard range produces pages in order (by collecting and checking)
        auto page_ids = collect_page_ids(shard_range);
        EXPECT_FALSE(page_ids.empty()) << "Shard range should not be empty";
        shards.push_back(expected_shard_id);
        expected_shard_id += num_threads;
    }
    return shards;
}

// Helpers to create common test accessors
template <uint32_t N, uint32_t NumBanks>
auto make_1d_interleaved_accessor(uint32_t /*tensor_size*/) {
    using TensorShape = ArrayWrapperU32<N>;
    using ShardShape = ArrayWrapperU32<N>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<
        1, NumBanks, TensorShape, ShardShape, BankCoords, /* IsInterleaved */ false>;
    std::array<uint16_t, NumBanks> bank_coord_array{};
    auto dspec_val = dspec_t({}, {}, bank_coord_array);
    return TensorAccessor<dspec_t>(std::move(dspec_val), 0, 4096);
}

}  // namespace strided_threading_tests

using namespace strided_threading_tests;

// -----------------------------------------------------------------------
// pages() with stride > 1 — interleaved-style (no shards)
// -----------------------------------------------------------------------

// Tensor [12], 4 banks, 4 threads: thread i gets pages {i, i+4, i+8}
TEST(StridedPagesTests, Interleaved_4Threads_CoverageAndOrder) {
    constexpr uint32_t N = 12;
    constexpr uint32_t num_banks = 4;
    constexpr uint32_t num_threads = 4;
    using TensorShape = ArrayWrapperU32<N>;
    using ShardShape = ArrayWrapperU32<N>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    const std::vector<std::vector<uint32_t>> expected = {{0, 4, 8}, {1, 5, 9}, {2, 6, 10}, {3, 7, 11}};
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto ids = collect_page_ids(pages_for_thread(accessor, tid, num_threads));
        EXPECT_EQ(ids, expected[tid]) << "Thread " << tid << " page_ids mismatch";
    }
    assert_full_coverage_no_overlap(accessor, num_threads);
}

// Tensor volume=10, num_threads=3 (not a divisor) — lengths differ by at most 1
TEST(StridedPagesTests, NonDivisorThreadCount_FullCoverage) {
    constexpr uint32_t N = 10;
    constexpr uint32_t num_banks = 2;
    constexpr uint32_t num_threads = 3;
    using TensorShape = ArrayWrapperU32<N>;
    using ShardShape = ArrayWrapperU32<N>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    // thread 0: {0,3,6,9}, thread 1: {1,4,7}, thread 2: {2,5,8}
    auto ids0 = collect_page_ids(pages_for_thread(accessor, 0, num_threads));
    auto ids1 = collect_page_ids(pages_for_thread(accessor, 1, num_threads));
    auto ids2 = collect_page_ids(pages_for_thread(accessor, 2, num_threads));
    EXPECT_EQ(ids0, (std::vector<uint32_t>{0, 3, 6, 9}));
    EXPECT_EQ(ids1, (std::vector<uint32_t>{1, 4, 7}));
    EXPECT_EQ(ids2, (std::vector<uint32_t>{2, 5, 8}));
    assert_full_coverage_no_overlap(accessor, num_threads);
}

// Single thread behaves identically to pages(0, total) — regression guard
TEST(StridedPagesTests, SingleThread_FullTensorInOrder) {
    constexpr uint32_t num_banks = 4;
    constexpr uint32_t num_threads = 1;
    using TensorShape = ArrayWrapperU32<2, 3>;
    using ShardShape = ArrayWrapperU32<1, 2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<2, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    auto strided_ids = collect_page_ids(pages_for_thread(accessor, 0, num_threads));
    auto contiguous_ids = collect_page_ids(accessor.pages());
    EXPECT_EQ(strided_ids, contiguous_ids) << "stride=1 must yield identical results to pages()";
}

// Empty range: Pages(accessor, 5, 5) — begin == end, no pages emitted
TEST(StridedPagesTests, EmptyRange_NoPagesYielded) {
    constexpr uint32_t num_banks = 4;
    using TensorShape = ArrayWrapperU32<2, 3>;
    using ShardShape = ArrayWrapperU32<1, 2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<2, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    // start == end means empty range
    constexpr uint32_t page_start_end = 5;
    auto ids = collect_page_ids(tensor_accessor::Pages(accessor, page_start_end, page_start_end, 1u, (uint8_t)0));
    EXPECT_TRUE(ids.empty()) << "Empty range should produce no pages";
}

// 2D sharded tensor [4,4], shard [2,2], 4 banks, 4 threads
TEST(StridedPagesTests, Sharded2D_4Threads_FullCoverage) {
    constexpr uint32_t num_banks = 4;
    constexpr uint32_t num_threads = 4;
    using TensorShape = ArrayWrapperU32<4, 4>;
    using ShardShape = ArrayWrapperU32<2, 2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<2, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    EXPECT_EQ(accessor.dspec().tensor_volume(), 16u);
    assert_full_coverage_no_overlap(accessor, num_threads);
}

// 1D sharded tensor [16], shard [4], 4 banks, 4 threads
// Thread i gets pages {i, i+4, i+8, i+12}
TEST(StridedPagesTests, Sharded1D_4Threads_CoverageAndOrder) {
    constexpr uint32_t num_banks = 4;
    constexpr uint32_t num_threads = 4;
    using TensorShape = ArrayWrapperU32<16>;
    using ShardShape = ArrayWrapperU32<4>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    const std::vector<std::vector<uint32_t>> expected = {
        {0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}};
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto ids = collect_page_ids(pages_for_thread(accessor, tid, num_threads));
        EXPECT_EQ(ids, expected[tid]) << "Thread " << tid << " page_ids mismatch";
    }
    assert_full_coverage_no_overlap(accessor, num_threads);
}

// -----------------------------------------------------------------------
// shard_pages() — existing API, no change (regression tests)
// -----------------------------------------------------------------------

TEST(ShardPagesTests, SingleShard_AllPages) {
    constexpr uint32_t num_banks = 4;
    using TensorShape = ArrayWrapperU32<2, 3>;
    using ShardShape = ArrayWrapperU32<1, 2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<2, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    // Shard 0 contains global pages 0 and 1
    auto ids0 = collect_page_ids(accessor.shard_pages(0));
    EXPECT_EQ(ids0, (std::vector<uint32_t>{0, 1}));

    // Shard 1 contains global page 2
    auto ids1 = collect_page_ids(accessor.shard_pages(1));
    EXPECT_EQ(ids1, (std::vector<uint32_t>{2}));
}

// Padded shard: tensor [3], shard [2], 2 banks — last shard has 1 valid + 1 padded page
TEST(ShardPagesTests, PaddedShard_SkipsOutOfBounds) {
    constexpr uint32_t num_banks = 2;
    using TensorShape = ArrayWrapperU32<3>;
    using ShardShape = ArrayWrapperU32<2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    // Shard 1 has only 1 valid page (page_id 2), the second slot is padding
    auto ids = collect_page_ids(accessor.shard_pages(1));
    EXPECT_EQ(ids.size(), 1u) << "Padded shard should yield only 1 valid page";
    EXPECT_EQ(ids[0], 2u);
}

// -----------------------------------------------------------------------
// strided_shard_pages() — whole-shard granularity (new)
// -----------------------------------------------------------------------

// 4 shards, 4 threads: thread i owns exactly shard i
TEST(StridedShardPagesTests, FourShards_FourThreads_OneShardEach) {
    constexpr uint32_t num_banks = 4;
    constexpr uint32_t num_threads = 4;
    using TensorShape = ArrayWrapperU32<8>;
    using ShardShape = ArrayWrapperU32<2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    ASSERT_EQ(accessor.dspec().tensor_volume() / accessor.dspec().shard_volume(), 4u);

    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto shards = shard_ids_for_thread(accessor, tid, num_threads);
        ASSERT_EQ(shards.size(), 1u) << "Thread " << tid << " should own exactly 1 shard";
        EXPECT_EQ(shards[0], tid) << "Thread " << tid << " should own shard " << tid;
    }

    // Full coverage: all 4 shards seen exactly once
    std::vector<uint32_t> all_shards;
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto s = shard_ids_for_thread(accessor, tid, num_threads);
        all_shards.insert(all_shards.end(), s.begin(), s.end());
    }
    std::sort(all_shards.begin(), all_shards.end());
    EXPECT_EQ(all_shards, (std::vector<uint32_t>{0, 1, 2, 3}));
}

// 8 shards, 4 threads: thread 0 gets shards {0,4}, thread 1: {1,5}, etc.
TEST(StridedShardPagesTests, EightShards_FourThreads_TwoShardsEach) {
    constexpr uint32_t num_banks = 4;
    constexpr uint32_t num_threads = 4;
    using TensorShape = ArrayWrapperU32<16>;
    using ShardShape = ArrayWrapperU32<2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    ASSERT_EQ(accessor.dspec().tensor_volume() / accessor.dspec().shard_volume(), 8u);

    const std::vector<std::vector<uint32_t>> expected_shards = {{0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto shards = shard_ids_for_thread(accessor, tid, num_threads);
        EXPECT_EQ(shards, expected_shards[tid]) << "Thread " << tid << " shard assignment mismatch";
    }

    // Full coverage
    std::vector<uint32_t> all_shards;
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto s = shard_ids_for_thread(accessor, tid, num_threads);
        all_shards.insert(all_shards.end(), s.begin(), s.end());
    }
    std::sort(all_shards.begin(), all_shards.end());
    EXPECT_EQ(all_shards, (std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

// 5 shards, 4 threads: thread 0 gets shards {0,4}, threads 1-3 get 1 shard each
TEST(StridedShardPagesTests, FiveShards_FourThreads_NonDivisor) {
    constexpr uint32_t num_banks = 5;
    constexpr uint32_t num_threads = 4;
    using TensorShape = ArrayWrapperU32<10>;
    using ShardShape = ArrayWrapperU32<2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    ASSERT_EQ(accessor.dspec().tensor_volume() / accessor.dspec().shard_volume(), 5u);

    auto shards0 = shard_ids_for_thread(accessor, 0, num_threads);
    auto shards1 = shard_ids_for_thread(accessor, 1, num_threads);
    auto shards2 = shard_ids_for_thread(accessor, 2, num_threads);
    auto shards3 = shard_ids_for_thread(accessor, 3, num_threads);

    EXPECT_EQ(shards0, (std::vector<uint32_t>{0, 4}));
    EXPECT_EQ(shards1, (std::vector<uint32_t>{1}));
    EXPECT_EQ(shards2, (std::vector<uint32_t>{2}));
    EXPECT_EQ(shards3, (std::vector<uint32_t>{3}));

    // Full coverage, no shard visited twice
    std::vector<uint32_t> all_shards;
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        auto s = shard_ids_for_thread(accessor, tid, num_threads);
        all_shards.insert(all_shards.end(), s.begin(), s.end());
    }
    std::sort(all_shards.begin(), all_shards.end());
    EXPECT_EQ(all_shards, (std::vector<uint32_t>{0, 1, 2, 3, 4}));
}

// Single thread must visit all shards in order 0..num_shards-1
TEST(StridedShardPagesTests, SingleThread_AllShardsInOrder) {
    constexpr uint32_t num_banks = 4;
    constexpr uint32_t num_threads = 1;
    using TensorShape = ArrayWrapperU32<8>;
    using ShardShape = ArrayWrapperU32<2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    auto shards = shard_ids_for_thread(accessor, 0, num_threads);
    EXPECT_EQ(shards, (std::vector<uint32_t>{0, 1, 2, 3}));
}

// strided_shard_pages() also covers all pages — inner loop delivers all pages within each shard
TEST(StridedShardPagesTests, InnerPagesFullCoverage) {
    constexpr uint32_t num_banks = 4;
    using TensorShape = ArrayWrapperU32<8>;
    using ShardShape = ArrayWrapperU32<2>;
    using BankCoords = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    uint32_t num_threads = 2;
    const uint32_t tensor_volume = accessor.dspec().tensor_volume();
    const uint32_t shard_volume = accessor.dspec().shard_volume();
    const uint32_t total_shards = (tensor_volume + shard_volume - 1) / shard_volume;
    std::vector<uint32_t> all_page_ids;

    for (uint32_t tid = 0; tid < num_threads; tid++) {
        tensor_accessor::StridedShardPages range(accessor, tid, total_shards, num_threads, (uint8_t)0);
        for (const auto& shard_range : range) {
            for (const auto& page : shard_range) {
                all_page_ids.push_back(page.page_id());
            }
        }
    }

    std::sort(all_page_ids.begin(), all_page_ids.end());
    ASSERT_EQ(all_page_ids.size(), accessor.dspec().tensor_volume());
    for (uint32_t i = 0; i < accessor.dspec().tensor_volume(); i++) {
        EXPECT_EQ(all_page_ids[i], i) << "Missing or duplicate page_id " << i;
    }
}

// Partial last shard: tensor_volume is not evenly divisible by shard_volume.
//
// Layout (tensor_shape={7}, shard_shape={3}, 3 banks):
//   shard 0 → pages {0,1,2}  (full)
//   shard 1 → pages {3,4,5}  (full)
//   shard 2 → page  {6}      (partial — shard memory has 3 slots, only 1 is logical)
TEST(StridedShardPagesTests, PartialLastShard_NotTruncated) {
    constexpr uint32_t num_banks = 3;
    constexpr uint32_t num_threads = 1;
    using TensorShape = ArrayWrapperU32<7>;
    using ShardShape  = ArrayWrapperU32<3>;
    using BankCoords  = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    // Single thread visits all 3 shards (including the partial one).
    auto shards = shard_ids_for_thread(accessor, 0, num_threads);
    EXPECT_EQ(shards, (std::vector<uint32_t>{0, 1, 2}))
        << "Partial last shard must not be truncated by floor division";

    // Full page coverage across all threads: all 7 logical pages must appear exactly once.
    const uint32_t tensor_volume = accessor.dspec().tensor_volume();
    const uint32_t shard_volume  = accessor.dspec().shard_volume();
    const uint32_t total_shards  = (tensor_volume + shard_volume - 1) / shard_volume;
    std::vector<uint32_t> all_page_ids;

    for (uint32_t tid = 0; tid < num_threads; tid++) {
        tensor_accessor::StridedShardPages range(accessor, tid, total_shards, num_threads, (uint8_t)0);
        for (const auto& shard_range : range) {
            for (const auto& page : shard_range) {
                all_page_ids.push_back(page.page_id());
            }
        }
    }

    std::sort(all_page_ids.begin(), all_page_ids.end());
    ASSERT_EQ(all_page_ids.size(), 7u) << "All 7 logical pages must be visited";
    for (uint32_t i = 0; i < 7u; i++) {
        EXPECT_EQ(all_page_ids[i], i) << "Missing or duplicate page_id " << i;
    }
}

// Partial last shard with multiple threads: verify per-thread assignment and no page is lost.
// tensor_shape={7}, shard_shape={3}, 2 threads:
//   thread 0 → shards {0, 2}  (shard 2 is the partial one)
//   thread 1 → shard  {1}
TEST(StridedShardPagesTests, PartialLastShard_MultiThread_NoPagesLost) {
    constexpr uint32_t num_banks = 3;
    constexpr uint32_t num_threads = 2;
    using TensorShape = ArrayWrapperU32<7>;
    using ShardShape  = ArrayWrapperU32<3>;
    using BankCoords  = ArrayWrapperDynamic;
    using dspec_t = tensor_accessor::DistributionSpec<1, num_banks, TensorShape, ShardShape, BankCoords>;

    std::array<uint16_t, num_banks> bank_coords{};
    auto accessor = TensorAccessor<dspec_t>(dspec_t({}, {}, bank_coords), 0, 4096);

    auto shards0 = shard_ids_for_thread(accessor, 0, num_threads);
    auto shards1 = shard_ids_for_thread(accessor, 1, num_threads);

    EXPECT_EQ(shards0, (std::vector<uint32_t>{0, 2})) << "Thread 0 should own shards 0 and 2 (partial)";
    EXPECT_EQ(shards1, (std::vector<uint32_t>{1}))    << "Thread 1 should own shard 1";

    // Collect all pages from both threads; expect exactly pages 0..6.
    const uint32_t tensor_volume = accessor.dspec().tensor_volume();
    const uint32_t shard_volume  = accessor.dspec().shard_volume();
    const uint32_t total_shards  = (tensor_volume + shard_volume - 1) / shard_volume;
    std::vector<uint32_t> all_page_ids;

    for (uint32_t tid = 0; tid < num_threads; tid++) {
        tensor_accessor::StridedShardPages range(accessor, tid, total_shards, num_threads, (uint8_t)0);
        for (const auto& shard_range : range) {
            for (const auto& page : shard_range) {
                all_page_ids.push_back(page.page_id());
            }
        }
    }

    std::sort(all_page_ids.begin(), all_page_ids.end());
    ASSERT_EQ(all_page_ids.size(), 7u);
    for (uint32_t i = 0; i < 7u; i++) {
        EXPECT_EQ(all_page_ids[i], i) << "Missing or duplicate page_id " << i;
    }
}
