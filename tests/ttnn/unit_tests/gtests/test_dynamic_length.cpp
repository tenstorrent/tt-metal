// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ttnn/dynamic_length.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::dynamic_length::test {

TEST(DynamicLength, SplitScalarPassesThrough) {
    const auto split_length = split(ValidLength{std::size_t{100}}, 128);
    EXPECT_EQ(split_length.value, 100);
    EXPECT_FALSE(split_length.tensor.has_value());
}

TEST(DynamicLength, SplitTensorYieldsCapacity) {
    const auto length = ttnn::zeros(ttnn::Shape({1}), DataType::UINT32, ttnn::ROW_MAJOR_LAYOUT);
    const auto split_length = split(ValidLength{length}, 128);
    EXPECT_EQ(split_length.value, 128);
    EXPECT_TRUE(split_length.tensor.has_value());
}

TEST(DynamicLength, AcceptsLengthUpToCapacity) {
    const std::array axes{
        DynamicAxis{.name = "n", .capacity = 128, .value = 128},
        DynamicAxis{.name = "m", .capacity = 128, .value = 0},
        DynamicAxis{.name = "l", .value = 1'000'000},
    };
    EXPECT_NO_THROW(validate_axes(axes, "op"));
}

TEST(DynamicLength, RejectsLengthAboveCapacity) {
    const std::array axes{DynamicAxis{.name = "n", .capacity = 128, .value = 129}};
    EXPECT_THROW(validate_axes(axes, "op"), std::exception);
}

TEST(DynamicLength, RejectsMisalignedLength) {
    const std::array aligned{DynamicAxis{.name = "n", .capacity = 128, .value = 64, .alignment = 32}};
    EXPECT_NO_THROW(validate_axes(aligned, "op"));
    const std::array misaligned{DynamicAxis{.name = "n", .capacity = 128, .value = 65, .alignment = 32}};
    EXPECT_THROW(validate_axes(misaligned, "op"), std::exception);
}

TEST(DynamicLength, RejectsHostLengthTensor) {
    const auto length = ttnn::zeros(ttnn::Shape({1}), DataType::UINT32, ttnn::ROW_MAJOR_LAYOUT);
    const std::array axes{DynamicAxis{.name = "n", .capacity = 128, .value = 128, .tensor = &length}};
    EXPECT_THROW(validate_axes(axes, "op"), std::exception);
}

TEST(DynamicLength, RejectsWrongDtypeLengthTensor) {
    const auto length = ttnn::zeros(ttnn::Shape({1}), DataType::BFLOAT16, ttnn::ROW_MAJOR_LAYOUT);
    const std::array axes{DynamicAxis{.name = "n", .capacity = 128, .value = 128, .tensor = &length}};
    EXPECT_THROW(validate_axes(axes, "op"), std::exception);
}

struct KeyAttributes {
    uint32_t mode = 0;
    uint32_t length = 0;
};

struct KeyTensorArgs {
    Tensor input;
    Tensor weight;
    std::optional<Tensor> output;
};

Tensor make_tensor(uint32_t rows, uint32_t cols) {
    return ttnn::zeros(ttnn::Shape({1, 1, rows, cols}), DataType::BFLOAT16, ttnn::TILE_LAYOUT);
}

// The rows of input and output are dynamic; the weight's are not.
std::string rows_free_key(const KeyTensorArgs& args, KeyAttributes attributes = {}) {
    DynamicKey dynamic_key;
    dynamic_key.dim(args.input, -2).dim(args.output, 2);
    return program_key(attributes, args, dynamic_key, "op");
}

TEST(DynamicKey, IgnoresDeclaredExtents) {
    const KeyTensorArgs short_args{make_tensor(32, 64), make_tensor(64, 32), make_tensor(32, 32)};
    const KeyTensorArgs long_args{make_tensor(96, 64), make_tensor(64, 32), make_tensor(128, 32)};
    EXPECT_EQ(rows_free_key(short_args), rows_free_key(long_args));
}

TEST(DynamicKey, KeepsEverythingElse) {
    const KeyTensorArgs args{make_tensor(32, 64), make_tensor(64, 32), make_tensor(32, 32)};
    const auto key = rows_free_key(args);
    EXPECT_NE(key, rows_free_key({make_tensor(32, 96), args.weight, args.output}));
    EXPECT_NE(key, rows_free_key({args.input, make_tensor(96, 32), args.output}));
    EXPECT_NE(key, rows_free_key({args.input, args.weight, std::nullopt}));
    EXPECT_NE(key, rows_free_key(args, {.mode = 1}));
    EXPECT_NE(key, rows_free_key(args, {.length = 1}));
    const auto row_major_input = ttnn::zeros(ttnn::Shape({1, 1, 32, 64}), DataType::BFLOAT16, ttnn::ROW_MAJOR_LAYOUT);
    EXPECT_NE(key, rows_free_key({row_major_input, args.weight, args.output}));
}

std::string length_free_key(const KeyAttributes& attributes, const KeyTensorArgs& args) {
    DynamicKey dynamic_key;
    dynamic_key.value(attributes.length);
    return program_key(attributes, args, dynamic_key, "op");
}

TEST(DynamicKey, IgnoresDeclaredValues) {
    const KeyTensorArgs args{make_tensor(32, 64), make_tensor(64, 32), std::nullopt};
    const auto key = length_free_key({.length = 16}, args);
    EXPECT_EQ(key, length_free_key({.length = 64}, args));
    EXPECT_NE(key, length_free_key({.mode = 1, .length = 16}, args));
}

TEST(DynamicKey, KeysDerivedValues) {
    const KeyTensorArgs short_args{make_tensor(32, 64), make_tensor(64, 32), std::nullopt};
    const KeyTensorArgs long_args{make_tensor(96, 64), make_tensor(64, 32), std::nullopt};
    const auto key = [](const KeyTensorArgs& args, uint32_t block_rows) {
        DynamicKey dynamic_key;
        dynamic_key.dim(args.input, -2).depends_on(block_rows);
        return program_key(KeyAttributes{}, args, dynamic_key, "op");
    };
    EXPECT_EQ(key(short_args, 1), key(long_args, 1));
    EXPECT_NE(key(short_args, 1), key(long_args, 3));
}

struct MapTensorArgs {
    std::map<int, Tensor> by_id;
    std::unordered_map<int, Tensor> by_slot;
};

TEST(DynamicKey, ReachesTensorsInMaps) {
    const auto key = [](uint32_t rows) {
        const MapTensorArgs args{{{0, make_tensor(rows, 64)}}, {{0, make_tensor(rows, 32)}}};
        DynamicKey dynamic_key;
        dynamic_key.dim(args.by_id.at(0), -2).dim(args.by_slot.at(0), -2);
        return program_key(KeyAttributes{}, args, dynamic_key, "op");
    };
    EXPECT_EQ(key(32), key(96));
}

TEST(DynamicKey, RejectsDimOutsideTensorArgs) {
    const KeyTensorArgs args{make_tensor(32, 64), make_tensor(64, 32), std::nullopt};
    const auto unrelated = make_tensor(32, 64);
    DynamicKey dynamic_key;
    dynamic_key.dim(unrelated, -2);
    EXPECT_THROW(program_key(KeyAttributes{}, args, dynamic_key, "op"), std::exception);
}

TEST(DynamicKey, RejectsDimOutOfRange) {
    const KeyTensorArgs args{make_tensor(32, 64), make_tensor(64, 32), std::nullopt};
    DynamicKey dynamic_key;
    dynamic_key.dim(args.input, 4);
    EXPECT_THROW(program_key(KeyAttributes{}, args, dynamic_key, "op"), std::exception);
}

TEST(DynamicKey, RejectsValueOutsideAttributes) {
    const KeyAttributes attributes;
    const KeyAttributes copy = attributes;
    const KeyTensorArgs args{make_tensor(32, 64), make_tensor(64, 32), std::nullopt};
    DynamicKey dynamic_key;
    dynamic_key.value(copy.length);
    EXPECT_THROW(program_key(attributes, args, dynamic_key, "op"), std::exception);
}

}  // namespace ttnn::dynamic_length::test
