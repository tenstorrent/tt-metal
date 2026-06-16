// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tests for the host-only layout conversion tt::tt_metal::to_layout(const HostTensor&, Layout).
//
// Group 1 (no sharding): a matrix of {conversion direction} x {dtype}. For each cell, to_layout's
// output is compared against an INDEPENDENTLY constructed reference tensor built directly in the
// target layout via HostTensor::from_vector (which reaches the tiling code through a different path,
// encode_tensor_data). Matching that reference byte-for-byte verifies the conversion produced exactly
// what a freshly constructed tensor would hold.
//
// Plus dedicated tests for the special dtypes whose to_layout behavior is not a generic conversion:
// BFLOAT8_B / BFLOAT4_B are no-ops, and FP8_E4M3 is ROW_MAJOR-only (throws when asked to tilize).

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/shape.hpp>

namespace tt::tt_metal {
namespace {

// Builds an interleaved spec with a chosen layout. The output memory config of to_layout is
// interleaved (it rebuilds the spec with MemoryConfig{}), so the reference tensor uses interleaved too.
TensorSpec make_spec(const Shape& shape, DataType dtype, Layout layout) {
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config));
}

// A deterministic ramp that is in-range for every dtype under test (incl. UINT8).
template <typename T>
std::vector<T> make_ramp(size_t count) {
    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(static_cast<float>(i % 251));
    }
    return data;
}

// Content-equivalence check that works for every dtype (including FP8_E4M3, which has no typed
// reader): pull the host shard out of each tensor and compare the raw bytes directly. The tensors
// here are single-device, so each has exactly one shard.
void expect_equal_shard_data(const HostTensor& actual, const HostTensor& expected) {
    const HostBuffer actual_buffer = host_buffer::get_host_buffer(actual);
    const HostBuffer expected_buffer = host_buffer::get_host_buffer(expected);
    const auto actual_bytes = actual_buffer.view_bytes();
    const auto expected_bytes = expected_buffer.view_bytes();
    ASSERT_EQ(actual_bytes.size(), expected_bytes.size());
    EXPECT_TRUE(std::equal(actual_bytes.begin(), actual_bytes.end(), expected_bytes.begin()));
}

std::string dtype_to_str(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "FLOAT32";
        case DataType::BFLOAT16: return "BFLOAT16";
        case DataType::INT32: return "INT32";
        case DataType::UINT32: return "UINT32";
        case DataType::UINT16: return "UINT16";
        case DataType::UINT8: return "UINT8";
        default: return "UNKNOWN";
    }
}

using ToLayoutParam = std::tuple<std::pair<Layout, Layout>, DataType>;

std::string to_layout_param_name(const ::testing::TestParamInfo<ToLayoutParam>& info) {
    const auto direction = std::get<0>(info.param);
    const auto dtype = std::get<1>(info.param);
    const std::string dir = (direction.first == Layout::ROW_MAJOR) ? "RmToTile" : "TileToRm";
    return dir + "_" + dtype_to_str(dtype);
}

class HostTensorToLayoutMatrix : public ::testing::TestWithParam<ToLayoutParam> {};

TEST_P(HostTensorToLayoutMatrix, MatchesFreshConstruction) {
    const auto direction = std::get<0>(GetParam());
    const auto dtype = std::get<1>(GetParam());
    const auto src = direction.first;
    const auto tgt = direction.second;

    // to_layout(src -> tgt) must produce the same physical contents as a tensor freshly constructed
    // directly in the target layout from the same logical data.
    auto check_matches_fresh_construction = [&]<typename T>() {
        const Shape shape{64, 64};  // 2x2 tiles; logical == physical (no padding edge cases)
        const auto data = make_ramp<T>(shape.volume());

        const auto source = HostTensor::from_vector<T>(data, make_spec(shape, dtype, src));
        const auto result = to_layout(source, tgt);
        const auto expected = HostTensor::from_vector<T>(data, make_spec(shape, dtype, tgt));

        EXPECT_EQ(result.layout(), tgt);
        EXPECT_EQ(result.logical_shape(), shape);
        EXPECT_EQ(result.padded_shape(), expected.padded_shape());
        EXPECT_EQ(result.dtype(), dtype);

        expect_equal_shard_data(result, expected);
    };

    switch (dtype) {
        case DataType::FLOAT32: check_matches_fresh_construction.operator()<float>(); break;
        case DataType::BFLOAT16: check_matches_fresh_construction.operator()<bfloat16>(); break;
        case DataType::INT32: check_matches_fresh_construction.operator()<int32_t>(); break;
        case DataType::UINT32: check_matches_fresh_construction.operator()<uint32_t>(); break;
        case DataType::UINT16: check_matches_fresh_construction.operator()<uint16_t>(); break;
        case DataType::UINT8: check_matches_fresh_construction.operator()<uint8_t>(); break;
        default: FAIL() << "Unhandled dtype in matrix: " << dtype_to_str(dtype);
    }
}

INSTANTIATE_TEST_SUITE_P(
    HostTensorToLayout,
    HostTensorToLayoutMatrix,
    ::testing::Combine(
        ::testing::Values(std::pair{Layout::ROW_MAJOR, Layout::TILE}, std::pair{Layout::TILE, Layout::ROW_MAJOR}),
        // Block-float formats (BFLOAT8_B / BFLOAT4_B) and FP8_E4M3 are intentionally excluded from
        // this matrix: their to_layout is a no-op / throws rather than a real conversion. They are
        // covered by the dedicated special-dtype tests below.
        ::testing::Values(
            DataType::FLOAT32,
            DataType::BFLOAT16,
            DataType::INT32,
            DataType::UINT32,
            DataType::UINT16,
            DataType::UINT8)),
    to_layout_param_name);

// ----------------------------------------------------------------------------------------------------
// Unsupported conversions.
// ----------------------------------------------------------------------------------------------------

// G2: converting a block-float tensor (BFLOAT8_B / BFLOAT4_B) from TILE to ROW_MAJOR.
//
// This SHOULD be a hard error. Block-float formats are only meaningful in TILE layout, so a request to
// row-major-ize them is nonsensical and ought to be rejected. Today it is not: to_row_major_layout (in
// tensor_apis.cpp) deliberately short-circuits these dtypes and returns the input tensor unchanged,
// emitting only a log warning -- see the #43763 TODO at that call site. The tests below pin down that
// current no-op behavior: the call simply does not throw. When #43763 flips the no-op into a hard
// failure, replace the EXPECT_NO_THROW assertions here with EXPECT_ANY_THROW.
TEST(HostTensorToLayout, Bfloat8BToRowMajorDoesNotThrow) {
    const Shape shape{32, 32};
    const auto data = make_ramp<float>(shape.volume());
    const auto source = HostTensor::from_vector<float>(data, make_spec(shape, DataType::BFLOAT8_B, Layout::TILE));

    EXPECT_NO_THROW(std::ignore = to_layout(source, Layout::ROW_MAJOR));
}

TEST(HostTensorToLayout, Bfloat4BToRowMajorDoesNotThrow) {
    const Shape shape{32, 32};
    const auto data = make_ramp<float>(shape.volume());
    const auto source = HostTensor::from_vector<float>(data, make_spec(shape, DataType::BFLOAT4_B, Layout::TILE));

    EXPECT_NO_THROW(std::ignore = to_layout(source, Layout::ROW_MAJOR));
}

// G3: dtypes that cannot be converted to TILE. FP8_E4M3 is constrained to ROW_MAJOR, so tilizing it
// is a caller error. (Block-float formats are TILE-native, so they have no "cannot tilize" case.)
TEST(HostTensorToLayout, Fp8E4m3CannotConvertToTile) {
    const Shape shape{32, 32};
    const auto data = make_ramp<float>(shape.volume());
    const auto source = HostTensor::from_vector<float>(data, make_spec(shape, DataType::FP8_E4M3, Layout::ROW_MAJOR));

    EXPECT_ANY_THROW(std::ignore = to_layout(source, Layout::TILE));
}

}  // namespace
}  // namespace tt::tt_metal
