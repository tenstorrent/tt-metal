// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor_host_pad_apis.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Not present in the ramp, so any element holding it was written by pad. Exact in every tested dtype.
constexpr float kPadValue = 250.0f;

template <typename T>
DataType dtype_of() {
    if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        return DataType::BFLOAT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DataType::UINT16;
    } else {
        static_assert(std::is_same_v<T, uint8_t>);
        return DataType::UINT8;
    }
}

template <typename T>
std::vector<T> make_ramp(size_t count, size_t offset = 0) {
    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(static_cast<float>(((i + offset) % 200) + 1));
    }
    return data;
}

TensorSpec row_major_spec(const Shape& shape, DataType dtype) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
}

// Raw buffer of one shard, i.e. the physical (padded-shape) data.
template <typename T>
std::vector<T> shard_values(const HostTensor& tensor, const distributed::MeshCoordinate& coord) {
    auto shard = tensor.buffer().get_shard(coord);
    if (!shard.has_value()) {
        ADD_FAILURE() << "missing shard at " << coord;
        return {};
    }
    auto view = shard->view_as<T>();
    return std::vector<T>(view.begin(), view.end());
}

template <typename T>
std::vector<T> first_shard_values(const HostTensor& tensor) {
    const auto& coords = tensor.buffer().shard_coords();
    if (coords.empty()) {
        ADD_FAILURE() << "tensor has no shards";
        return {};
    }
    return shard_values<T>(tensor, *coords.begin());
}

template <typename T>
std::vector<T> reference_pad(
    const std::vector<T>& input,
    const Shape& input_shape,
    const Shape& output_shape,
    const Shape& input_start,
    T pad_value) {
    std::vector<T> output(output_shape.volume(), pad_value);
    const int rank = static_cast<int>(input_shape.rank());
    for (size_t flat = 0; flat < input.size(); ++flat) {
        size_t remainder = flat;
        size_t output_index = 0;
        size_t output_stride = 1;
        for (int dim = rank - 1; dim >= 0; --dim) {
            const size_t coord = remainder % input_shape[dim];
            remainder /= input_shape[dim];
            output_index += (coord + input_start[dim]) * output_stride;
            output_stride *= output_shape[dim];
        }
        output[output_index] = input[flat];
    }
    return output;
}

template <typename T>
std::vector<T> reference_unpad(
    const std::vector<T>& input, const Shape& input_shape, const Shape& start, const Shape& end) {
    std::vector<T> output;
    const int rank = static_cast<int>(input_shape.rank());
    for (size_t flat = 0; flat < input.size(); ++flat) {
        size_t remainder = flat;
        bool inside = true;
        for (int dim = rank - 1; dim >= 0; --dim) {
            const size_t coord = remainder % input_shape[dim];
            remainder /= input_shape[dim];
            inside = inside && coord >= start[dim] && coord < end[dim];
        }
        if (inside) {
            output.push_back(input[flat]);
        }
    }
    return output;
}

Shape extent(const Shape& start, const Shape& end) {
    ttsl::SmallVector<uint32_t> dims;
    for (int dim = 0; dim < static_cast<int>(start.rank()); ++dim) {
        dims.push_back(end[dim] - start[dim]);
    }
    return Shape(std::move(dims));
}

Shape end_of(const Shape& start, const Shape& size) {
    ttsl::SmallVector<uint32_t> dims;
    for (int dim = 0; dim < static_cast<int>(start.rank()); ++dim) {
        dims.push_back(start[dim] + size[dim]);
    }
    return Shape(std::move(dims));
}

struct MultiDeviceSource {
    HostTensor tensor;
    std::vector<float> data_0;
    std::vector<float> data_1;
};

// 1x2 host mesh with different data per shard (separate allocations, so transform() runs once per shard).
MultiDeviceSource make_multi_device_source(const Shape& shape) {
    auto data_0 = make_ramp<float>(shape.volume());
    auto data_1 = make_ramp<float>(shape.volume(), /*offset=*/100);
    auto distributed_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 2));
    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 0), [&]() { return HostBuffer(std::vector<float>(data_0)); });
    distributed_buffer.emplace_shard(
        distributed::MeshCoordinate(0, 1), [&]() { return HostBuffer(std::vector<float>(data_1)); });
    auto topology = TensorTopology::create_fully_replicated_tensor_topology(distributed::MeshShape(1, 2));
    auto tensor = host_tensor_from_buffer_with_topology(
        std::move(distributed_buffer), row_major_spec(shape, DataType::FLOAT32), topology);
    return {std::move(tensor), std::move(data_0), std::move(data_1)};
}

HostTensor make_float_source(const Shape& shape, const std::vector<float>& data) {
    return HostTensor::from_vector<float>(data, row_major_spec(shape, DataType::FLOAT32));
}

// ---------------------------------------------------------------------------------------------
// pad: shapes and values across ranks and before/after placements
// ---------------------------------------------------------------------------------------------

struct PadCase {
    std::string name;
    Shape input_shape;
    Shape output_shape;
    Shape input_start;
};

class HostPadUnpadPad : public ::testing::TestWithParam<PadCase> {};

TEST_P(HostPadUnpadPad, MatchesReference) {
    const auto& param = GetParam();
    auto data = make_ramp<float>(param.input_shape.volume());
    auto source = make_float_source(param.input_shape, data);

    auto padded = pad(source, param.output_shape, param.input_start, kPadValue);

    EXPECT_EQ(padded.padded_shape(), param.output_shape);
    // The documented post-condition covers only the padded shape; host pad keeps the input logical shape.
    EXPECT_EQ(padded.logical_shape(), param.input_shape);
    EXPECT_EQ(padded.dtype(), DataType::FLOAT32);
    EXPECT_EQ(padded.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(padded.memory_config(), MemoryConfig{});
    EXPECT_EQ(
        first_shard_values<float>(padded),
        reference_pad<float>(data, param.input_shape, param.output_shape, param.input_start, kPadValue));
}

INSTANTIATE_TEST_SUITE_P(
    HostPadUnpad,
    HostPadUnpadPad,
    ::testing::Values(
        PadCase{"rank2_after", Shape{2, 3}, Shape{4, 6}, Shape{0, 0}},
        PadCase{"rank2_before", Shape{2, 3}, Shape{4, 6}, Shape{2, 3}},
        PadCase{"rank2_both", Shape{2, 3}, Shape{4, 6}, Shape{1, 1}},
        PadCase{"rank3_both", Shape{2, 2, 3}, Shape{3, 4, 5}, Shape{1, 1, 1}},
        PadCase{"rank4_after_last_dim", Shape{1, 1, 2, 4}, Shape{1, 1, 2, 5}, Shape{0, 0, 0, 0}},
        PadCase{"rank4_before_last_dim", Shape{1, 1, 2, 4}, Shape{1, 1, 2, 5}, Shape{0, 0, 0, 1}},
        PadCase{"rank4_before_upper_dim", Shape{1, 2, 2, 3}, Shape{3, 2, 2, 3}, Shape{1, 0, 0, 0}},
        PadCase{"rank4_both_all_dims", Shape{1, 2, 2, 3}, Shape{3, 4, 5, 6}, Shape{1, 1, 2, 2}},
        PadCase{"rank5_both", Shape{1, 2, 1, 2, 3}, Shape{2, 3, 2, 4, 5}, Shape{1, 0, 1, 1, 1}}),
    [](const ::testing::TestParamInfo<PadCase>& info) { return info.param.name; });

TEST(HostPadUnpad, PadRank1) {
    const Shape input_shape{5};
    const Shape output_shape{8};
    const Shape input_start{2};
    auto data = make_ramp<float>(input_shape.volume());
    auto source = make_float_source(input_shape, data);

    auto padded = pad(source, output_shape, input_start, kPadValue);

    EXPECT_EQ(padded.padded_shape(), output_shape);
    EXPECT_EQ(padded.logical_shape(), input_shape);
    EXPECT_EQ(
        first_shard_values<float>(padded),
        reference_pad<float>(data, input_shape, output_shape, input_start, kPadValue));
}

// from_torch with a pad value pads rank 0/1 tensors into a rank-2 tile shape; the input gets leading 1s.
TEST(HostPadUnpad, PadRank1IntoRank2Output) {
    const Shape input_shape{5};
    const Shape output_shape{2, 8};
    const Shape input_start{1, 2};
    auto data = make_ramp<float>(input_shape.volume());
    auto source = make_float_source(input_shape, data);

    auto padded = pad(source, output_shape, input_start, kPadValue);

    EXPECT_EQ(padded.padded_shape(), output_shape);
    EXPECT_EQ(padded.logical_shape(), input_shape);
    EXPECT_EQ(
        first_shard_values<float>(padded),
        reference_pad<float>(data, Shape{1, 5}, output_shape, input_start, kPadValue));
}

// A second pad reads the first pad's physical buffer (its padded shape), not its logical shape.
TEST(HostPadUnpad, ChainedPadReadsPhysicalData) {
    const Shape input_shape{1, 1, 2, 4};
    const Shape first_shape{1, 1, 2, 5};
    const Shape first_start{0, 0, 0, 0};
    const Shape second_shape{1, 1, 3, 6};
    const Shape second_start{0, 0, 1, 1};
    auto data = make_ramp<float>(input_shape.volume());
    auto source = make_float_source(input_shape, data);

    auto first = pad(source, first_shape, first_start, kPadValue);
    auto second = pad(first, second_shape, second_start, kPadValue);

    auto first_physical = reference_pad<float>(data, input_shape, first_shape, first_start, kPadValue);
    EXPECT_EQ(second.padded_shape(), second_shape);
    EXPECT_EQ(second.logical_shape(), input_shape);
    EXPECT_EQ(
        first_shard_values<float>(second),
        reference_pad<float>(first_physical, first_shape, second_shape, second_start, kPadValue));
}

// The whole padded input, placed at input_tensor_start, must fit in output_padded_shape.
TEST(HostPadUnpad, PadRejectsInputThatDoesNotFit) {
    const Shape input_shape{1, 1, 3, 5};
    auto source = make_float_source(input_shape, make_ramp<float>(input_shape.volume()));

    EXPECT_ANY_THROW(std::ignore = pad(source, Shape{1, 1, 3, 5}, Shape{0, 0, 1, 1}, kPadValue));
}

TEST(HostPadUnpad, PadRejectsRankMismatch) {
    const Shape input_shape{1, 1, 3, 5};
    auto source = make_float_source(input_shape, make_ramp<float>(input_shape.volume()));

    EXPECT_ANY_THROW(std::ignore = pad(source, Shape{1, 4, 6}, Shape{0, 0, 0}, kPadValue));
}

// The bounds check must hold in every build type, not only those that enable TT_ASSERT.
TEST(HostPadUnpad, UnpadRejectsEndPastInput) {
    const Shape input_shape{1, 1, 3, 5};
    auto source = make_float_source(input_shape, make_ramp<float>(input_shape.volume()));

    EXPECT_ANY_THROW(std::ignore = unpad(source, Shape{0, 0, 0, 0}, Shape{1, 1, 4, 5}));
}

// ---------------------------------------------------------------------------------------------
// unpad: end index is exclusive
// ---------------------------------------------------------------------------------------------

struct UnpadCase {
    std::string name;
    Shape input_shape;
    Shape start;
    Shape end;
};

class HostPadUnpadUnpad : public ::testing::TestWithParam<UnpadCase> {};

TEST_P(HostPadUnpadUnpad, MatchesReference) {
    const auto& param = GetParam();
    auto data = make_ramp<float>(param.input_shape.volume());
    auto source = make_float_source(param.input_shape, data);

    auto unpadded = unpad(source, param.start, param.end);

    const auto expected_shape = extent(param.start, param.end);
    EXPECT_EQ(unpadded.logical_shape(), expected_shape);
    EXPECT_EQ(unpadded.padded_shape(), expected_shape);
    EXPECT_EQ(unpadded.memory_config(), MemoryConfig{});
    EXPECT_EQ(
        first_shard_values<float>(unpadded), reference_unpad<float>(data, param.input_shape, param.start, param.end));
}

INSTANTIATE_TEST_SUITE_P(
    HostPadUnpad,
    HostPadUnpadUnpad,
    ::testing::Values(
        UnpadCase{"rank1", Shape{7}, Shape{2}, Shape{5}},
        UnpadCase{"rank2", Shape{5, 6}, Shape{1, 2}, Shape{4, 5}},
        UnpadCase{"rank3", Shape{3, 4, 5}, Shape{1, 0, 2}, Shape{3, 4, 4}},
        UnpadCase{"rank4", Shape{2, 3, 4, 5}, Shape{1, 1, 1, 1}, Shape{2, 3, 4, 4}},
        UnpadCase{"rank5", Shape{2, 2, 3, 4, 5}, Shape{0, 1, 1, 0, 2}, Shape{2, 2, 3, 3, 5}},
        UnpadCase{"issue_example_exclusive_end", Shape{1, 1, 5, 5}, Shape{0, 0, 1, 1}, Shape{1, 1, 4, 4}},
        UnpadCase{"full_range", Shape{1, 2, 3, 4}, Shape{0, 0, 0, 0}, Shape{1, 2, 3, 4}},
        UnpadCase{"zero_size", Shape{1, 1, 5, 5}, Shape{0, 0, 2, 2}, Shape{0, 0, 2, 2}}),
    [](const ::testing::TestParamInfo<UnpadCase>& info) { return info.param.name; });

TEST(HostPadUnpad, PadThenUnpadRoundTrip) {
    const Shape input_shape{1, 2, 3, 4};
    const Shape output_shape{2, 4, 6, 8};
    const Shape input_start{1, 1, 2, 3};
    auto data = make_ramp<float>(input_shape.volume());
    auto source = make_float_source(input_shape, data);

    auto padded = pad(source, output_shape, input_start, kPadValue);
    auto unpadded = unpad(padded, input_start, end_of(input_start, input_shape));

    EXPECT_EQ(unpadded.logical_shape(), input_shape);
    EXPECT_EQ(unpadded.padded_shape(), input_shape);
    EXPECT_EQ(first_shard_values<float>(unpadded), data);
}

// ---------------------------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------------------------

template <typename T>
class HostPadUnpadDtype : public ::testing::Test {};

using HostPadUnpadDtypes = ::testing::Types<float, bfloat16, int32_t, uint32_t, uint16_t, uint8_t>;
TYPED_TEST_SUITE(HostPadUnpadDtype, HostPadUnpadDtypes);

TYPED_TEST(HostPadUnpadDtype, PadMatchesReference) {
    using T = TypeParam;
    const Shape input_shape{1, 2, 3, 4};
    const Shape output_shape{2, 3, 5, 6};
    const Shape input_start{1, 0, 1, 2};
    auto data = make_ramp<T>(input_shape.volume());
    auto source = HostTensor::from_vector<T>(data, row_major_spec(input_shape, dtype_of<T>()));

    auto padded = pad(source, output_shape, input_start, kPadValue);

    EXPECT_EQ(padded.dtype(), dtype_of<T>());
    EXPECT_EQ(padded.padded_shape(), output_shape);
    EXPECT_EQ(
        first_shard_values<T>(padded),
        reference_pad<T>(data, input_shape, output_shape, input_start, static_cast<T>(kPadValue)));
}

TYPED_TEST(HostPadUnpadDtype, UnpadMatchesReference) {
    using T = TypeParam;
    const Shape input_shape{2, 3, 4, 5};
    const Shape start{1, 0, 1, 2};
    const Shape end{2, 2, 4, 5};
    auto data = make_ramp<T>(input_shape.volume());
    auto source = HostTensor::from_vector<T>(data, row_major_spec(input_shape, dtype_of<T>()));

    auto unpadded = unpad(source, start, end);

    EXPECT_EQ(unpadded.dtype(), dtype_of<T>());
    EXPECT_EQ(unpadded.logical_shape(), extent(start, end));
    EXPECT_EQ(first_shard_values<T>(unpadded), reference_unpad<T>(data, input_shape, start, end));
}

// ---------------------------------------------------------------------------------------------
// multi-device host storage (#21628)
// ---------------------------------------------------------------------------------------------

TEST(HostPadUnpad, UnpadMultiDeviceHostStoragePerShard) {
    const Shape input_shape{1, 2, 4, 6};
    const Shape start{0, 1, 1, 2};
    const Shape end{1, 2, 3, 5};
    auto source = make_multi_device_source(input_shape);

    auto unpadded = unpad(source.tensor, start, end);

    EXPECT_EQ(unpadded.logical_shape(), extent(start, end));
    EXPECT_EQ(get_tensor_topology(unpadded), get_tensor_topology(source.tensor));
    EXPECT_EQ(unpadded.buffer().shard_coords(), source.tensor.buffer().shard_coords());
    EXPECT_EQ(
        shard_values<float>(unpadded, distributed::MeshCoordinate(0, 0)),
        reference_unpad<float>(source.data_0, input_shape, start, end));
    EXPECT_EQ(
        shard_values<float>(unpadded, distributed::MeshCoordinate(0, 1)),
        reference_unpad<float>(source.data_1, input_shape, start, end));
}

TEST(HostPadUnpad, PadMultiDeviceHostStoragePerShard) {
    const Shape input_shape{1, 2, 4, 6};
    const Shape output_shape{2, 3, 6, 8};
    const Shape input_start{1, 0, 1, 2};
    auto source = make_multi_device_source(input_shape);

    auto padded = pad(source.tensor, output_shape, input_start, kPadValue);

    EXPECT_EQ(padded.padded_shape(), output_shape);
    EXPECT_EQ(get_tensor_topology(padded), get_tensor_topology(source.tensor));
    EXPECT_EQ(padded.buffer().shard_coords(), source.tensor.buffer().shard_coords());
    EXPECT_EQ(
        shard_values<float>(padded, distributed::MeshCoordinate(0, 0)),
        reference_pad<float>(source.data_0, input_shape, output_shape, input_start, kPadValue));
    EXPECT_EQ(
        shard_values<float>(padded, distributed::MeshCoordinate(0, 1)),
        reference_pad<float>(source.data_1, input_shape, output_shape, input_start, kPadValue));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace tt::tt_metal
