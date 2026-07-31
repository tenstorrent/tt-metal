// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor_apis_with_pad_values.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include "tensor_impl.hpp"

#include <tt_stl/span.hpp>
#include <tt_stl/fmt.hpp>

#include <algorithm>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
HostTensor from_span_impl(std::span<const T> buffer, const TensorSpec& spec, T pad_value) {
    auto buffer_dtype = convert_to_data_type<T>();
    auto buffer_spec = TensorSpec(
        spec.logical_shape(),
        tensor_layout_with_custom_alignment(
            buffer_dtype, spec.page_config(), spec.memory_config(), spec.tensor_layout().get_alignment()));

    size_t volume = spec.logical_shape().volume();

    TT_FATAL(
        spec.layout() != Layout::ROW_MAJOR || spec.logical_2d_shape() != spec.physical_shape(),
        "Logical matches physical, don't support that case, use Tensor::from_span instead!");

    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (spec.data_type() == DataType::BFLOAT8_B || spec.data_type() == DataType::BFLOAT4_B) {
        TT_FATAL(spec.layout() == Layout::TILE, "Block float types are only supported in TILE layout");
    }

    auto host_buffer = HostBuffer(tensor_impl::encode_tensor_data(ttsl::make_const_span(buffer), spec, pad_value));

    auto res = HostTensor::from_buffer(std::move(host_buffer), std::move(buffer_spec));
    return to_dtype(res, spec.data_type());
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
HostTensor host_tensor_from_span_with_pad_value(std::span<const T> buffer, TensorSpec spec, T pad_value) {
    if (spec.layout() != Layout::ROW_MAJOR || spec.logical_2d_shape() != spec.physical_shape()) {
        // If the logical shape doesn't match the physical shape, we need to encode the data
        // and write the result to a new buffer. This branch avoids the extra copy that
        // would otherwise occur in the from_vector function call.
        return CMAKE_UNIQUE_NAMESPACE::from_span_impl(buffer, spec, pad_value);
    }
    return host_tensor_from_vector_with_pad_value(
        std::vector<T>(buffer.begin(), buffer.end()), std::move(spec), pad_value);
}

template <typename T>
HostTensor HostTensor::from_span(std::span<const T> buffer, TensorSpec spec) {
    return host_tensor_from_span_with_pad_value(buffer, std::move(spec), T{0});
}

template <typename T>
HostTensor HostTensor::from_borrowed_data(ttsl::Span<T> buffer, const Shape& shape, MemoryPin pin) {
    size_t volume = shape.volume();
    TT_FATAL(buffer.size() == volume, "Buffer size {} differs from shape volume {}", buffer.size(), volume);

    auto host_buffer = HostBuffer(buffer, std::move(pin));
    auto buffer_dtype = convert_to_data_type<T>();
    TensorSpec tensor_spec(shape, TensorLayout(buffer_dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    return HostTensor::from_buffer(std::move(host_buffer), std::move(tensor_spec));
}

template <typename T>
HostTensor host_tensor_from_vector_with_pad_value(const std::vector<T>& buffer, TensorSpec spec, T pad_value) {
    return host_tensor_from_span_with_pad_value(ttsl::make_const_span(buffer), std::move(spec), pad_value);
}

template <typename T>
HostTensor host_tensor_from_vector_with_pad_value(std::vector<T>&& buffer, TensorSpec spec, T pad_value) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(buffer.size() == volume, "Buffer size {} differs from shape volume {}", buffer.size(), volume);

    if (spec.data_type() == DataType::BFLOAT8_B || spec.data_type() == DataType::BFLOAT4_B) {
        TT_FATAL(spec.layout() == Layout::TILE, "Block float types only supported in TILE layout");
    }

    auto buffer_dtype = convert_to_data_type<T>();
    auto buffer_spec = TensorSpec(
        spec.logical_shape(),
        tensor_layout_with_custom_alignment(
            buffer_dtype, spec.page_config(), spec.memory_config(), spec.tensor_layout().get_alignment()));

    auto host_buffer =
        buffer_spec.layout() == Layout::ROW_MAJOR && buffer_spec.logical_2d_shape() == buffer_spec.physical_shape()
            ? HostBuffer(std::move(buffer))
            : HostBuffer(tensor_impl::encode_tensor_data(ttsl::make_const_span(buffer), spec, pad_value));

    auto res = HostTensor::from_buffer(std::move(host_buffer), std::move(buffer_spec));
    return to_dtype(res, spec.data_type());
}

template <typename T>
HostTensor HostTensor::from_vector(const std::vector<T>& buffer, TensorSpec spec) {
    return host_tensor_from_vector_with_pad_value(buffer, std::move(spec), T{0});
}

template <typename T>
HostTensor HostTensor::from_vector(std::vector<T>&& buffer, TensorSpec spec) {
    return host_tensor_from_vector_with_pad_value(std::move(buffer), std::move(spec), T{0});
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
std::vector<T> to_vector_generic(const HostTensor& tensor) {
    TT_FATAL(
        tensor.dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        tensor.dtype(),
        convert_to_data_type<T>());

    const auto& tensor_spec = tensor.tensor_spec();
    auto data = host_buffer::get_as<const T>(tensor);
    if (tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape()) {
        return std::vector<T>(data.begin(), data.end());
    }
    return tensor_impl::decode_tensor_data(data, tensor_spec);
}

std::vector<float> to_vector_float(const HostTensor& tensor) {
    switch (tensor.dtype()) {
        case DataType::BFLOAT16: {
            auto buffer = host_buffer::get_as<bfloat16>(tensor);
            std::vector<float> physical_data;
            physical_data.reserve(buffer.size());
            std::transform(buffer.begin(), buffer.end(), std::back_inserter(physical_data), [](bfloat16 val) {
                return static_cast<float>(val);
            });
            const auto& tensor_spec = tensor.tensor_spec();
            if (tensor_spec.layout() == Layout::ROW_MAJOR &&
                tensor_spec.logical_2d_shape() == tensor_spec.physical_shape()) {
                return physical_data;
            }
            return tensor_impl::decode_tensor_data(ttsl::make_const_span(physical_data), tensor_spec);
        }
        case DataType::FLOAT32: {
            auto buffer = host_buffer::get_as<float>(tensor);
            return tensor_impl::decode_tensor_data(buffer, tensor.tensor_spec());
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = tensor.tensor_spec().tile();
            auto buffer = host_buffer::get_as<uint32_t>(tensor);
            std::vector<float> unpacked_data =
                tensor.dtype() == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                    : unpack_bfp4_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
            return tensor_impl::decode_tensor_data(ttsl::make_const_span(unpacked_data), tensor.tensor_spec());
        }
        default: {
            TT_THROW("Cannot convert HostTensor to vector<float> for data type: {}", tensor.dtype());
        }
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
std::vector<T> HostTensor::to_vector() const {
    if constexpr (std::is_same_v<T, float>) {
        return CMAKE_UNIQUE_NAMESPACE::to_vector_float(*this);
    }
    return CMAKE_UNIQUE_NAMESPACE::to_vector_generic<T>(*this);
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template HostTensor host_tensor_from_span_with_pad_value<bfloat16>(ttsl::Span<const bfloat16>, TensorSpec, bfloat16);
template HostTensor host_tensor_from_span_with_pad_value<float>(ttsl::Span<const float>, TensorSpec, float);
template HostTensor host_tensor_from_span_with_pad_value<int32_t>(ttsl::Span<const int32_t>, TensorSpec, int32_t);
template HostTensor host_tensor_from_span_with_pad_value<uint32_t>(ttsl::Span<const uint32_t>, TensorSpec, uint32_t);
template HostTensor host_tensor_from_span_with_pad_value<uint16_t>(ttsl::Span<const uint16_t>, TensorSpec, uint16_t);
template HostTensor host_tensor_from_span_with_pad_value<uint8_t>(ttsl::Span<const uint8_t>, TensorSpec, uint8_t);

template HostTensor HostTensor::from_span<bfloat16>(ttsl::Span<const bfloat16>, TensorSpec);
template HostTensor HostTensor::from_span<float>(ttsl::Span<const float>, TensorSpec);
template HostTensor HostTensor::from_span<int32_t>(ttsl::Span<const int32_t>, TensorSpec);
template HostTensor HostTensor::from_span<uint32_t>(ttsl::Span<const uint32_t>, TensorSpec);
template HostTensor HostTensor::from_span<uint16_t>(ttsl::Span<const uint16_t>, TensorSpec);
template HostTensor HostTensor::from_span<uint8_t>(ttsl::Span<const uint8_t>, TensorSpec);

template HostTensor HostTensor::from_borrowed_data<bfloat16>(ttsl::Span<bfloat16>, const Shape&, MemoryPin);
template HostTensor HostTensor::from_borrowed_data<float>(ttsl::Span<float>, const Shape&, MemoryPin);
template HostTensor HostTensor::from_borrowed_data<int32_t>(ttsl::Span<int32_t>, const Shape&, MemoryPin);
template HostTensor HostTensor::from_borrowed_data<uint32_t>(ttsl::Span<uint32_t>, const Shape&, MemoryPin);
template HostTensor HostTensor::from_borrowed_data<uint16_t>(ttsl::Span<uint16_t>, const Shape&, MemoryPin);
template HostTensor HostTensor::from_borrowed_data<uint8_t>(ttsl::Span<uint8_t>, const Shape&, MemoryPin);

template HostTensor host_tensor_from_vector_with_pad_value<bfloat16>(
    const std::vector<bfloat16>&, TensorSpec, bfloat16);
template HostTensor host_tensor_from_vector_with_pad_value<float>(const std::vector<float>&, TensorSpec, float);
template HostTensor host_tensor_from_vector_with_pad_value<int32_t>(const std::vector<int32_t>&, TensorSpec, int32_t);
template HostTensor host_tensor_from_vector_with_pad_value<uint32_t>(
    const std::vector<uint32_t>&, TensorSpec, uint32_t);
template HostTensor host_tensor_from_vector_with_pad_value<uint16_t>(
    const std::vector<uint16_t>&, TensorSpec, uint16_t);
template HostTensor host_tensor_from_vector_with_pad_value<uint8_t>(const std::vector<uint8_t>&, TensorSpec, uint8_t);

template HostTensor host_tensor_from_vector_with_pad_value<bfloat16>(std::vector<bfloat16>&&, TensorSpec, bfloat16);
template HostTensor host_tensor_from_vector_with_pad_value<float>(std::vector<float>&&, TensorSpec, float);
template HostTensor host_tensor_from_vector_with_pad_value<int32_t>(std::vector<int32_t>&&, TensorSpec, int32_t);
template HostTensor host_tensor_from_vector_with_pad_value<uint32_t>(std::vector<uint32_t>&&, TensorSpec, uint32_t);
template HostTensor host_tensor_from_vector_with_pad_value<uint16_t>(std::vector<uint16_t>&&, TensorSpec, uint16_t);
template HostTensor host_tensor_from_vector_with_pad_value<uint8_t>(std::vector<uint8_t>&&, TensorSpec, uint8_t);

template HostTensor HostTensor::from_vector<bfloat16>(const std::vector<bfloat16>&, TensorSpec);
template HostTensor HostTensor::from_vector<float>(const std::vector<float>&, TensorSpec);
template HostTensor HostTensor::from_vector<int32_t>(const std::vector<int32_t>&, TensorSpec);
template HostTensor HostTensor::from_vector<uint32_t>(const std::vector<uint32_t>&, TensorSpec);
template HostTensor HostTensor::from_vector<uint16_t>(const std::vector<uint16_t>&, TensorSpec);
template HostTensor HostTensor::from_vector<uint8_t>(const std::vector<uint8_t>&, TensorSpec);

template HostTensor HostTensor::from_vector<bfloat16>(std::vector<bfloat16>&&, TensorSpec);
template HostTensor HostTensor::from_vector<float>(std::vector<float>&&, TensorSpec);
template HostTensor HostTensor::from_vector<int32_t>(std::vector<int32_t>&&, TensorSpec);
template HostTensor HostTensor::from_vector<uint32_t>(std::vector<uint32_t>&&, TensorSpec);
template HostTensor HostTensor::from_vector<uint16_t>(std::vector<uint16_t>&&, TensorSpec);
template HostTensor HostTensor::from_vector<uint8_t>(std::vector<uint8_t>&&, TensorSpec);

template std::vector<float> HostTensor::to_vector() const;
template std::vector<bfloat16> HostTensor::to_vector() const;
template std::vector<int32_t> HostTensor::to_vector() const;
template std::vector<uint32_t> HostTensor::to_vector() const;
template std::vector<uint16_t> HostTensor::to_vector() const;
template std::vector<uint8_t> HostTensor::to_vector() const;

}  // namespace tt::tt_metal
