// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/tensor_utils.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/shape.hpp>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <algorithm>
#include <vector>

namespace tt::tt_metal {

// ======================================================================================
//                                  HostTensor Factory Functions
// ======================================================================================

template <typename T>
HostTensor HostTensor::from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (spec.data_type() == DataType::BFLOAT8_B || spec.data_type() == DataType::BFLOAT4_B) {
        TT_FATAL(spec.layout() == Layout::TILE, "Block float types are only supported in TILE layout");
    }

    auto buffer_dtype = convert_to_data_type<T>();
    auto buffer_spec =
        TensorSpec(spec.logical_shape(), TensorLayout(buffer_dtype, spec.page_config(), spec.memory_config()));

    auto host_buffer =
        logical_matches_physical(buffer_spec)
            ? HostBuffer(std::move(buffer))
            : HostBuffer(tensor_impl::encode_tensor_data(tt::stl::make_const_span(buffer), spec, pad_value));

    auto result = HostTensor(std::move(host_buffer), buffer_spec, TensorTopology{});
    return to_dtype(result, spec.data_type());
}

template HostTensor HostTensor::from_vector<bfloat16>(
    std::vector<bfloat16>&& buffer, const TensorSpec& spec, bfloat16 pad_value);
template HostTensor HostTensor::from_vector<float>(
    std::vector<float>&& buffer, const TensorSpec& spec, float pad_value);
template HostTensor HostTensor::from_vector<int32_t>(
    std::vector<int32_t>&& buffer, const TensorSpec& spec, int32_t pad_value);
template HostTensor HostTensor::from_vector<uint8_t>(
    std::vector<uint8_t>&& buffer, const TensorSpec& spec, uint8_t pad_value);
template HostTensor HostTensor::from_vector<uint16_t>(
    std::vector<uint16_t>&& buffer, const TensorSpec& spec, uint16_t pad_value);
template HostTensor HostTensor::from_vector<uint32_t>(
    std::vector<uint32_t>&& buffer, const TensorSpec& spec, uint32_t pad_value);

template <typename T>
HostTensor HostTensor::from_vector(const std::vector<T>& buffer, const TensorSpec& spec, T pad_value) {
    return from_vector(std::vector<T>(buffer), spec, pad_value);
}

template HostTensor HostTensor::from_vector<bfloat16>(
    const std::vector<bfloat16>& buffer, const TensorSpec& spec, bfloat16 pad_value);
template HostTensor HostTensor::from_vector<float>(
    const std::vector<float>& buffer, const TensorSpec& spec, float pad_value);
template HostTensor HostTensor::from_vector<int32_t>(
    const std::vector<int32_t>& buffer, const TensorSpec& spec, int32_t pad_value);
template HostTensor HostTensor::from_vector<uint8_t>(
    const std::vector<uint8_t>& buffer, const TensorSpec& spec, uint8_t pad_value);
template HostTensor HostTensor::from_vector<uint16_t>(
    const std::vector<uint16_t>& buffer, const TensorSpec& spec, uint16_t pad_value);
template HostTensor HostTensor::from_vector<uint32_t>(
    const std::vector<uint32_t>& buffer, const TensorSpec& spec, uint32_t pad_value);

template <typename T>
HostTensor HostTensor::from_span(tt::stl::Span<const T> buffer, const TensorSpec& spec, T pad_value) {
    return from_vector(std::vector<T>(buffer.begin(), buffer.end()), spec, pad_value);
}

template HostTensor HostTensor::from_span<bfloat16>(
    tt::stl::Span<const bfloat16> buffer, const TensorSpec& spec, bfloat16 pad_value);
template HostTensor HostTensor::from_span<float>(
    tt::stl::Span<const float> buffer, const TensorSpec& spec, float pad_value);
template HostTensor HostTensor::from_span<int32_t>(
    tt::stl::Span<const int32_t> buffer, const TensorSpec& spec, int32_t pad_value);
template HostTensor HostTensor::from_span<uint8_t>(
    tt::stl::Span<const uint8_t> buffer, const TensorSpec& spec, uint8_t pad_value);
template HostTensor HostTensor::from_span<uint16_t>(
    tt::stl::Span<const uint16_t> buffer, const TensorSpec& spec, uint16_t pad_value);
template HostTensor HostTensor::from_span<uint32_t>(
    tt::stl::Span<const uint32_t> buffer, const TensorSpec& spec, uint32_t pad_value);

template <typename T>
HostTensor HostTensor::from_borrowed_data(tt::stl::Span<T> buffer, const Shape& shape, MemoryPin buffer_pin) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);

    auto tensor_spec = TensorSpec(
        shape,
        TensorLayout::fromPaddedShape(
            convert_to_data_type<T>(), PageConfig(Layout::ROW_MAJOR, std::nullopt), MemoryConfig{}, shape, shape));

    return HostTensor(
        HostBuffer(tt::stl::Span<T>(buffer.data(), buffer.size()), std::move(buffer_pin)),
        tensor_spec,
        TensorTopology{});
}

template HostTensor HostTensor::from_borrowed_data<bfloat16>(
    tt::stl::Span<bfloat16> buffer, const Shape& shape, MemoryPin buffer_pin);
template HostTensor HostTensor::from_borrowed_data<float>(
    tt::stl::Span<float> buffer, const Shape& shape, MemoryPin buffer_pin);
template HostTensor HostTensor::from_borrowed_data<int32_t>(
    tt::stl::Span<int32_t> buffer, const Shape& shape, MemoryPin buffer_pin);
template HostTensor HostTensor::from_borrowed_data<uint8_t>(
    tt::stl::Span<uint8_t> buffer, const Shape& shape, MemoryPin buffer_pin);
template HostTensor HostTensor::from_borrowed_data<uint16_t>(
    tt::stl::Span<uint16_t> buffer, const Shape& shape, MemoryPin buffer_pin);
template HostTensor HostTensor::from_borrowed_data<uint32_t>(
    tt::stl::Span<uint32_t> buffer, const Shape& shape, MemoryPin buffer_pin);

// ======================================================================================
//                                  HostTensor to_vector()
// ======================================================================================

namespace {

std::vector<float> to_vector_float(const HostTensor& tensor) {
    switch (tensor.dtype()) {
        case DataType::BFLOAT16: {
            auto buffer = host_buffer::get_as<bfloat16>(tensor);
            std::vector<float> physical_data;
            physical_data.reserve(buffer.size());
            std::transform(buffer.begin(), buffer.end(), std::back_inserter(physical_data), [](bfloat16 val) {
                return static_cast<float>(val);
            });
            if (logical_matches_physical(tensor.tensor_spec())) {
                return physical_data;
            }
            return tensor_impl::decode_tensor_data(tt::stl::make_const_span(physical_data), tensor.tensor_spec());
        }
        case DataType::FLOAT32: {
            auto buffer = host_buffer::get_as<const float>(tensor);
            return tensor_impl::decode_tensor_data(buffer, tensor.tensor_spec());
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = tensor.tensor_spec().tile();
            auto buffer = host_buffer::get_as<const uint32_t>(tensor);
            std::vector<float> unpacked_data =
                tensor.tensor_spec().data_type() == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                    : unpack_bfp4_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
            return tensor_impl::decode_tensor_data(tt::stl::make_const_span(unpacked_data), tensor.tensor_spec());
        }
        default: {
            TT_THROW("Cannot convert tensor to vector for data type: {}", tensor.dtype());
        }
    }
}

}  // namespace

template <typename T>
std::vector<T> HostTensor::to_vector() const {
    if constexpr (std::is_same_v<T, float>) {
        return to_vector_float(*this);
    }
    TT_FATAL(
        dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        dtype(),
        convert_to_data_type<T>());
    auto data = host_buffer::get_as<const T>(*this);
    if (logical_matches_physical(tensor_spec())) {
        return std::vector<T>(data.begin(), data.end());
    }
    return tensor_impl::decode_tensor_data(data, tensor_spec());
}

template std::vector<float> HostTensor::to_vector<float>() const;
template std::vector<bfloat16> HostTensor::to_vector<bfloat16>() const;
template std::vector<int32_t> HostTensor::to_vector<int32_t>() const;
template std::vector<uint8_t> HostTensor::to_vector<uint8_t>() const;
template std::vector<uint16_t> HostTensor::to_vector<uint16_t>() const;
template std::vector<uint32_t> HostTensor::to_vector<uint32_t>() const;

}  // namespace tt::tt_metal
