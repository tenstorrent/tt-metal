// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include <internal/tensor/host_pad_apis.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>

#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

HostTensor pad_bfloat8_b(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto intermediate = HostTensor::from_buffer(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())));
    update_tensor_topology(intermediate, get_tensor_topology(tensor));
    auto float_tensor = pad(intermediate, output_padded_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT8_B
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    TensorSpec output_spec(
        float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT8_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            float_tensor.logical_shape(),
            float_tensor.padded_shape()));
    auto result = HostTensor::from_buffer(std::move(output_uint32_buffer), output_spec);
    update_tensor_topology(result, get_tensor_topology(tensor));
    return result;
}

HostTensor unpad_bfloat8_b(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));

    auto intermediate = HostTensor::from_buffer(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())));
    update_tensor_topology(intermediate, get_tensor_topology(tensor));
    auto float_tensor = unpad(intermediate, output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT8_B
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    auto result = HostTensor::from_buffer(
        std::move(output_uint32_buffer),
        TensorSpec(
            float_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT8_B,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                float_tensor.logical_shape(),
                float_tensor.padded_shape())));
    update_tensor_topology(result, get_tensor_topology(tensor));
    return result;
}

HostTensor pad_bfloat4_b(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto intermediate = HostTensor::from_buffer(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.logical_shape())));
    update_tensor_topology(intermediate, get_tensor_topology(tensor));
    auto float_tensor = pad(intermediate, output_padded_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT4_B
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    TensorSpec output_spec(
        float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT4_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            float_tensor.logical_shape(),
            float_tensor.padded_shape()));
    auto result = HostTensor::from_buffer(std::move(output_uint32_buffer), output_spec);
    update_tensor_topology(result, get_tensor_topology(tensor));
    return result;
}

HostTensor unpad_bfloat4_b(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto intermediate = HostTensor::from_buffer(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())));
    update_tensor_topology(intermediate, get_tensor_topology(tensor));
    auto float_tensor = unpad(intermediate, output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT4_B
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    auto result = HostTensor::from_buffer(
        std::move(output_uint32_buffer),
        TensorSpec(
            float_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT4_B,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                float_tensor.logical_shape(),
                float_tensor.padded_shape())));
    update_tensor_topology(result, get_tensor_topology(tensor));
    return result;
}

template <typename T>
HostTensor pad_impl(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    auto pad_value_ = static_cast<T>(pad_value);
    auto input_padded_shape = tensor.padded_shape();
    if (input_padded_shape.rank() < 2) {
        input_padded_shape = input_padded_shape.to_rank(2);
    }
    const auto input_strides = tensor.strides();

    auto pad = [&input_padded_shape, &output_padded_shape, &input_tensor_start, &pad_value_](
                   const HostBuffer& input_host_buffer) {
        const auto input_buffer = input_host_buffer.view_as<T>();
        const auto rank = input_padded_shape.rank();

        auto output_buffer = std::vector<T>(output_padded_shape.volume());
        std::fill(output_buffer.begin(), output_buffer.end(), pad_value_);

        if (input_padded_shape.volume() == 0) {
            return output_buffer;
        }

        if (rank == 1) {
            std::memcpy(
                output_buffer.data() + input_tensor_start[0],
                input_buffer.data(),
                static_cast<size_t>(input_padded_shape[0]) * sizeof(T));
            return output_buffer;
        }

        // Calculate strides
        auto input_strides = compute_strides(input_padded_shape);
        auto output_strides = compute_strides(output_padded_shape);

        // Process all coordinates except for the last dimension (it's copied with mempcy)
        ttsl::SmallVector<size_t> coords(rank - 1, 0);

        bool processed_all_coords = false;
        while (!processed_all_coords) {
            // Calculate offset for a given coordinate for input and output. Again, last dimension is ignored
            size_t input_idx = 0;
            size_t output_idx = 0;

            for (int i = 0; i < rank - 1; ++i) {
                input_idx += coords[i] * input_strides[i];
                output_idx += (coords[i] + static_cast<size_t>(input_tensor_start[i])) * output_strides[i];
            }

            // Add offset (left padding) for the innermost dimension
            output_idx += static_cast<size_t>(input_tensor_start[rank - 1]) * output_strides[rank - 1];

            // Copy entire input row with memcpy
            std::memcpy(
                output_buffer.data() + output_idx,
                input_buffer.data() + input_idx,
                static_cast<size_t>(input_padded_shape[rank - 1]) * sizeof(T));

            // Increment coordinates (from right to left), ignore last dimension
            processed_all_coords = true;
            for (int dim = rank - 2; dim >= 0; --dim) {
                coords[dim]++;
                // There are still coordinates to process in dim dimension
                if (coords[dim] < input_padded_shape[dim]) {
                    processed_all_coords = false;
                    break;
                }
                // This dim's coordinate overflowed, reset it and try to increment the next one
                coords[dim] = 0;
            }
        }

        return output_buffer;
    };

    auto transformed_buffer = tensor.buffer().transform(
        [&](const HostBuffer& buffer) { return HostBuffer(pad(buffer)); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    auto tile = tt::tt_metal::Tile();
    if (tensor.layout() == Layout::TILE) {
        tile = tensor.tensor_spec().tile();
    }

    auto output_spec = TensorSpec(
        tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            tensor.dtype(),
            PageConfig(tensor.layout(), tile),
            tensor.memory_config(),
            tensor.logical_shape(),
            output_padded_shape));

    const size_t expected_shard_size = output_spec.compute_packed_buffer_size_bytes();
    for (const auto& coord : transformed_buffer.shard_coords()) {
        auto shard = transformed_buffer.get_shard(coord);
        if (shard) {
            TT_FATAL(
                shard->view_bytes().size() == expected_shard_size,
                "pad shard size mismatch after conversion: actual {} != expected {}",
                shard->view_bytes().size(),
                expected_shard_size);
        }
    }

    return host_tensor_from_buffer_with_topology(
        std::move(transformed_buffer), output_spec, get_tensor_topology(tensor));
}

template <>
HostTensor pad_impl<tensor_impl::bfloat8_b>(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    return pad_bfloat8_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

template <>
HostTensor pad_impl<tensor_impl::bfloat4_b>(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    return pad_bfloat4_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

template <>
HostTensor pad_impl<float8_e4m3>(const HostTensor&, const tt::tt_metal::Shape&, const tt::tt_metal::Shape&, float) {
    // FP8_E4M3 host-side pad is not wired up; no current op needs it. The generic body
    // would actually compile (float8_e4m3 is a 1-byte trivially-copyable type with a float
    // constructor), but leaving this as an explicit throw documents the intentional scope.
    TT_THROW("pad: FP8_E4M3 is not supported");
}

template <typename T>
HostTensor unpad_impl(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    const auto& input_shape = tensor.padded_shape();
    const auto input_strides = compute_strides(input_shape);

    // Validate inputs and compute output shape
    ttsl::SmallVector<uint32_t> output_shape;
    for (auto i = 0; i < input_shape.rank(); i++) {
        // Check if tensor start and end indices are within input tensor shape
        TT_ASSERT(output_tensor_start[i] <= input_shape[i]);
        TT_ASSERT(output_tensor_end[i] <= input_shape[i]);
        // Check if start shape is < end shape
        TT_ASSERT(output_tensor_start[i] <= output_tensor_end[i]);
        // Figure out output tensor shape
        output_shape.push_back(output_tensor_end[i] - output_tensor_start[i]);
    }

    auto unpad = [&input_shape, &input_strides, &output_shape, &output_tensor_start, &output_tensor_end](
                     const HostBuffer& input_host_buffer) {
        const auto input_buffer = input_host_buffer.view_as<T>();
        ttsl::SmallVector<uint32_t> input_indices(input_shape.rank(), 0);

        auto flat_output_index = 0;
        auto output_buffer = std::vector<T>(tt::tt_metal::Shape(output_shape).volume());

        std::function<void(std::size_t)> unpad_from_tile = [&](std::size_t dim) -> void {
            for (auto i = output_tensor_start[dim]; i < output_tensor_end[dim]; i++) {
                input_indices[dim] = i;
                if (dim == input_shape.rank() - 1) {
                    auto flat_input_index = compute_flat_indices(input_indices, input_strides);
                    output_buffer[flat_output_index++] = input_buffer[flat_input_index];
                } else {
                    unpad_from_tile(dim + 1);
                }
            }
        };
        unpad_from_tile(0);

        return output_buffer;
    };

    auto transformed_buffer = tensor.buffer().transform(
        [&](const HostBuffer& buffer) { return HostBuffer(unpad(buffer)); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    // Exact authorship: padded == logical == cropped buffer. Do not re-apply source alignment.
    const auto output_tensor_shape = tt::tt_metal::Shape(output_shape);
    auto tile = tt::tt_metal::Tile();
    if (tensor.layout() == Layout::TILE) {
        tile = tensor.tensor_spec().tile();
    }
    auto output_spec = TensorSpec(
        output_tensor_shape,
        TensorLayout::fromPaddedShape(
            tensor.dtype(),
            PageConfig(tensor.layout(), tile),
            tensor.memory_config(),
            output_tensor_shape,
            output_tensor_shape));

    const size_t expected_shard_size = output_spec.compute_packed_buffer_size_bytes();
    for (const auto& coord : transformed_buffer.shard_coords()) {
        auto shard = transformed_buffer.get_shard(coord);
        if (shard) {
            TT_FATAL(
                shard->view_bytes().size() == expected_shard_size,
                "unpad shard size mismatch after conversion: actual {} != expected {}",
                shard->view_bytes().size(),
                expected_shard_size);
        }
    }

    return host_tensor_from_buffer_with_topology(
        std::move(transformed_buffer), output_spec, get_tensor_topology(tensor));
}

template <>
HostTensor unpad_impl<tensor_impl::bfloat8_b>(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    return unpad_bfloat8_b(tensor, output_tensor_start, output_tensor_end);
}

template <>
HostTensor unpad_impl<tensor_impl::bfloat4_b>(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    return unpad_bfloat4_b(tensor, output_tensor_start, output_tensor_end);
}

template <>
HostTensor unpad_impl<float8_e4m3>(const HostTensor&, const tt::tt_metal::Shape&, const tt::tt_metal::Shape&) {
    // See pad_impl<float8_e4m3>: not wired up, no current op needs it.
    TT_THROW("unpad: FP8_E4M3 is not supported");
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostTensor pad(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "Tensor layout must be ROW_MAJOR for padding");
    TT_FATAL(
        !tensor.memory_config().is_sharded(),
        "pad: sharded host tensors are not supported (legacy and ND). "
        "legacyShapeToAlignment short-circuits on shard_spec and ignores output_padded_shape for convertible ND; "
        "rederiving shard geometry under pad is out of scope.");
    return tensor_impl::dispatch(tensor.dtype(), [&]<typename T>() {
        return CMAKE_UNIQUE_NAMESPACE::pad_impl<T>(tensor, output_padded_shape, input_tensor_start, pad_value);
    });
}

HostTensor pad_to_tile(const HostTensor& tensor, float pad_value) {
    uint32_t height = tensor.padded_shape()[-2];
    uint32_t width = tensor.padded_shape()[-1];
    uint32_t padded_height = round_up(height, constants::TILE_HEIGHT);
    uint32_t padded_width = round_up(width, constants::TILE_WIDTH);

    ttsl::SmallVector<uint32_t> padded_shape;
    ttsl::SmallVector<uint32_t> input_tensor_start;

    for (auto index = 0; index < static_cast<int>(tensor.padded_shape().rank()) - 2; index++) {
        padded_shape.push_back(tensor.padded_shape()[index]);
        input_tensor_start.push_back(0);
    }

    padded_shape.push_back(padded_height);
    padded_shape.push_back(padded_width);
    input_tensor_start.push_back(0);
    input_tensor_start.push_back(0);

    return pad(
        tensor,
        tt::tt_metal::Shape(std::move(padded_shape)),
        tt::tt_metal::Shape{std::move(input_tensor_start)},
        pad_value);
}

HostTensor unpad(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "Tensor layout must be ROW_MAJOR for unpadding");
    TT_FATAL(
        !tensor.memory_config().is_sharded(),
        "unpad: sharded host tensors are not supported (legacy and ND). "
        "legacyShapeToAlignment short-circuits on shard_spec and ignores cropped geometry for convertible ND; "
        "rederiving shard geometry under unpad is out of scope.");
    return tensor_impl::dispatch(tensor.dtype(), [&]<typename T>() {
        return CMAKE_UNIQUE_NAMESPACE::unpad_impl<T>(tensor, output_tensor_start, output_tensor_end);
    });
}

HostTensor unpad_from_tile(const HostTensor& tensor, const tt::tt_metal::Shape& output_tensor_shape) {
    for (auto index = -3; index >= -static_cast<int>(tensor.padded_shape().rank()); index--) {
        TT_FATAL(
            tensor.logical_shape()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_FATAL(
        tensor.padded_shape()[-2] % constants::TILE_HEIGHT == 0 &&
            tensor.padded_shape()[-1] % constants::TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_FATAL(
        tensor.padded_shape()[-2] < output_tensor_shape[-2] + constants::TILE_HEIGHT &&
            tensor.padded_shape()[-1] < output_tensor_shape[-1] + constants::TILE_WIDTH,
        "Last 2 dims of output must be within range to have been padded to input");
    Shape output_tensor_start(ttsl::SmallVector<uint32_t>(tensor.padded_shape().rank(), 0));
    Shape output_tensor_end(ttsl::SmallVector<uint32_t>(tensor.padded_shape().rank(), 1));
    for (int index = -1; index >= -static_cast<int>(output_tensor_shape.rank()); index--) {
        output_tensor_end[index] = output_tensor_shape[index];
    }
    return unpad(tensor, output_tensor_start, output_tensor_end);
}

}  // namespace tt::tt_metal
