// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_set>

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/concepts.hpp>
#include <tt_stl/small_vector.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                            Transfer classification
// ======================================================================================

bool is_uniform_write(const HostTensor& host_tensor, const distributed::MeshDevice& device) {
    const auto& device_mesh_shape = device.shape();
    const auto& host_buffer = host_tensor.buffer();

    if (host_buffer.shape() != device_mesh_shape) {
        return false;
    }

    auto all_coords = distributed::MeshCoordinateRange(device_mesh_shape);
    return std::ranges::all_of(
        all_coords, [&](const auto& coord) { return host_buffer.shard_coords().contains(coord); });
}

// ======================================================================================
//                                Uniform Data movement APIs
// ======================================================================================

HostTensor enqueue_read_tensor(distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, bool blocking) {
    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();
    auto& device = device_tensor.device();

    auto distributed_host_buffer = DistributedHostBuffer::create(device.get_view());

    distributed::MeshCoordinateRange all_coords(device.shape());
    std::vector<distributed::MeshCoordinate> coords(all_coords.begin(), all_coords.end());
    distributed_host_buffer.emplace_shards(
        coords,
        [&](const distributed::MeshCoordinate&) {
            return tensor_impl::allocate_host_buffer(device_tensor.tensor_spec());
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    cq.enqueue_read(mesh_buffer, distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    return HostTensor(std::move(distributed_host_buffer), device_tensor.tensor_spec(), device_tensor.tensor_topology());
}

MeshTensor enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config) {
    TT_FATAL(
        is_uniform_write(host_tensor, mesh_device),
        "Incompatible shape between source host tensor and target MeshDevice. For non-uniform transfers, use the "
        "non-uniform data movement APIs.");
    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        tensor_spec_overriden_memory_config = host_tensor.tensor_spec().with_memory_config(*memory_config);
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &host_tensor.tensor_spec();

    auto result = MeshTensor::allocate_on_device(mesh_device, *tensor_spec, host_tensor.tensor_topology());
    enqueue_write_tensor(cq, host_tensor, result);
    return result;
}

void enqueue_read_tensor(
    distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, HostTensor& host_tensor, bool blocking) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();

    cq.enqueue_read(mesh_buffer, host_tensor.buffer(), /*shards=*/std::nullopt, blocking);
    host_tensor.update_tensor_topology(device_tensor.tensor_topology());
}

void enqueue_write_tensor(distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor) {
    TT_FATAL(
        is_uniform_write(host_tensor, device_tensor.device()),
        "Incompatible shape between source host tensor and target MeshDevice. For non-uniform transfers, use the "
        "non-uniform data movement APIs.");
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();

    // Uniform H2D copy.
    cq.enqueue_write(mesh_buffer, host_tensor.buffer(), /*blocking=*/false);
    device_tensor = MeshTensor(
        mesh_buffer,
        host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()),
        host_tensor.tensor_topology());
}

// ======================================================================================
//                    Unit Tensor enqueue_read/write_tensor
// ======================================================================================

void enqueue_read_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_read_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(dst)
            .region(region)};
    queue.enqueue_read_shards(shard_data_transfers, device_tensor.mesh_buffer_invariant_breaking(), blocking);
}

void enqueue_write_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_write_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(const_cast<std::byte*>(src))
            .region(region)};
    queue.enqueue_write_shards(device_tensor.mesh_buffer_invariant_breaking(), shard_data_transfers, false);
}

// ======================================================================================
//              Non-uniform enqueue_read/write_tensor
// ======================================================================================

namespace non_uniform_data_movement {

HostTensor enqueue_read_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking) {
    auto distributed_host_buffer = DistributedHostBuffer::create(device_tensor.device().get_view());
    distributed_host_buffer.emplace_shards(
        {coords.begin(), coords.end()},
        [&](const distributed::MeshCoordinate&) {
            return tensor_impl::allocate_host_buffer(device_tensor.tensor_spec());
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    HostTensor result(std::move(distributed_host_buffer), device_tensor.tensor_spec(), device_tensor.tensor_topology());
    enqueue_read_tensor(cq, device_tensor, result, coords, blocking);
    return result;
}

void enqueue_read_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    HostTensor& host_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& distributed_host_buffer = host_tensor.buffer();

    std::vector<std::pair<distributed::MeshCoordinate, std::optional<HostBuffer>>> shards;
    shards.reserve(coords.size());
    for (const auto& device_coord : coords) {
        shards.push_back({device_coord, distributed_host_buffer.get_shard(device_coord)});
    }

    DistributedHostBuffer dst_distributed_host_buffer =
        DistributedHostBuffer::create(device_tensor.device().get_view());
    const size_t expected_size_bytes = device_tensor.tensor_spec().compute_packed_buffer_size_bytes();
    for (const auto& [device_coord, host_buffer] : shards) {
        dst_distributed_host_buffer.emplace_shard(device_coord, [&]() {
            TT_FATAL(host_buffer.has_value(), "Host shard for device shard {} is not populated.", device_coord);
            TT_FATAL(
                host_buffer->view_bytes().size() == expected_size_bytes,
                "Host shard for device shard {} has invalid size: {} != {}",
                device_coord,
                host_buffer->view_bytes().size(),
                expected_size_bytes);
            return *host_buffer;
        });
    }

    std::unordered_set<distributed::MeshCoordinate> shard_set(coords.begin(), coords.end());
    cq.enqueue_read(device_tensor.mesh_buffer_invariant_breaking(), dst_distributed_host_buffer, shard_set, blocking);

    host_tensor = HostTensor(
        std::move(dst_distributed_host_buffer), device_tensor.tensor_spec(), device_tensor.tensor_topology());
}

std::pair<MeshTensor, std::vector<distributed::MeshCoordinate>> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config) {
    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        tensor_spec_overriden_memory_config = host_tensor.tensor_spec().with_memory_config(*memory_config);
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &host_tensor.tensor_spec();

    auto result = MeshTensor::allocate_on_device(mesh_device, *tensor_spec, host_tensor.tensor_topology());
    auto coords = non_uniform_data_movement::enqueue_write_tensor(cq, host_tensor, result);
    return {std::move(result), std::move(coords)};
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void h2d_as_replicate_tensor_on_1x1_mesh(
    const HostTensor& host_tensor, MeshTensor& device_tensor, distributed::MeshCommandQueue& command_queue) {
    const auto host_buffer = host_tensor.buffer().get_shard(distributed::MeshCoordinate(0, 0));
    auto data_to_write = host_buffer->view_bytes();
    const auto expected_packed_buffer_size_bytes = device_tensor.tensor_spec().compute_packed_buffer_size_bytes();
    const auto input_size_bytes = data_to_write.size();
    TT_FATAL(
        input_size_bytes == expected_packed_buffer_size_bytes,
        "Host data with total size {}B does not match expected size {}B of device buffer!",
        input_size_bytes,
        expected_packed_buffer_size_bytes);

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();
    command_queue.enqueue_write_mesh_buffer(mesh_buffer, data_to_write.data(), /*blocking=*/false);

    const auto& mesh_device_shape = mesh_buffer->device()->shape();
    auto topology = TensorTopology::create_fully_replicated_tensor_topology(mesh_device_shape);
    device_tensor =
        MeshTensor(mesh_buffer, host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()), topology);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<distributed::MeshCoordinate> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& host_storage_shape = host_tensor.buffer().shape();
    const auto& dst_device_shape = device_tensor.device().shape();

    // Special case of replicating tensors on 1x1 mesh across the entire mesh device.
    if (host_storage_shape.mesh_size() < dst_device_shape.mesh_size() &&
        host_storage_shape == distributed::MeshShape(1, 1)) {
        CMAKE_UNIQUE_NAMESPACE::h2d_as_replicate_tensor_on_1x1_mesh(host_tensor, device_tensor, cq);

        // All coordinates of the MeshDevice
        distributed::MeshCoordinateRange range(device_tensor.device().shape());
        return {range.begin(), range.end()};
    }

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();
    cq.enqueue_write(mesh_buffer, host_tensor.buffer(), /*blocking=*/false);

    // DistributedHostBuffer may not cover the entire MeshDevice, must preserve coords here.
    // Coordinates here represents the shards that are local to this instance, there maybe other shards that are on
    // another host.
    std::vector<distributed::MeshCoordinate> coords;
    const auto& shard_coords = host_tensor.buffer().shard_coords();
    coords.reserve(shard_coords.size());
    std::copy(shard_coords.begin(), shard_coords.end(), std::back_inserter(coords));

    device_tensor = MeshTensor(
        mesh_buffer,
        host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()),
        host_tensor.tensor_topology());

    return coords;
}

}  // namespace non_uniform_data_movement

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
HostTensor to_layout_impl(const HostTensor& tensor, Layout target_layout) {
    if (tensor.layout() == target_layout) {
        return tensor;
    }

    auto source_layout = tensor.layout();
    auto tile = tensor.tensor_spec().tile();
    auto physical_shape = tensor.tensor_spec().physical_shape();
    auto convert =
        [tile, &physical_shape, source_layout, target_layout](const HostBuffer& input_host_buffer) -> std::vector<T> {
        const auto input_data = input_host_buffer.view_as<T>();
        switch (source_layout) {
            case Layout::ROW_MAJOR:
                TT_FATAL(target_layout == Layout::TILE, "Unsupported layout conversion");
                return tensor_impl::to_tile_major_layout(physical_shape, tile, input_data);
            case Layout::TILE:
                TT_FATAL(target_layout == Layout::ROW_MAJOR, "Unsupported layout conversion");
                return tensor_impl::to_row_major_layout(physical_shape, tile, input_data);
            case Layout::INVALID: TT_THROW("Invalid layout");
        }
        TT_THROW("Unreachable");
    };

    auto transformed_buffer = tensor.buffer().transform(
        [&](const HostBuffer& buffer) { return HostBuffer(convert(buffer)); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
    return HostTensor(
        std::move(transformed_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                tensor.dtype(),
                PageConfig(target_layout, tensor.tensor_spec().tile()),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        tensor.tensor_topology());
}

template <typename T>
HostTensor to_layout_bfloat_impl(const HostTensor& tensor, Layout target_layout) {
    static_assert(
        std::is_same_v<T, tensor_impl::bfloat8_b> || std::is_same_v<T, tensor_impl::bfloat4_b>, "Invalid type T");
    // TODO: Flip to assert when we remove use cases in python and c++
    if (tensor.layout() != target_layout or tensor.layout() != Layout::TILE) {
        log_warning(
            tt::LogAlways,
            "Tensor layout must be Layout::TILE for bfloat8_b or bfloat4_b! Conversion from {} to {} was not executed!",
            tensor.layout(),
            target_layout);
    }
    return tensor;
}

template <>
HostTensor to_layout_impl<tensor_impl::bfloat8_b>(const HostTensor& tensor, Layout target_layout) {
    return to_layout_bfloat_impl<tensor_impl::bfloat8_b>(tensor, target_layout);
}

template <>
HostTensor to_layout_impl<tensor_impl::bfloat4_b>(const HostTensor& tensor, Layout target_layout) {
    return to_layout_bfloat_impl<tensor_impl::bfloat4_b>(tensor, target_layout);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostTensor to_layout(const HostTensor& tensor, Layout target_layout) {
    return tensor_impl::dispatch(
        tensor.dtype(), [&]<typename T>() { return CMAKE_UNIQUE_NAMESPACE::to_layout_impl<T>(tensor, target_layout); });
}

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
    auto intermediate = HostTensor(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        tensor.tensor_topology());
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
    return HostTensor(std::move(output_uint32_buffer), output_spec, tensor.tensor_topology());
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

    HostTensor intermediate(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        tensor.tensor_topology());
    auto float_tensor = unpad(intermediate, output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT8_B
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    return HostTensor(
        std::move(output_uint32_buffer),
        TensorSpec(
            float_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT8_B,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                float_tensor.logical_shape(),
                float_tensor.padded_shape())),
        tensor.tensor_topology());
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
    auto intermediate = HostTensor(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.logical_shape())),
        tensor.tensor_topology());
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
    return HostTensor(std::move(output_uint32_buffer), output_spec, tensor.tensor_topology());
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
    auto intermediate = HostTensor(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        tensor.tensor_topology());
    auto float_tensor = unpad(intermediate, output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT4_B
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    return HostTensor(
        std::move(output_uint32_buffer),
        TensorSpec(
            float_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT4_B,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                float_tensor.logical_shape(),
                float_tensor.padded_shape())),
        tensor.tensor_topology());
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
    return HostTensor(
        std::move(transformed_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                tensor.dtype(),
                PageConfig(tensor.layout(), tensor.tensor_spec().tile()),
                MemoryConfig{},
                tensor.logical_shape(),
                output_padded_shape)),
        tensor.tensor_topology());
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
    return HostTensor(
        std::move(transformed_buffer),
        TensorSpec(
            tt::tt_metal::Shape(output_shape),
            tt::tt_metal::TensorLayout(
                tensor.dtype(),
                tt::tt_metal::PageConfig(tensor.layout(), tensor.tensor_spec().tile()),
                tt::tt_metal::MemoryConfig{})),
        tensor.tensor_topology());
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

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostTensor pad(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    // TODO(#40993): Flip to assert when we remove use cases in python and c++
    if (tensor.layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp, "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!", tensor.layout());
        return tensor;
    }
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
    // TODO(#40993): This should be a FATAL
    TT_ASSERT(tensor.layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    return tensor_impl::dispatch(tensor.dtype(), [&]<typename T>() {
        return CMAKE_UNIQUE_NAMESPACE::unpad_impl<T>(tensor, output_tensor_start, output_tensor_end);
    });
}

HostTensor unpad_from_tile(const HostTensor& tensor, const tt::tt_metal::Shape& output_tensor_shape) {
    // TODO(#40993): These asserts should be FATAL
    for (auto index = -3; index >= -static_cast<int>(tensor.padded_shape().rank()); index--) {
        TT_ASSERT(
            tensor.logical_shape()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_ASSERT(
        tensor.padded_shape()[-2] % constants::TILE_HEIGHT == 0 &&
            tensor.padded_shape()[-1] % constants::TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(
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

// ======================================================================================
//                                  .to_dtype()
// ======================================================================================

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct bfloat4_tag {};
struct bfloat8_tag {};

// Preprocess the storage to unpack the bfloat8/4 tiles into float32.
tt::tt_metal::DistributedHostBuffer preprocess_buffers(
    const tt::tt_metal::DistributedHostBuffer& input_storage, const DataType input_dtype) {
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if (input_dtype == DataType::BFLOAT8_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    if (input_dtype == DataType::BFLOAT4_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    return input_storage;
}

template <typename SrcType, typename DstType>
tt::tt_metal::DistributedHostBuffer transform_buffers(
    const tt::tt_metal::TensorSpec& input_tensor_spec, const tt::tt_metal::DistributedHostBuffer& input_buffer) {
    if constexpr (std::is_same_v<SrcType, DstType>) {
        return input_buffer;
    } else if constexpr (std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag>) {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const SrcType> data = buffer.view_as<const SrcType>();
            std::vector<SrcType> tilized_data;  // empty if `data` is already in tile layout.
            if (input_tensor_spec.layout() == Layout::ROW_MAJOR) {
                tilized_data = tensor_impl::to_tile_major_layout(
                    input_tensor_spec.physical_shape(), input_tensor_spec.tile(), data);
                data = ttsl::make_const_span(tilized_data);
            }

            auto float_packed_data = [&]() {
                constexpr bool row_major_input = false;
                constexpr bool is_exp_a = false;
                if constexpr (std::is_same_v<DstType, bfloat8_tag>) {
                    return pack_as_bfp8_tiles(data, row_major_input, is_exp_a, input_tensor_spec.tile());
                } else if constexpr (std::is_same_v<DstType, bfloat4_tag>) {
                    return pack_as_bfp4_tiles(data, row_major_input, is_exp_a, input_tensor_spec.tile());
                } else {
                    static_assert(ttsl::concepts::always_false_v<DstType>, "Unsupported data type");
                }
            }();
            return tt::tt_metal::HostBuffer(std::move(float_packed_data));
        };

        return input_buffer.transform(transform_fn);
    } else {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            auto data = buffer.view_as<const SrcType>();
            std::vector<DstType> output_vector(data.size());
            std::transform(data.begin(), data.end(), output_vector.begin(), [](SrcType value) {
                return static_cast<DstType>(value);
            });
            return tt::tt_metal::HostBuffer(std::move(output_vector));
        };

        return input_buffer.transform(transform_fn);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype) {
    const auto src_type = input_tensor.dtype();
    if (src_type == dtype) {
        return input_tensor;
    }

    auto input_buffer = CMAKE_UNIQUE_NAMESPACE::preprocess_buffers(input_tensor.buffer(), src_type);

    auto output_storage = [src_type, dst_type = dtype, &input_tensor, &input_buffer]() {
        auto with_src_and_dst = [&]<typename SrcType, typename DstType>() {
            return CMAKE_UNIQUE_NAMESPACE::transform_buffers<SrcType, DstType>(
                input_tensor.tensor_spec(), input_buffer);
        };

        auto with_src = [dst_type, &with_src_and_dst]<typename SrcType>() {
            switch (dst_type) {
                case DataType::BFLOAT4_B:
                    return with_src_and_dst.operator()<SrcType, CMAKE_UNIQUE_NAMESPACE::bfloat4_tag>();
                case DataType::BFLOAT8_B:
                    return with_src_and_dst.operator()<SrcType, CMAKE_UNIQUE_NAMESPACE::bfloat8_tag>();
                case DataType::FLOAT32: return with_src_and_dst.operator()<SrcType, float>();
                case DataType::BFLOAT16: return with_src_and_dst.operator()<SrcType, bfloat16>();
                case DataType::UINT8: return with_src_and_dst.operator()<SrcType, uint8_t>();
                case DataType::UINT16: return with_src_and_dst.operator()<SrcType, uint16_t>();
                case DataType::UINT32: return with_src_and_dst.operator()<SrcType, uint32_t>();
                case DataType::INT32: return with_src_and_dst.operator()<SrcType, int32_t>();
                case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
            }
            TT_THROW("Unreachable");
        };

        switch (src_type) {
            case DataType::BFLOAT4_B:
            case DataType::BFLOAT8_B:
            case DataType::FLOAT32: return with_src.operator()<float>();
            case DataType::BFLOAT16: return with_src.operator()<bfloat16>();
            case DataType::UINT8: return with_src.operator()<uint8_t>();
            case DataType::UINT16: return with_src.operator()<uint16_t>();
            case DataType::UINT32: return with_src.operator()<uint32_t>();
            case DataType::INT32: return with_src.operator()<int32_t>();
            case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
        }
        TT_THROW("Unreachable");
    }();

    const auto layout =
        (dtype == DataType::BFLOAT4_B || dtype == DataType::BFLOAT8_B) ? Layout::TILE : input_tensor.layout();

    auto output_spec = TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            dtype,
            tt::tt_metal::PageConfig(layout, input_tensor.tensor_spec().tile()),
            input_tensor.tensor_spec().memory_config(),
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));

    return HostTensor(std::move(output_storage), output_spec, input_tensor.tensor_topology());
}

// ======================================================================================
//                                  Utility functions
// ======================================================================================

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

namespace host_buffer {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
void validate_datatype(DataType dtype) {
    using BaseType = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<BaseType, uint32_t>) {
        TT_FATAL(
            dtype == DataType::UINT32 or dtype == DataType::BFLOAT8_B or dtype == DataType::BFLOAT4_B,
            "Incorrect data type {}",
            dtype);
    } else if constexpr (std::is_same_v<BaseType, int32_t>) {
        TT_FATAL(dtype == DataType::INT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, float>) {
        TT_FATAL(dtype == DataType::FLOAT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, bfloat16>) {
        TT_FATAL(dtype == DataType::BFLOAT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint16_t>) {
        TT_FATAL(dtype == DataType::UINT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint8_t>) {
        TT_FATAL(dtype == DataType::UINT8, "Incorrect data type {}", dtype);
    } else {
        static_assert(sizeof(BaseType) == 0, "Unsupported DataType");
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostBuffer get_host_buffer(const HostTensor& tensor) {
    std::vector<HostBuffer> buffers;
    tensor.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
    TT_FATAL(
        buffers.size() == 1,
        "Can't get a single buffer from host storage distributed over mesh shape {}",
        tensor.buffer().shape());
    return buffers.front();
}

template <typename T>
ttsl::Span<const T> get_as(const HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<const T> get_as(const HostTensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(HostTensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

// Explicit template instantiations
#define INSTANTIATE_HOST_BUFFER_FUNCTIONS(T)                         \
    template ttsl::Span<const T> get_as<T>(const HostBuffer&);       \
    template ttsl::Span<const T> get_as<const T>(const HostBuffer&); \
    template ttsl::Span<T> get_as<T>(HostBuffer&);                   \
    template ttsl::Span<const T> get_as<const T>(HostBuffer&);       \
    template ttsl::Span<const T> get_as<T>(const HostTensor&);       \
    template ttsl::Span<const T> get_as<const T>(const HostTensor&); \
    template ttsl::Span<T> get_as<T>(HostTensor&);                   \
    template ttsl::Span<const T> get_as<const T>(HostTensor&);

INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(int32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(float)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(bfloat16)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint16_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint8_t)

#undef INSTANTIATE_HOST_BUFFER_FUNCTIONS

}  // namespace host_buffer

}  // namespace tt::tt_metal
