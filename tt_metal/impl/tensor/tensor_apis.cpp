// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape.hpp>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/concepts.hpp>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::tensor_impl {

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec) {
    const auto& memory_config = tensor_spec.tensor_layout().get_memory_config();

    distributed::DeviceLocalBufferConfig device_local_buffer_config{
        .page_size = tensor_spec.compute_page_size_bytes(),
        .buffer_type = memory_config.buffer_type(),
        .sharding_args = tensor_spec.compute_buffer_sharding_args(),
    };

    // Use replicated buffer, which supports both working with individual shards and replicating data across all shards.
    // This is required for the time being, as TTNN has rich multi-device sharding implementation.
    const distributed::ReplicatedBufferConfig replicated_buffer_config{
        .size = tensor_spec.compute_packed_buffer_size_bytes(),
    };

    return distributed::MeshBuffer::create(replicated_buffer_config, device_local_buffer_config, mesh_device);
}

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec) {
    const size_t size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    switch (tensor_spec.data_type()) {
        case DataType::BFLOAT16: return HostBuffer(std::vector<bfloat16>(size_bytes / sizeof(bfloat16)));
        case DataType::FLOAT32: return HostBuffer(std::vector<float>(size_bytes / sizeof(float)));
        case DataType::INT32: return HostBuffer(std::vector<int32_t>(size_bytes / sizeof(int32_t)));
        case DataType::UINT8: return HostBuffer(std::vector<uint8_t>(size_bytes / sizeof(uint8_t)));
        case DataType::UINT16: return HostBuffer(std::vector<uint16_t>(size_bytes / sizeof(uint16_t)));
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32: return HostBuffer(std::vector<uint32_t>(size_bytes / sizeof(uint32_t)));
        case DataType::INVALID: TT_THROW("Invalid data type");
    }
    TT_THROW("Unreachable");
}

// ======================================================================================
//                                         .to_host()
// ======================================================================================

HostTensor to_host(distributed::MeshCommandQueue& queue, const MeshTensor& tensor, bool blocking) {
    TT_FATAL(tensor.is_allocated(), "Buffer must be allocated on device!");
    const auto& storage = tensor.get_legacy_device_storage();
    const auto& mesh_buffer = storage.mesh_buffer;
    distributed::MeshDevice* device = mesh_buffer->device();

    // For performance, perform all allocations via DistributedHostBuffer::transform, run from multiple threads.
    auto distributed_host_buffer = DistributedHostBuffer::create(device->get_view());

    distributed_host_buffer.emplace_shards(
        storage.coords,
        [&](const distributed::MeshCoordinate&) { return allocate_host_buffer(tensor.tensor_spec()); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    queue.enqueue_read(mesh_buffer, distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    HostStorage host_storage(std::move(distributed_host_buffer));
    return HostTensor(std::move(host_storage), tensor.tensor_spec(), tensor.tensor_topology());
}

// ======================================================================================
//                               .to_device() details
// ======================================================================================

namespace {

DeviceStorage replicate_to_mesh_buffer(
    distributed::MeshCommandQueue& queue,
    const HostBuffer& buffer,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec) {
    auto* mesh_device = mesh_buffer->device();
    auto data_to_write = buffer.view_bytes();
    const auto expected_packed_buffer_size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    const auto input_size_bytes = data_to_write.size();
    TT_FATAL(
        input_size_bytes == expected_packed_buffer_size_bytes,
        "Host data with total size {}B does not match expected size {}B of device buffer!",
        input_size_bytes,
        expected_packed_buffer_size_bytes);

    queue.enqueue_write_mesh_buffer(mesh_buffer, data_to_write.data(), /*blocking=*/false);

    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(mesh_device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        coords.push_back(coord);
    }
    return DeviceStorage(mesh_buffer, std::move(coords));
}

DeviceStorage write_to_mesh_buffer(
    distributed::MeshCommandQueue& queue,
    const DistributedHostBuffer& distributed_host_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer) {
    queue.enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(distributed_host_buffer.shard_coords().size());
    std::copy(
        distributed_host_buffer.shard_coords().begin(),
        distributed_host_buffer.shard_coords().end(),
        std::back_inserter(coords));
    return DeviceStorage(mesh_buffer, std::move(coords));
}

}  // namespace

MeshTensor to_device_mesh_buffer(
    distributed::MeshCommandQueue& queue,
    const HostStorage& host_storage,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec,
    const TensorTopology& tensor_topology) {
    const auto& host_storage_shape = host_storage.buffer().shape();
    const auto& mesh_device_shape = mesh_buffer->device()->shape();
    if (host_storage_shape.mesh_size() < mesh_device_shape.mesh_size() &&
        host_storage_shape == distributed::MeshShape(1, 1)) {
        // Special case of replicating tensors on 1x1 mesh across the entire mesh device.
        const auto device_buffer = host_storage.buffer().get_shard(distributed::MeshCoordinate(0, 0));
        return MeshTensor(
            replicate_to_mesh_buffer(queue, *device_buffer, mesh_buffer, tensor_spec),
            tensor_spec,
            TensorTopology::create_fully_replicated_tensor_topology(mesh_device_shape));
    }
    TT_FATAL(
        host_storage_shape == mesh_device_shape,
        "Distributed host buffer has different shape {} than the mesh device {}",
        host_storage_shape,
        mesh_device_shape);
    return MeshTensor(write_to_mesh_buffer(queue, host_storage.buffer(), mesh_buffer), tensor_spec, tensor_topology);
}

MeshTensor to_device(
    distributed::MeshCommandQueue& queue,
    const HostTensor& tensor,
    ttsl::optional_reference<const MemoryConfig> memory_config) {
    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        tensor_spec_overriden_memory_config = tensor.tensor_spec().with_memory_config(*memory_config);
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &tensor.tensor_spec();
    auto mesh_buffer = allocate_device_buffer(queue.device(), *tensor_spec);
    return to_device_mesh_buffer(
        queue, tensor.get_legacy_host_storage(), mesh_buffer, *tensor_spec, tensor.tensor_topology());
}

// ======================================================================================
//                                  copy_to_host
// ======================================================================================

void copy_to_host(
    distributed::MeshCommandQueue& queue, const MeshTensor& device_tensor, HostTensor& host_tensor, bool blocking) {
    TT_FATAL(device_tensor.is_allocated(), "Buffer must be allocated on device.");

    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& device_storage = device_tensor.get_legacy_device_storage();
    const auto& mesh_buffer = device_storage.mesh_buffer;
    distributed::MeshDevice* device = mesh_buffer->device();

    const auto& distributed_host_buffer = host_tensor.get_distributed_host_buffer();

    // Host tensor must have pre-allocated buffers for all device shards.
    // However, it may have some extra shards. Drop them by "unwrapping" the distributed host buffer, and re-wrapping
    // only for those shards that are actually present on device.
    std::vector<std::pair<distributed::MeshCoordinate, std::optional<HostBuffer>>> shards;
    shards.reserve(device_storage.coords.size());
    for (const auto& device_coord : device_storage.coords) {
        shards.push_back({device_coord, distributed_host_buffer.get_shard(device_coord)});
    }

    DistributedHostBuffer dst_distributed_host_buffer = DistributedHostBuffer::create(device->get_view());
    const size_t expected_size_bytes = device_tensor.tensor_spec().compute_packed_buffer_size_bytes();
    for (const auto& [device_coord, host_buffer] : shards) {
        dst_distributed_host_buffer.emplace_shard(device_coord, [&]() {
            // Note the lambda is executed only for host-local shards.
            // If `host_buffer` is nullopt, the data was not correctly allocated on the host.
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

    queue.enqueue_read(mesh_buffer, dst_distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    host_tensor = HostTensor(
        HostStorage(std::move(dst_distributed_host_buffer)),
        device_tensor.tensor_spec(),
        device_tensor.tensor_topology());
}

void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    TT_FATAL(queue.device()->num_devices() == 1, "copy_to_host only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(dst)
            .region(region)};
    queue.enqueue_read_shards(shard_data_transfers, device_tensor.mesh_buffer_invariant_breaking(), blocking);
}

// ======================================================================================
//                                  copy_to_device
// ======================================================================================

void copy_to_device(distributed::MeshCommandQueue& queue, const HostTensor& host_tensor, MeshTensor& device_tensor) {
    TT_FATAL(device_tensor.is_allocated(), "Buffer must be allocated on device.");

    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();

    device_tensor = to_device_mesh_buffer(
        queue,
        host_tensor.get_legacy_host_storage(),
        mesh_buffer,
        device_tensor.tensor_spec(),
        host_tensor.tensor_topology());
}

void copy_to_device(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    TT_FATAL(queue.device()->num_devices() == 1, "copy_to_device only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(const_cast<std::byte*>(src))
            .region(region)};
    queue.enqueue_write_shards(device_tensor.mesh_buffer_invariant_breaking(), shard_data_transfers, false);
}

// ======================================================================================
//                                  .to_layout() helpers
// ======================================================================================

template <typename T>
std::vector<T> convert_layout_row_major_to_tile(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
    if (shape.width() * shape.height() == 0) {
        return std::vector<T>();
    }
    TT_FATAL(
        (shape.height() % tile.get_tile_shape()[0] == 0 && shape.width() % tile.get_tile_shape()[1] == 0),
        "Unsupported shape for tensor conversion from row-major to tile layout. The tensor shape height and width must "
        "be a multiple of tile height ({}) and width ({}), but the provided shape is {}",
        tile.get_tile_shape()[0],
        tile.get_tile_shape()[1],
        shape);

    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
}

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
                return convert_layout_row_major_to_tile(physical_shape, tile, input_data);
            case Layout::TILE:
                TT_FATAL(target_layout == Layout::ROW_MAJOR, "Unsupported layout conversion");
                return convert_layout_tile_to_row_major(physical_shape, tile, input_data);
            case Layout::INVALID: TT_THROW("Invalid layout");
        }
        TT_THROW("Unreachable");
    };

    return HostTensor(
        tensor.get_legacy_host_storage().transform(
            [&](const HostBuffer& buffer) { return HostBuffer(convert(buffer)); }),
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
    static_assert(std::is_same_v<T, bfloat8_b> || std::is_same_v<T, bfloat4_b>, "Invalid type T");
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
HostTensor to_layout_impl<bfloat8_b>(const HostTensor& tensor, Layout target_layout) {
    return to_layout_bfloat_impl<bfloat8_b>(tensor, target_layout);
}

template <>
HostTensor to_layout_impl<bfloat4_b>(const HostTensor& tensor, Layout target_layout) {
    return to_layout_bfloat_impl<bfloat4_b>(tensor, target_layout);
}

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

HostTensor to_layout(const HostTensor& tensor, Layout target_layout) {
    return dispatch(tensor.dtype(), [&]<typename T>() { return to_layout_impl<T>(tensor, target_layout); });
}

// ======================================================================================
//                                  .pad() and .unpad() helpers
// ======================================================================================

HostTensor pad_bfloat8_b(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value) {
    auto tile = tensor.tensor_spec().tile();

    auto input_buffer = tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(input_buffer.has_value(), "No host buffer available");
    auto input_packed_data = input_buffer->view_as<uint32_t>();
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_host_tensor = HostTensor(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        TensorTopology{});
    auto padded_float_tensor = tensor_impl::pad(float_host_tensor, output_padded_shape, input_tensor_start, pad_value);

    auto output_buffer =
        padded_float_tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(output_buffer.has_value(), "No host buffer available");
    auto output_float_data = output_buffer->view_as<const float>();
    auto output_packed_data =
        pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    TensorSpec output_spec(
        padded_float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT8_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            padded_float_tensor.logical_shape(),
            padded_float_tensor.padded_shape()));
    return HostTensor(std::move(output_uint32_buffer), output_spec, TensorTopology{});
}

HostTensor pad_bfloat4_b(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value) {
    auto tile = tensor.tensor_spec().tile();

    auto input_buffer = tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(input_buffer.has_value(), "No host buffer available");
    auto input_packed_data = input_buffer->view_as<uint32_t>();
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_host_tensor = HostTensor(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        TensorTopology{});
    auto padded_float_tensor = tensor_impl::pad(float_host_tensor, output_padded_shape, input_tensor_start, pad_value);

    auto output_buffer =
        padded_float_tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(output_buffer.has_value(), "No host buffer available");
    auto output_float_data = output_buffer->view_as<const float>();
    auto output_packed_data =
        pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    TensorSpec output_spec(
        padded_float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT4_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            padded_float_tensor.logical_shape(),
            padded_float_tensor.padded_shape()));
    return HostTensor(std::move(output_uint32_buffer), output_spec, TensorTopology{});
}

HostTensor unpad_bfloat8_b(const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    auto tile = tensor.tensor_spec().tile();

    auto input_buffer = tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(input_buffer.has_value(), "No host buffer available");
    auto input_packed_data = input_buffer->view_as<uint32_t>();
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_host_tensor = HostTensor(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        TensorTopology{});
    auto unpadded_float_tensor = tensor_impl::unpad(float_host_tensor, output_tensor_start, output_tensor_end);

    auto output_buffer =
        unpadded_float_tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(output_buffer.has_value(), "No host buffer available");
    auto output_float_data = output_buffer->view_as<const float>();
    auto output_packed_data =
        pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));

    TensorSpec output_spec(
        unpadded_float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT8_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            unpadded_float_tensor.logical_shape(),
            unpadded_float_tensor.padded_shape()));
    return HostTensor(std::move(output_uint32_buffer), output_spec, TensorTopology{});
}

HostTensor unpad_bfloat4_b(const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    auto tile = tensor.tensor_spec().tile();

    auto input_buffer = tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(input_buffer.has_value(), "No host buffer available");
    auto input_packed_data = input_buffer->view_as<uint32_t>();
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_host_tensor = HostTensor(
        std::move(input_float_buffer),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::FLOAT32,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        TensorTopology{});
    auto unpadded_float_tensor = tensor_impl::unpad(float_host_tensor, output_tensor_start, output_tensor_end);

    auto output_buffer =
        unpadded_float_tensor.get_legacy_host_storage().buffer().get_shard(distributed::MeshCoordinate(0, 0));
    TT_FATAL(output_buffer.has_value(), "No host buffer available");
    auto output_float_data = output_buffer->view_as<const float>();
    auto output_packed_data =
        pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));

    TensorSpec output_spec(
        unpadded_float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT4_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            unpadded_float_tensor.logical_shape(),
            unpadded_float_tensor.padded_shape()));
    return HostTensor(std::move(output_uint32_buffer), output_spec, TensorTopology{});
}

template <typename T>
HostTensor pad_impl(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value) {
    auto pad_value_ = static_cast<T>(pad_value);
    auto input_padded_shape = tensor.padded_shape();
    if (input_padded_shape.rank() < 2) {
        input_padded_shape = input_padded_shape.to_rank(2);
    }

    auto pad_fn = [&input_padded_shape, &output_padded_shape, &input_tensor_start, &pad_value_](
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

        auto input_strides = compute_strides(input_padded_shape);
        auto output_strides = compute_strides(output_padded_shape);

        ttsl::SmallVector<size_t> coords(rank - 1, 0);

        bool processed_all_coords = false;
        while (!processed_all_coords) {
            size_t input_offset = 0;
            size_t output_offset = 0;
            for (size_t i = 0; i < rank - 1; i++) {
                input_offset += coords[i] * input_strides[i];
                output_offset += (coords[i] + input_tensor_start[i]) * output_strides[i];
            }
            output_offset += input_tensor_start[rank - 1];

            std::memcpy(
                output_buffer.data() + output_offset,
                input_buffer.data() + input_offset,
                static_cast<size_t>(input_padded_shape[rank - 1]) * sizeof(T));

            processed_all_coords = true;
            for (int i = static_cast<int>(rank) - 2; i >= 0; i--) {
                coords[i]++;
                if (coords[i] < input_padded_shape[i]) {
                    processed_all_coords = false;
                    break;
                }
                coords[i] = 0;
            }
        }

        return output_buffer;
    };

    return HostTensor(
        tensor.get_legacy_host_storage().transform(
            [&](const HostBuffer& buffer) { return HostBuffer(pad_fn(buffer)); }),
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
HostTensor pad_impl<bfloat8_b>(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value) {
    return pad_bfloat8_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

template <>
HostTensor pad_impl<bfloat4_b>(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value) {
    return pad_bfloat4_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

template <typename T>
HostTensor unpad_impl(const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    const auto& input_shape = tensor.padded_shape();
    const auto input_strides = compute_strides(input_shape);

    ttsl::SmallVector<uint32_t> output_shape;
    for (auto i = 0; i < input_shape.rank(); i++) {
        TT_ASSERT(output_tensor_start[i] <= input_shape[i]);
        TT_ASSERT(output_tensor_end[i] <= input_shape[i]);
        TT_ASSERT(output_tensor_start[i] <= output_tensor_end[i]);
        output_shape.push_back(output_tensor_end[i] - output_tensor_start[i]);
    }

    auto unpad_fn = [&input_shape, &input_strides, &output_shape, &output_tensor_start, &output_tensor_end](
                        const HostBuffer& input_host_buffer) {
        const auto input_buffer = input_host_buffer.view_as<T>();
        ttsl::SmallVector<uint32_t> input_indices(input_shape.rank(), 0);

        auto flat_output_index = 0;
        auto output_buffer = std::vector<T>(Shape(output_shape).volume());

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

    return HostTensor(
        tensor.get_legacy_host_storage().transform(
            [&](const HostBuffer& buffer) { return HostBuffer(unpad_fn(buffer)); }),
        TensorSpec(
            Shape(output_shape),
            TensorLayout(tensor.dtype(), PageConfig(tensor.layout(), tensor.tensor_spec().tile()), MemoryConfig{})),
        tensor.tensor_topology());
}

template <>
HostTensor unpad_impl<bfloat8_b>(
    const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    return unpad_bfloat8_b(tensor, output_tensor_start, output_tensor_end);
}

template <>
HostTensor unpad_impl<bfloat4_b>(
    const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    return unpad_bfloat4_b(tensor, output_tensor_start, output_tensor_end);
}

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================

HostTensor pad(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value) {
    return dispatch(tensor.dtype(), [&]<typename T>() {
        return pad_impl<T>(tensor, output_padded_shape, input_tensor_start, pad_value);
    });
}

HostTensor unpad(const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    return dispatch(
        tensor.dtype(), [&]<typename T>() { return unpad_impl<T>(tensor, output_tensor_start, output_tensor_end); });
}

HostTensor pad_to_tile(const HostTensor& input_tensor, float pad_value) {
    uint32_t height = input_tensor.padded_shape()[-2];
    uint32_t width = input_tensor.padded_shape()[-1];
    uint32_t padded_height = round_up(height, constants::TILE_HEIGHT);
    uint32_t padded_width = round_up(width, constants::TILE_WIDTH);

    ttsl::SmallVector<uint32_t> padded_shape;
    ttsl::SmallVector<uint32_t> input_tensor_start;

    for (auto index = 0; index < static_cast<int>(input_tensor.padded_shape().rank()) - 2; index++) {
        padded_shape.push_back(input_tensor.padded_shape()[index]);
        input_tensor_start.push_back(0);
    }

    padded_shape.push_back(padded_height);
    padded_shape.push_back(padded_width);
    input_tensor_start.push_back(0);
    input_tensor_start.push_back(0);

    return pad(input_tensor, Shape(std::move(padded_shape)), Shape{std::move(input_tensor_start)}, pad_value);
}

HostTensor unpad_from_tile(const HostTensor& input_tensor, const Shape& output_tensor_shape) {
    for (auto index = -3; index >= -static_cast<int>(input_tensor.padded_shape().rank()); index--) {
        TT_ASSERT(
            input_tensor.logical_shape()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_ASSERT(
        input_tensor.padded_shape()[-2] % constants::TILE_HEIGHT == 0 &&
            input_tensor.padded_shape()[-1] % constants::TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(
        input_tensor.padded_shape()[-2] < output_tensor_shape[-2] + constants::TILE_HEIGHT &&
            input_tensor.padded_shape()[-1] < output_tensor_shape[-1] + constants::TILE_WIDTH,
        "Last 2 dims of output must be within range to have been padded to input");
    Shape output_tensor_start(ttsl::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 0));
    Shape output_tensor_end(ttsl::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 1));
    for (int index = -1; index >= -static_cast<int>(output_tensor_shape.rank()); index--) {
        output_tensor_end[index] = output_tensor_shape[index];
    }
    return unpad(input_tensor, output_tensor_start, output_tensor_end);
}

// ======================================================================================
//                                  .to_dtype() helpers
// ======================================================================================

namespace to_dtype_detail {

struct bfloat4_tag {};
struct bfloat8_tag {};

HostStorage preprocess_storage(const HostStorage& input_storage, const DataType input_dtype) {
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if (input_dtype == DataType::BFLOAT8_B) {
        return input_storage.transform([&](const HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return HostBuffer(std::move(float_unpacked_data));
        });
    }
    if (input_dtype == DataType::BFLOAT4_B) {
        return input_storage.transform([&](const HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return HostBuffer(std::move(float_unpacked_data));
        });
    }
    return input_storage;
}

template <typename SrcType, typename DstType>
HostStorage transform_storage(const TensorSpec& input_tensor_spec, const HostStorage& input_storage) {
    if constexpr (std::is_same_v<SrcType, DstType>) {
        return input_storage;
    } else if constexpr (std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag>) {
        auto transform_fn = [&](const HostBuffer& buffer) {
            ttsl::Span<const SrcType> data = buffer.view_as<const SrcType>();
            std::vector<SrcType> tilized_data;
            if (input_tensor_spec.layout() == Layout::ROW_MAJOR) {
                tilized_data = convert_layout_row_major_to_tile(
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
            return HostBuffer(std::move(float_packed_data));
        };

        return input_storage.transform(transform_fn);
    } else {
        auto transform_fn = [&](const HostBuffer& buffer) {
            auto data = buffer.view_as<const SrcType>();
            std::vector<DstType> output_vector(data.size());
            std::transform(data.begin(), data.end(), output_vector.begin(), [](SrcType value) {
                return static_cast<DstType>(value);
            });
            return HostBuffer(std::move(output_vector));
        };

        return input_storage.transform(transform_fn);
    }
}

}  // namespace to_dtype_detail

// ======================================================================================
//                                  .to_dtype()
// ======================================================================================

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype) {
    const auto src_type = input_tensor.dtype();
    if (src_type == dtype) {
        return input_tensor;
    }

    auto input_storage = to_dtype_detail::preprocess_storage(input_tensor.get_legacy_host_storage(), src_type);

    auto output_storage = [src_type, dst_type = dtype, &input_tensor, &input_storage]() {
        auto with_src_and_dst = [&]<typename SrcType, typename DstType>() {
            return to_dtype_detail::transform_storage<SrcType, DstType>(input_tensor.tensor_spec(), input_storage);
        };

        auto with_src = [dst_type, &with_src_and_dst]<typename SrcType>() {
            switch (dst_type) {
                case DataType::BFLOAT4_B: return with_src_and_dst.operator()<SrcType, to_dtype_detail::bfloat4_tag>();
                case DataType::BFLOAT8_B: return with_src_and_dst.operator()<SrcType, to_dtype_detail::bfloat8_tag>();
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
        TensorLayout::fromPaddedShape(
            dtype,
            PageConfig(layout, input_tensor.tensor_spec().tile()),
            input_tensor.tensor_spec().memory_config(),
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));

    return HostTensor(HostStorage(std::move(output_storage)), output_spec, input_tensor.tensor_topology());
}

}  // namespace tt::tt_metal::tensor_impl
