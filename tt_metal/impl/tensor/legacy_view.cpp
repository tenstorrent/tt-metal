// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/details/legacy_view.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::tensor_impl {

// TODO (#25340): Review tensor topology logic for reshape
HostTensor view(const HostTensor& tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    if (tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }
    bool is_row_major = tensor.layout() == Layout::ROW_MAJOR;
    bool changing_last_dim = new_padded_shape[-1] != tensor.padded_shape()[-1];
    const auto& input_memory_config = tensor.memory_config();
    TT_FATAL(
        !input_memory_config.is_sharded() || !changing_last_dim ||
            input_memory_config.shard_spec()->shape[1] == tensor.padded_shape()[-1],
        "Changing the last dimension of a sharded tensor is not supported unless the shard width matches the input "
        "last dimension. "
        "Input shape: {}, New shape: {}, Shard width: {}",
        tensor.padded_shape(),
        new_padded_shape,
        input_memory_config.shard_spec()->shape[1]);

    auto output_memory_config = input_memory_config;
    if (is_row_major && input_memory_config.is_sharded() && changing_last_dim) {
        auto shard_spec = input_memory_config.shard_spec().value();
        auto shard_volume = shard_spec.numel();
        shard_spec.shape[1] = new_padded_shape[-1];
        shard_spec.shape[0] = shard_volume / shard_spec.shape[1];
        output_memory_config =
            MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
    }

    auto new_spec = TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            tensor.dtype(),
            tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    return HostTensor(tensor.get_legacy_host_storage(), new_spec, tensor.tensor_topology());
}

// TODO (#25340): Review tensor topology logic for reshape
// TODO: We should force MeshTensor to be moved in. This essentially copies the MeshTensor.
MeshTensor view(const MeshTensor& tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    if (tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }
    bool is_row_major = tensor.layout() == Layout::ROW_MAJOR;
    bool changing_last_dim = new_padded_shape[-1] != tensor.padded_shape()[-1];
    const auto& input_memory_config = tensor.memory_config();
    TT_FATAL(
        !input_memory_config.is_sharded() || !changing_last_dim ||
            input_memory_config.shard_spec()->shape[1] == tensor.padded_shape()[-1],
        "Changing the last dimension of a sharded tensor is not supported unless the shard width matches the input "
        "last dimension. "
        "Input shape: {}, New shape: {}, Shard width: {}",
        tensor.padded_shape(),
        new_padded_shape,
        input_memory_config.shard_spec()->shape[1]);

    auto output_memory_config = input_memory_config;
    if (is_row_major && input_memory_config.is_sharded() && changing_last_dim) {
        auto shard_spec = input_memory_config.shard_spec().value();
        auto shard_volume = shard_spec.numel();
        shard_spec.shape[1] = new_padded_shape[-1];
        shard_spec.shape[0] = shard_volume / shard_spec.shape[1];
        output_memory_config =
            MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
    }

    auto new_spec = TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            tensor.dtype(),
            tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    auto device_storage = tensor.get_legacy_device_storage();
    if (tensor.layout() != Layout::ROW_MAJOR || !changing_last_dim) {
        return MeshTensor(std::move(device_storage), new_spec, tensor.tensor_topology());
    }
    if (!tensor.memory_config().is_sharded()) {
        auto* device_buffer = device_storage.get_buffer();
        auto page_size_bytes = new_spec.compute_page_size_bytes();
        device_buffer->set_page_size(page_size_bytes);
        return MeshTensor(std::move(device_storage), new_spec, tensor.tensor_topology());
    }

    ShardSpec new_shard_spec = output_memory_config.shard_spec().value();
    std::array<uint32_t, 2> shard_page_shape = {1, new_shard_spec.shape[1]};
    std::array<uint32_t, 2> tensor2d_shape_in_pages = {
        static_cast<uint32_t>(new_spec.physical_shape().height() / shard_page_shape[0]),
        static_cast<uint32_t>(new_spec.physical_shape().width() / shard_page_shape[1])};
    ShardSpecBuffer new_shard_spec_buffer = ShardSpecBuffer(new_shard_spec, shard_page_shape, tensor2d_shape_in_pages);

    Shape tensor_shape_pages(tensor2d_shape_in_pages);
    Shape shard_shape_pages(new_shard_spec_buffer.shape_in_pages());
    BufferDistributionSpec new_buffer_dist_spec =
        BufferDistributionSpec(tensor_shape_pages, shard_shape_pages, new_shard_spec.grid, new_shard_spec.orientation);

    auto device_local_config = device_storage.mesh_buffer->device_local_config();
    auto& sharding_args = device_local_config.sharding_args;
    BufferShardingArgs new_sharding_args(new_buffer_dist_spec, new_shard_spec_buffer, sharding_args.buffer_layout());

    distributed::DeviceLocalBufferConfig new_device_config = {
        .page_size = new_spec.compute_page_size_bytes(),
        .buffer_type = device_local_config.buffer_type,
        .sharding_args = new_sharding_args,
        .bottom_up = device_local_config.bottom_up};

    auto view_mesh_buffer = distributed::MeshBuffer::create(
        device_storage.mesh_buffer->global_config(),
        new_device_config,
        device_storage.mesh_buffer->device(),
        device_storage.mesh_buffer->address());
    DeviceStorage view_storage(view_mesh_buffer, device_storage.coords, device_storage.get_root_mesh_buffer());

    return MeshTensor(view_storage, new_spec, tensor.tensor_topology());
}

}  // namespace tt::tt_metal::tensor_impl
