// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>
#include <tt-metalium/experimental/tensor/device_tensor.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace {

using namespace tt::tt_metal;

DeviceStorage replicate_to_mesh_buffer(
    distributed::MeshCommandQueue& cq,
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

    cq.enqueue_write_mesh_buffer(mesh_buffer, data_to_write.data(), /*blocking=*/false);

    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(mesh_device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        coords.push_back(coord);
    }
    return DeviceStorage(mesh_buffer, std::move(coords));
}

DeviceStorage write_to_mesh_buffer(
    distributed::MeshCommandQueue& cq,
    const DistributedHostBuffer& distributed_host_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer) {
    cq.enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(distributed_host_buffer.shard_coords().size());
    std::copy(
        distributed_host_buffer.shard_coords().begin(),
        distributed_host_buffer.shard_coords().end(),
        std::back_inserter(coords));
    return DeviceStorage(mesh_buffer, std::move(coords));
}

}  // namespace

namespace tt::tt_metal {

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

namespace detail {

struct bfloat4_tag {};
struct bfloat8_tag {};

// Preprocess the storage to unpack the bfloat8/4 tiles into float32.
tt::tt_metal::HostStorage preprocess_storage(
    const tt::tt_metal::HostStorage& input_storage, const DataType input_dtype) {
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if (input_dtype == DataType::BFLOAT8_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    if (input_dtype == DataType::BFLOAT4_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    return input_storage;
}

template <typename SrcType, typename DstType>
tt::tt_metal::HostStorage transform_storage(
    const tt::tt_metal::TensorSpec& input_tensor_spec, const tt::tt_metal::HostStorage& input_storage) {
    if constexpr (std::is_same_v<SrcType, DstType>) {
        return input_storage;
    } else if constexpr (std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag>) {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const SrcType> data = buffer.view_as<const SrcType>();
            std::vector<SrcType> tilized_data;  // empty if `data` is already in tile layout.
            if (input_tensor_spec.layout() == Layout::ROW_MAJOR) {
                tilized_data = tensor_impl::convert_layout_row_major_to_tile(
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

        return input_storage.transform(transform_fn);
    } else {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            auto data = buffer.view_as<const SrcType>();
            std::vector<DstType> output_vector(data.size());
            std::transform(data.begin(), data.end(), output_vector.begin(), [](SrcType value) {
                return static_cast<DstType>(value);
            });
            return tt::tt_metal::HostBuffer(std::move(output_vector));
        };

        return input_storage.transform(transform_fn);
    }
}

}  // namespace detail

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype) {
    const auto src_type = input_tensor.dtype();
    if (src_type == dtype) {
        return input_tensor;
    }

    auto input_storage = detail::preprocess_storage(input_tensor.host_storage(), src_type);

    auto output_storage = [src_type, dst_type = dtype, &input_tensor, &input_storage]() {
        auto with_src_and_dst = [&]<typename SrcType, typename DstType>() {
            return detail::transform_storage<SrcType, DstType>(input_tensor.tensor_spec(), input_storage);
        };

        auto with_src = [dst_type, &with_src_and_dst]<typename SrcType>() {
            switch (dst_type) {
                case DataType::BFLOAT4_B: return with_src_and_dst.operator()<SrcType, detail::bfloat4_tag>();
                case DataType::BFLOAT8_B: return with_src_and_dst.operator()<SrcType, detail::bfloat8_tag>();
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

    return HostTensor(
        tt::tt_metal::HostStorage(std::move(output_storage)), output_spec, input_tensor.tensor_topology());
}

void TransferToDevice(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, DeviceTensor& device_tensor, bool blocking) {
    TT_FATAL(device_tensor.is_allocated(), "DeviceTensor must be allocated");
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Shape mismatch");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Dtype mismatch");

    const auto& host_buffer = host_tensor.host_storage().buffer();
    auto mesh_buffer = device_tensor.mesh_buffer();

    cq.enqueue_write(mesh_buffer, host_buffer, blocking);
}

namespace tensor_impl {

std::pair<DeviceStorage, TensorTopology> to_device_mesh_buffer(
    distributed::MeshCommandQueue& cq,
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
        return {
            replicate_to_mesh_buffer(cq, *device_buffer, mesh_buffer, tensor_spec),
            TensorTopology::create_fully_replicated_tensor_topology(mesh_device_shape)};
    }

    TT_FATAL(
        host_storage_shape == mesh_device_shape,
        "Distributed host buffer has different shape {} than the mesh device {}",
        host_storage_shape,
        mesh_device_shape);
    return {write_to_mesh_buffer(cq, host_storage.buffer(), mesh_buffer), tensor_topology};
}

}  // namespace tensor_impl

}  // namespace tt::tt_metal
