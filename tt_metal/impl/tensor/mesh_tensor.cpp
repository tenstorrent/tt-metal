// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <tt_stl/small_vector.hpp>

namespace tt::tt_metal {

MeshTensor MeshTensor::allocate_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* device) {
    auto mesh_buffer = tensor_impl::allocate_device_buffer(device, tensor_spec);
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(device->shape())) {
        coords.push_back(coord);
    }
    DeviceStorage device_storage(std::move(mesh_buffer), coords);

    ttsl::SmallVector<distributed::MeshMapperConfig::Placement> placements(device->shape().dims());
    for (size_t i = 0; i < device->shape().dims(); i++) {
        placements[i] = distributed::MeshMapperConfig::Replicate{};
    }

    auto tensor_topology = TensorTopology{device->shape(), placements, coords};
    return MeshTensor(std::move(device_storage), tensor_spec, tensor_topology);
}

}  // namespace tt::tt_metal
