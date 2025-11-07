// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/tensor/unit_mesh/unit_mesh_utils.hpp"

#include "tt-metalium/tensor/storage.hpp"
#include <tt-metalium/distributed/tensor_topology.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/allocator_state.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::unit_mesh {

namespace {

void synchronize_parent_allocator_with_submeshes(tt::tt_metal::distributed::MeshDevice* parent_mesh) {
    auto* parent_allocator = parent_mesh->allocator().get();
    TT_FATAL(parent_allocator != nullptr, "Parent mesh must have an allocator");

    tt::tt_metal::AllocatorState merged_state;
    for (const auto& submesh : parent_mesh->get_submeshes()) {
        auto* submesh_allocator = submesh->allocator().get();
        TT_FATAL(submesh_allocator != nullptr, "Submesh must have an allocator");
        merged_state.merge(submesh_allocator->extract_state());
    }

    parent_allocator->override_state(merged_state);
}

}  // namespace

Tensor aggregate(const std::vector<tt::tt_metal::Tensor>& tensors) {
    TT_FATAL(!tensors.empty(), "Cannot aggregate empty tensor vector");

    // Validate all tensors are allocated on the unit meshes.
    std::vector<tt::tt_metal::distributed::MeshDevice*> devices;
    devices.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        auto* device = tensor.device();
        TT_FATAL(device != nullptr, "All tensors must be on device");

        const auto& shape = device->shape();
        TT_FATAL(shape.mesh_size() == 1, "Expected unit mesh (1x1), but got mesh of size {}", shape.mesh_size());

        devices.push_back(device);
    }

    // Validate all devices have the same parent mesh.
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> parent_mesh = tensors[0].device()->get_parent_mesh();
    TT_FATAL(parent_mesh != nullptr, "First device must have a parent mesh");
    TT_FATAL(parent_mesh->shape().mesh_size() == tensors.size(), "Input tensors must span the entire parent mesh");

    for (size_t i = 1; i < devices.size(); i++) {
        TT_FATAL(
            devices[i]->get_parent_mesh().get() == parent_mesh.get(),
            "All tensors must belong to the same parent mesh");
    }

    // Validate all tensor specs and mesh buffer addresses are the same.
    const auto& reference_spec = tensors[0].tensor_spec();
    auto reference_address = tensors[0].mesh_buffer()->address();
    for (size_t i = 1; i < tensors.size(); i++) {
        TT_FATAL(tensors[i].tensor_spec() == reference_spec, "All tensors must have the same TensorSpec");
        TT_FATAL(
            tensors[i].mesh_buffer()->address() == reference_address, "All mesh buffers must be at the same address");
    }

    synchronize_parent_allocator_with_submeshes(parent_mesh.get());

    // Create a new mesh tensor for parent mesh.
    const auto& reference_buffer = tensors[0].mesh_buffer();
    auto mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        reference_buffer->global_config(),
        reference_buffer->device_local_config(),
        parent_mesh.get(),
        reference_address);

    std::vector<tt::tt_metal::distributed::MeshCoordinate> coords;
    coords.reserve(parent_mesh->shape().mesh_size());
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(parent_mesh->shape())) {
        coords.push_back(coord);
    }

    tt::tt_metal::DeviceStorage device_storage(std::move(mesh_buffer), std::move(coords));

    return Tensor(
        std::move(device_storage),
        reference_spec,
        tt::tt_metal::TensorTopology::create_sharded_tensor_topology(
            tt::tt_metal::distributed::MeshShape(parent_mesh->shape().mesh_size()), /*shard_dim=*/0));
}

std::vector<tt::tt_metal::Tensor> disaggregate(const tt::tt_metal::Tensor& tensor) {
    using namespace tt::tt_metal;

    // Validate the tensor is allocated on mesh device, that is parent mesh of unit meshes.
    auto* mesh_device = tensor.device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be allocated on a mesh device");
    const auto submeshes = mesh_device->get_submeshes();
    TT_FATAL(
        submeshes.size() == mesh_device->shape().mesh_size(),
        "Number of submeshes ({}) must match mesh size ({})",
        submeshes.size(),
        mesh_device->shape().mesh_size());
    for (const auto& submesh : submeshes) {
        const auto& submesh_shape = submesh->shape();
        TT_FATAL(
            submesh_shape.mesh_size() == 1,
            "All submeshes must be a unit mesh (1x1), but got mesh of size {}",
            submesh_shape.mesh_size());
    }

    const auto& input_mesh_buffer = tensor.mesh_buffer();
    const auto input_address = input_mesh_buffer->address();
    const auto& reference_spec = tensor.tensor_spec();

    // For all unit meshes, create individual mesh buffers with the same address.
    std::vector<Tensor> result;
    result.reserve(submeshes.size());
    for (const auto& submesh : submeshes) {
        auto mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            input_mesh_buffer->global_config(), input_mesh_buffer->device_local_config(), submesh.get(), input_address);

        DeviceStorage device_storage(
            std::move(mesh_buffer), std::vector<distributed::MeshCoordinate>{distributed::MeshCoordinate(0, 0)});

        result.push_back(Tensor(std::move(device_storage), reference_spec, TensorTopology{}));
    }

    return result;
}

}  // namespace tt::tt_metal::experimental::unit_mesh
