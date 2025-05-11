// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_mesh_buffer.hpp>
#include <tt-metalium/assert.hpp>

#include <functional>
#include <vector>

namespace tt::tt_metal {

HostMeshBuffer HostMeshBuffer::create_replicated(HostBuffer host_buffer) {
    return HostMeshBuffer(Replicated{std::move(host_buffer)});
}

HostMeshBuffer HostMeshBuffer::create_sharded(size_t global_size) {
    return HostMeshBuffer(Sharded{
        .global_to_local_index = [](size_t global_index) { return global_index; },
        .local_buffers = std::vector<HostBuffer>(global_size)});
}

HostMeshBuffer HostMeshBuffer::create_sharded(
    const distributed::MeshShape& global_shape,
    const distributed::MeshShape& local_shape,
    const distributed::MeshCoordinate& local_offset) {
    TT_FATAL(
        global_shape.dims() == local_shape.dims(),
        "Global shape {} and local shape {} must have the same dimensions",
        global_shape,
        local_shape);
    TT_FATAL(
        global_shape.dims() == local_offset.dims(),
        "Global shape {} and local offset {} must have the same dimensions",
        global_shape,
        local_offset);

    for (size_t dim = 0; dim < global_shape.dims(); ++dim) {
        TT_FATAL(
            local_offset[dim] + local_shape[dim] <= global_shape[dim],
            "Local shape exceeds global shape at dimension {}: {} + {} > {}",
            dim,
            local_offset[dim],
            local_shape[dim],
            global_shape[dim]);
    }

    auto global_to_local_index = [global_shape, local_shape, local_offset](size_t global_idx) -> std::optional<size_t> {
        TT_FATAL(
            global_idx < global_shape.mesh_size(),
            "Global index {} is out of bounds for global shape {}",
            global_idx,
            global_shape);

        size_t local_idx = 0;
        size_t remaining = global_idx;
        for (size_t dim = 0; dim < global_shape.dims(); ++dim) {
            const uint32_t coord = remaining / global_shape.get_stride(dim);
            remaining %= global_shape.get_stride(dim);

            if (coord < local_offset[dim] || coord >= local_offset[dim] + local_shape[dim]) {
                return std::nullopt;
            }

            local_idx += (coord - local_offset[dim]) * local_shape.get_stride(dim);
        }

        return local_idx;
    };
    return HostMeshBuffer(Sharded{
        .global_to_local_index = global_to_local_index,
        .local_buffers = std::vector<HostBuffer>(local_shape.mesh_size())});
}

std::optional<HostBuffer> HostMeshBuffer::get_buffer(size_t linear_index) const {
    return std::visit(
        tt::stl::overloaded{
            [](const Replicated& replicated) -> std::optional<HostBuffer> {
                return std::make_optional(replicated.buffer);
            },
            [linear_index](const Sharded& sharded) -> std::optional<HostBuffer> {
                const auto local_index = sharded.global_to_local_index(linear_index);
                if (!local_index) {
                    return std::nullopt;
                }
                return std::make_optional(sharded.local_buffers.at(local_index.value()));
            }},
        data_);
}

void HostMeshBuffer::emplace_buffer(size_t linear_index, HostBuffer buffer) {
    std::visit(
        tt::stl::overloaded{
            [&buffer](Replicated& replicated) { replicated.buffer = std::move(buffer); },
            [&buffer, linear_index](Sharded& sharded) {
                const auto local_index = sharded.global_to_local_index(linear_index);
                if (!local_index) {
                    return;
                }
                sharded.local_buffers.at(local_index.value()) = std::move(buffer);
            }},
        data_);
}

void HostMeshBuffer::transform(const TransformFn& fn) {
    std::visit(
        tt::stl::overloaded{
            [&fn](Replicated& replicated) { replicated.buffer = fn(replicated.buffer, 0); },
            [&fn](Sharded& sharded) {
                for (size_t i = 0; i < sharded.local_buffers.size(); ++i) {
                    sharded.local_buffers.at(i) = fn(sharded.local_buffers.at(i), i);
                }
            }},
        data_);
}

void HostMeshBuffer::apply(const ApplyFn& fn) {
    std::visit(
        tt::stl::overloaded{
            [&fn](Replicated& replicated) { fn(replicated.buffer, 0); },
            [&fn](Sharded& sharded) {
                for (size_t i = 0; i < sharded.local_buffers.size(); ++i) {
                    fn(sharded.local_buffers.at(i), i);
                }
            }},
        data_);
}

bool HostMeshBuffer::is_allocated() const {
    return std::visit(
        tt::stl::overloaded{
            [](const Replicated& replicated) { return replicated.buffer.is_allocated(); },
            [](const Sharded& sharded) {
                return std::all_of(sharded.local_buffers.begin(), sharded.local_buffers.end(), [](const HostBuffer& b) {
                    return b.is_allocated();
                });
            }},
        data_);
}

void HostMeshBuffer::deallocate() {
    std::visit(
        tt::stl::overloaded{
            [](Replicated& replicated) { replicated.buffer.deallocate(); },
            [](Sharded& sharded) {
                for (auto& buffer : sharded.local_buffers) {
                    buffer.deallocate();
                }
            }},
        data_);
}

}  // namespace tt::tt_metal
