// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/h2d_stream_service_descriptor.hpp>

#include "h2d_stream_service_descriptor_generated.h"
#include "hd_socket_descriptor_serialization.hpp"

#include <internal/descriptor_file.hpp>

#include <tt_stl/assert.hpp>

#include <variant>

namespace tt::tt_metal::distributed {

namespace {

std::pair<flatbuffer::PlacementKind, int32_t> encode_placement(
    const MeshMapperConfig::Placement& p) {
    return std::visit(
        [](auto&& arg) -> std::pair<flatbuffer::PlacementKind, int32_t> {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, MeshMapperConfig::Replicate>) {
                return {flatbuffer::PlacementKind::Replicate, 0};
            } else if constexpr (std::is_same_v<T, MeshMapperConfig::Shard>) {
                return {flatbuffer::PlacementKind::Shard, static_cast<int32_t>(arg.dim)};
            } else {
                static_assert(sizeof(T) == 0, "Unhandled placement variant");
            }
        },
        p);
}

MeshMapperConfig::Placement decode_placement(const flatbuffer::Placement& fb) {
    switch (fb.kind()) {
        case flatbuffer::PlacementKind::Replicate: return MeshMapperConfig::Replicate{};
        case flatbuffer::PlacementKind::Shard:
            return MeshMapperConfig::Shard{static_cast<int>(fb.dim())};
    }
    TT_THROW("Unknown PlacementKind: {}", static_cast<int>(fb.kind()));
}

}  // namespace

void H2DStreamServiceDescriptor::write_to_file(const std::string& path) const {
    flatbuffers::FlatBufferBuilder builder(2048);

    std::vector<uint32_t> shape_vec(global_shape.cbegin(), global_shape.cend());
    auto fb_shape = builder.CreateVector(shape_vec);
    auto fb_spec = flatbuffer::CreateTensorSpecLite(builder, fb_shape, static_cast<uint32_t>(global_dtype));

    std::vector<uint32_t> mesh_shape_vec(mesh_shape.cbegin(), mesh_shape.cend());
    auto fb_mesh_shape = builder.CreateVector(mesh_shape_vec);

    std::vector<flatbuffers::Offset<flatbuffer::Placement>> placement_offsets;
    placement_offsets.reserve(mapper_config.placements.size());
    for (const auto& p : mapper_config.placements) {
        auto [kind, dim] = encode_placement(p);
        placement_offsets.push_back(flatbuffer::CreatePlacement(builder, kind, dim));
    }
    auto fb_placements = builder.CreateVector(placement_offsets);

    std::vector<uint32_t> shape_override_vec;
    if (mapper_config.mesh_shape_override.has_value()) {
        const auto& sh = *mapper_config.mesh_shape_override;
        shape_override_vec.assign(sh.cbegin(), sh.cend());
    }
    auto fb_shape_override = builder.CreateVector(shape_override_vec);
    auto fb_completion_shm_name = builder.CreateString(completion_shm_name);

    std::vector<flatbuffers::Offset<flatbuffer::PerCoordEntry>> entry_offsets;
    entry_offsets.reserve(per_coord_entries.size());
    for (const auto& [coord, socket_desc] : per_coord_entries) {
        std::vector<uint32_t> coord_vec(coord.coords().begin(), coord.coords().end());
        auto fb_coord = builder.CreateVector(coord_vec);
        auto fb_socket = to_flatbuffer(builder, socket_desc);
        entry_offsets.push_back(flatbuffer::CreatePerCoordEntry(builder, fb_coord, fb_socket));
    }
    auto fb_entries = builder.CreateVector(entry_offsets);

    auto fb_desc = flatbuffer::CreateH2DStreamServiceDescriptor(
        builder,
        kVersion,
        fb_spec,
        fb_mesh_shape,
        fb_placements,
        fb_shape_override,
        socket_page_size,
        num_socket_pages,
        metadata_size_bytes,
        static_cast<uint32_t>(socket_buffer_type),
        fb_completion_shm_name,
        completion_shm_size,
        completion_issued_offset,
        completion_completed_offset,
        completion_completed_stride,
        fb_entries);
    builder.Finish(fb_desc);
    internal::write_file_atomically(path, ttsl::Span<const uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
}

H2DStreamServiceDescriptor H2DStreamServiceDescriptor::wait_and_read(
    const std::string& path, uint32_t timeout_ms) {
    const auto buf = internal::wait_and_read_file(path, timeout_ms);
    const auto* fb = flatbuffer::GetH2DStreamServiceDescriptor(buf.data());
    TT_FATAL(fb, "Failed to parse flatbuffer service descriptor from: {}", path);
    TT_FATAL(
        fb->version() == kVersion,
        "H2DStreamServiceDescriptor version mismatch at {}: got {}, expected {}",
        path,
        fb->version(),
        kVersion);

    H2DStreamServiceDescriptor desc;

    const auto* fb_spec = fb->global_spec();
    TT_FATAL(fb_spec != nullptr, "Service descriptor missing global_spec");
    {
        const auto* fb_shape = fb_spec->shape();
        TT_FATAL(fb_shape != nullptr, "Service descriptor missing global_spec.shape");
        desc.global_shape =
            tt::tt_metal::Shape(ttsl::Span<const uint32_t>(fb_shape->data(), fb_shape->size()));
        desc.global_dtype = static_cast<DataType>(fb_spec->dtype());
    }

    {
        const auto* fb_mesh_shape = fb->mesh_shape();
        TT_FATAL(fb_mesh_shape != nullptr, "Service descriptor missing mesh_shape");
        desc.mesh_shape =
            MeshShape(ttsl::Span<const uint32_t>(fb_mesh_shape->data(), fb_mesh_shape->size()));
    }

    {
        ttsl::SmallVector<MeshMapperConfig::Placement> placements;
        const auto* fb_placements = fb->mapper_placements();
        TT_FATAL(fb_placements != nullptr, "Service descriptor missing mapper_placements");
        placements.reserve(fb_placements->size());
        for (const auto* fb_p : *fb_placements) {
            placements.push_back(decode_placement(*fb_p));
        }
        std::optional<MeshShape> shape_override;
        const auto* fb_override = fb->mapper_shape_override();
        if (fb_override != nullptr && fb_override->size() > 0) {
            shape_override =
                MeshShape(ttsl::Span<const uint32_t>(fb_override->data(), fb_override->size()));
        }
        desc.mapper_config =
            MeshMapperConfig{.placements = placements, .mesh_shape_override = shape_override};
    }

    desc.socket_page_size = fb->socket_page_size();
    desc.num_socket_pages = fb->num_socket_pages();
    desc.metadata_size_bytes = fb->metadata_size_bytes();
    desc.socket_buffer_type = static_cast<BufferType>(fb->socket_buffer_type());
    desc.completion_shm_name = fb->completion_shm_name() ? fb->completion_shm_name()->str() : "";
    desc.completion_shm_size = fb->completion_shm_size();
    desc.completion_issued_offset = fb->completion_issued_offset();
    desc.completion_completed_offset = fb->completion_completed_offset();
    desc.completion_completed_stride = fb->completion_completed_stride();

    const auto* fb_entries = fb->per_coord_entries();
    TT_FATAL(fb_entries != nullptr, "Service descriptor missing per_coord_entries");
    desc.per_coord_entries.reserve(fb_entries->size());
    for (const auto* fb_entry : *fb_entries) {
        const auto* fb_coord = fb_entry->coord();
        TT_FATAL(fb_coord != nullptr, "PerCoordEntry missing coord");
        MeshCoordinate coord(ttsl::Span<const uint32_t>{fb_coord->data(), fb_coord->size()});

        const auto* fb_socket = fb_entry->socket_descriptor();
        TT_FATAL(fb_socket != nullptr, "PerCoordEntry missing socket_descriptor");
        desc.per_coord_entries.emplace_back(coord, from_flatbuffer(*fb_socket));
    }

    return desc;
}

}  // namespace tt::tt_metal::distributed
