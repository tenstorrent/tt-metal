// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

// Host-side resource description used to emit binding_details::LLKMetadata onto a binding token.
// Face layout is read from the tile.
struct LLKMetadata {
    DataFormat format;
    Tile tile;
    // Zero until resolved for a target, or when the format has no supported Tile storage encoding.
    uint32_t tile_size_bytes = 0;
};

// Returns metadata with its tile storage size resolved for the supplied target alignment.
// precondition: l1_alignment is a nonzero alignment in bytes.
LLKMetadata resolve_llk_metadata_tile_size(LLKMetadata metadata, uint32_t l1_alignment);

// Returns the C++ initializer for the device-side binding_details::LLKMetadata
// (internal/llk_metadata.h) that a generated binding token is constructed with, e.g.
//   ::binding_details::LLKMetadata{.format = 5u, .face_r_dim = 16u, .face_c_dim = 16u,
//                                  .num_faces_r_dim = 2u, .num_faces_c_dim = 2u, .tile_size_bytes = 2048u}
// `format` is the HW DataFormat code; the face fields are derived from `metadata.tile`.
// The tile byte size is resolved with the program target's L1 alignment before serialization.
// The JIT's kernel bindings header and the emulator's both embed this string verbatim.
std::string serialize_llk_metadata(const LLKMetadata& metadata);

}  // namespace tt::tt_metal
