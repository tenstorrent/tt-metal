// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/llk_metadata.h"

namespace binding_details {

class TokenWithLLKMetadata;

/**
 * Returns the host-supplied resource metadata of a binding token.
 */
constexpr const LLKMetadata& llk_metadata(const TokenWithLLKMetadata& token);

// Base of the binding tokens that carry host-supplied resource metadata.
class TokenWithLLKMetadata {
protected:
    constexpr TokenWithLLKMetadata() noexcept = default;
    explicit constexpr TokenWithLLKMetadata(LLKMetadata metadata) noexcept : llk_metadata_(metadata) {}

private:
    friend constexpr const LLKMetadata& llk_metadata(const TokenWithLLKMetadata& token);

    LLKMetadata llk_metadata_{};
};

constexpr const LLKMetadata& llk_metadata(const TokenWithLLKMetadata& token) { return token.llk_metadata_; }

}  // namespace binding_details

// Metadata getters for DFB, scratchpad and tensor binding tokens.
// They are global so that qualified calls such as ::get_tile_size(dfb::x) select them over the CB-ID getters.

/**
 * Extract DataFormat metadata associated with a resource binding.
 * This metadata is configured on the host through DataflowBufferSpec::data_format_metadata,
 * ScratchpadSpec::data_format_metadata, or the bound MeshTensor's format.
 *
 * precondition: The host has supplied a data format for this resource.
 * Missing metadata fails constant evaluation regardless of device assertion settings.
 */
constexpr DataFormat get_dataformat(const binding_details::TokenWithLLKMetadata& token) {
    const binding_details::LLKMetadata& metadata = binding_details::llk_metadata(token);
    binding_details::assert_dataformat_set(metadata);
    return static_cast<DataFormat>(metadata.format);
}

/**
 * Extract the face height in elements associated with a resource binding.
 * Returns 16 if tile metadata is omitted on the host, representing a default face (16x16 elements).
 * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
 */
constexpr uint8_t get_face_r_dim(const binding_details::TokenWithLLKMetadata& token) {
    return binding_details::llk_metadata(token).face_r_dim;
}

/**
 * Extract the face width in elements associated with a resource binding.
 * Returns 16 if tile metadata is omitted on the host, representing a default face (16x16 elements).
 * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
 */
constexpr uint8_t get_face_c_dim(const binding_details::TokenWithLLKMetadata& token) {
    return binding_details::llk_metadata(token).face_c_dim;
}

/**
 * Extract the number of faces along the tile's row dimension associated with a resource binding.
 * Returns 2 if tile metadata is omitted on the host, representing a default tile (2x2 faces).
 * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
 */
constexpr uint8_t get_num_faces_r_dim(const binding_details::TokenWithLLKMetadata& token) {
    return binding_details::llk_metadata(token).num_faces_r_dim;
}

/**
 * Extract the number of faces along the tile's column dimension associated with a resource binding.
 * Returns 2 if tile metadata is omitted on the host, representing a default tile (2x2 faces).
 * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
 */
constexpr uint8_t get_num_faces_c_dim(const binding_details::TokenWithLLKMetadata& token) {
    return binding_details::llk_metadata(token).num_faces_c_dim;
}

/**
 * Returns the tile storage size in bytes.
 *
 * precondition: The token has format metadata and a supported tile storage size.
 */
constexpr uint32_t get_tile_size(const binding_details::TokenWithLLKMetadata& token) {
    const binding_details::LLKMetadata& metadata = binding_details::llk_metadata(token);
    binding_details::assert_dataformat_set(metadata);
    if (metadata.tile_size_bytes == 0) {
        binding_details::tile_size_unsupported_for_format();
    }
    return metadata.tile_size_bytes;
}

/**
 * Returns the tile height in elements.
 */
constexpr uint32_t get_tile_r_dim(const binding_details::TokenWithLLKMetadata& token) {
    return get_face_r_dim(token) * get_num_faces_r_dim(token);
}

/**
 * Returns the tile width in elements.
 */
constexpr uint32_t get_tile_c_dim(const binding_details::TokenWithLLKMetadata& token) {
    return get_face_c_dim(token) * get_num_faces_c_dim(token);
}

/**
 * Returns the number of elements in a tile.
 */
constexpr uint32_t get_tile_hw(const binding_details::TokenWithLLKMetadata& token) {
    return get_tile_r_dim(token) * get_tile_c_dim(token);
}

/**
 * Returns the total number of faces in a tile.
 */
constexpr uint32_t get_tile_num_faces(const binding_details::TokenWithLLKMetadata& token) {
    return get_num_faces_r_dim(token) * get_num_faces_c_dim(token);
}

/**
 * Returns 1 if the tile height is less than 32 elements, or 0 otherwise.
 */
constexpr uint32_t get_partial_face(const binding_details::TokenWithLLKMetadata& token) {
    return static_cast<uint32_t>(get_tile_r_dim(token) < 32);
}

/**
 * Returns 1 if the tile width is less than 32 elements, or 0 otherwise.
 */
constexpr uint32_t get_narrow_tile(const binding_details::TokenWithLLKMetadata& token) {
    return static_cast<uint32_t>(get_tile_c_dim(token) < 32);
}
