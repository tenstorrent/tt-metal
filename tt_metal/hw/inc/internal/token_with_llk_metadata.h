// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/llk_metadata.h"

namespace binding_details {

// Stores host-supplied resource metadata and provides the common binding-token getters.
class TokenWithLLKMetadata {
public:
    /**
     * Extract DataFormat metadata associated with a resource binding.
     * This metadata is configured on the host through DataflowBufferSpec::data_format_metadata,
     * ScratchpadSpec::data_format_metadata, or the bound MeshTensor's format.
     *
     * precondition: The host has supplied a data format for this resource.
     * Missing metadata fails constant evaluation regardless of device assertion settings.
     */
    friend constexpr DataFormat get_dataformat(const TokenWithLLKMetadata& token) {
        assert_dataformat_set(token.llk_metadata_);
        return token.dataformat_or_invalid();
    }

    /**
     * Extract the face height in elements associated with a resource binding.
     * Returns 16 if tile metadata is omitted on the host, representing a default face (16x16 elements).
     * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
     */
    friend constexpr uint8_t get_face_r_dim(const TokenWithLLKMetadata& token) {
        return token.llk_metadata_.face_r_dim;
    }

    /**
     * Extract the face width in elements associated with a resource binding.
     * Returns 16 if tile metadata is omitted on the host, representing a default face (16x16 elements).
     * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
     */
    friend constexpr uint8_t get_face_c_dim(const TokenWithLLKMetadata& token) {
        return token.llk_metadata_.face_c_dim;
    }

    /**
     * Extract the number of faces along the tile's row dimension associated with a resource binding.
     * Returns 2 if tile metadata is omitted on the host, representing a default tile (2x2 faces).
     * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
     */
    friend constexpr uint8_t get_num_faces_r_dim(const TokenWithLLKMetadata& token) {
        return token.llk_metadata_.num_faces_r_dim;
    }

    /**
     * Extract the number of faces along the tile's column dimension associated with a resource binding.
     * Returns 2 if tile metadata is omitted on the host, representing a default tile (2x2 faces).
     * See DataflowBufferSpec::tile_format_metadata, ScratchpadSpec::tile_format_metadata, or the bound tensor's tile.
     */
    friend constexpr uint8_t get_num_faces_c_dim(const TokenWithLLKMetadata& token) {
        return token.llk_metadata_.num_faces_c_dim;
    }

    // Returns the tile storage size in bytes.
    // precondition: The token has format metadata and a supported tile storage size.
    friend constexpr uint32_t get_tile_size(const TokenWithLLKMetadata& token) {
        assert_dataformat_set(token.llk_metadata_);
        if (token.llk_metadata_.tile_size_bytes == 0) {
            tile_size_metadata_not_set();
        }
        return token.llk_metadata_.tile_size_bytes;
    }

    // Returns the tile height in elements.
    friend constexpr uint32_t get_tile_r_dim(const TokenWithLLKMetadata& token) {
        return get_face_r_dim(token) * get_num_faces_r_dim(token);
    }

    // Returns the tile width in elements.
    friend constexpr uint32_t get_tile_c_dim(const TokenWithLLKMetadata& token) {
        return get_face_c_dim(token) * get_num_faces_c_dim(token);
    }

    // Returns the number of elements in a tile.
    friend constexpr uint32_t get_tile_hw(const TokenWithLLKMetadata& token) {
        return get_tile_r_dim(token) * get_tile_c_dim(token);
    }

    // Returns the total number of faces in a tile.
    friend constexpr uint32_t get_tile_num_faces(const TokenWithLLKMetadata& token) {
        return get_num_faces_r_dim(token) * get_num_faces_c_dim(token);
    }

    // Returns whether the tile height is less than 32 elements.
    friend constexpr bool get_partial_face(const TokenWithLLKMetadata& token) { return get_tile_r_dim(token) < 32; }

    // Returns whether the tile width is less than 32 elements.
    friend constexpr bool get_narrow_tile(const TokenWithLLKMetadata& token) { return get_tile_c_dim(token) < 32; }

protected:
    constexpr TokenWithLLKMetadata() noexcept = default;
    explicit constexpr TokenWithLLKMetadata(LLKMetadata metadata) noexcept : llk_metadata_(metadata) {}

    // Returns the resource format, or Invalid when the host omitted format metadata.
    constexpr DataFormat dataformat_or_invalid() const { return static_cast<DataFormat>(llk_metadata_.format); }

private:
    LLKMetadata llk_metadata_{};
};

}  // namespace binding_details
