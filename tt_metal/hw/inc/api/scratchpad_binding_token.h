// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/token_with_llk_metadata.h"

// Forward declaration only: the token grants Scratchpad<T> access to its members, but nothing
// in this header needs the definition. Defined in api/scratchpad.h.
template <typename T>
class Scratchpad;

// Opaque handle for a Program-scope scratchpad binding (declared in kernel_bindings_generated.h).
// The user will never directly interact with this type.
//
// The user's host code declares an accessor_name when binding a scratchpad to a kernel.
// The user then uses that accessor_name to construct a Scratchpad in the kernel code.
//
// Usage example:
//   // (Host code declares "my_scratchpad_name" as the scratchpad accessor name for this kernel.)
//   // In the kernel code:
//   Scratchpad<int32_t> my_pad(scratch::my_scratchpad_name);
//
// Here my_scratchpad_name is a constexpr ScratchpadBindingToken, auto-included in
// kernel_bindings_generated.h.
//
// This header holds the token and its metadata getters, including device assertion support,
// so the generated bindings header does not have to pull in the Scratchpad implementation.
// See api/scratchpad.h for the Scratchpad class this token constructs.
//

class ScratchpadBindingToken : public binding_details::TokenWithLLKMetadata {
public:
    // Construct a Scratchpad from the offset and size supplied by the host.
    //
    // This Scratchpad does not contain any llk metadata.
    // Attempt to extract an LLKOperand from this token using LLKOperandFrom will cause a compile-time error.
    explicit constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes) {}

    // Binding token constructor used when the host supplies LLK metadata.
    // See "Entry format metadata" in ScratchpadSpec.
    constexpr ScratchpadBindingToken(
        uint32_t crta_offset, uint32_t size_in_bytes, binding_details::LLKMetadata llk) noexcept :
        TokenWithLLKMetadata(llk), crta_offset_(crta_offset), size_in_bytes_(size_in_bytes) {}

private:
    template <typename T>
    friend class Scratchpad;

    uint32_t crta_offset_;    // word index of the base-address slot in the CRTA buffer
    uint32_t size_in_bytes_;  // static per-node size
};

/**
 * Extract DataFormat metadata associated with a `ScratchpadBindingToken`.
 * This metadata is configured on the host.
 *
 * See: `ScratchpadSpec::data_format_metadata` in the Metal host API for configuration.
 *
 * pre-condition: `ScratchpadSpec::data_format_metadata` associated with this scratchpad has been set to a
 * non-empty value on the host. Missing metadata fails constant evaluation regardless of device assertion settings.
 */
constexpr DataFormat get_dataformat(const ScratchpadBindingToken& token) {
    return get_dataformat(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

/**
 * Extract the face height (in elements) associated with a `ScratchpadBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_face_r_dim(const ScratchpadBindingToken& token) {
    return get_face_r_dim(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

/**
 * Extract the face width (in elements) associated with a `ScratchpadBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_face_c_dim(const ScratchpadBindingToken& token) {
    return get_face_c_dim(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

/**
 * Extract the number of faces along the tile's row dimension associated with a `ScratchpadBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_num_faces_r_dim(const ScratchpadBindingToken& token) {
    return get_num_faces_r_dim(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

/**
 * Extract the number of faces along the tile's column dimension associated with a `ScratchpadBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_num_faces_c_dim(const ScratchpadBindingToken& token) {
    return get_num_faces_c_dim(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

// Returns the tile storage size in bytes.
// precondition: The token has format metadata and a supported tile storage size.
constexpr uint32_t get_tile_size(const ScratchpadBindingToken& token) {
    return get_tile_size(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

// Returns the tile height in elements.
constexpr uint32_t get_tile_r_dim(const ScratchpadBindingToken& token) {
    return get_tile_r_dim(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

// Returns the tile width in elements.
constexpr uint32_t get_tile_c_dim(const ScratchpadBindingToken& token) {
    return get_tile_c_dim(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

// Returns the number of elements in a tile.
constexpr uint32_t get_tile_hw(const ScratchpadBindingToken& token) {
    return get_tile_hw(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

// Returns the total number of faces in a tile.
constexpr uint32_t get_tile_num_faces(const ScratchpadBindingToken& token) {
    return get_tile_num_faces(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

// Returns whether the tile height is less than 32 elements.
constexpr bool get_partial_face(const ScratchpadBindingToken& token) {
    return get_partial_face(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}

// Returns whether the tile width is less than 32 elements.
constexpr bool get_narrow_tile(const ScratchpadBindingToken& token) {
    return get_narrow_tile(static_cast<const binding_details::TokenWithLLKMetadata&>(token));
}
