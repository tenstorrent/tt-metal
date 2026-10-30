// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dfb_access.h"
#include "internal/llk_metadata.h"

// Opaque handle for a DataflowBuffer binding (declared in kernel_bindings_generated.h).
// The user will never directly interact with this type.
//
// The user's host code declares an accessor_name when binding a DFB endpoint to a kernel.
// The user then uses that accessor_name to construct a DataflowBuffer in the kernel code.
//
// Usage example:
//   // (Host code declares "my_dfb_name" as the DFB accessor name for this kernel.)
//   // In the kernel code:
//   DataflowBuffer my_dfb(dfb::my_dfb_name);
//
// Here my_dfb_name is a constexpr DFBBindingToken, auto-included in kernel_bindings_generated.h.
//
// This header holds the tokens and their metadata getters, including device assertion support,
// so the generated bindings header does not have to pull in the DataflowBuffer implementation. See
// api/dataflow/dataflow_buffer.h for the DataflowBuffer class these tokens construct.
//

// The token's template arguments are the producer and consumer access patterns, so the
// DataflowBuffer built from it is specialized at compile time while the kernel still writes
// `DataflowBuffer dfb(dfb::name)`. A plain `DFBBindingToken t{id}` is <UNKNOWN, UNKNOWN>; a
// function parameter must spell `DFBBindingToken<P, C>` (or take the id).
template <dfb::AccessPattern Pap = dfb::AccessPattern::UNKNOWN, dfb::AccessPattern Cap = dfb::AccessPattern::UNKNOWN>
struct DFBBindingToken {
    explicit constexpr DFBBindingToken(uint16_t id) noexcept : id_(id) {}

    // Binding token constructor when host supplies LLK metadata.
    // See "Entry format metadata" in DataflowBufferSpec.
    // Read these metadata with get_dataformat(token) / get_face_r_dim(token) / ...,
    // or construct an operand with LLKOperandFrom<dfb::token_name>.
    //
    // Kernels that do not query the data format or construct LLK operands can use the id-only
    // constructor; tile geometry getters then return the default dimensions.
    constexpr DFBBindingToken(uint16_t id, binding_details::LLKMetadata llk) noexcept : id_(id), llk_metadata_(llk) {}

    // DFBBindingToken is backed by a compile-time ID (an implicit CTA).

    // Implicit conversion to uint32_t:
    // This lets a Metal 2.0 kernel pass a DFBBindingToken directly to Gen1 (WH/BH) LLK
    // compute APIs that expect a raw CB id.
    // This conversion is constexpr; it's intended for Gen1 use only.
    constexpr operator uint32_t() const noexcept { return id_; }

private:
    template <dfb::AccessPattern P, dfb::AccessPattern C>
    friend constexpr DataFormat get_dataformat(const DFBBindingToken<P, C>& token);
    template <dfb::AccessPattern P, dfb::AccessPattern C>
    friend constexpr uint8_t get_face_r_dim(const DFBBindingToken<P, C>& token);
    template <dfb::AccessPattern P, dfb::AccessPattern C>
    friend constexpr uint8_t get_face_c_dim(const DFBBindingToken<P, C>& token);
    template <dfb::AccessPattern P, dfb::AccessPattern C>
    friend constexpr uint8_t get_num_faces_r_dim(const DFBBindingToken<P, C>& token);
    template <dfb::AccessPattern P, dfb::AccessPattern C>
    friend constexpr uint8_t get_num_faces_c_dim(const DFBBindingToken<P, C>& token);

    uint16_t id_;
    binding_details::LLKMetadata llk_metadata_{};
};

/**
 * Extract DataFormat metadata associated with a `DFBBindingToken`.
 * This metadata is configured on the host.
 *
 * See: `DataflowBufferSpec::data_format_metadata` in the Metal host API for configuration.
 *
 * pre-condition: `DataflowBufferSpec::data_format_metadata` associated with this DFB has been set to a
 * non-empty value on the host. Missing metadata fails constant evaluation regardless of device assertion settings.
 */
template <dfb::AccessPattern P, dfb::AccessPattern C>
constexpr DataFormat get_dataformat(const DFBBindingToken<P, C>& token) {
    if (token.llk_metadata_.format == binding_details::LLKMetadata::kNoFormat) {
        binding_details::data_format_metadata_not_set();
    }
    return static_cast<DataFormat>(token.llk_metadata_.format);
}

/**
 * Extract the face height (in elements) associated with a `DFBBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
template <dfb::AccessPattern P, dfb::AccessPattern C>
constexpr uint8_t get_face_r_dim(const DFBBindingToken<P, C>& token) {
    return token.llk_metadata_.face_r_dim;
}

/**
 * Extract the face width (in elements) associated with a `DFBBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
template <dfb::AccessPattern P, dfb::AccessPattern C>
constexpr uint8_t get_face_c_dim(const DFBBindingToken<P, C>& token) {
    return token.llk_metadata_.face_c_dim;
}

/**
 * Extract the number of faces along the tile's row dimension associated with a `DFBBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
template <dfb::AccessPattern P, dfb::AccessPattern C>
constexpr uint8_t get_num_faces_r_dim(const DFBBindingToken<P, C>& token) {
    return token.llk_metadata_.num_faces_r_dim;
}

/**
 * Extract the number of faces along the tile's column dimension associated with a `DFBBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
template <dfb::AccessPattern P, dfb::AccessPattern C>
constexpr uint8_t get_num_faces_c_dim(const DFBBindingToken<P, C>& token) {
    return token.llk_metadata_.num_faces_c_dim;
}

// Compile-time handle for a CrossNode/PrefetcherPipe *relay* local DFB binding.
// Distinct from DFBBindingToken so kernels cannot silently treat a normal DFB as a
// relay (or vice versa) without a cast — no runtime "am I a relay?" check needed.
// Emitted into kernel_bindings_generated.h when the host DFB was created as a relay.
//
// PrefetcherPipe relays additionally carry the prefetcher_pipe_id so the TRISC-side
// DataflowBuffer constructor can O(1)-index that slot in the launch-msg persistent
// region and snap the borrowed local iface to the durable fifo_ptr checkpoint.
// CrossNode relays omit it (NO_PREFETCHER_PIPE): CrossNode state is re-zeroed every
// launch, so the dispatch-written local CB config is already correct.
struct RelayDFBBindingToken {
    static constexpr uint8_t NO_PREFETCHER_PIPE = 0xFF;

    explicit constexpr RelayDFBBindingToken(uint16_t id, uint8_t prefetcher_pipe_id = NO_PREFETCHER_PIPE) noexcept :
        id_(id), prefetcher_pipe_id_(prefetcher_pipe_id) {}

    constexpr operator uint32_t() const noexcept { return id_; }

    constexpr uint8_t prefetcher_pipe_id() const noexcept { return prefetcher_pipe_id_; }

private:
    uint16_t id_;
    uint8_t prefetcher_pipe_id_;
};
