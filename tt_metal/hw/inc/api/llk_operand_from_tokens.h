// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Metal 2 BindingToken → ckernel::experimental::LLKOperand<Format, Shape>.

#include <cstdint>
#include <type_traits>

#include "api/compute/experimental/2_0/llk_operand.h"
#include "api/dataflow/dfb_binding_token.h"
#include "api/debug/assert.h"
#include "api/scratchpad_binding_token.h"
#include "api/tensor/tensor_binding_token.h"

namespace binding_details {

/**
 * Only 3 types of BindingTokens have llk metadata: DFB, Scratchpad, and Tensor.
 */
template <typename T>
inline constexpr bool binding_token_with_llk_metadata = false;
template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline constexpr bool binding_token_with_llk_metadata<DFBBindingToken<Pap, Cap>> = true;
template <>
inline constexpr bool binding_token_with_llk_metadata<ScratchpadBindingToken> = true;
template <uint32_t Cta, uint32_t Addr>
inline constexpr bool binding_token_with_llk_metadata<tensor_accessor::TensorBindingToken<Cta, Addr>> = true;

// Helper struct to instantiate the LLKOperand type from a BindingToken.
template <const auto& Token>
struct LLKOperandExtractor {
    using TokenT = std::remove_cv_t<std::remove_reference_t<decltype(Token)>>;
    static_assert(
        binding_token_with_llk_metadata<TokenT>,
        "LLKOperandFrom requires a BindingToken with llk metadata (DFB / Scratchpad / Tensor)");
    static_assert(
        Token.llk_metadata_.format != binding_details::LLKMetadata::kNoFormat,
        "This token has no data format. Did you forget to set data_format_metadata on ScratchpadSpec/ "
        "DataflowBufferSpec?");

    static constexpr DataFormat format = static_cast<DataFormat>(Token.llk_metadata_.format);
    static constexpr ckernel::TensorShape shape{
        Token.llk_metadata_.face_r_dim,
        Token.llk_metadata_.face_c_dim,
        Token.llk_metadata_.num_faces_r_dim,
        Token.llk_metadata_.num_faces_c_dim};

    using OperandT = ckernel::experimental::LLKOperand<format, shape>;
};

}  // namespace binding_details

/**
 * Extract the LLKOperand type from a BindingToken.
 *
 * LLKOperand is associated with llk metadata needed to operate on compute kernels.
 * These metadata are specified/ inferred from host.
 *
 * pre-condition: Resource represented by the tokens has llk metadata configured on the host side.
 */
template <const auto& Token>
using LLKOperandFrom = typename binding_details::LLKOperandExtractor<Token>::OperandT;

/**
 * Extract DataFormat metadata associated with a `ScratchpadBindingToken`.
 * This metadata is configured on the host.
 *
 * See: `ScratchpadSpec::data_format_metadata` in the Metal host API for configuration.
 *
 * pre-condition: `ScratchpadSpec::data_format_metadata` associated with this scratchpad has been set to a
 * non-empty value on the host.
 */
constexpr DataFormat get_dataformat(const ScratchpadBindingToken& token) {
    ASSERT(token.llk_metadata_.format != binding_details::LLKMetadata::kNoFormat);
    return static_cast<DataFormat>(token.llk_metadata_.format);
}

/**
 * Extract DataFormat metadata associated with a `DFBBindingToken`.
 * This metadata is configured on the host.
 *
 * See: `DataflowBufferSpec::data_format_metadata` in the Metal host API for configuration.
 *
 * pre-condition: `DataflowBufferSpec::data_format_metadata` associated with this DFB has been set to a
 * non-empty value on the host.
 */
constexpr DataFormat get_dataformat(const DFBBindingToken& token) {
    ASSERT(token.llk_metadata_.format != binding_details::LLKMetadata::kNoFormat);
    return static_cast<DataFormat>(token.llk_metadata_.format);
}

/**
 * Extract DataFormat metadata associated with a `TensorBindingToken`.
 * This metadata describes the data format of the `MeshTensor` that this token represents.
 */
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr DataFormat get_dataformat(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return static_cast<DataFormat>(token.llk_metadata_.format);
}

/**
 * Extract the face height (in elements) associated with a `ScratchpadBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_face_r_dim(const ScratchpadBindingToken& token) { return token.llk_metadata_.face_r_dim; }

/**
 * Extract the face height (in elements) associated with a `DFBBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_face_r_dim(const DFBBindingToken& token) { return token.llk_metadata_.face_r_dim; }

/**
 * Extract the face height (in elements) associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_face_r_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.face_r_dim;
}

/**
 * Extract the face width (in elements) associated with a `ScratchpadBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_face_c_dim(const ScratchpadBindingToken& token) { return token.llk_metadata_.face_c_dim; }

/**
 * Extract the face width (in elements) associated with a `DFBBindingToken`.
 * returns 16 if tile metadata is omitted on host, representing a default face (16x16 elements).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_face_c_dim(const DFBBindingToken& token) { return token.llk_metadata_.face_c_dim; }

/**
 * Extract the face width (in elements) associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_face_c_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.face_c_dim;
}

/**
 * Extract the number of faces along the tile's row dimension associated with a `ScratchpadBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_num_faces_r_dim(const ScratchpadBindingToken& token) {
    return token.llk_metadata_.num_faces_r_dim;
}

/**
 * Extract the number of faces along the tile's row dimension associated with a `DFBBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_num_faces_r_dim(const DFBBindingToken& token) { return token.llk_metadata_.num_faces_r_dim; }

/**
 * Extract the number of faces along the tile's row dimension associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_num_faces_r_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.num_faces_r_dim;
}

/**
 * Extract the number of faces along the tile's column dimension associated with a `ScratchpadBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `ScratchpadSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_num_faces_c_dim(const ScratchpadBindingToken& token) {
    return token.llk_metadata_.num_faces_c_dim;
}

/**
 * Extract the number of faces along the tile's column dimension associated with a `DFBBindingToken`.
 * returns 2 if tile metadata is omitted on host, representing a default tile (2x2 faces).
 *
 * See: `DataflowBufferSpec::tile_format_metadata` in the Metal host API for configuration.
 */
constexpr uint8_t get_num_faces_c_dim(const DFBBindingToken& token) { return token.llk_metadata_.num_faces_c_dim; }

/**
 * Extract the number of faces along the tile's column dimension associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_num_faces_c_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.num_faces_c_dim;
}
