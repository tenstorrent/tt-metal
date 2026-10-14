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

// Watcher/lightweight asserts fire when the host left the format unset. LLKOperandFrom still static_asserts.
constexpr DataFormat get_dataformat(const ScratchpadBindingToken& token) {
    ASSERT(token.llk_metadata_.format != binding_details::LLKMetadata::kNoFormat);
    return static_cast<DataFormat>(token.llk_metadata_.format);
}
constexpr DataFormat get_dataformat(const DFBBindingToken& token) {
    ASSERT(token.llk_metadata_.format != binding_details::LLKMetadata::kNoFormat);
    return static_cast<DataFormat>(token.llk_metadata_.format);
}
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr DataFormat get_dataformat(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    ASSERT(token.llk_metadata_.format != binding_details::LLKMetadata::kNoFormat);
    return static_cast<DataFormat>(token.llk_metadata_.format);
}

constexpr uint8_t get_face_r_dim(const ScratchpadBindingToken& token) { return token.llk_metadata_.face_r_dim; }
constexpr uint8_t get_face_r_dim(const DFBBindingToken& token) { return token.llk_metadata_.face_r_dim; }
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_face_r_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.face_r_dim;
}

constexpr uint8_t get_face_c_dim(const ScratchpadBindingToken& token) { return token.llk_metadata_.face_c_dim; }
constexpr uint8_t get_face_c_dim(const DFBBindingToken& token) { return token.llk_metadata_.face_c_dim; }
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_face_c_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.face_c_dim;
}

constexpr uint8_t get_num_faces_r_dim(const ScratchpadBindingToken& token) {
    return token.llk_metadata_.num_faces_r_dim;
}
constexpr uint8_t get_num_faces_r_dim(const DFBBindingToken& token) { return token.llk_metadata_.num_faces_r_dim; }
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_num_faces_r_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.num_faces_r_dim;
}

constexpr uint8_t get_num_faces_c_dim(const ScratchpadBindingToken& token) {
    return token.llk_metadata_.num_faces_c_dim;
}
constexpr uint8_t get_num_faces_c_dim(const DFBBindingToken& token) { return token.llk_metadata_.num_faces_c_dim; }
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
constexpr uint8_t get_num_faces_c_dim(const tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>& token) {
    return token.llk_metadata_.num_faces_c_dim;
}
