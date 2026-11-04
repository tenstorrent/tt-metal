// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Metal 2 binding-token geometry queries and LLKOperand type extraction.

#include <cstdint>
#include <type_traits>

#include "api/compute/experimental/2_0/llk_operand.h"
#include "api/dataflow/dfb_binding_token.h"
#include "api/scratchpad_binding_token.h"
#include "api/tensor/tensor_binding_token.h"

namespace binding_details {

// Returns the LLK tile and face geometry of a resource binding.
constexpr ckernel::TensorShape tensor_shape_from_metadata(const TokenWithLLKMetadata& token) {
    return ckernel::TensorShape{
        get_face_r_dim(token), get_face_c_dim(token), get_num_faces_r_dim(token), get_num_faces_c_dim(token)};
}

}  // namespace binding_details

// Returns the DFB binding's LLK tile and face geometry.
template <dfb::AccessPattern P, dfb::AccessPattern C>
constexpr ckernel::TensorShape get_tensor_shape(const DFBBindingToken<P, C>& token) {
    return binding_details::tensor_shape_from_metadata(token);
}

// Returns the scratchpad binding's LLK tile and face geometry.
constexpr ckernel::TensorShape get_tensor_shape(const ScratchpadBindingToken& token) {
    return binding_details::tensor_shape_from_metadata(token);
}

namespace tensor_accessor {

// Returns the tensor binding's LLK tile and face geometry.
template <uint32_t Cta, uint32_t Addr>
constexpr ckernel::TensorShape get_tensor_shape(const TensorBindingToken<Cta, Addr>& token) {
    return binding_details::tensor_shape_from_metadata(token);
}

// Returns the per-node tensor binding's LLK tile and face geometry.
template <uint32_t Cta, uint32_t Addr>
constexpr ckernel::TensorShape get_tensor_shape(const PerNodeTensorBindingToken<Cta, Addr>& token) {
    return binding_details::tensor_shape_from_metadata(token);
}

}  // namespace tensor_accessor

namespace binding_details {

// Identifies binding tokens that carry resource metadata.
template <typename T>
inline constexpr bool binding_token_with_llk_metadata = std::is_base_of_v<TokenWithLLKMetadata, T>;

// Helper struct to instantiate the LLKOperand type from a BindingToken.
template <const auto& Token>
struct LLKOperandExtractor {
    using TokenT = std::remove_cv_t<std::remove_reference_t<decltype(Token)>>;
    static_assert(
        binding_token_with_llk_metadata<TokenT>,
        "LLKOperandFrom requires a BindingToken with llk metadata (DFB / Scratchpad / Tensor)");

    static constexpr DataFormat format = get_dataformat(Token);
    static constexpr ckernel::TensorShape shape = get_tensor_shape(Token);

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
