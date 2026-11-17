// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Metal 2 binding-token geometry queries and LLKOperand type extraction.

#include <cstdint>

#include "api/dataflow/dfb_binding_token.h"
#include "api/scratchpad_binding_token.h"
#include "api/tensor/tensor_binding_token.h"
#include "tensor_shape.h"

/**
 * Returns the LLK tile and face geometry of a resource binding.
 */
constexpr ckernel::TensorShape get_tile_shape(const binding_details::TokenWithLLKMetadata& token) {
    return ckernel::TensorShape{
        get_face_r_dim(token), get_face_c_dim(token), get_num_faces_r_dim(token), get_num_faces_c_dim(token)};
}

namespace tensor_accessor {
using ::get_tile_shape;
}  // namespace tensor_accessor

#if defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_TRISC)

#include <type_traits>

#include "api/compute/experimental/2_0/llk_operand.h"

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
    static constexpr ckernel::TensorShape shape = get_tile_shape(Token);

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

#endif  // defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_TRISC)
