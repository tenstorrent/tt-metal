// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Metal 2 BindingToken → ckernel::experimental::LLKOperand<Format, Shape>.
// Device: C++17 + C++20-style NTTP. No concepts / requires.
//
// Blackhole-only (pulls llk_operand.h). Compute kernels that need the translator include this
// header; data-movement kernels must not.

#include <type_traits>

#include "api/compute/experimental/2_0/llk_operand.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/tensor/tensor_binding_token.h"

namespace binding_details {

template <typename T>
inline constexpr bool binding_token_with_llk_metadata = false;
template <>
inline constexpr bool binding_token_with_llk_metadata<DFBBindingToken> = true;
template <>
inline constexpr bool binding_token_with_llk_metadata<ScratchpadBindingToken> = true;
template <uint32_t Cta, uint32_t Addr>
inline constexpr bool binding_token_with_llk_metadata<tensor_accessor::TensorBindingToken<Cta, Addr>> = true;

template <const auto& Token>
struct LLKOperandExtractor {
    using TokenT = std::remove_cv_t<std::remove_reference_t<decltype(Token)>>;
    static_assert(
        binding_token_with_llk_metadata<TokenT>,
        "LLKOperandFrom requires a BindingToken with llk metadata (DFB / Scratchpad / Tensor)");

    // Named constexprs first — the device toolchain rejects feeding a braced
    // TensorShape{...} temporary built from Token.llk_metadata_ directly into
    // LLKOperand's NTTP list. DataFormat is global (tensix_types); TensorShape is ckernel::.
    static constexpr DataFormat format = static_cast<DataFormat>(Token.llk_metadata_.format);
    static constexpr ckernel::TensorShape shape{
        Token.llk_metadata_.face_r_dim,
        Token.llk_metadata_.face_c_dim,
        Token.llk_metadata_.num_faces_r_dim,
        Token.llk_metadata_.num_faces_c_dim};

    using type = ckernel::experimental::LLKOperand<format, shape>;
};

}  // namespace binding_details

template <const auto& Token>
using LLKOperandFrom = typename binding_details::LLKOperandExtractor<Token>::type;
