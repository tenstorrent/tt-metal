// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/llk_metadata.h"
#include "api/tensor/tensor_accessor_args.h"

namespace tensor_accessor {

// TensorBindingToken:
//
// == What is it? ==
// This is a codegen-emitted handle for a Metal 2.0 kernel's tensor binding.
// The user never interacts with this type directly; they use an opaque token (defined in the
// auto-generated kernel_bindings_generated.h) to construct an accessor from it.
// (Either a TensorAccessor or a LocalTensorAccessor, depending on the kernel's needs.)
//
// The user's kernel code looks like:
//  auto a = TensorAccessor(tensor::my_host_declared_accessor_name);         // DM kernels only
//  auto b = LocalTensorAccessor<T>(tensor::my_host_declared_accessor_name); // DM or compute kernels
//
// No more fussing around with TensorAccessorArgs!
// All of the boilerplate, nasty args offset logic, and raw base pointer are now fully hidden
// from the kernel author.
//
// == How does it work? ==
// For each kernel tensor binding, headergen emits the following into kernel_bindings_generated.h:
//   - A type alias:  using my_TA_name_t = TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>;
//   - A token value: constexpr my_TA_name_t my_TA_name{{.format = ..., .face_r_dim = ...}};
//
// This indirection gives us ultimate future-proofing flexibility over what actually goes into the
// TensorBindingToken. We can change TensorBindingToken at any time, or add a wrapper-type indirection,
// all without disturbing any existing Metal 2.0 kernel code. (Probably overkill, but cheap insurance.)
//
// == Current limitations ==
// The Metal 2.0 binding flow currently supports only a subset of the CRTA-dynamic DSpec metadata that
// TensorAccessorArgs nominally supports.
//
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
struct TensorBindingToken {
    using args_t = TensorAccessorArgs<CTA_OFFSET>;
    static constexpr args_t args{};
    static constexpr uint32_t addr_crta_offset = ADDR_CRTA_OFFSET;  // in bytes

    constexpr TensorBindingToken(LLKMetadata llk) noexcept : llk_metadata_(llk) {}

private:
    LLKMetadata llk_metadata_;
};

// NullTensorBindingToken: the "this name is not bound" result of a binding lookup.
//
// tensor::get_token_if_present<"name">() returns a pointer to the named TensorBindingToken when
// the host bound that name, and a null NullTensorBindingToken pointer when it did not. That lets a
// kernel written against an optional binding compile either way:
//
//   if (const auto* token = tensor::get_token_if_present<"maybe">()) {
//       TensorAccessor accessor(*token);   // instantiated, but only reached when "maybe" is bound
//       ...
//   }
//
// Both branches are type-checked even though the guard is a compile-time constant, so the absent
// case still has to name a constructible accessor. It cannot reuse TensorBindingToken with dummy
// offsets: that constructor always reads a compile-time arg at CTA_OFFSET, which fails outright in
// a kernel with no (or differently laid out) tensor CTAs. This distinct type selects overloads that
// read nothing; see NullDSpec in tensor_accessor.h.
//
// NullTensorBindingToken is not meant to be constructed, only used as a type for the null pointer.
struct NullTensorBindingToken {
    NullTensorBindingToken() = delete;
    NullTensorBindingToken(const NullTensorBindingToken&) = delete;
    NullTensorBindingToken(NullTensorBindingToken&&) = delete;
    NullTensorBindingToken& operator=(const NullTensorBindingToken&) = delete;
    NullTensorBindingToken& operator=(NullTensorBindingToken&&) = delete;
};

}  // namespace tensor_accessor
