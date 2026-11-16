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
//   - A token value: constexpr my_TA_name_t my_TA_name{LLKMetadata{.format = ..., .face_r_dim = ...}};
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

    // Construct a TensorBindingToken with LLK metadata.
    // Read these metadata with get_dataformat(token) / get_face_r_dim(token) / ...,
    // or construct an operand with LLKOperandFrom<token>.
    explicit constexpr TensorBindingToken(binding_details::LLKMetadata llk) noexcept : llk_metadata_(llk) {}

private:
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr DataFormat get_dataformat(const TensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_face_r_dim(const TensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_face_c_dim(const TensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_num_faces_r_dim(const TensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_num_faces_c_dim(const TensorBindingToken<Cta, Addr>& token);

    binding_details::LLKMetadata llk_metadata_;
};

/**
 * Extract DataFormat metadata associated with a `TensorBindingToken`.
 * This metadata describes the data format of the `MeshTensor` that this token represents.
 * Missing metadata fails constant evaluation regardless of device assertion settings.
 */
template <uint32_t Cta, uint32_t Addr>
constexpr DataFormat get_dataformat(const TensorBindingToken<Cta, Addr>& token) {
    if (token.llk_metadata_.format == binding_details::LLKMetadata::kNoFormat) {
        binding_details::data_format_metadata_not_set();
    }
    return static_cast<DataFormat>(token.llk_metadata_.format);
}

/**
 * Extract the face height (in elements) associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_face_r_dim(const TensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.face_r_dim;
}

/**
 * Extract the face width (in elements) associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_face_c_dim(const TensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.face_c_dim;
}

/**
 * Extract the number of faces along the tile's row dimension associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_num_faces_r_dim(const TensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.num_faces_r_dim;
}

/**
 * Extract the number of faces along the tile's column dimension associated with a `TensorBindingToken`.
 * This metadata describes the tile layout of the `MeshTensor` that this token represents.
 */
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_num_faces_c_dim(const TensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.num_faces_c_dim;
}

// PerNodeTensorBindingToken: the binding token for a per-node tensor.
//
// A per-node allocated tensor has an independent base address on every node, so the address
// cannot ride the kernel's common runtime args the way TensorBindingToken's does. Instead, each node
// receives its own address in its per-node runtime args, at byte offset ADDR_RTA_OFFSET.
//
// Only this node's address is known to the kernel, so a per-node token builds only a LocalTensorAccessor:
//  auto a = LocalTensorAccessor<T>(tensor::my_host_declared_accessor_name); // OK
//  auto b = TensorAccessor(tensor::my_host_declared_accessor_name);         // compile error
template <uint32_t CTA_OFFSET, uint32_t ADDR_RTA_OFFSET>
struct PerNodeTensorBindingToken {
    using args_t = TensorAccessorArgs<CTA_OFFSET>;
    static constexpr args_t args{};
    static constexpr uint32_t addr_rta_offset = ADDR_RTA_OFFSET;  // in bytes

    explicit constexpr PerNodeTensorBindingToken(binding_details::LLKMetadata llk) noexcept : llk_metadata_(llk) {}

private:
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr DataFormat get_dataformat(const PerNodeTensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_face_r_dim(const PerNodeTensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_face_c_dim(const PerNodeTensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_num_faces_r_dim(const PerNodeTensorBindingToken<Cta, Addr>& token);
    template <uint32_t Cta, uint32_t Addr>
    friend constexpr uint8_t get_num_faces_c_dim(const PerNodeTensorBindingToken<Cta, Addr>& token);

    binding_details::LLKMetadata llk_metadata_;
};

// LLK metadata getters for PerNodeTensorBindingToken, as for TensorBindingToken above.
template <uint32_t Cta, uint32_t Addr>
constexpr DataFormat get_dataformat(const PerNodeTensorBindingToken<Cta, Addr>& token) {
    if (token.llk_metadata_.format == binding_details::LLKMetadata::kNoFormat) {
        binding_details::data_format_metadata_not_set();
    }
    return static_cast<DataFormat>(token.llk_metadata_.format);
}
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_face_r_dim(const PerNodeTensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.face_r_dim;
}
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_face_c_dim(const PerNodeTensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.face_c_dim;
}
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_num_faces_r_dim(const PerNodeTensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.num_faces_r_dim;
}
template <uint32_t Cta, uint32_t Addr>
constexpr uint8_t get_num_faces_c_dim(const PerNodeTensorBindingToken<Cta, Addr>& token) {
    return token.llk_metadata_.num_faces_c_dim;
}

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
