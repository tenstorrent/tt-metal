// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The LLK operand conversion below is compute-only: llk_mem_descriptor.h reaches into the ckernel/LLK
// headers, which are not on the include path for data-movement builds.
#ifdef COMPILE_FOR_TRISC
#include "api/compute/experimental/2_0/llk_mem_descriptor.h"
#endif

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

#ifdef COMPILE_FOR_TRISC
    // Converts to a LLKMemDescriptor, which contains LLK relevant information about the tensor.
    // These metadata are automatically derived from the host side through the TensorSpec associated with the tensor
    // binding.
    friend constexpr ckernel::experimental::LLKMemDescriptor to_llk_mem_descriptor(TensorBindingToken token) {
        static_assert(!args_t::is_dram, "to_llk_mem_descriptor: DRAM TensorBindingToken has no node-local L1 region");
        return {
            token.llk_metadata_.format,
            ckernel::TensorShape{
                token.llk_metadata_.face_r_dim,
                token.llk_metadata_.face_c_dim,
                token.llk_metadata_.num_faces_r_dim,
                token.llk_metadata_.num_faces_c_dim}};
    }
#endif

private:
    LLKMetadata llk_metadata_;
};

}  // namespace tensor_accessor
