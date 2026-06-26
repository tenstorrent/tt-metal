// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace scratchpad {

// ScratchpadBindingToken:
//
// == What is it? ==
// This is a codegen-emitted handle for a Metal 2.0 kernel's scratchpad binding.
// The user never interacts with this type directly; they use an opaque token (defined in the
// auto-generated kernel_bindings_generated.h) to construct a Scratchpad accessor from it.
//
// The user's kernel code looks like:
//   auto s = Scratchpad<uint32_t>(scratch::my_host_declared_accessor_name);
//
// == How does it work? ==
// For each kernel scratchpad binding, headergen emits the following into kernel_bindings_generated.h:
//   - A type alias:  using my_scratch_name_t = ScratchpadBindingToken<SIZE_BYTES, ADDR_CRTA_OFFSET>;
//   - A token value: constexpr my_scratch_name_t my_scratch_name{};
//
// The token carries two pieces, both fixed by the host ProgramSpec:
//   - SIZE_BYTES:        the scratchpad's per-node size, delivered as an implicit compile-time arg (CTA).
//   - ADDR_CRTA_OFFSET:  the byte offset, within the kernel's common-runtime-arg (CRTA) buffer, of the
//                        word holding the scratchpad's allocated L1 base address. The framework fills
//                        that word at program-compile time (the scratchpad is allocated then), and the
//                        Scratchpad ctor reads it via get_common_arg_val.
//
// This indirection mirrors the tensor binding token (api/tensor/tensor_binding_token.h): it lets the
// framework change what goes into the token later without disturbing any existing Metal 2.0 kernel
// code. The token header is intentionally NOC-free so it compiles on both compute (TRISC) and
// data-movement builds.
//
template <uint32_t SIZE_BYTES, uint32_t ADDR_CRTA_OFFSET>
struct ScratchpadBindingToken {
    static constexpr uint32_t size_bytes = SIZE_BYTES;
    static constexpr uint32_t addr_crta_offset = ADDR_CRTA_OFFSET;  // in bytes
};

}  // namespace scratchpad
