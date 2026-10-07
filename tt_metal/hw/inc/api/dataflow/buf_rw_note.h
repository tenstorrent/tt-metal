// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "internal/tensor/binding_id.h"  // tensor_accessor::NO_BINDING_ID

// Op-to-op R/W inference (POC) -- device emit side. When a kernel reads/writes a bound tensor via the Noc
// APIs, we emit one 8-byte (object-slot, kind) record into a non-allocatable SHT_NOTE ".tt.BUF_RW" section.
// The loader harvests it (see ll_api::parse_binary_metadata) so the host can recover which bound objects a
// kernel reads vs writes -- the input a future trace-mode barrier-relaxation optimizer needs. Pure data,
// zero instructions; not loaded to device. Eventually the compiler emits these directly.
namespace tt_buf_rw {

// Record kinds -- MUST match ll_api::BufRwKind (tt_metal/llrt/binary_metadata.hpp). OPAQUE=0 is the
// failure-safe default: a zero-filled/truncated record reads as un-analyzable, never as a plain read.
inline constexpr uint32_t OPAQUE = 0;
inline constexpr uint32_t READ = 1;
inline constexpr uint32_t WRITE = 2;

// Emit one (slot, kind) record into the non-allocated .tt.BUF_RW note section. Both are compile-time
// immediates, so this adds section data only -- no instructions.
template <uint32_t Slot, uint32_t Kind>
inline void note() {
    __asm__ volatile(  // NOLINT(hicpp-no-assembler)
        ".pushsection .tt.BUF_RW,\"\",@note\n\t"
        ".4byte %0\n\t"
        ".4byte %1\n\t"
        ".popsection" ::"n"(Slot),
        "n"(Kind));
}

// Recover an endpoint's op-to-op binding id at the NoC call site. A bound-tensor accessor carries it as
// T::DSpec::binding_id (threaded in by the TensorAccessor deduction guide); non-tensor endpoints (e.g.
// Scratchpad) have no such member and select the primary template. NO_BINDING_ID means "an accessor that
// was not built from a Metal 2.0 binding token" -> nothing to attribute, so not emitted.
template <typename T, typename = void>
struct endpoint {
    static constexpr bool present = false;
    static constexpr uint32_t slot = 0;
};
template <typename T>
struct endpoint<T, std::void_t<decltype(T::DSpec::binding_id)>> {
    static constexpr uint32_t slot = T::DSpec::binding_id;
    static constexpr bool present = (slot != tensor_accessor::NO_BINDING_ID);
};

// Endpoints that stand for a bound accessor without being one -- PageView / ShardView, the AccessorPage / ShardPage an
// accessor's iterators yield -- specialize endpoint to carry the accessor's binding (api/tensor/noc_traits.h).

// Emit a READ/WRITE record for endpoint T iff it is a token-bound tensor accessor.
template <uint32_t Kind, typename T>
inline void note_if_bound() {
    if constexpr (endpoint<T>::present) {
        note<endpoint<T>::slot, Kind>();
    }
}

// The binding id endpoint T carries, or NO_BINDING_ID.
template <typename T>
inline constexpr uint32_t binding_of = endpoint<T>::present ? endpoint<T>::slot : tensor_accessor::NO_BINDING_ID;

// Where a binding is erased -- a type-erased wrapper, a handle that gives the kernel direct access to the tensor's
// memory -- the accesses that follow can't be attributed, but the binding is known right there. Note it as both read
// and written: over-approximating one binding is safe, where a missing record is not and OPAQUE would give up on the
// whole kernel.
template <uint32_t Slot>
inline void note_read_write() {
    if constexpr (Slot != tensor_accessor::NO_BINDING_ID) {
        note<Slot, READ>();
        note<Slot, WRITE>();
    }
}

}  // namespace tt_buf_rw
