// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/scratchpad/scratchpad_binding_token.h"
#include "internal/risc_attribs.h"  // tt_l1_ptr (named in get_unsafe_ptr's return type)

// Forward declared from the kernel's API header (api/dataflow/dataflow_api.h for data movement,
// api/compute/common.h for compute/TRISC), which the kernel TU includes before the generated
// kernel_bindings header. Declaring it here — rather than including either API — keeps this
// header usable on both kernel types (mirrors api/tensor/tensor_accessor_args.h).
template <typename T>
T get_common_arg_val(int arg_idx);

/**
 * @brief A private, node-local L1 scratch region for a kernel's working memory.
 *
 * Scratchpad is the device-side accessor for a Metal 2.0 kernel scratchpad (see ScratchpadSpec on
 * the host). It hands a kernel typed, node-local read/write access to a private chunk of L1 — the
 * sanctioned replacement for abusing a dataflow buffer (DFB) as scratch. It works on both data
 * movement and compute (TRISC) kernels.
 *
 * USAGE:
 *   // The kernel author declares the element type T of the scratch region.
 *   auto s = Scratchpad<uint32_t>(scratch::my_host_declared_accessor_name);  // Metal 2.0 ctor
 *   s[0] = 42;       // read or write
 *   auto n = s.size();  // number of T-elements that fit
 *
 * The region is uninitialized; the kernel must write before it reads. Its base address is allocated
 * by the framework at program-compile time and delivered as an implicit common runtime arg; its
 * size is delivered as an implicit compile-time arg. Both are carried by the binding token, so the
 * kernel author never touches an offset or a raw pointer.
 *
 * @tparam T  Element type viewed over the scratch region.
 */
template <typename T>
class Scratchpad {
public:
    // Construct from a Metal 2.0 binding token. The base address is the token's CRTA word; the size
    // is the token's compile-time SIZE_BYTES.
    template <uint32_t SIZE_BYTES, uint32_t ADDR_CRTA_OFFSET>
    explicit Scratchpad(scratchpad::ScratchpadBindingToken<SIZE_BYTES, ADDR_CRTA_OFFSET>) :
        mem_(static_cast<uintptr_t>(get_common_arg_val<uint32_t>(ADDR_CRTA_OFFSET / sizeof(uint32_t)))),
        size_bytes_(SIZE_BYTES) {
        // ADDR_CRTA_OFFSET is a byte offset host codegen produces as crta_word_index * sizeof(u32), so
        // it is word-aligned by construction; the /sizeof(u32) conversion would silently truncate otherwise.
        static_assert(
            ADDR_CRTA_OFFSET % sizeof(uint32_t) == 0,
            "ScratchpadBindingToken: ADDR_CRTA_OFFSET must be 4-byte aligned");
    }

    // Legacy constructor: from a raw node-local L1 base address (byte address) and the region's size
    // in bytes. For hand-written / non-Metal-2.0 kernels.
    explicit Scratchpad(uint32_t base_address, uint32_t size_bytes = 0) :
        mem_(static_cast<uintptr_t>(base_address)), size_bytes_(size_bytes) {}

    // Element access into the scratch region (read and write), bounds-checked in debug builds.
    T& operator[](uint32_t index) const { return mem_[index]; }

    // Raw, directly-dereferenceable L1 pointer to the start of the scratch region.
    tt_l1_ptr T* get_unsafe_ptr() const { return mem_.get_unsafe_ptr(); }

    // L1 base address of the scratch region.
    uint32_t get_base_address() const { return static_cast<uint32_t>(mem_.get_address()); }

    // Size of the scratch region, in bytes.
    uint32_t size_bytes() const { return size_bytes_; }

    // Number of T-elements the scratch region holds (size_bytes / sizeof(T)).
    uint32_t size() const { return size_bytes_ / sizeof(T); }

    // The underlying typed L1 view, for callers wanting the full CoreLocalMem<T> surface
    // (pointer arithmetic, scoped_lock, comparisons, ...).
    const CoreLocalMem<T>& local_mem() const { return mem_; }

private:
    CoreLocalMem<T> mem_;
    uint32_t size_bytes_;
};
