// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor_args.h"
#include "api/tensor/tensor_binding_token.h"
#include "internal/risc_attribs.h"  // tt_l1_ptr (named in get_unsafe_ptr's return type)

/**
 * @brief A minimal accessor for a tensor's node-local L1 shard.
 *
 * LocalTensorAccessor is the local-only counterpart to TensorAccessor.
 * Unlike TensorAccessor, it can be used on both data movement and compute kernels.
 *
 * USAGE:
 *   // The kernel author declares the element type T of the local shard
 *   auto a = LocalTensorAccessor<T>(tensor::my_host_declared_accessor_name); // Metal 2.0 ctor
 *   auto& lut = a[0];   // read or write
 *
 * NOTE:
 *   LocalTensorAccessor replaces the "pinned CB as L1 pointer" legacy pattern (get_pointer_to_cb_data<T>).
 *
 * FEATURES:
 *   Constructors are provided for both Metal 2.0 and legacy kernels.
 *   Basic memory access via operator[].
 *   Basic getters for tensor info.
 *
 * @tparam T  Element type stored in the local shard.
 */
template <typename T>
class LocalTensorAccessor {
public:
    // Construct from a Metal 2.0 binding token:
    template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
    explicit LocalTensorAccessor(tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>) :
        // Base address is stored in the first CRTA word
        mem_(static_cast<uintptr_t>(get_common_arg_val<uint32_t>(ADDR_CRTA_OFFSET / sizeof(uint32_t)))),
        // Runtime accessor fields (if any) follow the base-address slot
        aligned_page_size_(
            TensorAccessorArgs<CTA_OFFSET, ADDR_CRTA_OFFSET / sizeof(uint32_t) + 1>{}.get_aligned_page_size()),
        rank_(TensorAccessorArgs<CTA_OFFSET, ADDR_CRTA_OFFSET / sizeof(uint32_t) + 1>{}.get_rank()) {
        // ADDR_CRTA_OFFSET must be word_aligned
        static_assert(
            ADDR_CRTA_OFFSET % sizeof(uint32_t) == 0, "TensorBindingToken: ADDR_CRTA_OFFSET must be 4-byte aligned");
    }

    // Element access into the local shard (read and write), bounds-checked in debug builds.
    T& operator[](uint32_t index) const { return mem_[index]; }

    // Raw, directly-dereferenceable L1 pointer to the start of the local shard.
    tt_l1_ptr T* get_unsafe_ptr() const { return mem_.get_unsafe_ptr(); }

    // L1 base address of the local shard.
    uint32_t get_bank_base_address() const { return static_cast<uint32_t>(mem_.get_address()); }

    // Page size in bytes.
    uint32_t get_aligned_page_size() const { return aligned_page_size_; }

    // Tensor rank.
    uint32_t rank() const { return rank_; }

    // The underlying typed L1 view, for callers wanting the full CoreLocalMem<T> surface
    // (pointer arithmetic, scoped_lock, comparisons, ...).
    const CoreLocalMem<T>& local_mem() const { return mem_; }

private:
    CoreLocalMem<T> mem_;
    uint32_t aligned_page_size_;
    uint32_t rank_;
};
