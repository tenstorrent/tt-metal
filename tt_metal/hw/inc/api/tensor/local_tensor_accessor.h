// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/tensor/tensor_accessor_args.h"
#include "api/tensor/tensor_binding_token.h"

/**
 * @brief A minimal accessor for a tensor's node-local L1 region.
 * LocalTensorAccessor is the local-only counterpart to TensorAccessor.
 * Unlike TensorAccessor, it can be used on both data movement and compute kernels.
 *
 * "Node-local L1 region" is the part of the tensor that lives in this node's SRAM.
 *  - For a sharded tensor, this is the local shard.
 *  - For an interleaved tensor (in SRAM/L1), this is the contiguous local memory region
 *    storing tensor data for this node (in physical layout-dictated order).
 *
 * To construct a LocalTensorAccessor:
 *  - Metal 2.0: use the token name your host code declared on the kernel's tensor binding
 *  - Legacy: construct from a raw L1 base address
 * @code
 *   // T is the element type of the local region, chosen by the kernel author.
 *   LocalTensorAccessor<T> a(tensor::my_host_declared_accessor_name); // Metal 2.0
 *   LocalTensorAccessor<T> b(l1_base_address);                        // legacy
 *   auto& elem = a[0];                                                // read or write
 * @endcode
 *
 * Notes:
 *  - LocalTensorAccessor replaces the legacy "pinned CB as L1 pointer" pattern.
 *  - The current API is deliberately minimal; more features will be added as use cases arise.
 *  - Element access is currently NOT bounds-checked against the region's extent
 *    (this should be added in the future).
 *
 * Template parameters:
 * @tparam T  Element type stored in the local region.
 */
template <typename T>
class LocalTensorAccessor {
public:
    // Construct from a Metal 2.0 binding token, from the host-declared accessor name.
    // (tensor::<accessor_name> constant is in the generated, auto-included kernel_bindings_generated.h.)
    // e.g.
    // LocalTensorAccessor<T> my_local_accessor(tensor::my_host_declared_accessor_name);
    //
    template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
    [[nodiscard]] explicit LocalTensorAccessor(
        tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>) noexcept :
        // The region's L1 base address is stored in the CRTA at the token-supplied offset.
        // Delegates to the legacy constructor, which takes a raw L1 base address.
        LocalTensorAccessor(get_common_arg_val<uint32_t>(ADDR_CRTA_OFFSET / sizeof(uint32_t))) {
        // ADDR_CRTA_OFFSET is a byte offset; dividing recovers the word index
        static_assert(
            ADDR_CRTA_OFFSET % sizeof(uint32_t) == 0, "TensorBindingToken: ADDR_CRTA_OFFSET must be 4-byte aligned");
    }

    // Legacy constructor: from a raw node-local L1 base address (a byte address).
    // (Typically a legacy Buffer's address passed into the kernel as a CRTA.)
    [[nodiscard]] explicit LocalTensorAccessor(uint32_t bank_base_address) noexcept :
        mem_(static_cast<uintptr_t>(bank_base_address)) {
        ASSERT(mem_.get_address() % alignof(T) == 0);
    }

    /** @brief Access the element at the given index (read or write).
     *
     * Watcher validates the addresses in debug builds.
     * Currently NOT bounds-checked against the region's extent.
     *
     * @param index Element index into the local region.
     * @return Reference to the element at the given index.
     */
    [[nodiscard]] T& operator[](uint32_t index) const { return mem_[index]; }

    /** @brief L1 base address of the local region, as a raw uint32_t byte address.
     *
     * This is the form most kernel-side APIs consume (NOC transfers, CB/LLK configuration, ...).
     *
     * @return the local region's L1 base address (as uint32_t).
     */
    [[nodiscard]] uint32_t get_bank_base_address() const noexcept {
        // static_cast narrows to uint32_t (uintptr_t is 64-bit on Gen2); an L1 address always fits.
        return static_cast<uint32_t>(mem_.get_address());
    }

    /** @brief The underlying typed L1 view, for callers wanting the full CoreLocalMem<T> surface
     * (pointer arithmetic, scoped_lock, comparisons, ...).
     *
     * For element access, prefer operator[]; use this only when you need the raw underlying handle
     * (e.g. local_mem().get_unsafe_ptr()).
     */
    // Returned by value: CoreLocalMem<T> is trivially copyable and pointer-sized.
    [[nodiscard]] CoreLocalMem<T> local_mem() const noexcept { return mem_; }

private:
    CoreLocalMem<T> mem_;
};
