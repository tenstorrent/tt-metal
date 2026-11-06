// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <tuple>

#include "api/core_local_mem.h"
#include "api/dataflow/buf_rw_note.h"
#include "api/debug/assert.h"
#include "api/tensor/tensor_accessor_args.h"
#include "api/tensor/tensor_binding_token.h"

// Forward declared from dataflow_api.h / compute/common.h
template <typename T>
T get_arg_val(int arg_idx);

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
        // LocalTensorAccessor is legal only for tensors stored in L1 (SRAM) node-local memory.
        // Hard error if TensorBindingToken represents a DRAM tensor.
        static_assert(
            !tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>::args_t::is_dram,
            "LocalTensorAccessor requires an L1-resident tensor, but this binding token is for a DRAM "
            "tensor. A DRAM tensor has no node-local L1 region; use TensorAccessor instead.");
        // ADDR_CRTA_OFFSET is a byte offset; dividing recovers the word index
        static_assert(
            ADDR_CRTA_OFFSET % sizeof(uint32_t) == 0, "TensorBindingToken: ADDR_CRTA_OFFSET must be 4-byte aligned");
        // Op-to-op R/W inference: this hands the kernel the tensor's local memory (CPU loads/stores, LLK operands, raw
        // NoC addresses), none of which carries the binding. Note it as both read and written; the slot is the binding
        // id a TensorAccessor built from the same token uses.
        tt_buf_rw::note_read_write<ADDR_CRTA_OFFSET>();
    }

    // Construct from a Metal 2.0 per-node binding token.
    template <uint32_t CTA_OFFSET, uint32_t ADDR_RTA_OFFSET>
    [[nodiscard]] explicit LocalTensorAccessor(
        tensor_accessor::PerNodeTensorBindingToken<CTA_OFFSET, ADDR_RTA_OFFSET>) noexcept :
        LocalTensorAccessor(get_arg_val<uint32_t>(ADDR_RTA_OFFSET / sizeof(uint32_t))) {
        static_assert(
            !tensor_accessor::PerNodeTensorBindingToken<CTA_OFFSET, ADDR_RTA_OFFSET>::args_t::is_dram,
            "LocalTensorAccessor requires an L1-resident tensor, but this binding token is for a DRAM "
            "tensor. A DRAM tensor has no node-local L1 region; use TensorAccessor instead.");
        static_assert(
            ADDR_RTA_OFFSET % sizeof(uint32_t) == 0,
            "PerNodeTensorBindingToken: ADDR_RTA_OFFSET must be 4-byte aligned");
        // Op-to-op R/W inference keys a binding by its one address, which a per-node tensor doesn't have.
        // The accesses can't be attributed, so give up on the kernel rather than miss one.
        tt_buf_rw::note_opaque();
    }

    // Construct from the "binding not present" token.
    // Meant to be used with `get_token_if_present` to make the token-not-exist branch compile.
    // This will never be run in runtime & NullTensorBindingToken is not constructible.
    [[nodiscard]] explicit LocalTensorAccessor(const tensor_accessor::NullTensorBindingToken&) noexcept :
        LocalTensorAccessor(uint32_t{0}) {}

    // Legacy constructor: from a raw node-local L1 base address (a byte address).
    // (Typically a legacy Buffer's address passed into the kernel as a CRTA.)
    [[nodiscard]] explicit LocalTensorAccessor(uint32_t bank_base_address) noexcept :
        mem_(static_cast<uintptr_t>(bank_base_address)) {
        ASSERT(bank_base_address % alignof(T) == 0);
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

#if defined(COMPILE_FOR_TRISC)
    // Returns the LLKOperand pointing to the base address of the local tensor accessor
    // Functionalities meant to be used with LLKOperand for LLK 2.0
    //
    // Parameters:
    // - Operand: The LLKOperand type to return. (hint: use LLKOperandFrom<tensor::token>)
    template <typename Operand>
    [[nodiscard]] Operand operand() const noexcept {
        return Operand{(static_cast<uint32_t>(local_mem::detail::AddressAccess::address(mem_)) >> 4) - 1};
    }
#endif

    /** @brief Lock num_elements elements starting at element `offset`.
     *
     * @param offset       Index of the first element to lock.
     * @param num_elements Number of T elements to lock.
     */
    [[nodiscard]] auto scoped_lock(uint32_t offset, uint32_t num_elements) const {
        return (mem_ + offset).scoped_lock(num_elements);
    }

    /** @brief The underlying typed L1 view, for callers wanting the full CoreLocalMem<T> surface
     * (pointer arithmetic, comparisons, ...).
     *
     * For element access, prefer operator[] within a scoped_lock() scope. Use this only when you
     * need the underlying handle (e.g. pointer arithmetic, or a helper's CoreLocalMem overload).
     */
    // Returned by value: CoreLocalMem<T> is trivially copyable and pointer-sized.
    [[nodiscard]] CoreLocalMem<T> local_mem() const noexcept { return mem_; }

private:
    CoreLocalMem<T> mem_;
};

// A LocalTensorAccessor names this node's L1 region of a tensor, so it can be either endpoint of a NoC transaction (DM
// kernels only): see internal/local_mem/noc_traits.h.
#include "internal/local_mem/noc_traits.h"

/**
 * @brief Map a tensor sequence (tuple of TensorBindingToken) to an array of LocalTensorAccessor.
 *
 * A tuple of TensorBindingTokens is obtained from a TensorBindingSequence: host codegen emits
 * `tensor::<sequence_name>` as `std::tuple` of the named binding tokens. This helper
 * maps that sequence into a std::array of LocalTensorAccessors, one per token, in the same order.
 * LocalTensorAccessor is only templated on the element type T, so all entries share one type.
 *
 * All tokens must refer to L1-resident tensors; a DRAM token fails at compile time. For DRAM or
 * NOC address generation, use make_tensor_accessors instead (DM kernels only).
 *
 * Usage:
 *   auto local_accessors = make_local_tensor_accessors<uint32_t>(tensor::inputs);
 *   auto& acc = local_accessors[i];  // i-th LocalTensorAccessor in the sequence
 *   auto& elem = acc[0];            // element in that accessor's local L1 region
 *
 * @tparam T Element type stored in each local region.
 * @param tokens constexpr tuple of TensorBindingTokens from a TensorBindingSequence
 *               (tensor::<sequence_name>).
 * @return std::array<LocalTensorAccessor<T>, N>, one per token, in sequence order.
 */
template <typename T, typename... Tokens>
std::array<LocalTensorAccessor<T>, sizeof...(Tokens)> make_local_tensor_accessors(const std::tuple<Tokens...>& tokens) {
    return std::apply(
        [](const auto&... toks) {
            return std::array<LocalTensorAccessor<T>, sizeof...(Tokens)>{LocalTensorAccessor<T>(toks)...};
        },
        tokens);
}
