// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <tuple>

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
        // LocalTensorAccessor is legal only for tensors stored in L1 (SRAM) node-local memory.
        // Hard error if TensorBindingToken represents a DRAM tensor.
        static_assert(
            !tensor_accessor::TensorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>::args_t::is_dram,
            "LocalTensorAccessor requires an L1-resident tensor, but this binding token is for a DRAM "
            "tensor. A DRAM tensor has no node-local L1 region; use TensorAccessor instead.");
        // ADDR_CRTA_OFFSET is a byte offset; dividing recovers the word index
        static_assert(
            ADDR_CRTA_OFFSET % sizeof(uint32_t) == 0, "TensorBindingToken: ADDR_CRTA_OFFSET must be 4-byte aligned");
    }

    // Cannot construct a LocalTensorAccessor from a NullTensorBindingToken, consider binding the token to an actual
    // resource on host. See: ProgramSpec on host.
    explicit LocalTensorAccessor(tensor_accessor::NullTensorBindingToken) = delete;

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

// A LocalTensorAccessor names this node's L1 region of a tensor, so it can be either endpoint of a NoC
// transaction. This specialization makes it usable directly as the Src or Dst of any Noc operation.
//
// Notes:
//  - `offset_bytes` addresses within the local region. LocalTensorAccessor does not expose the
//    region's extent, so no bounds ASSERT is possible here.
//  - LocalTensorAccessor may be used by both DM and compute (TRISC) kernels, but only DM kernels have
//    NoC access.
#if !defined(COMPILE_FOR_TRISC)
template <typename T>
struct noc_traits_t<LocalTensorAccessor<T>> {
    struct src_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_mcast_type {};

    template <Noc::AddressType address_type>
    static auto src_addr(const LocalTensorAccessor<T>& src, const Noc&, const src_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1, "LocalTensorAccessor can only be used as a local L1 source");
        return src.get_bank_base_address() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const LocalTensorAccessor<T>& dst, const Noc&, const dst_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "LocalTensorAccessor can only be used as a local L1 destination");
        return dst.get_bank_base_address() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const LocalTensorAccessor<T>&, const Noc&, const dst_args_mcast_type&) {
        static_assert(false, "LocalTensorAccessor cannot be used as a NoC multicast destination");
    }
};

template <typename T>
inline constexpr bool noc_zero_l1_endpoint_v<LocalTensorAccessor<T>> = true;
#endif  // !defined(COMPILE_FOR_TRISC)

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
