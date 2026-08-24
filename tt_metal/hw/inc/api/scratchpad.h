// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/null_state.h"
#include "experimental/kernel_args.h"

// Opaque handle for a Program-scope scratchpad binding (declared in kernel_bindings_generated.h).
// The user will never directly interact with this type.
//
// The user's host code declares an accessor_name when binding a scratchpad to a kernel.
// The user then uses that accessor_name to construct a Scratchpad in the kernel code.
//
// Usage example:
//   // (Host code declares "my_scratchpad_name" as the scratchpad accessor name for this kernel.)
//   // In the kernel code:
//   Scratchpad<int32_t> my_pad(scratch::my_scratchpad_name);
//
// Here my_scratchpad_name is a constexpr ScratchpadBindingToken, auto-included in
// kernel_bindings_generated.h.
//
// NullState tags whether this KernelSpec attached the accessor to a real scratchpad (NonNull,
// default) or left the resource id empty (Null). Constructing Scratchpad from a Null token is deleted.
//
template <NullState S = NonNull>
class ScratchpadBindingToken;

template <>
class ScratchpadBindingToken<NonNull> {
public:
    explicit constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes) {}

    static constexpr bool is_null = false;
    constexpr operator bool() const noexcept { return true; }

private:
    template <typename T>
    friend class Scratchpad;

    uint32_t crta_offset_;    // word index of the base-address slot in the CRTA buffer
    uint32_t size_in_bytes_;  // static per-node size
};

template <>
class ScratchpadBindingToken<Null> {
public:
    constexpr ScratchpadBindingToken() noexcept = default;

    static constexpr bool is_null = true;
    constexpr operator bool() const noexcept { return false; }
};

// CTAD: `ScratchpadBindingToken{word, size}` → NonNull (matches codegen's unadorned emission).
ScratchpadBindingToken(uint32_t, uint32_t) -> ScratchpadBindingToken<NonNull>;

/**
 * @brief Kernel-side typed span over a Program-scope scratchpad.
 *
 * A Scratchpad is the device-side counterpart to the host ScratchpadSpec: it provides indexed
 * access to the scratchpad region reserved in per-node SRAM ("L1") for the duration of a Program.
 *
 * Construct one from the binding token your host code declared on the kernel's scratchpad binding:
 * @code
 *   // Host code declares the accessor name "my_scratchpad_name" for this kernel.
 *   // In the kernel:
 *   Scratchpad<int32_t> my_pad(scratch::my_scratchpad_name);
 * @endcode
 *
 * The region is provided as raw, uninitialized memory, with no synchronization of read/write
 * across threads. Avoiding undefined-behavior access is the user's responsibility; typical kinds
 * of undefined behavior access here include:
 * 1. Reading uninitialized data.
 * 2. Data races across threads in multi-threaded kernels.
 * 3. Data races across pipeline stages (Unpack/Math/Pack) in compute kernels.
 *
 * Indexed access via operator[] is bounds-checked with ASSERT (see api/debug/assert.h for when it's activated).
 *
 * @tparam T Element type the region is viewed as.
 * @see scratchpad_spec.hpp (host-side ScratchpadSpec)
 */
template <typename T>
class Scratchpad {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = uint32_t;

    // Use CoreLocalMem as pointer type as Scratchpad will always be in CoreLocal memory space (SRAM).
    using pointer = CoreLocalMem<T>;
    using const_pointer = CoreLocalMem<const T>;
    using reference = T&;
    using const_reference = const T&;

    // Metal 2.0 ctor: Create a Scratchpad from its binding token:
    [[nodiscard]] explicit Scratchpad(const ScratchpadBindingToken<NonNull>& token) noexcept :
        Scratchpad(pointer{get_common_arg_val<uint32_t>(token.crta_offset_)}, token.size_in_bytes_) {}

    // Null bindings have no allocated region — constructing a Scratchpad from one is a compile error.
    explicit Scratchpad(const ScratchpadBindingToken<Null>&) = delete;

    /** @brief Get the element at the given index
     *
     * The index is bounds-checked with ASSERT (see api/debug/assert.h for when it's activated).
     *
     * @param index The index of the element to get
     * @return Reference to the element at the given index
     */
    [[nodiscard]] reference operator[](uint32_t index) const {
        auto location = start_addr_ + index;
        ASSERT(location < sentinel_addr_);
        return *location;
    }

    /** @brief Get the size of this scratchpad in number of T elements
     *
     * @return Number of T-sized elements in this scratchpad.
     */
    [[nodiscard]] size_type size() const noexcept {
        // sizeof(T) is size_t (64-bit on Gen2); narrow explicitly to size_type (always safe).
        return static_cast<size_type>(size_in_bytes() / sizeof(T));
    }

    /** @brief Get the size of this scratchpad in number of bytes.
     *
     * Equals ScratchpadSpec::size_per_node from the host spec.
     *
     * @return size of the scratchpad in bytes.
     */
    [[nodiscard]] size_type size_in_bytes() const noexcept {
        // get_address() returns uintptr_t (64-bit on Gen2); narrow explicitly (always safe).
        return static_cast<size_type>(sentinel_addr_.get_address() - start_addr_.get_address());
    }

    /** @brief L1 base address of the scratchpad, as a raw uint32_t byte address.
     *
     * This is the form most kernel-side APIs consume (NOC transfers, CB/LLK configuration, ...).
     *
     * @return the base address of the scratchpad (as uint32_t)
     */
    [[nodiscard]] uint32_t get_base_address() const noexcept {
        return static_cast<uint32_t>(start_addr_.get_address());
    }

    // begin/end pair to enable range-based-for over the entire scratchpad region.
    // NOTE: This does not support standard-library algorithms that require a conforming iterator:
    //       `CoreLocalMem<T>` does not satisfy the named iterator requirements.
    using iterator = pointer;
    [[nodiscard]] iterator begin() const noexcept { return start_addr_; }
    [[nodiscard]] iterator end() const noexcept { return sentinel_addr_; }

    /** @brief The underlying typed L1 view, for callers wanting the full CoreLocalMem<T> surface
     * (pointer arithmetic, scoped_lock, comparisons, ...).
     *
     * For element access, prefer operator[] (bounds-checked); reach for this only when you need the
     * raw underlying handle (e.g. local_mem().get_unsafe_ptr()).
     */
    // Returned by value: CoreLocalMem<T> is trivially copyable and pointer-sized (matches begin()/end()).
    [[nodiscard]] pointer local_mem() const noexcept { return start_addr_; }

private:
    // Create a Scratchpad from an SRAM (L1) base address and size in bytes.
    // This ctor is private because a Scratchpad represents an allocated SRAM (L1) region.
    // A user may NOT construct a Scratchpad from an arbitrary address/size. (Use CoreLocalMem for that use case.)
    [[nodiscard]] Scratchpad(pointer base_addr, size_type size_in_bytes) noexcept :
        start_addr_(base_addr), sentinel_addr_(pointer{base_addr.get_address() + uintptr_t{size_in_bytes}}) {
        ASSERT(base_addr.get_address() % alignof(T) == 0);
        ASSERT(size_in_bytes % sizeof(T) == 0);
    }

    // constexpr note:
    // The following members could be `constexpr` if `CoreLocalMem<T>` supported constexpr
    // construction/copy and a constexpr `get_address()`:
    //   - Scratchpad(pointer, size_type)
    //   - size(), size_in_bytes()
    //   - get_base_address(), local_mem()
    //   - begin(), end()
    // They are currently runtime-only because `CoreLocalMem<T>` does not provide constexpr support
    // for those operations.

    // Invariant:
    // - `start_addr_` and `sentinel_addr_` are always aligned to the alignment of T.
    // - `sentinel_addr_` - `start_addr_` is always a multiple of sizeof(T).
    //
    // Note:
    // sentinel_addr_ could be omitted in class layout if we inject the size information as a template parameter.
    pointer start_addr_, sentinel_addr_;
};

// A scratchpad is a node-local SRAM (L1) allocation, so it can be either endpoint of a NoC transaction.
// This specialization makes it usable directly as the Src or Dst of any Noc operation.
//
// Notes:
//  - `offset_bytes` addresses within the region. For Scratchpad, this is bounds-checked with an assertion.
//  - Scratchpad may be used by both DM and compute (TRISC) kernels, but only DM kernels have NoC access.
//
#if !defined(COMPILE_FOR_TRISC)
template <typename T>
struct noc_traits_t<Scratchpad<T>> {
    struct src_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t offset_bytes = 0;
    };
    struct dst_args_mcast_type {};

    template <Noc::AddressType address_type>
    static auto src_addr(const Scratchpad<T>& src, const Noc&, const src_args_type& args) {
        static_assert(address_type == Noc::AddressType::LOCAL_L1, "Scratchpad can only be used as a local L1 source");
        ASSERT(args.offset_bytes < src.size_in_bytes());
        return src.get_base_address() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const Scratchpad<T>& dst, const Noc&, const dst_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1, "Scratchpad can only be used as a local L1 destination");
        ASSERT(args.offset_bytes < dst.size_in_bytes());
        return dst.get_base_address() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const Scratchpad<T>&, const Noc&, const dst_args_mcast_type&) {
        // A scratchpad is private to one kernel instance per node, so multicasting into it would be
        // writing into other kernels' private memory. Revisit only if a real use case appears.
        static_assert(false, "Scratchpad cannot be used as a NoC multicast destination");
    }
};

template <typename T>
inline constexpr bool noc_zero_l1_endpoint_v<Scratchpad<T>> = true;
template <typename T>
inline constexpr bool is_scratchpad_v<Scratchpad<T>> = true;
#endif  // !defined(COMPILE_FOR_TRISC)
