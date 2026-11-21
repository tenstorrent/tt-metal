// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/scratchpad_binding_token.h"
#include "experimental/kernel_args.h"

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
 * 4. Data races across kernels when ScratchpadSpec::advanced_options.allow_instance_multi_binding is enabled.
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
    [[nodiscard]] explicit Scratchpad(const ScratchpadBindingToken& token) noexcept :
        Scratchpad(
            pointer{get_common_arg_val<uint32_t>(token.crta_offset_)},
            token.size_in_bytes_,
            static_cast<DataFormat>(token.llk_metadata_.format)) {}

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
        // Addresses are uintptr_t (64-bit on Gen2); narrow explicitly (always safe).
        return static_cast<size_type>(
            local_mem::detail::AddressAccess::address(sentinel_addr_) -
            local_mem::detail::AddressAccess::address(start_addr_));
    }

    /** @brief Get the LLK data format configured for this scratchpad.
     *
     * Returns DataFormat::Invalid when the host ScratchpadSpec did not provide data format metadata.
     */
    [[nodiscard]] constexpr DataFormat get_dataformat() const noexcept { return data_format_; }

#if defined(COMPILE_FOR_TRISC)
    // Returns the LLKOperand pointing to the base address of the scratchpad
    // Functionalities meant to be used with LLKOperand for LLK 2.0
    //
    // Parameters:
    // - Operand: The LLKOperand type to return. (hint: use LLKOperandFrom<scratchpad::token>)
    template <typename Operand>
    [[nodiscard]] Operand operand() const noexcept {
        return Operand{(static_cast<uint32_t>(local_mem::detail::AddressAccess::address(start_addr_)) >> 4) - 1};
    }
#endif

    // begin/end pair to enable range-based-for over the entire scratchpad region.
    // NOTE: This does not support standard-library algorithms that require a conforming iterator:
    //       `CoreLocalMem<T>` does not satisfy the named iterator requirements.
    using iterator = pointer;
    [[nodiscard]] iterator begin() const noexcept { return start_addr_; }
    [[nodiscard]] iterator end() const noexcept { return sentinel_addr_; }

    /** @brief The underlying typed L1 view, for callers wanting the full CoreLocalMem<T> surface
     * (pointer arithmetic, comparisons, ...).
     *
     * For element access, prefer operator[] (bounds-checked) within a scoped_lock() scope. Reach
     * for this only when you need the underlying handle (e.g. pointer arithmetic, or a helper's CoreLocalMem
     * overload).
     */
    // Returned by value: CoreLocalMem<T> is trivially copyable and pointer-sized (matches begin()/end()).
    [[nodiscard]] pointer local_mem() const noexcept { return start_addr_; }

    /** @brief Lock num_elements elements starting at element `offset`.
     *
     * @param offset       Index of the first element to lock.
     * @param num_elements Number of T elements to lock.
     */
    [[nodiscard]] auto scoped_lock(uint32_t offset, uint32_t num_elements) const {
        ASSERT(start_addr_ + offset + num_elements <= sentinel_addr_);
        return (start_addr_ + offset).scoped_lock(num_elements);
    }

private:
    // Create a Scratchpad from an SRAM (L1) base address and size in bytes.
    // This ctor is private because a Scratchpad represents an allocated SRAM (L1) region.
    // A user may NOT construct a Scratchpad from an arbitrary address/size. (Use CoreLocalMem for that use case.)
    [[nodiscard]] Scratchpad(pointer base_addr, size_type size_in_bytes, DataFormat data_format) noexcept :
        start_addr_(base_addr),
        sentinel_addr_(pointer{local_mem::detail::AddressAccess::address(base_addr) + uintptr_t{size_in_bytes}}),
        data_format_(data_format) {
        ASSERT(local_mem::detail::AddressAccess::address(base_addr) % alignof(T) == 0);
        ASSERT(size_in_bytes % sizeof(T) == 0);
    }

    // constexpr note:
    // The following members could be `constexpr` if `CoreLocalMem<T>` supported constexpr
    // construction/copy and constexpr address access:
    //   - Scratchpad(pointer, size_type, DataFormat)
    //   - size(), size_in_bytes()
    //   - local_mem()
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
    DataFormat data_format_;
};

// A scratchpad is a node-local SRAM (L1) allocation, so it can be either endpoint of a NoC transaction (DM kernels
// only): see internal/local_mem/noc_traits.h.
#include "internal/local_mem/noc_traits.h"
