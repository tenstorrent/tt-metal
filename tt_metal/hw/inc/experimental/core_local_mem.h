// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/noc.h"
#include "experimental/lock.h"

namespace experimental {

/**
 * @brief Provides a safe pointer to a structure of type T in the core's local memory
 *
 * Pointers are tagged with tt_l1_ptr to give the compiler latency information.
 *
 * Usage:
 * - For non-volatile access with compiler optimizations: CoreLocalMem<uint32_t>
 * - For volatile access (prevents optimization): CoreLocalMem<volatile uint32_t>
 *
 * Note: When using non-volatile types with NOC operations, you must ensure proper
 * memory ordering with compiler barriers (e.g., asm volatile("" ::: "memory"))
 * or L1 cache invalidation as needed.
 */
template <typename T, typename AddressType = uintptr_t>
class CoreLocalMem {
    using difference_type = std::ptrdiff_t;

    static_assert(std::is_integral<AddressType>::value, "AddressType must be an integral type");
    static_assert(std::is_unsigned<AddressType>::value, "AddressType must be unsigned for address representation");
    static_assert(
        sizeof(AddressType) >= sizeof(difference_type),
        "AddressType must be large enough to hold difference_type for safe pointer arithmetic");

public:
    /** @brief Construct a CoreLocalMem instance from a raw address
     *
     * @param address The raw address of the structure in the core's local memory
     */
    CoreLocalMem(AddressType address) : address_(address) {}

    /** @brief Construct a CoreLocalMem instance from a raw pointer
     *
     * @param ptr The pointer to the structure in the core's local memory
     */
    CoreLocalMem(T* ptr) : address_(reinterpret_cast<AddressType>(ptr)) {}

    /** @brief Copy constructor
     *
     * @param other The other CoreLocalMem to copy from
     */
    CoreLocalMem(const CoreLocalMem&) = default;

    /** @brief Copy assignment operator
     *
     * @param other The other CoreLocalMem to copy from
     * @return A reference to the assigned CoreLocalMem
     */
    CoreLocalMem& operator=(const CoreLocalMem&) = default;

    /** @brief Get the raw pointer to the structure in the core's local memory
     *
     * @return The raw pointer to the structure in the core's local memory
     */
    tt_l1_ptr T* get_unsafe_ptr() const { return reinterpret_cast<tt_l1_ptr T*>(address_); }

    /** @brief Get the memory address
     *
     * @return The address
     */
    AddressType get_address() const { return address_; }

    /** @brief Get the element at the given index
     *
     * @param index The index of the element to get
     * @return Reference to the element at the given index
     */
    T& operator[](uint32_t index) const {
        DEBUG_SANITIZE_L1_ADDR(address_ + (index + 1) * sizeof(T), sizeof(T));
        return get_unsafe_ptr()[index];
    }

    /** @brief Dereference operator to get reference to the value
     *
     * @return Reference to the value at the address
     */
    T& operator*() const {
        DEBUG_SANITIZE_L1_ADDR(address_, sizeof(T));
        return get_unsafe_ptr()[0];
    }

    /** @brief Arrow operator for struct/class member access
     *
     * @return Pointer to the structure in the core's local memory
     */
    tt_l1_ptr T* operator->() const {
        DEBUG_SANITIZE_L1_ADDR(address_, sizeof(T));
        return get_unsafe_ptr();
    }

    CoreLocalMem& operator+=(difference_type offset) {
        address_ += offset * sizeof(T);
        return *this;
    }

    CoreLocalMem& operator-=(difference_type offset) {
        address_ -= offset * sizeof(T);
        return *this;
    }

    CoreLocalMem& operator++() {
        address_ += sizeof(T);
        return *this;
    }

    CoreLocalMem& operator--() {
        address_ -= sizeof(T);
        return *this;
    }

    CoreLocalMem operator++(int) {
        CoreLocalMem tmp = *this;
        operator++();
        return tmp;
    }

    CoreLocalMem operator--(int) {
        CoreLocalMem tmp = *this;
        operator--();
        return tmp;
    }

    CoreLocalMem operator+(difference_type offset) const { return CoreLocalMem(address_ + offset * sizeof(T)); }

    CoreLocalMem operator-(difference_type offset) const { return CoreLocalMem(address_ - offset * sizeof(T)); }

    difference_type operator-(const CoreLocalMem& other) const {
        difference_type byte_diff =
            static_cast<difference_type>(address_) - static_cast<difference_type>(other.address_);
        // Compiler automatically optimizes division to a shift if T is pow2
        return byte_diff / sizeof(T);
    }

    [[nodiscard]] auto scoped_lock() {
        return Lock([this]() { release_scoped_lock(); });
    }

    bool operator==(const CoreLocalMem& other) const { return address_ == other.address_; }
    bool operator!=(const CoreLocalMem& other) const { return address_ != other.address_; }
    bool operator<(const CoreLocalMem& other) const { return address_ < other.address_; }
    bool operator<=(const CoreLocalMem& other) const { return address_ <= other.address_; }
    bool operator>(const CoreLocalMem& other) const { return address_ > other.address_; }
    bool operator>=(const CoreLocalMem& other) const { return address_ >= other.address_; }
    explicit operator bool() const { return address_ != 0; }

private:
    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    AddressType address_;
};

template <typename T, typename AddressType>
struct noc_traits_t<CoreLocalMem<T, AddressType>> {
    struct src_args_type {
        AddressType offset_bytes = 0;
    };
    struct dst_args_type {
        AddressType offset_bytes = 0;
    };
    struct dst_args_mcast_type {};

    template <Noc::AddressType address_type>
    static auto src_addr(const CoreLocalMem<T, AddressType>& src, const Noc&, const src_args_type& args) {
        static_assert(address_type == Noc::AddressType::LOCAL_L1, "CoreLocalMem can only be used as local L1 source");
        return src.get_address() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const CoreLocalMem<T, AddressType>& dst, const Noc& noc, const dst_args_type& args) {
        static_assert(address_type == Noc::AddressType::LOCAL_L1, "CoreLocalMem can only be used as local L1 dest");
        return dst.get_address() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(
        const CoreLocalMem<T, AddressType>& dst, const Noc& noc, const dst_args_mcast_type& args) {
        static_assert(false, "CoreLocalMem cannot be used as NoC mcast destination");
    }
};

}  // namespace experimental
