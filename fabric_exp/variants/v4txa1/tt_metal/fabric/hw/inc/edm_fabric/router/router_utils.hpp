// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Router-independent helpers both router programs use: compile-time loops and channel-mask helpers, members present
// only where their channel or VC is, and the placement primitives (compiler-only). Nothing here reads a compile-time
// arg; router_common.hpp holds the router's facts.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace fabric_router {

// ---------------------------------------------------------------------------------------------------------------------
// Compile-time loops
// ---------------------------------------------------------------------------------------------------------------------

// Calls f(std::integral_constant<size_t, I>) for each I in [0, N).
template <size_t N, typename F>
FORCE_INLINE void for_each_index(F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) { (f(std::integral_constant<size_t, Is>{}), ...); }(
        std::make_index_sequence<N>{});
}

constexpr uint32_t popcount(uint32_t mask) { return static_cast<uint32_t>(__builtin_popcount(mask)); }

// For plain recursion over a channel mask: the lowest channel in MASK, and MASK without it. The per-pass code recurses
// over its masks instead of iterating with a closure (router_sender.hpp, "The loop function takes no closures").
constexpr size_t lowest_bit(uint32_t mask) { return static_cast<size_t>(__builtin_ctz(mask)); }
constexpr uint32_t without_lowest_bit(uint32_t mask) { return mask & (mask - 1); }

// ---------------------------------------------------------------------------------------------------------------------
// Presence
// ---------------------------------------------------------------------------------------------------------------------

// A member that exists only where its channel or VC is present on this router (by configuration or channel trimming),
// and is an empty Absent<TAG> elsewhere (TAG keeps two absent members of one struct at distinct addresses). Code names
// such a member only under `if constexpr` on the same flag, from template-dependent code, so an absent one never has to
// compile.
template <size_t TAG>
struct Absent {};
template <bool PRESENT, typename T, size_t TAG>
using Present = std::conditional_t<PRESENT, T, Absent<TAG>>;

// `value` where PRESENT, Absent<TAG> elsewhere: to initialize a Present member.
template <bool PRESENT, size_t TAG, typename T>
FORCE_INLINE constexpr Present<PRESENT, T, TAG> present(const T& value) {
    if constexpr (PRESENT) {
        return value;
    } else {
        return {};
    }
}

// One T<I> for each bit I of MASK, lowest first, found by its index at compile time (get<I>()); empty where MASK is 0.
// A channel or VC this router does not run has no member. Code names a member only under `if constexpr` on the same
// mask, from template-dependent code.
template <uint32_t MASK, template <size_t> class T>
struct Each {
    [[no_unique_address]] T<lowest_bit(MASK)> first;
    [[no_unique_address]] Each<without_lowest_bit(MASK), T> rest;

    template <size_t I>
    FORCE_INLINE T<I>& get() {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template get<I>();
        }
    }
    template <size_t I>
    FORCE_INLINE const T<I>& get() const {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template get<I>();
        }
    }
};
template <size_t I>
struct NotHeld : std::false_type {};
template <template <size_t> class T>
struct Each<0, T> {
    // Naming a member this container does not hold is an error where it is instantiated, not where it is written.
    template <size_t I>
    T<I>& get() {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
    template <size_t I>
    const T<I>& get() const {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Placement primitives
// ---------------------------------------------------------------------------------------------------------------------
// Each is compiler-only: it emits no instruction, it only constrains where the compiler may keep or move things.

// Keeps `obj` in memory at this point: its current value is stored before it and read back after it (GCC's documented
// semantics of an asm memory operand). The whole object is affected, so state meant to stay in registers must not
// share an object with it. It says nothing about where `obj` lives between such points. Use: state touched only on some
// passes (a forwarding channel's record, a later VC's receiver state), so the registers go to the state every pass
// touches (V4_PLACEMENT.md).
template <typename T>
FORCE_INLINE void keep_in_memory(T& obj) {
    asm volatile("" : "+m"(obj));
}

// Read group boundaries (std::atomic_signal_fence): a compiler barrier for memory. No access to memory other code could
// see (device registers, L1, any object whose address escapes) moves across either one, so the reads written between
// a begin and an end stay together, with no other such access among them, and their latencies overlap when their
// values are first used after the end. They do not stop arithmetic, or locals kept in registers, moving between the
// reads, and such memory used on both sides of a boundary is stored before it and reloaded after it.
FORCE_INLINE void begin_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }
FORCE_INLINE void end_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }

// An ordering point outside a read group (std::atomic_signal_fence): the same compiler barrier for memory, where code
// relies on the order of the memory accesses around it but they form no group.
FORCE_INLINE void order_point() { std::atomic_signal_fence(std::memory_order_seq_cst); }

}  // namespace fabric_router
