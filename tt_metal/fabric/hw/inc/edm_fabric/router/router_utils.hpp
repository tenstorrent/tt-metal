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
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (f(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<N>{});
}

constexpr uint32_t popcount(uint32_t mask) { return static_cast<uint32_t>(__builtin_popcount(mask)); }
// The number of bits up to and including the highest set bit (0 for an empty mask).
constexpr size_t mask_width(uint32_t mask) { return mask == 0 ? 0 : 32 - static_cast<size_t>(__builtin_clz(mask)); }

// For plain recursion over a channel mask: the lowest channel in MASK, and MASK without it. The per-pass code recurses
// over its masks instead of iterating with a closure: a lambda capturing a loop's state can keep that state in memory.
constexpr size_t lowest_bit(uint32_t mask) { return static_cast<size_t>(__builtin_ctz(mask)); }
constexpr uint32_t without_lowest_bit(uint32_t mask) { return mask & (mask - 1); }

// ---------------------------------------------------------------------------------------------------------------------
// Presence
// ---------------------------------------------------------------------------------------------------------------------

// A member that exists only where its channel or VC is present on this router (by configuration or channel trimming),
// and is an empty Absent<TAG> elsewhere (TAG keeps two absent members of one struct at distinct addresses). Code names
// it only under `if constexpr` on the same flag, from template-dependent code, so an absent one never has to compile.
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

// Keeps `obj` in memory at this point: its value is stored before and read back after (GCC's documented semantics of an
// asm memory operand), for the whole object, so state meant for registers must not share an object with it. Use: state
// only some passes touch, so the registers go to the state every pass touches.
template <typename T>
FORCE_INLINE void keep_in_memory(T& obj) {
    asm volatile("" : "+m"(obj));
}

// Read group boundaries (std::atomic_signal_fence), a compiler memory barrier: no access to memory other code could see
// (device registers, L1, escaped objects) crosses one, so the reads between a begin and an end stay together and their
// latencies overlap until their first use after the end. Register locals may still move; such memory used on both
// sides is stored before and reloaded after.
FORCE_INLINE void begin_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }
FORCE_INLINE void end_read_group() { std::atomic_signal_fence(std::memory_order_seq_cst); }

}  // namespace fabric_router
