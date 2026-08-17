// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace dest_order {

enum class client : std::uint32_t {
    UNPACK = 0,
    FPU = 1,
    SFPU = 2,
    PACK = 3,
};

constexpr std::uint32_t client_count = 4;

namespace stage {
inline constexpr client unpack = client::UNPACK;
inline constexpr client fpu = client::FPU;
inline constexpr client sfpu = client::SFPU;
inline constexpr client pack = client::PACK;
}  // namespace stage

constexpr std::uint32_t bit_of(client c) { return 1u << static_cast<std::uint32_t>(c); }

constexpr std::uint32_t popcount(std::uint32_t mask) {
    std::uint32_t n = 0;
    while (mask != 0) {
        mask &= mask - 1;
        ++n;
    }
    return n;
}

constexpr std::uint32_t lowest_bit(std::uint32_t mask) { return mask & (~mask + 1); }

template <client... STAGES>
struct chain;

namespace detail {

template <std::uint32_t N>
struct stage_buffer {
    client stages[N > 0 ? N : 1];
    std::uint32_t size;
};

template <client... STAGES>
constexpr stage_buffer<sizeof...(STAGES)> collapse_runs() {
    constexpr std::uint32_t n = sizeof...(STAGES);
    const client in[n > 0 ? n : 1] = {STAGES...};

    stage_buffer<n> out{};
    for (std::uint32_t i = 0; i < n; ++i) {
        if (out.size == 0 || out.stages[out.size - 1] != in[i]) {
            out.stages[out.size++] = in[i];
        }
    }
    if (out.size > 1 && out.stages[out.size - 1] == out.stages[0]) {
        --out.size;
    }
    return out;
}

template <client... STAGES>
struct collapser {
    static constexpr stage_buffer<sizeof...(STAGES)> data = collapse_runs<STAGES...>();

    template <std::size_t... INDEX>
    static constexpr auto rebuild(std::index_sequence<INDEX...>) -> chain<data.stages[INDEX]...>;

    using type = decltype(rebuild(std::make_index_sequence<data.size>{}));
};

}  // namespace detail

template <client... STAGES>
struct chain {
    static constexpr bool declared = true;

    static constexpr std::uint32_t size = sizeof...(STAGES);
    static_assert(size > 0, "a dest order chain must contain at least one stage");

    static constexpr client stages[size] = {STAGES...};

    static constexpr client at(std::uint32_t index) { return stages[index]; }

    static constexpr std::uint32_t next_index(std::uint32_t index) { return (index + 1u) % size; }

    static constexpr std::uint32_t prev_index(std::uint32_t index) { return (index + size - 1u) % size; }

    static constexpr client first() { return stages[0]; }

    static constexpr client last() { return stages[size - 1u]; }

    static constexpr std::uint32_t mask() {
        std::uint32_t m = 0;
        for (std::uint32_t i = 0; i < size; ++i) {
            m |= bit_of(stages[i]);
        }
        return m;
    }

    static constexpr std::uint32_t distinct_count() { return popcount(mask()); }

    static constexpr bool contains(client c) { return (mask() & bit_of(c)) != 0u; }

    static constexpr std::uint32_t count_of(client c) {
        std::uint32_t n = 0;
        for (std::uint32_t i = 0; i < size; ++i) {
            if (stages[i] == c) {
                ++n;
            }
        }
        return n;
    }

    static constexpr std::uint32_t index_of(client c) {
        for (std::uint32_t i = 0; i < size; ++i) {
            if (stages[i] == c) {
                return i;
            }
        }
        return size;
    }

    static constexpr bool is_first_occurrence(std::uint32_t index) { return index_of(stages[index]) == index; }

    static constexpr bool is_run_start(std::uint32_t index) { return stages[prev_index(index)] != stages[index]; }

    using collapsed = typename detail::collapser<STAGES...>::type;

    static constexpr bool is_collapsed() { return collapsed::size == size; }

    static constexpr std::uint32_t successors_mask(client c) {
        std::uint32_t m = 0;
        for (std::uint32_t i = 0; i < size; ++i) {
            if (stages[i] == c) {
                const client n = stages[next_index(i)];
                if (n != c) {
                    m |= bit_of(n);
                }
            }
        }
        return m;
    }

    static constexpr std::uint32_t predecessors_mask(client c) {
        std::uint32_t m = 0;
        for (std::uint32_t i = 0; i < size; ++i) {
            if (stages[i] == c) {
                const client p = stages[prev_index(i)];
                if (p != c) {
                    m |= bit_of(p);
                }
            }
        }
        return m;
    }

    static constexpr client successor(client c) {
        return static_cast<client>(index_of_bit(lowest_bit(successors_mask(c))));
    }

    static constexpr client predecessor(client c) {
        return static_cast<client>(index_of_bit(lowest_bit(predecessors_mask(c))));
    }

    static constexpr bool is_simple_ring() {
        for (std::uint32_t c = 0; c < client_count; ++c) {
            const client candidate = static_cast<client>(c);
            if (contains(candidate) && popcount(successors_mask(candidate)) != 1u) {
                return false;
            }
        }
        return true;
    }

    template <typename FN>
    static constexpr void for_each(FN&& fn) {
        (fn(std::integral_constant<client, STAGES>{}), ...);
    }

    template <typename FN>
    static constexpr void for_each_run(FN&& fn) {
        for_each_run_impl(fn, std::make_index_sequence<size>{});
    }

    template <typename FN>
    static constexpr void for_each_distinct(FN&& fn) {
        for_each_distinct_impl(fn, std::make_index_sequence<size>{});
    }

private:
    static constexpr std::uint32_t index_of_bit(std::uint32_t bit) {
        for (std::uint32_t i = 0; i < client_count; ++i) {
            if (bit == (1u << i)) {
                return i;
            }
        }
        return client_count;
    }

    template <std::size_t INDEX, typename FN>
    static constexpr void call_if_first(FN& fn) {
        if constexpr (is_first_occurrence(INDEX)) {
            fn(std::integral_constant<client, stages[INDEX]>{});
        }
    }

    template <std::size_t INDEX, typename FN>
    static constexpr void call_if_run_start(FN& fn) {
        if constexpr (is_run_start(INDEX)) {
            fn(std::integral_constant<client, stages[INDEX]>{});
        }
    }

    template <typename FN, std::size_t... INDEX>
    static constexpr void for_each_run_impl(FN& fn, std::index_sequence<INDEX...>) {
        (call_if_run_start<INDEX>(fn), ...);
    }

    template <typename FN, std::size_t... INDEX>
    static constexpr void for_each_distinct_impl(FN& fn, std::index_sequence<INDEX...>) {
        (call_if_first<INDEX>(fn), ...);
    }
};

inline std::uint32_t touched_mask = 0;

template <client UNIT>
inline __attribute__((always_inline)) void touch() {
    touched_mask |= bit_of(UNIT);
}

inline __attribute__((always_inline)) void touch_unpack() { touch<client::UNPACK>(); }

inline __attribute__((always_inline)) void touch_fpu() { touch<client::FPU>(); }

inline __attribute__((always_inline)) void touch_sfpu() { touch<client::SFPU>(); }

inline __attribute__((always_inline)) void touch_pack() { touch<client::PACK>(); }

inline __attribute__((always_inline)) bool was_touched(client c) { return (touched_mask & bit_of(c)) != 0; }

inline __attribute__((always_inline)) void reset_touched() { touched_mask = 0; }

}  // namespace dest_order
