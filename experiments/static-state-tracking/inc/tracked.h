// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A `Tracked<T>` is a value that the compiler either KNOWS (a concrete `T`) or
// does NOT know ("unknown"). `merge` is used
// at control-flow joins and loop fixed points:
//
//     merge(known a, known a) = known a      (agreement is preserved)
//     merge(known a, known b) = unknown      (disagreement widens to unknown)
//     merge(anything, unknown) = unknown     (unknown is absorbing)
//
// This is the single primitive that makes static state tracking sound: after a
// branch whose arms leave the hardware in different configs, the tracked field
// widens to unknown, so the next op is forced to (re)configure. When all paths
// agree, the field stays known and the reconfigure is compiled out.

#ifndef SST_TRACKED_H
#define SST_TRACKED_H

#include <cstdint>

namespace sst {

template <typename T>
struct Tracked {
    T value{};
    bool known = false;

    constexpr Tracked() = default;
    constexpr Tracked(T v) : value(v), known(true) {}

    // True if we KNOW the value and it equals `v`. This is the query the
    // reconfigure guards use: "is the hardware already in state `v`?"
    constexpr bool matches(const T& v) const { return known && value == v; }
};

template <typename T>
constexpr bool operator==(const Tracked<T>& a, const Tracked<T>& b) {
    if (a.known != b.known) {
        return false;
    }
    if (!a.known) {
        return true;  // unknown == unknown
    }
    return a.value == b.value;
}

template <typename T>
constexpr bool operator!=(const Tracked<T>& a, const Tracked<T>& b) {
    return !(a == b);
}

// Keeps the value only if both sides KNOW the same thing.
template <typename T>
constexpr Tracked<T> merge(const Tracked<T>& a, const Tracked<T>& b) {
    Tracked<T> r;
    r.known = a.known && b.known && (a.value == b.value);
    if (r.known) {
        r.value = a.value;
    }
    return r;
}

}  // namespace sst

#endif  // SST_TRACKED_H
