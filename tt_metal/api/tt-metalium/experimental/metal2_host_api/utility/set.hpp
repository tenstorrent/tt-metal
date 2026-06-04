// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <ranges>
#include <utility>
#include <vector>

namespace tt::tt_metal::experimental {

// A flat collection of unique keys.
//
//  - Each key appears at most once.
//  - Iteration order is unspecified.
//  - Iterates as const K&, so elements can't be mutated in place. Range
//    construction of other containers (`std::set<K>(s.begin(), s.end())`) works
//    directly.
//  - The interface mirrors std::set: contains() / find() look a key up, insert()
//    adds a key only if it is absent, and erase() removes it.
//  - Uniqueness and lookup compare keys through `Proj`, which defaults to
//    comparing whole keys. Pass a pointer-to-member (or any callable usable as a
//    projection) to compare on a member instead, e.g.
//    `FlatSet<MyStruct, &MyStruct::id>` treats two entries as the same key iff
//    their `.id` members compare equal.
template <typename K, auto Proj = std::identity{}>
class FlatSet {
public:
    using key_type = K;
    using value_type = K;
    using size_type = std::size_t;

private:
    using Storage = std::vector<K>;

public:
    // Elements are immutable, so both iterators expose const K&.
    using iterator = typename Storage::const_iterator;
    using const_iterator = typename Storage::const_iterator;

    FlatSet() noexcept = default;

    // Builds from a `{k, k, ...}` list. Duplicate keys are ignored (only the first
    // is kept), as with std::set / std::unordered_set.
    FlatSet(std::initializer_list<K> keys) {
        for (const auto& key : keys) {
            insert(key);
        }
    }

    // Builds from any range of keys convertible to K (e.g. another FlatSet, a
    // std::set / std::unordered_set, or a std::vector / std::span). Each key is
    // copied in; duplicate keys are ignored.
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>, K>
    explicit FlatSet(const R& keys) {
        for (const auto& key : keys) {
            insert(key);
        }
    }

    // Iterates the keys as const K&.
    [[nodiscard]] const_iterator begin() const noexcept { return keys_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return keys_.end(); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return keys_.cbegin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return keys_.cend(); }

    [[nodiscard]] bool empty() const noexcept { return keys_.empty(); }
    [[nodiscard]] size_type size() const noexcept { return keys_.size(); }
    void clear() noexcept { keys_.clear(); }

    // Returns an iterator to the entry whose projected key equals `key`'s, or
    // end() if absent.
    [[nodiscard]] const_iterator find(const K& key) const {
        return std::ranges::find(keys_, std::invoke(Proj, key), Proj);
    }

    // Returns whether `key` is present.
    [[nodiscard]] bool contains(const K& key) const { return find(key) != end(); }

    // Returns the number of keys equal to `key` (0 or 1).
    [[nodiscard]] size_type count(const K& key) const { return contains(key) ? 1 : 0; }

    // Inserts a key only if it is absent. Returns {iterator to the key, true} when
    // inserted, or {iterator to the existing key, false} otherwise.
    std::pair<const_iterator, bool> insert(const K& key) {
        if (auto it = find(key); it != end()) {
            return {it, false};
        }
        keys_.push_back(key);
        return {std::prev(keys_.end()), true};
    }
    std::pair<const_iterator, bool> insert(K&& key) {
        if (auto it = find(key); it != end()) {
            return {it, false};
        }
        keys_.push_back(std::move(key));
        return {std::prev(keys_.end()), true};
    }

    // Removes `key` if present. Returns the number of keys removed (0 or 1).
    // Invalidates iterators at or after the removed key.
    size_type erase(const K& key) {
        auto it = find(key);
        if (it == end()) {
            return 0;
        }
        keys_.erase(it);
        return 1;
    }

    // Equal iff both hold the same elements (iteration order is ignored): same
    // size, and every element compares equal — by full value, not just its
    // projected key — to its counterpart in the other set. So under a projection,
    // entries that share a key but differ in their unprojected members are not
    // equal (mirroring how std::set's operator== ignores the comparator).
    [[nodiscard]] friend bool operator==(const FlatSet& a, const FlatSet& b)
        requires std::equality_comparable<K>
    {
        if (a.size() != b.size()) {
            return false;
        }
        for (const auto& key : a.keys_) {
            const auto it = b.find(key);
            if (it == b.end() || !(*it == key)) {
                return false;
            }
        }
        return true;
    }

private:
    Storage keys_;
};

template <typename K, auto Proj = std::identity{}>
using set = FlatSet<K, Proj>;

}  // namespace tt::tt_metal::experimental
