// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/optional_reference.hpp>
#include <tt_stl/small_vector.hpp>

// Forward declarations of the ttsl::hash entry points used by the std::hash<Table> specialization
// below, so this header need not pull in the (heavy) <tt_stl/reflection.hpp>. The definitions live
// there; any translation unit that actually hashes a Table must include reflection.hpp so the
// specialization can be instantiated. These signatures must stay in sync with reflection.hpp.
namespace ttsl::hash {

using hash_t = std::uint64_t;

template <typename... Types>
hash_t hash_objects(hash_t seed, const Types&... args) noexcept;  // NOLINT(readability-redundant-declaration)

template <typename T>
void hash_combine(std::size_t& seed, const T& value);  // NOLINT(readability-redundant-declaration)

}  // namespace ttsl::hash

namespace tt::tt_metal::experimental {

// A key -> value map with unique keys.
//
//  - K must be hashable with std::hash and comparable with ==.
//  - Each key appears at most once.
//  - Iteration order is unspecified.
//  - Iterates as std::pair<const K, V>, so keys can't be mutated through an
//    iterator. Structured bindings (`for (const auto& [k, v] : table)`) and
//    std::map / std::unordered_map range construction
//    (`std::map<K, V>(t.begin(), t.end())`) work directly.
//  - The interface mirrors std::map: find() / get() look a key up, insert() and
//    emplace() add an entry only if the key is absent, and operator[] inserts a
//    key or overwrites the existing value.
template <typename K, typename V>
class Table {
    // Reference mapped types (e.g. Table<K, T&>) are not supported: the backing
    // vector relocates elements via assignment (erase/copy-assign), and assigning a
    // pair with a reference member writes through the reference rather than
    // repointing it.
    static_assert(!std::is_reference_v<V>, "Table does not support reference mapped types (V)");

public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const K, V>;
    using size_type = std::size_t;

private:
    // Entries are stored as a mutable std::pair<K, V> (so the backing vector stays
    // easy to grow, erase, and assign), but exposed through the iterators below as
    // the const-key value_type.
    // Small-vector backing: Tables typically hold only a handful of entries, so the inline storage
    // avoids a heap allocation in the common case. Larger Tables spill to the heap with identical
    // semantics, so correctness never depends on the inline capacity.
    static constexpr std::size_t kInlineStorageCapacity = 8;
    using Storage = ttsl::SmallVector<std::pair<K, V>, kInlineStorageCapacity>;

    // Open-addressing hash index over entries_, with linear probing. A slot holds 0 when empty, or
    // p when it refers to entries_[p - 1]. The inline capacity holds the index of a full inline
    // Storage at the maximum load factor, so small Tables stay heap-free.
    static constexpr std::size_t kInlineIndexCapacity = 2 * kInlineStorageCapacity;
    static_assert(std::has_single_bit(kInlineIndexCapacity), "The index size must be a power of two");
    using Index = ttsl::SmallVector<std::uint32_t, kInlineIndexCapacity>;
    static constexpr std::size_t kMinIndexSlots = kInlineIndexCapacity;

public:
    // Wraps a backing iterator and re-presents the stored pair it points at as the
    // const-key value_type, so a key can't be mutated through an iterator.
    // ExposedPair is value_type for the mutable iterator and const value_type for
    // the const one.
    template <typename BackingIt, typename ExposedPair>
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Table::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = ExposedPair*;
        using reference = ExposedPair&;

        Iterator() = default;
        explicit Iterator(BackingIt it) : it_(it) {}

        // Implicit mutable -> const conversion (mirrors std::vector's iterators).
        template <typename OtherBackingIt, typename OtherExposedPair>
            requires std::convertible_to<OtherBackingIt, BackingIt>
        Iterator(const Iterator<OtherBackingIt, OtherExposedPair>& other) : it_(other.backing()) {}

        reference operator*() const { return reinterpret_cast<reference>(*it_); }
        pointer operator->() const { return reinterpret_cast<pointer>(&*it_); }

        Iterator& operator++() {
            ++it_;
            return *this;
        }
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++it_;
            return tmp;
        }

        [[nodiscard]] BackingIt backing() const { return it_; }

        friend bool operator==(const Iterator& a, const Iterator& b) = default;

    private:
        BackingIt it_{};
    };

    using iterator = Iterator<typename Storage::iterator, value_type>;
    using const_iterator = Iterator<typename Storage::const_iterator, const value_type>;

    Table() noexcept = default;

    // Builds from a `{{k, v}, {k, v}, ...}` list. Throws if a key appears more than once.
    Table(std::initializer_list<value_type> entries) {
        for (const auto& entry : entries) {
            const bool inserted = insert(entry).second;
            TT_FATAL(inserted, "Table: duplicate key in initializer list");
        }
    }

    // Builds from any range of entries convertible to value_type (e.g. another
    // Table, a std::map / std::unordered_map, or a std::vector / std::span of
    // pairs). Each entry is copied in. Throws if a key appears more than once.
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>, value_type>
    explicit Table(const R& entries) {
        for (const auto& entry : entries) {
            const bool inserted = insert(entry).second;
            TT_FATAL(inserted, "Table: duplicate key in input range");
        }
    }

    // Iterates the entries as std::pair<const K, V>.
    [[nodiscard]] iterator begin() { return iterator{entries_.begin()}; }
    [[nodiscard]] iterator end() { return iterator{entries_.end()}; }
    [[nodiscard]] const_iterator begin() const { return const_iterator{entries_.begin()}; }
    [[nodiscard]] const_iterator end() const { return const_iterator{entries_.end()}; }
    [[nodiscard]] const_iterator cbegin() const { return const_iterator{entries_.cbegin()}; }
    [[nodiscard]] const_iterator cend() const { return const_iterator{entries_.cend()}; }

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] size_type size() const noexcept { return entries_.size(); }
    void clear() noexcept {
        entries_.clear();
        index_.clear();
    }

    // Returns an iterator to the entry for `key`, or end() if the key is absent.
    [[nodiscard]] iterator find(const K& key) { return iterator{entries_.begin() + lookup(key).position}; }
    [[nodiscard]] const_iterator find(const K& key) const {
        return const_iterator{entries_.begin() + lookup(key).position};
    }

    // Returns true if `key` is present in the table, false otherwise.
    [[nodiscard]] bool contains(const K& key) const { return find(key) != end(); }

    // Inserts an entry only if its key is absent. Returns {iterator to the entry,
    // true} when inserted, or {iterator to the existing entry, false} otherwise.
    std::pair<iterator, bool> insert(const value_type& entry) {
        const Lookup found = lookup(entry.first);
        if (found.position != entries_.size()) {
            return {iterator{entries_.begin() + found.position}, false};
        }
        return {append(found, entry), true};
    }
    std::pair<iterator, bool> insert(value_type&& entry) {
        const Lookup found = lookup(entry.first);
        if (found.position != entries_.size()) {
            return {iterator{entries_.begin() + found.position}, false};
        }
        return {append(found, std::move(entry)), true};
    }

    // Removes the entry for `key` if present. Returns the number of entries removed
    // (0 or 1). Invalidates iterators at or after the removed entry.
    size_type erase(const K& key) {
        const Lookup found = lookup(key);
        if (found.position == entries_.size()) {
            return 0;
        }
        entries_.erase(entries_.begin() + found.position);
        index_ = make_index(entries_, index_.size());
        return 1;
    }

    // Returns a reference to the value for `key`, inserting a default-constructed
    // value if the key is absent (so `table[key] = value;` works). Overwrite an
    // existing value by assigning through the reference. (Requires V to be
    // default-constructible.)
    V& operator[](const K& key) {
        const Lookup found = lookup(key);
        if (found.position != entries_.size()) {
            return entries_[found.position].second;
        }
        return append(found, key, V{})->second;
    }

    // Builds a value_type from `args` and inserts it only if its key is absent.
    // Returns {iterator, true} when inserted, else {iterator to existing, false}.
    template <typename... Args>
        requires std::constructible_from<value_type, Args...>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert(value_type(std::forward<Args>(args)...));
    }

    // Looks `key` up without inserting: returns an optional reference to its
    // value, or an empty optional_reference if the key is absent. Never throws.
    // Usage: `if (auto v = table.get(key)) { use(*v); }`.
    [[nodiscard]] ttsl::optional_reference<V> get(const K& key) noexcept {
        if (auto it = find(key); it != end()) {
            return it->second;
        }
        return std::nullopt;
    }
    [[nodiscard]] ttsl::optional_reference<const V> get(const K& key) const noexcept {
        if (auto it = find(key); it != end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Equal iff both hold the same key/value pairs (iteration order is ignored).
    [[nodiscard]] friend bool operator==(const Table& a, const Table& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (const auto& [key, value] : a.entries_) {
            const auto it = b.find(key);
            if (it == b.end() || !(it->second == value)) {
                return false;
            }
        }
        return true;
    }

private:
    // The result of looking a key up in the table.
    struct Lookup {
        // The position of the key in entries_, or entries_.size() if the key is absent.
        size_type position;
        // The index slot that refers to the key, or the empty slot where the key would go.
        std::size_t slot;
    };

    // Returns where `key` is in the table.
    [[nodiscard]] Lookup lookup(const K& key) const {
        if (index_.empty()) {
            return {entries_.size(), 0};
        }
        const std::size_t mask = index_.size() - 1;
        for (std::size_t slot = home_slot(key, index_.size());; slot = (slot + 1) & mask) {
            const std::uint32_t value = index_[slot];
            if (value == 0) {
                return {entries_.size(), slot};
            }
            if (entries_[value - 1].first == key) {
                return {value - 1, slot};
            }
        }
    }

    // Appends an entry built from `args` and returns an iterator to it.
    // precondition: `absent` is the result of lookup() for the entry's key, and that key is absent.
    template <typename... Args>
    iterator append(const Lookup& absent, Args&&... args) {
        TT_ASSERT(absent.position == entries_.size());
        entries_.emplace_back(std::forward<Args>(args)...);
        if (2 * entries_.size() > index_.size()) {
            index_ = make_index(entries_, std::max(kMinIndexSlots, 2 * index_.size()));
        } else {
            index_[absent.slot] = static_cast<std::uint32_t>(entries_.size());
        }
        return iterator{std::prev(entries_.end())};
    }

    // Returns the slot where a search for `key` starts, in an index of `slot_count` slots.
    // precondition: slot_count is a power of two, at least 2.
    [[nodiscard]] static std::size_t home_slot(const K& key, std::size_t slot_count) {
        constexpr std::uint64_t kFibonacciMultiplier = 0x9E3779B97F4A7C15ULL;
        const int shift = 64 - std::countr_zero(slot_count);
        return static_cast<std::size_t>(
            (static_cast<std::uint64_t>(std::hash<K>{}(key)) * kFibonacciMultiplier) >> shift);
    }

    // Returns an index of `slot_count` slots that refers to every entry of `entries`.
    // precondition: slot_count is a power of two, and is larger than entries.size().
    [[nodiscard]] static Index make_index(const Storage& entries, std::size_t slot_count) {
        Index index(slot_count, 0);
        const std::size_t mask = slot_count - 1;
        for (size_type position = 0; position < entries.size(); ++position) {
            std::size_t slot = home_slot(entries[position].first, slot_count);
            while (index[slot] != 0) {
                slot = (slot + 1) & mask;
            }
            index[slot] = static_cast<std::uint32_t>(position + 1);
        }
        return index;
    }

    Storage entries_;
    // invariant: index_ refers to every entry of entries_, and is at most half full.
    // invariant: index_ is empty only when entries_ is empty.
    Index index_;
};

}  // namespace tt::tt_metal::experimental

// Makes Table hashable via std::hash, and therefore via ttsl::hash, which dispatches to it. Like
// operator==, the result ignores the entries' storage order. A translation unit that hashes a Table
// must include <tt_stl/reflection.hpp> so this specialization can be instantiated.
template <typename K, typename V>
struct std::hash<tt::tt_metal::experimental::Table<K, V>> {
    std::size_t operator()(const tt::tt_metal::experimental::Table<K, V>& table) const {
        using ttsl::hash::hash_t;

        // Hash each entry on its own, recursing through K and V via ttsl::hash.
        std::vector<hash_t> entry_hashes;
        entry_hashes.reserve(table.size());
        for (const auto& entry : table) {
            entry_hashes.push_back(ttsl::hash::hash_objects(hash_t{0}, entry));
        }

        // Fold the per-entry hashes in sorted order, so the result is independent of the entries'
        // storage order — consistent with operator==, which ignores order.
        std::ranges::sort(entry_hashes);
        return std::accumulate(
            entry_hashes.begin(), entry_hashes.end(), std::size_t{0}, [](std::size_t seed, hash_t entry_hash) {
                ttsl::hash::hash_combine(seed, entry_hash);
                return seed;
            });
    }
};
