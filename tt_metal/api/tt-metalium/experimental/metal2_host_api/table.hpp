// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>              // TT_FATAL (duplicate-key check in constructors)
#include <tt_stl/optional_reference.hpp>  // get() return type

namespace tt::tt_metal::experimental {

// A key -> value map with unique keys.
//
//  - Each key appears at most once.
//  - Iteration order is unspecified.
//  - Iterates as std::pair<const K, mapped>, so keys can't be mutated through an
//    iterator. Structured bindings (`for (const auto& [k, v] : table)`) and
//    std::map / std::unordered_map range construction
//    (`std::map<K, V>(t.begin(), t.end())`) work directly.
//  - The interface mirrors std::map: find() / get() look a key up, insert() and
//    emplace() add an entry only if the key is absent, and operator[] inserts a
//    key or overwrites the existing value.
//  - V may be an object type or an lvalue reference type (e.g. Table<K, T&>).
//    Reference values are stored as a rebindable std::reference_wrapper, so the
//    `mapped` exposed by iteration is that wrapper; get() unwraps it and returns
//    an optional_reference to the referent; and operator[] is unavailable (a
//    reference can't be default-inserted).
template <typename K, typename V>
class Table {
public:
    using key_type = K;
    using mapped_type = V;

private:
    static constexpr bool kMappedIsReference = std::is_reference_v<V>;

    // The referent type (V itself for value mappeds; the referred-to type, which
    // may be const, for reference mappeds).
    using referent_type = std::remove_reference_t<V>;

    // A reference mapped is stored as a rebindable std::reference_wrapper so the
    // backing vector element stays a regular (assignable, growable) object; a
    // value mapped is stored as-is.
    using stored_mapped_type = std::conditional_t<kMappedIsReference, std::reference_wrapper<referent_type>, V>;

    // Entries are stored as a mutable std::pair<K, stored_mapped_type> (so the
    // backing vector stays easy to grow, erase, and assign), but exposed through
    // the iterators below as the const-key value_type.
    using Storage = std::vector<std::pair<K, stored_mapped_type>>;

public:
    using value_type = std::pair<const K, stored_mapped_type>;
    using size_type = std::size_t;

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
    void clear() noexcept { entries_.clear(); }

    // Returns an iterator to the entry for `key`, or end() if the key is absent.
    [[nodiscard]] iterator find(const K& key) {
        return iterator{std::ranges::find(entries_, key, &Storage::value_type::first)};
    }
    [[nodiscard]] const_iterator find(const K& key) const {
        return const_iterator{std::ranges::find(entries_, key, &Storage::value_type::first)};
    }

    // Inserts an entry only if its key is absent. Returns {iterator to the entry,
    // true} when inserted, or {iterator to the existing entry, false} otherwise.
    std::pair<iterator, bool> insert(const value_type& entry) {
        if (auto it = find(entry.first); it != end()) {
            return {it, false};
        }
        entries_.push_back(entry);
        return {iterator{std::prev(entries_.end())}, true};
    }
    std::pair<iterator, bool> insert(value_type&& entry) {
        if (auto it = find(entry.first); it != end()) {
            return {it, false};
        }
        entries_.push_back(std::move(entry));
        return {iterator{std::prev(entries_.end())}, true};
    }

    // Removes the entry for `key` if present. Returns the number of entries removed
    // (0 or 1). Invalidates iterators at or after the removed entry.
    size_type erase(const K& key) {
        auto it = find(key);
        if (it == end()) {
            return 0;
        }
        entries_.erase(it.backing());
        return 1;
    }

    // Returns a reference to the value for `key`, inserting a default-constructed
    // value if the key is absent (so `table[key] = value;` works). Overwrite an
    // existing value by assigning through the reference. (Requires V to be a
    // default-constructible object type; unavailable when V is a reference.)
    V& operator[](const K& key)
        requires(!kMappedIsReference)
    {
        if (auto it = find(key); it != end()) {
            return it->second;
        }
        return insert(value_type{key, V{}}).first->second;
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
    // For reference mappeds the reference_wrapper is unwrapped, so get() yields an
    // optional_reference to the referent (whose const-ness comes from V, not from
    // the const-ness of the table).
    [[nodiscard]] ttsl::optional_reference<referent_type> get(const K& key) noexcept {
        if (auto it = find(key); it != end()) {
            if constexpr (kMappedIsReference) {
                return it->second.get();
            } else {
                return it->second;
            }
        }
        return std::nullopt;
    }
    [[nodiscard]] auto get(const K& key) const noexcept {
        using Result = std::conditional_t<
            kMappedIsReference,
            ttsl::optional_reference<referent_type>,
            ttsl::optional_reference<const V>>;
        if (auto it = find(key); it != end()) {
            if constexpr (kMappedIsReference) {
                return Result{it->second.get()};
            } else {
                return Result{it->second};
            }
        }
        return Result{std::nullopt};
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
    Storage entries_;
};

}  // namespace tt::tt_metal::experimental
