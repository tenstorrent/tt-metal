// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>              // TT_FATAL (duplicate-key check in constructors)
#include <tt_stl/optional_reference.hpp>  // get() return type

namespace tt::tt_metal::experimental {

namespace table_detail {

// The default storage backend for Table: a vector of key/value pairs providing
// iteration, lookup (find), and insert-if-absent. Plug a different type into
// Table's third template parameter to change the backing representation.
template <typename K, typename V>
class VectorBackedTableBase {
    using Storage = std::vector<std::pair<K, V>>;

public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<K, V>;
    using size_type = std::size_t;

    using iterator = typename Storage::iterator;
    using const_iterator = typename Storage::const_iterator;

    [[nodiscard]] iterator begin() { return entries_.begin(); }
    [[nodiscard]] iterator end() { return entries_.end(); }
    [[nodiscard]] const_iterator begin() const { return entries_.begin(); }
    [[nodiscard]] const_iterator end() const { return entries_.end(); }
    [[nodiscard]] const_iterator cbegin() const { return entries_.cbegin(); }
    [[nodiscard]] const_iterator cend() const { return entries_.cend(); }

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] size_type size() const noexcept { return entries_.size(); }
    void clear() noexcept { entries_.clear(); }

    [[nodiscard]] iterator find(const K& key) {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->first == key) {
                return it;
            }
        }
        return entries_.end();
    }
    [[nodiscard]] const_iterator find(const K& key) const {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->first == key) {
                return it;
            }
        }
        return entries_.end();
    }

    std::pair<iterator, bool> insert(const value_type& entry) {
        if (auto it = find(entry.first); it != end()) {
            return {it, false};
        }
        entries_.push_back(entry);
        return {std::prev(entries_.end()), true};
    }
    std::pair<iterator, bool> insert(value_type&& entry) {
        if (auto it = find(entry.first); it != end()) {
            return {it, false};
        }
        entries_.push_back(std::move(entry));
        return {std::prev(entries_.end()), true};
    }

private:
    Storage entries_;
};

}  // namespace table_detail

// A key -> value map with unique keys.
//
//  - Each key appears at most once.
//  - Iteration order is unspecified.
//  - Iterates as std::pair<K, V>, so structured bindings
//    (`for (const auto& [k, v] : table)`) and std::map / std::unordered_map
//    range construction (`std::map<K, V>(t.begin(), t.end())`) work directly.
//  - The interface mirrors std::map: find() / get() look a key up, insert() and
//    emplace() add an entry only if the key is absent, and operator[] inserts a
//    key or overwrites the existing value.
template <typename K, typename V, typename TableBase = table_detail::VectorBackedTableBase<K, V>>
class Table : private TableBase {
public:
    using typename TableBase::const_iterator;
    using typename TableBase::iterator;
    using typename TableBase::key_type;
    using typename TableBase::mapped_type;
    using typename TableBase::size_type;
    using typename TableBase::value_type;

    // Iterates the entries as std::pair<K, V>.
    using TableBase::begin;
    using TableBase::cbegin;
    using TableBase::cend;
    using TableBase::end;

    using TableBase::clear;
    using TableBase::empty;
    using TableBase::size;

    // Returns an iterator to the entry for `key`, or end() if the key is absent.
    using TableBase::find;

    // Inserts an entry only if its key is absent. Returns {iterator to the entry,
    // true} when inserted, or {iterator to the existing entry, false} otherwise.
    using TableBase::insert;

    Table() = default;

    // Builds from a `{{k, v}, {k, v}, ...}` list. Throws if a key appears more than once.
    Table(std::initializer_list<value_type> entries) {
        for (const auto& entry : entries) {
            const bool inserted = this->insert(entry).second;
            TT_FATAL(inserted, "Table: duplicate key in initializer list");
        }
    }

    // Builds from a span of entries (each is copied in). Throws if a key appears
    // more than once.
    explicit Table(std::span<const value_type> entries) {
        for (const auto& entry : entries) {
            const bool inserted = this->insert(entry).second;
            TT_FATAL(inserted, "Table: duplicate key in input span");
        }
    }

    // Returns a reference to the value for `key`, inserting a default-constructed
    // value if the key is absent (so `table[key] = value;` works). The K&& form
    // avoids copying the key on insert. (Requires V to be default-constructible.)
    V& operator[](const K& key) {
        if (auto it = this->find(key); it != this->end()) {
            return it->second;
        }
        return this->insert(value_type{key, V{}}).first->second;
    }

    // Builds a value_type from `args` and inserts it only if its key is absent.
    // Returns {iterator, true} when inserted, else {iterator to existing, false}.
    template <typename... Args>
        requires std::constructible_from<value_type, Args...>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return this->insert(value_type(std::forward<Args>(args)...));
    }

    // Looks `key` up without inserting: returns an optional reference to its
    // value, or an empty optional_reference if the key is absent. Never throws.
    // Usage: `if (auto v = table.get(key)) { use(*v); }`.
    [[nodiscard]] ttsl::optional_reference<V> get(const K& key) noexcept {
        if (auto it = this->find(key); it != this->end()) {
            return it->second;
        }
        return std::nullopt;
    }
    [[nodiscard]] ttsl::optional_reference<const V> get(const K& key) const noexcept {
        if (auto it = this->find(key); it != this->end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Equal iff both hold the same key/value pairs (iteration order is ignored).
    [[nodiscard]] friend bool operator==(const Table& a, const Table& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (const auto& [key, value] : a) {
            const auto found = b.get(key);
            if (!found || !(*found == value)) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace tt::tt_metal::experimental
