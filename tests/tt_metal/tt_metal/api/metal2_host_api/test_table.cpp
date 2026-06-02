// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API Table<K, V> container (table.hpp).
//
// Pure host-side data-structure tests — no device required. The core behaviors
// are exercised against BOTH storage backings (the default vector-backed base and
// the unordered_map-backed base) via a typed test suite, so the swappable-backing
// contract is validated end to end.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/table.hpp>

namespace {

namespace m2 = tt::tt_metal::experimental;

// A user-defined alternative backing (std::unordered_map), defined here in the test
// rather than shipped in the public header. Plugging it into Table's third template
// parameter exercises the swappable-backing extensibility point.
template <typename K, typename V>
class UnorderedMapBackedTableBase {
    using Storage = std::unordered_map<K, V>;

public:
    using key_type = K;
    using mapped_type = V;
    using value_type = typename Storage::value_type;
    using size_type = std::size_t;
    using iterator = typename Storage::iterator;
    using const_iterator = typename Storage::const_iterator;

    iterator begin() { return entries_.begin(); }
    iterator end() { return entries_.end(); }
    const_iterator begin() const { return entries_.begin(); }
    const_iterator end() const { return entries_.end(); }
    const_iterator cbegin() const { return entries_.cbegin(); }
    const_iterator cend() const { return entries_.cend(); }

    bool empty() const noexcept { return entries_.empty(); }
    size_type size() const noexcept { return entries_.size(); }
    void clear() noexcept { entries_.clear(); }

    iterator find(const K& key) { return entries_.find(key); }
    const_iterator find(const K& key) const { return entries_.find(key); }

    std::pair<iterator, bool> insert(const value_type& entry) { return entries_.insert(entry); }
    std::pair<iterator, bool> insert(value_type&& entry) { return entries_.insert(std::move(entry)); }

    // Order-independent equality via std::unordered_map::operator== (the more
    // optimal implementation this backing exists to demonstrate).
    [[nodiscard]] bool operator==(const UnorderedMapBackedTableBase& other) const { return entries_ == other.entries_; }

private:
    Storage entries_;
};

// The two backings under test, both presenting as Table<std::string, int>.
using VectorBacked = m2::Table<std::string, int>;
using UnorderedBacked = m2::Table<std::string, int, UnorderedMapBackedTableBase<std::string, int>>;

template <typename TableT>
class TableTest : public ::testing::Test {};

using Backings = ::testing::Types<VectorBacked, UnorderedBacked>;
TYPED_TEST_SUITE(TableTest, Backings);

// ---- construction / emptiness ------------------------------------------------

TYPED_TEST(TableTest, DefaultConstructedIsEmpty) {
    TypeParam t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.begin(), t.end());
}

TYPED_TEST(TableTest, InitializerListConstruction) {
    TypeParam t{{"a", 1}, {"b", 2}, {"c", 3}};
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
    EXPECT_EQ(*t.get("c"), 3);
}

TYPED_TEST(TableTest, SpanConstruction) {
    std::vector<typename TypeParam::value_type> entries{{"a", 1}, {"b", 2}};
    std::span<const typename TypeParam::value_type> entries_view(entries);
    TypeParam t(entries_view);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
}

// ---- operator[] --------------------------------------------------------------

TYPED_TEST(TableTest, SubscriptInsertsDefaultOnMiss) {
    TypeParam t;
    EXPECT_EQ(t["a"], 0);  // default-constructed int
    EXPECT_EQ(t.size(), 1u);
}

TYPED_TEST(TableTest, SubscriptAssignThenOverwrite) {
    TypeParam t;
    t["a"] = 1;
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
    t["a"] = 2;  // overwrite, no new entry
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 2);
}

TYPED_TEST(TableTest, SubscriptReadsExistingWithoutOverwriting) {
    TypeParam t{{"a", 5}};
    EXPECT_EQ(t["a"], 5);
    EXPECT_EQ(t.size(), 1u);
}

// ---- insert (insert-if-absent) -----------------------------------------------

TYPED_TEST(TableTest, InsertNewKey) {
    TypeParam t;
    auto [it, inserted] = t.insert({"a", 1});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

TYPED_TEST(TableTest, InsertExistingKeyDoesNotOverwrite) {
    TypeParam t{{"a", 1}};
    auto [it, inserted] = t.insert({"a", 2});
    EXPECT_FALSE(inserted);
    EXPECT_EQ(it->second, 1);  // unchanged
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
}

// ---- emplace (insert-if-absent, in place) ------------------------------------

TYPED_TEST(TableTest, EmplaceNewKey) {
    TypeParam t;
    auto [it, inserted] = t.emplace("a", 1);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

TYPED_TEST(TableTest, EmplaceExistingKeyDoesNotOverwrite) {
    TypeParam t;
    t.emplace("a", 1);
    auto [it, inserted] = t.emplace("a", 2);
    EXPECT_FALSE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

// ---- get ---------------------------------------------------------------------

TYPED_TEST(TableTest, GetPresentAndAbsent) {
    TypeParam t{{"a", 7}};
    auto present = t.get("a");
    ASSERT_TRUE(present);
    EXPECT_EQ(*present, 7);
    EXPECT_FALSE(t.get("missing"));
}

TYPED_TEST(TableTest, GetMutatesThroughReference) {
    TypeParam t{{"a", 1}};
    *t.get("a") = 42;
    EXPECT_EQ(*t.get("a"), 42);
}

TYPED_TEST(TableTest, GetOnConstTable) {
    const TypeParam t{{"a", 3}};
    auto v = t.get("a");
    ASSERT_TRUE(v);
    EXPECT_EQ(*v, 3);
    EXPECT_FALSE(t.get("x"));
}

// ---- find --------------------------------------------------------------------

TYPED_TEST(TableTest, FindHitAndMiss) {
    TypeParam t{{"a", 1}};
    auto it = t.find("a");
    ASSERT_NE(it, t.end());
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.find("missing"), t.end());
}

// ---- size / clear ------------------------------------------------------------

TYPED_TEST(TableTest, Clear) {
    TypeParam t{{"a", 1}, {"b", 2}};
    EXPECT_EQ(t.size(), 2u);
    t.clear();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

// ---- iteration ---------------------------------------------------------------

TYPED_TEST(TableTest, IterationVisitsAllEntries) {
    TypeParam t{{"a", 1}, {"b", 2}, {"c", 3}};
    std::map<std::string, int> seen;
    for (const auto& [k, v] : t) {
        seen[k] = v;
    }
    EXPECT_EQ(seen, (std::map<std::string, int>{{"a", 1}, {"b", 2}, {"c", 3}}));
}

TYPED_TEST(TableTest, RangeConstructsStdMap) {
    TypeParam t{{"a", 1}, {"b", 2}};
    std::map<std::string, int> m(t.begin(), t.end());
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m.at("a"), 1);
    EXPECT_EQ(m.at("b"), 2);
}

// ---- equality (order independent) --------------------------------------------

TYPED_TEST(TableTest, EqualityIsOrderIndependent) {
    TypeParam a{{"a", 1}, {"b", 2}};
    TypeParam b;
    b["b"] = 2;
    b["a"] = 1;  // inserted in the opposite order
    EXPECT_EQ(a, b);
}

TYPED_TEST(TableTest, InequalityCases) {
    TypeParam base{{"a", 1}, {"b", 2}};
    EXPECT_NE(base, (TypeParam{{"a", 1}}));             // different size
    EXPECT_NE(base, (TypeParam{{"a", 1}, {"b", 99}}));  // different value
    EXPECT_NE(base, (TypeParam{{"a", 1}, {"z", 2}}));   // different key
}

// ---- duplicate-key rejection at construction ---------------------------------

TYPED_TEST(TableTest, InitializerListDuplicateThrows) {
    auto build = [] { return TypeParam{{"a", 1}, {"a", 2}}; };
    EXPECT_ANY_THROW(build());
}

TYPED_TEST(TableTest, SpanDuplicateThrows) {
    std::vector<typename TypeParam::value_type> dup{{"a", 1}, {"a", 2}};
    std::span<const typename TypeParam::value_type> sp(dup);
    EXPECT_ANY_THROW(TypeParam{sp});
}

}  // namespace

// ---- genericity over other K/V types + move semantics (default backing) ------

namespace {

TEST(TableMiscTest, IntKeyStringValue) {
    m2::Table<int, std::string> t;
    t[1] = "one";
    t.emplace(2, "two");
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get(1), "one");
    EXPECT_EQ(*t.get(2), "two");
    EXPECT_FALSE(t.get(3));
}

}  // namespace
