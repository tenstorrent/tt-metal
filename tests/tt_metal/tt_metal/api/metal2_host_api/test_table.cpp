// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API Table<K, V> container (table.hpp).
//
// Pure host-side data-structure tests — no device required.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <concepts>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt_stl/reflection.hpp>

namespace {

namespace m2 = tt::tt_metal::experimental;

using StrIntTable = m2::Table<std::string, int>;

// ---- iterator contract (compile-time) ----------------------------------------

static_assert(std::forward_iterator<StrIntTable::iterator>, "Table::iterator must model std::forward_iterator");
static_assert(
    std::forward_iterator<StrIntTable::const_iterator>, "Table::const_iterator must model std::forward_iterator");
static_assert(
    std::convertible_to<StrIntTable::iterator, StrIntTable::const_iterator>,
    "Table::iterator must implicitly convert to const_iterator");
static_assert(
    !std::convertible_to<StrIntTable::const_iterator, StrIntTable::iterator>,
    "Table::const_iterator must not convert to mutable iterator");

// ---- construction / emptiness ------------------------------------------------

TEST(TableTest, DefaultConstructedIsEmpty) {
    StrIntTable t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.begin(), t.end());
}

TEST(TableTest, InitializerListConstruction) {
    StrIntTable t{{"a", 1}, {"b", 2}, {"c", 3}};
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
    EXPECT_EQ(*t.get("c"), 3);
}

TEST(TableTest, SpanConstruction) {
    std::vector<StrIntTable::value_type> entries{{"a", 1}, {"b", 2}};
    std::span<const StrIntTable::value_type> entries_view(entries);
    StrIntTable t(entries_view);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
}

TEST(TableTest, RangeConstructionFromUnorderedMap) {
    std::unordered_map<std::string, int> src{{"a", 1}, {"b", 2}, {"c", 3}};
    StrIntTable t(src);
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
    EXPECT_EQ(*t.get("c"), 3);
}

TEST(TableTest, RangeConstructionFromMap) {
    std::map<std::string, int> src{{"a", 1}, {"b", 2}};
    StrIntTable t(src);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
}

TEST(TableTest, RangeConstructionFromVectorOfPlainPairs) {
    std::vector<std::pair<std::string, int>> src{{"a", 1}, {"b", 2}};  // non-const key
    StrIntTable t(src);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
}

TEST(TableTest, RangeConstructionDuplicateThrows) {
    std::vector<std::pair<std::string, int>> src{{"a", 1}, {"a", 2}};
    EXPECT_ANY_THROW(StrIntTable{src});
}

// ---- operator[] --------------------------------------------------------------

TEST(TableTest, SubscriptInsertsDefaultOnMiss) {
    StrIntTable t;
    EXPECT_EQ(t["a"], 0);  // default-constructed int
    EXPECT_EQ(t.size(), 1u);
}

TEST(TableTest, SubscriptAssignThenOverwrite) {
    StrIntTable t;
    t["a"] = 1;
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
    t["a"] = 2;  // overwrite, no new entry
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 2);
}

TEST(TableTest, SubscriptReadsExistingWithoutOverwriting) {
    StrIntTable t{{"a", 5}};
    EXPECT_EQ(t["a"], 5);
    EXPECT_EQ(t.size(), 1u);
}

// ---- insert (insert-if-absent) -----------------------------------------------

TEST(TableTest, InsertNewKey) {
    StrIntTable t;
    auto [it, inserted] = t.insert({"a", 1});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

TEST(TableTest, InsertExistingKeyDoesNotOverwrite) {
    StrIntTable t{{"a", 1}};
    auto [it, inserted] = t.insert({"a", 2});
    EXPECT_FALSE(inserted);
    EXPECT_EQ(it->second, 1);  // unchanged
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
}

// ---- erase -------------------------------------------------------------------

TEST(TableTest, EraseRemovesPresentKey) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_EQ(t.erase("a"), 1u);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_FALSE(t.get("a"));
    EXPECT_EQ(*t.get("b"), 2);  // surviving entry intact
}

TEST(TableTest, EraseAbsentKeyIsNoop) {
    StrIntTable t{{"a", 1}};
    EXPECT_EQ(t.erase("missing"), 0u);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
}

TEST(TableTest, EraseThenReinsert) {
    StrIntTable t{{"a", 1}};
    EXPECT_EQ(t.erase("a"), 1u);
    EXPECT_TRUE(t.empty());
    auto [it, inserted] = t.insert({"a", 2});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(*t.get("a"), 2);
}

// ---- emplace (insert-if-absent, in place) ------------------------------------

TEST(TableTest, EmplaceNewKey) {
    StrIntTable t;
    auto [it, inserted] = t.emplace("a", 1);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

TEST(TableTest, EmplaceExistingKeyDoesNotOverwrite) {
    StrIntTable t;
    t.emplace("a", 1);
    auto [it, inserted] = t.emplace("a", 2);
    EXPECT_FALSE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

// ---- get ---------------------------------------------------------------------

TEST(TableTest, GetPresentAndAbsent) {
    StrIntTable t{{"a", 7}};
    auto present = t.get("a");
    ASSERT_TRUE(present);
    EXPECT_EQ(*present, 7);
    EXPECT_FALSE(t.get("missing"));
}

TEST(TableTest, GetMutatesThroughReference) {
    StrIntTable t{{"a", 1}};
    *t.get("a") = 42;
    EXPECT_EQ(*t.get("a"), 42);
}

TEST(TableTest, GetOnConstTable) {
    const StrIntTable t{{"a", 3}};
    auto v = t.get("a");
    ASSERT_TRUE(v);
    EXPECT_EQ(*v, 3);
    EXPECT_FALSE(t.get("x"));
}

// ---- find --------------------------------------------------------------------

TEST(TableTest, FindHitAndMiss) {
    StrIntTable t{{"a", 1}};
    auto it = t.find("a");
    ASSERT_NE(it, t.end());
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.find("missing"), t.end());
}

// ---- size / clear ------------------------------------------------------------

TEST(TableTest, Clear) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_EQ(t.size(), 2u);
    t.clear();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

// ---- iteration ---------------------------------------------------------------

TEST(TableTest, IterationVisitsAllEntries) {
    StrIntTable t{{"a", 1}, {"b", 2}, {"c", 3}};
    std::map<std::string, int> seen;
    for (const auto& [k, v] : t) {
        seen[k] = v;
    }
    EXPECT_EQ(seen, (std::map<std::string, int>{{"a", 1}, {"b", 2}, {"c", 3}}));
}

TEST(TableTest, RangeConstructsStdMap) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    std::map<std::string, int> m(t.begin(), t.end());
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m.at("a"), 1);
    EXPECT_EQ(m.at("b"), 2);
}

// ---- equality (order independent) --------------------------------------------

TEST(TableTest, EqualityIsOrderIndependent) {
    StrIntTable a{{"a", 1}, {"b", 2}};
    StrIntTable b;
    b["b"] = 2;
    b["a"] = 1;  // inserted in the opposite order
    EXPECT_EQ(a, b);
}

TEST(TableTest, InequalityCases) {
    StrIntTable base{{"a", 1}, {"b", 2}};
    EXPECT_NE(base, (StrIntTable{{"a", 1}}));             // different size
    EXPECT_NE(base, (StrIntTable{{"a", 1}, {"b", 99}}));  // different value
    EXPECT_NE(base, (StrIntTable{{"a", 1}, {"z", 2}}));   // different key
}

// ---- duplicate-key rejection at construction ---------------------------------

TEST(TableTest, InitializerListDuplicateThrows) {
    auto build = [] { return StrIntTable{{"a", 1}, {"a", 2}}; };
    EXPECT_ANY_THROW(build());
}

TEST(TableTest, SpanDuplicateThrows) {
    std::vector<StrIntTable::value_type> dup{{"a", 1}, {"a", 2}};
    std::span<const StrIntTable::value_type> sp(dup);
    EXPECT_ANY_THROW(StrIntTable{sp});
}

// ---- genericity over other K/V types -----------------------------------------

TEST(TableMiscTest, IntKeyStringValue) {
    m2::Table<int, std::string> t;
    t[1] = "one";
    t.emplace(2, "two");
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get(1), "one");
    EXPECT_EQ(*t.get(2), "two");
    EXPECT_FALSE(t.get(3));
}

// ---- contains ----------------------------------------------------------------

TEST(TableTest, ContainsReturnsTrueForPresentKey) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_TRUE(t.contains("a"));
    EXPECT_TRUE(t.contains("b"));
}

TEST(TableTest, ContainsReturnsFalseForAbsentKey) {
    StrIntTable t{{"a", 1}};
    EXPECT_FALSE(t.contains("missing"));
    EXPECT_FALSE(t.contains(""));
}

TEST(TableTest, ContainsOnEmptyTable) {
    StrIntTable t;
    EXPECT_FALSE(t.contains("anything"));
}

TEST(TableTest, ContainsReflectsInsertAndErase) {
    StrIntTable t;
    EXPECT_FALSE(t.contains("a"));
    t.insert({"a", 1});
    EXPECT_TRUE(t.contains("a"));
    t.erase("a");
    EXPECT_FALSE(t.contains("a"));
}

TEST(TableTest, ContainsWorksOnConstTable) {
    const StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_TRUE(t.contains("a"));
    EXPECT_FALSE(t.contains("z"));
}

TEST(TableTest, ContainsDoesNotMutateTable) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    const auto size_before = t.size();
    EXPECT_TRUE(t.contains("a"));
    EXPECT_FALSE(t.contains("missing"));
    EXPECT_EQ(t.size(), size_before);
}

TEST(TableMiscTest, ContainsWithIntKey) {
    m2::Table<int, std::string> t;
    t[1] = "one";
    t.emplace(2, "two");
    EXPECT_TRUE(t.contains(1));
    EXPECT_TRUE(t.contains(2));
    EXPECT_FALSE(t.contains(3));
    EXPECT_FALSE(t.contains(0));
}

// ---- reflection-based hashing (tt_stl/reflection.hpp) ------------------------

// Canonical hash entry point (matches how reflected structs hash themselves).
template <typename T>
ttsl::hash::hash_t hash_of(const T& object) {
    return ttsl::hash::hash_objects_with_default_seed(object);
}

// A reflected struct that holds a Table, to exercise Table as a nested field
// (its real use in the Metal 2.0 host API). Must live at namespace scope because
// a local class can't have the static attribute_names member.
struct TaggedTable {
    int tag = 0;
    StrIntTable table;

    static constexpr auto attribute_names = std::forward_as_tuple("tag", "table");
    auto attribute_values() const { return std::forward_as_tuple(tag, table); }
};

TEST(TableHashTest, HashIsDeterministic) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_EQ(hash_of(t), hash_of(t));  // same object hashes identically every time
}

TEST(TableHashTest, EqualTablesHashEqual) {
    StrIntTable a{{"a", 1}, {"b", 2}, {"c", 3}};
    StrIntTable b{{"a", 1}, {"b", 2}, {"c", 3}};
    ASSERT_EQ(a, b);
    EXPECT_EQ(hash_of(a), hash_of(b));
}

TEST(TableHashTest, EmptyTables) {
    EXPECT_EQ(hash_of(StrIntTable{}), hash_of(StrIntTable{}));
    EXPECT_NE(hash_of(StrIntTable{}), hash_of(StrIntTable{{"a", 1}}));
}

TEST(TableHashTest, DifferentValueChangesHash) {
    StrIntTable base{{"a", 1}, {"b", 2}};
    StrIntTable diff{{"a", 1}, {"b", 99}};
    EXPECT_NE(hash_of(base), hash_of(diff));
}

TEST(TableHashTest, DifferentKeyChangesHash) {
    EXPECT_NE(hash_of(StrIntTable{{"a", 1}}), hash_of(StrIntTable{{"z", 1}}));
}

TEST(TableHashTest, DifferentSizeChangesHash) {
    StrIntTable small{{"a", 1}};
    StrIntTable big{{"a", 1}, {"b", 2}};
    EXPECT_NE(hash_of(small), hash_of(big));
}

TEST(TableHashTest, HashIsOrderIndependent) {
    // Two tables that compare equal but were built in opposite order must hash equally:
    // std::hash<Table> folds the per-entry hashes in sorted order, so insertion order drops out
    // (consistent with operator==, which ignores order).
    StrIntTable a;
    a["x"] = 1;
    a["y"] = 2;
    StrIntTable b;
    b["y"] = 2;
    b["x"] = 1;
    ASSERT_EQ(a, b);
    EXPECT_EQ(hash_of(a), hash_of(b));
}

TEST(TableHashTest, NonStringKeyValueHashes) {
    // Reflection recurses into K and V, so any hashable key/value type works.
    m2::Table<int, std::string> a;
    a[1] = "one";
    a[2] = "two";
    m2::Table<int, std::string> same;
    same[1] = "one";
    same[2] = "two";
    EXPECT_EQ(hash_of(a), hash_of(same));
    EXPECT_NE(hash_of(a), hash_of(m2::Table<int, std::string>{{1, "one"}}));
}

TEST(TableHashTest, NestedInReflectedStruct) {
    TaggedTable a{.tag = 7, .table = {{"a", 1}, {"b", 2}}};
    TaggedTable b{.tag = 7, .table = {{"a", 1}, {"b", 2}}};
    EXPECT_EQ(hash_of(a), hash_of(b));

    TaggedTable diff_tag{.tag = 8, .table = {{"a", 1}, {"b", 2}}};
    EXPECT_NE(hash_of(a), hash_of(diff_tag));

    TaggedTable diff_table{.tag = 7, .table = {{"a", 1}, {"b", 3}}};
    EXPECT_NE(hash_of(a), hash_of(diff_table));

    // The Table is reachable as a named attribute of the enclosing reflected type.
    EXPECT_STREQ(std::get<1>(TaggedTable::attribute_names), "table");
}

}  // namespace
