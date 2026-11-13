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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
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

TEST(TableTest, CPU_DefaultConstructedIsEmpty) {
    StrIntTable t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.begin(), t.end());
}

TEST(TableTest, CPU_InitializerListConstruction) {
    StrIntTable t{{"a", 1}, {"b", 2}, {"c", 3}};
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
    EXPECT_EQ(*t.get("c"), 3);
}

TEST(TableTest, CPU_SpanConstruction) {
    std::vector<StrIntTable::value_type> entries{{"a", 1}, {"b", 2}};
    std::span<const StrIntTable::value_type> entries_view(entries);
    StrIntTable t(entries_view);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
}

TEST(TableTest, CPU_RangeConstructionFromUnorderedMap) {
    std::unordered_map<std::string, int> src{{"a", 1}, {"b", 2}, {"c", 3}};
    StrIntTable t(src);
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
    EXPECT_EQ(*t.get("c"), 3);
}

TEST(TableTest, CPU_RangeConstructionFromMap) {
    std::map<std::string, int> src{{"a", 1}, {"b", 2}};
    StrIntTable t(src);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
}

TEST(TableTest, CPU_RangeConstructionFromVectorOfPlainPairs) {
    std::vector<std::pair<std::string, int>> src{{"a", 1}, {"b", 2}};  // non-const key
    StrIntTable t(src);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get("a"), 1);
    EXPECT_EQ(*t.get("b"), 2);
}

TEST(TableTest, CPU_RangeConstructionDuplicateThrows) {
    std::vector<std::pair<std::string, int>> src{{"a", 1}, {"a", 2}};
    EXPECT_ANY_THROW(StrIntTable{src});
}

// ---- operator[] --------------------------------------------------------------

TEST(TableTest, CPU_SubscriptInsertsDefaultOnMiss) {
    StrIntTable t;
    EXPECT_EQ(t["a"], 0);  // default-constructed int
    EXPECT_EQ(t.size(), 1u);
}

TEST(TableTest, CPU_SubscriptAssignThenOverwrite) {
    StrIntTable t;
    t["a"] = 1;
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
    t["a"] = 2;  // overwrite, no new entry
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 2);
}

TEST(TableTest, CPU_SubscriptReadsExistingWithoutOverwriting) {
    StrIntTable t{{"a", 5}};
    EXPECT_EQ(t["a"], 5);
    EXPECT_EQ(t.size(), 1u);
}

// ---- insert (insert-if-absent) -----------------------------------------------

TEST(TableTest, CPU_InsertNewKey) {
    StrIntTable t;
    auto [it, inserted] = t.insert({"a", 1});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

TEST(TableTest, CPU_InsertExistingKeyDoesNotOverwrite) {
    StrIntTable t{{"a", 1}};
    auto [it, inserted] = t.insert({"a", 2});
    EXPECT_FALSE(inserted);
    EXPECT_EQ(it->second, 1);  // unchanged
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
}

// ---- erase -------------------------------------------------------------------

TEST(TableTest, CPU_EraseRemovesPresentKey) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_EQ(t.erase("a"), 1u);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_FALSE(t.get("a"));
    EXPECT_EQ(*t.get("b"), 2);  // surviving entry intact
}

TEST(TableTest, CPU_EraseAbsentKeyIsNoop) {
    StrIntTable t{{"a", 1}};
    EXPECT_EQ(t.erase("missing"), 0u);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.get("a"), 1);
}

TEST(TableTest, CPU_EraseThenReinsert) {
    StrIntTable t{{"a", 1}};
    EXPECT_EQ(t.erase("a"), 1u);
    EXPECT_TRUE(t.empty());
    auto [it, inserted] = t.insert({"a", 2});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(*t.get("a"), 2);
}

// ---- emplace (insert-if-absent, in place) ------------------------------------

TEST(TableTest, CPU_EmplaceNewKey) {
    StrIntTable t;
    auto [it, inserted] = t.emplace("a", 1);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

TEST(TableTest, CPU_EmplaceExistingKeyDoesNotOverwrite) {
    StrIntTable t;
    t.emplace("a", 1);
    auto [it, inserted] = t.emplace("a", 2);
    EXPECT_FALSE(inserted);
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.size(), 1u);
}

// ---- get ---------------------------------------------------------------------

TEST(TableTest, CPU_GetPresentAndAbsent) {
    StrIntTable t{{"a", 7}};
    auto present = t.get("a");
    ASSERT_TRUE(present);
    EXPECT_EQ(*present, 7);
    EXPECT_FALSE(t.get("missing"));
}

TEST(TableTest, CPU_GetMutatesThroughReference) {
    StrIntTable t{{"a", 1}};
    *t.get("a") = 42;
    EXPECT_EQ(*t.get("a"), 42);
}

TEST(TableTest, CPU_GetOnConstTable) {
    const StrIntTable t{{"a", 3}};
    auto v = t.get("a");
    ASSERT_TRUE(v);
    EXPECT_EQ(*v, 3);
    EXPECT_FALSE(t.get("x"));
}

// ---- find --------------------------------------------------------------------

TEST(TableTest, CPU_FindHitAndMiss) {
    StrIntTable t{{"a", 1}};
    auto it = t.find("a");
    ASSERT_NE(it, t.end());
    EXPECT_EQ(it->second, 1);
    EXPECT_EQ(t.find("missing"), t.end());
}

// ---- size / clear ------------------------------------------------------------

TEST(TableTest, CPU_Clear) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_EQ(t.size(), 2u);
    t.clear();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

// ---- iteration ---------------------------------------------------------------

TEST(TableTest, CPU_IterationVisitsAllEntries) {
    StrIntTable t{{"a", 1}, {"b", 2}, {"c", 3}};
    std::map<std::string, int> seen;
    for (const auto& [k, v] : t) {
        seen[k] = v;
    }
    EXPECT_EQ(seen, (std::map<std::string, int>{{"a", 1}, {"b", 2}, {"c", 3}}));
}

TEST(TableTest, CPU_RangeConstructsStdMap) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    std::map<std::string, int> m(t.begin(), t.end());
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m.at("a"), 1);
    EXPECT_EQ(m.at("b"), 2);
}

// ---- equality (order independent) --------------------------------------------

TEST(TableTest, CPU_EqualityIsOrderIndependent) {
    StrIntTable a{{"a", 1}, {"b", 2}};
    StrIntTable b;
    b["b"] = 2;
    b["a"] = 1;  // inserted in the opposite order
    EXPECT_EQ(a, b);
}

TEST(TableTest, CPU_InequalityCases) {
    StrIntTable base{{"a", 1}, {"b", 2}};
    EXPECT_NE(base, (StrIntTable{{"a", 1}}));             // different size
    EXPECT_NE(base, (StrIntTable{{"a", 1}, {"b", 99}}));  // different value
    EXPECT_NE(base, (StrIntTable{{"a", 1}, {"z", 2}}));   // different key
}

// ---- duplicate-key rejection at construction ---------------------------------

TEST(TableTest, CPU_InitializerListDuplicateThrows) {
    auto build = [] { return StrIntTable{{"a", 1}, {"a", 2}}; };
    EXPECT_ANY_THROW(build());
}

TEST(TableTest, CPU_SpanDuplicateThrows) {
    std::vector<StrIntTable::value_type> dup{{"a", 1}, {"a", 2}};
    std::span<const StrIntTable::value_type> sp(dup);
    EXPECT_ANY_THROW(StrIntTable{sp});
}

// ---- genericity over other K/V types -----------------------------------------

TEST(TableMiscTest, CPU_IntKeyStringValue) {
    m2::Table<int, std::string> t;
    t[1] = "one";
    t.emplace(2, "two");
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(*t.get(1), "one");
    EXPECT_EQ(*t.get(2), "two");
    EXPECT_FALSE(t.get(3));
}

// ---- contains ----------------------------------------------------------------

TEST(TableTest, CPU_ContainsReturnsTrueForPresentKey) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_TRUE(t.contains("a"));
    EXPECT_TRUE(t.contains("b"));
}

TEST(TableTest, CPU_ContainsReturnsFalseForAbsentKey) {
    StrIntTable t{{"a", 1}};
    EXPECT_FALSE(t.contains("missing"));
    EXPECT_FALSE(t.contains(""));
}

TEST(TableTest, CPU_ContainsOnEmptyTable) {
    StrIntTable t;
    EXPECT_FALSE(t.contains("anything"));
}

TEST(TableTest, CPU_ContainsReflectsInsertAndErase) {
    StrIntTable t;
    EXPECT_FALSE(t.contains("a"));
    t.insert({"a", 1});
    EXPECT_TRUE(t.contains("a"));
    t.erase("a");
    EXPECT_FALSE(t.contains("a"));
}

TEST(TableTest, CPU_ContainsWorksOnConstTable) {
    const StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_TRUE(t.contains("a"));
    EXPECT_FALSE(t.contains("z"));
}

TEST(TableTest, CPU_ContainsDoesNotMutateTable) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    const auto size_before = t.size();
    EXPECT_TRUE(t.contains("a"));
    EXPECT_FALSE(t.contains("missing"));
    EXPECT_EQ(t.size(), size_before);
}

TEST(TableMiscTest, CPU_ContainsWithIntKey) {
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

TEST(TableHashTest, CPU_HashIsDeterministic) {
    StrIntTable t{{"a", 1}, {"b", 2}};
    EXPECT_EQ(hash_of(t), hash_of(t));  // same object hashes identically every time
}

TEST(TableHashTest, CPU_EqualTablesHashEqual) {
    StrIntTable a{{"a", 1}, {"b", 2}, {"c", 3}};
    StrIntTable b{{"a", 1}, {"b", 2}, {"c", 3}};
    ASSERT_EQ(a, b);
    EXPECT_EQ(hash_of(a), hash_of(b));
}

TEST(TableHashTest, CPU_EmptyTables) {
    EXPECT_EQ(hash_of(StrIntTable{}), hash_of(StrIntTable{}));
    EXPECT_NE(hash_of(StrIntTable{}), hash_of(StrIntTable{{"a", 1}}));
}

TEST(TableHashTest, CPU_DifferentValueChangesHash) {
    StrIntTable base{{"a", 1}, {"b", 2}};
    StrIntTable diff{{"a", 1}, {"b", 99}};
    EXPECT_NE(hash_of(base), hash_of(diff));
}

TEST(TableHashTest, CPU_DifferentKeyChangesHash) {
    EXPECT_NE(hash_of(StrIntTable{{"a", 1}}), hash_of(StrIntTable{{"z", 1}}));
}

TEST(TableHashTest, CPU_DifferentSizeChangesHash) {
    StrIntTable small{{"a", 1}};
    StrIntTable big{{"a", 1}, {"b", 2}};
    EXPECT_NE(hash_of(small), hash_of(big));
}

TEST(TableHashTest, CPU_HashIsOrderIndependent) {
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

TEST(TableHashTest, CPU_NonStringKeyValueHashes) {
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

TEST(TableHashTest, CPU_NestedInReflectedStruct) {
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

// A key type whose keys all have the same std::hash value.
struct CollidingKey {
    int id = 0;
    bool operator==(const CollidingKey&) const = default;
};

// A key type whose std::hash value is chosen by the test.
struct KeyWithChosenHash {
    int id = 0;
    std::size_t hash = 0;
    bool operator==(const KeyWithChosenHash&) const = default;
};

}  // namespace

template <>
struct std::hash<CollidingKey> {
    std::size_t operator()(const CollidingKey& /*key*/) const noexcept { return 0; }
};

template <>
struct std::hash<KeyWithChosenHash> {
    std::size_t operator()(const KeyWithChosenHash& key) const noexcept { return key.hash; }
};

namespace {

// ---- behavior at many sizes --------------------------------------------------

// The parameter is the number of entries that each test adds to the Table.
class TableAtSizeTest : public ::testing::TestWithParam<int> {};

// Sizes on each side of the inline capacity, and sizes that need many growth steps.
INSTANTIATE_TEST_SUITE_P(EntryCounts, TableAtSizeTest, ::testing::Values(1, 8, 9, 16, 17, 100, 1000));

TEST_P(TableAtSizeTest, CPU_FindsEveryAddedKey) {
    // get() returns the value that operator[] assigned, for every key.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key * 10;
    }
    for (int key = 0; key < entry_count; ++key) {
        const auto value = table.get(key);
        ASSERT_TRUE(value);
        EXPECT_EQ(*value, key * 10);
    }
}

TEST_P(TableAtSizeTest, CPU_ContainsIsFalseForAbsentKey) {
    // contains() is false for a key that was never added.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    EXPECT_FALSE(table.contains(entry_count));
}

TEST_P(TableAtSizeTest, CPU_FindReturnsEndForAbsentKey) {
    // find() returns end() for a key that was never added.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    EXPECT_EQ(table.find(-1), table.end());
}

TEST_P(TableAtSizeTest, CPU_RangeConstructedFindsEveryKey) {
    // A Table built from a range of entries finds the value of every entry.
    const int entry_count = GetParam();
    std::vector<std::pair<int, int>> entries;
    for (int key = 0; key < entry_count; ++key) {
        entries.emplace_back(key, key * 10);
    }
    const m2::Table<int, int> table(entries);
    for (int key = 0; key < entry_count; ++key) {
        const auto value = table.get(key);
        ASSERT_TRUE(value);
        EXPECT_EQ(*value, key * 10);
    }
}

TEST_P(TableAtSizeTest, CPU_EmplacedKeysAreFound) {
    // get() returns the value that emplace() added, for every key.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table.emplace(key, key * 10);
    }
    for (int key = 0; key < entry_count; ++key) {
        const auto value = table.get(key);
        ASSERT_TRUE(value);
        EXPECT_EQ(*value, key * 10);
    }
}

TEST_P(TableAtSizeTest, CPU_SubscriptOnExistingKeysKeepsSize) {
    // Assigning through operator[] to keys that are present does not add entries.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = 0;
    }
    for (int key = 0; key < entry_count; ++key) {
        table[key] = 1;
    }
    EXPECT_EQ(table.size(), static_cast<std::size_t>(entry_count));
}

TEST_P(TableAtSizeTest, CPU_InsertOfExistingKeyKeepsOldValue) {
    // insert() of a key that is present returns the entry that is already there.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = 1;
    }
    for (int key = 0; key < entry_count; ++key) {
        auto [it, inserted] = table.insert({key, 2});
        EXPECT_FALSE(inserted);
        EXPECT_EQ(it->second, 1);
    }
}

TEST_P(TableAtSizeTest, CPU_EraseKeepsOtherKeysFindable) {
    // After the even keys are erased, only the odd keys are found.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    for (int key = 0; key < entry_count; key += 2) {
        table.erase(key);
    }
    for (int key = 0; key < entry_count; ++key) {
        EXPECT_EQ(table.contains(key), key % 2 != 0) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_ErasedKeyCanBeAddedAgain) {
    // A key that was erased can be added again with a new value.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    table.erase(0);
    table[0] = 42;
    const auto value = table.get(0);
    ASSERT_TRUE(value);
    EXPECT_EQ(*value, 42);
}

TEST_P(TableAtSizeTest, CPU_KeysAddedAfterEraseAreFound) {
    // After the even keys are erased and new keys are added, the odd keys and the new keys are found.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    for (int key = 0; key < entry_count; key += 2) {
        table.erase(key);
    }
    for (int key = entry_count; key < 2 * entry_count; ++key) {
        table[key] = key;
    }
    for (int key = 0; key < 2 * entry_count; ++key) {
        EXPECT_EQ(table.contains(key), key >= entry_count || key % 2 != 0) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_CopyFindsEveryKey) {
    // A copy-constructed Table finds every key of its source.
    const int entry_count = GetParam();
    m2::Table<int, int> source;
    for (int key = 0; key < entry_count; ++key) {
        source[key] = key;
    }
    const m2::Table<int, int> copy = source;
    for (int key = 0; key < entry_count; ++key) {
        EXPECT_TRUE(copy.contains(key)) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_CopyAssignedFindsEveryKey) {
    // A copy-assigned Table finds every key of its source.
    const int entry_count = GetParam();
    m2::Table<int, int> source;
    for (int key = 0; key < entry_count; ++key) {
        source[key] = key;
    }
    m2::Table<int, int> target{{-1, -1}};
    target = source;
    for (int key = 0; key < entry_count; ++key) {
        EXPECT_TRUE(target.contains(key)) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_EraseFromCopyLeavesSourceUnchanged) {
    // A copy is independent of its source.
    const int entry_count = GetParam();
    m2::Table<int, int> source;
    for (int key = 0; key < entry_count; ++key) {
        source[key] = key;
    }
    m2::Table<int, int> copy = source;
    copy.erase(0);
    EXPECT_TRUE(source.contains(0));
    EXPECT_EQ(source.size(), static_cast<std::size_t>(entry_count));
}

TEST_P(TableAtSizeTest, CPU_MoveConstructedFindsEveryKey) {
    // A move-constructed Table finds every key of its source.
    const int entry_count = GetParam();
    m2::Table<int, int> source;
    for (int key = 0; key < entry_count; ++key) {
        source[key] = key;
    }
    const m2::Table<int, int> moved = std::move(source);
    for (int key = 0; key < entry_count; ++key) {
        EXPECT_TRUE(moved.contains(key)) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_MoveAssignedFindsEveryKeyOfSource) {
    // A move-assigned Table finds every key of its source.
    const int entry_count = GetParam();
    m2::Table<int, int> source;
    for (int key = 0; key < entry_count; ++key) {
        source[key] = key;
    }
    m2::Table<int, int> target{{-1, -1}};
    target = std::move(source);
    for (int key = 0; key < entry_count; ++key) {
        EXPECT_TRUE(target.contains(key)) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_MoveAssignedDoesNotFindOldKeys) {
    // A move-assigned Table no longer finds the keys it held before the assignment.
    const int entry_count = GetParam();
    m2::Table<int, int> source;
    for (int key = 0; key < entry_count; ++key) {
        source[key] = key;
    }
    m2::Table<int, int> target{{-1, -1}};
    target = std::move(source);
    EXPECT_FALSE(target.contains(-1));
}

TEST_P(TableAtSizeTest, CPU_MovedFromTableCanBeClearedAndReused) {
    // A moved-from Table finds every key that is added after clear().
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    const m2::Table<int, int> moved = std::move(table);
    table.clear();  // NOLINT(bugprone-use-after-move)
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    for (int key = 0; key < entry_count; ++key) {
        EXPECT_TRUE(table.contains(key)) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_ClearThenReuseFindsOnlyNewKeys) {
    // After clear(), the old keys are absent and the keys added later are found.
    const int entry_count = GetParam();
    m2::Table<int, int> table;
    for (int key = 0; key < entry_count; ++key) {
        table[key] = key;
    }
    table.clear();
    for (int key = entry_count; key < 2 * entry_count; ++key) {
        table[key] = key;
    }
    for (int key = 0; key < 2 * entry_count; ++key) {
        EXPECT_EQ(table.contains(key), key >= entry_count) << "key " << key;
    }
}

TEST_P(TableAtSizeTest, CPU_TablesBuiltInOppositeOrderAreEqual) {
    // Equality does not depend on the order in which keys were added.
    const int entry_count = GetParam();
    m2::Table<int, int> forward;
    m2::Table<int, int> backward;
    for (int key = 0; key < entry_count; ++key) {
        forward[key] = key;
        backward[entry_count - 1 - key] = entry_count - 1 - key;
    }
    EXPECT_EQ(forward, backward);
}

TEST_P(TableAtSizeTest, CPU_TablesWithOneDifferentValueAreNotEqual) {
    // Tables with the same keys are not equal when one value differs.
    const int entry_count = GetParam();
    m2::Table<int, int> a;
    m2::Table<int, int> b;
    for (int key = 0; key < entry_count; ++key) {
        a[key] = key;
        b[key] = key;
    }
    b[entry_count - 1] = -1;
    EXPECT_NE(a, b);
}

// ---- keys with equal hashes --------------------------------------------------

// The parameter is the number of entries that each test adds to the Table.
class TableWithCollidingKeysAtSizeTest : public ::testing::TestWithParam<int> {};

// Sizes on each side of the inline capacity, kept small because every lookup reads all colliding keys.
INSTANTIATE_TEST_SUITE_P(EntryCounts, TableWithCollidingKeysAtSizeTest, ::testing::Values(1, 8, 9, 17, 100));

TEST_P(TableWithCollidingKeysAtSizeTest, CPU_KeysWithEqualHashesAreAllFound) {
    // Keys that have the same std::hash value are still told apart.
    const int entry_count = GetParam();
    m2::Table<CollidingKey, int> table;
    for (int id = 0; id < entry_count; ++id) {
        table[CollidingKey{id}] = id;
    }
    for (int id = 0; id < entry_count; ++id) {
        const auto value = table.get(CollidingKey{id});
        ASSERT_TRUE(value);
        EXPECT_EQ(*value, id);
    }
}

TEST_P(TableWithCollidingKeysAtSizeTest, CPU_EraseAmongKeysWithEqualHashesKeepsOthersFindable) {
    // After the even keys are erased, only the odd keys are found, when all keys have the same hash.
    const int entry_count = GetParam();
    m2::Table<CollidingKey, int> table;
    for (int id = 0; id < entry_count; ++id) {
        table[CollidingKey{id}] = id;
    }
    for (int id = 0; id < entry_count; id += 2) {
        table.erase(CollidingKey{id});
    }
    for (int id = 0; id < entry_count; ++id) {
        EXPECT_EQ(table.contains(CollidingKey{id}), id % 2 != 0) << "id " << id;
    }
}

// The parameter is the std::hash value that all keys of a test share.
class TableWithSharedHashTest : public ::testing::TestWithParam<std::size_t> {};

// Many shared hash values, so that the keys' common search starts at many different places in the Table.
INSTANTIATE_TEST_SUITE_P(SharedHashes, TableWithSharedHashTest, ::testing::Range<std::size_t>(0, 64));

TEST_P(TableWithSharedHashTest, CPU_KeysThatShareTheHashValueAreAllFound) {
    // Keys that all have the same std::hash value are told apart, whatever that value is.
    const std::size_t shared_hash = GetParam();
    constexpr int kKeyCount = 12;
    m2::Table<KeyWithChosenHash, int> table;
    for (int id = 0; id < kKeyCount; ++id) {
        table[KeyWithChosenHash{id, shared_hash}] = id;
    }
    for (int id = 0; id < kKeyCount; ++id) {
        const auto value = table.get(KeyWithChosenHash{id, shared_hash});
        ASSERT_TRUE(value);
        EXPECT_EQ(*value, id);
    }
}

// ---- NodeCoord keys ----------------------------------------------------------

TEST(TableNodeCoordTest, CPU_FindsEveryNodeOfAFullGrid) {
    // A per-node table for every node of an 11 x 10 grid finds the value of each node.
    constexpr std::size_t kGridWidth = 11;
    constexpr std::size_t kNodeCount = 110;
    m2::Table<m2::NodeCoord, std::uint32_t> arg_per_node;
    for (std::size_t i = 0; i < kNodeCount; ++i) {
        arg_per_node[m2::NodeCoord{i % kGridWidth, i / kGridWidth}] = static_cast<std::uint32_t>(i);
    }
    for (std::size_t i = 0; i < kNodeCount; ++i) {
        const auto value = arg_per_node.get(m2::NodeCoord{i % kGridWidth, i / kGridWidth});
        ASSERT_TRUE(value);
        EXPECT_EQ(*value, i);
    }
}

}  // namespace
