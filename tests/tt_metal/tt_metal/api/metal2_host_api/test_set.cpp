// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API FlatSet<K, Proj> container (set.hpp).
//
// Pure host-side data-structure tests — no device required.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <concepts>
#include <iterator>
#include <set>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/utility/set.hpp>

namespace {

namespace m2 = tt::tt_metal::experimental;

using IntSet = m2::FlatSet<int>;
using StrSet = m2::FlatSet<std::string>;

// ---- iterator contract (compile-time) ----------------------------------------

static_assert(std::forward_iterator<IntSet::iterator>, "FlatSet::iterator must model std::forward_iterator");
static_assert(
    std::forward_iterator<IntSet::const_iterator>, "FlatSet::const_iterator must model std::forward_iterator");
// Elements are immutable, so both iterators are the same const iterator type.
static_assert(
    std::same_as<IntSet::iterator, IntSet::const_iterator>, "FlatSet exposes elements as const through both iterators");

// ---- construction / emptiness ------------------------------------------------

TEST(SetTest, DefaultConstructedIsEmpty) {
    IntSet s;
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.begin(), s.end());
}

TEST(SetTest, InitializerListConstruction) {
    IntSet s{1, 2, 3};
    EXPECT_FALSE(s.empty());
    EXPECT_EQ(s.size(), 3u);
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
    EXPECT_TRUE(s.contains(3));
}

TEST(SetTest, InitializerListIgnoresDuplicates) {
    IntSet s{1, 1, 2, 2, 2, 3};
    EXPECT_EQ(s.size(), 3u);
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
    EXPECT_TRUE(s.contains(3));
}

TEST(SetTest, SpanConstruction) {
    std::vector<int> keys{1, 2};
    std::span<const int> keys_view(keys);
    IntSet s(keys_view);
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
}

TEST(SetTest, RangeConstructionFromUnorderedSet) {
    std::unordered_set<int> src{1, 2, 3};
    IntSet s(src);
    EXPECT_EQ(s.size(), 3u);
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
    EXPECT_TRUE(s.contains(3));
}

TEST(SetTest, RangeConstructionFromStdSet) {
    std::set<int> src{1, 2};
    IntSet s(src);
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
}

TEST(SetTest, RangeConstructionFromVectorIgnoresDuplicates) {
    std::vector<int> src{1, 2, 1};
    IntSet s(src);
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
}

// ---- insert (insert-if-absent) -----------------------------------------------

TEST(SetTest, InsertNewKey) {
    IntSet s;
    auto [it, inserted] = s.insert(1);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(s.size(), 1u);
}

TEST(SetTest, InsertExistingKeyIsNoop) {
    IntSet s{1};
    auto [it, inserted] = s.insert(1);
    EXPECT_FALSE(inserted);
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(s.size(), 1u);
}

// ---- erase -------------------------------------------------------------------

TEST(SetTest, EraseRemovesPresentKey) {
    IntSet s{1, 2};
    EXPECT_EQ(s.erase(1), 1u);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_FALSE(s.contains(1));
    EXPECT_TRUE(s.contains(2));  // surviving element intact
}

TEST(SetTest, EraseAbsentKeyIsNoop) {
    IntSet s{1};
    EXPECT_EQ(s.erase(99), 0u);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_TRUE(s.contains(1));
}

TEST(SetTest, EraseThenReinsert) {
    IntSet s{1};
    EXPECT_EQ(s.erase(1), 1u);
    EXPECT_TRUE(s.empty());
    auto [it, inserted] = s.insert(1);
    EXPECT_TRUE(inserted);
    EXPECT_TRUE(s.contains(1));
}

// ---- find / contains / count -------------------------------------------------

TEST(SetTest, FindHitAndMiss) {
    IntSet s{1};
    auto it = s.find(1);
    ASSERT_NE(it, s.end());
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(s.find(99), s.end());
}

TEST(SetTest, ContainsAndCount) {
    IntSet s{1, 2};
    EXPECT_TRUE(s.contains(1));
    EXPECT_FALSE(s.contains(99));
    EXPECT_EQ(s.count(1), 1u);
    EXPECT_EQ(s.count(99), 0u);
}

// ---- size / clear ------------------------------------------------------------

TEST(SetTest, Clear) {
    IntSet s{1, 2};
    EXPECT_EQ(s.size(), 2u);
    s.clear();
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0u);
}

// ---- iteration ---------------------------------------------------------------

TEST(SetTest, IterationVisitsAllKeys) {
    IntSet s{1, 2, 3};
    std::set<int> seen;
    for (const auto& k : s) {
        seen.insert(k);
    }
    EXPECT_EQ(seen, (std::set<int>{1, 2, 3}));
}

TEST(SetTest, RangeConstructsStdSet) {
    IntSet s{1, 2};
    std::set<int> m(s.begin(), s.end());
    EXPECT_EQ(m, (std::set<int>{1, 2}));
}

// ---- equality (order independent) --------------------------------------------

TEST(SetTest, EqualityIsOrderIndependent) {
    IntSet a{1, 2};
    IntSet b;
    b.insert(2);
    b.insert(1);  // inserted in the opposite order
    EXPECT_EQ(a, b);
}

TEST(SetTest, InequalityCases) {
    IntSet base{1, 2};
    EXPECT_NE(base, (IntSet{1}));     // different size
    EXPECT_NE(base, (IntSet{1, 3}));  // different key
}

// ---- genericity over other key types -----------------------------------------

TEST(SetMiscTest, StringKeys) {
    StrSet s;
    s.insert("one");
    s.insert("two");
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.contains("one"));
    EXPECT_TRUE(s.contains("two"));
    EXPECT_FALSE(s.contains("three"));
}

// ---- projected comparison (compare on a member of K) -------------------------

namespace projection {

struct Entry {
    int id = 0;
    std::string payload;
    bool operator==(const Entry&) const = default;
};

using EntrySet = m2::FlatSet<Entry, &Entry::id>;

TEST(SetProjectionTest, UniquenessIsByProjectedMember) {
    EntrySet s;
    auto [it, inserted] = s.insert({1, "first"});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->payload, "first");

    // Same .id, different payload -> rejected, original kept.
    auto [it2, inserted2] = s.insert({1, "second"});
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(it2->payload, "first");
    EXPECT_EQ(s.size(), 1u);
}

TEST(SetProjectionTest, LookupIgnoresUnprojectedMembers) {
    EntrySet s{{1, "a"}, {2, "b"}};
    EXPECT_TRUE(s.contains(Entry{1, "ignored"}));
    EXPECT_EQ(s.count(Entry{2, ""}), 1u);
    EXPECT_FALSE(s.contains(Entry{3, "a"}));

    auto it = s.find(Entry{1, ""});
    ASSERT_NE(it, s.end());
    EXPECT_EQ(it->payload, "a");
}

TEST(SetProjectionTest, InitializerListDedupesByProjectedMember) {
    EntrySet s{{1, "a"}, {1, "b"}, {2, "c"}};
    EXPECT_EQ(s.size(), 2u);
    EXPECT_EQ(s.find(Entry{1, ""})->payload, "a");  // first kept
}

TEST(SetProjectionTest, EqualityComparesFullValueNotJustProjectedKey) {
    // Same projected key (.id), differing unprojected member (.payload) -> unequal,
    // even though each set "contains" the other's key under the projection.
    EntrySet a{{1, "a"}};
    EntrySet b{{1, "b"}};
    EXPECT_NE(a, b);
    EXPECT_EQ(a, (EntrySet{{1, "a"}}));
}

TEST(SetProjectionTest, EraseByProjectedMember) {
    EntrySet s{{1, "a"}, {2, "b"}};
    EXPECT_EQ(s.erase(Entry{1, "ignored"}), 1u);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_FALSE(s.contains(Entry{1, ""}));
    EXPECT_TRUE(s.contains(Entry{2, ""}));
}

}  // namespace projection

}  // namespace
