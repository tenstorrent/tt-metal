// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/tt_stl/slotmap.hpp"


MAKE_SLOTMAP_KEY(IntKey, uint16_t, 10);
using IntSlotMap = tt::stl::SlotMap<IntKey, int>;


MAKE_SLOTMAP_KEY(StringKey, uint16_t, 10);
using StringSlotMap = tt::stl::SlotMap<StringKey, std::string>;

TEST(SlotMapTest, CanCreateSlotMap) {
  IntSlotMap slotmap;
  EXPECT_TRUE(slotmap.empty());
}

TEST(SlotMapTest, CanInsertIntoSlotMap) {
  IntSlotMap slotmap;
  auto key = slotmap.insert(42);

  EXPECT_TRUE(slotmap.contains(key));
  EXPECT_EQ(slotmap.size(), 1);
  EXPECT_EQ(*slotmap.get(key), 42);
}

TEST(SlotMapTest, CanInsertIntoStringSlotMap) {
  StringSlotMap slotmap;
  auto key = slotmap.insert("hello");

  EXPECT_TRUE(slotmap.contains(key));
  EXPECT_EQ(slotmap.size(), 1);
  EXPECT_EQ(*slotmap.get(key), "hello");
}

TEST(SlotMapTest, CanInsertMultipleValuesIntoSlotMap) {
  IntSlotMap slotmap;

  auto key1 = slotmap.insert(42);
  auto key2 = slotmap.insert(43);
  auto key3 = slotmap.insert(44);

  EXPECT_TRUE(slotmap.contains(key1));
  EXPECT_TRUE(slotmap.contains(key2));
  EXPECT_TRUE(slotmap.contains(key3));
  EXPECT_EQ(slotmap.size(), 3);
  EXPECT_EQ(*slotmap.get(key1), 42);
  EXPECT_EQ(*slotmap.get(key2), 43);
  EXPECT_EQ(*slotmap.get(key3), 44);
}

TEST(SlotMapTest, CanRemoveValueFromSlotMap) {
  IntSlotMap slotmap;

  auto key1 = slotmap.insert(42);
  auto key2 = slotmap.insert(43);

  EXPECT_TRUE(slotmap.contains(key1));
  EXPECT_TRUE(slotmap.contains(key2));
  EXPECT_EQ(slotmap.size(), 2);

  slotmap.remove(key2);

  EXPECT_TRUE(slotmap.contains(key1));
  EXPECT_FALSE(slotmap.contains(key2));
  EXPECT_EQ(slotmap.size(), 1);
  EXPECT_EQ(*slotmap.get(key1), 42);
}

TEST(SlotMapTest, CanRemoveValueFromStringSlotMap) {
  StringSlotMap slotmap{};

  auto key1 = slotmap.insert("hello");
  auto key2 = slotmap.insert("world");

  EXPECT_TRUE(slotmap.contains(key1));
  EXPECT_TRUE(slotmap.contains(key2));
  EXPECT_EQ(slotmap.size(), 2);

  slotmap.remove(key1);

  EXPECT_FALSE(slotmap.contains(key1));
  EXPECT_TRUE(slotmap.contains(key2));
  EXPECT_EQ(slotmap.size(), 1);
  EXPECT_EQ(*slotmap.get(key2), "world");
}

TEST(SlotMapTest, CanIterateOverSlotMap) {
  IntSlotMap slotmap;

  slotmap.insert(42);
  slotmap.insert(43);
  slotmap.insert(44);

  std::vector<int> expected_values = {42, 43, 44};
  std::vector<int> actual_values;
  std::copy(slotmap.cbegin(), slotmap.cend(), std::back_inserter(actual_values));

  EXPECT_EQ(actual_values, expected_values);
}

TEST(KeyTest, CanCreateKeyFromRaw) {
    uint16_t raw = 0b0000000101000011;
    IntKey key(raw);

    EXPECT_EQ(key.index(), 0b101);
    EXPECT_EQ(key.version(), 0b11);
}

TEST(SlotMapTest, ThrowsOnInsertIfMaxIndex) {
    IntSlotMap slotmap;

    IntKey key;
    for (int i = 0; i < IntKey::max_index + 1; i++) {
        key = slotmap.insert(i);
    }

    EXPECT_EQ(key.index(), IntKey::max_index);
    EXPECT_THROW(slotmap.insert(0), std::runtime_error);
}
