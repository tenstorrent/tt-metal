// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <tt_stl/strong_type.hpp>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

using MyIntId = tt::stl::StrongType<int, struct MyIntIdTag>;
using MyStringId = tt::stl::StrongType<std::string, struct MyStringIdTag>;

namespace tt::stl {
namespace {

using ::testing::ElementsAre;
using ::testing::IsNull;
using ::testing::UnorderedElementsAre;

TEST(StrongTypeTest, Basic) {
    MyIntId my_int_id1(42);
    MyIntId my_int_id2(43);

    EXPECT_EQ(*my_int_id1, 42);
    EXPECT_LT(*my_int_id1, *my_int_id2);

    my_int_id1 = MyIntId(43);
    EXPECT_EQ(my_int_id1, my_int_id2);
}

TEST(StrongTypeTest, GuarenteedUnique) {
    StrongType<int> one{1};
    StrongType<int> otherone{1};

    static_assert(not std::same_as<decltype(one), decltype(otherone)>);
    static_assert(not std::is_convertible_v<decltype(one), decltype(otherone)>);

    auto runtime_same = std::is_same_v<decltype(one), decltype(otherone)>;
    EXPECT_FALSE(runtime_same);

    EXPECT_EQ(*one, *otherone);
}

TEST(StrongTypeTest, UseInContainers) {
    std::unordered_set<MyIntId> unordered;
    std::set<MyIntId> ordered;

    unordered.insert(MyIntId(42));
    unordered.insert(MyIntId(43));

    ordered.insert(MyIntId(1));
    ordered.insert(MyIntId(2));
    ordered.insert(MyIntId(3));

    EXPECT_THAT(unordered, UnorderedElementsAre(MyIntId(42), MyIntId(43)));
    EXPECT_THAT(ordered, ElementsAre(MyIntId(1), MyIntId(2), MyIntId(3)));
}

TEST(StrongTypeTest, StreamingOperator) {
    std::stringstream ss;
    ss << MyStringId("hello world");
    EXPECT_EQ(ss.str(), "hello world");
}

TEST(StrongTypeTest, MoveOnlyType) {
    using MoveOnlyType = StrongType<std::unique_ptr<int>, struct MoveOnlyTag>;

    MoveOnlyType from(std::make_unique<int>(42));
    EXPECT_EQ(**from, 42);

    MoveOnlyType to = std::move(from);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(*from, IsNull());
    EXPECT_EQ(**to, 42);
}

}  // namespace
}  // namespace tt::stl
