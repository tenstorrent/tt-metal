// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <optional>
#include <string>
#include <vector>
#include <sstream>

#include <tt-metalium/maybe_remote.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal::distributed {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;

struct TestType {
    int value;
    bool operator==(const TestType& other) const { return value == other.value; }
};

std::ostream& operator<<(std::ostream& os, const TestType& t) {
    return os << "TestType{" << t.value << "}";
}

TEST(MaybeRemoteTest, LocalConstruction) {
    auto local_int = MaybeRemote<int>::local(42);
    EXPECT_TRUE(local_int.is_local());
    EXPECT_FALSE(local_int.is_remote());
    EXPECT_EQ(local_int.value(), 42);

    auto local_string = MaybeRemote<std::string>::local("hello");
    EXPECT_TRUE(local_string.is_local());
    EXPECT_FALSE(local_string.is_remote());
    EXPECT_EQ(local_string.value(), "hello");

    auto local_test = MaybeRemote<TestType>::local({123});
    EXPECT_TRUE(local_test.is_local());
    EXPECT_FALSE(local_test.is_remote());
    EXPECT_EQ(local_test.value().value, 123);
}

TEST(MaybeRemoteTest, RemoteConstruction) {
    auto remote_int = MaybeRemote<int>::remote();
    EXPECT_FALSE(remote_int.is_local());
    EXPECT_TRUE(remote_int.is_remote());

    auto remote_string = MaybeRemote<std::string>::remote();
    EXPECT_FALSE(remote_string.is_local());
    EXPECT_TRUE(remote_string.is_remote());

    auto remote_test = MaybeRemote<TestType>::remote();
    EXPECT_FALSE(remote_test.is_local());
    EXPECT_TRUE(remote_test.is_remote());
}

TEST(MaybeRemoteTest, ValueAccess) {
    auto local = MaybeRemote<int>::local(42);
    EXPECT_EQ(local.value(), 42);

    // Test mutable access
    local.value() = 100;
    EXPECT_EQ(local.value(), 100);

    // Test const access
    const auto const_local = MaybeRemote<int>::local(200);
    EXPECT_EQ(const_local.value(), 200);
}

TEST(MaybeRemoteTest, ValueAccessThrowsForRemote) {
    auto remote = MaybeRemote<int>::remote();
    EXPECT_THROW((void)remote.value(), std::exception);

    const auto const_remote = MaybeRemote<std::string>::remote();
    EXPECT_THROW((void)const_remote.value(), std::exception);
}

TEST(MaybeRemoteTest, WhenPatternMatching) {
    auto local = MaybeRemote<int>::local(42);

    int result = local.when(
        [](int value) { return value * 2; },
        []() { return -1; }
    );
    EXPECT_EQ(result, 84);

    auto remote = MaybeRemote<int>::remote();
    result = remote.when(
        [](int value) { return value * 2; },
        []() { return -1; }
    );
    EXPECT_EQ(result, -1);

    // Test with different return types
    std::string str_result = local.when(
        [](int value) { return std::to_string(value); },
        []() { return std::string("remote"); }
    );
    EXPECT_EQ(str_result, "42");

    str_result = remote.when(
        [](int value) { return std::to_string(value); },
        []() { return std::string("remote"); }
    );
    EXPECT_EQ(str_result, "remote");
}

TEST(MaybeRemoteTest, Equality) {
    auto local1 = MaybeRemote<int>::local(42);
    auto local2 = MaybeRemote<int>::local(42);
    auto local3 = MaybeRemote<int>::local(100);
    auto remote1 = MaybeRemote<int>::remote();
    auto remote2 = MaybeRemote<int>::remote();

    EXPECT_EQ(local1, local2);
    EXPECT_NE(local1, local3);
    EXPECT_NE(local1, remote1);
    EXPECT_EQ(remote1, remote2);

    // Test with custom types
    auto local_test1 = MaybeRemote<TestType>::local({123});
    auto local_test2 = MaybeRemote<TestType>::local({123});
    auto local_test3 = MaybeRemote<TestType>::local({456});

    EXPECT_EQ(local_test1, local_test2);
    EXPECT_NE(local_test1, local_test3);
}

TEST(MaybeRemoteTest, ToString) {
    auto local = MaybeRemote<int>::local(42);
    EXPECT_EQ(local.to_string(), "MaybeRemote{42}");

    auto remote = MaybeRemote<int>::remote();
    EXPECT_EQ(remote.to_string(), "MaybeRemote{remote}");

    auto local_string = MaybeRemote<std::string>::local("hello");
    EXPECT_EQ(local_string.to_string(), "MaybeRemote{hello}");

    auto local_test = MaybeRemote<TestType>::local({789});
    EXPECT_EQ(local_test.to_string(), "MaybeRemote{TestType{789}}");
}

TEST(MaybeRemoteTest, ExtractLocals) {
    std::vector<MaybeRemote<int>> mixed_values = {
        MaybeRemote<int>::local(1),
        MaybeRemote<int>::remote(),
        MaybeRemote<int>::local(2),
        MaybeRemote<int>::local(3),
        MaybeRemote<int>::remote(),
        MaybeRemote<int>::local(4)
    };

    auto locals = extract_locals(mixed_values);
    EXPECT_THAT(locals, ElementsAre(1, 2, 3, 4));

    // Test with all remote
    std::vector<MaybeRemote<int>> all_remote = {
        MaybeRemote<int>::remote(),
        MaybeRemote<int>::remote(),
        MaybeRemote<int>::remote()
    };

    auto no_locals = extract_locals(all_remote);
    EXPECT_TRUE(no_locals.empty());

    // Test with all local
    std::vector<MaybeRemote<int>> all_local = {
        MaybeRemote<int>::local(10),
        MaybeRemote<int>::local(20),
        MaybeRemote<int>::local(30)
    };

    auto all_locals = extract_locals(all_local);
    EXPECT_THAT(all_locals, ElementsAre(10, 20, 30));
}

TEST(MaybeRemoteTest, ExtractLocalsWithCustomType) {
    std::vector<MaybeRemote<TestType>> mixed_values = {
        MaybeRemote<TestType>::local({100}),
        MaybeRemote<TestType>::remote(),
        MaybeRemote<TestType>::local({200}),
        MaybeRemote<TestType>::local({300})
    };

    auto locals = extract_locals(mixed_values);
    EXPECT_EQ(locals.size(), 3);
    EXPECT_EQ(locals[0].value, 100);
    EXPECT_EQ(locals[1].value, 200);
    EXPECT_EQ(locals[2].value, 300);
}

TEST(MaybeRemoteTest, TypeAliases) {
    // Test MaybeRemoteDeviceId
    auto local_device_id = MaybeRemoteDeviceId::local(5);
    EXPECT_TRUE(local_device_id.is_local());
    EXPECT_EQ(local_device_id.value(), 5);

    auto remote_device_id = MaybeRemoteDeviceId::remote();
    EXPECT_TRUE(remote_device_id.is_remote());

    // Test MaybeRemoteDevice
    // Note: We can't actually create a real Device object in unit tests,
    // but we can test the type exists and basic operations work
    auto remote_device = MaybeRemoteDevice::remote();
    EXPECT_TRUE(remote_device.is_remote());
}

TEST(MaybeRemoteTest, MoveSemantics) {
    // Test with movable type
    auto local_vec = MaybeRemote<std::vector<int>>::local({1, 2, 3, 4, 5});
    EXPECT_EQ(local_vec.value().size(), 5);

    // Move construction
    auto moved_vec = std::move(local_vec);
    EXPECT_EQ(moved_vec.value().size(), 5);
    EXPECT_THAT(moved_vec.value(), ElementsAre(1, 2, 3, 4, 5));

    // Move assignment
    auto another_vec = MaybeRemote<std::vector<int>>::local({10, 20});
    another_vec = std::move(moved_vec);
    EXPECT_EQ(another_vec.value().size(), 5);
    EXPECT_THAT(another_vec.value(), ElementsAre(1, 2, 3, 4, 5));
}

TEST(MaybeRemoteTest, CopySemantics) {
    auto local = MaybeRemote<int>::local(42);

    // Copy construction
    auto copied = local;
    EXPECT_EQ(copied.value(), 42);
    EXPECT_EQ(local.value(), 42);  // Original still valid

    // Copy assignment
    auto another = MaybeRemote<int>::local(100);
    another = local;
    EXPECT_EQ(another.value(), 42);
    EXPECT_EQ(local.value(), 42);  // Original still valid
}

TEST(MaybeRemoteTest, IfLocal) {
    // Test with local value
    auto local = MaybeRemote<int>::local(42);

    // Test if_local with return value
    auto result = local.if_local([](int value) { return value * 2; });
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 84);

    // Test if_local with different return type
    auto str_result = local.if_local([](int value) { return std::to_string(value); });
    ASSERT_TRUE(str_result.has_value());
    EXPECT_EQ(str_result.value(), "42");

    // Test if_local with mutable lambda
    int captured = 0;
    local.if_local([&captured](int value) { captured = value; });
    EXPECT_EQ(captured, 42);

    // Test with remote value - if_local should not execute
    auto remote = MaybeRemote<int>::remote();
    captured = 0;
    remote.if_local([&captured](int value) { captured = value; });
    EXPECT_EQ(captured, 0);  // Should remain unchanged

    // Test with custom type
    auto local_test = MaybeRemote<TestType>::local({123});
    auto test_result = local_test.if_local([](const TestType& t) { return t.value * 3; });
    ASSERT_TRUE(test_result.has_value());
    EXPECT_EQ(test_result.value(), 369);

    // Test with void return
    bool was_called = false;
    local.if_local([&was_called](int) { was_called = true; });
    EXPECT_TRUE(was_called);

    was_called = false;
    remote.if_local([&was_called](int) { was_called = true; });
    EXPECT_FALSE(was_called);

    // Test if_local on remote returns empty optional
    auto remote_result = remote.if_local([](int value) { return value * 2; });
    EXPECT_FALSE(remote_result.has_value());
}

TEST(MaybeRemoteTest, IfRemote) {
    // Test with remote value
    auto remote = MaybeRemote<int>::remote();

    // Test if_remote with return value
    auto result = remote.if_remote([]() { return 999; });
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 999);

    // Test if_remote with different return type
    auto str_result = remote.if_remote([]() { return std::string("is_remote"); });
    ASSERT_TRUE(str_result.has_value());
    EXPECT_EQ(str_result.value(), "is_remote");

    // Test if_remote with mutable lambda
    int captured = 0;
    remote.if_remote([&captured]() { captured = 100; });
    EXPECT_EQ(captured, 100);

    // Test with local value - if_remote should not execute
    auto local = MaybeRemote<int>::local(42);
    captured = 0;
    local.if_remote([&captured]() { captured = 100; });
    EXPECT_EQ(captured, 0);  // Should remain unchanged

    // Test with void return
    bool was_called = false;
    remote.if_remote([&was_called]() { was_called = true; });
    EXPECT_TRUE(was_called);

    was_called = false;
    local.if_remote([&was_called]() { was_called = true; });
    EXPECT_FALSE(was_called);

    // Test if_remote on local returns empty optional
    auto local_result = local.if_remote([]() { return 999; });
    EXPECT_FALSE(local_result.has_value());
}

TEST(MaybeRemoteTest, IfLocalIfRemoteCombined) {
    // Test using both if_local and if_remote together
    auto process_maybe_remote = [](const MaybeRemote<int>& maybe) -> std::string {
        std::string result = "none";
        maybe.if_local([&result](int value) { result = "local:" + std::to_string(value); });
        maybe.if_remote([&result]() { result = "remote"; });
        return result;
    };

    auto local = MaybeRemote<int>::local(42);
    EXPECT_EQ(process_maybe_remote(local), "local:42");

    auto remote = MaybeRemote<int>::remote();
    EXPECT_EQ(process_maybe_remote(remote), "remote");

    // Test chaining behavior - only one should execute
    int counter = 0;
    local.if_local([&counter](int) { counter++; });
    local.if_remote([&counter]() { counter += 10; });
    EXPECT_EQ(counter, 1);

    counter = 0;
    remote.if_local([&counter](int) { counter++; });
    remote.if_remote([&counter]() { counter += 10; });
    EXPECT_EQ(counter, 10);
}

TEST(MaybeRemoteTest, ComplexScenarios) {
    // Test with nested MaybeRemote (though probably not a real use case)
    using NestedType = MaybeRemote<MaybeRemote<int>>;
    auto nested_local = NestedType::local(MaybeRemote<int>::local(42));
    EXPECT_TRUE(nested_local.is_local());
    EXPECT_TRUE(nested_local.value().is_local());
    EXPECT_EQ(nested_local.value().value(), 42);

    // Test with container of MaybeRemote
    std::vector<MaybeRemote<std::string>> container;
    container.push_back(MaybeRemote<std::string>::local("first"));
    container.push_back(MaybeRemote<std::string>::remote());
    container.push_back(MaybeRemote<std::string>::local("third"));

    int local_count = 0;
    int remote_count = 0;
    for (const auto& item : container) {
        item.when(
            [&](const std::string&) { local_count++; },
            [&]() { remote_count++; }
        );
    }

    EXPECT_EQ(local_count, 2);
    EXPECT_EQ(remote_count, 1);
}

}  // namespace
}  // namespace tt::tt_metal::distributed
