// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string_view>
#include <reflect>

using namespace std::literals;

enum E { A, B };
struct foo { int a; E b; };

constexpr auto f = foo{.a = 42, .b = B};

// Compile-time checks using static_assert
static_assert(reflect::size(f) == 2, "Reflect size check failed");
static_assert(reflect::type_id(f.a) != reflect::type_id(f.b), "Reflect type_id check failed");
static_assert("foo"sv == reflect::type_name(f), "Reflect type_name (obj) check failed");
static_assert("int"sv == reflect::type_name(f.a), "Reflect type_name (a) check failed");
static_assert("E"sv == reflect::type_name(f.b), "Reflect type_name (b) check failed");
static_assert("B"sv == reflect::enum_name(f.b), "Reflect enum_name check failed");
static_assert("a"sv == reflect::member_name<0>(f), "Reflect member_name<0> check failed");
static_assert("b"sv == reflect::member_name<1>(f), "Reflect member_name<1> check failed");
static_assert(42 == reflect::get<0>(f), "Reflect get<0> check failed");
static_assert(B == reflect::get<1>(f), "Reflect get<1> check failed");
static_assert(42 == reflect::get<"a">(f), "Reflect get<\"a\"> check failed");
static_assert(B == reflect::get<"b">(f), "Reflect get<\"b\"> check failed");

constexpr auto t = reflect::to<std::tuple>(f);
static_assert(42 == std::get<0>(t), "Reflect to<std::tuple> get<0> check failed");
static_assert(B == std::get<1>(t), "Reflect to<std::tuple> get<1> check failed");
