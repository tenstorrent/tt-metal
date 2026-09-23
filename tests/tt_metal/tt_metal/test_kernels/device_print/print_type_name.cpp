// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

namespace test::deep {
struct Foo {};

template <typename T>
struct Wrapper {};

enum class Bar : uint8_t {
    A = 0,
};
}  // namespace test::deep

/*
 * Test printing type names from a kernel running on BRISC.
 */

void kernel_main() {
    float f = 1.0f;
    DEVICE_PRINT("builtin type: {}\n", dp_type_name_t<int>());
    DEVICE_PRINT("struct type: {}\n", dp_type_name_t<test::deep::Foo>());
    DEVICE_PRINT("enum type: {}\n", dp_type_name_t<test::deep::Bar>());
    DEVICE_PRINT("template type: {}\n", dp_type_name_t<test::deep::Wrapper<int>>());
    DEVICE_PRINT("decltype: {}\n", dp_type_name_t<decltype(f)>());
    DEVICE_PRINT("with value: {} = {}\n", dp_type_name_t<decltype(f)>(), f);
}
