// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tools/tt_builder/builder.hpp"

TEST(BuilderTest, test_firmware_build) {
    build_firmware();
    EXPECT_EQ(1, 1);
}
