// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum fabric_mode {
    PULL,
    PUSH,
};

enum test_mode {
    TEST_ASYNC_WRITE,
    TEST_ATOMIC_INC,
    TEST_ASYNC_WRITE_ATOMIC_INC,
    TEST_ASYNC_WRITE_MULTICAST,
    TEST_ASYNC_WRITE_MULTICAST_MULTIDIRECTION,
};
