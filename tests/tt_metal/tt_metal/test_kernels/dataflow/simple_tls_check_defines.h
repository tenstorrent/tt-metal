// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Result slot layout: one slot per DM core, shared by test and kernels.
enum TlsCheckResultIndex : uint32_t {
    TLS_CHECK_KERNEL_ID = 0,
    TLS_CHECK_NUM_THREADS = 1,
    TLS_CHECK_MY_THREAD_ID = 2,
    TLS_CHECK_HART_ID = 3,
    TLS_CHECK_THREAD_0_HART_ID = 4,
    TLS_CHECK_GLOBAL_START = 5,
    TLS_CHECK_GLOBAL_END = 6,
    TLS_CHECK_GLOBAL_ADDR_LO = 7,
    TLS_CHECK_GLOBAL_ADDR_HI = 8,
    TLS_CHECK_UNINITIALIZED_GLOBAL_START = 9,
    TLS_CHECK_UNINITIALIZED_GLOBAL_END = 10,
    TLS_CHECK_THREAD_LOCAL_START = 11,
    TLS_CHECK_THREAD_LOCAL_END = 12,
    TLS_CHECK_THREAD_LOCAL_ADDR_LO = 13,
    TLS_CHECK_THREAD_LOCAL_ADDR_HI = 14,
    TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_START = 15,
    TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_END = 16,
};

constexpr uint32_t NUM_TLS_CHECK_RESULT_SLOTS = 17;
// Rounding to nearest multiple of 4 to ensure alignment for DRAM writes.
constexpr uint32_t TLS_CHECK_RESULT_SLOT_WORDS = ((NUM_TLS_CHECK_RESULT_SLOTS + 3) / 4) * 4;
constexpr uint32_t TLS_CHECK_RESULT_SLOT_BYTES = TLS_CHECK_RESULT_SLOT_WORDS * sizeof(uint32_t);
