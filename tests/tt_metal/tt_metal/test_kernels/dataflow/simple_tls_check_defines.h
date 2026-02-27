// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Result slot layout: one slot per DM core, shared by test and kernels.
// Layout: 9 x uint32_t (36 bytes) per slot.
enum TlsCheckResultIndex : uint32_t {
    TLS_CHECK_KERNEL_ID = 0,
    TLS_CHECK_NUM_KERNEL_THREADS = 1,
    TLS_CHECK_MY_THREAD_ID = 2,
    TLS_CHECK_HART_ID = 3,
    TLS_CHECK_THREAD_0_HART_ID = 4,
    TLS_CHECK_GLOBAL_START = 5,
    TLS_CHECK_GLOBAL_END = 6,
    TLS_CHECK_GLOBAL_ADDR_LO = 7,
    TLS_CHECK_GLOBAL_ADDR_HI = 8,
};

constexpr uint32_t TLS_CHECK_RESULT_SLOT_WORDS = 9;
constexpr uint32_t TLS_CHECK_RESULT_SLOT_BYTES = TLS_CHECK_RESULT_SLOT_WORDS * sizeof(uint32_t);
