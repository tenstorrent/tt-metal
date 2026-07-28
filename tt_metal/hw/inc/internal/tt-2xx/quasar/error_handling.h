// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DM errors
// Values are taken from
// https://yyz-gitlab.local.tenstorrent.com/tensix/soc/overlay/-/blob/main/docs/riscv/exceptions/riscv_exception_handling_doc.md
// and represent the error code returned in `mcause` register.
enum class DmErrors : uint32_t {
    MISALIGNED_PC = 0,
    PC_ADDRESS_FAULT = 1,
    ILLEGAL_INSTRUCTION = 2,
    HW_BREAKPOINT = 3,
    UNALIGNED_LOAD = 4,
    LOAD_ACCESS_FAULT = 5,
    UNALIGNED_STORE = 6,
    STORE_ACCESS_FAULT = 7,

};

// TRISC errors
// Values are taken from https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/564527286/Error+Aggregator, additional
// error information is available in `ERR_DATA` register for errors 0-3.
enum class TriscErrors : uint32_t {
    ERROR_TRISC0 = 0,
    ERROR_TRISC1 = 1,
    ERROR_TRISC2 = 2,
    ERROR_TRISC3 = 3,
    UNPACKER_0 = 5,
    UNPACKER_1 = 6,
    UNPACKER_2 = 7,
    PACKER_0 = 8,
    PACKER_1 = 9,
    EDC_FATAL_ERROR = 10,
    EDC_CORRECTABLE_ERROR = 11,
    NEO_SEMAPHORES = 12,
    GLOBAL_SEMAPHORES = 13,
    SFPU = 14,
    TILE_COUNTERS = 15,
    ILLEGAL_INSTRUCTION_TRISC3 = 32,
    ILLEGAL_INSTRUCTION_TRISC2 = 33,
    ILLEGAL_INSTRUCTION_TRISC1 = 34,
    ILLEGAL_INSTRUCTION_TRISC0 = 35,
};
