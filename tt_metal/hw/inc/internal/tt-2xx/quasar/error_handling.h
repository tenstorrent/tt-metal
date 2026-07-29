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
//
// This is the block ID in error_code[13:8]. The other fields are error_code[15:14] = compute
// core (Neo) ID and error_code[7:0] = error index within the block. The index is decoded by
// one of the per-block enums below; it has no meaning on its own.
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

// Below: error_code[7:0], the error index within the block named by TriscErrors.

// Index for TriscErrors::ERROR_TRISC0..3. Values come from
// https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/461602877 via
// https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/565445671, which defines the
// per-TRISC error report. They are the bases of the legacy ranges on the former page
// (22-26, 31-34, 40-44). Those ranges are no longer a per-TRISC spread, so read the TRISC
// from error_code[13:8], not from here. The faulting PC is in `ERR_DATA`.
//
// Only one is reported at a time, in this priority order:
//   L1_ILLEGAL_ACCESS > STACK_OVERFLOW > MEM_ACCESS_HANG > TTI_BUFFER_HANG
// If several conditions are live, the higher one wins and the lower is never seen. Note that
// MEM_ACCESS_HANG (25) outranks TTI_BUFFER_HANG (22) and the two can overlap: a store to a
// full instruction buffer stalls the load/store queue, so it may report as 25. Use the PC in
// `ERR_DATA` to tell them apart.
enum class TriscRiscErrors : uint32_t {
    TTI_BUFFER_HANG = 22,  // TT instruction buffer is full
    // A load/store did not complete. Can also be a blocked instruction buffer push, since
    // this outranks TTI_BUFFER_HANG.
    MEM_ACCESS_HANG = 25,
    STACK_OVERFLOW = 31,     // stack pointer dropped below the stack limit
    L1_ILLEGAL_ACCESS = 40,  // L1 address check failed
};

// Index for TriscErrors::UNPACKER_0..2 and PACKER_0..1.
// Hardware reports only one at a time: 0 masks 1, and 1:0 mask 2.
enum class TdmaErrors : uint32_t {
    ILLEGAL_FORMAT_CONVERSION = 0,
    BUFFER_LIMIT_BELOW_START = 1,  // buffer limit addr < buffer start addr
    ILLEGAL_TILE_SIZE = 2,
};

// Index for TriscErrors::NEO_SEMAPHORES and GLOBAL_SEMAPHORES (error_code[2:0]).
// The offending semaphore number is captured in `ERR_DATA`.
// WAIT_ON_UNINITIALIZED is not supported for GLOBAL_SEMAPHORES.
enum class SemaphoreErrors : uint32_t {
    WAIT_ON_UNINITIALIZED = 1,
    POST_ON_UNINITIALIZED = 2,
    GET_ON_UNINITIALIZED = 3,
    POST_OVERFLOW = 4,
    GET_UNDERFLOW = 5,
};

// Index for TriscErrors::SFPU. Unlike the others this is a BITMASK and it is sticky,
// so both bits can be set at once. error_code[7:2] are reserved.
enum class SfpuErrors : uint32_t {
    CC_STACK_OVERFLOW = 0x1,
    CC_STACK_UNDERFLOW = 0x2,
};
