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

// Error code layout. Both sides decode these bits: the device to work out which Neo/TRISC to
// blame, the host to render the message. Keep them here so the two can't drift apart.
//   [15:14] Neo ID
//   [13:8]  block ID, see TriscErrors
//   [7:0]   error index within the block, see the per-block enums further down
constexpr uint32_t kQuasarErrNeoShift = 14;
constexpr uint32_t kQuasarErrNeoMask = 0x3;
constexpr uint32_t kQuasarErrBlockShift = 8;
constexpr uint32_t kQuasarErrBlockMask = 0x3f;
constexpr uint32_t kQuasarErrIndexMask = 0xff;

// TRISC errors
// Values are taken from https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/564527286/Error+Aggregator, additional
// error information is available in `ERR_DATA` register for errors 0-3.
//
// Block ID, error_code[13:8]. The index in [7:0] only means anything alongside its block, see
// the enums below.
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

// Index for TriscErrors::ERROR_TRISC0..3. These are the bases of the old 22-26 / 31-34 / 40-44
// ranges on https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/461602877. Those ranges used
// to spread across the TRISCs, they don't any more, so take the TRISC from error_code[13:8] and
// not from the value here. Note that page predates the current encoding, so treat the value
// meanings below as the reference rather than its table. Reported PC lands in `ERR_DATA`.
//
// Only one gets reported, in this order:
//   L1_ILLEGAL_ACCESS > STACK_OVERFLOW > MEM_READ_NO_RESPONSE > TTI_BUFFER_HANG
// so a lower one is invisible while a higher one is live.
enum class TriscRiscErrors : uint32_t {
    // Instruction buffer sat idle for DBG_IBUFFER_TIMEOUT cycles. Opt-in: nothing is counted
    // unless DBG_IBUFFER_CNT_EN is set, so this stays quiet on a normal run.
    TTI_BUFFER_HANG = 22,
    // A read went out and nothing answered within CSR_TIMEOUT_COUNT cycles (65536 after reset).
    // Look at the target rather than the core: a bad NOC address, a core that isn't running,
    // unmapped L1. Beware that the load still completes, with zero data, so the kernel carries
    // on with a bogus value and whatever breaks next is a knock-on. Nothing to do with
    // TTI_BUFFER_HANG despite sitting inside the old 22-26 range.
    MEM_READ_NO_RESPONSE = 25,
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
