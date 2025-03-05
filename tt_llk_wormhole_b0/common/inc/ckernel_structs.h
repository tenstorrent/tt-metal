// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel
{

// Semaphores mapping and trisc space -> tensix space conversion
struct semaphore
{
    constexpr static uint32_t MATH_PACK           = 1; // math <-> pack sync on dest register
    constexpr static uint32_t UNPACK_TO_DEST      = 2; // unpack <-> math sync on unpack to dest
    constexpr static uint32_t UNPACK_OPERAND_SYNC = 3; // unpack <-> pack, math sync on operand get/release
    constexpr static uint32_t PACK_DONE           = 4; // Wait for beinning and end of each pack-iteration. For recording perf events and inserting delay.
    constexpr static uint32_t UNPACK_SYNC         = 5; // trisc <-> unpack sync on hw kernel
    // Wait for beinning and end of each unpack or math iteration. For recording perf events and inserting delay.
    // This semaphore should only be used for either unpack or math. Not both at the same time.
    constexpr static uint32_t UNPACK_MATH_DONE = 6;
    constexpr static uint32_t MATH_DONE        = 7; // wait for math to finish when unpacking to dest

    constexpr static uint16_t t6_sem(const uint8_t sem_index)
    {
        return (1 << sem_index);
    }
};

struct mutex
{
    constexpr static uint32_t REG_RMW = 0; // used for atomic register read-modify-write from different threads
};

constexpr uint8_t PC_BUF_SEMAPHORE_BASE = 8;  // base address for semaphores in PC buffer
constexpr uint8_t MATH_HALF_DEST_SIZE   = 32; // arch specific 1/2 dest registers size in 16x16 faces
constexpr uint8_t MAX_CONFIG_STATES     = 2;

// Firmware messages to ckernels
enum firmware_msg_e
{
    FLIP_STATE_ID        = 1,
    RUN_INSTRUCTIONS     = 2,
    RESET_DEST_OFFSET_ID = 3,
    SET_PERF_SCRATCH     = 4
};

} // namespace ckernel
