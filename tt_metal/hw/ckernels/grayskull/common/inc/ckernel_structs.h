// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "circular_buffer.h"
#include "hostdevcommon/kernel_structs.h"

namespace ckernel
{

// Semaphores mapping and trisc space -> tensix space conversion
struct semaphore
{
    constexpr static uint32_t MATH_PACK = 1;   // math <-> pack sync on dest register
    constexpr static uint32_t UNPACK_PACK = 2; // pack <-> unpack sync on scratch buffer
    constexpr static uint32_t UNPACK_OPERAND_SYNC = 3; // unpack <-> pack, math sync on operand get/release
    constexpr static uint32_t PACK_DONE = 4; // Wait for beinning and end of each pack-iteration. For recording perf events.
    constexpr static uint32_t UNPACK_SYNC = 5; // trisc <-> unpack sync on hw kernel
    constexpr static uint32_t PACK_SYNC = 6; // thcon <-> pack sync on tile write to l1
    // Zahi may be using this register, although I'm not sure if it's the same set of stream registers...
    // Should ask MT
    constexpr static uint32_t UNPACK_PACK_CONFIG_SYNC = 7; // unpack <-> pack config sync to safely change common registers

    constexpr static uint16_t t6_sem(const uint8_t sem_index)
    {
        return (1 << sem_index);
    }
};

struct mutex
{
    constexpr static uint32_t REG_RMW = 0;   // used for atomic register read-modify-write from different threads
};

constexpr uint8_t PC_BUF_SEMAPHORE_BASE = 8; // base address for semaphores in PC buffer
constexpr uint8_t MATH_HALF_DEST_SIZE = 32;  // arch specific 1/2 dest registers size in 16x16 faces
constexpr uint8_t MAX_CONFIG_STATES = 2;

// Firmware messages to ckernels
enum firmware_msg_e
{
    FLIP_STATE_ID = 1,
    RUN_INSTRUCTIONS = 2,
    RESET_DEST_OFFSET_ID = 3,
    SET_PERF_SCRATCH = 4
};

} // namespace ckelimitrnel
