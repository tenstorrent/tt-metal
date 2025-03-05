// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Aux variable used to align addresses to platform specific width. WH requires 32B alignment.
constexpr auto NOC_ADDRESS_ALIGNMENT = 32;

namespace l1_mem
{

struct mailbox_type
{
    constexpr static int TRISC0 = 0;
    constexpr static int TRISC1 = 1;
    constexpr static int TRISC2 = 2;
    constexpr static int NCRISC = 3;
    constexpr static int BRISC  = 4;
};

struct address_map
{
    // Sizes
    static constexpr std::uint32_t FIRMWARE_SIZE             = 20 * 1024;            // 20KB = 7KB + 1KB zeros + 12KB perf buffers
    static constexpr std::uint32_t L1_BARRIER_SIZE           = 0x20;                 // 32 bytes reserved for L1 Barrier
    static constexpr std::uint32_t BRISC_FIRMWARE_SIZE       = 7 * 1024 + 512 + 768; // Taking an extra 768B from perf buffer space
    static constexpr std::uint32_t ZEROS_SIZE                = 512;
    static constexpr std::uint32_t NCRISC_FIRMWARE_SIZE      = 32 * 1024; // 16KB in L0, 16KB in L1
    static constexpr std::uint32_t TRISC0_SIZE               = 20 * 1024; // 20KB = 16KB + 4KB local memory
    static constexpr std::uint32_t TRISC1_SIZE               = 16 * 1024; // 16KB = 12KB + 4KB local memory
    static constexpr std::uint32_t TRISC2_SIZE               = 20 * 1024; // 20KB = 16KB + 4KB local memory
    static constexpr std::uint32_t TRISC_LOCAL_MEM_SIZE      = 4 * 1024;  //
    static constexpr std::uint32_t NCRISC_LOCAL_MEM_SIZE     = 4 * 1024;  //
    static constexpr std::uint32_t NCRISC_L1_SCRATCH_SIZE    = 4 * 1024;  //
    static constexpr std::uint32_t NCRISC_L1_CODE_SIZE       = 16 * 1024; // Size of code block that is L1 resident
    static constexpr std::uint32_t NCRISC_IRAM_CODE_SIZE     = 16 * 1024; // Size of code block that is IRAM resident
    static constexpr std::uint32_t NCRISC_DATA_SIZE          = 4 * 1024;  // 4KB
    static constexpr std::uint32_t EPOCH_RUNTIME_CONFIG_SIZE = 128;       //
    static constexpr std::uint32_t OVERLAY_BLOB_SIZE         = (64 * 1024) - EPOCH_RUNTIME_CONFIG_SIZE;
    static constexpr std::uint32_t TILE_HEADER_BUF_SIZE      = 32 * 1024; //
    static constexpr std::uint32_t NCRISC_L1_EPOCH_Q_SIZE    = 32;
    static constexpr std::uint32_t FW_L1_BLOCK_SIZE =
        FIRMWARE_SIZE + NCRISC_FIRMWARE_SIZE + TRISC0_SIZE + TRISC1_SIZE + TRISC2_SIZE + OVERLAY_BLOB_SIZE + EPOCH_RUNTIME_CONFIG_SIZE + TILE_HEADER_BUF_SIZE;

    // Base addresses
    static constexpr std::uint32_t FIRMWARE_BASE         = 0;
    static constexpr std::uint32_t L1_BARRIER_BASE       = 0x16dfc0;
    static constexpr std::uint32_t ZEROS_BASE            = FIRMWARE_BASE + BRISC_FIRMWARE_SIZE;
    static constexpr std::uint32_t NCRISC_FIRMWARE_BASE  = FIRMWARE_BASE + FIRMWARE_SIZE;
    static constexpr std::uint32_t NCRISC_L1_CODE_BASE   = NCRISC_FIRMWARE_BASE + NCRISC_IRAM_CODE_SIZE;
    static constexpr std::uint32_t NCRISC_LOCAL_MEM_BASE = NCRISC_FIRMWARE_BASE + NCRISC_FIRMWARE_SIZE - NCRISC_LOCAL_MEM_SIZE; // Copy of the local memory
    static constexpr std::uint32_t NCRISC_L1_SCRATCH_BASE =
        NCRISC_FIRMWARE_BASE +
        0x200; // L1 Scratch used by NCRISC sized NCRISC_L1_SCRATCH_SIZE, skip 0x200 because some of the beginning of NCRISC is used .e.g. TEST_MAILBOX
    static constexpr std::uint32_t NCRISC_L1_CONTEXT_BASE =
        NCRISC_FIRMWARE_BASE + 0x20; // If changing make sure to modify src/firmware/riscv/targets/ncrisc/contextASM.S
    static constexpr std::uint32_t NCRISC_L1_DRAM_POLLING_CTRL_BASE = NCRISC_FIRMWARE_BASE + 0x40;
    static constexpr std::uint32_t NCRISC_PERF_QUEUE_HEADER_SIZE    = 8 * 8;                                         // Half of this value must be 32B aligned
    static constexpr std::uint32_t NCRISC_PERF_QUEUE_HEADER_ADDR    = NCRISC_FIRMWARE_BASE + NCRISC_L1_SCRATCH_SIZE; // L1 Performance Buffer used by NCRISC
    static constexpr std::uint32_t NCRISC_L1_PERF_BUF_BASE =
        NCRISC_PERF_QUEUE_HEADER_ADDR + NCRISC_PERF_QUEUE_HEADER_SIZE;                                      // L1 Performance Buffer used by NCRISC
    static constexpr std::uint32_t NCRISC_PERF_BUF_SIZE_LEVEL_0 = 640;                                      // smaller buffer size for limited logging
    static constexpr std::uint32_t NCRISC_PERF_BUF_SIZE_LEVEL_1 = 4 * 1024 - NCRISC_PERF_QUEUE_HEADER_SIZE; // NCRISC performance buffer
    static constexpr std::uint32_t NCRISC_L1_EPOCH_Q_BASE       = NCRISC_L1_PERF_BUF_BASE + NCRISC_PERF_BUF_SIZE_LEVEL_1; // Epoch Q start in L1.

    static constexpr std::uint32_t TRISC_BASE                = NCRISC_FIRMWARE_BASE + NCRISC_FIRMWARE_SIZE;
    static constexpr std::uint32_t TRISC0_BASE               = NCRISC_FIRMWARE_BASE + NCRISC_FIRMWARE_SIZE;
    static constexpr std::uint32_t TRISC0_LOCAL_MEM_BASE     = TRISC0_BASE + TRISC0_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
    static constexpr std::uint32_t TRISC1_BASE               = TRISC0_BASE + TRISC0_SIZE;
    static constexpr std::uint32_t TRISC1_LOCAL_MEM_BASE     = TRISC1_BASE + TRISC1_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
    static constexpr std::uint32_t TRISC2_BASE               = TRISC1_BASE + TRISC1_SIZE;
    static constexpr std::uint32_t TRISC2_LOCAL_MEM_BASE     = TRISC2_BASE + TRISC2_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
    static constexpr std::uint32_t EPOCH_RUNTIME_CONFIG_BASE = TRISC2_BASE + TRISC2_SIZE + TILE_HEADER_BUF_SIZE;
    static constexpr std::uint32_t OVERLAY_BLOB_BASE         = EPOCH_RUNTIME_CONFIG_BASE + EPOCH_RUNTIME_CONFIG_SIZE;
    static constexpr std::uint32_t DATA_BUFFER_SPACE_BASE    = EPOCH_RUNTIME_CONFIG_BASE + EPOCH_RUNTIME_CONFIG_SIZE + OVERLAY_BLOB_SIZE;

    static_assert(FIRMWARE_BASE % NOC_ADDRESS_ALIGNMENT == 0, "FIRMWARE_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(NCRISC_FIRMWARE_BASE % NOC_ADDRESS_ALIGNMENT == 0, "NCRISC_FIRMWARE_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(TRISC0_BASE % NOC_ADDRESS_ALIGNMENT == 0, "TRISC0_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(TRISC1_BASE % NOC_ADDRESS_ALIGNMENT == 0, "TROSC1_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(TRISC2_BASE % NOC_ADDRESS_ALIGNMENT == 0, "TRISC2_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(EPOCH_RUNTIME_CONFIG_BASE % NOC_ADDRESS_ALIGNMENT == 0, "EPOCH_RUNTIME_CONFIG_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(OVERLAY_BLOB_BASE % NOC_ADDRESS_ALIGNMENT == 0, "OVERLAY_BLOB_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(DATA_BUFFER_SPACE_BASE % NOC_ADDRESS_ALIGNMENT == 0, "DATA_BUFFER_SPACE_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");
    static_assert(L1_BARRIER_BASE % NOC_ADDRESS_ALIGNMENT == 0, "L1_BARRIER_BASE must be aligned to NOC_ADDRESS_ALIGNMENT");

    // Trisc Mailboxes
    static constexpr std::uint32_t TRISC_L1_MAILBOX_OFFSET = 4;
    static constexpr std::uint32_t BRISC_L1_MAILBOX_OFFSET = 4;
    static constexpr std::uint32_t NRISC_L1_MAILBOX_OFFSET = 4;

    static constexpr std::uint32_t TRISC0_MAILBOX_BASE = TRISC0_BASE + TRISC_L1_MAILBOX_OFFSET;
    static constexpr std::uint32_t TRISC1_MAILBOX_BASE = TRISC1_BASE + TRISC_L1_MAILBOX_OFFSET;
    static constexpr std::uint32_t TRISC2_MAILBOX_BASE = TRISC2_BASE + TRISC_L1_MAILBOX_OFFSET;

    static constexpr std::uint32_t FW_MAILBOX_BASE        = 32;
    static constexpr std::uint32_t DEBUG_MAILBOX_BUF_BASE = 112;

    static constexpr std::uint32_t FW_MAILBOX_BUF_SIZE    = 64;
    static constexpr std::uint32_t DEBUG_MAILBOX_BUF_SIZE = 64; // For each T0/T1/T2/FW

    // Upper 2KB of local space is used as debug buffer
    static constexpr std::uint32_t DEBUG_BUFFER_SIZE        = 2 * 1024;
    static constexpr std::uint32_t TRISC0_DEBUG_BUFFER_BASE = TRISC0_LOCAL_MEM_BASE + DEBUG_BUFFER_SIZE;
    static constexpr std::uint32_t TRISC1_DEBUG_BUFFER_BASE = TRISC1_LOCAL_MEM_BASE + DEBUG_BUFFER_SIZE;
    static constexpr std::uint32_t TRISC2_DEBUG_BUFFER_BASE = TRISC2_LOCAL_MEM_BASE + DEBUG_BUFFER_SIZE;

    static constexpr std::uint32_t MAX_SIZE            = 1499136;
    static constexpr std::uint32_t MAX_L1_LOADING_SIZE = 1 * 1024 * 1024;

    static constexpr std::uint32_t RISC_LOCAL_MEM_BASE = 0xffb00000; // Actaul local memory address as seen from risc firmware
                                                                     // As part of the init risc firmware will copy local memory data from
                                                                     // l1 locations listed above into internal local memory that starts
                                                                     // at RISC_LOCAL_MEM_BASE address

    static constexpr std::uint32_t NCRISC_IRAM_MEM_BASE = 0xffc00000; // NCRISC instruction RAM base address

    // Perf buffer (FIXME - update once location of the perf data buffer is finalized)
    // Parameter UNPACK_PACK_PERF_BUF_SIZE_LEVEL_1 assumes the following PERF_BUF_SIZE = 12KB - 768
    static constexpr std::uint32_t PERF_BUF_SIZE = FIRMWARE_SIZE - BRISC_FIRMWARE_SIZE - ZEROS_SIZE;

    // This value must be equal to the sum of all all the subsequent sizes in this section
    static constexpr std::uint32_t PERF_TOTAL_SETUP_BUFFER_SIZE = 64;
    // Queue header below is used for the concurrent performance trace
    static constexpr std::uint32_t PERF_QUEUE_HEADER_SIZE = 16;
    // Input and output overlay decoupling masks
    static constexpr std::uint32_t PERF_RISC_MAILBOX_SIZE = 8;
    // Signal to ncrisc that the performance buffer has been drained.
    static constexpr std::uint32_t PERF_RESET_PTR_MAILBOX_SIZE = 4;
    // Signal to all cores within a chip that the epoch comamnd for all cores has been sent;
    static constexpr std::uint32_t PERF_ANALYZER_COMMAND_START_PTR_SIZE = 8;
    static constexpr std::uint32_t PERF_ANALYZER_COMMAND_START_VAL_SIZE = 4;
    static constexpr std::uint32_t PERF_UNUSED_SIZE                     = 24;

    static constexpr std::uint32_t MATH_PERF_BUF_SIZE                = 64;
    static constexpr std::uint32_t BRISC_PERF_BUF_SIZE               = 640; // Half of this value must be 32B aligned
    static constexpr std::uint32_t UNPACK_PACK_PERF_BUF_SIZE_LEVEL_0 = 640; // smaller buffer size for limited logging
    static constexpr std::uint32_t UNPACK_PACK_PERF_BUF_SIZE_LEVEL_1 =
        (12 * 1024 - 768) / 2 - MATH_PERF_BUF_SIZE / 2 - (PERF_TOTAL_SETUP_BUFFER_SIZE) / 2 - BRISC_PERF_BUF_SIZE / 2;

    static constexpr std::uint32_t PERF_QUEUE_HEADER_ADDR               = FIRMWARE_BASE + BRISC_FIRMWARE_SIZE + ZEROS_SIZE;
    static constexpr std::uint32_t PERF_RISC_MAILBOX_ADDR               = PERF_QUEUE_HEADER_ADDR + PERF_QUEUE_HEADER_SIZE;
    static constexpr std::uint32_t PERF_RESET_PTR_MAILBOX_ADDR          = PERF_RISC_MAILBOX_ADDR + PERF_RISC_MAILBOX_SIZE;
    static constexpr std::uint32_t PERF_ANALYZER_COMMAND_START_PTR_ADDR = PERF_RESET_PTR_MAILBOX_ADDR + PERF_RESET_PTR_MAILBOX_SIZE;
    static constexpr std::uint32_t PERF_ANALYZER_COMMAND_START_VAL_ADDR = PERF_ANALYZER_COMMAND_START_PTR_ADDR + PERF_ANALYZER_COMMAND_START_PTR_SIZE;
    static constexpr std::uint32_t BRISC_PERF_BUF_BASE_ADDR = PERF_ANALYZER_COMMAND_START_VAL_SIZE + PERF_UNUSED_SIZE + PERF_ANALYZER_COMMAND_START_VAL_ADDR;
    static constexpr std::uint32_t MATH_PERF_BUF_BASE_ADDR  = BRISC_PERF_BUF_BASE_ADDR + BRISC_PERF_BUF_SIZE;
    static constexpr std::uint32_t UNPACK_PACK_PERF_BUF_BASE_ADDR = MATH_PERF_BUF_BASE_ADDR + MATH_PERF_BUF_SIZE;
    static constexpr std::int32_t PERF_NUM_THREADS                = 5;

    static constexpr std::uint32_t PERF_QUEUE_PTRS    = PERF_QUEUE_HEADER_ADDR;
    static constexpr std::uint32_t PERF_THREAD_HEADER = PERF_QUEUE_PTRS + 8;
    static constexpr std::uint32_t PERF_WR_PTR_COPY   = PERF_QUEUE_PTRS + 12;

    static constexpr std::uint32_t WALL_CLOCK_L = 0xFFB121F0;
    static constexpr std::uint32_t WALL_CLOCK_H = 0xFFB121F8;
};
} // namespace l1_mem
