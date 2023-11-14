/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

namespace epoch_queue {
/**
 * @brief Configuration parameters of the Epoch Queue on the "Silicon" device.
*/
    static constexpr std::int32_t EPOCH_Q_NUM_SLOTS = 32; // needs to match param with same name in ncrisc - parametrized by arch
    static constexpr std::int32_t EPOCH_Q_SLOT_SIZE = 32; // needs to match param with same name in ncrisc - parametrized by arch
    static constexpr std::int32_t GridSizeRow = 16;
    static constexpr std::int32_t GridSizeCol = 16;
    static constexpr std::int32_t EpochEndReached = 0xFFFFFFFF;


/**
 * @brief Silicon device epoch queue command interpreted by NCRISC/ERISC FW.
 */
    enum EpochQueueCmd
    {
        EpochCmdValid = 0x1,
        EpochCmdNotValid = 0x2,
        EpochCmdIOQueueUpdate = 0x3,
        EpochCmdEndProgram = 0xF,
    };

    struct IOQueueUpdateCmdInfo {
        uint64_t queue_header_addr;

        uint8_t num_buffers;
        uint8_t reader_index;
        uint8_t num_readers;
        // In full update mode: update_mask = 0xff
        uint8_t update_mask;

        uint32_t header[5]; // The first 5 words of the header
    };

    static constexpr std::int32_t EPOCH_Q_WRPTR_OFFSET = 4;
    static constexpr std::int32_t EPOCH_Q_RDPTR_OFFSET = 0;
    static constexpr std::int32_t EPOCH_Q_SLOTS_OFFSET = 32;

    static constexpr std::int32_t EPOCH_TABLE_ENTRY_SIZE_BYTES = EPOCH_Q_NUM_SLOTS*EPOCH_Q_SLOT_SIZE+EPOCH_Q_SLOTS_OFFSET;
    static constexpr std::int32_t QUEUE_UPDATE_BLOB_SIZE_BYTES = 120 * 8;

    static constexpr std::int32_t DRAM_PERF_SCRATCH_SIZE_BYTES =   8 * 1024 * 1024;
    // Starting from address 0, epoch queues start at 40MByte - sizeof(All epoch queues on the chip)
    // i.e top of epoch q table is @ 40MByte.
    static constexpr std::int32_t EPOCH_TABLE_DRAM_ADDR = DRAM_PERF_SCRATCH_SIZE_BYTES-GridSizeCol*GridSizeRow*EPOCH_TABLE_ENTRY_SIZE_BYTES;

} // namespace epoch_queue
