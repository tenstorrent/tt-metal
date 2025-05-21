// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the
    // completion region Equation for issue fifo size is | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t cq_start);

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B
    // aligned and remaining space is dedicated for completion queue Smaller issue queues can lead to more stalls for
    // applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t cq_start = 0;
    const uint32_t command_completion_region_size = 0;
    const uint32_t command_issue_region_size = 0;
    const uint8_t id = 0;

    uint32_t issue_fifo_size = 0;
    uint32_t issue_fifo_limit = 0;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr = 0;
    bool issue_fifo_wr_toggle = false;

    uint32_t completion_fifo_size = 0;
    uint32_t completion_fifo_limit = 0;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr = 0;
    bool completion_fifo_rd_toggle = false;

    // TODO add the host addresses from dispatch constants in here
};

}  // namespace tt::tt_metal
