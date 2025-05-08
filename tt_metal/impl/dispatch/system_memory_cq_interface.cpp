// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "system_memory_cq_interface.hpp"

#include "assert.hpp"
#include "command_queue_common.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch_settings.hpp"

namespace tt::tt_metal {

SystemMemoryCQInterface::SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t cq_start) :
    cq_start(cq_start),
    command_completion_region_size(
        (((cq_size - cq_start) / DispatchSettings::TRANSFER_PAGE_SIZE) / 4) * DispatchSettings::TRANSFER_PAGE_SIZE),
    command_issue_region_size((cq_size - cq_start) - this->command_completion_region_size),
    issue_fifo_size(command_issue_region_size >> 4),
    issue_fifo_limit(
        ((cq_start + this->command_issue_region_size) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4),
    completion_fifo_size(command_completion_region_size >> 4),
    completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
    offset(get_absolute_cq_offset(channel, cq_id, cq_size)),
    id(cq_id) {
    TT_ASSERT(
        this->command_completion_region_size % MetalContext::instance().hal().get_alignment(HalMemType::HOST) == 0 and
            this->command_issue_region_size % MetalContext::instance().hal().get_alignment(HalMemType::HOST) == 0,
        "Issue queue and completion queue need to be {}B aligned!",
        MetalContext::instance().hal().get_alignment(HalMemType::HOST));
    TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
    // queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    this->issue_fifo_wr_ptr = (this->cq_start + this->offset) >> 4;  // In 16B words
    this->issue_fifo_wr_toggle = false;

    this->completion_fifo_rd_ptr = this->issue_fifo_limit;
    this->completion_fifo_rd_toggle = false;
}

}  // namespace tt::tt_metal
