// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/device_command.hpp"

#include <atomic>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/env_lib.hpp"

bool DeviceCommand::zero_init_disable = tt::parse_env<bool>("TT_METAL_ZERO_INIT_DISABLE", false);

DeviceCommand::DeviceCommand(void *cmd_region, uint32_t cmd_sequence_sizeB) : cmd_sequence_sizeB(cmd_sequence_sizeB), cmd_region(cmd_region), cmd_write_offsetB(0) {
    TT_FATAL(cmd_sequence_sizeB % sizeof(uint32_t) == 0, "Command sequence size B={} is not {}-byte aligned", cmd_sequence_sizeB, sizeof(uint32_t));
}

DeviceCommand::DeviceCommand(uint32_t cmd_sequence_sizeB) : cmd_sequence_sizeB(cmd_sequence_sizeB), cmd_write_offsetB(0) {
    TT_FATAL(cmd_sequence_sizeB % sizeof(uint32_t) == 0, "Command sequence size B={} is not {}-byte aligned", cmd_sequence_sizeB, sizeof(uint32_t));
    this->cmd_region_vector.resize(cmd_sequence_sizeB / sizeof(uint32_t), 0);
    this->cmd_region = this->cmd_region_vector.data();
}

void DeviceCommand::deepcopy(const DeviceCommand &other) {
    if (other.cmd_region_vector.empty() and other.cmd_region != nullptr) {
        this->cmd_region = other.cmd_region;
    } else if (not other.cmd_region_vector.empty()) {
        TT_ASSERT(other.cmd_region != nullptr);
        this->cmd_region = this->cmd_region_vector.data();
        memcpy(this->cmd_region, other.cmd_region_vector.data(), this->cmd_sequence_sizeB);
    }
}

DeviceCommand &DeviceCommand::operator=(DeviceCommand &&other) {
    this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
    this->cmd_write_offsetB = other.cmd_write_offsetB;
    this->cmd_region_vector = other.cmd_region_vector;
    this->deepcopy(other);
    return *this;
}

DeviceCommand &DeviceCommand::operator=(const DeviceCommand &other) {
    this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
    this->cmd_write_offsetB = other.cmd_write_offsetB;
    this->cmd_region_vector = other.cmd_region_vector;
    this->deepcopy(other);
    return *this;
}

DeviceCommand::DeviceCommand(const DeviceCommand &other) : cmd_sequence_sizeB(other.cmd_sequence_sizeB), cmd_write_offsetB(other.cmd_write_offsetB), cmd_region_vector(other.cmd_region_vector) {
    this->deepcopy(other);
}

DeviceCommand::DeviceCommand(DeviceCommand &&other) : cmd_sequence_sizeB(other.cmd_sequence_sizeB), cmd_write_offsetB(other.cmd_write_offsetB), cmd_region_vector(other.cmd_region_vector) {
    this->deepcopy(other);
}

void DeviceCommand::add_dispatch_wait(uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count) {
    CQPrefetchCmd *relay_wait = this->reserve_space<CQPrefetchCmd *>(sizeof(CQPrefetchCmd));

    relay_wait->base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
    relay_wait->relay_inline.length = sizeof(CQDispatchCmd);
    relay_wait->relay_inline.stride = CQ_PREFETCH_CMD_BARE_MIN_SIZE;

    CQDispatchCmd *wait_cmd = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

    wait_cmd->base.cmd_id = CQ_DISPATCH_CMD_WAIT;
    wait_cmd->wait.barrier = barrier;
    wait_cmd->wait.notify_prefetch = false;
    wait_cmd->wait.addr = address;
    wait_cmd->wait.count = count;
    wait_cmd->wait.clear_count = clear_count;
}

void DeviceCommand::add_dispatch_wait_with_prefetch_stall(uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count) {
    this->add_dispatch_wait(barrier, address, count, clear_count);

    // modify the added CQ_DISPATCH_CMD_WAIT command to set notify prefetch
    uint32_t dispatch_wait_offsetB = this->cmd_write_offsetB - sizeof(CQDispatchCmd);
    CQDispatchCmd *wait_cmd = (CQDispatchCmd*)((char *)this->cmd_region + dispatch_wait_offsetB);
    wait_cmd->wait.notify_prefetch = true;

    uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
    CQPrefetchCmd *stall_cmd = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);
    stall_cmd->base.cmd_id = CQ_PREFETCH_CMD_STALL;
}

void DeviceCommand::add_prefetch_relay_inline(bool flush, uint32_t lengthB) {
    CQPrefetchCmd *relay_write = this->reserve_space<CQPrefetchCmd *>(sizeof(CQPrefetchCmd));

    relay_write->base.cmd_id = flush ? CQ_PREFETCH_CMD_RELAY_INLINE : CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH;
    relay_write->relay_inline.length = lengthB;
    relay_write->relay_inline.stride = align(sizeof(CQPrefetchCmd) + lengthB, PCIE_ALIGNMENT);
}

void DeviceCommand::add_prefetch_relay_linear(uint32_t noc_xy_addr, uint32_t lengthB, uint32_t addr) {
    uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
    CQPrefetchCmd *relay_linear_cmd = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

    relay_linear_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR;
    relay_linear_cmd->relay_linear.noc_xy_addr = noc_xy_addr;
    relay_linear_cmd->relay_linear.length = lengthB;
    relay_linear_cmd->relay_linear.addr = addr;
}

void DeviceCommand::add_prefetch_relay_paged(uint8_t is_dram, uint8_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages, uint16_t length_adjust) {
    uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
    CQPrefetchCmd *relay_paged_cmd = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

    relay_paged_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;
    relay_paged_cmd->relay_paged.packed_page_flags = (is_dram << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT) | (start_page << CQ_PREFETCH_RELAY_PAGED_START_PAGE_SHIFT);
    relay_paged_cmd->relay_paged.length_adjust = length_adjust;
    relay_paged_cmd->relay_paged.base_addr = base_addr;
    relay_paged_cmd->relay_paged.page_size = page_size;
    relay_paged_cmd->relay_paged.pages = pages;
}

void DeviceCommand::add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages) {
    uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
    CQPrefetchCmd *exec_buf_cmd = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

    exec_buf_cmd->base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF;
    exec_buf_cmd->exec_buf.base_addr = base_addr;
    exec_buf_cmd->exec_buf.log_page_size = log_page_size;
    exec_buf_cmd->exec_buf.pages = pages;
}

void DeviceCommand::add_dispatch_terminate() {
    this->add_prefetch_relay_inline(true, sizeof(CQDispatchCmd));

    CQDispatchCmd *terminate_cmd = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

    terminate_cmd->base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
}

void DeviceCommand::add_prefetch_terminate() {
    uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
    CQPrefetchCmd *terminate_cmd = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

    terminate_cmd->base.cmd_id = CQ_PREFETCH_CMD_TERMINATE;
}

void DeviceCommand::add_prefetch_exec_buf_end() {
    uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
    CQPrefetchCmd *exec_buf_end_cmd = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

    exec_buf_end_cmd->base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF_END;
}

void DeviceCommand::add_data(const void *data, uint32_t data_size_to_copyB, uint32_t cmd_write_offset_incrementB) {
    this->validate_cmd_write(cmd_write_offset_incrementB);
    memcpy((char *)this->cmd_region + this->cmd_write_offsetB, data, data_size_to_copyB);
    this->cmd_write_offsetB += cmd_write_offset_incrementB;
}
