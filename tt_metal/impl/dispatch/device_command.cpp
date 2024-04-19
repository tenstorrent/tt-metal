// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/device_command.hpp"

#include <atomic>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/env_lib.hpp"

bool DeviceCommand::zero_init_disable = tt::parse_env<bool>("TT_METAL_ZERO_INIT_DISABLE", false);

DeviceCommand::DeviceCommand(uint32_t cmd_sequence_sizeB) : cmd_write_idx(0) {
    this->cmd_sequence.resize(cmd_sequence_sizeB / sizeof(uint32_t), 0);
}

void DeviceCommand::add_dispatch_wait(uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count) {
    TT_ASSERT(this->cmd_write_idx + (sizeof(CQPrefetchCmd) / sizeof(uint32_t)) < this->cmd_sequence.size()); // turn to api

    CQPrefetchCmd relay_wait;
    if (!zero_init_disable) DeviceCommand::zero(relay_wait);

    relay_wait.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
    relay_wait.relay_inline.length = sizeof(CQDispatchCmd);
    relay_wait.relay_inline.stride = CQ_PREFETCH_CMD_BARE_MIN_SIZE;

    this->write_to_cmd_sequence(&relay_wait, sizeof(CQPrefetchCmd));

    CQDispatchCmd wait_cmd;
    if (!zero_init_disable) DeviceCommand::zero(wait_cmd);

    wait_cmd.base.cmd_id = CQ_DISPATCH_CMD_WAIT;
    wait_cmd.wait.barrier = barrier;
    wait_cmd.wait.notify_prefetch = false;
    wait_cmd.wait.addr = address;
    wait_cmd.wait.count = count;
    wait_cmd.wait.clear_count = clear_count;

    this->write_to_cmd_sequence(&wait_cmd, sizeof(CQDispatchCmd));
}

void DeviceCommand::add_dispatch_wait_with_prefetch_stall(uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count) {
    this->add_dispatch_wait(barrier, address, count, clear_count);

    uint32_t dispatch_wait_idx = this->cmd_write_idx - (sizeof(CQDispatchCmd) / sizeof(uint32_t));
    CQDispatchCmd *wait_cmd = (CQDispatchCmd*)(this->cmd_sequence.data() + dispatch_wait_idx);
    wait_cmd->wait.notify_prefetch = true;

    CQPrefetchCmd stall_cmd;
    if (!zero_init_disable) DeviceCommand::zero(stall_cmd);

    stall_cmd.base.cmd_id = CQ_PREFETCH_CMD_STALL;

    this->write_to_cmd_sequence(&stall_cmd, sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
}

void DeviceCommand::add_prefetch_relay_inline(bool flush, uint32_t lengthB) {
    CQPrefetchCmd relay_write;
    if (!zero_init_disable) DeviceCommand::zero(relay_write);

    relay_write.base.cmd_id = flush ? CQ_PREFETCH_CMD_RELAY_INLINE : CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH;
    relay_write.relay_inline.length = lengthB;
    relay_write.relay_inline.stride = align(sizeof(CQPrefetchCmd) + lengthB, PCIE_ALIGNMENT);

    this->write_to_cmd_sequence(&relay_write, sizeof(CQPrefetchCmd));
}

void DeviceCommand::add_prefetch_relay_linear(uint32_t noc_xy_addr, uint32_t lengthB, uint32_t addr) {
    CQPrefetchCmd relay_linear_cmd;
    if (!zero_init_disable) DeviceCommand::zero(relay_linear_cmd);

    relay_linear_cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR;
    relay_linear_cmd.relay_linear.noc_xy_addr = noc_xy_addr;
    relay_linear_cmd.relay_linear.length = lengthB;
    relay_linear_cmd.relay_linear.addr = addr;

    this->write_to_cmd_sequence(&relay_linear_cmd, sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
}

void DeviceCommand::add_prefetch_relay_paged(uint8_t is_dram, uint8_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages, uint16_t length_adjust) {
    CQPrefetchCmd relay_paged_cmd;
    if (!zero_init_disable) DeviceCommand::zero(relay_paged_cmd);

    relay_paged_cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;
    relay_paged_cmd.relay_paged.packed_page_flags = (is_dram << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT) | (start_page << CQ_PREFETCH_RELAY_PAGED_START_PAGE_SHIFT);
    relay_paged_cmd.relay_paged.length_adjust = length_adjust;
    relay_paged_cmd.relay_paged.base_addr = base_addr;
    relay_paged_cmd.relay_paged.page_size = page_size;
    relay_paged_cmd.relay_paged.pages = pages;

    this->write_to_cmd_sequence(&relay_paged_cmd, sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
}

void DeviceCommand::add_dispatch_write_linear(bool flush_prefetch, uint8_t num_mcast_dests, uint32_t noc_xy_addr, uint32_t addr, uint32_t data_sizeB, const void *data) {
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    CQDispatchCmd write_cmd;
    if (!zero_init_disable) DeviceCommand::zero(write_cmd);

    write_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    write_cmd.write_linear.num_mcast_dests = num_mcast_dests;
    write_cmd.write_linear.noc_xy_addr = noc_xy_addr;
    write_cmd.write_linear.addr = addr;
    write_cmd.write_linear.length = data_sizeB;

    this->write_to_cmd_sequence(&write_cmd, sizeof(CQDispatchCmd));

    if (data != nullptr) {
        this->write_to_cmd_sequence(data, data_sizeB, PCIE_ALIGNMENT);
    }
}

void DeviceCommand::add_dispatch_write_paged(bool flush_prefetch, uint8_t is_dram, uint16_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages, const void *data) {
    uint32_t data_sizeB = page_size * pages;
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    CQDispatchCmd write_cmd;
    if (!zero_init_disable) DeviceCommand::zero(write_cmd);

    write_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
    write_cmd.write_paged.is_dram = is_dram;
    write_cmd.write_paged.start_page = start_page;
    write_cmd.write_paged.base_addr = base_addr;
    write_cmd.write_paged.page_size = page_size;
    write_cmd.write_paged.pages = pages;

    this->write_to_cmd_sequence(&write_cmd, sizeof(CQDispatchCmd));

    if (data != nullptr) {
        this->write_to_cmd_sequence(data, data_sizeB, PCIE_ALIGNMENT);
    }
}

void DeviceCommand::add_dispatch_write_host(bool flush_prefetch, uint32_t data_sizeB, const void *data) {
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    CQDispatchCmd write_cmd;
    if (!zero_init_disable) DeviceCommand::zero(write_cmd);

    write_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
    write_cmd.write_linear_host.length = payload_sizeB; // CQ_DISPATCH_CMD_WRITE_LINEAR_HOST writes dispatch cmd back to completion queue

    this->write_to_cmd_sequence(&write_cmd, sizeof(CQDispatchCmd));

    if (data != nullptr) {
        this->write_to_cmd_sequence(data, data_sizeB, PCIE_ALIGNMENT);
    }
}

void DeviceCommand::add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages) {
    CQPrefetchCmd exec_buf_cmd;
    if (!zero_init_disable) DeviceCommand::zero(exec_buf_cmd);

    exec_buf_cmd.base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF;
    exec_buf_cmd.exec_buf.base_addr = base_addr;
    exec_buf_cmd.exec_buf.log_page_size = log_page_size;
    exec_buf_cmd.exec_buf.pages = pages;

    this->write_to_cmd_sequence(&exec_buf_cmd, sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
}

void DeviceCommand::add_dispatch_terminate() {
    this->add_prefetch_relay_inline(true, sizeof(CQDispatchCmd));
    CQDispatchCmd terminate_cmd;
    if (!zero_init_disable) DeviceCommand::zero(terminate_cmd);

    terminate_cmd.base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
    this->write_to_cmd_sequence(&terminate_cmd, sizeof(CQDispatchCmd));
}

void DeviceCommand::add_prefetch_terminate() {
    CQPrefetchCmd terminate_cmd;
    if (!zero_init_disable) DeviceCommand::zero(terminate_cmd);

    terminate_cmd.base.cmd_id = CQ_PREFETCH_CMD_TERMINATE;
    this->write_to_cmd_sequence(&terminate_cmd, sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
}

void DeviceCommand::add_prefetch_exec_buf_end() {
    CQPrefetchCmd exec_buf_end_cmd;
    if (!zero_init_disable) DeviceCommand::zero(exec_buf_end_cmd);

    exec_buf_end_cmd.base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF_END;
    this->write_to_cmd_sequence(&exec_buf_end_cmd, sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
}
