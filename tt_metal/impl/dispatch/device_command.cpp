// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/device_command.hpp"

#include <atomic>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"

DeviceCommand::DeviceCommand(uint32_t cmd_sequence_sizeB) : cmd_write_idx(0) {
    this->cmd_sequence.resize(cmd_sequence_sizeB / sizeof(uint32_t), 0);
}

void DeviceCommand::add_dispatch_wait(uint8_t barrier, uint32_t address, uint32_t count) {
    TT_ASSERT(this->cmd_write_idx + (sizeof(CQPrefetchCmd) / sizeof(uint32_t)) < this->cmd_sequence.size()); // turn to api

    CQPrefetchCmd relay_wait;
    relay_wait.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
    relay_wait.relay_inline.length = sizeof(CQDispatchCmd);
    relay_wait.relay_inline.stride = CQ_PREFETCH_CMD_BARE_MIN_SIZE;

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &relay_wait, sizeof(CQPrefetchCmd));
    this->cmd_write_idx += (sizeof(CQPrefetchCmd) / sizeof(uint32_t));

    CQDispatchCmd wait_cmd;
    wait_cmd.base.cmd_id = CQ_DISPATCH_CMD_WAIT;
    wait_cmd.wait.barrier = barrier;
    wait_cmd.wait.notify_prefetch = false;
    wait_cmd.wait.addr = address;
    wait_cmd.wait.count = count;

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &wait_cmd, sizeof(CQDispatchCmd));
    this->cmd_write_idx += (sizeof(CQDispatchCmd) / sizeof(uint32_t));
}

void DeviceCommand::add_dispatch_wait_with_prefetch_stall(uint8_t barrier, uint32_t address, uint32_t count) {
    this->add_dispatch_wait(barrier, address, count);

    uint32_t dispatch_wait_idx = this->cmd_write_idx - (sizeof(CQDispatchCmd) / sizeof(uint32_t));
    CQDispatchCmd *wait_cmd = (CQDispatchCmd*)(this->cmd_sequence.data() + dispatch_wait_idx);
    wait_cmd->wait.notify_prefetch = true;

    CQPrefetchCmd stall_cmd;
    stall_cmd.base.cmd_id = CQ_PREFETCH_CMD_STALL;
    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &stall_cmd, sizeof(CQPrefetchCmd));

    this->cmd_write_idx += (CQ_PREFETCH_CMD_BARE_MIN_SIZE / sizeof(uint32_t));
}

void DeviceCommand::add_prefetch_relay_inline(bool flush, uint32_t lengthB) {
    CQPrefetchCmd relay_write;
    relay_write.base.cmd_id = flush ? CQ_PREFETCH_CMD_RELAY_INLINE : CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH;
    relay_write.relay_inline.length = lengthB;
    relay_write.relay_inline.stride = align(sizeof(CQPrefetchCmd) + lengthB, PCIE_ALIGNMENT);

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &relay_write, sizeof(CQPrefetchCmd));
    this->cmd_write_idx += (sizeof(CQPrefetchCmd) / sizeof(uint32_t));
}

void DeviceCommand::add_prefetch_relay_linear(uint32_t noc_xy_addr, uint32_t lengthB, uint32_t addr) {
    CQPrefetchCmd relay_linear_cmd;
    relay_linear_cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR;
    relay_linear_cmd.relay_linear.noc_xy_addr = noc_xy_addr;
    relay_linear_cmd.relay_linear.length = lengthB;
    relay_linear_cmd.relay_linear.addr = addr;

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &relay_linear_cmd, sizeof(CQPrefetchCmd));
    this->cmd_write_idx += (CQ_PREFETCH_CMD_BARE_MIN_SIZE / sizeof(uint32_t));
}

void DeviceCommand::add_prefetch_relay_paged(uint8_t is_dram, uint8_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages) {
    CQPrefetchCmd relay_paged_cmd;
    relay_paged_cmd.base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;
    relay_paged_cmd.relay_paged.is_dram = is_dram;
    relay_paged_cmd.relay_paged.start_page = start_page;
    relay_paged_cmd.relay_paged.base_addr = base_addr;
    relay_paged_cmd.relay_paged.page_size = page_size;
    relay_paged_cmd.relay_paged.pages = pages;

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &relay_paged_cmd, sizeof(CQPrefetchCmd));
    this->cmd_write_idx += (CQ_PREFETCH_CMD_BARE_MIN_SIZE / sizeof(uint32_t));
}

void DeviceCommand::add_dispatch_write_linear(bool flush_prefetch, uint8_t num_mcast_dests, uint32_t noc_xy_addr, uint32_t addr, uint32_t data_sizeB, const void *data) {
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    CQDispatchCmd write_cmd;
    write_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    write_cmd.write_linear.num_mcast_dests = num_mcast_dests;
    write_cmd.write_linear.noc_xy_addr = noc_xy_addr;
    write_cmd.write_linear.addr = addr;
    write_cmd.write_linear.length = data_sizeB;

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &write_cmd, sizeof(CQDispatchCmd));
    this->cmd_write_idx += (sizeof(CQDispatchCmd) / sizeof(uint32_t));

    if (data != nullptr) {
        memcpy(this->cmd_sequence.data() + this->cmd_write_idx, data, data_sizeB);
        this->cmd_write_idx += (align(data_sizeB, PCIE_ALIGNMENT) / sizeof(uint32_t));
    }
}

void DeviceCommand::add_dispatch_write_paged(bool flush_prefetch, uint8_t is_dram, uint16_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages, const void *data) {
    uint32_t data_sizeB = page_size * pages;
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    CQDispatchCmd write_cmd;
    write_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
    write_cmd.write_paged.is_dram = is_dram;
    write_cmd.write_paged.start_page = start_page;
    write_cmd.write_paged.base_addr = base_addr;
    write_cmd.write_paged.page_size = page_size;
    write_cmd.write_paged.pages = pages;

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &write_cmd, sizeof(CQDispatchCmd));
    this->cmd_write_idx += (sizeof(CQDispatchCmd) / sizeof(uint32_t));

    if (data != nullptr) {
        memcpy(this->cmd_sequence.data() + this->cmd_write_idx, data, data_sizeB);
        this->cmd_write_idx += (align(data_sizeB, PCIE_ALIGNMENT) / sizeof(uint32_t));
    }
}

void DeviceCommand::add_dispatch_write_host(bool flush_prefetch, uint32_t data_sizeB, const void *data) {
    uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
    this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

    CQDispatchCmd write_cmd;
    write_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_HOST;
    write_cmd.write_linear_host.length = payload_sizeB; // CQ_DISPATCH_CMD_WRITE_LINEAR_HOST writes dispatch cmd back to completion queue

    memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &write_cmd, sizeof(CQDispatchCmd));
    this->cmd_write_idx += (sizeof(CQDispatchCmd) / sizeof(uint32_t));

    if (data != nullptr) {
        memcpy(this->cmd_sequence.data() + this->cmd_write_idx, data, data_sizeB);
        this->cmd_write_idx += (align(data_sizeB, PCIE_ALIGNMENT) / sizeof(uint32_t));
    }
}
