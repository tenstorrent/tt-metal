// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "dev_mem_map.h"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"

class DeviceCommand {
   public:
    DeviceCommand() = default;
    DeviceCommand(void *cmd_region, uint32_t cmd_sequence_sizeB);
    DeviceCommand(uint32_t cmd_sequence_sizeB);

    DeviceCommand &operator=(const DeviceCommand &other);
    DeviceCommand &operator=(DeviceCommand &&other);
    DeviceCommand(const DeviceCommand &other);
    DeviceCommand(DeviceCommand &&other);

    // Constants
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;  // TODO: Move this somewhere else

    uint32_t size_bytes() const { return this->cmd_sequence_sizeB; }

    void* data() const { return this->cmd_region; }

    vector<uint32_t> cmd_vector() const { return this->cmd_region_vector; }

    void add_dispatch_wait(uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count = 0);

    void add_dispatch_wait_with_prefetch_stall(uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count = 0);

    void add_prefetch_relay_linear(uint32_t noc_xy_addr, uint32_t lengthB, uint32_t addr);

    void add_prefetch_relay_paged(uint8_t is_dram, uint8_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages, uint16_t length_adjust = 0);

    template <bool inline_data = false>
    void add_dispatch_write_linear(bool flush_prefetch, uint8_t num_mcast_dests, uint32_t noc_xy_addr, uint32_t addr, uint32_t data_sizeB, const void *data = nullptr) {
        uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        CQDispatchCmd *write_cmd = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
        write_cmd->write_linear.num_mcast_dests = num_mcast_dests;
        write_cmd->write_linear.noc_xy_addr = noc_xy_addr;
        write_cmd->write_linear.addr = addr;
        write_cmd->write_linear.length = data_sizeB;

        if (inline_data) {
            TT_ASSERT(data != nullptr); // compiled out?
            uint32_t increment_sizeB = align(data_sizeB, PCIE_ALIGNMENT);
            this->add_data(data, data_sizeB, increment_sizeB);
        }
    }

    template <bool inline_data = false>
    void add_dispatch_write_paged(bool flush_prefetch, uint8_t is_dram, uint16_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages, const void *data = nullptr) {
        uint32_t data_sizeB = page_size * pages;
        uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        CQDispatchCmd *write_cmd = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
        write_cmd->write_paged.is_dram = is_dram;
        write_cmd->write_paged.start_page = start_page;
        write_cmd->write_paged.base_addr = base_addr;
        write_cmd->write_paged.page_size = page_size;
        write_cmd->write_paged.pages = pages;

        if (inline_data) {
            TT_ASSERT(data != nullptr); // compiled out?
            uint32_t increment_sizeB = align(data_sizeB, PCIE_ALIGNMENT);
            this->add_data(data, data_sizeB, increment_sizeB);
        }
    }

    template <bool inline_data = false>
    void add_dispatch_write_host(bool flush_prefetch, uint32_t data_sizeB, const void *data = nullptr) {
        uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        CQDispatchCmd *write_cmd = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
        write_cmd->write_linear_host.length = payload_sizeB; // CQ_DISPATCH_CMD_WRITE_LINEAR_HOST writes dispatch cmd back to completion queue

        if (inline_data) {
            TT_ASSERT(data != nullptr); // compiled out?
            uint32_t increment_sizeB = align(data_sizeB, PCIE_ALIGNMENT);
            this->add_data(data, data_sizeB, increment_sizeB);
        }
    }

    void add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages);

    void add_dispatch_terminate();

    void add_prefetch_terminate();

    void add_prefetch_exec_buf_end();

    void update_cmd_sequence(uint32_t cmd_offsetB, const void *new_data, uint32_t data_sizeB) {
        memcpy((char*)this->cmd_region + cmd_offsetB, new_data, data_sizeB);
    }

    void add_data(const void *data, uint32_t data_size_to_copyB, uint32_t cmd_write_offset_incrementB);

    template<typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint16_t num_sub_cmds,
        uint32_t common_addr,
        uint16_t packed_data_sizeB,
        uint32_t payload_sizeB,
        const std::vector<PackedSubCmd> &sub_cmds,
        const std::vector<const void *> &data_collection) {
        static_assert(std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value or std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value);
        bool multicast = std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value;

        static constexpr uint32_t max_num_packed_sub_cmds = (dispatch_constants::TRANSFER_PAGE_SIZE - sizeof(CQDispatchCmd)) / sizeof(PackedSubCmd);
        TT_ASSERT(num_sub_cmds <= max_num_packed_sub_cmds, "Max number of packed sub commands are {} but requesting {}", max_num_packed_sub_cmds, num_sub_cmds);

        bool flush_prefetch = true;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        CQDispatchCmd *write_packed_cmd = (CQDispatchCmd *)((char *)this->cmd_region + this->cmd_write_offsetB);
        if (!zero_init_disable) DeviceCommand::zero(write_packed_cmd);

        write_packed_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
        write_packed_cmd->write_packed.is_multicast = multicast;
        write_packed_cmd->write_packed.count = num_sub_cmds;
        write_packed_cmd->write_packed.addr = common_addr;
        write_packed_cmd->write_packed.size = packed_data_sizeB;

        this->cmd_write_offsetB += sizeof(CQDispatchCmd);

        static_assert(sizeof(PackedSubCmd) % sizeof(uint32_t) == 0);
        uint32_t sub_cmds_sizeB = sub_cmds.size() * sizeof(PackedSubCmd);

        memcpy((char*)this->cmd_region + this->cmd_write_offsetB, sub_cmds.data(), sub_cmds_sizeB);
        uint32_t increment_sizeB = align(sub_cmds_sizeB, L1_ALIGNMENT);
        this->cmd_write_offsetB += increment_sizeB;

        // copy the actual data
        for (const void *data : data_collection) {
            memcpy((char*)this->cmd_region + this->cmd_write_offsetB, data, packed_data_sizeB);
            uint32_t increment_sizeB = align(packed_data_sizeB, L1_ALIGNMENT);
            this->cmd_write_offsetB += increment_sizeB;
        }

        this->cmd_write_offsetB = align(this->cmd_write_offsetB, PCIE_ALIGNMENT);
    }

   private:

    static bool zero_init_disable;

    void add_prefetch_relay_inline(bool flush, uint32_t lengthB);

    void validate_cmd_write(uint32_t data_sizeB) const {
        uint32_t data_endB = this->cmd_write_offsetB + data_sizeB;
        TT_ASSERT(data_endB <= this->cmd_sequence_sizeB,
            "Out of bounds command sequence write: attemping to write {} B but only {} B available",
            data_sizeB, this->cmd_sequence_sizeB - this->cmd_write_offsetB);
    }

    template<typename Command>
    static void zero(Command *cmd) {
        for (int i = 0; i < sizeof(Command); i++) {
            ((uint8_t *)cmd)[i] = 0;
        }
    }

    template<typename CommandPtr>
    CommandPtr reserve_space(uint32_t size_to_writeB) {
        this->validate_cmd_write(size_to_writeB);
        CommandPtr cmd = (CommandPtr)((char *)this->cmd_region + this->cmd_write_offsetB);
        if (!zero_init_disable) DeviceCommand::zero(cmd);
        this->cmd_write_offsetB += size_to_writeB;
        return cmd;
    }

    void deepcopy(const DeviceCommand &other);

    uint32_t cmd_sequence_sizeB = 0;
    void *cmd_region = nullptr;
    uint32_t cmd_write_offsetB = 0;

    std::vector<uint32_t> cmd_region_vector;
};
