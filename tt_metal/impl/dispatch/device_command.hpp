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
    DeviceCommand(uint32_t cmd_sequence_sizeB);

    // Constants
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048; // TODO: Move this somewhere else

    uint32_t size_bytes() const { return this->cmd_sequence.size() * sizeof(uint32_t); }

    const void* data() const { return this->cmd_sequence.data(); }

    void add_dispatch_wait(uint8_t barrier, uint32_t address, uint32_t count);

    void add_dispatch_wait_with_prefetch_stall(uint8_t barrier, uint32_t address, uint32_t count);

    void add_prefetch_relay_linear(uint32_t noc_xy_addr, uint32_t lengthB, uint32_t addr);

    void add_prefetch_relay_paged(uint8_t is_dram, uint8_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages);

    void add_dispatch_write_linear(bool flush_prefetch, uint8_t num_mcast_dests, uint32_t noc_xy_addr, uint32_t addr, uint32_t data_sizeB, const void *data = nullptr);

    void add_dispatch_write_paged(bool flush_prefetch, uint8_t is_dram, uint16_t start_page, uint32_t base_addr, uint32_t page_size, uint32_t pages, const void *data = nullptr);

    void add_dispatch_write_host(bool flush_prefetch, uint32_t data_sizeB, const void *data = nullptr);

    void add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages);

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

        static constexpr uint32_t max_num_packed_sub_cmds = (TRANSFER_PAGE_SIZE - sizeof(CQDispatchCmd)) / sizeof(PackedSubCmd);
        TT_ASSERT(num_sub_cmds <= max_num_packed_sub_cmds, "Max number of packed sub commands are {} but requesting {}", max_num_packed_sub_cmds, num_sub_cmds);

        bool flush_prefetch = true;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        CQDispatchCmd write_packed_cmd;
        write_packed_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
        write_packed_cmd.write_packed.is_multicast = multicast;
        write_packed_cmd.write_packed.count = num_sub_cmds;
        write_packed_cmd.write_packed.addr = common_addr;
        write_packed_cmd.write_packed.size = packed_data_sizeB;

        this->write_to_cmd_sequence(&write_packed_cmd, sizeof(CQDispatchCmd));

        static_assert(sizeof(PackedSubCmd) % sizeof(uint32_t) == 0);
        uint32_t sub_cmds_sizeB = sub_cmds.size() * sizeof(PackedSubCmd);

        this->write_to_cmd_sequence(sub_cmds.data(), sub_cmds_sizeB, L1_ALIGNMENT);

        // copy the actual data
        for (const void *data : data_collection) {
            this->write_to_cmd_sequence(data, packed_data_sizeB, L1_ALIGNMENT);
        }

        this->cmd_write_idx += (align(this->cmd_write_idx, PCIE_ALIGNMENT / sizeof(uint32_t)) - this->cmd_write_idx);
    }

   private:
    void add_prefetch_relay_inline(bool flush, uint32_t lengthB);

    void validate_cmd_write(uint32_t data_sizeB) const {
        uint32_t data_end_index = this->cmd_write_idx + (data_sizeB / sizeof(uint32_t));
        TT_ASSERT(data_end_index <= this->cmd_sequence.size(),
            "Attemping to write {} B at index {} which is out of bounds for command sequence vector of size {}",
            data_sizeB, data_end_index, this->cmd_sequence.size());
    }

    void write_to_cmd_sequence(const void *data, uint32_t data_sizeB, std::optional<uint32_t> alignmentB = std::nullopt) {
        this->validate_cmd_write(data_sizeB);
        memcpy(this->cmd_sequence.data() + this->cmd_write_idx, data, data_sizeB);
        uint32_t increment_sizeB = alignmentB.has_value() ? align(data_sizeB, alignmentB.value()) : data_sizeB;
        this->cmd_write_idx += (increment_sizeB / sizeof(uint32_t));
    }

    std::vector<uint32_t> cmd_sequence;
    uint32_t cmd_write_idx;
};
