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

    template<typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint16_t num_sub_cmds,
        uint32_t common_addr,
        uint16_t packed_data_sizeB,
        uint32_t payload_sizeB,
        const std::vector<PackedSubCmd> &sub_cmds,
        const std::vector<std::pair<const void *, uint32_t>> &data_and_size_descriptors) {
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

        memcpy(this->cmd_sequence.data() + this->cmd_write_idx, &write_packed_cmd, sizeof(CQDispatchCmd));
        this->cmd_write_idx += (sizeof(CQDispatchCmd) / sizeof(uint32_t));

        static_assert(sizeof(PackedSubCmd) % sizeof(uint32_t) == 0);
        uint32_t sub_cmds_sizeB = sub_cmds.size() * sizeof(PackedSubCmd);

        memcpy(this->cmd_sequence.data() + this->cmd_write_idx, sub_cmds.data(), sub_cmds_sizeB);

        this->cmd_write_idx += (align(sub_cmds_sizeB, L1_ALIGNMENT) / sizeof(uint32_t));

        // copy the actual data
        for (const std::pair<const void *, uint32_t> &data_and_size : data_and_size_descriptors) {
            memcpy(this->cmd_sequence.data() + this->cmd_write_idx, data_and_size.first, data_and_size.second);
            this->cmd_write_idx += (align(data_and_size.second, L1_ALIGNMENT) / sizeof(uint32_t));
        }

        this->cmd_write_idx += (align(this->cmd_write_idx, PCIE_ALIGNMENT) / sizeof(uint32_t));
    }

   private:
    void add_prefetch_relay_inline(bool flush, uint32_t lengthB);

    std::vector<uint32_t> cmd_sequence;
    uint32_t cmd_write_idx;
};
