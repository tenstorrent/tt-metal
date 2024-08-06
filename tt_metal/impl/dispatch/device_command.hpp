// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstddef>
#include <vector>

#include "common/env_lib.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/memcpy.hpp"
#include "tt_metal/tt_stl/aligned_allocator.hpp"

namespace tt::tt_metal {
template <typename T>
using vector_memcpy_aligned = std::vector<T, tt::stl::aligned_allocator<T, MEMCPY_ALIGNMENT>>;

template <bool hugepage_write = false>
class DeviceCommand {
   public:
    DeviceCommand() = default;
    DeviceCommand(void *cmd_region, uint32_t cmd_sequence_sizeB) :
        cmd_sequence_sizeB(cmd_sequence_sizeB), cmd_region(cmd_region), cmd_write_offsetB(0) {
        TT_FATAL(
            cmd_sequence_sizeB % sizeof(uint32_t) == 0,
            "Command sequence size B={} is not {}-byte aligned",
            cmd_sequence_sizeB,
            sizeof(uint32_t));
    }
    template <bool hp_w = hugepage_write, typename std::enable_if_t<!hp_w, int> = 0>
    DeviceCommand(uint32_t cmd_sequence_sizeB) : cmd_sequence_sizeB(cmd_sequence_sizeB), cmd_write_offsetB(0) {
        TT_FATAL(
            cmd_sequence_sizeB % sizeof(uint32_t) == 0,
            "Command sequence size B={} is not {}-byte aligned",
            cmd_sequence_sizeB,
            sizeof(uint32_t));
        this->cmd_region_vector.resize(cmd_sequence_sizeB / sizeof(uint32_t), 0);
        this->cmd_region = this->cmd_region_vector.data();
    }

    DeviceCommand &operator=(const DeviceCommand &other) {
        this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
        this->cmd_write_offsetB = other.cmd_write_offsetB;
        this->cmd_region_vector = other.cmd_region_vector;
        this->deepcopy(other);
        return *this;
    }
    DeviceCommand &operator=(DeviceCommand &&other) {
        this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
        this->cmd_write_offsetB = other.cmd_write_offsetB;
        this->cmd_region_vector = other.cmd_region_vector;
        this->deepcopy(other);
        return *this;
    }
    DeviceCommand(const DeviceCommand &other) :
        cmd_sequence_sizeB(other.cmd_sequence_sizeB),
        cmd_write_offsetB(other.cmd_write_offsetB),
        cmd_region_vector(other.cmd_region_vector) {
        this->deepcopy(other);
    }
    DeviceCommand(DeviceCommand &&other) :
        cmd_sequence_sizeB(other.cmd_sequence_sizeB),
        cmd_write_offsetB(other.cmd_write_offsetB),
        cmd_region_vector(other.cmd_region_vector) {
        this->deepcopy(other);
    }

    // Constants
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;  // TODO: Move this somewhere else
    static constexpr uint32_t LOG2_PROGRAM_PAGE_SIZE = std::bit_width(PROGRAM_PAGE_SIZE) - 1;

    uint32_t size_bytes() const { return this->cmd_sequence_sizeB; }

    void *data() const { return this->cmd_region; }

    uint32_t write_offset_bytes() const { return this->cmd_write_offsetB; }

    vector_memcpy_aligned<uint32_t> cmd_vector() const { return this->cmd_region_vector; }

    void add_dispatch_wait(
        uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count = 0, bool notify_prefetch = false, bool do_wait = true);

    void add_dispatch_wait_with_prefetch_stall(
        uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count = 0, bool do_wait = true);

    void add_prefetch_wait_for_event(uint32_t event_id, uint32_t event_addr);

    void add_dispatch_write_remote(uint32_t data, uint32_t noc_xy_addr, uint32_t addr);

    void add_prefetch_relay_linear(uint32_t noc_xy_addr, uint32_t lengthB, uint32_t addr);

    void add_prefetch_relay_paged(
        uint8_t is_dram,
        uint8_t start_page,
        uint32_t base_addr,
        uint32_t page_size,
        uint32_t pages,
        uint16_t length_adjust = 0);

    void add_prefetch_relay_paged_packed(
        uint32_t length,
        std::vector<CQPrefetchRelayPagedPackedSubCmd> & sub_cmds,
        uint16_t num_sub_cmds,
        uint32_t offset_idx = 0);

    template <bool inline_data = false>
    void add_dispatch_write_linear(
        bool flush_prefetch,
        uint8_t num_mcast_dests,
        uint32_t noc_xy_addr,
        uint32_t addr,
        uint32_t data_sizeB,
        const void *data = nullptr,
        uint32_t write_offset_index = 0);

    template <bool inline_data = false>
    void add_dispatch_write_paged(
        bool flush_prefetch,
        uint8_t is_dram,
        uint16_t start_page,
        uint32_t base_addr,
        uint32_t page_size,
        uint32_t pages,
        const void *data = nullptr);

    template <bool inline_data = false>
    void add_dispatch_write_host(bool flush_prefetch, uint32_t data_sizeB, bool is_event, const void *data = nullptr);

    void add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages);

    void add_dispatch_set_write_offsets(uint32_t write_offset0, uint32_t write_offset1, uint32_t write_offset2);

    void add_dispatch_terminate();

    void add_prefetch_terminate();

    void add_prefetch_exec_buf_end();

    void update_cmd_sequence(uint32_t cmd_offsetB, const void *new_data, uint32_t data_sizeB);

    void add_data(const void *data, uint32_t data_size_to_copyB, uint32_t cmd_write_offset_incrementB);

    template <typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint16_t num_sub_cmds,
        uint32_t common_addr,
        uint16_t packed_data_sizeB,
        uint32_t payload_sizeB,
        const std::vector<PackedSubCmd> &sub_cmds,
        const std::vector<std::pair<const void *, uint32_t>> &data_collection,
        uint32_t packed_write_max_unicast_sub_cmds,
        const uint32_t offset_idx = 0,
        const bool no_stride = false,
        uint32_t write_offset_index = 0);

    // Tuple in data_collection is:
    //  0:address, 1:size, 2:stride
    template <typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint16_t num_sub_cmds,
        uint32_t common_addr,
        uint16_t packed_data_sizeB,
        uint32_t payload_sizeB,
        const std::vector<PackedSubCmd> &sub_cmds,
        const std::vector<std::vector<std::tuple<const void *, uint32_t, uint32_t>>> &data_collection,
        uint32_t packed_write_max_unicast_sub_cmds,
        const uint32_t offset_idx = 0,
        const bool no_stride = false,
        uint32_t write_offset_index = 0);

    void add_dispatch_write_packed_large(
        uint16_t alignment,
        uint16_t num_sub_cmds,
        const std::vector<CQDispatchWritePackedLargeSubCmd> &sub_cmds,
        const uint32_t offset_idx = 0,
        uint32_t write_offset_index = 0);

    template <typename CommandPtr, bool data = false>
    CommandPtr reserve_space(uint32_t size_to_writeB) {
        this->validate_cmd_write(size_to_writeB);
        CommandPtr cmd = (CommandPtr)((char *)this->cmd_region + this->cmd_write_offsetB);
        // Only zero out cmds
        if constexpr (!data) {
            if (zero_init_enable)
                DeviceCommand::zero(cmd);
        }
        this->cmd_write_offsetB += size_to_writeB;
        return cmd;
    }

   private:
    static bool zero_init_enable;

    void add_prefetch_relay_inline(bool flush, uint32_t lengthB);

    void validate_cmd_write(uint32_t data_sizeB) const;

    template <typename Command>
    void zero(Command *cmd);

    void deepcopy(const DeviceCommand &other) {
        if (other.cmd_region_vector.empty() and other.cmd_region != nullptr) {
            this->cmd_region = other.cmd_region;
        } else if (not other.cmd_region_vector.empty()) {
            TT_ASSERT(other.cmd_region != nullptr);
            this->cmd_region = this->cmd_region_vector.data();
            this->memcpy(this->cmd_region, other.cmd_region_vector.data(), this->cmd_sequence_sizeB);
        }
    }
    void memcpy(void *__restrict dst, const void *__restrict src, size_t n);

    uint32_t cmd_sequence_sizeB = 0;
    void *cmd_region = nullptr;
    uint32_t cmd_write_offsetB = 0;

    vector_memcpy_aligned<uint32_t> cmd_region_vector;
};

template <bool hugepage_write>
bool DeviceCommand<hugepage_write>::zero_init_enable = tt::parse_env<bool>("TT_METAL_ZERO_INIT_ENABLE", false);

using HugepageDeviceCommand = DeviceCommand<true>;
using HostMemDeviceCommand = DeviceCommand<false>;

}
