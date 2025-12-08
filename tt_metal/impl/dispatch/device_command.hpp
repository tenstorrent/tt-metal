// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt_stl/aligned_allocator.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "env_lib.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "memcpy.hpp"
#include <tt_stl/span.hpp>
#include "tt_align.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "vector_aligned.hpp"

namespace tt::tt_metal {

template <bool hugepage_write = false>
class DeviceCommand {
public:
    DeviceCommand() = default;
    DeviceCommand(void* cmd_region, uint32_t cmd_sequence_sizeB);

    template <bool hp_w = hugepage_write, typename std::enable_if_t<!hp_w, int> = 0>
    DeviceCommand(uint32_t cmd_sequence_sizeB);

    DeviceCommand& operator=(const DeviceCommand& other);
    DeviceCommand& operator=(DeviceCommand&& other) noexcept;
    DeviceCommand(const DeviceCommand& other);
    DeviceCommand(DeviceCommand&& other) noexcept;

    // Constants
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;  // TODO: Move this somewhere else
    static constexpr uint32_t LOG2_PROGRAM_PAGE_SIZE = std::bit_width(PROGRAM_PAGE_SIZE) - 1;

    uint32_t size_bytes() const;

    void* data() const;

    uint32_t write_offset_bytes() const;

    vector_aligned<uint32_t> cmd_vector() const;

    void add_dispatch_wait(
        uint32_t flags, uint32_t address, uint32_t stream, uint32_t count, uint8_t dispatcher_type = 0);

    void add_dispatch_wait_with_prefetch_stall(uint32_t flags, uint32_t address, uint32_t stream, uint32_t count);

    void add_prefetch_relay_linear(uint32_t noc_xy_addr, DeviceAddr lengthB, uint32_t addr);

    void add_prefetch_relay_paged(
        uint8_t is_dram,
        uint8_t start_page,
        uint32_t base_addr,
        uint32_t page_size,
        uint32_t pages,
        uint16_t length_adjust = 0);

    void add_prefetch_relay_paged_packed(
        uint32_t length,
        const std::vector<CQPrefetchRelayPagedPackedSubCmd>& sub_cmds,
        uint16_t num_sub_cmds,
        uint32_t offset_idx = 0);

    void add_prefetch_paged_to_ringbuffer(const CQPrefetchPagedToRingbufferCmd& paged_to_ringbuffer_info);

    void add_prefetch_set_ringbuffer_offset(uint32_t offset, bool update_wp = false);

    void add_prefetch_relay_ringbuffer(
        uint32_t num_sub_cmds, const std::vector<CQPrefetchRelayRingbufferSubCmd>& sub_cmds, uint32_t offset_idx = 0);

    template <bool flush_prefetch = true, bool inline_data = false>
    void add_dispatch_write_linear(
        uint8_t num_mcast_dests,
        uint32_t noc_xy_addr,
        DeviceAddr addr,
        DeviceAddr data_sizeB,
        const void* data = nullptr,
        uint32_t write_offset_index = 0);

    // Like add_dispatch_write_linear, but emits CQ_DISPATCH_CMD_WRITE_LINEAR_H (dispatch_h variant).
    template <bool flush_prefetch = true, bool inline_data = false>
    void add_dispatch_write_linear_h(
        uint8_t num_mcast_dests,
        uint32_t noc_xy_addr,
        DeviceAddr addr,
        DeviceAddr data_sizeB,
        const void* data = nullptr,
        uint32_t write_offset_index = 0);

    void add_dispatch_go_signal_mcast(
        uint32_t wait_count,
        uint32_t go_signal,
        uint32_t wait_stream,
        uint8_t multicast_go_offset,
        uint8_t num_unicast_txns,
        uint8_t noc_data_start_index,
        DispatcherSelect dispatcher_type);

    void add_notify_dispatch_s_go_signal_cmd(uint8_t wait, uint16_t index_bitmask);

    template <bool inline_data = false>
    void add_dispatch_write_paged(
        bool flush_prefetch,
        uint8_t is_dram,
        uint16_t start_page,
        uint32_t base_addr,
        uint32_t page_size,
        uint32_t pages,
        const void* data = nullptr);

    template <bool inline_data = false>
    void add_dispatch_write_host(
        bool flush_prefetch, uint64_t data_sizeB, bool is_event, uint16_t pad1, const void* data = nullptr);

    void add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages);

    void add_dispatch_set_num_worker_sems(uint32_t num_worker_sems, DispatcherSelect dispatcher_type);

    void add_dispatch_set_go_signal_noc_data(
        const vector_aligned<uint32_t>& noc_mcast_unicast_data, DispatcherSelect dispatcher_type);

    void add_dispatch_set_write_offsets(tt::stl::Span<const uint32_t> write_offsets);

    void add_dispatch_terminate(DispatcherSelect dispatcher_type = DispatcherSelect::DISPATCH_MASTER);

    void add_prefetch_terminate();

    void add_prefetch_exec_buf_end();

    void update_cmd_sequence(uint32_t cmd_offsetB, const void* new_data, uint32_t data_sizeB);

    void add_data(const void* data, uint32_t data_size_to_copyB, uint32_t cmd_write_offset_incrementB)
        __attribute((nonnull(2)));

    void align_write_offset();

    template <typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint8_t type,
        uint16_t num_sub_cmds,
        uint32_t common_addr,
        uint16_t packed_data_sizeB,
        uint32_t payload_sizeB,
        const std::vector<PackedSubCmd>& sub_cmds,
        const std::vector<std::pair<const void*, uint32_t>>& data_collection,
        uint32_t packed_write_max_unicast_sub_cmds,
        uint32_t offset_idx = 0,
        bool no_stride = false,
        uint32_t write_offset_index = 0);

    // Tuple in data_collection is:
    //  0:address, 1:size, 2:stride
    template <typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint8_t type,
        uint16_t num_sub_cmds,
        uint32_t common_addr,
        uint16_t packed_data_sizeB,
        uint32_t payload_sizeB,
        const std::vector<PackedSubCmd>& sub_cmds,
        const std::vector<std::vector<std::tuple<const void*, uint32_t, uint32_t>>>& data_collection,
        uint32_t packed_write_max_unicast_sub_cmds,
        uint32_t offset_idx = 0,
        bool no_stride = false,
        uint32_t write_offset_index = 0);

    // Add write packed large, with no data.
    void add_dispatch_write_packed_large(
        uint8_t type,
        uint16_t alignment,
        uint16_t num_sub_cmds,
        const std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
        uint32_t offset_idx = 0,
        uint32_t write_offset_index = 0);

    // Add write packed large, with data inlined.
    void add_dispatch_write_packed_large(
        uint8_t type,
        uint16_t alignment,
        uint16_t num_sub_cmds,
        const std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
        const std::vector<tt::stl::Span<const uint8_t>>& data_collection,
        std::vector<uint8_t*>*
            data_collection_buffer_ptr,  // optional. Stores the location each data segment was written to
        uint32_t offset_idx = 0,
        uint32_t write_offset_index = 0);

    template <typename CommandPtr, bool data = false>
    CommandPtr reserve_space(uint32_t size_to_writeB) {
        this->validate_cmd_write(size_to_writeB);
        CommandPtr cmd = (CommandPtr)((char*)this->cmd_region + this->cmd_write_offsetB);
        // Only zero out cmds
        if constexpr (!data) {
            if (zero_init_enable) {
                DeviceCommand::zero(cmd);
            }
        }
        this->cmd_write_offsetB += size_to_writeB;
        return cmd;
    }

    // This value is random, but stable for the lifetime of the program. It is used to pad the command for event
    // commands, so we can check the value on the host side.
    static uint32_t random_padding_value();

private:
    static bool zero_init_enable;

    void add_prefetch_relay_inline(
        bool flush, uint32_t lengthB, DispatcherSelect dispatcher_type = DispatcherSelect::DISPATCH_MASTER);

    // Write packed large cmd and subcmds, but not data.
    void add_dispatch_write_packed_large_internal(
        uint8_t type,
        bool flush_prefetch,
        uint16_t alignment,
        uint32_t payload_sizeB,
        uint16_t num_sub_cmds,
        const std::vector<CQDispatchWritePackedLargeSubCmd>& sub_cmds,
        uint32_t offset_idx,
        uint32_t write_offset_index);

    void validate_cmd_write(uint32_t data_sizeB) const;

    void deepcopy(const DeviceCommand& other);

    void memcpy(void* __restrict dst, const void* __restrict src, size_t n) __attribute__((nonnull(2, 3)));

    template <typename Command>
    void zero(Command* cmd) {
        if constexpr (hugepage_write) {
            vector_aligned<char> zero_cmd(sizeof(Command), 0);
            this->memcpy(cmd, zero_cmd.data(), sizeof(Command));
        } else {
            std::fill((uint8_t*)cmd, (uint8_t*)cmd + sizeof(Command), 0);
        }
    }

    uint32_t cmd_sequence_sizeB = 0;
    void* cmd_region = nullptr;
    uint32_t cmd_write_offsetB = 0;
    uint32_t pcie_alignment =
        tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST);
    uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);

    vector_aligned<uint32_t> cmd_region_vector;
};

template <bool hugepage_write>
bool DeviceCommand<hugepage_write>::zero_init_enable = tt::parse_env<bool>("TT_METAL_ZERO_INIT_ENABLE", false);

using HugepageDeviceCommand = DeviceCommand<true>;
using HostMemDeviceCommand = DeviceCommand<false>;

}  // namespace tt::tt_metal
