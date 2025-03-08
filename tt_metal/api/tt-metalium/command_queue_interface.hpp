// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <climits>
#include <magic_enum/magic_enum.hpp>
#include <mutex>
#include <tt-metalium/tt_align.hpp>
#include <unordered_map>

#include "cq_commands.hpp"
#include "dispatch_core_manager.hpp"
#include "launch_message_ring_buffer_state.hpp"
#include "memcpy.hpp"
#include "hal.hpp"
#include "dispatch_settings.hpp"
#include "helpers.hpp"
#include "buffer.hpp"
#include "umd/device/tt_core_coordinates.h"

namespace tt::tt_metal {

enum class CommandQueueDeviceAddrType : uint8_t {
    PREFETCH_Q_RD = 0,
    // Used to notify host of how far device has gotten, doesn't need L1 alignment because it's only written locally by
    // prefetch kernel.
    PREFETCH_Q_PCIE_RD = 1,
    COMPLETION_Q_WR = 2,
    COMPLETION_Q_RD = 3,
    // Max of 2 CQs. COMPLETION_Q*_LAST_EVENT_PTR track the last completed event in the respective CQs
    COMPLETION_Q0_LAST_EVENT = 4,
    COMPLETION_Q1_LAST_EVENT = 5,
    DISPATCH_S_SYNC_SEM = 6,
    UNRESERVED = 7
};

enum class CommandQueueHostAddrType : uint8_t {
    ISSUE_Q_RD = 0,
    ISSUE_Q_WR = 1,
    COMPLETION_Q_WR = 2,
    COMPLETION_Q_RD = 3,
    UNRESERVED = 4
};

//
// Dispatch Memory Map
// Assigns each CommandQueueDeviceAddrType in a linear
// order. The size of each address type and L1 base is
// set by DispatchSettings.
//
class DispatchMemMap {
public:
    DispatchMemMap& operator=(const DispatchMemMap&) = delete;
    DispatchMemMap& operator=(DispatchMemMap&& other) noexcept = delete;
    DispatchMemMap(const DispatchMemMap&) = delete;
    DispatchMemMap(DispatchMemMap&& other) noexcept = delete;

    //
    // Returns the instance. The instance is reset if the core_type and/or num_hw_cqs changed from
    // the last call. The memory region sizes can be configured using DispatchSettings.
    //
    // If the settings changed, then force_reinit_with_settings will recreate the instance with
    // the settings for the given core_type / num_hw_cqs.
    //
    static const DispatchMemMap& get(
        const CoreType& core_type, const uint32_t num_hw_cqs = 0, const bool force_reinit_with_settings = false) {
        auto& instance = get_instance();

        if (num_hw_cqs > 0 && (core_type != instance.last_core_type || num_hw_cqs != instance.hw_cqs) ||
            force_reinit_with_settings) {
            instance.reset(core_type, num_hw_cqs);
        }

        TT_FATAL(
            instance.hw_cqs > 0,
            "Command Queue is not initialized. Call DispatchMemMap::get with non zero num_hw_cqs.");
        return instance;
    }

    uint32_t prefetch_q_entries() const { return settings.prefetch_q_entries_; }

    uint32_t prefetch_q_size() const { return settings.prefetch_q_size_; }

    uint32_t max_prefetch_command_size() const { return settings.prefetch_max_cmd_size_; }

    uint32_t cmddat_q_base() const { return cmddat_q_base_; }

    uint32_t cmddat_q_size() const { return settings.prefetch_cmddat_q_size_; }

    uint32_t scratch_db_base() const { return scratch_db_base_; }

    uint32_t scratch_db_size() const { return settings.prefetch_scratch_db_size_; }

    uint32_t dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

    uint32_t dispatch_buffer_base() const { return dispatch_buffer_base_; }

    uint32_t dispatch_buffer_pages() const { return settings.dispatch_pages_; }

    uint32_t prefetch_d_buffer_size() const { return settings.prefetch_d_buffer_size_; }

    uint32_t prefetch_d_buffer_pages() const { return settings.prefetch_d_pages_; }

    uint32_t mux_buffer_size(uint8_t num_hw_cqs = 1) const { return settings.tunneling_buffer_size_ / num_hw_cqs; }

    uint32_t mux_buffer_pages(uint8_t num_hw_cqs = 1) const { return settings.tunneling_buffer_pages_ / num_hw_cqs; }

    uint32_t dispatch_s_buffer_size() const { return settings.dispatch_s_buffer_size_; }

    uint32_t dispatch_s_buffer_pages() const {
        return settings.dispatch_s_buffer_size_ /
               (1 << tt::tt_metal::DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
    }

    uint32_t get_device_command_queue_addr(const CommandQueueDeviceAddrType& device_addr_type) const {
        uint32_t index = tt::utils::underlying_type<CommandQueueDeviceAddrType>(device_addr_type);
        TT_ASSERT(index < this->device_cq_addrs_.size());
        return device_cq_addrs_[index];
    }

    uint32_t get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const {
        return tt::utils::underlying_type<CommandQueueHostAddrType>(host_addr) *
               tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST);
    }

    uint32_t get_sync_offset(uint32_t index) const {
        TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
        uint32_t offset = index * hal.get_alignment(HalMemType::L1);
        return offset;
    }

    uint32_t get_dispatch_message_addr_start() const {
        // Address of the first dispatch message entry. Remaining entries are each offset by
        // get_noc_stream_reg_space_size() bytes.
        return tt::tt_metal::hal.get_noc_overlay_start_addr() +
               tt::tt_metal::hal.get_noc_stream_reg_space_size() * get_dispatch_stream_index(0) +
               tt::tt_metal::hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index() * sizeof(uint32_t);
    }

    uint32_t get_dispatch_stream_index(uint32_t index) const {
        if (last_core_type == CoreType::WORKER) {
            // There are 64 streams. CBs use entries 8-39.
            return 48u + index;
        } else if (last_core_type == CoreType::ETH) {
            // There are 32 streams.
            return 16u + index;
        } else {
            TT_THROW("get_dispatch_starting_stream_index not implemented for core type");
        }
    }

    // Offset to be passed in the go message.
    uint8_t get_dispatch_message_update_offset(uint32_t index) const {
        TT_ASSERT(index < tt::tt_metal::DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
        return index;
    }

private:
    DispatchMemMap() = default;

    static DispatchMemMap& get_instance() {
        static DispatchMemMap instance;
        return instance;
    }

    // Reset the instance using the settings for the core_type and num_hw_cqs.
    void reset(const CoreType& core_type, const uint32_t num_hw_cqs) {
        const auto dispatch_settings = DispatchSettings::get(core_type, num_hw_cqs);
        this->settings = dispatch_settings;
        last_core_type = settings.core_type_;
        hw_cqs = settings.num_hw_cqs_;

        const auto dispatch_buffer_block_size = settings.dispatch_size_;
        const auto [l1_base, l1_size] = get_device_l1_info(settings.core_type_);
        const auto pcie_alignment = tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST);
        const auto l1_alignment = tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::L1);

        TT_ASSERT(settings.prefetch_cmddat_q_size_ >= 2 * settings.prefetch_max_cmd_size_);
        TT_ASSERT(settings.prefetch_scratch_db_size_ % 2 == 0);
        TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);
        TT_ASSERT(
            DispatchSettings::DISPATCH_MESSAGE_ENTRIES <= DispatchSettings::DISPATCH_MESSAGES_MAX_OFFSET / l1_alignment + 1,
            "Number of dispatch message entries exceeds max representable offset");

        uint8_t num_dev_cq_addrs = magic_enum::enum_count<CommandQueueDeviceAddrType>();
        std::vector<uint32_t> device_cq_addr_sizes_(num_dev_cq_addrs, 0);
        for (auto dev_addr_idx = 0; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
            CommandQueueDeviceAddrType dev_addr_type =
                magic_enum::enum_cast<CommandQueueDeviceAddrType>(dev_addr_idx).value();
            if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_RD) {
                device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_rd_ptr_size_;
            } else if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD) {
                device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_pcie_rd_ptr_size_;
            } else if (dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM) {
                device_cq_addr_sizes_[dev_addr_idx] = settings.dispatch_s_sync_sem_;
            } else {
                device_cq_addr_sizes_[dev_addr_idx] = settings.other_ptrs_size;
            }
        }

        device_cq_addrs_.resize(num_dev_cq_addrs);
        device_cq_addrs_[0] = l1_base;
        for (auto dev_addr_idx = 1; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
            device_cq_addrs_[dev_addr_idx] =
                device_cq_addrs_[dev_addr_idx - 1] + device_cq_addr_sizes_[dev_addr_idx - 1];
            CommandQueueDeviceAddrType dev_addr_type = magic_enum::enum_value<CommandQueueDeviceAddrType>(dev_addr_idx);
            if (dev_addr_type == CommandQueueDeviceAddrType::UNRESERVED) {
                device_cq_addrs_[dev_addr_idx] = align(device_cq_addrs_[dev_addr_idx], pcie_alignment);
            }
        }

        uint32_t prefetch_dispatch_unreserved_base =
            device_cq_addrs_[tt::utils::underlying_type<CommandQueueDeviceAddrType>(
                CommandQueueDeviceAddrType::UNRESERVED)];
        cmddat_q_base_ = prefetch_dispatch_unreserved_base + round_size(settings.prefetch_q_size_, pcie_alignment);
        scratch_db_base_ = cmddat_q_base_ + round_size(settings.prefetch_cmddat_q_size_, pcie_alignment);
        dispatch_buffer_base_ = align(prefetch_dispatch_unreserved_base, 1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
        dispatch_buffer_block_size_pages_ = settings.dispatch_pages_ / DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
        const uint32_t dispatch_cb_end = dispatch_buffer_base_ + settings.dispatch_size_;

        TT_ASSERT(scratch_db_base_ + settings.prefetch_scratch_db_size_ < l1_size);
        TT_ASSERT(dispatch_cb_end < l1_size);
    }

    std::pair<uint32_t, uint32_t> get_device_l1_info(const CoreType& core_type) const {
        uint32_t l1_base;
        uint32_t l1_size;
        if (core_type == CoreType::WORKER) {
            l1_base = hal.get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
            l1_size =
                hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
        } else if (core_type == CoreType::ETH) {
            l1_base = hal.get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
            l1_size =
                hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE);
        } else {
            TT_THROW("get_base_device_command_queue_addr not implemented for core type");
        }

        return {l1_base, l1_size};
    }

    uint32_t cmddat_q_base_;
    uint32_t scratch_db_base_;
    uint32_t dispatch_buffer_base_;

    uint32_t dispatch_buffer_block_size_pages_;
    std::vector<uint32_t> device_cq_addrs_;

    DispatchSettings settings;

    uint32_t hw_cqs{0}; // 0 means uninitialized
    CoreType last_core_type{CoreType::WORKER};
};

/// @brief Get offset of the command queue relative to its channel
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t relative offset
inline uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) { return cq_id * cq_size; }

inline uint16_t get_umd_channel(uint16_t channel) { return channel & 0x3; }

/// @brief Get absolute offset of the command queue
/// @param channel uint16_t channel ID (hugepage)
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t absolute offset
inline uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    return (DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel)) + ((channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE) +
           get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_issue_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the
    // completion region Equation for issue fifo size is | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t cq_start) :
        cq_start(cq_start),
        command_completion_region_size(
            (((cq_size - cq_start) / DispatchSettings::TRANSFER_PAGE_SIZE) / 4) *
            DispatchSettings::TRANSFER_PAGE_SIZE),
        command_issue_region_size((cq_size - cq_start) - this->command_completion_region_size),
        issue_fifo_size(command_issue_region_size >> 4),
        issue_fifo_limit(
            ((cq_start + this->command_issue_region_size) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4),
        completion_fifo_size(command_completion_region_size >> 4),
        completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
        offset(get_absolute_cq_offset(channel, cq_id, cq_size)),
        id(cq_id) {
        TT_ASSERT(
            this->command_completion_region_size % hal.get_alignment(HalMemType::HOST) == 0 and
                this->command_issue_region_size % hal.get_alignment(HalMemType::HOST) == 0,
            "Issue queue and completion queue need to be {}B aligned!",
            hal.get_alignment(HalMemType::HOST));
        TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        this->issue_fifo_wr_ptr = (this->cq_start + this->offset) >> 4;  // In 16B words
        this->issue_fifo_wr_toggle = 0;

        this->completion_fifo_rd_ptr = this->issue_fifo_limit;
        this->completion_fifo_rd_toggle = 0;
    }

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B
    // aligned and remaining space is dedicated for completion queue Smaller issue queues can lead to more stalls for
    // applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t cq_start;
    const uint32_t command_completion_region_size;
    const uint32_t command_issue_region_size;
    const uint8_t id;

    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr;
    bool issue_fifo_wr_toggle;

    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr;
    bool completion_fifo_rd_toggle;

    // TODO add the host addresses from dispatch constants in here
};

class SystemMemoryManager {
private:
    chip_id_t device_id;
    uint8_t num_hw_cqs;
    const std::function<void(uint32_t, uint32_t, const uint8_t*)> fast_write_callable;
    std::vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size;
    uint32_t channel_offset;
    std::vector<int> cq_to_event;
    std::vector<int> cq_to_last_completed_event;
    std::vector<std::mutex> cq_to_event_locks;
    std::vector<tt_cxy_pair> prefetcher_cores;
    std::vector<tt::Writer> prefetch_q_writers;
    std::vector<uint32_t> prefetch_q_dev_ptrs;
    std::vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable;
    std::vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset;
    std::array<LaunchMessageRingBufferState, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>
        worker_launch_message_buffer_state;

public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs);

    uint32_t get_next_event(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        uint32_t next_event = ++this->cq_to_event[cq_id];  // Event ids start at 1
        cq_to_event_locks[cq_id].unlock();
        return next_event;
    }

    void reset_event_id(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        this->cq_to_event[cq_id] = 0;
        cq_to_event_locks[cq_id].unlock();
    }

    void increment_event_id(const uint8_t cq_id, const uint32_t val) {
        cq_to_event_locks[cq_id].lock();
        this->cq_to_event[cq_id] += val;
        cq_to_event_locks[cq_id].unlock();
    }

    void set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
        TT_ASSERT(
            event_id >= this->cq_to_last_completed_event[cq_id],
            "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
            "completed event is {}",
            event_id,
            this->cq_to_last_completed_event[cq_id]);
        cq_to_event_locks[cq_id].lock();
        this->cq_to_last_completed_event[cq_id] = event_id;
        cq_to_event_locks[cq_id].unlock();
    }

    uint32_t get_last_completed_event(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        uint32_t last_completed_event = this->cq_to_last_completed_event[cq_id];
        cq_to_event_locks[cq_id].unlock();
        return last_completed_event;
    }

    void reset(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = 0;
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = 0;
    }

    void set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_size = (issue_queue_size >> 4);
        cq_interface.issue_fifo_limit = (cq_interface.cq_start + cq_interface.offset + issue_queue_size) >> 4;
    }

    void set_bypass_mode(const bool enable, const bool clear) {
        this->bypass_enable = enable;
        if (clear) {
            this->bypass_buffer.clear();
            this->bypass_buffer_write_offset = 0;
        }
    }

    bool get_bypass_mode() { return this->bypass_enable; }

    std::vector<uint32_t>& get_bypass_data() { return this->bypass_buffer; }

    uint32_t get_issue_queue_size(const uint8_t cq_id) const { return this->cq_interfaces[cq_id].issue_fifo_size << 4; }

    uint32_t get_issue_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
    }

    uint32_t get_completion_queue_size(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_size << 4;
    }

    uint32_t get_completion_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
    }

    uint32_t get_issue_queue_write_ptr(const uint8_t cq_id) const {
        if (this->bypass_enable) {
            return this->bypass_buffer_write_offset;
        } else {
            return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
        }
    }

    uint32_t get_completion_queue_read_ptr(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
    }

    uint32_t get_completion_queue_read_toggle(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
    }

    uint32_t get_cq_size() const { return this->cq_size; }

    chip_id_t get_device_id() const { return this->device_id; }

    std::vector<SystemMemoryCQInterface>& get_cq_interfaces() { return this->cq_interfaces; }

    void* issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
        if (this->bypass_enable) {
            uint32_t curr_size = this->bypass_buffer.size();
            uint32_t new_size = curr_size + (cmd_size_B / sizeof(uint32_t));
            this->bypass_buffer.resize(new_size);
            return (void*)((char*)this->bypass_buffer.data() + this->bypass_buffer_write_offset);
        }

        uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

        const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
        if (issue_q_write_ptr + align(cmd_size_B, tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST)) >
            command_issue_limit) {
            this->wrap_issue_queue_wr_ptr(cq_id);
            issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
        }

        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on
        // https://github.com/tenstorrent/tt-metal/issues/4757
        void* issue_q_region = this->cq_sysmem_start + (issue_q_write_ptr - this->channel_offset);

        return issue_q_region;
    }

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on
        // https://github.com/tenstorrent/tt-metal/issues/4757
        void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

        if (this->bypass_enable) {
            std::copy((uint8_t*)data, (uint8_t*)data + size_in_bytes, (uint8_t*)this->bypass_buffer.data() + write_ptr);
        } else {
            memcpy_to_device(user_scratchspace, data, size_in_bytes);
        }
    }

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id);

    uint32_t completion_queue_wait_front(const uint8_t cq_id, volatile bool& exit_condition) const {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        do {
            write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
        } while (cq_interface.completion_fifo_rd_ptr == write_ptr and
                 cq_interface.completion_fifo_rd_toggle == write_toggle and not exit_condition);
        return write_ptr_and_toggle;
    }

    void send_completion_queue_read_ptr(const uint8_t cq_id) const;

    void wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
        if (this->bypass_enable) {
            return;
        }
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
    }

    void wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    void completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
        uint32_t data_read_B = num_pages_read * DispatchSettings::TRANSFER_PAGE_SIZE;
        uint32_t data_read_16B = data_read_B >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr += data_read_16B;
        if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
            cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
            cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
        }

        // Notify dispatch core
        this->send_completion_queue_read_ptr(cq_id);
    }

    void fetch_queue_reserve_back(const uint8_t cq_id);

    void fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id, bool stall_prefetcher = false) {
        CoreType dispatch_core_type =
            tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(this->device_id);
        uint32_t max_command_size_B = DispatchMemMap::get(dispatch_core_type, num_hw_cqs).max_prefetch_command_size();
        TT_ASSERT(
            command_size_B <= max_command_size_B,
            "Generated prefetcher command of size {} B exceeds max command size {} B",
            command_size_B,
            max_command_size_B);
        TT_ASSERT(
            (command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF,
            "FetchQ command too large to represent");
        if (this->bypass_enable) {
            return;
        }
        tt_driver_atomics::sfence();
        DispatchSettings::prefetch_q_entry_type command_size_16B =
            command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE;

        // stall_prefetcher is used for enqueuing traces, as replaying a trace will hijack the cmd_data_q
        // so prefetcher fetches multiple cmds that include the trace cmd, they will be corrupted by trace pulling data
        // from DRAM stall flag prevents pulling prefetch q entries that occur after the stall entry Stall flag for
        // prefetcher is MSB of FetchQ entry.
        if (stall_prefetcher) {
            command_size_16B |= (1 << ((sizeof(DispatchSettings::prefetch_q_entry_type) * 8) - 1));
        }
        this->prefetch_q_writers[cq_id].write(this->prefetch_q_dev_ptrs[cq_id], command_size_16B);
        this->prefetch_q_dev_ptrs[cq_id] += sizeof(DispatchSettings::prefetch_q_entry_type);
    }

    std::array<LaunchMessageRingBufferState, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>&
    get_worker_launch_message_buffer_state() {
        return this->worker_launch_message_buffer_state;
    }

    void reset_worker_launch_message_buffer_state(const uint32_t num_entries) {
        std::for_each(
            this->worker_launch_message_buffer_state.begin(),
            this->worker_launch_message_buffer_state.begin() + num_entries,
            std::mem_fn(&LaunchMessageRingBufferState::reset));
    }
};

}  // namespace tt::tt_metal
