// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/common/math.hpp"

using namespace tt::tt_metal;

// todo consider moving these to dispatch_addr_map
static constexpr uint32_t PCIE_ALIGNMENT = 32;
static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30; // 1GB;

struct dispatch_constants {
   public:
    dispatch_constants &operator=(const dispatch_constants &) = delete;
    dispatch_constants &operator=(dispatch_constants &&other) noexcept = delete;
    dispatch_constants(const dispatch_constants &) = delete;
    dispatch_constants(dispatch_constants &&other) noexcept = delete;

    static const dispatch_constants &get(const CoreType &core_type) {
        static dispatch_constants inst = dispatch_constants(core_type);
        return inst;
    }

    typedef uint32_t prefetch_q_entry_type;
    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;
    static constexpr uint32_t PREFETCH_Q_ENTRIES = 128;
    static constexpr uint32_t PREFETCH_Q_SIZE = PREFETCH_Q_ENTRIES * sizeof(prefetch_q_entry_type);
    static constexpr uint32_t PREFETCH_Q_BASE = DISPATCH_L1_UNRESERVED_BASE;

    static constexpr uint32_t CMDDAT_Q_BASE = PREFETCH_Q_BASE + ((PREFETCH_Q_SIZE + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;
    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;
    static constexpr uint32_t DISPATCH_BUFFER_BASE = ((DISPATCH_L1_UNRESERVED_BASE - 1) | ((1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) + 1;

    static constexpr uint32_t PREFETCH_D_BUFFER_SIZE = 256 * 1024;
    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t PREFETCH_D_BUFFER_BLOCKS = 4;
    static constexpr uint32_t PREFETCH_D_BUFFER_PAGES = PREFETCH_D_BUFFER_SIZE >> PREFETCH_D_BUFFER_LOG_PAGE_SIZE;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;
    // When page size of buffer to write/read exceeds MAX_PREFETCH_COMMAND_SIZE, the PCIe aligned page size is broken down into equal sized partial pages
    // BASE_PARTIAL_PAGE_SIZE denotes the initial partial page size to use, it is incremented by PCIe alignment until page size can be evenly split
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE = 4096;

    uint32_t max_prefetch_command_size() const { return max_prefetch_command_size_; }

    uint32_t cmddat_q_size() const { return cmddat_q_size_; }

    uint32_t scratch_db_base() const { return scratch_db_base_; }

    uint32_t scratch_db_size() const { return scratch_db_size_; }

    uint32_t dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

    uint32_t dispatch_buffer_pages() const { return dispatch_buffer_pages_; }

   private:
    dispatch_constants(const CoreType &core_type) {
        TT_ASSERT(core_type == CoreType::WORKER or core_type == CoreType::ETH);
        // make this 2^N as required by the packetized stages
        uint32_t dispatch_buffer_block_size;
        if (core_type == CoreType::WORKER) {
            max_prefetch_command_size_ = 64 * 1024;
            cmddat_q_size_ = 128 * 1024;
            scratch_db_size_ = 128 * 1024;
            dispatch_buffer_block_size = 512 * 1024;
        } else {
            max_prefetch_command_size_ = 32 * 1024;
            cmddat_q_size_ = 64 * 1024;
            scratch_db_size_ = 64 * 1024;
            dispatch_buffer_block_size = 128 * 1024;
        }
        TT_ASSERT(cmddat_q_size_ >= 2 * max_prefetch_command_size_);
        TT_ASSERT(scratch_db_size_ % 2 == 0);
        TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);
        scratch_db_base_ = CMDDAT_Q_BASE + ((cmddat_q_size_ + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);
        uint32_t l1_size = core_type == CoreType::WORKER ? MEM_L1_SIZE : MEM_ETH_SIZE;
        TT_ASSERT(scratch_db_base_ + scratch_db_size_ < l1_size);
        dispatch_buffer_block_size_pages_ = dispatch_buffer_block_size / (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) / DISPATCH_BUFFER_SIZE_BLOCKS;
        dispatch_buffer_pages_ = dispatch_buffer_block_size_pages_ * DISPATCH_BUFFER_SIZE_BLOCKS;
        uint32_t dispatch_cb_end = DISPATCH_BUFFER_BASE + (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_buffer_pages_;
        TT_ASSERT(dispatch_cb_end < l1_size);
    }

    uint32_t max_prefetch_command_size_;
    uint32_t cmddat_q_size_;
    uint32_t scratch_db_base_;
    uint32_t scratch_db_size_;
    uint32_t dispatch_buffer_block_size_pages_;
    uint32_t dispatch_buffer_pages_;
};

/// @brief Get offset of the command queue relative to its channel
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t relative offset
inline uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) {
    return cq_id * cq_size;
}

/// @brief Get absolute offset of the command queue
/// @param channel uint16_t channel ID (hugepage)
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t absolute offset
inline uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    return (MAX_HUGEPAGE_SIZE * channel) + get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
inline uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_ISSUE_READ_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_COMPLETION_WRITE_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the completion region
    // Equation for issue fifo size is
    // | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size):
      command_completion_region_size((((cq_size - CQ_START) / dispatch_constants::TRANSFER_PAGE_SIZE) / 4) * dispatch_constants::TRANSFER_PAGE_SIZE),
      command_issue_region_size((cq_size - CQ_START) - this->command_completion_region_size),
      issue_fifo_size(command_issue_region_size >> 4),
      issue_fifo_limit(((CQ_START + this->command_issue_region_size) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4),
      completion_fifo_size(command_completion_region_size >> 4),
      completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
      offset(get_absolute_cq_offset(channel, cq_id, cq_size))
     {
        TT_ASSERT(this->command_completion_region_size % PCIE_ALIGNMENT == 0 and this->command_issue_region_size % PCIE_ALIGNMENT == 0, "Issue queue and completion queue need to be {}B aligned!", PCIE_ALIGNMENT);
        TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        this->issue_fifo_wr_ptr = (CQ_START + this->offset) >> 4;  // In 16B words
        this->issue_fifo_wr_toggle = 0;

        this->completion_fifo_rd_ptr = this->issue_fifo_limit;
        this->completion_fifo_rd_toggle = 0;
    }

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B aligned and remaining space is dedicated for completion queue
    // Smaller issue queues can lead to more stalls for applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t command_completion_region_size;
    const uint32_t command_issue_region_size;

    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr;
    bool issue_fifo_wr_toggle;

    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr;
    bool completion_fifo_rd_toggle;
};

class SystemMemoryManager {
   private:
    chip_id_t device_id;
    uint8_t num_hw_cqs;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t*, uint32_t)> fast_write_callable;
    vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start;
    vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size;
    uint32_t channel_offset;
    vector<int> cq_to_event;
    vector<int> cq_to_last_completed_event;
    vector<std::mutex> cq_to_event_locks;
    vector<tt_cxy_pair> prefetcher_cores;
    vector<uint32_t> prefetch_q_dev_ptrs;
    vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable;
    vector<uint32_t> bypass_buffer;

   public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
        device_id(device_id),
        num_hw_cqs(num_hw_cqs),
        m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device_id)),
        fast_write_callable(tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)),
        bypass_enable(false) {
        this->completion_byte_addrs.resize(num_hw_cqs);
        this->prefetcher_cores.resize(num_hw_cqs);
        this->prefetch_q_dev_ptrs.resize(num_hw_cqs);
        this->prefetch_q_dev_fences.resize(num_hw_cqs);

        // Split hugepage into however many pieces as there are CQs
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        char* hugepage_start = (char*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        this->cq_sysmem_start = hugepage_start;

        // TODO(abhullar): Remove env var and expose sizing at the API level
        char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
        if (cq_size_override_env != nullptr) {
            uint32_t cq_size_override = std::stoi(string(cq_size_override_env));
            this->cq_size = cq_size_override;
        } else {
            this->cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
        }
        this->channel_offset = MAX_HUGEPAGE_SIZE * channel;

        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair prefetcher_core = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
            CoreType core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
            tt_cxy_pair prefetcher_physical_core = tt_cxy_pair(prefetcher_core.chip, tt::get_physical_core_coordinate(prefetcher_core, core_type));
            this->prefetcher_cores[cq_id] = prefetcher_physical_core;

            tt_cxy_pair completion_queue_writer_core = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
            const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(completion_queue_writer_core.chip, tt::get_physical_core_coordinate(completion_queue_writer_core, core_type))).value();
            auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;
            this->completion_byte_addrs[cq_id] = completion_tlb_offset + CQ_COMPLETION_READ_PTR % completion_tlb_size;

            this->cq_interfaces.push_back(SystemMemoryCQInterface(channel, cq_id, this->cq_size));
            this->cq_to_event.push_back(0);
            this->cq_to_last_completed_event.push_back(0);
            this->prefetch_q_dev_ptrs[cq_id] = dispatch_constants::PREFETCH_Q_BASE;
            this->prefetch_q_dev_fences[cq_id] = dispatch_constants::PREFETCH_Q_BASE + dispatch_constants::PREFETCH_Q_ENTRIES * sizeof(dispatch_constants::prefetch_q_entry_type);
        }
        vector<std::mutex> temp_mutexes(num_hw_cqs);
        cq_to_event_locks.swap(temp_mutexes);
    }

    uint32_t get_next_event(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        uint32_t next_event = this->cq_to_event[cq_id]++;
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
        TT_ASSERT(event_id >= this->cq_to_last_completed_event[cq_id], "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded completed event is {}", event_id, this->cq_to_last_completed_event[cq_id]);
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
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = 0;
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = 0;
    }

    void set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_size = (issue_queue_size >> 4);
        cq_interface.issue_fifo_limit = (CQ_START + cq_interface.offset + issue_queue_size) >> 4;
    }

    void set_bypass_mode(const bool enable, const bool clear) {
        this->bypass_enable = enable;
        if (clear) {
            this->bypass_buffer.clear();
        }
    }

    bool get_bypass_mode() {
        return this->bypass_enable;
    }

    std::vector<uint32_t> get_bypass_data() {
        return std::move(this->bypass_buffer);
    }

    uint32_t get_issue_queue_size(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].issue_fifo_size << 4;
    }

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
        return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
    }

    uint32_t get_completion_queue_read_ptr(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
    }

    uint32_t get_completion_queue_read_toggle(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
    }

    uint32_t get_cq_size() const {
        return this->cq_size;
    }

    // TODO: rename issue_queue_reserve?
    void issue_queue_reserve_back(uint32_t cmd_size_B, const uint8_t cq_id) {
        if (this->bypass_enable) return;

        uint32_t cmd_size_16B = align(cmd_size_B, 32) >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        uint32_t rd_ptr_and_toggle;
        uint32_t rd_ptr;
        uint32_t rd_toggle;
        do {
            rd_ptr_and_toggle = get_cq_issue_rd_ptr<true>(this->device_id, cq_id, this->cq_size);
            rd_ptr = rd_ptr_and_toggle & 0x7fffffff;
            rd_toggle = rd_ptr_and_toggle >> 31;
        } while (
            cq_interface
                .issue_fifo_wr_ptr < rd_ptr and cq_interface.issue_fifo_wr_ptr + cmd_size_16B > rd_ptr or

            // This is the special case where we wrapped our wr ptr and our rd ptr
            // has not yet moved
            (rd_toggle != cq_interface.issue_fifo_wr_toggle and cq_interface.issue_fifo_wr_ptr == rd_ptr));
    }

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on https://github.com/tenstorrent/tt-metal/issues/4757
        void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

        if (this->bypass_enable) {
            TT_FATAL(size_in_bytes % sizeof(uint32_t) == 0, "Data size_in_bytes={} is not {}-byte aligned", size_in_bytes, sizeof(uint32_t));
            this->bypass_buffer.insert(this->bypass_buffer.end(), (uint32_t*)data, (uint32_t*)data + size_in_bytes / sizeof(uint32_t));
        } else {
            memcpy(user_scratchspace, data, size_in_bytes);
        }
    }

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
        if (this->bypass_enable) return;

        // All data needs to be 32B aligned
        uint32_t push_size_16B = align(push_size_B, 32) >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
            cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;  // In 16B words
            cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle; // Flip the toggle
        } else {
            cq_interface.issue_fifo_wr_ptr += push_size_16B;
        }
    }

    void completion_queue_wait_front(const uint8_t cq_id, volatile bool& exit_condition) const {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        do {
            write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
        } while (cq_interface.completion_fifo_rd_ptr == write_ptr and cq_interface.completion_fifo_rd_toggle == write_toggle and not exit_condition);
    }

    void send_completion_queue_read_ptr(const uint8_t cq_id) const {
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        uint32_t read_ptr_and_toggle =
            cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
        this->fast_write_callable(this->completion_byte_addrs[cq_id], 4, (uint8_t*)&read_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
        if (this->bypass_enable) return;
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
    }

    void wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    void completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
        uint32_t data_read_B = num_pages_read * dispatch_constants::TRANSFER_PAGE_SIZE;
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

    void fetch_queue_reserve_back(const uint8_t cq_id) {
        if (this->bypass_enable) return;

        // Wait for space in the FetchQ
        uint32_t fence;
        while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
            tt::Cluster::instance().read_core(&fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], CQ_PREFETCH_Q_RD_PTR);
            this->prefetch_q_dev_fences[cq_id] = fence;
        }

        // Wrap FetchQ if possible
        uint32_t prefetch_q_base = DISPATCH_L1_UNRESERVED_BASE;
        uint32_t prefetch_q_limit = prefetch_q_base + dispatch_constants::PREFETCH_Q_ENTRIES * sizeof(dispatch_constants::prefetch_q_entry_type);
        if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
            this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;

            while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
                tt::Cluster::instance().read_core(&fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], CQ_PREFETCH_Q_RD_PTR);
                this->prefetch_q_dev_fences[cq_id] = fence;
            }
        }
    }

    void fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id) {
        CoreType dispatch_core_type = dispatch_core_manager::get(this->num_hw_cqs).get_dispatch_core_type(this->device_id);
        uint32_t max_command_size_B = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
        TT_FATAL(command_size_B <= max_command_size_B, "Generated prefetcher command of size {} B exceeds max command size {} B", command_size_B, max_command_size_B);
        TT_FATAL((command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

        if (this->bypass_enable) return;
        uint32_t command_size_16B = command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE;
        tt::Cluster::instance().write_reg(&command_size_16B, this->prefetcher_cores[cq_id], this->prefetch_q_dev_ptrs[cq_id]);
        this->prefetch_q_dev_ptrs[cq_id] += sizeof(dispatch_constants::prefetch_q_entry_type);
        tt_driver_atomics::sfence();
    }
};
