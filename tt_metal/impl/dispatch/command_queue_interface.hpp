// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/common/math.hpp"

using namespace tt::tt_metal;

template <bool channel_offet>
inline uint32_t cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    uint32_t offset = (cq_id * cq_size);
    if (channel_offet) {
        offset += (DeviceCommand::MAX_HUGEPAGE_SIZE * channel);
    }
    return offset;
}

template <bool addr_16B>
inline uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    // no need to account for channle in the cq offset here just use offset using cq_id ? ...
    static constexpr bool add_channel_offset = false;
    std::cout << "Reading cq issue read ptr for device " << chip_id
              << " from addr: " << HOST_CQ_ISSUE_READ_PTR + cq_offset<add_channel_offset>(channel, cq_id, cq_size)
              << " at channel: " << channel << std::endl;
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_ISSUE_READ_PTR + cq_offset<add_channel_offset>(channel, cq_id, cq_size), mmio_device_id, channel);
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
    // no need to account for channle in the cq offset here just use offset using cq_id ? ...
    static constexpr bool add_channel_offset = false;
    std::cout << "Reading cq completion write ptr for device " << chip_id
              << " from addr: " << HOST_CQ_COMPLETION_WRITE_PTR + cq_offset<add_channel_offset>(channel, cq_id, cq_size)
              << " at channel: " << channel << std::endl;
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_COMPLETION_WRITE_PTR + cq_offset<add_channel_offset>(channel, cq_id, cq_size), mmio_device_id, channel);
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
      command_issue_region_size(tt::round_up((cq_size - CQ_START) * this->default_issue_queue_split, 32)),
      command_completion_region_size((cq_size - CQ_START) - this->command_issue_region_size),
      issue_fifo_size(command_issue_region_size >> 4),
      issue_fifo_limit(((CQ_START + this->command_issue_region_size) + cq_offset<true>(channel, cq_id, cq_size)) >> 4),
      completion_fifo_size(command_completion_region_size >> 4),
      completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
      offset(cq_offset<true>(channel, cq_id, cq_size))
     {
        TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
        this->issue_fifo_wr_ptr = (CQ_START + this->offset) >> 4;  // In 16B words
        this->issue_fifo_wr_toggle = 0;

        this->completion_fifo_rd_ptr = this->issue_fifo_limit;
        this->completion_fifo_rd_toggle = 0;

        std::cout << "System Memory CQ Interface" << std::endl;
        std::cout << "\tcommand_issue_region_size: " << command_issue_region_size
                  << " command_completion_region_size: " << command_completion_region_size
                  << " issue_fifo_size: " << issue_fifo_size
                  << " issue_fifo_limit: " << issue_fifo_limit
                  << " offset: " << offset
                  << " issue_fifo_wr_ptr: " << issue_fifo_wr_ptr
                  << " completion_fifo_size: " << completion_fifo_size
                  << " completion_fifo_limit: " << completion_fifo_limit
                  << " completion_fifo_rd_ptr: " << completion_fifo_rd_ptr << std::endl;
    }

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B aligned and remaining space is dedicated for completion queue
    // Smaller issue queues can lead to more stalls for applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t command_issue_region_size;
    const uint32_t command_completion_region_size;

    const uint32_t issue_fifo_size;
    const uint32_t issue_fifo_limit;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr;
    bool issue_fifo_wr_toggle;

    const uint32_t completion_fifo_size;
    const uint32_t completion_fifo_limit;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr;
    bool completion_fifo_rd_toggle;
};

class SystemMemoryManager {
   private:
    chip_id_t device_id;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t*, uint32_t)> fast_write_callable;
    vector<uint32_t> issue_byte_addrs;
    vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start;
    vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size;
    uint32_t channel_offset;

   public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
        device_id(device_id),
        m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device_id)),
        fast_write_callable(
            tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)) {

        this->issue_byte_addrs.resize(num_hw_cqs);
        this->completion_byte_addrs.resize(num_hw_cqs);

        // Split hugepage into however many pieces as there are CQs
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        char* hugepage_start = (char*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        this->cq_sysmem_start = hugepage_start;
        this->cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
        this->channel_offset = DeviceCommand::MAX_HUGEPAGE_SIZE * channel;

        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair issue_queue_interface_core = dispatch_core_manager::get(num_hw_cqs).issue_queue_interface_core(device_id, channel, cq_id);
            const std::tuple<uint32_t, uint32_t> issue_interface_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(issue_queue_interface_core.chip, get_physical_dispatch_coord(issue_queue_interface_core))).value();
            auto [issue_tlb_offset, issue_tlb_size] = issue_interface_tlb_data;
            this->issue_byte_addrs[cq_id] = issue_tlb_offset + CQ_ISSUE_WRITE_PTR % issue_tlb_size;

            tt_cxy_pair completion_queue_interface_core = dispatch_core_manager::get(num_hw_cqs).completion_queue_interface_core(device_id, channel, cq_id);
            const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(completion_queue_interface_core.chip, get_physical_dispatch_coord(completion_queue_interface_core))).value();
            auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;
            this->completion_byte_addrs[cq_id] = completion_tlb_offset + CQ_COMPLETION_READ_PTR % completion_tlb_size;

            this->cq_interfaces.push_back(SystemMemoryCQInterface(channel, cq_id, this->cq_size));
        }
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

    void issue_queue_reserve_back(uint32_t cmd_size_B, const uint8_t cq_id) const {
        uint32_t cmd_size_16B = align(cmd_size_B, 32) >> 4;

        uint32_t rd_ptr_and_toggle;
        uint32_t rd_ptr;
        uint32_t rd_toggle;
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
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

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) const {
        // this->cq_sysmem_start gives start of hugepage for a given channel from perspective of host
        //  but all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        std::cout << "memcopy to address cq_sysmem_start + " << (write_ptr - this->channel_offset) << std::endl;
        void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

        memcpy(user_scratchspace, data, size_in_bytes);
    }

    void send_issue_queue_write_ptr(const uint8_t cq_id) const {
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        uint32_t write_ptr_and_toggle =
            cq_interface.issue_fifo_wr_ptr | (cq_interface.issue_fifo_wr_toggle << 31);
        // std::cout << "Sending " << (cq_interface.issue_fifo_wr_ptr << 4) << " to device" << std::endl;
        this->fast_write_callable(this->issue_byte_addrs[cq_id], 4, (uint8_t*)&write_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void issue_queue_push_back(uint32_t push_size_B, bool lazy, const uint8_t cq_id) {
        // All data needs to be 32B aligned

        uint32_t push_size_16B = align(push_size_B, 32) >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        cq_interface.issue_fifo_wr_ptr += push_size_16B;

        if (cq_interface.issue_fifo_wr_ptr >= cq_interface.issue_fifo_limit) {
            cq_interface.issue_fifo_wr_ptr -= cq_interface.issue_fifo_size;

            // Flip the toggle
            cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
        }

        // Notify dispatch core
        if (not lazy) {
            this->send_issue_queue_write_ptr(cq_id);
        }
    }

    void completion_queue_wait_front(const uint8_t cq_id) {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        do {
            write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
        } while (cq_interface.completion_fifo_rd_ptr == write_ptr and cq_interface.completion_fifo_rd_toggle == write_toggle);
    }

    void send_completion_queue_read_ptr(const uint8_t cq_id) const {
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        uint32_t read_ptr_and_toggle =
            cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
        this->fast_write_callable(this->completion_byte_addrs[cq_id], 4, (uint8_t*)&read_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
        this->send_issue_queue_write_ptr(cq_id);
    }

    void wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    void completion_queue_pop_front(uint32_t data_read_B, const uint8_t cq_id) {
        uint32_t data_read_16B = align(data_read_B, 32) >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr += data_read_16B;
        if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
            cq_interface.completion_fifo_rd_ptr = cq_interface.command_issue_region_size >> 4;
            cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
        }

        // Notify dispatch core
        this->send_completion_queue_read_ptr(cq_id);
    }
};
