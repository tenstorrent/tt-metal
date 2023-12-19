// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/common/math.hpp"

using namespace tt::tt_metal;

inline uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_ISSUE_READ_PTR, mmio_device_id, channel);
    return recv;
}

inline uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_COMPLETION_WRITE_PTR, mmio_device_id, channel);
    return recv;
}

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the completion region
    // Equation for issue fifo size is
    // | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint32_t command_queue_size, std::optional<float> command_issue_queue_split) : command_queue_size(command_queue_size) {
        float issue_queue_split = command_issue_queue_split.has_value() ? command_issue_queue_split.value() : default_issue_queue_split;

        this->command_issue_region_size = tt::round_up(command_queue_size * issue_queue_split, 32);
        this->command_completion_region_size = this->command_queue_size - this->command_issue_region_size;

        this->issue_fifo_size = (this->command_issue_region_size - CQ_START) >> 4;
        this->issue_fifo_limit = (this->command_issue_region_size >> 4) - 1;
        this->issue_fifo_wr_ptr = CQ_START >> 4;  // In 16B words
        this->issue_fifo_wr_toggle =
            0;  // This is used for the edge case where we wrap and our read pointer has not yet moved

        this->completion_fifo_size = this->command_completion_region_size >> 4;
        this->completion_fifo_limit = (this->command_queue_size >> 4) - 1;
        this->completion_fifo_rd_ptr = this->command_issue_region_size >> 4; // completion region is below issue region
        this->completion_fifo_rd_toggle = 0;
    }

    const uint32_t command_queue_size;
    static constexpr float default_issue_queue_split = 0.75;
    uint32_t command_issue_region_size;
    uint32_t command_completion_region_size;

    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // Last possible FIFO address
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
    // Data required for fast writes to write pointer location
    // in prefetch core's L1
    // const std::tuple<uint32_t, uint32_t> tlb_data;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t*, uint32_t)> fast_write_callable;
    const std::set<CoreCoord> dispatch_cores;
    const std::function<CoreCoord (CoreCoord)>worker_from_logical_callable;
    uint32_t issue_byte_addr;
    uint32_t completion_byte_addr;
    char* hugepage_start;

   public:
    SystemMemoryCQInterface cq_interface;
    SystemMemoryManager(chip_id_t device_id, const std::set<CoreCoord> &dev_dispatch_cores, const std::function<CoreCoord (CoreCoord)> &worker_from_logical) :
        device_id(device_id),
        m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device->id())),
        hugepage_start(
            (char*) tt::Cluster::instance().host_dma_address(0, tt::Cluster::instance().get_associated_mmio_device(device_id), tt::Cluster::instance().get_assigned_channel_for_device(device_id))),
        fast_write_callable(
            tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)),
        dispatch_cores(dev_dispatch_cores),
        worker_from_logical_callable(worker_from_logical) {
        
        auto dispatch_cores_iter = dispatch_cores.begin();
        const std::tuple<uint32_t, uint32_t> producer_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(device_id, this->worker_from_logical_callable(*dispatch_cores_iter++))).value();
        auto [producer_tlb_offset, producer_tlb_size] = producer_tlb_data;
        this->issue_byte_addr = producer_tlb_offset + CQ_ISSUE_WRITE_PTR % producer_tlb_size;

        const std::tuple<uint32_t, uint32_t> consumer_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(device_id, this->worker_from_logical_callable(*dispatch_cores_iter))).value();
        auto [consumer_tlb_offset, consumer_tlb_size] = consumer_tlb_data;
        this->completion_byte_addr = consumer_tlb_offset + CQ_COMPLETION_READ_PTR % consumer_tlb_size;
    }

    uint32_t get_issue_queue_write_ptr() const {
        return this->cq_interface.issue_fifo_wr_ptr << 4;
    }

    uint32_t get_completion_queue_read_ptr() const {
        return this->cq_interface.completion_fifo_rd_ptr << 4;
    }

    void cq_reserve_back(uint32_t cmd_size_B) const {
        uint32_t cmd_size_16B =
            (((cmd_size_B - 1) | 31) + 1) >> 4;  // Terse way to find next multiple of 32 in 16B words

        uint32_t rd_ptr_and_toggle;
        uint32_t rd_ptr;
        uint32_t rd_toggle;
        do {
            rd_ptr_and_toggle = get_cq_issue_rd_ptr(this->device_id);
            rd_ptr = rd_ptr_and_toggle & 0x7fffffff;
            rd_toggle = rd_ptr_and_toggle >> 31;

        } while (
            this->cq_interface
                .issue_fifo_wr_ptr<rd_ptr and this->cq_interface.issue_fifo_wr_ptr + cmd_size_16B> rd_ptr or

            // This is the special case where we wrapped our wr ptr and our rd ptr
            // has not yet moved
            (rd_toggle != this->cq_interface.issue_fifo_wr_toggle and this->cq_interface.issue_fifo_wr_ptr == rd_ptr));
    }

    // Ideally, data should be an array or pointer, but vector for time-being
    // TODO ALMEET: MEASURE THIS
    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) const {
        // There is a 50% overhead if hugepage_start is not made static.
        // Eventually when we want to have multiple hugepages, we may need to template
        // the sysmem writer to get this optimization.
        /*static*/ char* hugepage_start = this->hugepage_start;
        void* user_scratchspace = hugepage_start + write_ptr;
        memcpy(user_scratchspace, data, size_in_bytes);
    }

    void send_write_ptr() const {
        static CoreCoord dispatch_core =
            this->worker_from_logical_callable(*this->dispatch_cores.begin());

        uint32_t write_ptr_and_toggle =
            this->cq_interface.issue_fifo_wr_ptr | (this->cq_interface.issue_fifo_wr_toggle << 31);
        this->fast_write_callable(this->issue_byte_addr, 4, (uint8_t*)&write_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void cq_push_back(uint32_t push_size_B) {
        // All data needs to be 32B aligned
        uint32_t push_size_16B =
            (((push_size_B - 1) | 31) + 1) >> 4;  // Terse way to find next multiple of 32 in 16B words

        this->cq_interface.issue_fifo_wr_ptr += push_size_16B;

        if (this->cq_interface.issue_fifo_wr_ptr > this->cq_interface.issue_fifo_limit) {
            this->cq_interface.issue_fifo_wr_ptr = CQ_START >> 4;

            // Flip the toggle
            this->cq_interface.issue_fifo_wr_toggle = not this->cq_interface.issue_fifo_wr_toggle;
        }

        // Notify dispatch core
        this->send_write_ptr();
    }

    void cq_wait_front() {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;
        do {
            write_ptr_and_toggle = get_cq_completion_wr_ptr(this->device_id);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
        } while (this->cq_interface.completion_fifo_rd_ptr == write_ptr and this->cq_interface.completion_fifo_rd_toggle == write_toggle);
    }

    void send_read_ptr() const {
        static CoreCoord dispatch_core =
            this->worker_from_logical_callable(*(++this->dispatch_cores.begin()));

        uint32_t read_ptr_and_toggle =
            this->cq_interface.completion_fifo_rd_ptr | (this->cq_interface.completion_fifo_rd_toggle << 31);
        this->fast_write_callable(this->completion_byte_addr, 4, (uint8_t*)&read_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void wrap_completion_queue_locally() {
        cq_interface.completion_fifo_rd_ptr = cq_interface.command_issue_region_size >> 4;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    void cq_pop_front(uint32_t data_read_B) {
        uint32_t data_read_16B =
            (((data_read_B - 1) | 31) + 1) >> 4;  // Terse way to find next multiple of 32 in 16B words
        cq_interface.completion_fifo_rd_ptr += data_read_16B;
        if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
            cq_interface.completion_fifo_rd_ptr = cq_interface.command_issue_region_size >> 4;
            cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
        }

        // Notify dispatch core
        this->send_read_ptr();
    }
};
