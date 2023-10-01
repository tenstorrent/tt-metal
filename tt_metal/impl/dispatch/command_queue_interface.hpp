// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/llrt/llrt.hpp"

using namespace tt::tt_metal;

inline uint32_t get_cq_rd_ptr(chip_id_t chip_id) {
    uint32_t recv;
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_READ_PTR, chip_id);
    return recv;
}

struct SystemMemoryCQWriteInterface {
    // Equation for fifo size is
    // | fifo_wr_ptr + command size B - fifo_rd_ptr |
    // Space available would just be fifo_limit - fifo_size
    SystemMemoryCQWriteInterface() {
        this->fifo_wr_ptr = CQ_START >> 4;  // In 16B words
        this->fifo_wr_toggle =
            0;  // This is used for the edge case where we wrap and our read pointer has not yet moved
    }

    const uint32_t fifo_size = ((DeviceCommand::HUGE_PAGE_SIZE)-CQ_START) >> 4;
    const uint32_t fifo_limit = ((DeviceCommand::HUGE_PAGE_SIZE) >> 4) - 1;  // Last possible FIFO address

    uint32_t fifo_wr_ptr;
    bool fifo_wr_toggle;
};

class SystemMemoryWriter {
   private:
    Device* device;
    // Data required for fast writes to write pointer location
    // in prefetch core's L1
    // const std::tuple<uint32_t, uint32_t> tlb_data;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t*, uint32_t)> fast_write_callable;
    uint32_t byte_addr;
    char* hugepage_start;

   public:
    SystemMemoryCQWriteInterface cq_write_interface;
    SystemMemoryWriter(Device* device) :
        m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device->id())),
        hugepage_start((char*) tt::Cluster::instance().host_dma_address(0, device->id(), 0)),
        fast_write_callable(
            tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device->id())) {

        this->device = device;
        const std::tuple<uint32_t, uint32_t> tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(device->id(), this->device->worker_core_from_logical_core(*device->dispatch_cores().begin()))).value();
        auto [tlb_offset, tlb_size] = tlb_data;
        this->byte_addr = tlb_offset + CQ_WRITE_PTR % tlb_size;
    }

    void cq_reserve_back(uint32_t cmd_size_B) const {
        uint32_t cmd_size_16B =
            (((cmd_size_B - 1) | 31) + 1) >> 4;  // Terse way to find next multiple of 32 in 16B words

        uint32_t rd_ptr_and_toggle;
        uint32_t rd_ptr;
        uint32_t rd_toggle;
        do {
            rd_ptr_and_toggle = get_cq_rd_ptr(this->device->id());
            rd_ptr = rd_ptr_and_toggle & 0x7fffffff;
            rd_toggle = rd_ptr_and_toggle >> 31;

        } while (
            this->cq_write_interface
                .fifo_wr_ptr<rd_ptr and this->cq_write_interface.fifo_wr_ptr + cmd_size_16B> rd_ptr or

            // This is the special case where we wrapped our wr ptr and our rd ptr
            // has not yet moved
            (rd_toggle != this->cq_write_interface.fifo_wr_toggle and this->cq_write_interface.fifo_wr_ptr == rd_ptr));
    }

    // Ideally, data should be an array or pointer, but vector for time-being
    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) const {
        // There is a 50% overhead if hugepage_start is not made static.
        // Eventually when we want to have multiple hugepages, we may need to template
        // the sysmem writer to get this optimization.
        static char* hugepage_start = this->hugepage_start;
        void* user_scratchspace = hugepage_start + write_ptr;
        memcpy(user_scratchspace, data, size_in_bytes);
    }

    void send_write_ptr() const {
        static CoreCoord dispatch_core =
            this->device->worker_core_from_logical_core(*this->device->dispatch_cores().begin());

        uint32_t write_ptr_and_toggle =
            this->cq_write_interface.fifo_wr_ptr | (this->cq_write_interface.fifo_wr_toggle << 31);
        this->fast_write_callable(this->byte_addr, 4, (uint8_t*)&write_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void cq_push_back(uint32_t push_size_B) {
        // All data needs to be 32B aligned
        uint32_t push_size_16B =
            (((push_size_B - 1) | 31) + 1) >> 4;  // Terse way to find next multiple of 32 in 16B words

        this->cq_write_interface.fifo_wr_ptr += push_size_16B;

        if (this->cq_write_interface.fifo_wr_ptr > this->cq_write_interface.fifo_limit) {
            this->cq_write_interface.fifo_wr_ptr = CQ_START >> 4;

            // Flip the toggle
            this->cq_write_interface.fifo_wr_toggle = not this->cq_write_interface.fifo_wr_toggle;
        }

        // Notify dispatch core
        this->send_write_ptr();
    }
};
