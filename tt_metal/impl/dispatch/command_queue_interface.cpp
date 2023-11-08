// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_interface.hpp"

uint32_t get_cq_rd_ptr(chip_id_t chip_id) {
    vector<uint32_t> recv;
    tt::Cluster::instance().read_sysmem_vec(recv, HOST_CQ_READ_PTR, 4, chip_id);
    return recv[0];
}

SystemMemoryWriter::SystemMemoryWriter() {
    this->cq_write_interface.fifo_wr_ptr = CQ_START >> 4;  // In 16B words
    this->cq_write_interface.fifo_wr_toggle = 0; // This is used for the edge case where we wrap and our read pointer has not yet moved
}

// Ensure that there is enough space to push to the queue first
void SystemMemoryWriter::cq_reserve_back(Device* device, uint32_t cmd_size_B) const {
    uint32_t cmd_size_16B = (((cmd_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words

    uint32_t rd_ptr_and_toggle;
    uint32_t rd_ptr;
    uint32_t rd_toggle;
    do {
        rd_ptr_and_toggle = get_cq_rd_ptr(device->id());
        rd_ptr = rd_ptr_and_toggle & 0x7fffffff;
        rd_toggle = rd_ptr_and_toggle >> 31;

    } while (this->cq_write_interface.fifo_wr_ptr < rd_ptr and
             this->cq_write_interface.fifo_wr_ptr + cmd_size_16B > rd_ptr or

             // This is the special case where we wrapped our wr ptr and our rd ptr
             // has not yet moved
             (rd_toggle != this->cq_write_interface.fifo_wr_toggle and this->cq_write_interface.fifo_wr_ptr == rd_ptr));
}

void SystemMemoryWriter::send_write_ptr(Device* device) const {
    static CoreCoord dispatch_core = device->worker_core_from_logical_core(*device->dispatch_cores().begin());
    uint32_t chip_id = 0;  // TODO(agrebenisan): Remove hard-coding

    tt_driver_atomics::sfence();

    uint32_t write_ptr_and_toggle = this->cq_write_interface.fifo_wr_ptr | (this->cq_write_interface.fifo_wr_toggle << 31);
    tt::llrt::write_hex_vec_to_core(chip_id, dispatch_core, {write_ptr_and_toggle}, CQ_WRITE_PTR, false);

    tt_driver_atomics::sfence();
}

void SystemMemoryWriter::cq_push_back(Device* device, uint32_t push_size_B) {

    // All data needs to be 32B aligned
    uint32_t push_size_16B = (((push_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words

    this->cq_write_interface.fifo_wr_ptr += push_size_16B;

    if (this->cq_write_interface.fifo_wr_ptr > this->cq_write_interface.fifo_limit) {
        this->cq_write_interface.fifo_wr_ptr = CQ_START >> 4;

        // Flip the toggle
        this->cq_write_interface.fifo_wr_toggle = not this->cq_write_interface.fifo_wr_toggle;
    }

    // Notify dispatch core
    this->send_write_ptr(device);
}
