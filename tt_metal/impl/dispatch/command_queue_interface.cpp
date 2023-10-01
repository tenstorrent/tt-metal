// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_interface.hpp"
#include "tt_metal/common/concurrency_interface.hpp"

uint32_t get_cq_rd_ptr(chip_id_t chip_id) {
    vector<uint32_t> recv;
    tt::Cluster::instance().read_sysmem_vec(recv, HOST_CQ_READ_PTR, 4, chip_id);
    return recv.at(0);
}

uint32_t get_cq_rd_toggle(chip_id_t chip_id) {
    vector<uint32_t> recv;
    tt::Cluster::instance().read_sysmem_vec(recv, HOST_CQ_READ_TOGGLE_PTR, 4, chip_id);
    return recv.at(0);
}

SystemMemoryWriter::SystemMemoryWriter() {}

// Ensure that there is enough space to push to the queue first

void SystemMemoryWriter::cq_reserve_back(Device* device, uint32_t cmd_size_B) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device_id(device->id());
    tt::concurrent::sys_mem_cq_write_interface_t *cq_write_interface = tt::concurrent::get_cq_write_interface(mmio_device_id).first;

    uint32_t cmd_size_16B = (((cmd_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words


    uint32_t rd_ptr;
    uint32_t rd_toggle;
    do {
        rd_ptr = get_cq_rd_ptr(mmio_device_id);
        rd_toggle = get_cq_rd_toggle(mmio_device_id);
    } while (cq_write_interface->fifo_wr_ptr < rd_ptr and
             cq_write_interface->fifo_wr_ptr + cmd_size_16B >= rd_ptr or

             // This is the special case where we wrapped our wr ptr and our rd ptr
             // has not yet moved
             (rd_toggle != cq_write_interface->fifo_wr_toggle and cq_write_interface->fifo_wr_ptr == rd_ptr));
}

// Ideally, data should be an array or pointer, but vector for time-being
void SystemMemoryWriter::cq_write(Device* device, const uint32_t* data, uint32_t size, uint32_t write_ptr) {
    tt::Cluster::instance().write_sysmem_vec(data, size, write_ptr, device->id());
}

void SystemMemoryWriter::send_write_ptr(Device* device) {
    static CoreCoord dispatch_core = device->worker_core_from_logical_core(*device->dispatch_cores().begin());
    uint32_t chip_id = 0;  // TODO(agrebenisan): Remove hard-coding

    tt_driver_atomics::sfence();

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device_id(device->id());
    tt::concurrent::sys_mem_cq_write_interface_t *cq_write_interface = tt::concurrent::get_cq_write_interface(mmio_device_id).first;

    tt::llrt::write_hex_vec_to_core(chip_id, dispatch_core, {cq_write_interface->fifo_wr_ptr}, CQ_WRITE_PTR, false);

    tt_driver_atomics::sfence();
}

void SystemMemoryWriter::send_write_toggle(Device* device) {
    static CoreCoord dispatch_core = device->worker_core_from_logical_core(*device->dispatch_cores().begin());
    uint32_t chip_id = 0;  // TODO(agrebenisan): Remove hard-coding

    tt_driver_atomics::sfence();

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device_id(device->id());
    tt::concurrent::sys_mem_cq_write_interface_t *cq_write_interface = tt::concurrent::get_cq_write_interface(mmio_device_id).first;

    tt::llrt::write_hex_vec_to_core(chip_id, dispatch_core, {cq_write_interface->fifo_wr_toggle}, CQ_WRITE_TOGGLE, true);

    tt_driver_atomics::sfence();
}

void SystemMemoryWriter::cq_push_back(Device* device, uint32_t push_size_B) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device_id(device->id());
    tt::concurrent::sys_mem_cq_write_interface_t *cq_write_interface = tt::concurrent::get_cq_write_interface(mmio_device_id).first;


    // All data needs to be 32B aligned
    uint32_t push_size_16B = (((push_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words

    cq_write_interface->fifo_wr_ptr += push_size_16B;

    if (cq_write_interface->fifo_wr_ptr > cq_write_interface->fifo_limit) {
        cq_write_interface->fifo_wr_ptr = CQ_START >> 4;

        // Flip the toggle
        cq_write_interface->fifo_wr_toggle = not cq_write_interface->fifo_wr_toggle;
        this->send_write_toggle(device);
    }

    // Notify dispatch core
    this->send_write_ptr(device);
}
