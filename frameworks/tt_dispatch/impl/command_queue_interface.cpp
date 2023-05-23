#include "command_queue_interface.hpp"

u32 get_cq_rd_ptr(Device* device) {
    u32 chip_id = 0;  // TODO(agrebenisan): Remove hard-coding
    vector<u32> recv;
    u32 rd_ptr_addr = 0;
    device->cluster()->read_sysmem_vec(recv, rd_ptr_addr, 4, chip_id);
    return recv.at(0);
}

SystemMemoryWriter::SystemMemoryWriter() {
    this->cq_write_interface.fifo_wr_ptr = (HOST_CQ_FINISH_PTR + 32) >> 4;  // In 16B words, so this would be address 64B
    this->cq_write_interface.fifo_size = 0;
}

// Ensure that there is enough space to push to the queue first
void SystemMemoryWriter::cq_reserve_back(Device* device, u32 cmd_size_B) {
    u32 cmd_size_16B = ((cmd_size_B + 31) / 32) * 2; // Terse way to find next multiple of 32 in 16B words

    // Need to create a NOP to fill in the remaining space
    if (this->cq_write_interface.fifo_wr_ptr + cmd_size_16B > this->cq_write_interface.fifo_limit) {
        TT_ASSERT("Edge case not yet handled");
    }

    u32 rd_ptr;
    do {
        rd_ptr = get_cq_rd_ptr(device);

    } while (this->cq_write_interface.fifo_wr_ptr < rd_ptr and
             this->cq_write_interface.fifo_wr_ptr + cmd_size_16B >= rd_ptr);
}

// Ideally, data should be an array or pointer, but vector for time-being
void SystemMemoryWriter::cq_write(Device* device, vector<u32>& data, u32 write_ptr) {
    device->cluster()->write_sysmem_vec(data, write_ptr, 0);
}

void SystemMemoryWriter::send_write_ptr(Device* device) {
    tt_xy_pair dispatch_core = {1, 11};
    u32 chip_id = 0;  // TODO(agrebenisan): Remove hard-coding

    tt::llrt::write_hex_vec_to_core(
        device->cluster(), chip_id, dispatch_core, {this->cq_write_interface.fifo_wr_ptr}, CQ_WRITE_PTR, false);
}

void SystemMemoryWriter::cq_push_back(Device* device, u32 push_size_B) {
    // All data needs to be 32B aligned
    u32 push_size_16B = ((push_size_B + 31) / 32) * 2; // Terse way to find next multiple of 32 in 16B words

    this->cq_write_interface.fifo_wr_ptr += push_size_16B;

    // Notify dispatch core
    this->send_write_ptr(device);

    // if (this->cq_write_interface.fifo_wr_ptr > this->cq_write_interface.fifo_limit) {
    //     this->cq_write_interface.fifo_wr_ptr -= this->cq_write_interface.fifo_size;
    // }
}
