#include "command_queue_interface.hpp"

u32 get_cq_rd_ptr(Device* device) {
    u32 chip_id = 0;  // TODO(agrebenisan): Remove hard-coding
    vector<u32> recv;
    device->cluster()->read_sysmem_vec(recv, HOST_CQ_READ_PTR, 4, chip_id);
    return recv.at(0);
}

u32 get_cq_rd_toggle(Device* device) {
    u32 chip_id = 0;  // TODO(agrebenisan): Remove hard-coding
    vector<u32> recv;
    device->cluster()->read_sysmem_vec(recv, HOST_CQ_READ_TOGGLE_PTR, 4, chip_id);
    return recv.at(0);
}

SystemMemoryWriter::SystemMemoryWriter() {
    this->cq_write_interface.fifo_wr_ptr = CQ_START >> 4;  // In 16B words
    this->cq_write_interface.fifo_wr_toggle = 0; // This is used for the edge case where we wrap and our read pointer has not yet moved
}

// Ensure that there is enough space to push to the queue first
void SystemMemoryWriter::cq_reserve_back(Device* device, u32 cmd_size_B) {
    u32 cmd_size_16B = (((cmd_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words

    u32 rd_ptr;
    u32 rd_toggle;
    do {
        rd_ptr = get_cq_rd_ptr(device);
        rd_toggle = get_cq_rd_toggle(device);
    } while (this->cq_write_interface.fifo_wr_ptr < rd_ptr and
             this->cq_write_interface.fifo_wr_ptr + cmd_size_16B >= rd_ptr or

             // This is the special case where we wrapped our wr ptr and our rd ptr
             // has not yet moved
             (rd_toggle != this->cq_write_interface.fifo_wr_toggle and this->cq_write_interface.fifo_wr_ptr == rd_ptr));
}

// Ideally, data should be an array or pointer, but vector for time-being
void SystemMemoryWriter::cq_write(Device* device, vector<u32>& data, u32 write_ptr) {
    device->cluster()->write_sysmem_vec(data, write_ptr, 0);
}

void SystemMemoryWriter::send_write_ptr(Device* device) {
    CoreCoord dispatch_core = {1, 11};
    u32 chip_id = 0;  // TODO(agrebenisan): Remove hard-coding

    _mm_sfence();

    tt::llrt::write_hex_vec_to_core(
        device->cluster(), chip_id, dispatch_core, {this->cq_write_interface.fifo_wr_ptr}, CQ_WRITE_PTR, false);

    _mm_sfence();
}

void SystemMemoryWriter::send_write_toggle(Device* device) {
    CoreCoord dispatch_core = {1, 11};
    u32 chip_id = 0;  // TODO(agrebenisan): Remove hard-coding

    _mm_sfence();

    tt::llrt::write_hex_vec_to_core(
        device->cluster(), chip_id, dispatch_core, {this->cq_write_interface.fifo_wr_toggle}, CQ_WRITE_TOGGLE, true);

    _mm_sfence();
}

void SystemMemoryWriter::cq_push_back(Device* device, u32 push_size_B) {

    // All data needs to be 32B aligned
    u32 push_size_16B = (((push_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words

    this->cq_write_interface.fifo_wr_ptr += push_size_16B;

    if (this->cq_write_interface.fifo_wr_ptr > this->cq_write_interface.fifo_limit) {
        this->cq_write_interface.fifo_wr_ptr = CQ_START >> 4;

        // Flip the toggle
        this->cq_write_interface.fifo_wr_toggle = not this->cq_write_interface.fifo_wr_toggle;
        this->send_write_toggle(device);
    }

    // Notify dispatch core
    this->send_write_ptr(device);
}
