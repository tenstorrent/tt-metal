#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool test_interleaved_l1_buffers_basic_allocator(tt_metal::Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) {
    bool pass = true;

    uint32_t buffer_size = num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry;

    auto interleaved_buffer = tt_metal::CreateInterleavedL1Buffer(device, num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);

    std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    tt_metal::WriteToDeviceL1Interleaved(interleaved_buffer, host_buffer);

    std::vector<uint32_t> readback_buffer;
    tt_metal::ReadFromDeviceL1Interleaved(interleaved_buffer, readback_buffer);

    pass &= (host_buffer == readback_buffer);

    return pass;
}

bool test_interleaved_l1_buffers_l1_banking_allocator(tt_metal::Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) {
    bool pass = true;

    uint32_t buffer_size = num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry;

    auto interleaved_buffer = tt_metal::CreateInterleavedL1Buffer(device, num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);

    std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    tt_metal::WriteToDeviceL1Interleaved(interleaved_buffer, host_buffer);

    std::vector<uint32_t> readback_buffer;
    tt_metal::ReadFromDeviceL1Interleaved(interleaved_buffer, readback_buffer);

    pass &= (host_buffer == readback_buffer);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        uint32_t single_tile_size = 2 * 1024;
        int num_entries_per_bank_unit = 512;
        int num_bytes_per_entry = 4;

        int num_bank_units_one = 258;
        int num_bank_units_two = 378;

        // First run tests with basic memory allocator
        pass &= tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::BASIC);
        pass &= test_interleaved_l1_buffers_basic_allocator(device, num_bank_units_one, num_entries_per_bank_unit, num_bytes_per_entry);
        pass &= test_interleaved_l1_buffers_basic_allocator(device, num_bank_units_two, num_entries_per_bank_unit, num_bytes_per_entry);

        // Close device and re-initialize it with L1 banking allocator
        pass &= tt_metal::CloseDevice(device);
        pass &= tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);

        pass &= test_interleaved_l1_buffers_l1_banking_allocator(device, num_bank_units_one, num_entries_per_bank_unit, num_bytes_per_entry);
        pass &= test_interleaved_l1_buffers_l1_banking_allocator(device, num_bank_units_two, num_entries_per_bank_unit, num_bytes_per_entry);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
