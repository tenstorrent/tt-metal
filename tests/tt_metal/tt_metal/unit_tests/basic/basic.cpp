#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

#include "catch.hpp"

using namespace tt;

// Ping a set number of bytes into the specified address of L1
bool l1_ping(tt_metal::Device* device, const size_t& byte_size, const size_t& l1_byte_address, const CoreCoord& grid_size) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    auto inputs = tt::test_utils::generate_uniform_int_random_vector<uint32_t>(0, UINT32_MAX, byte_size);
    for(int y = 0 ; y < grid_size.y; y++) {
        for(int x = 0 ; x < grid_size.x; x++) {
            CoreCoord dest_core({.x=static_cast<size_t>(x), .y=static_cast<size_t>(y)});
            tt_metal::WriteToDeviceL1(device, dest_core, l1_byte_address, inputs);
        }
    }
    for(int y = 0 ; y < grid_size.y; y++) {
        for(int x = 0 ; x < grid_size.x; x++) {
            CoreCoord dest_core({.x=static_cast<size_t>(x), .y=static_cast<size_t>(y)});
            std::vector<uint32_t> dest_core_data;
            tt_metal::ReadFromDeviceL1(device, dest_core, l1_byte_address, byte_size, dest_core_data);
            pass &= (dest_core_data == inputs);
            if (not pass) {
                INFO("Mismatch at Core: " << dest_core.str());
            }
        }
    }
    return pass;
}

// Ping a set number of bytes into the specified address of DRAM
bool dram_ping(tt_metal::Device* device, const size_t& byte_size, const size_t& dram_byte_address, const unsigned int& num_channels) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    auto inputs = tt::test_utils::generate_uniform_int_random_vector<uint32_t>(0, UINT32_MAX, byte_size);
    for(unsigned int channel = 0 ; channel < num_channels; channel++) {
        tt_metal::WriteToDeviceDRAMChannel(device, channel, dram_byte_address, inputs);
    }
    for(unsigned int channel = 0 ; channel < num_channels; channel++) {
        std::vector<uint32_t> dest_channel_data;
        tt_metal::ReadFromDeviceDRAMChannel(device, channel, dram_byte_address, byte_size, dest_channel_data);
        pass &= (dest_channel_data == inputs);
        if (not pass) {
            INFO("Mismatch at Channel: " << channel);
        }
    }
    return pass;
}

// load_blank_kernels into all cores and ensure all cores hit mailbox values
bool load_all_blank_kernels(tt_metal::Device* device) {
    bool pass = true;
    CoreCoord grid_size = device->logical_grid_size();
    constexpr int INIT_VALUE = 42;
    constexpr int DONE_VALUE = 1;
    const std::unordered_map<string, uint32_t> mailbox_addresses = {
        {"BRISC", MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_BRISC_OFFSET},
        {"TRISC0", MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC0_OFFSET},
        {"TRISC1", MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC1_OFFSET},
        {"TRISC2", MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC2_OFFSET},
        {"NCRISC", MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_NCRISC_OFFSET},
    };
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    for(int y = 0 ; y < grid_size.y; y++) {
        for(int x = 0 ; x < grid_size.x; x++) {
            CoreCoord dest_core({.x=static_cast<size_t>(x), .y=static_cast<size_t>(y)});
            for (const auto& [name, address] : mailbox_addresses) {
                std::vector<uint32_t> mailbox_value;
                tt_metal::ReadFromDeviceL1(device, dest_core, address, 4, mailbox_value);
                pass &= (mailbox_value.at(0) == INIT_VALUE);
                if (not pass) {
                    INFO("Wrong INIT_VALUE at Core: " << dest_core.str() << " " << name);
                }
            }
        }
    }
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::LaunchKernels(device, program);
    return pass;
}

TEST_CASE(
    "dram_ping", "[basic][dram][ping]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    SECTION( "low_addr" ) {
        size_t start_byte_address = 0;
        SECTION ("small") {
            REQUIRE(dram_ping(device, 4, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 12, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 16, start_byte_address, device->num_dram_channels()));
        }
        SECTION ("tiles") {
            REQUIRE(dram_ping(device, 1024, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 2*1024, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 32*1024, start_byte_address, device->num_dram_channels()));
        }
    }
    SECTION( "high_addr" ) {
        size_t start_byte_address = device->dram_bank_size() - 32*1024;
        SECTION ("small") {
            REQUIRE(dram_ping(device, 4, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 12, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 16, start_byte_address, device->num_dram_channels()));
        }
        SECTION ("tiles") {
            REQUIRE(dram_ping(device, 1024, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 2*1024, start_byte_address, device->num_dram_channels()));
            REQUIRE(dram_ping(device, 32*1024, start_byte_address, device->num_dram_channels()));
        }
    }
    tt_metal::CloseDevice(device);
}
TEST_CASE(
    "Force DRAM Ping Outside Logical Grid Size", "[basic][dram][ping][error]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    size_t start_byte_address = UNRESERVED_BASE;
    auto num_channels = device->num_dram_channels();
    num_channels++;
    REQUIRE_THROWS(dram_ping(device, 4, start_byte_address, num_channels), Catch::Contains("Bounds-Error"));
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "l1_ping_logical", "[basic][l1][ping]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    SECTION( "low_addr" ) {
        size_t start_byte_address = UNRESERVED_BASE;
        SECTION ("small") {
            REQUIRE(l1_ping(device, 4, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 12, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 16, start_byte_address, device->logical_grid_size()));
        }
        SECTION ("tiles") {
            REQUIRE(l1_ping(device, 1024, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 2*1024, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 32*1024, start_byte_address, device->logical_grid_size()));
        }
    }
    SECTION( "high_addr" ) {
        size_t start_byte_address = device->l1_size() - 32*1024;
        SECTION ("small") {
            REQUIRE(l1_ping(device, 4, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 12, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 16, start_byte_address, device->logical_grid_size()));
        }
        SECTION ("tiles") {
            REQUIRE(l1_ping(device, 1024, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 2*1024, start_byte_address, device->logical_grid_size()));
            REQUIRE(l1_ping(device, 32*1024, start_byte_address, device->logical_grid_size()));
        }
    }
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "Force L1 Ping Outside Logical Grid Size", "[basic][l1][ping][error]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    size_t start_byte_address = UNRESERVED_BASE;
    auto grid_size = device->logical_grid_size();
    grid_size.x++;
    grid_size.y++;
    REQUIRE_THROWS_WITH(l1_ping(device, 4, start_byte_address, grid_size), Catch::Contains("Bounds-Error"));
    tt_metal::CloseDevice(device);
}

TEST_CASE(
    "Logical grid blank Kernels", "[basic][kernels]") {
    const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);
    load_all_blank_kernels(device);
    tt_metal::CloseDevice(device);
}
