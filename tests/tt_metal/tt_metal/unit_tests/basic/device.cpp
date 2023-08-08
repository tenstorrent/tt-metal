#include <algorithm>
#include <functional>
#include <random>

#include "doctest.h"
#include "multi_device_fixture.hpp"
#include "single_device_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::device {

/// @brief Ping number of bytes for specified grid_size
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param grid_size - grid size. will ping all cores from {0,0} to grid_size (non-inclusive)
/// @return
bool l1_ping(
    tt_metal::Device* device, const size_t& byte_size, const size_t& l1_byte_address, const CoreCoord& grid_size) {
    bool pass = true;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core({.x = static_cast<size_t>(x), .y = static_cast<size_t>(y)});
            tt_metal::detail::WriteToDeviceL1(device, dest_core, l1_byte_address, inputs);
        }
    }
    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core({.x = static_cast<size_t>(x), .y = static_cast<size_t>(y)});
            std::vector<uint32_t> dest_core_data;
            tt_metal::detail::ReadFromDeviceL1(device, dest_core, l1_byte_address, byte_size, dest_core_data);
            pass &= (dest_core_data == inputs);
            if (not pass) {
                MESSAGE("Mismatch at Core: ", dest_core.str());
            }
        }
    }
    return pass;
}

/// @brief Ping number of bytes for specified channels
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param num_channels - num_channels. will ping all channels from {0} to num_channels (non-inclusive)
/// @return
bool dram_ping(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& dram_byte_address,
    const unsigned int& num_channels) {
    bool pass = true;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    for (unsigned int channel = 0; channel < num_channels; channel++) {
        tt_metal::detail::WriteToDeviceDRAMChannel(device, channel, dram_byte_address, inputs);
    }
    for (unsigned int channel = 0; channel < num_channels; channel++) {
        std::vector<uint32_t> dest_channel_data;
        tt_metal::detail::ReadFromDeviceDRAMChannel(device, channel, dram_byte_address, byte_size, dest_channel_data);
        pass &= (dest_channel_data == inputs);
        if (not pass) {
            cout << "Mismatch at Channel: " << channel << std::endl;
        }
    }
    return pass;
}

/// @brief load_blank_kernels into all cores and will launch
/// @param device
/// @return
bool load_all_blank_kernels(tt_metal::Device* device) {
    bool pass = true;
    tt_metal::Program program = tt_metal::Program();
    pass &= tt_metal::CompileProgram(device, program);
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::LaunchKernels(device, program);
    return pass;
}
}  // namespace unit_tests::basic

bool is_multi_device_gs_machine() {
    tt::ARCH arch = tt::get_arch_from_string(get_env_arch_name());
    const size_t num_devices = tt::tt_metal::Device::detect_num_available_devices();

    return arch == tt::ARCH::GRAYSKULL && num_devices > 1;
}

TEST_SUITE(
    "BasicMultiDeviceTest" *
    doctest::description("Basic device tests should just test simple APIs and shouldn't take more than 1s per chip, "
                         "but can scale beyond for many devices.") *
    doctest::timeout(10) *
    doctest::skip(is_multi_device_gs_machine())
    ) {
    TEST_CASE("Multi Device Initialize and Teardown" * doctest::timeout(2)) {
        auto arch = tt::get_arch_from_string(get_env_arch_name());
        const size_t num_devices = tt::tt_metal::Device::detect_num_available_devices();
        REQUIRE(num_devices > 0);
        std::vector<tt::tt_metal::Device*> devices;

        if (arch != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }
        for (unsigned int id = 0; id < num_devices; id++) {
            devices.push_back(tt::tt_metal::CreateDevice(arch, id));
            REQUIRE(tt::tt_metal::InitializeDevice(devices.at(id)));
        }
        for (unsigned int id = 0; id < num_devices; id++) {
            REQUIRE(tt::tt_metal::CloseDevice(devices.at(id)));
        }
    }
    TEST_CASE("Multi Device Load Blank Kernels" * doctest::timeout(5)) {
        auto arch = tt::get_arch_from_string(get_env_arch_name());
        const size_t num_devices = tt::tt_metal::Device::detect_num_available_devices();
        REQUIRE(num_devices > 0);
        std::vector<tt::tt_metal::Device*> devices;

        if (arch != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }

        for (unsigned int id = 0; id < num_devices; id++) {
            devices.push_back(tt::tt_metal::CreateDevice(arch, id));
            REQUIRE(tt::tt_metal::InitializeDevice(devices.at(id)));
        }
        for (unsigned int id = 0; id < num_devices; id++) {
            unit_tests::basic::device::load_all_blank_kernels(devices.at(id));
        }
        for (unsigned int id = 0; id < num_devices; id++) {
            REQUIRE(tt::tt_metal::CloseDevice(devices.at(id)));
        }
    }
    TEST_CASE_FIXTURE(unit_tests::MultiDeviceFixture, "Ping all legal dram channels") {
        for (unsigned int id = 0; id < num_devices_; id++) {
            auto device_ = devices_.at(id);
            SUBCASE("Low Address Dram") {
                size_t start_byte_address = 0;
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 4, start_byte_address, device_->num_dram_channels()));
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 12, start_byte_address, device_->num_dram_channels()));
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 16, start_byte_address, device_->num_dram_channels()));
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 1024, start_byte_address, device_->num_dram_channels()));
                REQUIRE(
                    unit_tests::basic::device::dram_ping(device_, 2 * 1024, start_byte_address, device_->num_dram_channels()));
                REQUIRE(
                    unit_tests::basic::device::dram_ping(device_, 32 * 1024, start_byte_address, device_->num_dram_channels()));
            }
            SUBCASE("High Address Dram") {
                size_t start_byte_address = device_->dram_bank_size() - 32 * 1024;
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 4, start_byte_address, device_->num_dram_channels()));
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 12, start_byte_address, device_->num_dram_channels()));
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 16, start_byte_address, device_->num_dram_channels()));
                REQUIRE(unit_tests::basic::device::dram_ping(device_, 1024, start_byte_address, device_->num_dram_channels()));
                REQUIRE(
                    unit_tests::basic::device::dram_ping(device_, 2 * 1024, start_byte_address, device_->num_dram_channels()));
                REQUIRE(
                    unit_tests::basic::device::dram_ping(device_, 32 * 1024, start_byte_address, device_->num_dram_channels()));
            }
        }
    }
    TEST_CASE_FIXTURE(unit_tests::MultiDeviceFixture, "Ping all legal dram channels + illegal channel") {
        for (unsigned int id = 0; id < num_devices_; id++) {
            auto device_ = devices_.at(id);
            auto num_channels = device_->num_dram_channels() + 1;
            size_t start_byte_address = 0;
            REQUIRE_THROWS_WITH(
                unit_tests::basic::device::dram_ping(device_, 4, start_byte_address, num_channels),
                doctest::Contains("Bounds-Error"));
        }
    }

    TEST_CASE_FIXTURE(unit_tests::MultiDeviceFixture, "Ping all legal l1 cores") {
        for (unsigned int id = 0; id < num_devices_; id++) {
            auto device_ = devices_.at(id);
            SUBCASE("Low Address L1") {
                size_t start_byte_address = UNRESERVED_BASE;  // FIXME: Should remove dependency on
                                                              // hostdevcommon/common_runtime_address_map.h header.
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 4, start_byte_address, device_->logical_grid_size()));
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 12, start_byte_address, device_->logical_grid_size()));
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 16, start_byte_address, device_->logical_grid_size()));
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 1024, start_byte_address, device_->logical_grid_size()));
                REQUIRE(
                    unit_tests::basic::device::l1_ping(device_, 2 * 1024, start_byte_address, device_->logical_grid_size()));
                REQUIRE(
                    unit_tests::basic::device::l1_ping(device_, 32 * 1024, start_byte_address, device_->logical_grid_size()));
            }
            SUBCASE("High Address L1") {
                size_t start_byte_address = device_->l1_size() - 32 * 1024;
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 4, start_byte_address, device_->logical_grid_size()));
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 12, start_byte_address, device_->logical_grid_size()));
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 16, start_byte_address, device_->logical_grid_size()));
                REQUIRE(unit_tests::basic::device::l1_ping(device_, 1024, start_byte_address, device_->logical_grid_size()));
                REQUIRE(
                    unit_tests::basic::device::l1_ping(device_, 2 * 1024, start_byte_address, device_->logical_grid_size()));
                REQUIRE(
                    unit_tests::basic::device::l1_ping(device_, 32 * 1024, start_byte_address, device_->logical_grid_size()));
            }
        }
    }

    TEST_CASE_FIXTURE(unit_tests::MultiDeviceFixture, "Ping all legal l1 + illegal cores") {
        for (unsigned int id = 0; id < num_devices_; id++) {
            auto device_ = devices_.at(id);
            auto grid_size = device_->logical_grid_size();
            grid_size.x++;
            grid_size.y++;
            size_t start_byte_address = UNRESERVED_BASE;  // FIXME: Should remove dependency on
                                                          // hostdevcommon/common_runtime_address_map.h header.
            REQUIRE_THROWS_WITH(
                unit_tests::basic::device::l1_ping(device_, 4, start_byte_address, grid_size),
                doctest::Contains("Bounds-Error"));
        }
    }
}

TEST_SUITE(
    "BasicSingleDeviceTest" *
    doctest::description("Basic device tests should just test simple APIs and shouldn't take more than 5s") *
    doctest::timeout(5)) {
    TEST_CASE("Single Device Initialize and Teardown") {
        auto arch = tt::get_arch_from_string(get_env_arch_name());
        tt::tt_metal::Device* device;
        const unsigned int pcie_id = 0;
        device = tt::tt_metal::CreateDevice(arch, pcie_id);
        if (arch != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }
        REQUIRE(tt::tt_metal::InitializeDevice(device));
        REQUIRE(tt::tt_metal::CloseDevice(device));
    }
    TEST_CASE("HarvestingPrints") {
        auto arch = tt::get_arch_from_string(get_env_arch_name());
        tt::tt_metal::Device* device;
        const unsigned int pcie_id = 0;
        device = tt::tt_metal::CreateDevice(arch, pcie_id);
        if (arch != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }
        REQUIRE(tt::tt_metal::InitializeDevice(device));
        CoreCoord unharvested_logical_grid_size = {.x=12, .y = 10};
        if (arch == tt::ARCH::WORMHOLE_B0) {
            unharvested_logical_grid_size = {.x=8, .y=10};
        }
        auto logical_grid_size = device->logical_grid_size();
        if (logical_grid_size == unharvested_logical_grid_size) {
            tt::log_info("Harvesting Disabled in SW");
        } else {
            tt::log_info("Harvesting Enabled in SW");
            tt::log_info("Number of Harvested Rows={}", unharvested_logical_grid_size.y - logical_grid_size.y);
        }

        tt::log_info("Logical -- Noc Coordinates Mapping");
        tt::log_info("[Logical <-> NOC0] Coordinates");
        for (int r = 0; r < logical_grid_size.y; r ++) {
            string output_row = "";
            for (int c = 0; c < logical_grid_size.x; c ++) {
                const CoreCoord logical_coord(c, r);
                const auto noc_coord = device->worker_core_from_logical_core(logical_coord);
                output_row += "{L[x" + std::to_string(c);
                output_row += "-y" + std::to_string(r);
                output_row += "]:N[x" + std::to_string(noc_coord.x);
                output_row += "-y" + std::to_string(noc_coord.y);
                output_row += "]}, ";
            }
            tt::log_info("{}", output_row);
        }
        REQUIRE(tt::tt_metal::CloseDevice(device));
    }

    TEST_CASE("Single Device Load Blank Kernels") {
        auto arch = tt::get_arch_from_string(get_env_arch_name());
        tt::tt_metal::Device* device;
        const unsigned int pcie_id = 0;
        device = tt::tt_metal::CreateDevice(arch, pcie_id);
        if (arch != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }
        REQUIRE(tt::tt_metal::InitializeDevice(device));
        REQUIRE(tt::tt_metal::CloseDevice(device));
    }
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "Ping all legal dram channels") {
        SUBCASE("Low Address Dram") {
            size_t start_byte_address = 0;
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 4, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 12, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 16, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 1024, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 2 * 1024, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 32 * 1024, start_byte_address, device_->num_dram_channels()));
        }
        SUBCASE("High Address Dram") {
            size_t start_byte_address = device_->dram_bank_size() - 32 * 1024;
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 4, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 12, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 16, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 1024, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 2 * 1024, start_byte_address, device_->num_dram_channels()));
            REQUIRE(unit_tests::basic::device::dram_ping(device_, 32 * 1024, start_byte_address, device_->num_dram_channels()));
        }
    }
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "Ping all legal dram channels + illegal channel") {
        auto num_channels = device_->num_dram_channels() + 1;
        size_t start_byte_address = 0;
        REQUIRE_THROWS_WITH(
            unit_tests::basic::device::dram_ping(device_, 4, start_byte_address, num_channels),
            doctest::Contains("Bounds-Error"));
    }

    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "Ping all legal l1 cores") {
        SUBCASE("Low Address L1") {
            size_t start_byte_address = UNRESERVED_BASE;  // FIXME: Should remove dependency on
                                                          // hostdevcommon/common_runtime_address_map.h header.
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 4, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 12, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 16, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 1024, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 2 * 1024, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 32 * 1024, start_byte_address, device_->logical_grid_size()));
        }
        SUBCASE("High Address L1") {
            size_t start_byte_address = device_->l1_size() - 32 * 1024;
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 4, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 12, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 16, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 1024, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 2 * 1024, start_byte_address, device_->logical_grid_size()));
            REQUIRE(unit_tests::basic::device::l1_ping(device_, 32 * 1024, start_byte_address, device_->logical_grid_size()));
        }
    }

    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "Ping all legal l1 + illegal cores") {
        auto grid_size = device_->logical_grid_size();
        grid_size.x++;
        grid_size.y++;
        size_t start_byte_address =
            UNRESERVED_BASE;  // FIXME: Should remove dependency on hostdevcommon/common_runtime_address_map.h header.
        REQUIRE_THROWS_WITH(
            unit_tests::basic::device::l1_ping(device_, 4, start_byte_address, grid_size), doctest::Contains("Bounds-Error"));
    }
}
