// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <unistd.h>

#include "concurrent_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/executor.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

using namespace tt::tt_metal;
using namespace tt::test_utils;

namespace unit_tests::concurrent::device {

/// @brief Ping number of bytes for specified grid_size
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param grid_size - grid size. will ping all cores from {0,0} to grid_size (non-inclusive)
/// @return
void l1_ping(
    Device* device, const size_t& byte_size, const size_t& l1_byte_address, const CoreCoord& grid_size) {
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t), std::chrono::system_clock::now().time_since_epoch().count());
    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core({.x = static_cast<size_t>(x), .y = static_cast<size_t>(y)});
            detail::WriteToDeviceL1(device, dest_core, l1_byte_address, inputs);
        }
    }

    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core({.x = static_cast<size_t>(x), .y = static_cast<size_t>(y)});
            std::vector<uint32_t> dest_core_data;
            detail::ReadFromDeviceL1(device, dest_core, l1_byte_address, byte_size, dest_core_data);
            EXPECT_EQ(dest_core_data, inputs);
            bool pass = (dest_core_data == inputs);
            if (not pass) {
                tt::log_error("Mismatch at address {} in core {}", l1_byte_address, dest_core.str());
            }
        }
    }
}

vector<uint32_t> generate_constant_vector(uint32_t size_bytes, uint32_t val) {
    TT_ASSERT(size_bytes % sizeof(uint32_t) == 0);
    vector<uint32_t> src(size_bytes / sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = val;
    }
    return src;
}

/// @brief Ping number of bytes for specified channels
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param num_channels - num_channels. will ping all channels from {0} to num_channels (non-inclusive)
/// @return
void dram_ping(
    Device* device,
    const size_t& byte_size,
    const size_t& dram_byte_address,
    const unsigned int& num_channels) {
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t), std::chrono::system_clock::now().time_since_epoch().count());
    for (unsigned int channel = 0; channel < num_channels; channel++) {
        detail::WriteToDeviceDRAMChannel(device, channel, dram_byte_address, inputs);
    }

    for (unsigned int channel = 0; channel < num_channels; channel++) {
        std::vector<uint32_t> dest_channel_data;
        detail::ReadFromDeviceDRAMChannel(device, channel, dram_byte_address, byte_size, dest_channel_data);
        EXPECT_EQ(dest_channel_data, inputs);
        bool pass = (dest_channel_data == inputs);
        if (not pass) {
            tt::log_error("Mismatch at address {} in channel {}", dram_byte_address, channel);
        }
    }
}

/// @brief Ping number of bytes for sysmem
/// @param device
/// @param byte_size - data to write
/// @param byte_size - size in bytes
/// @param address - address in sysmem
/// @return
void sysmem_ping(Device* device, std::vector<uint32_t> &data, uint64_t address) {
    std::vector<uint32_t> data_readback;
    tt::Cluster::instance().write_sysmem_vec(data, address, device->id());
    tt::Cluster::instance().read_sysmem_vec(data_readback, address, data.size() * sizeof(uint32_t), device->id());
    EXPECT_EQ(data, data_readback);
}

}   // unit_tests::concurrent::device


TEST_F(ConcurrentFixture, TestOpenCloseSameDevice) {
    const chip_id_t device_id = 0;

    EXPECT_NO_THROW({
        Device *first_device = CreateDevice(device_id);
        CloseDevice(first_device);
    });

    EXPECT_NO_THROW({
        Device *second_device = CreateDevice(device_id);
        CloseDevice(second_device);
    });
}

TEST_F(ConcurrentFixture, TestMultiThreadAttemptSameDevice) {
    std::vector<std::future<void>> events;
    const chip_id_t device_id = 0;

    const int num_threads = 3;
    int num_exceptions = 0;
    for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
        events.emplace_back(
            detail::async (
                [&] {
                    try {
                        Device *device = CreateDevice(device_id);
                        sleep(1);
                        CloseDevice(device);
                    } catch (...) {
                        num_exceptions++;
                    }
                }
            )
        );
    }

    for (auto &f : events) {
        f.wait();
    }

    // expect num_threads - 1 exceptions because the same device cannot be opened multiple times in one process
    EXPECT_EQ(num_exceptions, num_threads - 1);
}

TEST_F(ConcurrentFixture, TestMultiProcessInitializeDevice) {
    pid_t child_a, child_b;
    child_a = fork();

    const chip_id_t device_id = 0;

    if (child_a == 0) {
        try {
            Device *device = CreateDevice(device_id);
            // sleep(2);
            CloseDevice(device);
        } catch (...) {
            exit(1);
        }
        exit(0);
    } else {
        child_b = fork();

        if (child_b == 0) {
            try {
                Device *device = CreateDevice(device_id);
                // sleep(1);
                CloseDevice(device);
            } catch (...) {
                exit(1);
            }
            exit(0);
        } else {
            // parent process
            EXPECT_NO_THROW({
                Device *device = CreateDevice(device_id);
                                CloseDevice(device);
            });

            int status_a;
            waitpid(child_a, &status_a, 0);

            int status_b;
            waitpid(child_b, &status_b, 0);

            if ( WIFEXITED(status_a) ) {
                const int exit_status = WEXITSTATUS(status_a);
                ASSERT_EQ(0, exit_status);
            }

            if ( WIFEXITED(status_b) ) {
                const int exit_status = WEXITSTATUS(status_b);
                ASSERT_EQ(0, exit_status);
            }
        }
    }
}

TEST_F(ConcurrentFixture, TestMultiThreadDeviceDramWriteRead) {
    const chip_id_t device_id = 0;

    Device *device = CreateDevice(device_id);
    uint32_t base_dram_address = DRAM_UNRESERVED_BASE;
    uint32_t size_bytes = 2048 * 10;

    std::vector<std::future<void>> events;

    const int num_threads = 3;
    for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
        events.emplace_back(
            detail::async (
                [&] {
                    uint32_t dram_address = base_dram_address + (size_bytes * thread_idx);
                    unit_tests::concurrent::device::dram_ping(device, size_bytes, dram_address, device->num_dram_channels());
                }
            )
        );
    }

    for (auto &f : events) {
        f.wait();
    }

    CloseDevice(device);
}

TEST_F(ConcurrentFixture, TestMultiProcessDeviceDramWriteRead) {
    pid_t child_a, child_b;
    child_a = fork();

    const chip_id_t device_id = 0;
    uint32_t base_address = DRAM_UNRESERVED_BASE;
    uint32_t size_bytes = 2048 * 10;

    if (child_a == 0) {
        Device *device = CreateDevice(device_id);
        uint32_t dram_address = base_address;
        unit_tests::concurrent::device::dram_ping(device, size_bytes, base_address, device->num_dram_channels());
        CloseDevice(device);
        exit(0);
    } else {
        child_b = fork();

        if (child_b == 0) {
            Device *device = CreateDevice(device_id);
            uint32_t dram_address = base_address + size_bytes;
            unit_tests::concurrent::device::dram_ping(device, size_bytes, dram_address, device->num_dram_channels());
            CloseDevice(device);
            exit(0);
        } else {
            // parent process
            Device *device = CreateDevice(device_id);
            uint32_t dram_address = base_address + (2 * size_bytes);
            unit_tests::concurrent::device::dram_ping(device, size_bytes, dram_address, device->num_dram_channels());
            CloseDevice(device);

            int status_a;
            waitpid(child_a, &status_a, 0);

            int status_b;
            waitpid(child_b, &status_b, 0);

            if ( WIFEXITED(status_a) ) {
                const int exit_status = WEXITSTATUS(status_a);
                ASSERT_EQ(0, exit_status);
            }

            if ( WIFEXITED(status_b) ) {
                const int exit_status = WEXITSTATUS(status_b);
                ASSERT_EQ(0, exit_status);
            }
        }
    }
}

TEST_F(ConcurrentFixture, TestMultiThreadDeviceL1WriteRead) {
    const chip_id_t device_id = 0;

    Device *device = CreateDevice(device_id);
    int num_channels = device->num_dram_channels();
    uint32_t base_address = L1_UNRESERVED_BASE;
    uint32_t size_bytes = 2048 * 10;

    std::vector<std::future<void>> events;

    const int num_threads = 3;
    for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
        events.emplace_back(
            detail::async (
                [&] {
                    uint32_t l1_address = base_address + (size_bytes * thread_idx);
                    unit_tests::concurrent::device::l1_ping(device, size_bytes, l1_address, device->logical_grid_size());
                }
            )
        );
    }

    for (auto &f : events) {
        f.wait();
    }

    CloseDevice(device);
}

TEST_F(ConcurrentFixture, TestMultiProcessDeviceL1WriteRead) {
    pid_t child_a, child_b;
    child_a = fork();

    const chip_id_t device_id = 0;
    uint32_t base_address = L1_UNRESERVED_BASE;
    uint32_t size_bytes = 2048 * 10;

    if (child_a == 0) {
        Device *device = CreateDevice(device_id);
        uint32_t l1_address = base_address;
        unit_tests::concurrent::device::l1_ping(device, size_bytes, base_address, device->logical_grid_size());
        CloseDevice(device);
        exit(0);
    } else {
        child_b = fork();

        if (child_b == 0) {
            Device *device = CreateDevice(device_id);
            uint32_t l1_address = base_address + size_bytes;
            unit_tests::concurrent::device::l1_ping(device, size_bytes, l1_address, device->logical_grid_size());
            CloseDevice(device);
            exit(0);
        } else {
            // parent process
            Device *device = CreateDevice(device_id);
            uint32_t l1_address = base_address + (2 * size_bytes);
            unit_tests::concurrent::device::l1_ping(device, size_bytes, l1_address, device->logical_grid_size());
            CloseDevice(device);

            int status_a;
            waitpid(child_a, &status_a, 0);

            int status_b;
            waitpid(child_b, &status_b, 0);

            if ( WIFEXITED(status_a) ) {
                const int exit_status = WEXITSTATUS(status_a);
                ASSERT_EQ(0, exit_status);
            }

            if ( WIFEXITED(status_b) ) {
                const int exit_status = WEXITSTATUS(status_b);
                ASSERT_EQ(0, exit_status);
            }
        }
    }
}

// Two processes targetting sysmem on the same device
// Child process writes to sysmem, parent process reads from the same address after child process is completed and should see the same value
TEST_F(ConcurrentFixture, TestSysmemSharedAcrossProcesses) {
    pid_t child_a = fork();
    pid_t wpid;

    const chip_id_t device_id = 0;
    uint32_t base_address = 0;
    uint32_t size_bytes = 2048 * 10;
    uint32_t value = 0xAB;
    std::vector<uint32_t> data = unit_tests::concurrent::device::generate_constant_vector(size_bytes, value);

    if (child_a == 0) {
        Device *device = CreateDevice(device_id);
        unit_tests::concurrent::device::sysmem_ping(device, data, base_address);
        CloseDevice(device);
        exit(0);
    } else {
        // parent process
        int status_a;
        waitpid(child_a, &status_a, 0);

        Device *device1 = CreateDevice(device_id);
        std::vector<uint32_t> data_readback;
        tt::Cluster::instance().read_sysmem_vec(data_readback, base_address, size_bytes, device1->id());
        EXPECT_EQ(data, data_readback);
        CloseDevice(device1);
    }

}

TEST_F(ConcurrentFixture, TestMultiThreadSysmemWriteRead) {
    const chip_id_t device_id = 0;

    Device *device = CreateDevice(device_id);

    uint32_t size_bytes = 2048 * 10;
    uint64_t base_address = 0;

    std::vector<std::future<void>> events;

    const int num_threads = 3;
    for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
        auto data = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, size_bytes / sizeof(uint32_t), std::chrono::system_clock::now().time_since_epoch().count());
        unit_tests::concurrent::device::sysmem_ping(device, data, base_address + (size_bytes * thread_idx));
    }

    for (auto &f : events) {
        f.wait();
    }

    CloseDevice(device);
}

TEST_F(ConcurrentFixture, TestMultiProcessSysmemWriteRead) {
    pid_t child_a, child_b;
    child_a = fork();

    const chip_id_t device_id = 0;
    uint32_t base_address = 0;
    uint32_t size_bytes = 2048 * 10;

    if (child_a == 0) {
        Device *device = CreateDevice(device_id);
        auto data1 = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, size_bytes / sizeof(uint32_t), std::chrono::system_clock::now().time_since_epoch().count());
        unit_tests::concurrent::device::sysmem_ping(device, data1, base_address);
        CloseDevice(device);
        exit(0);
    } else {
        child_b = fork();

        if (child_b == 0) {
            Device *device = CreateDevice(device_id);
            auto data2 = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, size_bytes / sizeof(uint32_t), std::chrono::system_clock::now().time_since_epoch().count());
            unit_tests::concurrent::device::sysmem_ping(device, data2, base_address + size_bytes);
            CloseDevice(device);
            exit(0);
        } else {
            // parent process
            Device *device = CreateDevice(device_id);
            auto data3 = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, size_bytes / sizeof(uint32_t), std::chrono::system_clock::now().time_since_epoch().count());
            unit_tests::concurrent::device::sysmem_ping(device, data3, base_address + (2 * size_bytes));
            CloseDevice(device);

            int status_a;
            waitpid(child_a, &status_a, 0);

            int status_b;
            waitpid(child_b, &status_b, 0);

            if ( WIFEXITED(status_a) ) {
                const int exit_status = WEXITSTATUS(status_a);
                ASSERT_EQ(0, exit_status);
            }

            if ( WIFEXITED(status_b) ) {
                const int exit_status = WEXITSTATUS(status_b);
                ASSERT_EQ(0, exit_status);
            }
        }
    }
}
