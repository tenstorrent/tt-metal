// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "command_queue_fixture.hpp"
#include "hal_types.hpp"
#include "llrt.hpp"
#include "tt_metal/impl/dispatch/hardware_command_queue.hpp"
#include "dispatch_test_utils.hpp"

using namespace tt::tt_metal;

TEST_F(CommandQueueSingleCardFixture, TestBasicReadWriteL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core_l1(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core_l1(virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestBasicReadL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        tt::llrt::write_hex_vec_to_core(device->id(), virtual_core, src_data, address);

        std::vector<uint32_t> dst_data(num_elements, 0);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core_l1(virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestBasicWriteL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core_l1(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        const std::vector<uint32_t> dst_data =
            tt::llrt::read_hex_vec_from_core(device->id(), virtual_core, address, num_elements * sizeof(uint32_t));

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestInvalidReadWriteAddressL1) {
    const uint32_t num_elements = 1010;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr l1_end_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    const DeviceAddr l1_end_address_offset = 256;
    const DeviceAddr address = l1_end_address + l1_end_address_offset;

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        EXPECT_THROW(
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_write_to_core_l1(
                    virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_read_from_core_l1(
                    virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestReadWriteMultipleTensixCoresL1) {
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    for (IDevice* device : this->devices_) {
        for (uint32_t core_x = 0; core_x < device->compute_with_storage_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->compute_with_storage_grid_size().y; ++core_y) {
                const CoreCoord core = device->worker_core_from_logical_core({core_x, core_y});
                dynamic_cast<HWCommandQueue&>(device->command_queue())
                    .enqueue_write_to_core_l1(core, src_data.data(), address, num_elements * sizeof(uint32_t), false);
            }
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
            std::vector<uint32_t>(num_elements, 0));
        uint32_t i = 0;
        for (uint32_t core_x = 0; core_x < device->compute_with_storage_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->compute_with_storage_grid_size().y; ++core_y) {
                const CoreCoord core = device->worker_core_from_logical_core({core_x, core_y});
                dynamic_cast<HWCommandQueue&>(device->command_queue())
                    .enqueue_read_from_core_l1(
                        core, all_cores_dst_data[i].data(), address, num_elements * sizeof(uint32_t), false);
                i++;
            }
        }

        Finish(device->command_queue());

        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(src_data, dst_data);
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, TestReadWriteZeroElementsL1) {
    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const std::vector<uint32_t> src_data = {};

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core_l1(virtual_core, src_data.data(), address, 0, false);

        std::vector<uint32_t> dst_data;
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core_l1(virtual_core, dst_data.data(), address, 0, false);

        Finish(device->command_queue());

        EXPECT_TRUE(dst_data.empty());
    }
}

TEST_F(CommandQueueSingleCardFixture, TestReadWriteEntireL1) {
    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t size = MetalContext::instance().hal().get_dev_size(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t num_elements = size / sizeof(uint32_t);
    const std::vector<uint32_t> src_data = generate_arange_vector(size);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core_l1(virtual_core, src_data.data(), address, size, false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core_l1(virtual_core, dst_data.data(), address, size, false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}
