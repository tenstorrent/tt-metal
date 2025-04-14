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
    const std::vector<uint32_t>& src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address =
        hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        device->command_queue().enqueue_write_to_core_l1(
            virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        device->command_queue().enqueue_read_from_core_l1(
            virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestBasicReadL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t>& src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address =
        hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        tt::llrt::write_hex_vec_to_core(device->id(), virtual_core, src_data, address);

        std::vector<uint32_t> dst_data(num_elements, 0);
        device->command_queue().enqueue_read_from_core_l1(
            virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestBasicWriteL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t>& src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address =
        hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        device->command_queue().enqueue_write_to_core_l1(
            virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        const std::vector<uint32_t>& dst_data =
            tt::llrt::read_hex_vec_from_core(device->id(), virtual_core, address, num_elements * sizeof(uint32_t));

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestInvalidReadWriteAddressL1) {
    const uint32_t num_elements = 1010;
    const std::vector<uint32_t>& src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord core = {0, 0};
    const DeviceAddr address = hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                               hal_ref.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) + 256;

    for (IDevice* device : this->devices_) {
        EXPECT_THROW(
            device->command_queue().enqueue_write_to_core_l1(
                core, src_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            device->command_queue().enqueue_read_from_core_l1(
                core, dst_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestReadWriteMultipleTensixCoresL1) {
    const DeviceAddr address =
        hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t num_bytes =
        hal_ref.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t num_elements = num_bytes / sizeof(uint32_t);
    const std::vector<uint32_t>& src_data = generate_arange_vector(num_elements * sizeof(uint32_t), 1);

    for (IDevice* device : this->devices_) {
        tt::log_info("l1_size_per_core: {}", device->l1_size_per_core());
        for (uint32_t core_x = 0; core_x < device->compute_with_storage_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->compute_with_storage_grid_size().y; ++core_y) {
                tt::log_info("Writing to core: {}, {}", core_x, core_y);
                const CoreCoord core = device->worker_core_from_logical_core({core_x, core_y});
                device->command_queue().enqueue_write_to_core_l1(
                    core, src_data.data(), address, num_elements * sizeof(uint32_t), false);
            }
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
            std::vector<uint32_t>(num_elements, 0));
        tt::log_info("all_cores_dst_data size: {}", all_cores_dst_data.size());
        uint32_t i = 0;
        for (uint32_t core_x = 0; core_x < device->compute_with_storage_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->compute_with_storage_grid_size().y; ++core_y) {
                tt::log_info("Reading from core: {}, {}", core_x, core_y);
                const CoreCoord core = device->worker_core_from_logical_core({core_x, core_y});
                device->command_queue().enqueue_read_from_core_l1(
                    core, all_cores_dst_data[i].data(), address, num_elements * sizeof(uint32_t), false);
                i++;
            }
        }

        Finish(device->command_queue());

        i = 0;
        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            tt::log_info("iteration: {}", i);
            EXPECT_EQ(src_data, dst_data);
            i++;
        }
    }
}

// TEST_F(L1ReadWriteTest, MultipleSubDevices) {
//     constexpr uint32_t size = 512;
//     std::vector<uint32_t> src_data = generate_arange_vector(size * sizeof(uint32_t));
//     std::vector<uint32_t> dst_data(size, 0);

//     CoreCoord core = {0, 0};
//     DeviceAddr address = 0x2000;
//     std::vector<SubDeviceId> sub_device_ids = {0, 1};

//     // Write to multiple sub-devices
//     command_queue_->enqueue_write_to_core_l1(
//         core,
//         src_data.data(),
//         address,
//         size * sizeof(uint32_t),
//         true,
//         sub_device_ids
//     );

//     // Read from multiple sub-devices
//     command_queue_->enqueue_read_from_core_l1(
//         core,
//         dst_data.data(),
//         address,
//         size * sizeof(uint32_t),
//         true,
//         sub_device_ids
//     );

//     EXPECT_EQ(src_data, dst_data);
// }

// TEST_F(L1ReadWriteTest, NonBlockingOperations) {
//     constexpr uint32_t size = 256;
//     std::vector<uint32_t> src_data = generate_arange_vector(size * sizeof(uint32_t));
//     std::vector<uint32_t> dst_data(size, 0);

//     CoreCoord core = {0, 0};
//     DeviceAddr address = 0x3000;

//     // Non-blocking write
//     command_queue_->enqueue_write_to_core_l1(
//         core,
//         src_data.data(),
//         address,
//         size * sizeof(uint32_t),
//         false
//     );

//     // Non-blocking read
//     command_queue_->enqueue_read_from_core_l1(
//         core,
//         dst_data.data(),
//         address,
//         size * sizeof(uint32_t),
//         false
//     );

//     // Need to wait for operations to complete
//     command_queue_->finish();

//     EXPECT_EQ(src_data, dst_data);
// }

// TEST_F(L1ReadWriteTest, InvalidAddress) {
//     constexpr uint32_t size = 128;
//     std::vector<uint32_t> data = generate_arange_vector(size * sizeof(uint32_t));

//     CoreCoord core = {0, 0};
//     DeviceAddr invalid_address = 0xFFFFFFFF; // Invalid address

//     // Should throw when trying to write to invalid address
//     EXPECT_THROW(
//         command_queue_->enqueue_write_to_core_l1(
//             core,
//             data.data(),
//             invalid_address,
//             size * sizeof(uint32_t),
//             true
//         ),
//         std::runtime_error
//     );
// }

// TEST_F(L1ReadWriteTest, ZeroSize) {
//     std::vector<uint32_t> data = generate_arange_vector(sizeof(uint32_t));

//     CoreCoord core = {0, 0};
//     DeviceAddr address = 0x4000;

//     // Should handle zero size gracefully
//     command_queue_->enqueue_write_to_core_l1(
//         core,
//         data.data(),
//         address,
//         0,
//         true
//     );

//     command_queue_->enqueue_read_from_core_l1(
//         core,
//         data.data(),
//         address,
//         0,
//         true
//     );

//     // No data corruption should occur
//     EXPECT_EQ(data[0], 0);
// }
