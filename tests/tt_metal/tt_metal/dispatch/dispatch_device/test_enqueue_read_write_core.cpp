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

TEST_F(CommandQueueSingleCardFixture, TensixTestBasicReadWriteL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core(virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TensixTestBasicReadL1) {
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
            .enqueue_read_from_core(virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TensixTestBasicWriteL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);

        Finish(device->command_queue());

        const std::vector<uint32_t> dst_data =
            tt::llrt::read_hex_vec_from_core(device->id(), virtual_core, address, num_elements * sizeof(uint32_t));

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, TensixTestInvalidReadWriteAddressL1) {
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
                .enqueue_write_to_core(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_read_from_core(virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(CommandQueueSingleCardFixture, TensixTestReadWriteMultipleCoresL1) {
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    for (IDevice* device : this->devices_) {
        for (uint32_t core_x = 0; core_x < device->compute_with_storage_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->compute_with_storage_grid_size().y; ++core_y) {
                const CoreCoord core = device->worker_core_from_logical_core({core_x, core_y});
                dynamic_cast<HWCommandQueue&>(device->command_queue())
                    .enqueue_write_to_core(core, src_data.data(), address, num_elements * sizeof(uint32_t), false);
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
                    .enqueue_read_from_core(
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

TEST_F(CommandQueueSingleCardFixture, TensixTestReadWriteZeroElementsL1) {
    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const std::vector<uint32_t> src_data = {};

    for (IDevice* device : this->devices_) {
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core(virtual_core, src_data.data(), address, 0, false);

        std::vector<uint32_t> dst_data;
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core(virtual_core, dst_data.data(), address, 0, false);

        Finish(device->command_queue());

        EXPECT_TRUE(dst_data.empty());
    }
}

TEST_F(CommandQueueSingleCardFixture, TensixTestReadWriteEntireL1) {
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
            .enqueue_write_to_core(virtual_core, src_data.data(), address, size, false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core(virtual_core, dst_data.data(), address, size, false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, ActiveEthTestReadWriteEntireL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t size =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = size / sizeof(uint32_t);
    const std::vector<uint32_t> src_data = generate_arange_vector(size);

    for (IDevice* device : this->devices_) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "No active ethernet cores found";
        }

        std::unordered_set<CoreCoord> active_ethernet_cores = device->get_active_ethernet_cores(true);
        const CoreCoord eth_core = *active_ethernet_cores.begin();
        const CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core(virtual_core, src_data.data(), address, size, false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core(virtual_core, dst_data.data(), address, size, false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, ActiveEthTestReadWriteMultipleCoresL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    for (IDevice* device : this->devices_) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "No active ethernet cores found";
        }

        std::unordered_set<CoreCoord> active_ethernet_cores = device->get_active_ethernet_cores(true);
        for (const CoreCoord& core : active_ethernet_cores) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_write_to_core(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            active_ethernet_cores.size(), std::vector<uint32_t>(num_elements, 0));
        uint32_t i = 0;
        for (const CoreCoord& core : active_ethernet_cores) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_read_from_core(
                    virtual_core, all_cores_dst_data[i].data(), address, num_elements * sizeof(uint32_t), false);
            i++;
        }

        Finish(device->command_queue());

        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(src_data, dst_data);
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, IdleEthTestReadWriteEntireL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t size =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = size / sizeof(uint32_t);
    const std::vector<uint32_t> src_data = generate_arange_vector(size);

    for (IDevice* device : this->devices_) {
        if (!does_device_have_idle_eth_cores(device)) {
            GTEST_SKIP() << "No idle ethernet cores found";
        }

        std::unordered_set<CoreCoord> idle_ethernet_cores = device->get_inactive_ethernet_cores();
        const CoreCoord eth_core = *idle_ethernet_cores.begin();
        const CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);

        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_write_to_core(virtual_core, src_data.data(), address, size, false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        dynamic_cast<HWCommandQueue&>(device->command_queue())
            .enqueue_read_from_core(virtual_core, dst_data.data(), address, size, false);

        Finish(device->command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(CommandQueueSingleCardFixture, IdleEthTestInvalidReadWriteAddressL1) {
    const uint32_t num_elements = 1010;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const DeviceAddr l1_end_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED) +
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::BASE);
    const DeviceAddr l1_end_address_offset = 256;
    const DeviceAddr address = l1_end_address + l1_end_address_offset;

    for (IDevice* device : this->devices_) {
        if (!does_device_have_idle_eth_cores(device)) {
            GTEST_SKIP() << "No idle ethernet cores found";
        }

        std::unordered_set<CoreCoord> idle_ethernet_cores = device->get_inactive_ethernet_cores();
        const CoreCoord eth_core = *idle_ethernet_cores.begin();
        const CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);

        EXPECT_THROW(
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_write_to_core(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_read_from_core(virtual_core, dst_data.data(), address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(CommandQueueSingleCardFixture, IdleEthTestReadWriteMultipleCoresL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    for (IDevice* device : this->devices_) {
        if (!does_device_have_idle_eth_cores(device)) {
            GTEST_SKIP() << "No idle ethernet cores found";
        }

        std::unordered_set<CoreCoord> idle_ethernet_cores = device->get_inactive_ethernet_cores();
        dispatch_core_manager& dispatch_core_manager = MetalContext::instance().get_dispatch_core_manager();
        const CoreType dispatch_core_type = dispatch_core_manager.get_dispatch_core_type();
        if (dispatch_core_type == CoreType::ETH) {
            const std::vector<CoreCoord> eth_dispatch_cores =
                dispatch_core_manager.get_all_logical_dispatch_cores(device->id());
            for (const CoreCoord& core : eth_dispatch_cores) {
                idle_ethernet_cores.erase(core);
            }
        }

        for (const CoreCoord& core : idle_ethernet_cores) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_write_to_core(virtual_core, src_data.data(), address, num_elements * sizeof(uint32_t), false);
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            idle_ethernet_cores.size(), std::vector<uint32_t>(num_elements, 0));
        uint32_t i = 0;
        for (const CoreCoord& core : idle_ethernet_cores) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_read_from_core(
                    virtual_core, all_cores_dst_data[i].data(), address, num_elements * sizeof(uint32_t), false);
            i++;
        }

        Finish(device->command_queue());

        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(src_data, dst_data);
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, TestInvalidReadWriteAddressDRAM) {
    for (IDevice* device : this->devices_) {
        const uint32_t num_elements = 1010;
        const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

        const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
        const uint32_t size = MetalContext::instance().hal().get_dev_size(HalDramMemAddrType::UNRESERVED);
        const uint32_t dram_end_address = address + size;
        const DeviceAddr dram_end_address_offset = 256;
        const DeviceAddr dram_invalid_address = dram_end_address + dram_end_address_offset;

        const CoreCoord logical_core = {0, 0};
        const CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, CoreType::DRAM);

        EXPECT_THROW(
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_write_to_core(
                    virtual_core, src_data.data(), dram_invalid_address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            dynamic_cast<HWCommandQueue&>(device->command_queue())
                .enqueue_read_from_core(
                    virtual_core, dst_data.data(), dram_invalid_address, num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestReadWriteMultipleCoresDRAM) {
    for (IDevice* device : this->devices_) {
        const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
        const uint32_t num_elements = 1000;

        uint32_t dram_core_value = 1;
        for (uint32_t core_x = 0; core_x < device->dram_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->dram_grid_size().y; ++core_y) {
                const CoreCoord core = device->virtual_core_from_logical_core({core_x, core_y}, CoreType::DRAM);
                const std::vector<uint32_t> src_data(num_elements, dram_core_value);
                dynamic_cast<HWCommandQueue&>(device->command_queue())
                    .enqueue_write_to_core(core, src_data.data(), address, num_elements * sizeof(uint32_t), false);
                dram_core_value++;
            }
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            device->dram_grid_size().x * device->dram_grid_size().y, std::vector<uint32_t>(num_elements, 0));
        uint32_t j = 0;
        for (uint32_t core_x = 0; core_x < device->dram_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->dram_grid_size().y; ++core_y) {
                const CoreCoord core = device->virtual_core_from_logical_core({core_x, core_y}, CoreType::DRAM);
                dynamic_cast<HWCommandQueue&>(device->command_queue())
                    .enqueue_read_from_core(
                        core, all_cores_dst_data[j].data(), address, num_elements * sizeof(uint32_t), false);
                j++;
            }
        }

        Finish(device->command_queue());

        uint32_t expected_dram_core_value = 1;
        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(dst_data, std::vector<uint32_t>(num_elements, expected_dram_core_value));
            expected_dram_core_value++;
        }
    }
}
