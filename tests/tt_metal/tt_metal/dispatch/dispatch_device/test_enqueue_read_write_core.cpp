// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "command_queue_fixture.hpp"
#include "hal_types.hpp"
#include "llrt.hpp"
#include "tt_metal/impl/dispatch/hardware_command_queue.hpp"
#include "dispatch_test_utils.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt::tt_metal;

TEST_F(UnitMeshCQSingleCardFixture, TensixTestBasicReadWriteL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (const auto& mesh_device : this->devices_) {
        auto device = mesh_device->get_devices()[0];
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        const CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(logical_core);

        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        fd_cq.enqueue_write_shard_to_core(
            device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false);

        std::vector<uint32_t> dst_data(num_elements, 0);
        fd_cq.enqueue_read_shard_from_core(
            device_memory_address, dst_data.data(), num_elements * sizeof(uint32_t), false);

        distributed::Finish(fd_cq);

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestBasicReadL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(device->id(), virtual_core, src_data, address);
        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};

        std::vector<uint32_t> dst_data(num_elements, 0);
        fd_cq.enqueue_read_shard_from_core(
            device_memory_address, dst_data.data(), num_elements * sizeof(uint32_t), false);

        distributed::Finish(mesh_device->mesh_command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestBasicWriteL1) {
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        fd_cq.enqueue_write_shard_to_core(
            device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false);

        distributed::Finish(mesh_device->mesh_command_queue());

        const std::vector<uint32_t> dst_data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            device->id(), virtual_core, address, num_elements * sizeof(uint32_t));

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestInvalidReadWriteAddressL1) {
    const uint32_t num_elements = 1010;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const CoreCoord logical_core = {0, 0};
    const DeviceAddr l1_end_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    const DeviceAddr l1_end_address_offset = 256;
    const DeviceAddr address = l1_end_address + l1_end_address_offset;

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        EXPECT_THROW(
            fd_cq.enqueue_write_shard_to_core(
                device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            fd_cq.enqueue_read_shard_from_core(
                device_memory_address, dst_data.data(), num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestReadWriteMultipleCoresL1) {
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        for (uint32_t core_x = 0; core_x < device->compute_with_storage_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->compute_with_storage_grid_size().y; ++core_y) {
                const CoreCoord core = device->worker_core_from_logical_core({core_x, core_y});
                const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
                const distributed::DeviceMemoryAddress device_memory_address = {
                    device_coord, core, reinterpret_cast<DeviceAddr>(address)};
                fd_cq.enqueue_write_shard_to_core(
                    device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), true);
            }
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
            std::vector<uint32_t>(num_elements, 0));
        uint32_t i = 0;
        for (uint32_t core_x = 0; core_x < device->compute_with_storage_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->compute_with_storage_grid_size().y; ++core_y) {
                const CoreCoord core = device->worker_core_from_logical_core({core_x, core_y});
                const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
                const distributed::DeviceMemoryAddress device_memory_address = {
                    device_coord, core, reinterpret_cast<DeviceAddr>(address)};
                fd_cq.enqueue_read_shard_from_core(
                    device_memory_address, all_cores_dst_data[i].data(), num_elements * sizeof(uint32_t), true);
                i++;
            }
        }

        distributed::Finish(mesh_device->mesh_command_queue());

        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(src_data, dst_data);
        }
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestReadWriteZeroElementsL1) {
    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const std::vector<uint32_t> src_data = {};

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        fd_cq.enqueue_write_shard_to_core(device_memory_address, src_data.data(), 0, false);

        std::vector<uint32_t> dst_data;
        fd_cq.enqueue_read_shard_from_core(device_memory_address, dst_data.data(), 0, false);

        distributed::Finish(mesh_device->mesh_command_queue());

        EXPECT_TRUE(dst_data.empty());
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestReadWriteEntireL1) {
    const CoreCoord logical_core = {0, 0};
    const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t size = MetalContext::instance().hal().get_dev_size(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t num_elements = size / sizeof(uint32_t);
    const std::vector<uint32_t> src_data = generate_arange_vector(size);

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        fd_cq.enqueue_write_shard_to_core(device_memory_address, src_data.data(), size, true);

        std::vector<uint32_t> dst_data(num_elements, 0);
        fd_cq.enqueue_read_shard_from_core(device_memory_address, dst_data.data(), size, true);

        distributed::Finish(mesh_device->mesh_command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, ActiveEthTestReadWriteEntireL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t size =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = size / sizeof(uint32_t);
    const std::vector<uint32_t> src_data = generate_arange_vector(size);

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "No active ethernet cores found";
        }

        std::unordered_set<CoreCoord> active_ethernet_cores = device->get_active_ethernet_cores(true);
        const CoreCoord eth_core = *active_ethernet_cores.begin();
        const CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);
        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        fd_cq.enqueue_write_shard_to_core(device_memory_address, src_data.data(), size, true);

        std::vector<uint32_t> dst_data(num_elements, 0);
        fd_cq.enqueue_read_shard_from_core(device_memory_address, dst_data.data(), size, true);

        distributed::Finish(mesh_device->mesh_command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, ActiveEthTestReadWriteMultipleCoresL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "No active ethernet cores found";
        }

        std::unordered_set<CoreCoord> active_ethernet_cores = device->get_active_ethernet_cores(true);
        for (const CoreCoord& core : active_ethernet_cores) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
            const distributed::DeviceMemoryAddress device_memory_address = {
                device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
            fd_cq.enqueue_write_shard_to_core(
                device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false);
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            active_ethernet_cores.size(), std::vector<uint32_t>(num_elements, 0));
        uint32_t i = 0;
        for (const CoreCoord& core : active_ethernet_cores) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
            const distributed::DeviceMemoryAddress device_memory_address = {
                device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
            fd_cq.enqueue_read_shard_from_core(
                device_memory_address, all_cores_dst_data[i].data(), num_elements * sizeof(uint32_t), false);
            i++;
        }

        distributed::Finish(mesh_device->mesh_command_queue());

        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(src_data, dst_data);
        }
    }
}

TEST_F(UnitMeshCQSingleCardFixture, IdleEthTestReadWriteEntireL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t size =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = size / sizeof(uint32_t);
    const std::vector<uint32_t> src_data = generate_arange_vector(size);

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        if (!does_device_have_idle_eth_cores(device)) {
            GTEST_SKIP() << "No idle ethernet cores found";
        }

        std::unordered_set<CoreCoord> idle_ethernet_cores = device->get_inactive_ethernet_cores();
        const CoreCoord eth_core = *idle_ethernet_cores.begin();
        const CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);

        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        fd_cq.enqueue_write_shard_to_core(device_memory_address, src_data.data(), size, true);

        std::vector<uint32_t> dst_data(num_elements, 0);
        fd_cq.enqueue_read_shard_from_core(device_memory_address, dst_data.data(), size, true);

        distributed::Finish(mesh_device->mesh_command_queue());

        EXPECT_EQ(src_data, dst_data);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, IdleEthTestInvalidReadWriteAddressL1) {
    const uint32_t num_elements = 1010;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    const DeviceAddr l1_end_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED) +
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::BASE);
    const DeviceAddr l1_end_address_offset = 256;
    const DeviceAddr address = l1_end_address + l1_end_address_offset;

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        if (!does_device_have_idle_eth_cores(device)) {
            GTEST_SKIP() << "No idle ethernet cores found";
        }

        std::unordered_set<CoreCoord> idle_ethernet_cores = device->get_inactive_ethernet_cores();
        const CoreCoord eth_core = *idle_ethernet_cores.begin();
        const CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);

        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
        EXPECT_THROW(
            fd_cq.enqueue_write_shard_to_core(
                device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            fd_cq.enqueue_read_shard_from_core(
                device_memory_address, dst_data.data(), num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, IdleEthTestReadWriteMultipleCoresL1) {
    const DeviceAddr address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_elements = 1000;
    const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
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
            const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
            const distributed::DeviceMemoryAddress device_memory_address = {
                device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
            fd_cq.enqueue_write_shard_to_core(
                device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false);
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            idle_ethernet_cores.size(), std::vector<uint32_t>(num_elements, 0));
        uint32_t i = 0;
        for (const CoreCoord& core : idle_ethernet_cores) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
            const distributed::DeviceMemoryAddress device_memory_address = {
                device_coord, virtual_core, reinterpret_cast<DeviceAddr>(address)};
            fd_cq.enqueue_read_shard_from_core(
                device_memory_address, all_cores_dst_data[i].data(), num_elements * sizeof(uint32_t), false);
            i++;
        }

        distributed::Finish(mesh_device->mesh_command_queue());

        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(src_data, dst_data);
        }
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TestInvalidReadWriteAddressDRAM) {
    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        const uint32_t num_elements = 1010;
        const std::vector<uint32_t> src_data = generate_arange_vector(num_elements * sizeof(uint32_t));

        const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
        const uint32_t size = MetalContext::instance().hal().get_dev_size(HalDramMemAddrType::UNRESERVED);
        const uint32_t dram_end_address = address + size;
        const DeviceAddr dram_end_address_offset = 256;
        const DeviceAddr dram_invalid_address = dram_end_address + dram_end_address_offset;

        const CoreCoord logical_core = {0, 0};
        const CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, CoreType::DRAM);

        const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
        const distributed::DeviceMemoryAddress device_memory_address = {
            device_coord, virtual_core, reinterpret_cast<DeviceAddr>(dram_invalid_address)};
        EXPECT_THROW(
            fd_cq.enqueue_write_shard_to_core(
                device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false),
            std::runtime_error);

        std::vector<uint32_t> dst_data(num_elements, 0);
        EXPECT_THROW(
            fd_cq.enqueue_read_shard_from_core(
                device_memory_address, dst_data.data(), num_elements * sizeof(uint32_t), false),
            std::runtime_error);
    }
}

TEST_F(UnitMeshCQSingleCardFixture, TestReadWriteMultipleCoresDRAM) {
    for (const auto& mesh_device : this->devices_) {
        auto& fd_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
        auto device = mesh_device->get_devices()[0];
        const DeviceAddr address = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
        const uint32_t num_elements = 1000;

        uint32_t dram_core_value = 1;
        for (uint32_t core_x = 0; core_x < device->dram_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->dram_grid_size().y; ++core_y) {
                const CoreCoord core = device->virtual_core_from_logical_core({core_x, core_y}, CoreType::DRAM);
                const std::vector<uint32_t> src_data(num_elements, dram_core_value);
                const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
                const distributed::DeviceMemoryAddress device_memory_address = {
                    device_coord, core, reinterpret_cast<DeviceAddr>(address)};
                fd_cq.enqueue_write_shard_to_core(
                    device_memory_address, src_data.data(), num_elements * sizeof(uint32_t), false);
                dram_core_value++;
            }
        }

        std::vector<std::vector<uint32_t>> all_cores_dst_data(
            device->dram_grid_size().x * device->dram_grid_size().y, std::vector<uint32_t>(num_elements, 0));
        uint32_t j = 0;
        for (uint32_t core_x = 0; core_x < device->dram_grid_size().x; ++core_x) {
            for (uint32_t core_y = 0; core_y < device->dram_grid_size().y; ++core_y) {
                const CoreCoord core = device->virtual_core_from_logical_core({core_x, core_y}, CoreType::DRAM);
                const distributed::MeshCoordinate device_coord = mesh_device->get_view().find_device(device->id());
                const distributed::DeviceMemoryAddress device_memory_address = {
                    device_coord, core, reinterpret_cast<DeviceAddr>(address)};
                fd_cq.enqueue_read_shard_from_core(
                    device_memory_address, all_cores_dst_data[j].data(), num_elements * sizeof(uint32_t), false);
                j++;
            }
        }

        distributed::Finish(mesh_device->mesh_command_queue());

        uint32_t expected_dram_core_value = 1;
        for (const std::vector<uint32_t>& dst_data : all_cores_dst_data) {
            EXPECT_EQ(dst_data, std::vector<uint32_t>(num_elements, expected_dram_core_value));
            expected_dram_core_value++;
        }
    }
}
