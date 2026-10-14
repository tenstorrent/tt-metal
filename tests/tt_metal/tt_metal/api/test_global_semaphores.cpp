// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <vector>

#include <tt-metalium/device.hpp>
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {

TEST_F(MeshDispatchFixture, InitializeGlobalSemaphores) {
    CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));

    auto cores_vec = corerange_to_cores(cores);
    for (const auto& mesh_device : devices_) {
        auto device_id = mesh_device->get_device_ids()[0];
        {
            uint32_t initial_value = 1;
            auto global_semaphore = tt::tt_metal::GlobalSemaphore(*mesh_device, cores, initial_value);
            auto address = global_semaphore.address();
            distributed::Synchronize(*mesh_device, std::nullopt);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    device_id, mesh_device->worker_core_from_logical_core(core), address, sizeof(uint32_t));

                EXPECT_EQ(sem_vals[0], initial_value);
            }
        }
        {
            uint32_t initial_value = 2;
            auto global_semaphore = tt::tt_metal::GlobalSemaphore(*mesh_device, cores, initial_value);
            auto address = global_semaphore.address();
            distributed::Synchronize(*mesh_device, std::nullopt);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    device_id, mesh_device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
                EXPECT_EQ(sem_vals[0], initial_value);
            }
        }
    }
}

TEST_F(MeshDispatchFixture, CreateMultipleGlobalSemaphoresOnSameCore) {
    std::vector<CoreRangeSet> cores{CoreRange({0, 0}, {1, 1}), CoreRange({0, 0}, {2, 2}), CoreRange({3, 3}, {5, 6})};
    std::vector<std::vector<CoreCoord>> cores_vecs;
    cores_vecs.reserve(cores.size());
    std::vector<uint32_t> initial_values{1, 2, 3};
    for (const auto& crs : cores) {
        cores_vecs.push_back(corerange_to_cores(crs));
    }
    for (const auto& mesh_device : devices_) {
        auto device_id = mesh_device->get_device_ids()[0];
        {
            std::vector<tt::tt_metal::GlobalSemaphore> global_semaphores;
            global_semaphores.reserve(cores.size());
            std::vector<DeviceAddr> addresses;
            addresses.reserve(cores.size());
            for (size_t i = 0; i < cores.size(); i++) {
                global_semaphores.push_back(tt::tt_metal::GlobalSemaphore(*mesh_device, cores[i], initial_values[i]));
                addresses.push_back(global_semaphores[i].address());
            }
            distributed::Synchronize(*mesh_device, std::nullopt);
            for (size_t i = 0; i < cores.size(); i++) {
                const auto& address = addresses[i];
                const auto& initial_value = initial_values[i];
                const auto& cores_vec = cores_vecs[i];
                for (const auto& core : cores_vec) {
                    auto sem_vals = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                        device_id,
                        mesh_device->worker_core_from_logical_core(core),
                        address,
                        sizeof(uint32_t));
                    EXPECT_EQ(sem_vals[0], initial_value);
                }
            }
        }
    }
}

TEST_F(MeshDispatchFixture, ResetGlobalSemaphores) {
    CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));

    auto cores_vec = corerange_to_cores(cores);
    for (const auto& mesh_device : devices_) {
        auto device_id = mesh_device->get_device_ids()[0];
        {
            uint32_t initial_value = 1;
            uint32_t reset_value = 2;
            std::vector<uint32_t> overwrite_value = {2};
            auto global_semaphore = tt::tt_metal::GlobalSemaphore(*mesh_device, cores, initial_value);
            auto address = global_semaphore.address();
            distributed::Synchronize(*mesh_device, std::nullopt);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    device_id, mesh_device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    device_id, mesh_device->worker_core_from_logical_core(core), overwrite_value, address);
                EXPECT_EQ(sem_vals[0], initial_value);
            }
            tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_id);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    device_id, mesh_device->worker_core_from_logical_core(core), address, sizeof(uint32_t));

                EXPECT_EQ(sem_vals[0], overwrite_value[0]);
            }
            global_semaphore.reset_semaphore_value(reset_value);
            distributed::Synchronize(*mesh_device, std::nullopt);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    device_id, mesh_device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    device_id, mesh_device->worker_core_from_logical_core(core), overwrite_value, address);
                EXPECT_EQ(sem_vals[0], reset_value);
            }
        }
    }
}

namespace {

void ExpectPersistentValueOnEveryCore(
    const distributed::MeshDevice& mesh_device, const GlobalSemaphore& sem, const CoreRangeSet& cores, uint32_t value) {
    const auto& cluster = MetalContext::instance().get_cluster();
    for (IDevice* device : mesh_device.get_devices()) {
        for (const auto& core : corerange_to_cores(cores)) {
            auto sem_vals =
                cluster.read_core(device->id(), device->worker_core_from_logical_core(core), sem.address(), sizeof(uint32_t));
            EXPECT_EQ(sem_vals[0], value) << "device " << device->id() << " core " << core.str();
        }
    }
}

}  // namespace

// Initial value and reset land on every core of the grid, on every device, at the one arena address.
// The full grid makes the fast-dispatch reset a multi-row multicast.
TEST_F(MeshDispatchFixture, PersistentGlobalSemaphoreInitAndResetReachEveryCore) {
    for (const auto& mesh_device : devices_) {
        const auto grid = mesh_device->compute_with_storage_grid_size();
        const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

        auto sem = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0x1234);
        EXPECT_TRUE(std::get<2>(sem.attribute_values()));
        distributed::Synchronize(*mesh_device, std::nullopt);
        ExpectPersistentValueOnEveryCore(*mesh_device, sem, cores, 0x1234);

        sem.reset_semaphore_value(0xABCD);
        distributed::Synchronize(*mesh_device, std::nullopt);
        ExpectPersistentValueOnEveryCore(*mesh_device, sem, cores, 0xABCD);
    }
}

// Several disjoint ranges in one CoreRangeSet: every range is reset, nothing between them is touched.
TEST_F(MeshDispatchFixture, PersistentGlobalSemaphoreResetCoversEveryRange) {
    for (const auto& mesh_device : devices_) {
        const CoreRangeSet cores(std::vector<CoreRange>{CoreRange({0, 0}, {1, 1}), CoreRange({3, 0}, {3, 2})});
        const CoreCoord untouched{2, 0};

        auto sem = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0);
        const auto& cluster = MetalContext::instance().get_cluster();
        for (IDevice* device : mesh_device->get_devices()) {
            cluster.write_core(
                device->id(), device->worker_core_from_logical_core(untouched), std::vector<uint32_t>{0x5A5A}, sem.address());
            cluster.l1_barrier(device->id());
        }

        sem.reset_semaphore_value(7);
        distributed::Synchronize(*mesh_device, std::nullopt);
        ExpectPersistentValueOnEveryCore(*mesh_device, sem, cores, 7);
        ExpectPersistentValueOnEveryCore(*mesh_device, sem, CoreRangeSet(CoreRange(untouched)), 0x5A5A);
    }
}

TEST_F(MeshDispatchFixture, PersistentGlobalSemaphoreFreedSlotIsReused) {
    for (const auto& mesh_device : devices_) {
        const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
        DeviceAddr first_address = 0;
        {
            auto first = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0);
            first_address = first.address();
            auto second = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0);
            EXPECT_NE(second.address(), first_address);
        }
        auto again = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0);
        EXPECT_EQ(again.address(), first_address);
    }
}

// Copies share one arena slot; it is freed only when the last copy goes away.
TEST_F(MeshDispatchFixture, PersistentGlobalSemaphoreCopiesShareSlot) {
    for (const auto& mesh_device : devices_) {
        const CoreRangeSet cores(CoreRange({0, 0}, {0, 0}));
        std::optional<GlobalSemaphore> copy;
        DeviceAddr address = 0;
        {
            auto sem = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0);
            address = sem.address();
            copy = sem;
            EXPECT_EQ(copy->address(), address);
        }
        auto other = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0);
        EXPECT_NE(other.address(), address);
    }
}

// Placing program-local L1 seals the core's arena: creation fails while the program is alive and
// succeeds again once it is destroyed.
TEST_F(MeshDispatchFixture, PersistentGlobalSemaphoreCreationFailsOnSealedCores) {
    for (const auto& mesh_device : devices_) {
        const CoreRange core_range({0, 0}, {0, 0});
        {
            distributed::MeshWorkload workload;
            Program program = CreateProgram();
            const uint32_t page_size = tt::tile_size(tt::DataFormat::Float16_b);
            CreateCircularBuffer(
                program,
                core_range,
                CircularBufferConfig(2 * page_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_0, page_size));
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
                core_range,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
            RunProgram(mesh_device, workload);

            EXPECT_ANY_THROW(experimental::CreatePersistentGlobalSemaphore(*mesh_device, CoreRangeSet(core_range), 0));
        }
        EXPECT_NO_THROW(experimental::CreatePersistentGlobalSemaphore(*mesh_device, CoreRangeSet(core_range), 0));
    }
}

TEST_F(MeshDispatchFixture, PersistentGlobalSemaphoreHashDiffersFromBufferBacked) {
    for (const auto& mesh_device : devices_) {
        const CoreRangeSet cores(CoreRange({0, 0}, {0, 0}));
        auto persistent = experimental::CreatePersistentGlobalSemaphore(*mesh_device, cores, 0);
        auto buffer_backed = CreateGlobalSemaphore(*mesh_device, cores, 0);
        EXPECT_NE(std::hash<GlobalSemaphore>{}(persistent), std::hash<GlobalSemaphore>{}(buffer_backed));
    }
}

} // namespace tt::tt_metal
