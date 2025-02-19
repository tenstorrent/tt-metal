// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "device_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_semaphore.hpp>

TEST_F(DispatchFixture, InitializeGlobalSemaphores) {
    CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));

    auto cores_vec = corerange_to_cores(cores);
    for (auto device : devices_) {
        {
            uint32_t initial_value = 1;
            auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(device, cores, initial_value);
            auto address = global_semaphore.address();
            Synchronize(device);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::llrt::read_hex_vec_from_core(
                    device->id(), device->worker_core_from_logical_core(core), address, sizeof(uint32_t));

                EXPECT_EQ(sem_vals[0], initial_value);
            }
        }
        {
            uint32_t initial_value = 2;
            auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(device, cores, initial_value);
            auto address = global_semaphore.address();
            Synchronize(device);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::llrt::read_hex_vec_from_core(
                    device->id(), device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
                EXPECT_EQ(sem_vals[0], initial_value);
            }
        }
    }
}

TEST_F(DispatchFixture, CreateMultipleGlobalSemaphoresOnSameCore) {
    std::vector<CoreRangeSet> cores{CoreRange({0, 0}, {1, 1}), CoreRange({0, 0}, {2, 2}), CoreRange({3, 3}, {5, 6})};
    std::vector<std::vector<CoreCoord>> cores_vecs;
    cores_vecs.reserve(cores.size());
    std::vector<uint32_t> initial_values{1, 2, 3};
    for (const auto& crs : cores) {
        cores_vecs.push_back(corerange_to_cores(crs));
    }
    for (auto device : devices_) {
        {
            std::vector<tt::tt_metal::GlobalSemaphore> global_semaphores;
            global_semaphores.reserve(cores.size());
            std::vector<DeviceAddr> addresses;
            addresses.reserve(cores.size());
            for (size_t i = 0; i < cores.size(); i++) {
                global_semaphores.push_back(tt::tt_metal::CreateGlobalSemaphore(device, cores[i], initial_values[i]));
                addresses.push_back(global_semaphores[i].address());
            }
            Synchronize(device);
            for (size_t i = 0; i < cores.size(); i++) {
                const auto& address = addresses[i];
                const auto& initial_value = initial_values[i];
                const auto& cores_vec = cores_vecs[i];
                for (const auto& core : cores_vec) {
                    auto sem_vals = tt::llrt::read_hex_vec_from_core(
                        device->id(),
                        device->worker_core_from_logical_core(core),
                        address,
                        sizeof(uint32_t));
                    EXPECT_EQ(sem_vals[0], initial_value);
                }
            }
        }
    }
}

TEST_F(DispatchFixture, ResetGlobalSemaphores) {
    CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));

    auto cores_vec = corerange_to_cores(cores);
    for (auto device : devices_) {
        {
            uint32_t initial_value = 1;
            uint32_t reset_value = 2;
            std::vector<uint32_t> overwrite_value = {2};
            auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(device, cores, initial_value);
            auto address = global_semaphore.address();
            Synchronize(device);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::llrt::read_hex_vec_from_core(
                    device->id(), device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
                tt::llrt::write_hex_vec_to_core(
                    device->id(), device->worker_core_from_logical_core(core), overwrite_value, address);
                EXPECT_EQ(sem_vals[0], initial_value);
            }
            tt::Cluster::instance().l1_barrier(device->id());
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::llrt::read_hex_vec_from_core(
                    device->id(), device->worker_core_from_logical_core(core), address, sizeof(uint32_t));

                EXPECT_EQ(sem_vals[0], overwrite_value[0]);
            }
            global_semaphore.reset_semaphore_value(reset_value);
            Synchronize(device);
            for (const auto& core : cores_vec) {
                auto sem_vals = tt::llrt::read_hex_vec_from_core(
                    device->id(), device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
                tt::llrt::write_hex_vec_to_core(
                    device->id(), device->worker_core_from_logical_core(core), overwrite_value, address);
                EXPECT_EQ(sem_vals[0], reset_value);
            }
        }
    }
}
