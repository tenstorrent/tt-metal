// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"

using std::vector;
using namespace tt;

TEST_F(DeviceFixture, InitializeGlobalSemaphores) {
    CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));

    auto cores_vec = corerange_to_cores(cores);
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto device = devices_.at(id);
        {
            uint32_t initial_value = 1;
            auto global_semaphore = tt_metal::CreateGlobalSemaphore(device, cores, initial_value);
            auto address = global_semaphore->address();

            for (const auto& core : cores_vec) {
                vector<uint32_t> sem_vals =
                    tt::llrt::read_hex_vec_from_core(device->id(), core, address, sizeof(uint32_t));
                EXPECT_EQ(sem_vals[0], initial_value);
            }
        }
        {
            uint32_t initial_value = 2;
            auto global_semaphore = tt_metal::CreateGlobalSemaphore(device, cores, initial_value);
            auto address = global_semaphore->address();

            for (const auto& core : cores_vec) {
                vector<uint32_t> sem_vals =
                    tt::llrt::read_hex_vec_from_core(device->id(), core, address, sizeof(uint32_t));
                EXPECT_EQ(sem_vals[0], initial_value);
            }
        }
    }
}

TEST_F(DeviceFixture, CreateMultipleGlobalSemaphoresOnSameCore) {
    std::vector<CoreRangeSet> cores{CoreRange({0, 0}, {1, 1}), CoreRange({0, 0}, {2, 2}), CoreRange({3, 2}, {2, 3})};
    std::vector<std::vector<CoreCoord>> cores_vecs;
    cores_vecs.reserve(cores.size());
    std::vector<uint32_t> initial_values{1, 2, 3};
    for (const auto& crs : cores) {
        cores_vecs.push_back(corerange_to_cores(crs));
    }
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto device = devices_.at(id);
        {
            std::vector<std::shared_ptr<tt_metal::GlobalSemaphore>> global_semaphores;
            global_semaphores.reserve(cores.size());
            std::vector<DeviceAddr> addresses;
            addresses.reserve(cores.size());
            for (size_t i = 0; i < cores.size(); i++) {
                global_semaphores.push_back(tt_metal::CreateGlobalSemaphore(device, cores[i], initial_values[i]));
                addresses.push_back(global_semaphores[i]->address());
            }
            for (size_t i = 0; i < cores.size(); i++) {
                const auto& address = addresses[i];
                const auto& initial_value = initial_values[i];
                const auto& cores_vec = cores_vecs[i];
                for (const auto& core : cores_vec) {
                    vector<uint32_t> sem_vals =
                        tt::llrt::read_hex_vec_from_core(device->id(), core, address, sizeof(uint32_t));
                    EXPECT_EQ(sem_vals[0], initial_value);
                }
            }
        }
    }
}
