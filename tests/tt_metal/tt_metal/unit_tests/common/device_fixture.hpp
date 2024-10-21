// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

class DeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            TT_THROW("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

        // Some CI machines have lots of cards, running all tests on all cards is slow
        // Coverage for multidevices is decent if we just confirm 2 work
        if (arch_ == tt::ARCH::GRAYSKULL && num_devices_ > 2) {
            num_devices_ = 2;
        }

        vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            ids.push_back(id);
        }
        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        devices_ = tt::DevicePool::instance().get_all_active_devices();
    }

    void TearDown() override {
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (unsigned int id = 0; id < devices_.size(); id++) {
            if (devices_.at(id)->is_initialized()) {
                tt::tt_metal::CloseDevice(devices_.at(id));
            }
        }
    }

    std::vector<tt::tt_metal::v1::DeviceHandle> devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};


class DeviceSingleCardFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            TT_THROW("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        const chip_id_t mmio_device_id = 0;
        reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id});
        device_ = reserved_devices_.at(mmio_device_id);


        num_devices_ = reserved_devices_.size();
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    tt::tt_metal::Device* device_;
    std::map<chip_id_t, tt::tt_metal::Device*> reserved_devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};

class GalaxyFixture : public ::testing::Test {
   protected:
    void SkipTestSuiteIfNotGalaxyMotherboard()
    {
        const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (!(arch == tt::ARCH::WORMHOLE_B0 && num_devices >= 32))
        {
            GTEST_SKIP();
        }
    }

    void InitializeDevices()
    {
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        vector<chip_id_t> ids;
        for (uint32_t id = 0; id < num_devices; id++)
        {
            ids.push_back(id);
        }
        this->device_ids_to_devices_ = tt::tt_metal::detail::CreateDevices(ids);
        this->devices_ = tt::DevicePool::instance().get_all_active_devices();
    }

    void SetUp() override
    {
        this->SkipTestSuiteIfNotGalaxyMotherboard();
        this->InitializeDevices();
    }

    void TearDown() override
    {
        tt::tt_metal::detail::CloseDevices(this->device_ids_to_devices_);
        this->device_ids_to_devices_.clear();
        this->devices_.clear();
    }

    std::vector<tt::tt_metal::v1::DeviceHandle> devices_;

   private:
    std::map<chip_id_t, Device*> device_ids_to_devices_;
};

class TGFixture : public GalaxyFixture
{
   protected:
    void SkipTestSuiteIfNotTG()
    {
        this->SkipTestSuiteIfNotGalaxyMotherboard();
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 32 && num_pcie_devices == 4))
        {
            GTEST_SKIP();
        }
    }

    void SetUp() override
    {
        this->SkipTestSuiteIfNotTG();
        this->InitializeDevices();
    }
};

class TGGFixture : public GalaxyFixture
{
   protected:
    void SkipTestSuiteIfNotTGG()
    {
        this->SkipTestSuiteIfNotGalaxyMotherboard();
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 64 && num_pcie_devices == 8))
        {
            GTEST_SKIP();
        }
    }

    void SetUp() override
    {
        this->SkipTestSuiteIfNotTGG();
        this->InitializeDevices();
    }
};
