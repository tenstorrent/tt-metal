// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include "dispatch_fixture.hpp"

using namespace tt::tt_metal;

class ProgramWithKernelCreatedFromStringFixture : public DispatchFixture {
protected:
    void SetUp() override {
        DispatchFixture::SetUp();
        for (IDevice* device : this->devices_) {
            const chip_id_t device_id = device->id();
            this->device_ids_to_devices_[device_id] = device;
        }
    }

    void TearDown() override { detail::CloseDevices(this->device_ids_to_devices_); }

private:
    std::map<chip_id_t, IDevice*> device_ids_to_devices_;
};
