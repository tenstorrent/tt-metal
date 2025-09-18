// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include "mesh_dispatch_fixture.hpp"

using namespace tt::tt_metal;

class ProgramWithKernelCreatedFromStringFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        MeshDispatchFixture::SetUp();
        for (const auto& mesh_device : this->devices_) {
            auto device = mesh_device->get_devices()[0];
            const chip_id_t device_id = device->id();
            this->device_ids_to_devices_[device_id] = mesh_device;
        }
    }

private:
    std::map<chip_id_t, std::shared_ptr<distributed::MeshDevice>> device_ids_to_devices_;
};
