// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
            const auto device_id = mesh_device->get_device_ids()[0];
            this->device_ids_to_devices_[device_id] = mesh_device;
        }
    }

private:
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> device_ids_to_devices_;
};
