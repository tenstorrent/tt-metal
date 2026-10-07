// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared fixture for the DRAM-sender transports (GlobalCircularBuffer and PrefetcherPipe): a
// single Blackhole card whose DRAM cores are programmable.
//
// Slow dispatch, inherited from BlackholeSingleCardFixture, which is slow-dispatch-only. That is
// also what lets a DRISC sender and its worker receivers share one Program: slow dispatch
// materializes a program's per-core remote-buffer dense index in ConfigureDeviceWithProgram.
//
// The skip gate lives here rather than in each test file so both transports agree on it: DRAM
// programmable cores need firmware above a fleet-wide floor, and a test that quietly stopped
// skipping (or started) would misreport what a green run proved.

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>

#include "device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal {

class DramSenderFixture : public BlackholeSingleCardFixture {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;
        }
        mesh_device_ = devices_[0].get();
        if (!MetalContext::instance(mesh_device_->impl().get_context_id())
                 .hal()
                 .has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }

    distributed::MeshDevice* mesh_device_{};
};

}  // namespace tt::tt_metal
