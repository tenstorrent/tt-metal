// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared fixture for DRAM-sender PrefetcherPipe tests: a single Blackhole card whose DRAM cores are
// programmable.
//
// Slow dispatch, inherited from BlackholeSingleCardFixture, which is slow-dispatch-only. That is
// also what lets a DRISC sender and its worker receivers share one Program: slow dispatch
// materializes a program's per-core remote-buffer dense index in ConfigureDeviceWithProgram.
//
// The skip gate lives here rather than in each fixture so every DRAM-sender fixture agrees on it: DRAM
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

// The skip gate: whether `mesh_device`'s DRAM cores can run a DRISC sender.
inline bool dram_cores_programmable(distributed::MeshDevice& mesh_device) {
    return MetalContext::instance(mesh_device.impl().get_context_id())
        .hal()
        .has_programmable_core_type(HalProgrammableCoreType::DRAM);
}

class DramSenderFixture : public BlackholeSingleCardFixture {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;
        }
        mesh_device_ = devices_[0].get();
        if (!dram_cores_programmable(*mesh_device_)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }

    distributed::MeshDevice* mesh_device_{};
};

}  // namespace tt::tt_metal
