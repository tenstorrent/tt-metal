// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/metal_runner.hpp -- gtest fixture for a case that launches a Metal
// kernel on a core-local RISC (DRISC or idle ERISC) and needs no X280 runtime.

#ifndef X280_HARNESS_METAL_RUNNER_HPP
#define X280_HARNESS_METAL_RUNNER_HPP

#include "harness/slow_dispatch.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace x280::harness {

// SetUp fails unless TT_METAL_SLOW_DISPATCH_MODE=1 (slow_dispatch_linked), then
// opens a unit mesh on chip 0 as `mesh`; TearDown releases it.
template <typename Axis>
class MetalRunner : public ::testing::TestWithParam<Axis> {
public:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh;

protected:
    void SetUp() override {
        ASSERT_TRUE(slow_dispatch_linked());
        mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    }

    void TearDown() override { mesh.reset(); }
};

}  // namespace x280::harness

#endif  // X280_HARNESS_METAL_RUNNER_HPP
