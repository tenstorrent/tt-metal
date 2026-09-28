// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="UnitMeshFixture.ComputeConfigMathFidelity*"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "impl/program/program_impl.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {
namespace {

// A kernel that compiles only at one fidelity. The compile itself is the assertion, so this
// needs no CBs, no DST and no readback — a missing or wrong MATH_FIDELITY fails the JIT.
std::string kernel_asserting_fidelity(int expected) {
    return R"(
        #include "api/compute/common.h"
        void kernel_main() {
            static_assert(static_cast<int>(MATH_FIDELITY) == )" +
           std::to_string(expected) + R"(, "ComputeConfig.math_fidelity did not reach the kernel");
        }
    )";
}

void launch_with_fidelity(distributed::MeshDevice& device, const std::string& src, MathFidelity fidelity) {
    Program program = CreateProgram();
    CreateKernelFromString(program, src, CoreCoord{0, 0}, ComputeConfig{.math_fidelity = fidelity});
    LaunchProgram(device, std::move(program));
}

}  // namespace

// The host ComputeConfig's math_fidelity has to reach the compute kernel's TU.
TEST_F(UnitMeshFixture, ComputeConfigMathFidelityReachesTheKernel) {
    for (const auto fidelity : {MathFidelity::LoFi, MathFidelity::HiFi2, MathFidelity::HiFi3, MathFidelity::HiFi4}) {
        EXPECT_NO_THROW(
            launch_with_fidelity(this->device(), kernel_asserting_fidelity(static_cast<int>(fidelity)), fidelity))
            << "MathFidelity " << static_cast<int>(fidelity) << " did not reach the kernel";
    }
}

// ... and it has to be part of the JIT cache key. Same source, same defines, same compile args:
// a fidelity-blind key would serve the HiFi2 artifact to the HiFi3 launch and compile nothing,
// so the kernel that must not compile would launch clean.
TEST_F(UnitMeshFixture, ComputeConfigMathFidelitySeparatesJitCacheEntries) {
    const std::string hifi2_only = kernel_asserting_fidelity(static_cast<int>(MathFidelity::HiFi2));

    ASSERT_NO_THROW(launch_with_fidelity(this->device(), hifi2_only, MathFidelity::HiFi2));
    EXPECT_ANY_THROW(launch_with_fidelity(this->device(), hifi2_only, MathFidelity::HiFi3))
        << "the HiFi3 launch reused the HiFi2 artifact: math_fidelity is missing from the cache key";
}

}  // namespace tt::tt_metal
