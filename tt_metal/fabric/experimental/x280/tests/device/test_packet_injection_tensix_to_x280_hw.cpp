// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// test_packet_injection_tensix_to_x280_hw.cpp -- hardware device test entry
// for the Tensix -> X280 packet injection characterization.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/packet_injection/tensix_to_x280.hpp"

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/program/program_impl.hpp"

#include <exception>
#include <memory>
#include <vector>

namespace {

namespace suite = x280::suites::packet_injection::tensix_to_x280;
namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hreg_pi = x280::harness::regions::packet_injection;

constexpr suite::SweepPoint kSweepPoints[] = {
    {PI_MODE_LATENCY, 1, "LatencyInclude"},
    {PI_MODE_LATENCY, 0, "LatencyExclude"},
    {PI_MODE_THROUGHPUT, 0, "Throughput"},
    {PI_MODE_BACKPRESSURE, 1, "BackpressureInclude"},
    {PI_MODE_BACKPRESSURE, 0, "BackpressureExclude"},
    {PI_MODE_THROUGHPUT, 0, "RecoveryBeforeArrival", PI_ERROR_BAD_ARGS},
};

class PacketInjectionTensixToX280 : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_PACKET_INJECTION_FW_IMAGE",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                    .lim_layout = x280::LimLayout::PacketInjection,
                },
            .suite_label = "packet_injection_tensix_to_x280",
        };
    }
};

}  // namespace

TEST_P(PacketInjectionTensixToX280, Run) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    Config,
    PacketInjectionTensixToX280,
    ::testing::ValuesIn(kSweepPoints),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });

// JIT-compiles the producer for both data-movement RISCs across every
// mode/creation-flag variant, without enqueue, dispatch, X280 runtime or
// prepared LIM state. Compiling a program needs only an initialised device.
TEST(PacketInjectionCompile, BothRiscsCompileForEveryVariant) {
    std::shared_ptr<ttmd::MeshDevice> mesh;
    try {
        mesh = ttmd::MeshDevice::create_unit_mesh(0);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "no device could be opened: " << e.what();
    }
    if (!mesh) {
        GTEST_SKIP() << "no device could be opened: create_unit_mesh(0) returned no mesh";
    }

    const auto kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__);
    const ttm::CoreRangeSet cores(ttm::CoreRange(ttm::CoreCoord{0, 0}, ttm::CoreCoord{0, 0}));

    for (const auto& point : kSweepPoints) {
        hreg_pi::Config cfg{};
        cfg.mode = point.mode;
        cfg.include_creation = point.include_creation;
        cfg.family = hreg_pi::WorkerFamily::Paired;

        try {
            ttm::Program program;
            const suite::ProducerKernels kernels = suite::create_producer_kernels(program, cfg, kernel_dir, cores);
            EXPECT_NE(kernels.brisc, kernels.ncrisc) << point.label << ": expected distinct BRISC and NCRISC kernels";
            program.impl().compile(mesh->get_device(0, 0));
        } catch (const std::exception& e) {
            ADD_FAILURE() << point.label << ": producer JIT compile failed: " << e.what();
        }
    }

    mesh->close();
}
