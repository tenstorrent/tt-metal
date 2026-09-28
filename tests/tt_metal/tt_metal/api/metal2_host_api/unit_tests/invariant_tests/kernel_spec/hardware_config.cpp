// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec::hw_config against the kernel's own bindings (kernel_spec.hpp):
// unpack_modes keys name bound DFBs; on Gen1, UnpackToDest on a consumed DFB needs enable_32_bit_dest.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_ComputeConfigUnpackToDestModeReferencesUnboundDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    // Set an unpack_modes entry referencing a DFB this kernel doesn't bind
    // (in this case, a DFB that doesn't exist in the spec at all).
    auto& compute_config = std::get<ComputeHardwareConfig>(consumer.hw_config);
    compute_config.unpack_modes = {{DFBSpecName{"nonexistent_dfb"}, UnpackMode::UnpackToDest}};

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'consumer' unpack_modes entry references DFB 'nonexistent_dfb', "
                                 "which the kernel does not bind")));
}

TEST_F(ProgramSpecTestGen1, CPU_ConsumerUnpackToDestBelow32BitWithoutEnableFailsForPerf) {
    // On Gen1, UnpackToDest on a consumed <=16-bit DFB without enable_32_bit_dest bypasses the
    // SrcA/B path for no precision benefit — rejected as bad-for-perf. (On Gen2 the identical spec
    // is accepted — see ProgramSpecTestQuasar.ConsumerUnpackToDestBelow32BitWithoutEnableSucceeds.)
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();  // dfb_0 is Float16_b, consumed by compute_kernel
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            // enable_32_bit_dest stays at its default (false).
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("leads to worse performance")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ConsumerUnpackToDestBelow32BitWithoutEnableSucceeds) {
    // Gen2 has no unpack-to-Dest performance penalty, so UnpackToDest on a consumed <=16-bit DFB
    // is accepted even without enable_32_bit_dest (a 16-bit Dest holds a <=16-bit datum). On Gen1
    // the same spec is rejected as bad-for-perf — see the ProgramSpecTestGen1 counterpart.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b, consumed by compute_kernel
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            // enable_32_bit_dest stays at its default (false).
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ComputeConfigMathFidelitySucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Find the compute kernel and set math fidelity options
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            config.fpu_math_fidelity = MathFidelity::LoFi;
            config.enable_32_bit_dest = true;
            config.sfpu_precision_mode = tt::tt_metal::Precision::Approximate;
        }
    }

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
