// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ComputeHardwareConfig -> internal compute config translation done by MakeProgramFromSpec.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/hal.hpp>

#include "impl/host_api/temp_quasar_api.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestBlackhole;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

// ============================================================================
// Compute-config translation stability (defaults + inversion/enum)
// ============================================================================
// These tests pin the public ComputeHardwareConfig -> internal
// ComputeConfig / QuasarComputeConfig translation at the boundary where the
// field rename is absorbed (MakeGen1ComputeConfig / MakeGen2ComputeConfig).
//
// Several of these fields are performance / numerical-precision settings that do
// NOT change a functional pass/fail result, so a flipped inversion
// (dst_full_sync_en <-> double_buffer_dest) or a wrong precision-enum direction
// would be invisible to the behavioral tests. Asserting the internal values
// documents that the defaults must not change and guards the conversions
// against silent drift. (The public -> internal translation is Metal-side and
// testable here; the TTNN ComputeKernelConfig -> public bridge lives above this
// layer and is out of scope for a Metal unit test.)

TEST_F(ProgramSpecTestQuasar, CPU_ComputeHardwareConfigDefaultsMapToInternalDefaults) {
    // A default ComputeHardwareConfig{} must yield the historical internal QuasarComputeConfig defaults.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // compute_kernel carries a default ComputeHardwareConfig
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<experimental::quasar::QuasarComputeConfig>(built_variant);
    EXPECT_EQ(built.math_fidelity, MathFidelity::HiFi4);
    EXPECT_FALSE(built.fp32_dest_acc_en);
    EXPECT_FALSE(built.dst_full_sync_en);   // double_buffer_dest defaults true -> !true
    EXPECT_FALSE(built.math_approx_mode);   // sfpu_precision_mode defaults Precise
    EXPECT_FALSE(built.enable_trisc0_rvv);  // config_2xx unset
}

// Gen1 counterpart of the compute-config translation-stability tests (the Gen2 pair lives in the
// Quasar suite): a default ComputeHardwareConfig{} must yield the historical internal ComputeConfig
// defaults. Guards the perf/precision settings that don't move a functional pass/fail result.
TEST_F(ProgramSpecTestGen1, CPU_ComputeHardwareConfigDefaultsMapToInternalDefaults) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();  // compute_kernel carries a default ComputeHardwareConfig
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<ComputeConfig>(built_variant);
    EXPECT_EQ(built.math_fidelity, MathFidelity::HiFi4);
    EXPECT_FALSE(built.fp32_dest_acc_en);
    EXPECT_FALSE(built.dst_full_sync_en);   // double_buffer_dest defaults true -> !true
    EXPECT_FALSE(built.bfp8_pack_precise);  // bfp_pack_precision_mode defaults Approximate
    EXPECT_FALSE(built.math_approx_mode);   // sfpu_precision_mode defaults Precise
    EXPECT_FALSE(built.enable_trisc2_rvv);  // config_1xx unset
}

// The generation-specific RVV opt-ins must reach the internal configs. MakeProgramFromSpec also
// compiles, so each test runs on a mock architecture that supports its opt-in.
TEST_F(ProgramSpecTestQuasar, CPU_Config2xxTrisc0RvvMapsToInternal) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            std::get<ComputeHardwareConfig>(kernel.hw_config).config_2xx =
                ComputeHardwareConfig::Compute2XXConfig{.enable_trisc0_rvv = true};
        }
    }
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<experimental::quasar::QuasarComputeConfig>(built_variant);
    EXPECT_TRUE(built.enable_trisc0_rvv);
}

TEST_F(ProgramSpecTestBlackhole, CPU_Config1xxTrisc2RvvMapsToInternal) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            std::get<ComputeHardwareConfig>(kernel.hw_config).config_1xx =
                ComputeHardwareConfig::Compute1XXConfig{.enable_trisc2_rvv = true};
        }
    }
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<ComputeConfig>(built_variant);
    EXPECT_TRUE(built.enable_trisc2_rvv);
}

TEST_F(ProgramSpecTestQuasar, CPU_ComputeHardwareConfigInversionAndEnumMapToInternal) {
    // Non-default polarity: the double_buffer_dest inversion and the SFPU precision-enum mapping
    // must reach the internal config correctly. Guards the case a defaults-only check would miss
    // (a flip compensated by a changed default).
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            config.double_buffer_dest = false;                                  // -> internal dst_full_sync_en == true
            config.sfpu_precision_mode = tt::tt_metal::Precision::Approximate;  // -> internal math_approx_mode == true
        }
    }
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<experimental::quasar::QuasarComputeConfig>(built_variant);
    EXPECT_TRUE(built.dst_full_sync_en);
    EXPECT_TRUE(built.math_approx_mode);
}

TEST_F(ProgramSpecTestQuasar, CPU_UnpackToDestModePlacedAtDfbIdSlot) {
    // Regression test for the unpack_to_dest_mode sizing bug: the JIT consumer
    // iterates hal::get_num_dataflow_buffers() slots, so BuildUnpackToDestModeVector
    // must size the vector to that count and place each user-supplied mode at slot dfb_id.
    // Pre-fix code sized the vector to the number of DFBs, which produced silent
    // OOB reads downstream when num_dfbs < max_dfbs.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb0 = MakeMinimalDFB("dfb_0");
    dfb0.data_format_metadata = tt::DataFormat::Float16_b;
    auto dfb1 = MakeMinimalDFB("dfb_1");
    // dfb_1 is FP32 so the user can opt into UnpackToDest on it.
    dfb1.data_format_metadata = tt::DataFormat::Float32;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out0"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_1"}, "out1"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in0"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_1"}, "in1"));

    auto& compute_config = std::get<ComputeHardwareConfig>(consumer.hw_config);
    compute_config.enable_32_bit_dest = true;
    compute_config.unpack_modes = {{DFBSpecName{"dfb_1"}, UnpackMode::UnpackToDest}};

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb0, dfb1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Inspect the constructed compute kernel's QuasarComputeConfig:
    //  - vector must be sized to max_dfbs (so JIT's iteration up to max_dfbs is in-bounds)
    //  - the user-supplied mode must land at slot dfb_id (not at iteration order)
    //  - other slots stay Default
    const auto& impl = program.impl();
    auto consumer_kernel = impl.get_kernel_by_spec_name("consumer");
    const auto built_config_variant = consumer_kernel->config();
    const auto& built_config = std::get<experimental::quasar::QuasarComputeConfig>(built_config_variant);

    EXPECT_EQ(built_config.unpack_to_dest_mode.size(), tt::tt_metal::hal::get_num_dataflow_buffers());
    EXPECT_EQ(built_config.unpack_to_dest_mode[impl.get_dfb_handle("dfb_1")], UnpackToDestMode::UnpackToDestFp32);
    EXPECT_EQ(built_config.unpack_to_dest_mode[impl.get_dfb_handle("dfb_0")], UnpackToDestMode::Default);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
