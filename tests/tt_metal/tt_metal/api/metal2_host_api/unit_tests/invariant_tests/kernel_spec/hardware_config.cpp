// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec::hw_config against the kernel's bindings and the DFBs they name (kernel_spec.hpp):
// unpack_modes keys name bound DFBs; for each DFB a compute kernel consumes, UnpackToDest needs enable_32_bit_dest
// (32-bit formats on every generation, every format on Gen1), and Float32 with enable_32_bit_dest needs an explicit
// unpack_modes entry.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

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

TEST_F(ProgramSpecTestQuasar, CPU_NonFP32DFBWithoutUnpackToDestModeEntrySucceeds) {
    // Non-FP32 DFBs default to Default; omitting an entry is the expected idiom.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b (non-FP32)
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_NonFP32DFBWithExplicitDefaultUnpackToDestModeSucceeds) {
    // Existing call sites that explicitly spell out Default for non-FP32 DFBs keep working.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToSrc}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_NonFP32DFBWithUnpackToDestFp32ModeSucceeds) {
    // UnpackToDest on a non-Float32 DFB is INERT: the LLK ignores the mode where the data
    // isn't FP32. The validator tolerates it (rejecting it would force porters to dtype-gate
    // legacy unpack_to_dest_mode vectors that set UnpackToDestFp32 unconditionally). With
    // enable_32_bit_dest=true the entry is coherent, so the spec validates.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b (non-FP32)
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            config.enable_32_bit_dest = true;
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_FP32ConsumerWithFp32DestAccEnAndNoEntryFails) {
    // The narrow case where a choice is required: CONSUMER + FP32 + enable_32_bit_dest=true.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            config.enable_32_bit_dest = true;
        }
    }
    // Compute kernel intentionally has no unpack_modes entry.

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "Compute kernel 'compute_kernel' consumes FP32 DFB 'dfb_0' with enable_32_bit_dest=true, but "
            "provides no unpack_modes entry for this DFB")));
}

TEST_F(ProgramSpecTestQuasar, CPU_FP32ConsumerWithoutFp32DestAccEnDoesNotRequireEntry) {
    // Without enable_32_bit_dest, UnpackToDest is incoherent (Dest is 16-bit), so there's
    // no real choice — UnpackToSrc is the only valid value. No explicit entry required.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    // enable_32_bit_dest stays at its default (false). No unpack_modes entry.
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_FP32ProducerOnlyBindingDoesNotRequireEntry) {
    // A compute kernel that only PRODUCES an FP32 DFB never unpacks it, so the unpack mode
    // is dead config — no explicit entry required regardless of enable_32_bit_dest.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_compute = MakeMinimalGen2ComputeKernel("producer_compute");
    auto& producer_config = std::get<ComputeHardwareConfig>(producer_compute.hw_config);
    producer_config.enable_32_bit_dest = true;

    auto consumer_dm = MakeMinimalGen2DMKernel("consumer_dm");

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float32;

    producer_compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out"));
    consumer_dm.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in"));

    spec.kernels = {producer_compute, consumer_dm};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer_compute", "consumer_dm"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_UnpackToDestFp32OnProducerBindingSucceeds) {
    // UnpackToDest on a producer-only binding is INERT (producers don't unpack), so the
    // validator tolerates it rather than rejecting. With enable_32_bit_dest=true it is coherent.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_compute = MakeMinimalGen2ComputeKernel("producer_compute");
    auto& producer_config = std::get<ComputeHardwareConfig>(producer_compute.hw_config);
    producer_config.enable_32_bit_dest = true;
    producer_config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};

    auto consumer_dm = MakeMinimalGen2DMKernel("consumer_dm");

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float32;

    producer_compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out"));
    consumer_dm.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in"));

    spec.kernels = {producer_compute, consumer_dm};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer_compute", "consumer_dm"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_UnpackToDestFp32WithoutFp32DestAccEnFails) {
    // A 32-bit-format DFB (here Float32) cannot be unpacked into a 16-bit Dest, so UnpackToDest
    // on a consumed 32-bit DFB with enable_32_bit_dest=false is rejected on every generation.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            // enable_32_bit_dest stays at its default (false).
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("A 32-bit datum cannot be unpacked into a 16-bit Dest register")));
}

TEST_F(ProgramSpecTestQuasar, CPU_FP32DFBWithDefaultUnpackToDestModeSucceeds) {
    // UnpackToSrc is always a valid value, even outside the (CONSUMER + FP32 + enable_32_bit_dest) triple.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            config.enable_32_bit_dest = true;
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToSrc}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ValidUnpackToDestModeSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // The full meaningfulness triple: FP32 DFB, consumed by a compute kernel with
    // enable_32_bit_dest=true. UnpackToDest is meaningful here.
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeHardwareConfig>(kernel.hw_config);
            config.enable_32_bit_dest = true;
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
