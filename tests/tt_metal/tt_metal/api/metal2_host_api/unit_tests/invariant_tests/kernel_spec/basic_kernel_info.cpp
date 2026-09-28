// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec basic info (kernel_spec.hpp): num_threads per architecture and kernel kind, source.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_KernelWithZeroThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel", 0);  // 0 threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("KernelSpec 'kernel' has no threads")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DMKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Quasar has 8 DM cores per node (we reserve 2 for internal use)
    auto kernel = MakeMinimalGen2DMKernel("kernel", 9);  // Too many threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("KernelSpec 'kernel' has too many data movement threads")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ComputeKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Quasar has 4 Tensix cores per node
    auto kernel = MakeMinimalGen2ComputeKernel("kernel", 5);  // Too many threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "KernelSpec 'kernel' has too many threads. The architecture supports up to 4 for compute kernels")));
}

TEST_F(ProgramSpecTestQuasar, CPU_MaxComputeThreadsSucceeds) {
    // Use exactly 4 compute threads (the maximum available)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "max_compute_threads";

    auto dm = MakeMinimalGen2DMKernel("dm");
    auto compute = MakeMinimalGen2ComputeKernel("compute", 4);  // Max threads

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dfb.num_entries = 4;  // must be a multiple of the number of threads

    dm.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {dm, compute};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm", "compute"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_MultiThreadedDMKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    kernel.num_threads = 2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not support multi-threaded kernels")));
}

TEST_F(ProgramSpecTestGen1, CPU_MultiThreadedComputeKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    kernel.num_threads = 2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not support multi-threaded kernels")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SourceCodeKernelSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Change to inline source code
    spec.kernels[0].source = KernelSpec::SourceCode{"void kernel_main() {}"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
