// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KernelSpec options that MakeProgramFromSpec forwards into the lowered kernel / DFB config.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestGen1, CPU_CompilerIncludePathsForwardedToKernelConfig) {
    // KernelSpec.compiler_options.include_paths should be picked up as `-I<path>` flags
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    const std::vector<std::filesystem::path> dm_paths = {"/tmp/dm_first", "/tmp/dm_second"};
    const std::vector<std::filesystem::path> compute_paths = {"/tmp/compute_only"};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    dm_kernel.compiler_options.include_paths = dm_paths;

    auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    compute_kernel.compiler_options.include_paths = compute_paths;

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "input_dfb"));
    compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "input_dfb"));

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel", "compute_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto& impl = program.impl();
    auto built_dm = impl.get_kernel_by_spec_name("dm_kernel");
    auto built_compute = impl.get_kernel_by_spec_name("compute_kernel");

    const auto built_dm_variant = built_dm->config();
    const auto& built_dm_config = std::get<DataMovementConfig>(built_dm_variant);
    EXPECT_EQ(built_dm_config.compiler_include_paths, dm_paths);

    const auto built_compute_variant = built_compute->config();
    const auto& built_compute_config = std::get<ComputeConfig>(built_compute_variant);
    EXPECT_EQ(built_compute_config.compiler_include_paths, compute_paths);
}

TEST_F(ProgramSpecTestGen1, CPU_SharedDedicatedNocIsBriscNoc) {
    // BRISC firmware issues its own NoC traffic on the launch message's brisc_noc_id. When the two DM
    // kernels share one dedicated NOC, that is the shared NOC, whichever kernel the lowering visits last.
    for (const NOC noc : {NOC::NOC_0, NOC::NOC_1}) {
        SCOPED_TRACE("noc=" + std::to_string(static_cast<int>(noc)));
        auto brisc_kernel = MakeMinimalGen1DMKernel("brisc_kernel", DataMovementProcessor::RISCV_0);
        auto ncrisc_kernel = MakeMinimalGen1DMKernel("ncrisc_kernel", DataMovementProcessor::RISCV_1);
        for (auto* kernel : {&brisc_kernel, &ncrisc_kernel}) {
            auto& cfg = (*std::get<DataMovementHardwareConfig>(kernel->hw_config).config_1xx);
            cfg.noc = noc;
            cfg.dedicated_noc_sharing = DedicatedNocSharing::SHARED_DISJOINT_TRAFFIC;
        }

        ProgramSpec spec;
        spec.name = "test_program";
        spec.kernels = {brisc_kernel, ncrisc_kernel};
        spec.work_units = std::vector<WorkUnitSpec>{
            MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"brisc_kernel", "ncrisc_kernel"})};
        Program program = MakeProgramFromSpec(*mesh_device_, spec);

        const uint32_t tensix_index = MetalContext::instance(program.impl().get_context_id())
                                          .hal()
                                          .get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const auto& kernel_groups = program.impl().get_kernel_groups(tensix_index);
        ASSERT_EQ(kernel_groups.size(), 1u);
        EXPECT_EQ(kernel_groups[0]->launch_msg.view().kernel_config().brisc_noc_id(), static_cast<uint8_t>(noc));
    }
}

// ----------------------------------------------------------------------------
// DFB implicit-sync opt-out (Gen2)
// ----------------------------------------------------------------------------
// Implicit sync is ON by default for any DFB side that has a DM endpoint. A DM kernel can
// opt out per-DFB (disable_dfb_implicit_sync_for) or for all the DFBs it binds at once
// (disable_dfb_implicit_sync_for_all). These tests pin the per-kernel "all" hammer.

TEST_F(ProgramSpecTestQuasar, CPU_DisableImplicitSyncForAllDisablesProducerSide) {
    // Build the canonical DM-producer -> compute-consumer DFB, optionally hammering the
    // producer's implicit sync off, and read back the lowered DataflowBufferConfig.
    auto make_spec = [](bool disable_all) {
        ProgramSpec spec;
        spec.name = "test_program";

        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        auto compute_kernel = MakeMinimalGen2ComputeKernel("compute_kernel");
        if (disable_all) {
            std::get<DataMovementHardwareConfig>(dm_kernel.hw_config).config_2xx =
                DataMovementHardwareConfig::DataMovement2XXConfig{.disable_dfb_implicit_sync_for_all = true};
        }

        auto dfb = MakeMinimalDFB("dfb_0");
        dfb.data_format_metadata = tt::DataFormat::Float16_b;

        dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out"));
        compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in"));

        spec.kernels = {dm_kernel, compute_kernel};
        spec.dataflow_buffers = {dfb};
        spec.work_units = {MakeMinimalWorkUnit("work_unit_0", NodeCoord{0, 0}, {"dm_kernel", "compute_kernel"})};
        return spec;
    };

    // Default: the producer side has a DM kernel, so implicit sync is on.
    {
        auto program = MakeProgramFromSpec(*mesh_device_, make_spec(/*disable_all=*/false));
        const uint32_t dfb_id = program.impl().get_dfb_handle("dfb_0");
        EXPECT_TRUE(program.impl().get_dataflow_buffer(dfb_id)->config.enable_producer_implicit_sync);
    }
    // disable_dfb_implicit_sync_for_all turns it off for every DFB the kernel binds.
    {
        auto program = MakeProgramFromSpec(*mesh_device_, make_spec(/*disable_all=*/true));
        const uint32_t dfb_id = program.impl().get_dfb_handle("dfb_0");
        EXPECT_FALSE(program.impl().get_dataflow_buffer(dfb_id)->config.enable_producer_implicit_sync);
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
