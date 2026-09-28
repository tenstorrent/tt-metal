// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariant on Gen1 data-movement placement (program_spec.hpp): within a
// WorkUnitSpec, DM kernels use distinct processors, share noc_mode, and DM_DEDICATED_NOC kernels use
// distinct NOCs.

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
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalReaderDMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeMinimalWriterDMKernel;
using test_helpers::ProgramSpecTestGen1;

TEST_F(ProgramSpecTestGen1, CPU_DMOnlyProgramSucceeds) {
    // Two DM kernels on different processors (RISCV_0 producer, RISCV_1 consumer)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "dm_only_program";

    auto producer = MakeMinimalWriterDMKernel("producer");
    auto consumer = MakeMinimalReaderDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_TwoDMKernelsDifferentProcessorsSucceeds) {
    // RISCV_0 and RISCV_1 on the same node — should succeed. (MakeMinimalGen1DMKernel gives them
    // distinct NOCs, so they also satisfy the dedicated-NOC distinctness rule.)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "two_dm_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_ProcessorConflictFails) {
    // Two DM kernels both targeting RISCV_0 on the same node
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_0);  // conflict

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("both claim the same DM processor")));
}

TEST_F(ProgramSpecTestGen1, CPU_TwoDMKernelsSameNocDedicatedFails) {
    // Two DM kernels on distinct processors (RISCV_0, RISCV_1) but pinned to the SAME NOC in
    // dedicated mode. Each kernel's NoC traffic is statically compiled to its config.noc, so both
    // would drive NOC_0 and hang the device. Validation must reject this.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    // Force both onto NOC_0 (the helper would otherwise assign complementary NOCs). noc_mode
    // defaults to DM_DEDICATED_NOC.
    (*std::get<DataMovementHardwareConfig>(k0.hw_config).config_1xx).noc = NOC::NOC_0;
    (*std::get<DataMovementHardwareConfig>(k1.hw_config).config_1xx).noc = NOC::NOC_0;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("pinned to NOC_0")));
}

TEST_F(ProgramSpecTestGen1, CPU_TwoDMKernelsDistinctNocDedicatedSucceeds) {
    // Two dedicated-NOC DM kernels on distinct processors AND distinct NOCs — the correct pairing.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    (*std::get<DataMovementHardwareConfig>(k0.hw_config).config_1xx).noc = NOC::NOC_0;
    (*std::get<DataMovementHardwareConfig>(k1.hw_config).config_1xx).noc = NOC::NOC_1;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_TwoDMKernelsSameNocDynamicSucceeds) {
    // In DM_DYNAMIC_NOC mode, two DM kernels may intentionally share a NOC (it frees the other NOC
    // for fabric). The NOC-distinctness rule is dedicated-mode only, so this is accepted even though
    // both kernels name NOC_0.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    auto& cfg0 = (*std::get<DataMovementHardwareConfig>(k0.hw_config).config_1xx);
    cfg0.noc = NOC::NOC_0;
    cfg0.noc_mode = NOC_MODE::DM_DYNAMIC_NOC;
    auto& cfg1 = (*std::get<DataMovementHardwareConfig>(k1.hw_config).config_1xx);
    cfg1.noc = NOC::NOC_0;
    cfg1.noc_mode = NOC_MODE::DM_DYNAMIC_NOC;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_TwoDMKernelsMixedNocModeFails) {
    // NOC mode configures shared per-core NOC hardware (and is compiled into each kernel binary), so
    // two DM kernels on the same node must agree on it. One dedicated + one dynamic is incoherent.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Distinct processors and distinct NOCs (helper defaults: RISCV_0->NOC_0, RISCV_1->NOC_1), so
    // neither the processor nor the NOC-distinctness check fires — only the mode disagreement trips.
    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    (*std::get<DataMovementHardwareConfig>(k0.hw_config).config_1xx).noc_mode = NOC_MODE::DM_DEDICATED_NOC;
    (*std::get<DataMovementHardwareConfig>(k1.hw_config).config_1xx).noc_mode = NOC_MODE::DM_DYNAMIC_NOC;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("different NOC modes")));
}

TEST_F(ProgramSpecTestGen1, CPU_DMKernelsSameProcessorAndNocOnDistinctNodesSucceeds) {
    // Node-scoping guard: two DM kernels with identical processor (RISCV_0), NOC (NOC_0), and
    // DM_DEDICATED_NOC mode are legal when placed on DISTINCT nodes — the processor- and
    // NOC-distinctness censuses are per-node. This would wrongly fail if either map were keyed
    // without the node component.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k_a = MakeMinimalGen1DMKernel("k_a", DataMovementProcessor::RISCV_0);  // NOC_0, dedicated
    auto k_b = MakeMinimalGen1DMKernel("k_b", DataMovementProcessor::RISCV_0);  // NOC_0, dedicated

    spec.kernels = {k_a, k_b};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node_a, {"k_a"}),
        MakeMinimalWorkUnit("wu_b", node_b, {"k_b"}),
    };

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_DMKernelsDifferentNocModesOnDistinctNodesSucceeds) {
    // Node-scoping guard for NOC-mode agreement: a DM_DEDICATED_NOC kernel on one node and a
    // DM_DYNAMIC_NOC kernel on another are legal — agreement is enforced per-node, not globally.
    // This would wrongly fail if node_noc_mode were keyed without the node component.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k_a = MakeMinimalGen1DMKernel("k_a", DataMovementProcessor::RISCV_0);
    auto k_b = MakeMinimalGen1DMKernel("k_b", DataMovementProcessor::RISCV_0);
    (*std::get<DataMovementHardwareConfig>(k_a.hw_config).config_1xx).noc_mode = NOC_MODE::DM_DEDICATED_NOC;
    (*std::get<DataMovementHardwareConfig>(k_b.hw_config).config_1xx).noc_mode = NOC_MODE::DM_DEDICATED_NOC;

    spec.kernels = {k_a, k_b};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node_a, {"k_a"}),
        MakeMinimalWorkUnit("wu_b", node_b, {"k_b"}),
    };

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_ReaderAndWriterRolesOnSameNodeSucceed) {
    // A READER and a WRITER role resolve to distinct processors (RISCV_1 and RISCV_0
    // respectively), so two role-driven DM kernels coexist on one node without conflict.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto reader = MakeMinimalReaderDMKernel("reader");
    auto writer = MakeMinimalWriterDMKernel("writer");

    spec.kernels = {reader, writer};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"reader", "writer"})};

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestGen1, CPU_TwoReaderRolesOnSameNodeConflict) {
    // Both READER kernels resolve to the same processor (RISCV_1), so placing them on the
    // same node is a conflict — confirming the role hint resolves to a fixed, deterministic
    // processor (the same uniqueness rule as explicit configs).
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto r0 = MakeMinimalReaderDMKernel("r0");
    auto r1 = MakeMinimalReaderDMKernel("r1");

    spec.kernels = {r0, r1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"r0", "r1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("both claim the same DM processor")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
