// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke test for fabric connection bindings (mock 2-chip Wormhole with 1D fabric).

#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecFabricTestGen1;

TEST_F(ProgramSpecFabricTestGen1, CPU_FabricConnectionBindingJITSmoke) {
    // A DM kernel builds a fabric sender from its binding token, then opens and closes the connection.
    KernelSpec writer = MakeMinimalGen1DMKernel("writer");
    writer.source = KernelSpec::SourceCode{R"(
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
void kernel_main() {
    if (tt::tt_fabric::has_fabric_connection(fabric::fwd)) {
        auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::fwd);
        sender.open();
        sender.close();
    }
}
)"};
    writer.advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "fwd"}};
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{"fwd"}, .src_fabric_node_id = fixture_source_fabric_node_id()};
    connection.targets[NodeCoord{0, 0}] = {{.dst_fabric_node_id = fixture_destination_fabric_node_id()}};
    ProgramSpec spec{.kernels = {writer}, .work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})}};
    spec.advanced_options.fabric_connections = {connection};

    Program program = MakeProgramFromSpec(fixture_source_device(), spec);
    EXPECT_NO_THROW(program.impl().compile(&fixture_source_device()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
