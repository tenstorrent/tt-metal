// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/dfb_placement_test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

template <typename Arch>
class DFBPlacementRunArgsTest : public test_helpers::MockMeshDeviceFixture<Arch::value> {
protected:
    distributed::MeshDevice& fixture_device() { return *this->mesh_device_; }
};

using Architectures = ::testing::
    Types<std::integral_constant<tt::ARCH, tt::ARCH::WORMHOLE_B0>, std::integral_constant<tt::ARCH, tt::ARCH::QUASAR>>;
TYPED_TEST_SUITE(DFBPlacementRunArgsTest, Architectures);

TYPED_TEST(DFBPlacementRunArgsTest, CPU_DFBPlacementResizeReservesLargerRegion) {
    // Growing a DFB moves the next allocation beyond its new size on the selected node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value, 2);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    ProgramRunArgs args{.dfb_run_overrides = {{.dfb = DFBSpecName{"dfb_0"}, .num_entries = 8}}};
    SetProgramRunArgs(program, args);
    program.impl().compile_and_allocate(&fixture_device, false);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto next = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_GE(next->uniform_alloc_addr(), first->uniform_alloc_addr() + 8 * spec.dataflow_buffers[0].entry_size);
}

TYPED_TEST(DFBPlacementRunArgsTest, CPU_DFBPlacementResizeKeepsExcludedMemoryAvailable) {
    // Growing a DFB preserves independent backing allocations on excluded nodes.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDisjointDFBPlacementsForTest(TypeParam::value);
    auto program = MakeProgramFromSpec(fixture_device, spec);
    ProgramRunArgs args{.dfb_run_overrides = {{.dfb = DFBSpecName{"dfb_0"}, .num_entries = 8}}};
    SetProgramRunArgs(program, args);
    program.impl().compile_and_allocate(&fixture_device, false);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(first->uniform_alloc_addr(), second->uniform_alloc_addr());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
