// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/dfb_placement_test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace test_helpers {

static uint32_t TensixIndex(const Program& program) {
    return MetalContext::instance(program.impl().get_context_id())
        .hal()
        .get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

}  // namespace test_helpers
namespace {

template <typename Arch>
class DFBPlacementRunArgsTest : public test_helpers::MockMeshDeviceFixture<Arch::value> {
protected:
    distributed::MeshDevice& fixture_device() { return *this->mesh_device_; }
};

using Architectures = ::testing::
    Types<std::integral_constant<tt::ARCH, tt::ARCH::WORMHOLE_B0>, std::integral_constant<tt::ARCH, tt::ARCH::QUASAR>>;
TYPED_TEST_SUITE(DFBPlacementRunArgsTest, Architectures);

TYPED_TEST(DFBPlacementRunArgsTest, CPU_DFBPlacementResizePreservesNodes) {
    // Increasing the DFB entry count preserves its disjoint reservation nodes.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    ProgramRunArgs args;
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"dfb_0"}, .num_entries = 8}};
    SetProgramRunArgs(program, args);
    program.impl().compile_and_allocate(&fixture_device, false);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({2, 0})));
    EXPECT_EQ(dfb->config.num_entries, 8u);
}

TYPED_TEST(DFBPlacementRunArgsTest, CPU_DFBPlacementRepeatedPreparationKeepsExcludedMaskClear) {
    // Repeated argument updates and program preparation keep the DFB disabled on excluded kernel nodes.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    for (int repeat = 0; repeat < 2; ++repeat) {
        SetProgramRunArgs(program, {});
        program.impl().compile_and_allocate(&fixture_device, false);
        test_helpers::FinalizeMockDFBConfig(program);
    }
    const auto* group = program.impl().kernels_on_core({1, 0}, test_helpers::TensixIndex(program));
    EXPECT_EQ(group->launch_msg.view().kernel_config().local_cb_mask(), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
