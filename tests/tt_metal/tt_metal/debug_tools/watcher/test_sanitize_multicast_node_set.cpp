// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Watcher check on a multicast into a sharded tensor: tensor data belongs only on the tensor's own nodes, so the
// multicast rectangle must stay within them. A rectangle over the tensor's nodes runs clean; one that reaches a node
// past them, or wraps around the grid edge, trips the check.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include "debug_tools_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;
using experimental::NodeCoord;
using experimental::NodeRange;
using experimental::NodeRangeSet;

const std::string kMulticastKernel =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_multicast_writer.cpp";
// Outside every rectangle below.
const NodeCoord kSenderOutside{0, 1};

// One 32x32 tile per node, height-sharded over `nodes`.
TensorSpec MakeTileShardedSpec(const NodeRangeSet& nodes) {
    return TensorSpec(
        Shape{32 * nodes.num_cores(), 32},
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::TILE),
            MemoryConfig{
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                ShardSpec{nodes, {32, 32}, ShardOrientation::ROW_MAJOR},
            }));
}

// The sender multicasts its one-tile `source` over the rectangle from logical node (start_x,0) to (end_x,0) on `noc`.
void RunMulticastNodeSetTest(
    distributed::MeshDevice& mesh_device,
    NodeCoord sender,
    uint32_t start_x,
    uint32_t end_x,
    NOC noc,
    bool expect_error,
    bool includes_sender = false) {
    if (MetalContext::instance().rtoptions().watcher_noc_sanitize_disabled()) {
        GTEST_SKIP();
    }
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    if (grid.x < 3 || grid.y < 2) {
        GTEST_SKIP() << "Needs a 3x2 worker grid";
    }
    auto& cq = mesh_device.mesh_command_queue();

    const TensorSpec source_spec = MakeTileShardedSpec(NodeRangeSet(sender));
    const TensorSpec target_spec = MakeTileShardedSpec(NodeRangeSet(NodeRange({0, 0}, {1, 0})));
    MeshTensor source = MeshTensor::allocate_on_device(mesh_device, source_spec);
    MeshTensor target = MeshTensor::allocate_on_device(mesh_device, target_spec);

    std::vector<bfloat16> data(source_spec.logical_shape().volume());
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = bfloat16(static_cast<float>(i));
    }
    cq.enqueue_write_tensor(HostTensor::from_vector(data, source_spec), source);
    // Clear the target, so the in-range cases can't pass on data a previous run left at the same address.
    cq.enqueue_write_tensor(
        HostTensor::from_vector(
            std::vector<bfloat16>(target_spec.logical_shape().volume(), bfloat16(0.0f)), target_spec),
        target);

    // `noc` runs from start to end: NoC 0 rightward, NoC 1 leftward. Corners against its direction wrap around the grid
    // edge.
    const CoreCoord rect_start = mesh_device.worker_core_from_logical_core({start_x, 0});
    const CoreCoord rect_end = mesh_device.worker_core_from_logical_core({end_x, 0});
    // The rectangle's nodes, less a sender inside it unless the multicast is a loopback one. A wrapping rectangle stops
    // in the check before the multicast is issued.
    const uint32_t first_x = std::min(start_x, end_x);
    const uint32_t last_x = std::max(start_x, end_x);
    const bool skips_sender = !includes_sender && sender.y == 0 && sender.x >= first_x && sender.x <= last_x;
    const uint32_t num_targets = last_x - first_x + 1 - (skips_sender ? 1u : 0u);
    const experimental::KernelSpecName kernel_name{"multicast_writer"};
    Program program = experimental::MakeProgramFromSpec(
        mesh_device,
        experimental::ProgramSpec{
            .name = "watcher_multicast_node_set",
            .kernels = {experimental::KernelSpec{
                .unique_id = kernel_name,
                .source = kMulticastKernel,
                .tensor_bindings =
                    {
                        {.tensor_parameter_name = experimental::TensorParamName{"source"}, .accessor_name = "source"},
                        {.tensor_parameter_name = experimental::TensorParamName{"target"}, .accessor_name = "target"},
                    },
                .compile_time_args =
                    {
                        {"noc_x_start", static_cast<uint32_t>(rect_start.x)},
                        {"noc_y_start", static_cast<uint32_t>(rect_start.y)},
                        {"noc_x_end", static_cast<uint32_t>(rect_end.x)},
                        {"noc_y_end", static_cast<uint32_t>(rect_end.y)},
                        {"num_targets", num_targets},
                        {"size_bytes", static_cast<uint32_t>(source_spec.compute_page_size_bytes())},
                        // A sender inside the rectangle is skipped unless this makes the multicast a loopback one.
                        {"includes_sender", static_cast<uint32_t>(includes_sender)},
                    },
                .hw_config =
                    experimental::DataMovementHardwareConfig{
                        .config_1xx =
                            experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                                .processor =
                                    noc == NOC::NOC_0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
                                .noc = noc,
                            },
                    },
            }},
            .tensor_parameters =
                {
                    {.unique_id = experimental::TensorParamName{"source"}, .spec = source_spec},
                    {.unique_id = experimental::TensorParamName{"target"}, .spec = target_spec},
                },
            .work_units = {experimental::WorkUnitSpec{
                .name = "main",
                .kernels = {kernel_name},
                .target_nodes = sender,
            }},
        });
    experimental::SetProgramRunArgs(
        program,
        experimental::ProgramRunArgs{
            .tensor_args =
                {
                    {experimental::TensorParamName{"source"}, experimental::TensorArgument{source}},
                    {experimental::TensorParamName{"target"}, experimental::TensorArgument{target}},
                },
        });
    if (!expect_error) {
        LaunchProgram(mesh_device, std::move(program));
        std::vector<bfloat16> expected = data;
        expected.insert(expected.end(), data.begin(), data.end());
        EXPECT_EQ(cq.enqueue_read_tensor(target).to_vector<bfloat16>(), expected);
        return;
    }
    // On a Watcher error the kernel hangs in the check, so the blocking launch aborts with the Watcher report.
    try {
        LaunchProgram(mesh_device, std::move(program));
        ADD_FAILURE() << "Watcher did not stop the multicast";
    } catch (const std::runtime_error& exception) {
        EXPECT_THAT(
            exception.what(), ::testing::HasSubstr("(multicast rectangle reaches nodes outside the destination)."));
    }
}

}  // namespace

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastWithinShardedTensor) {
    RunMulticastNodeSetTest(
        *this->devices_[0], kSenderOutside, /*start_x=*/0, /*end_x=*/1, NOC::NOC_0, /*expect_error=*/false);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastWithinShardedTensorNoc1) {
    RunMulticastNodeSetTest(
        *this->devices_[0], kSenderOutside, /*start_x=*/1, /*end_x=*/0, NOC::NOC_1, /*expect_error=*/false);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastPastShardedTensor) {
    RunMulticastNodeSetTest(
        *this->devices_[0], kSenderOutside, /*start_x=*/0, /*end_x=*/2, NOC::NOC_0, /*expect_error=*/true);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastWrapAroundShardedTensor) {
    RunMulticastNodeSetTest(
        *this->devices_[0], kSenderOutside, /*start_x=*/1, /*end_x=*/0, NOC::NOC_0, /*expect_error=*/true);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastSenderInsideRectangle) {
    // The sender on (2,0) holds no tile of `target`, but the multicast skips it.
    RunMulticastNodeSetTest(
        *this->devices_[0], /*sender=*/{2, 0}, /*start_x=*/0, /*end_x=*/2, NOC::NOC_0, /*expect_error=*/false);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastLoopbackSenderInsideRectangle) {
    // As above, but the loopback multicast also writes the sender, which holds no tile of `target`.
    RunMulticastNodeSetTest(
        *this->devices_[0],
        /*sender=*/{2, 0},
        /*start_x=*/0,
        /*end_x=*/2,
        NOC::NOC_0,
        /*expect_error=*/true,
        /*includes_sender=*/true);
}
