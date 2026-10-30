// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Watcher check on a multicast into a DFB or semaphore: its address is reserved only on the nodes it exists on, or for
// a borrowed-memory DFB on its backing tensor's nodes, so the multicast rectangle must stay within them. A rectangle
// over those nodes runs clean; one that reaches a node past them trips the check.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
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

const std::string kKernelDirectory = "tests/tt_metal/tt_metal/test_kernels/dataflow/";
// The resource exists on logical nodes (0,0) and (1,0), and on the sender (0,1), which binds it.
const NodeCoord kSender{0, 1};
const NodeRangeSet kResourceNodes(std::vector<NodeRange>{NodeRange({0, 0}, {1, 0}), NodeRange(kSender)});
// A borrowed-memory DFB's backing tensor also covers logical node (2,0), past the DFB's own nodes.
const NodeRangeSet kBackingNodes(std::vector<NodeRange>{NodeRange({0, 0}, {2, 0}), NodeRange(kSender)});

experimental::DataMovementHardwareConfig DataMovementConfig(DataMovementProcessor processor, NOC noc) {
    return {
        .config_1xx =
            experimental::DataMovementHardwareConfig::DataMovement1XXConfig{.processor = processor, .noc = noc},
    };
}

// Skips unless Watcher sanitizes NoC transactions and the grid has the nodes the tests use.
bool ShouldSkip(distributed::MeshDevice& mesh_device) {
    const NodeCoord grid = mesh_device.compute_with_storage_grid_size();
    return MetalContext::instance().rtoptions().watcher_noc_sanitize_disabled() || grid.x < 4 || grid.y < 2;
}

// One 32x32 tile per node, height-sharded over `nodes`: as large as the DFB's two entries.
TensorSpec MakeBackingSpec(const NodeRangeSet& nodes) {
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

// Launches the program; with expect_error, expects Watcher to stop the multicast instead.
void LaunchAndCheck(distributed::MeshDevice& mesh_device, Program program, bool expect_error) {
    if (!expect_error) {
        LaunchProgram(mesh_device, std::move(program));
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

// The sender sets the semaphore over the rectangle from logical node (0,0) to (end_x,0).
void RunSemaphoreMulticastTest(distributed::MeshDevice& mesh_device, uint32_t end_x, bool expect_error) {
    if (ShouldSkip(mesh_device)) {
        GTEST_SKIP();
    }
    const NodeCoord rect_start = mesh_device.worker_core_from_logical_core({0, 0});
    const NodeCoord rect_end = mesh_device.worker_core_from_logical_core({end_x, 0});
    const experimental::KernelSpecName kernel_name{"semaphore_multicast_writer"};
    const experimental::SemaphoreSpecName semaphore_name{"flag"};
    const experimental::ProgramSpec spec{
        .name = "watcher_multicast_semaphore",
        .kernels = {experimental::KernelSpec{
            .unique_id = kernel_name,
            .source = kKernelDirectory + "semaphore_multicast_writer.cpp",
            .semaphore_bindings = {{.semaphore_spec_name = semaphore_name, .accessor_name = "flag"}},
            .compile_time_args =
                {
                    {"noc_x_start", static_cast<uint32_t>(rect_start.x)},
                    {"noc_y_start", static_cast<uint32_t>(rect_start.y)},
                    {"noc_x_end", static_cast<uint32_t>(rect_end.x)},
                    {"noc_y_end", static_cast<uint32_t>(rect_end.y)},
                    // Every node of the rectangle, which excludes the sender.
                    {"num_targets", end_x + 1},
                },
            .hw_config = DataMovementConfig(DataMovementProcessor::RISCV_0, NOC::NOC_0),
        }},
        .semaphores = {experimental::SemaphoreSpec{.unique_id = semaphore_name, .target_nodes = kResourceNodes}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {kernel_name},
            .target_nodes = kSender,
        }},
    };
    LaunchAndCheck(mesh_device, experimental::MakeProgramFromSpec(mesh_device, spec), expect_error);
}

// The sender multicasts one entry of its DFB over the rectangle from logical node (0,0) to (end_x,0). The DFB's
// producer and consumer kernels run on all its nodes, which is what places the DFB there. With `backing_nodes`, the DFB
// borrows the memory of a tensor sharded over them.
void RunDataflowBufferMulticastTest(
    distributed::MeshDevice& mesh_device,
    uint32_t end_x,
    bool expect_error,
    const std::optional<NodeRangeSet>& backing_nodes = std::nullopt) {
    if (ShouldSkip(mesh_device)) {
        GTEST_SKIP();
    }
    const NodeCoord sender = mesh_device.worker_core_from_logical_core(kSender);
    const NodeCoord rect_start = mesh_device.worker_core_from_logical_core({0, 0});
    const NodeCoord rect_end = mesh_device.worker_core_from_logical_core({end_x, 0});
    const experimental::KernelSpecName producer_name{"dataflow_buffer_multicast_writer"};
    const experimental::KernelSpecName consumer_name{"idle_consumer"};
    const experimental::DFBSpecName dataflow_buffer_name{"target"};
    const experimental::TensorParamName backing_name{"backing"};
    constexpr uint32_t entry_size = 1024;
    experimental::ProgramSpec spec{
        .name = "watcher_multicast_dataflow_buffer",
        .kernels =
            {
                experimental::KernelSpec{
                    .unique_id = producer_name,
                    .source = kKernelDirectory + "dataflow_buffer_multicast_writer.cpp",
                    .dfb_bindings = {experimental::ProducerOf(dataflow_buffer_name, "target")},
                    .compile_time_args =
                        {
                            {"sender_x", static_cast<uint32_t>(sender.x)},
                            {"sender_y", static_cast<uint32_t>(sender.y)},
                            {"noc_x_start", static_cast<uint32_t>(rect_start.x)},
                            {"noc_y_start", static_cast<uint32_t>(rect_start.y)},
                            {"noc_x_end", static_cast<uint32_t>(rect_end.x)},
                            {"noc_y_end", static_cast<uint32_t>(rect_end.y)},
                            // Every node of the rectangle, which excludes the sender.
                            {"num_targets", end_x + 1},
                            {"size_bytes", entry_size},
                        },
                    .hw_config = DataMovementConfig(DataMovementProcessor::RISCV_0, NOC::NOC_0),
                },
                experimental::KernelSpec{
                    .unique_id = consumer_name,
                    .source = kKernelDirectory + "blank.cpp",
                    .dfb_bindings = {experimental::ConsumerOf(dataflow_buffer_name, "target")},
                    .hw_config = DataMovementConfig(DataMovementProcessor::RISCV_1, NOC::NOC_1),
                },
            },
        .dataflow_buffers = {experimental::DataflowBufferSpec{
            .unique_id = dataflow_buffer_name,
            .entry_size = entry_size,
            .num_entries = 2,
            .data_format_metadata = tt::DataFormat::Float16_b,
        }},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {producer_name, consumer_name},
            .target_nodes = kResourceNodes,
        }},
    };
    if (!backing_nodes.has_value()) {
        LaunchAndCheck(mesh_device, experimental::MakeProgramFromSpec(mesh_device, spec), expect_error);
        return;
    }
    const TensorSpec backing_spec = MakeBackingSpec(*backing_nodes);
    spec.dataflow_buffers[0].borrowed_from = backing_name;
    spec.tensor_parameters = {{.unique_id = backing_name, .spec = backing_spec}};
    const MeshTensor backing = MeshTensor::allocate_on_device(mesh_device, backing_spec);
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::SetProgramRunArgs(
        program,
        experimental::ProgramRunArgs{
            .tensor_args = {{backing_name, experimental::TensorArgument{backing}}},
        });
    LaunchAndCheck(mesh_device, std::move(program), expect_error);
}

}  // namespace

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastWithinSemaphore) {
    RunSemaphoreMulticastTest(*this->devices_[0], /*end_x=*/1, /*expect_error=*/false);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastPastSemaphore) {
    RunSemaphoreMulticastTest(*this->devices_[0], /*end_x=*/2, /*expect_error=*/true);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastWithinDataflowBuffer) {
    RunDataflowBufferMulticastTest(*this->devices_[0], /*end_x=*/1, /*expect_error=*/false);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastPastDataflowBuffer) {
    RunDataflowBufferMulticastTest(*this->devices_[0], /*end_x=*/2, /*expect_error=*/true);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastWithinBorrowedDataflowBuffer) {
    RunDataflowBufferMulticastTest(*this->devices_[0], /*end_x=*/2, /*expect_error=*/false, kBackingNodes);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastPastBorrowedDataflowBuffer) {
    RunDataflowBufferMulticastTest(*this->devices_[0], /*end_x=*/3, /*expect_error=*/true, kBackingNodes);
}
