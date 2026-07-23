// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hal.hpp"
#include "llrt/rtoptions.hpp"
#include <algorithm>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarMultiSemaphorePipeline) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    const experimental::NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr uint32_t num_elements = 10;
    const uint32_t buf_a_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);
    const uint32_t dram_src_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dram_dst_addr = dram_src_addr + (1000 * 1024);

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_src_addr, initial_data);

    const experimental::KernelSpecName DM_READER{"dm_reader"};
    const experimental::KernelSpecName DM_TRANSFORM{"dm_transform"};
    const experimental::KernelSpecName DM_WRITER{"dm_writer"};

    experimental::SemaphoreSpec sem0_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem0"},
        .target_nodes = node,
    };
    experimental::SemaphoreSpec sem1_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem1"},
        .target_nodes = node,
    };

    experimental::KernelSpec dm_reader_spec{
        .unique_id = DM_READER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_transform_spec{
        .unique_id = DM_TRANSFORM,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}, {"INCOMING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0"}, .accessor_name = "sem_in"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1"}, .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_writer_spec{
        .unique_id = DM_WRITER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_READER, DM_TRANSFORM, DM_WRITER},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "multi_semaphore_pipeline",
        .kernels = {dm_reader_spec, dm_transform_spec, dm_writer_spec},
        .semaphores = {sem0_spec, sem1_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dram_addr", dram_src_addr},
                 {"l1_addr", buf_a_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dram_addr", dram_dst_addr},
                 {"l1_addr", buf_b_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(
        mesh_device->get_devices()[0], 0, dram_dst_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    ASSERT_EQ(actual_data, expected_data);
}

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarMultipleClustersMultiSemaphorePipeline) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];

    if (mesh_device->compute_with_storage_grid_size().x < 2) {
        GTEST_SKIP() << "This test requires at least 2 worker nodes.";
    }

    const experimental::NodeCoord node_0{0, 0};
    const experimental::NodeCoord node_1{1, 0};

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr uint32_t num_elements = 10;
    const uint32_t buf_a_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);
    const uint32_t dram_mid_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dram_dst_addr = dram_mid_addr + (1000 * 1024);

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], node_0, buf_a_addr, initial_data);

    const CoreCoord core_1_virtual = mesh_device->worker_core_from_logical_core(node_1);

    experimental::SemaphoreSpec sem_core_0_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem_core_0"},
        .target_nodes = node_0,
    };
    experimental::SemaphoreSpec sem_cross_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem_cross"},
        .target_nodes = experimental::NodeRange{node_0, node_1},
    };
    experimental::SemaphoreSpec sem0_core_1_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem0_core_1"},
        .target_nodes = node_1,
    };
    experimental::SemaphoreSpec sem1_core_1_spec{
        .unique_id = experimental::SemaphoreSpecName{"sem1_core_1"},
        .target_nodes = node_1,
    };

    const experimental::KernelSpecName DM_TRANSFORM_0{"dm_transform_0"};
    const experimental::KernelSpecName DM_WRITER_0{"dm_writer_0"};
    const experimental::KernelSpecName DM_READER_1{"dm_reader_1"};
    const experimental::KernelSpecName DM_TRANSFORM_1{"dm_transform_1"};
    const experimental::KernelSpecName DM_WRITER_1{"dm_writer_1"};

    experimental::KernelSpec dm_transform_0_spec{
        .unique_id = DM_TRANSFORM_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}}},
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_core_0"}, .accessor_name = "sem_out"}},
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_writer_0_spec{
        .unique_id = DM_WRITER_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCREMENT_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_core_0"}, .accessor_name = "sem"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_cross"}, .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"dram_addr", "l1_addr", "num_elements", "dram_bank_id", "remote_noc_x", "remote_noc_y"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_reader_1_spec{
        .unique_id = DM_READER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"WAIT_FOR_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0_core_1"}, .accessor_name = "sem"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_cross"}, .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_transform_1_spec{
        .unique_id = DM_TRANSFORM_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCOMING_SEM", "1"}, {"OUTGOING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem0_core_1"}, .accessor_name = "sem_in"},
                {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1_core_1"}, .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec dm_writer_1_spec{
        .unique_id = DM_WRITER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem1_core_1"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::WorkUnitSpec wu_core_0{
        .name = "wu_core_0",
        .kernels = {DM_TRANSFORM_0, DM_WRITER_0},
        .target_nodes = node_0,
    };
    experimental::WorkUnitSpec wu_core_1{
        .name = "wu_core_1",
        .kernels = {DM_READER_1, DM_TRANSFORM_1, DM_WRITER_1},
        .target_nodes = node_1,
    };

    experimental::ProgramSpec spec{
        .name = "multi_cluster_multi_semaphore_pipeline",
        .kernels = {dm_transform_0_spec, dm_writer_0_spec, dm_reader_1_spec, dm_transform_1_spec, dm_writer_1_spec},
        .semaphores = {sem_core_0_spec, sem_cross_spec, sem0_core_1_spec, sem1_core_1_spec},
        .work_units = {wu_core_0, wu_core_1},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM_0},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM_1},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER_0,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node_0,
                {{"dram_addr", dram_mid_addr},
                 {"l1_addr", buf_b_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u},
                 {"remote_noc_x", static_cast<uint32_t>(core_1_virtual.x)},
                 {"remote_noc_y", static_cast<uint32_t>(core_1_virtual.y)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_READER_1,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node_1,
                {{"dram_addr", dram_mid_addr},
                 {"l1_addr", buf_a_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER_1,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node_1,
                {{"dram_addr", dram_dst_addr},
                 {"l1_addr", buf_b_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(
        mesh_device->get_devices()[0], 0, dram_dst_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    ASSERT_EQ(actual_data, expected_data);
}

// ---------------------------------------------------------------------------
// L3: semaphore + DRAM-staged SNAKE CHAIN across an arbitrary ordered node list.
// Data flows node[0] -> node[1] -> ... -> node[N-1]. Each hop: node[i]'s writer
// stages its (transformed) buffer to DRAM stage[i+1] and increments node[i+1]'s
// cross-node semaphore; node[i+1]'s reader waits on it, reads stage[i+1], transforms
// (+1), and relays onward. Final DRAM stage = seed + N. Proves every node in the
// list can receive-from-prev and send-to-next. Generalizes QuasarMultipleClusters...
// (the proven 2-node cross-node pipeline) to N nodes.
// ---------------------------------------------------------------------------
static void run_snake_chain(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<experimental::NodeCoord>& nodes,
    bool ring = false,
    uint32_t num_elements = 10) {
    using experimental::KernelSpec;
    using experimental::KernelSpecName;
    using experimental::SemaphoreSpec;
    using experimental::SemaphoreSpecName;
    using experimental::WorkUnitSpec;
    const uint32_t N = static_cast<uint32_t>(nodes.size());
    ASSERT_GE(N, 2u);
    IDevice* dev = mesh_device->get_devices()[0];

    const uint32_t buf_a = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t buf_b = buf_a + num_elements * sizeof(uint32_t);
    const uint32_t buf_wrap = buf_b + num_elements * sizeof(uint32_t);  // ring: node[0] receives the wrapped token here
    const SemaphoreSpecName WRAP{"wrap_sem"};                           // ring: node[N-1] -> node[0] closing edge
    const SemaphoreSpecName WRAP_L0{"wrap_l0"};                         // ring: local sem for node[0]'s wrap reader
    const uint32_t dram_base = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    constexpr uint32_t stage_stride = 4096;  // per-hop DRAM staging spacing
    auto stage_addr = [&](uint32_t i) { return dram_base + i * stage_stride; };
    // Each hop stages num_elements words at stage_addr(i); guard against a payload that would
    // overflow one stage into the next (stages are stage_stride bytes apart).
    ASSERT_LE(num_elements * sizeof(uint32_t), stage_stride);

    // Seed stage 0 with [0..num_elements).
    std::vector<uint32_t> seed(num_elements);
    for (uint32_t i = 0; i < num_elements; ++i) {
        seed[i] = i;
    }
    tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, stage_addr(0), seed);
    // Sentinel the OUTPUT stage so a stale-data pass is impossible: callers reuse identical
    // stage addresses across runs, so the final EXPECT_EQ must be forced to observe fresh data
    // written by THIS run's chain (not leftover values from a prior same-N run).
    std::vector<uint32_t> out_sentinel(num_elements, 0xdeadbeefu);
    tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, stage_addr(N), out_sentinel);
    // Ring mode additionally verifies the wrapped token in node[0]'s buf_wrap; sentinel it too so
    // the ring-edge assertion cannot pass on stale/uninitialized data.
    if (ring) {
        tt_metal::detail::WriteToDeviceL1(dev, nodes[0], buf_wrap, out_sentinel);
    }

    experimental::ProgramSpec spec{.name = "snake_chain"};
    experimental::ProgramRunArgs params;

    for (uint32_t i = 0; i < N; ++i) {
        const bool is_src = (i == 0);
        const bool is_sink = (i == N - 1);
        const std::string si = std::to_string(i);
        const KernelSpecName READER{"reader_" + si};
        const KernelSpecName XFORM{"xform_" + si};
        const KernelSpecName WRITER{"writer_" + si};
        const SemaphoreSpecName L0{"l0_" + si};
        const SemaphoreSpecName L1{"l1_" + si};
        const SemaphoreSpecName CROSS_IN{"cross_" + std::to_string(i == 0 ? 0 : i - 1)};  // hop (i-1)->i
        const SemaphoreSpecName CROSS_OUT{"cross_" + si};                                 // hop i->(i+1)

        // Local intra-node sems (reader->xform, xform->writer).
        spec.semaphores.push_back(SemaphoreSpec{.unique_id = L0, .target_nodes = nodes[i]});
        spec.semaphores.push_back(SemaphoreSpec{.unique_id = L1, .target_nodes = nodes[i]});
        // Cross-node sem for the hop i->i+1, allocated on both endpoints (same L1 addr).
        // NodeRange is a bounding box (end >= start), so order the two adjacent coords by
        // componentwise min/max — the snake's right->left rows have nodes[i].x > nodes[i+1].x.
        if (!is_sink) {
            const auto a = nodes[i];
            const auto b = nodes[i + 1];
            const experimental::NodeCoord lo{std::min(a.x, b.x), std::min(a.y, b.y)};
            const experimental::NodeCoord hi{std::max(a.x, b.x), std::max(a.y, b.y)};
            spec.semaphores.push_back(
                SemaphoreSpec{.unique_id = CROSS_OUT, .target_nodes = experimental::NodeRange{lo, hi}});
        } else if (ring) {
            // Wrap edge node[N-1] -> node[0] (may be a non-adjacent long hop; bbox may span a column).
            const auto a = nodes[i];
            const auto b = nodes[0];
            const experimental::NodeCoord lo{std::min(a.x, b.x), std::min(a.y, b.y)};
            const experimental::NodeCoord hi{std::max(a.x, b.x), std::max(a.y, b.y)};
            spec.semaphores.push_back(
                SemaphoreSpec{.unique_id = WRAP, .target_nodes = experimental::NodeRange{lo, hi}});
        }

        // Reader: source reads without waiting; others wait on the incoming cross sem.
        KernelSpec reader{
            .unique_id = READER,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
            .hw_config = experimental::DataMovementGen2Config{}};
        if (is_src) {
            reader.semaphore_bindings = {{.semaphore_spec_name = L0, .accessor_name = "sem"}};
        } else {
            reader.compiler_options = {.defines = {{"WAIT_FOR_REMOTE_SEM", "1"}}};
            reader.semaphore_bindings = {
                {.semaphore_spec_name = L0, .accessor_name = "sem"},
                {.semaphore_spec_name = CROSS_IN, .accessor_name = "remote_sem"}};
        }
        spec.kernels.push_back(reader);

        // Transform: +1 per element, buf_a -> buf_b.
        spec.kernels.push_back(KernelSpec{
            .unique_id = XFORM,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
            .num_threads = 1,
            .compiler_options = {.defines = {{"INCOMING_SEM", "1"}, {"OUTGOING_SEM", "1"}}},
            .semaphore_bindings =
                {{.semaphore_spec_name = L0, .accessor_name = "sem_in"},
                 {.semaphore_spec_name = L1, .accessor_name = "sem_out"}},
            .compile_time_args = {{"num_elements", num_elements}, {"buf_a", buf_a}, {"buf_b", buf_b}},
            .hw_config = experimental::DataMovementGen2Config{}});

        // Writer: sink writes final; others also increment the next node's cross sem.
        KernelSpec writer{
            .unique_id = WRITER,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
            .num_threads = 1,
            .hw_config = experimental::DataMovementGen2Config{}};
        const bool writer_relays = (!is_sink) || ring;  // ring: last node relays back to node[0]
        if (!writer_relays) {
            writer.runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}};
            writer.semaphore_bindings = {{.semaphore_spec_name = L1, .accessor_name = "sem"}};
        } else {
            const SemaphoreSpecName rsem = is_sink ? WRAP : CROSS_OUT;
            writer.compiler_options = {.defines = {{"INCREMENT_REMOTE_SEM", "1"}}};
            writer.runtime_arg_schema = {
                .runtime_arg_names = {
                    "dram_addr", "l1_addr", "num_elements", "dram_bank_id", "remote_noc_x", "remote_noc_y"}};
            writer.semaphore_bindings = {
                {.semaphore_spec_name = L1, .accessor_name = "sem"},
                {.semaphore_spec_name = rsem, .accessor_name = "remote_sem"}};
        }
        spec.kernels.push_back(writer);

        std::vector<KernelSpecName> wu_kernels = {READER, XFORM, WRITER};
        if (ring && is_src) {
            // Ring: node[0] also hosts the wrap-reader. It MUST live in node[0]'s own work unit —
            // two work units may not target the same node (program_spec.cpp:1830).
            const KernelSpecName WRAP_READER{"wrap_reader"};
            spec.semaphores.push_back(SemaphoreSpec{.unique_id = WRAP_L0, .target_nodes = nodes[0]});
            spec.kernels.push_back(KernelSpec{
                .unique_id = WRAP_READER,
                .source =
                    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
                .num_threads = 1,
                .compiler_options = {.defines = {{"WAIT_FOR_REMOTE_SEM", "1"}}},
                .semaphore_bindings =
                    {{.semaphore_spec_name = WRAP_L0, .accessor_name = "sem"},
                     {.semaphore_spec_name = WRAP, .accessor_name = "remote_sem"}},
                .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
                .hw_config = experimental::DataMovementGen2Config{}});
            wu_kernels.push_back(WRAP_READER);
            params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = WRAP_READER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    nodes[0],
                    {{"dram_addr", stage_addr(N)},
                     {"l1_addr", buf_wrap},
                     {"num_elements", num_elements},
                     {"dram_bank_id", 0u}})});
        }
        spec.work_units.push_back(WorkUnitSpec{.name = "wu_" + si, .kernels = wu_kernels, .target_nodes = nodes[i]});

        // Runtime args.
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                nodes[i],
                {{"dram_addr", stage_addr(i)},
                 {"l1_addr", buf_a},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}})});
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{.kernel = XFORM});
        if (!writer_relays) {
            params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    nodes[i],
                    {{"dram_addr", stage_addr(i + 1)},
                     {"l1_addr", buf_b},
                     {"num_elements", num_elements},
                     {"dram_bank_id", 0u}})});
        } else {
            const CoreCoord nxt = mesh_device->worker_core_from_logical_core(is_sink ? nodes[0] : nodes[i + 1]);
            params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    nodes[i],
                    {{"dram_addr", stage_addr(i + 1)},
                     {"l1_addr", buf_b},
                     {"num_elements", num_elements},
                     {"dram_bank_id", 0u},
                     {"remote_noc_x", static_cast<uint32_t>(nxt.x)},
                     {"remote_noc_y", static_cast<uint32_t>(nxt.y)}})});
        }
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::SetProgramRunArgs(program, params);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> out(num_elements, 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(dev, 0, stage_addr(N), num_elements * sizeof(uint32_t), out);
    std::vector<uint32_t> expected(num_elements);
    for (uint32_t i = 0; i < num_elements; ++i) {
        expected[i] = i + N;  // +1 per hop
    }
    std::cout << "[snake] N=" << N << " hops, final stage[0]=" << out[0] << " expected=" << expected[0]
              << (out == expected ? "  PASS" : "  FAIL") << std::endl;
    EXPECT_EQ(out, expected);

    if (ring) {
        std::vector<uint32_t> wrapped(num_elements, 0);
        tt_metal::detail::ReadFromDeviceL1(dev, nodes[0], buf_wrap, num_elements * sizeof(uint32_t), wrapped);
        std::cout << "[snake] RING wrap-back to node0 L1[0]=" << wrapped[0] << " expected=" << expected[0]
                  << (wrapped == expected ? "  PASS" : "  FAIL") << std::endl;
        EXPECT_EQ(wrapped, expected);
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, RingChainFull) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    std::vector<experimental::NodeCoord> nodes;
    for (uint32_t y = 0; y < g.y; ++y) {
        if (y % 2 == 0) {
            for (uint32_t x = 0; x < g.x; ++x) {
                nodes.push_back(experimental::NodeCoord{x, y});
            }
        } else {
            for (int x = static_cast<int>(g.x) - 1; x >= 0; --x) {
                nodes.push_back(experimental::NodeCoord{static_cast<uint32_t>(x), y});
            }
        }
    }
    run_snake_chain(md, nodes, /*ring=*/true);  // 32-node snake + wrap edge back to node[0]
}

// L6 stress: longest single cross-node hop, opposite corners {0,0} <-> {7,3}.
TEST_F(QuasarMeshDeviceSingleCardFixture, SnakeCornerToCorner) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    run_snake_chain(md, {experimental::NodeCoord{0, 0}, experimental::NodeCoord{7, 3}});
}

// L6 stress: payload-size sweep across the full 32-node snake.
TEST_F(QuasarMeshDeviceSingleCardFixture, SnakePayloadSweep) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    std::vector<experimental::NodeCoord> nodes;
    for (uint32_t y = 0; y < g.y; ++y) {
        if (y % 2 == 0) {
            for (uint32_t x = 0; x < g.x; ++x) {
                nodes.push_back(experimental::NodeCoord{x, y});
            }
        } else {
            for (int x = static_cast<int>(g.x) - 1; x >= 0; --x) {
                nodes.push_back(experimental::NodeCoord{static_cast<uint32_t>(x), y});
            }
        }
    }
    for (uint32_t ne : {64u, 256u}) {
        std::cout << "[L6] payload sweep: num_elements=" << ne << std::endl;
        run_snake_chain(md, nodes, /*ring=*/false, ne);
    }
}

// L5 FAN-OUT: one DRAM root value read concurrently by ALL 32 nodes (broadcast via shared DRAM).
TEST_F(QuasarMeshDeviceSingleCardFixture, FanOutBroadcast) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    IDevice* dev = md->get_devices()[0];
    constexpr uint32_t ne = 8;
    const uint32_t buf_a = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t dram_bcast = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    std::vector<uint32_t> seed(ne);
    for (uint32_t i = 0; i < ne; ++i) {
        seed[i] = 0xa000 + i;
    }
    tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, dram_bcast, seed);
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            std::vector<uint32_t> sentinel(ne, 0xffffffffu);
            tt_metal::detail::WriteToDeviceL1(dev, experimental::NodeCoord{x, y}, buf_a, sentinel);
        }
    }

    const experimental::SemaphoreSpecName SEM{"sem"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::NodeRange all_nodes(experimental::NodeCoord{0, 0}, experimental::NodeCoord{g.x - 1, g.y - 1});
    experimental::ProgramSpec spec{.name = "fanout"};
    spec.semaphores.push_back(experimental::SemaphoreSpec{.unique_id = SEM, .target_nodes = all_nodes});
    spec.kernels.push_back(experimental::KernelSpec{
        .unique_id = READER,
        .source = std::string(OVERRIDE_KERNEL_PREFIX) +
                  "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SEM, .accessor_name = "sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
        .hw_config = experimental::DataMovementGen2Config{}});
    spec.work_units.push_back(experimental::WorkUnitSpec{.name = "wu", .kernels = {READER}, .target_nodes = all_nodes});
    experimental::ProgramRunArgs params;
    experimental::ProgramRunArgs::KernelRunArgs kra{.kernel = READER};
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            experimental::AddRuntimeArgsForNode(
                kra.runtime_arg_values,
                experimental::NodeCoord{x, y},
                {{"dram_addr", dram_bcast}, {"l1_addr", buf_a}, {"num_elements", ne}, {"dram_bank_id", 0u}});
        }
    }
    params.kernel_run_args = {kra};
    distributed::MeshWorkload workload;
    Program program = experimental::MakeProgramFromSpec(*md, spec);
    experimental::SetProgramRunArgs(program, params);
    workload.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(md->mesh_command_queue(), workload, true);

    uint32_t ok = 0, fail = 0;
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            std::vector<uint32_t> r(ne, 0);
            tt_metal::detail::ReadFromDeviceL1(dev, experimental::NodeCoord{x, y}, buf_a, ne * sizeof(uint32_t), r);
            (r == seed) ? ++ok : ++fail;
        }
    }
    std::cout << "[L5-fanout] broadcast to all nodes: ok=" << ok << " fail=" << fail << std::endl;
    EXPECT_EQ(fail, 0u);
}

// L5 FAN-IN / GATHER: all 32 nodes concurrently write a coord-unique value into a shared DRAM
// gather region (each node -> its own slot). Reader reads a per-node DRAM seed, writer stages it out.
TEST_F(QuasarMeshDeviceSingleCardFixture, FanInGather) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    IDevice* dev = md->get_devices()[0];
    constexpr uint32_t ne = 8;
    constexpr uint32_t stride = 4096;
    const uint32_t buf_a = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t dram_src = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dram_gather = dram_src + 64u * stride;  // gather region well past the per-node src slots
    auto sig_of = [&](uint32_t x, uint32_t y) { return 0xb000u + x + y * g.x; };

    // Seed each node's own DRAM src slot with its coord-unique value, and sentinel its gather
    // destination so the post-run gather check cannot pass on stale data from a prior run.
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const uint32_t idx = x + y * g.x;
            std::vector<uint32_t> s(ne, sig_of(x, y));
            tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, dram_src + idx * stride, s);
            std::vector<uint32_t> sentinel(ne, 0xffffffffu);
            tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, dram_gather + idx * stride, sentinel);
        }
    }

    const experimental::SemaphoreSpecName SEM{"sem"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::NodeRange all_nodes(experimental::NodeCoord{0, 0}, experimental::NodeCoord{g.x - 1, g.y - 1});
    experimental::ProgramSpec spec{.name = "fanin"};
    spec.semaphores.push_back(experimental::SemaphoreSpec{.unique_id = SEM, .target_nodes = all_nodes});
    spec.kernels.push_back(experimental::KernelSpec{
        .unique_id = READER,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SEM, .accessor_name = "sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
        .hw_config = experimental::DataMovementGen2Config{}});
    spec.kernels.push_back(experimental::KernelSpec{
        .unique_id = WRITER,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SEM, .accessor_name = "sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
        .hw_config = experimental::DataMovementGen2Config{}});
    spec.work_units.push_back(
        experimental::WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER}, .target_nodes = all_nodes});

    experimental::ProgramRunArgs params;
    experimental::ProgramRunArgs::KernelRunArgs kr{.kernel = READER};
    experimental::ProgramRunArgs::KernelRunArgs kw{.kernel = WRITER};
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const uint32_t idx = x + y * g.x;
            experimental::AddRuntimeArgsForNode(
                kr.runtime_arg_values,
                experimental::NodeCoord{x, y},
                {{"dram_addr", dram_src + idx * stride},
                 {"l1_addr", buf_a},
                 {"num_elements", ne},
                 {"dram_bank_id", 0u}});
            experimental::AddRuntimeArgsForNode(
                kw.runtime_arg_values,
                experimental::NodeCoord{x, y},
                {{"dram_addr", dram_gather + idx * stride},
                 {"l1_addr", buf_a},
                 {"num_elements", ne},
                 {"dram_bank_id", 0u}});
        }
    }
    params.kernel_run_args = {kr, kw};
    distributed::MeshWorkload workload;
    Program program = experimental::MakeProgramFromSpec(*md, spec);
    experimental::SetProgramRunArgs(program, params);
    workload.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(md->mesh_command_queue(), workload, true);

    uint32_t ok = 0, fail = 0;
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const uint32_t idx = x + y * g.x;
            std::vector<uint32_t> r(ne, 0);
            tt_metal::detail::ReadFromDeviceDRAMChannel(dev, 0, dram_gather + idx * stride, ne * sizeof(uint32_t), r);
            (r == std::vector<uint32_t>(ne, sig_of(x, y))) ? ++ok : ++fail;
        }
    }
    std::cout << "[L5-fanin] gather from all nodes: ok=" << ok << " fail=" << fail << std::endl;
    EXPECT_EQ(fail, 0u);
}

// L6 per-row chains: exercise each of the 4 rows as its own independent chain.
TEST_F(QuasarMeshDeviceSingleCardFixture, PerRowChains) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    for (uint32_t y = 0; y < g.y; ++y) {
        std::vector<experimental::NodeCoord> row;
        row.reserve(g.x);
        for (uint32_t x = 0; x < g.x; ++x) {
            row.push_back(experimental::NodeCoord{x, y});
        }
        std::cout << "[L6] per-row chain y=" << y << std::endl;
        run_snake_chain(md, row);
    }
}

// Vertical analog of PerRowChains: each of the 8 columns is a top->bottom 4-node chain.
// run_snake_chain blocks per call (EnqueueMeshWorkload is blocking), so the 8 column chains
// run SEQUENTIALLY -- one completes (enqueue + readback + assert) before the next starts.
// This adds vertical-NoC-path coverage vs PerRowChains' horizontal hops; it is NOT
// concurrent-chain contention (which the single-NoC functional sim would not model anyway).
TEST_F(QuasarMeshDeviceSingleCardFixture, PerColumnChains) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    for (uint32_t x = 0; x < g.x; ++x) {
        std::vector<experimental::NodeCoord> col;
        col.reserve(g.y);
        for (uint32_t y = 0; y < g.y; ++y) {
            col.push_back(experimental::NodeCoord{x, y});
        }
        std::cout << "[L6] per-column chain x=" << x << std::endl;
        run_snake_chain(md, col);
    }
}

// Grid all-to-one barrier: every node except the target (logical {0,0}) remote-increments the
// target's barrier semaphore once; the target waits for all N-1 increments (one down() each) and
// records completion in L1. Exercises maximum semaphore fan-in onto a single counter. Non-DFB.
TEST_F(QuasarMeshDeviceSingleCardFixture, GridBarrierAllToOne) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    IDevice* dev = md->get_devices()[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x != 8 || g.y != 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    const experimental::NodeCoord target{0, 0};
    const uint32_t num_signalers = g.x * g.y - 1;  // everyone except the target
    const uint32_t result_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    // Seed the target's result slot so an incomplete barrier is detectable.
    std::vector<uint32_t> seed{0xffffffffu};
    tt_metal::detail::WriteToDeviceL1(dev, target, result_addr, seed);

    const experimental::SemaphoreSpecName BARRIER{"barrier_sem"};
    const experimental::KernelSpecName BK{"barrier_kernel"};
    const CoreCoord tgt_phys = md->worker_core_from_logical_core(target);
    const experimental::NodeRange all_nodes{experimental::NodeCoord{0, 0}, experimental::NodeCoord{g.x - 1, g.y - 1}};

    experimental::ProgramSpec spec{.name = "grid_barrier"};
    // One barrier semaphore, allocated at the same L1 address on all 32 nodes (so the remote-inc
    // address computation and the target's local read agree).
    spec.semaphores.push_back(experimental::SemaphoreSpec{.unique_id = BARRIER, .target_nodes = all_nodes});
    experimental::KernelSpec bk{
        .unique_id = BK,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/grid_barrier.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = BARRIER, .accessor_name = "barrier_sem"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"remote_noc_x", "remote_noc_y", "is_target", "num_elements", "result_addr"}},
        .hw_config = experimental::DataMovementGen2Config{}};
    spec.kernels.push_back(bk);
    spec.work_units.push_back(experimental::WorkUnitSpec{.name = "wu", .kernels = {BK}, .target_nodes = all_nodes});

    Program program = experimental::MakeProgramFromSpec(*md, spec);

    experimental::ProgramRunArgs params;
    experimental::ProgramRunArgs::KernelRunArgs kra{.kernel = BK};
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const bool is_tgt = (x == 0 && y == 0);
            experimental::AddRuntimeArgsForNode(
                kra.runtime_arg_values,
                experimental::NodeCoord{x, y},
                {{"remote_noc_x", static_cast<uint32_t>(tgt_phys.x)},
                 {"remote_noc_y", static_cast<uint32_t>(tgt_phys.y)},
                 {"is_target", is_tgt ? 1u : 0u},
                 {"num_elements", num_signalers},
                 {"result_addr", result_addr}});
        }
    }
    params.kernel_run_args = {kra};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshCommandQueue& cq = md->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(md->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> r(1, 0u);
    tt_metal::detail::ReadFromDeviceL1(dev, target, result_addr, sizeof(uint32_t), r);
    // The target writes the RELEASED sentinel only after down(num_signalers) AND wait(0) both
    // pass -- i.e. exactly num_signalers increments arrived and the counter drained to 0. An
    // under-count blocks in down() (hang/timeout); an over-count blocks in wait(0). So this
    // asserts an OBSERVED device property, not the input constant it was seeded/sent. (A single
    // counting semaphore cannot verify signaler identity/distinctness -- inherent to a counter.)
    constexpr uint32_t kReleased = 0xC0DEBA11u;
    std::cout << "[BARRIER] target result=0x" << std::hex << r[0] << " expected released=0x" << kReleased << std::dec
              << (r[0] == kReleased ? "  PASS" : "  FAIL") << std::endl;
    EXPECT_EQ(r[0], kReleased);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, SnakeChain3) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    // Uses logical nodes {0,0},{1,0},{2,0}; skip (don't fault) on grids narrower than 3 (e.g. 1x3 CI).
    if (devices_[0]->compute_with_storage_grid_size().x < 3) {
        GTEST_SKIP() << "need a >=3-wide grid";
    }
    // First rung: source -> relay -> sink (3 adjacent nodes in row 0).
    run_snake_chain(devices_[0], {{0, 0}, {1, 0}, {2, 0}});
}

TEST_F(QuasarMeshDeviceSingleCardFixture, SnakeChainRow) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    if (md->compute_with_storage_grid_size().x < 8) {
        GTEST_SKIP() << "need an 8-wide grid";
    }
    std::vector<experimental::NodeCoord> nodes;  // full row 0, left->right (8 hops)
    nodes.reserve(8);
    for (uint32_t x = 0; x < 8; ++x) {
        nodes.push_back(experimental::NodeCoord{x, 0});
    }
    run_snake_chain(md, nodes);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, SnakeChainFull) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    auto md = devices_[0];
    const auto g = md->compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    // Boustrophedon: row0 L->R, row1 R->L, ... so every hop is nearest-neighbor.
    std::vector<experimental::NodeCoord> nodes;
    for (uint32_t y = 0; y < g.y; ++y) {
        if (y % 2 == 0) {
            for (uint32_t x = 0; x < g.x; ++x) {
                nodes.push_back(experimental::NodeCoord{x, y});
            }
        } else {
            for (int x = static_cast<int>(g.x) - 1; x >= 0; --x) {
                nodes.push_back(experimental::NodeCoord{static_cast<uint32_t>(x), y});
            }
        }
    }
    run_snake_chain(md, nodes);  // all 32 nodes, one continuous snake
}
