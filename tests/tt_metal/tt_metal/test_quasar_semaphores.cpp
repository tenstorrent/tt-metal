// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hal.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "impl/program/program_impl.hpp"
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
    const NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr uint32_t num_elements = 10;
    const uint32_t buf_a_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t buf_b_addr = buf_a_addr + num_elements * sizeof(uint32_t);

    // Source/destination live in DRAM MeshBuffers (single contiguous page in bank 0); the kernels
    // read/write them at {bank 0, buf->address()} and host I/O goes through the mesh command queue.
    constexpr uint32_t dram_bytes = num_elements * sizeof(uint32_t);
    auto make_dram_buf = [&] {
        distributed::DeviceLocalBufferConfig lc{.page_size = dram_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig gc{.size = dram_bytes};
        return distributed::MeshBuffer::create(gc, lc, mesh_device.get());
    };
    auto dram_src_buf = make_dram_buf();
    auto dram_dst_buf = make_dram_buf();
    const uint32_t dram_src_addr = dram_src_buf->address();
    const uint32_t dram_dst_addr = dram_dst_buf->address();

    std::vector<uint32_t> initial_data(num_elements, 0);
    for (uint32_t i = 0; i < num_elements; i++) {
        initial_data[i] = i;
    }
    distributed::EnqueueWriteMeshBuffer(cq, dram_src_buf, initial_data);

    const KernelSpecName DM_READER{"dm_reader"};
    const KernelSpecName DM_TRANSFORM{"dm_transform"};
    const KernelSpecName DM_WRITER{"dm_writer"};

    SemaphoreSpec sem0_spec{
        .unique_id = SemaphoreSpecName{"sem0"},
        .target_nodes = node,
    };
    SemaphoreSpec sem1_spec{
        .unique_id = SemaphoreSpecName{"sem1"},
        .target_nodes = node,
    };

    KernelSpec dm_reader_spec{
        .unique_id = DM_READER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"sem0"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    KernelSpec dm_transform_spec{
        .unique_id = DM_TRANSFORM,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}, {"INCOMING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = SemaphoreSpecName{"sem0"}, .accessor_name = "sem_in"},
                {.semaphore_spec_name = SemaphoreSpecName{"sem1"}, .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    KernelSpec dm_writer_spec{
        .unique_id = DM_WRITER,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"sem1"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_READER, DM_TRANSFORM, DM_WRITER},
        .target_nodes = node,
    };

    ProgramSpec spec{
        .name = "multi_semaphore_pipeline",
        .kernels = {dm_reader_spec, dm_transform_spec, dm_writer_spec},
        .semaphores = {sem0_spec, sem1_spec},
        .work_units = {main_wu},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = DM_READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"dram_addr", dram_src_addr},
                 {"l1_addr", buf_a_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
        ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM},
        ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"dram_addr", dram_dst_addr},
                 {"l1_addr", buf_b_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
    };
    tt_metal::experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data;
    distributed::EnqueueReadMeshBuffer(cq, actual_data, dram_dst_buf, /*blocking=*/true);

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

    const NodeCoord node_0{0, 0};
    const NodeCoord node_1{1, 0};

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
    slow_dispatch::WriteToL1(*mesh_device, node_0, buf_a_addr, initial_data);

    const CoreCoord core_1_virtual = mesh_device->worker_core_from_logical_core(node_1);

    SemaphoreSpec sem_core_0_spec{
        .unique_id = SemaphoreSpecName{"sem_core_0"},
        .target_nodes = node_0,
    };
    SemaphoreSpec sem_cross_spec{
        .unique_id = SemaphoreSpecName{"sem_cross"},
        .target_nodes = NodeRange{node_0, node_1},
    };
    SemaphoreSpec sem0_core_1_spec{
        .unique_id = SemaphoreSpecName{"sem0_core_1"},
        .target_nodes = node_1,
    };
    SemaphoreSpec sem1_core_1_spec{
        .unique_id = SemaphoreSpecName{"sem1_core_1"},
        .target_nodes = node_1,
    };

    const KernelSpecName DM_TRANSFORM_0{"dm_transform_0"};
    const KernelSpecName DM_WRITER_0{"dm_writer_0"};
    const KernelSpecName DM_READER_1{"dm_reader_1"};
    const KernelSpecName DM_TRANSFORM_1{"dm_transform_1"};
    const KernelSpecName DM_WRITER_1{"dm_writer_1"};

    KernelSpec dm_transform_0_spec{
        .unique_id = DM_TRANSFORM_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"OUTGOING_SEM", "1"}}},
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"sem_core_0"}, .accessor_name = "sem_out"}},
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    KernelSpec dm_writer_0_spec{
        .unique_id = DM_WRITER_0,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCREMENT_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = SemaphoreSpecName{"sem_core_0"}, .accessor_name = "sem"},
                {.semaphore_spec_name = SemaphoreSpecName{"sem_cross"}, .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"dram_addr", "l1_addr", "num_elements", "dram_bank_id", "remote_noc_x", "remote_noc_y"},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    KernelSpec dm_reader_1_spec{
        .unique_id = DM_READER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"WAIT_FOR_REMOTE_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = SemaphoreSpecName{"sem0_core_1"}, .accessor_name = "sem"},
                {.semaphore_spec_name = SemaphoreSpecName{"sem_cross"}, .accessor_name = "remote_sem"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    KernelSpec dm_transform_1_spec{
        .unique_id = DM_TRANSFORM_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/transform_pipeline.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"INCOMING_SEM", "1"}, {"OUTGOING_SEM", "1"}}},
        .semaphore_bindings =
            {
                {.semaphore_spec_name = SemaphoreSpecName{"sem0_core_1"}, .accessor_name = "sem_in"},
                {.semaphore_spec_name = SemaphoreSpecName{"sem1_core_1"}, .accessor_name = "sem_out"},
            },
        .compile_time_args =
            {
                {"num_elements", num_elements},
                {"buf_a", buf_a_addr},
                {"buf_b", buf_b_addr},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    KernelSpec dm_writer_1_spec{
        .unique_id = DM_WRITER_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"sem1_core_1"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"},
            },
        .hw_config = DataMovementHardwareConfig{},
    };

    WorkUnitSpec wu_core_0{
        .name = "wu_core_0",
        .kernels = {DM_TRANSFORM_0, DM_WRITER_0},
        .target_nodes = node_0,
    };
    WorkUnitSpec wu_core_1{
        .name = "wu_core_1",
        .kernels = {DM_READER_1, DM_TRANSFORM_1, DM_WRITER_1},
        .target_nodes = node_1,
    };

    ProgramSpec spec{
        .name = "multi_cluster_multi_semaphore_pipeline",
        .kernels = {dm_transform_0_spec, dm_writer_0_spec, dm_reader_1_spec, dm_transform_1_spec, dm_writer_1_spec},
        .semaphores = {sem_core_0_spec, sem_cross_spec, sem0_core_1_spec, sem1_core_1_spec},
        .work_units = {wu_core_0, wu_core_1},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM_0},
        ProgramRunArgs::KernelRunArgs{.kernel = DM_TRANSFORM_1},
        ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER_0,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node_0,
                {{"dram_addr", dram_mid_addr},
                 {"l1_addr", buf_b_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u},
                 {"remote_noc_x", static_cast<uint32_t>(core_1_virtual.x)},
                 {"remote_noc_y", static_cast<uint32_t>(core_1_virtual.y)}}),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = DM_READER_1,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node_1,
                {{"dram_addr", dram_mid_addr},
                 {"l1_addr", buf_a_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = DM_WRITER_1,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node_1,
                {{"dram_addr", dram_dst_addr},
                 {"l1_addr", buf_b_addr},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}}),
        },
    };
    tt_metal::experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_data(num_elements, 0);
    slow_dispatch::ReadFromDRAMChannel(*mesh_device, 0, dram_dst_addr, num_elements * sizeof(uint32_t), actual_data);

    const std::vector<uint32_t> expected_data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    ASSERT_EQ(actual_data, expected_data);
}

// Semaphore + DRAM-staged snake chain across an ordered list of nodes.
// Data flows node[0] -> node[1] -> ... -> node[N-1]. At each hop node[i]'s writer
// stages its transformed buffer to DRAM and bumps the next node's cross-node semaphore;
// node[i+1]'s reader waits on it, reads the stage, adds 1, and passes it along.
// Final DRAM stage = seed + N. Shows every node can both receive from the previous node
// and send to the next, the 2-node cross-node pipeline generalized to N nodes.
static void run_snake_chain(
    distributed::MeshDevice& mesh_device,
    const std::vector<NodeCoord>& nodes,
    bool ring = false,
    uint32_t num_elements = 10) {
    const uint32_t N = static_cast<uint32_t>(nodes.size());
    ASSERT_GE(N, 2u);

    const uint32_t buf_a = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t buf_b = buf_a + num_elements * sizeof(uint32_t);
    const uint32_t buf_wrap = buf_b + num_elements * sizeof(uint32_t);
    const SemaphoreSpecName WRAP{"wrap_sem"};
    const SemaphoreSpecName WRAP_L0{"wrap_l0"};
    const uint32_t dram_base = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    constexpr uint32_t stage_stride = 4096;  // per-hop DRAM staging spacing
    auto stage_addr = [&](uint32_t i) { return dram_base + i * stage_stride; };

    ASSERT_LE(num_elements * sizeof(uint32_t), stage_stride);

    // Seed stage 0 with [0..num_elements).
    std::vector<uint32_t> seed(num_elements);
    for (uint32_t i = 0; i < num_elements; ++i) {
        seed[i] = i;
    }
    slow_dispatch::WriteToDRAMChannel(mesh_device, 0, stage_addr(0), seed);
    // Pre-fill the output stage with a sentinel value.
    std::vector<uint32_t> out_sentinel(num_elements, 0xdeadbeefu);
    slow_dispatch::WriteToDRAMChannel(mesh_device, 0, stage_addr(N), out_sentinel);
    // Ring mode also checks the token that wraps back to node[0]'s buf_wrap; pre-fill it with a
    // sentinel too so that check can't pass on stale or uninitialized data.
    if (ring) {
        slow_dispatch::WriteToL1(mesh_device, nodes[0], buf_wrap, out_sentinel);
    }

    ProgramSpec spec{.name = "snake_chain"};
    ProgramRunArgs params;

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

        // Local intra-node sems.
        spec.semaphores.push_back(SemaphoreSpec{.unique_id = L0, .target_nodes = nodes[i]});
        spec.semaphores.push_back(SemaphoreSpec{.unique_id = L1, .target_nodes = nodes[i]});

        if (!is_sink) {
            const auto a = nodes[i];
            const auto b = nodes[i + 1];
            const NodeCoord lo{std::min(a.x, b.x), std::min(a.y, b.y)};
            const NodeCoord hi{std::max(a.x, b.x), std::max(a.y, b.y)};
            spec.semaphores.push_back(SemaphoreSpec{.unique_id = CROSS_OUT, .target_nodes = NodeRange{lo, hi}});
        } else if (ring) {
            const auto a = nodes[i];
            const auto b = nodes[0];
            const NodeCoord lo{std::min(a.x, b.x), std::min(a.y, b.y)};
            const NodeCoord hi{std::max(a.x, b.x), std::max(a.y, b.y)};
            spec.semaphores.push_back(SemaphoreSpec{.unique_id = WRAP, .target_nodes = NodeRange{lo, hi}});
        }

        // Reader: source reads without waiting; others wait on the incoming cross sem.
        KernelSpec reader{
            .unique_id = READER,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
            .hw_config = DataMovementHardwareConfig{}};
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
            .hw_config = DataMovementHardwareConfig{}});

        // Writer: sink writes final; others also increment the next node's cross sem.
        KernelSpec writer{
            .unique_id = WRITER,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
            .num_threads = 1,
            .hw_config = DataMovementHardwareConfig{}};
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
            // Ring: node[0] also runs the wrap-reader. It must go in node[0]'s existing work unit
            // because two work units are not allowed to target the same node.
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
                .hw_config = DataMovementHardwareConfig{}});
            wu_kernels.push_back(WRAP_READER);
            params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
                .kernel = WRAP_READER,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    nodes[0],
                    {{"dram_addr", stage_addr(N)},
                     {"l1_addr", buf_wrap},
                     {"num_elements", num_elements},
                     {"dram_bank_id", 0u}})});
        }
        spec.work_units.push_back(WorkUnitSpec{.name = "wu_" + si, .kernels = wu_kernels, .target_nodes = nodes[i]});

        // Runtime args.
        params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                nodes[i],
                {{"dram_addr", stage_addr(i)},
                 {"l1_addr", buf_a},
                 {"num_elements", num_elements},
                 {"dram_bank_id", 0u}})});
        params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{.kernel = XFORM});
        if (!writer_relays) {
            params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    nodes[i],
                    {{"dram_addr", stage_addr(i + 1)},
                     {"l1_addr", buf_b},
                     {"num_elements", num_elements},
                     {"dram_bank_id", 0u}})});
        } else {
            const CoreCoord nxt = mesh_device.worker_core_from_logical_core(is_sink ? nodes[0] : nodes[i + 1]);
            params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    nodes[i],
                    {{"dram_addr", stage_addr(i + 1)},
                     {"l1_addr", buf_b},
                     {"num_elements", num_elements},
                     {"dram_bank_id", 0u},
                     {"remote_noc_x", static_cast<uint32_t>(nxt.x)},
                     {"remote_noc_y", static_cast<uint32_t>(nxt.y)}})});
        }
    }

    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    Program program = tt_metal::experimental::MakeProgramFromSpec(mesh_device, spec);
    tt_metal::experimental::SetProgramRunArgs(program, params);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> out(num_elements, 0);
    slow_dispatch::ReadFromDRAMChannel(mesh_device, 0, stage_addr(N), num_elements * sizeof(uint32_t), out);
    std::vector<uint32_t> expected(num_elements);
    for (uint32_t i = 0; i < num_elements; ++i) {
        expected[i] = i + N;  // +1 per hop
    }
    std::cout << "[snake] N=" << N << " hops, final stage[0]=" << out[0] << " expected=" << expected[0]
              << (out == expected ? "  PASS" : "  FAIL") << std::endl;
    EXPECT_EQ(out, expected);

    if (ring) {
        std::vector<uint32_t> wrapped(num_elements, 0);
        slow_dispatch::ReadFromL1(mesh_device, nodes[0], buf_wrap, num_elements * sizeof(uint32_t), wrapped);
        std::cout << "[snake] RING wrap-back to node0 L1[0]=" << wrapped[0] << " expected=" << expected[0]
                  << (wrapped == expected ? "  PASS" : "  FAIL") << std::endl;
        EXPECT_EQ(wrapped, expected);
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, RingChainFull) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    std::vector<NodeCoord> nodes;
    for (uint32_t y = 0; y < g.y; ++y) {
        if (y % 2 == 0) {
            for (uint32_t x = 0; x < g.x; ++x) {
                nodes.push_back(NodeCoord{x, y});
            }
        } else {
            for (int x = static_cast<int>(g.x) - 1; x >= 0; --x) {
                nodes.push_back(NodeCoord{static_cast<uint32_t>(x), y});
            }
        }
    }
    run_snake_chain(this->device(), nodes, /*ring=*/true);  // 32-node snake + wrap edge back to node[0]
}

// Longest single cross-node hop: opposite corners {0,0} <-> {7,3}.
TEST_F(QuasarMeshDeviceSingleCardFixture, SnakeCornerToCorner) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    run_snake_chain(this->device(), {NodeCoord{0, 0}, NodeCoord{7, 3}});
}

// Sweep a few payload sizes across the full 32-node snake.
TEST_F(QuasarMeshDeviceSingleCardFixture, SnakePayloadSweep) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    std::vector<NodeCoord> nodes;
    for (uint32_t y = 0; y < g.y; ++y) {
        if (y % 2 == 0) {
            for (uint32_t x = 0; x < g.x; ++x) {
                nodes.push_back(NodeCoord{x, y});
            }
        } else {
            for (int x = static_cast<int>(g.x) - 1; x >= 0; --x) {
                nodes.push_back(NodeCoord{static_cast<uint32_t>(x), y});
            }
        }
    }
    for (uint32_t ne : {64u, 256u}) {
        std::cout << "[L6] payload sweep: num_elements=" << ne << std::endl;
        run_snake_chain(this->device(), nodes, /*ring=*/false, ne);
    }
}

// Fan-out: all 32 nodes read the same DRAM value at once (broadcast through shared DRAM).
TEST_F(QuasarMeshDeviceSingleCardFixture, FanOutBroadcast) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    constexpr uint32_t ne = 8;
    const uint32_t buf_a = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t dram_bcast = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    std::vector<uint32_t> seed(ne);
    for (uint32_t i = 0; i < ne; ++i) {
        seed[i] = 0xa000 + i;
    }
    slow_dispatch::WriteToDRAMChannel(this->device(), 0, dram_bcast, seed);
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            std::vector<uint32_t> sentinel(ne, 0xffffffffu);
            slow_dispatch::WriteToL1(this->device(), NodeCoord{x, y}, buf_a, sentinel);
        }
    }

    const SemaphoreSpecName SEM{"sem"};
    const KernelSpecName READER{"reader"};
    const NodeRange all_nodes(NodeCoord{0, 0}, NodeCoord{g.x - 1, g.y - 1});
    ProgramSpec spec{.name = "fanout"};
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM, .target_nodes = all_nodes});
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = std::string(OVERRIDE_KERNEL_PREFIX) +
                  "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SEM, .accessor_name = "sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
        .hw_config = DataMovementHardwareConfig{}});
    spec.work_units.push_back(WorkUnitSpec{.name = "wu", .kernels = {READER}, .target_nodes = all_nodes});
    ProgramRunArgs params;
    ProgramRunArgs::KernelRunArgs kra{.kernel = READER};
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            AddRuntimeArgsForNode(
                kra.runtime_arg_values,
                NodeCoord{x, y},
                {{"dram_addr", dram_bcast}, {"l1_addr", buf_a}, {"num_elements", ne}, {"dram_bank_id", 0u}});
        }
    }
    params.kernel_run_args = {kra};
    distributed::MeshWorkload workload;
    Program program = tt_metal::experimental::MakeProgramFromSpec(this->device(), spec);
    tt_metal::experimental::SetProgramRunArgs(program, params);
    workload.add_program(distributed::MeshCoordinateRange(this->device().shape()), std::move(program));
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), workload, true);

    uint32_t ok = 0, fail = 0;
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            std::vector<uint32_t> r(ne, 0);
            slow_dispatch::ReadFromL1(this->device(), NodeCoord{x, y}, buf_a, ne * sizeof(uint32_t), r);
            (r == seed) ? ++ok : ++fail;
        }
    }
    std::cout << "[L5-fanout] broadcast to all nodes: ok=" << ok << " fail=" << fail << std::endl;
    EXPECT_EQ(fail, 0u);
}

// Fan-in / gather: all 32 nodes write at once into a shared DRAM gather region, each into its
// own slot with a value unique to its coordinates. Reader loads a per-node DRAM seed, writer
// stages it back out.
TEST_F(QuasarMeshDeviceSingleCardFixture, FanInGather) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    constexpr uint32_t ne = 8;
    constexpr uint32_t stride = 4096;
    const uint32_t buf_a = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t dram_src = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dram_gather = dram_src + 64u * stride;  // gather region well past the per-node src slots
    auto sig_of = [&](uint32_t x, uint32_t y) { return 0xb000u + x + y * g.x; };

    // Seed each node's own DRAM source slot with a value unique to its coordinates, and pre-fill
    // its gather destination with a sentinel so the gather check can't pass on stale data from a
    // prior run.
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const uint32_t idx = x + y * g.x;
            std::vector<uint32_t> s(ne, sig_of(x, y));
            slow_dispatch::WriteToDRAMChannel(this->device(), 0, dram_src + idx * stride, s);
            std::vector<uint32_t> sentinel(ne, 0xffffffffu);
            slow_dispatch::WriteToDRAMChannel(this->device(), 0, dram_gather + idx * stride, sentinel);
        }
    }

    const SemaphoreSpecName SEM{"sem"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const NodeRange all_nodes(NodeCoord{0, 0}, NodeCoord{g.x - 1, g.y - 1});
    ProgramSpec spec{.name = "fanin"};
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM, .target_nodes = all_nodes});
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SEM, .accessor_name = "sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
        .hw_config = DataMovementHardwareConfig{}});
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SEM, .accessor_name = "sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"dram_addr", "l1_addr", "num_elements", "dram_bank_id"}},
        .hw_config = DataMovementHardwareConfig{}});
    spec.work_units.push_back(WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER}, .target_nodes = all_nodes});

    ProgramRunArgs params;
    ProgramRunArgs::KernelRunArgs kr{.kernel = READER};
    ProgramRunArgs::KernelRunArgs kw{.kernel = WRITER};
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const uint32_t idx = x + y * g.x;
            AddRuntimeArgsForNode(
                kr.runtime_arg_values,
                NodeCoord{x, y},
                {{"dram_addr", dram_src + idx * stride},
                 {"l1_addr", buf_a},
                 {"num_elements", ne},
                 {"dram_bank_id", 0u}});
            AddRuntimeArgsForNode(
                kw.runtime_arg_values,
                NodeCoord{x, y},
                {{"dram_addr", dram_gather + idx * stride},
                 {"l1_addr", buf_a},
                 {"num_elements", ne},
                 {"dram_bank_id", 0u}});
        }
    }
    params.kernel_run_args = {kr, kw};
    distributed::MeshWorkload workload;
    Program program = tt_metal::experimental::MakeProgramFromSpec(this->device(), spec);
    tt_metal::experimental::SetProgramRunArgs(program, params);
    workload.add_program(distributed::MeshCoordinateRange(this->device().shape()), std::move(program));
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), workload, true);

    uint32_t ok = 0, fail = 0;
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const uint32_t idx = x + y * g.x;
            std::vector<uint32_t> r(ne, 0);
            slow_dispatch::ReadFromDRAMChannel(this->device(), 0, dram_gather + idx * stride, ne * sizeof(uint32_t), r);
            (r == std::vector<uint32_t>(ne, sig_of(x, y))) ? ++ok : ++fail;
        }
    }
    std::cout << "[L5-fanin] gather from all nodes: ok=" << ok << " fail=" << fail << std::endl;
    EXPECT_EQ(fail, 0u);
}

// Run each of the 4 rows as its own independent chain.
TEST_F(QuasarMeshDeviceSingleCardFixture, PerRowChains) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    for (uint32_t y = 0; y < g.y; ++y) {
        std::vector<NodeCoord> row;
        row.reserve(g.x);
        for (uint32_t x = 0; x < g.x; ++x) {
            row.push_back(NodeCoord{x, y});
        }
        std::cout << "[L6] per-row chain y=" << y << std::endl;
        run_snake_chain(this->device(), row);
    }
}

// Column version of PerRowChains: each of the 8 columns is a top->bottom 4-node chain.
// Each run_snake_chain call blocks, so the 8 columns run one after another, each finishes its
// enqueue, readback, and check before the next starts.
TEST_F(QuasarMeshDeviceSingleCardFixture, PerColumnChains) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    for (uint32_t x = 0; x < g.x; ++x) {
        std::vector<NodeCoord> col;
        col.reserve(g.y);
        for (uint32_t y = 0; y < g.y; ++y) {
            col.push_back(NodeCoord{x, y});
        }
        std::cout << "[L6] per-column chain x=" << x << std::endl;
        run_snake_chain(this->device(), col);
    }
}

// Grid all-to-one barrier: every node except the target (logical {0,0}) increments the target's
// barrier semaphore once over the NoC. The target waits for all N-1 increments, then records
// completion in L1. Exercises maximum semaphore fan-in onto a single counter.
TEST_F(QuasarMeshDeviceSingleCardFixture, GridBarrierAllToOne) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x != 8 || g.y != 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    const NodeCoord target{0, 0};
    const uint32_t num_signalers = g.x * g.y - 1;  // everyone except the target
    // The target's result slot is a single-core L1 MeshBuffer on the target node, result_addr is
    // threaded into the kernel and host I/O (seed + read-back) goes through the mesh command queue.
    distributed::MeshCommandQueue& cq = this->device().mesh_command_queue();
    const CoreRangeSet result_grid(CoreRange(target, target));
    const ShardSpecBuffer result_shard(
        result_grid,
        /*shard_shape=*/{1, 1},
        ShardOrientation::ROW_MAJOR,
        /*page_shape=*/{1, 1},
        /*tensor2d_shape_in_pages=*/{1, 1});
    distributed::DeviceLocalBufferConfig result_local{
        .page_size = sizeof(uint32_t),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(result_shard, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    distributed::ReplicatedBufferConfig result_global{.size = sizeof(uint32_t)};
    auto result_buf = distributed::MeshBuffer::create(result_global, result_local, &this->device());
    const uint32_t result_addr = result_buf->address();

    // Seed the target's result slot so an incomplete barrier is detectable.
    std::vector<uint32_t> seed{0xffffffffu};
    distributed::EnqueueWriteMeshBuffer(cq, result_buf, seed);

    const SemaphoreSpecName BARRIER{"barrier_sem"};
    const KernelSpecName BK{"barrier_kernel"};
    const CoreCoord tgt_phys = this->device().worker_core_from_logical_core(target);
    const NodeRange all_nodes{NodeCoord{0, 0}, NodeCoord{g.x - 1, g.y - 1}};

    ProgramSpec spec{.name = "grid_barrier"};
    // One barrier semaphore at the same L1 address on all 32 nodes, so the remote increments and
    // the target's local read all land on the same location.
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = BARRIER, .target_nodes = all_nodes});
    KernelSpec bk{
        .unique_id = BK,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/grid_barrier.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = BARRIER, .accessor_name = "barrier_sem"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"remote_noc_x", "remote_noc_y", "is_target", "num_elements", "result_addr"}},
        .hw_config = DataMovementHardwareConfig{}};
    spec.kernels.push_back(bk);
    spec.work_units.push_back(WorkUnitSpec{.name = "wu", .kernels = {BK}, .target_nodes = all_nodes});

    Program program = tt_metal::experimental::MakeProgramFromSpec(this->device(), spec);

    ProgramRunArgs params;
    ProgramRunArgs::KernelRunArgs kra{.kernel = BK};
    for (uint32_t y = 0; y < g.y; ++y) {
        for (uint32_t x = 0; x < g.x; ++x) {
            const bool is_tgt = (x == 0 && y == 0);
            AddRuntimeArgsForNode(
                kra.runtime_arg_values,
                NodeCoord{x, y},
                {{"remote_noc_x", static_cast<uint32_t>(tgt_phys.x)},
                 {"remote_noc_y", static_cast<uint32_t>(tgt_phys.y)},
                 {"is_target", is_tgt ? 1u : 0u},
                 {"num_elements", num_signalers},
                 {"result_addr", result_addr}});
        }
    }
    params.kernel_run_args = {kra};
    tt_metal::experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(this->device().shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> r;
    distributed::EnqueueReadMeshBuffer(cq, r, result_buf, /*blocking=*/true);
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
    if (this->device().compute_with_storage_grid_size().x < 3) {
        GTEST_SKIP() << "need a >=3-wide grid";
    }
    // Source -> relay -> sink across 3 adjacent nodes in row 0.
    run_snake_chain(this->device(), {{0, 0}, {1, 0}, {2, 0}});
}

TEST_F(QuasarMeshDeviceSingleCardFixture, SnakeChainRow) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    if (this->device().compute_with_storage_grid_size().x < 8) {
        GTEST_SKIP() << "need an 8-wide grid";
    }
    std::vector<NodeCoord> nodes;  // full row 0, left->right (8 hops)
    nodes.reserve(8);
    for (uint32_t x = 0; x < 8; ++x) {
        nodes.push_back(NodeCoord{x, 0});
    }
    run_snake_chain(this->device(), nodes);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, SnakeChainFull) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "simulator/emulator only";
    }
    const auto g = this->device().compute_with_storage_grid_size();
    if (g.x < 8 || g.y < 4) {
        GTEST_SKIP() << "need the full 8x4 grid";
    }
    // Snake order: row0 left->right, row1 right->left, ... so every hop is to a neighbor.
    std::vector<NodeCoord> nodes;
    for (uint32_t y = 0; y < g.y; ++y) {
        if (y % 2 == 0) {
            for (uint32_t x = 0; x < g.x; ++x) {
                nodes.push_back(NodeCoord{x, y});
            }
        } else {
            for (int x = static_cast<int>(g.x) - 1; x >= 0; --x) {
                nodes.push_back(NodeCoord{static_cast<uint32_t>(x), y});
            }
        }
    }
    run_snake_chain(this->device(), nodes);  // all 32 nodes, one continuous snake
}
