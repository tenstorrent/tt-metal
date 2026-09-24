// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "host_api.hpp"
#include "llk_device_fixture.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

namespace {

// A scratch slot at the top of L1 where the kernels write results for the host to read back.
// `size_bytes` is rounded up to the L1 alignment.
std::uint32_t top_of_l1_result_addr(IDevice& device, std::uint32_t size_bytes) {
    const std::uint32_t l1_alignment = device.allocator()->get_alignment(BufferType::L1);
    const std::uint32_t aligned = (size_bytes + l1_alignment - 1) / l1_alignment * l1_alignment;
    return static_cast<std::uint32_t>(device.l1_size_per_core()) - aligned;
}

}  // namespace

// Minimal reproducer for the QuasarCbL1ReadApi fault: a single mailbox_write (UNPACK ->
// MathThreadId) matched by a single mailbox_read (MATH <- UnpackThreadId). No CB/DFB is
// involved, isolating whether the mailbox mechanism itself faults independent of the
// dataflow-buffer address computation in QuasarCbL1ReadApi.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarMailboxMinimal) {
    constexpr CoreCoord WORKER_CORE = {0, 0};
    // Must match kValue in quasar_mailbox_minimal_compute.cpp -- the kernel and host are separate
    // TUs, so nothing enforces this at compile time.
    constexpr std::uint32_t MAILBOX_MIN_EXPECTED_VALUE = 0xfacefaceu;

    const experimental::NodeCoord node{WORKER_CORE.x, WORKER_CORE.y};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_mailbox_minimal_compute.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    const experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {COMPUTE},
        .target_nodes = node,
    };

    const experimental::ProgramSpec spec{
        .name = "quasar_mailbox_minimal",
        .kernels = {compute_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    // Kernel (MATH thread) writes the mailbox-read value here; host reads it back after the run.
    const std::uint32_t result_l1_addr = top_of_l1_result_addr(this->device(), sizeof(std::uint32_t));
    std::vector<std::uint32_t> result_init(1, 0);
    slow_dispatch::WriteToL1(this->device(), WORKER_CORE, result_l1_addr, result_init);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = COMPUTE,
            .runtime_arg_values =
                experimental::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_l1_addr}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(this->device(), std::move(program));

    std::vector<std::uint32_t> host_buffer;
    slow_dispatch::ReadFromL1(this->device(), WORKER_CORE, result_l1_addr, sizeof(std::uint32_t), host_buffer);

    ASSERT_EQ(host_buffer.size(), 1u);
    EXPECT_EQ(host_buffer[0], MAILBOX_MIN_EXPECTED_VALUE);
}

// DM -> TRISC mailbox path over plain MMIO, covering all three compute-thread receivers: a
// data-movement kernel writes one value per receiver into NEO0's TRISC mailbox queues
// k = reader*4 + writer (reader in {T0=UNPACK, T1=MATH, T2=PACK}, writer slot T3=IsolateSfpu):
//   UNPACK: k = 0*4+3 = 3  (NEO_REGS_0 TRISC_MAILBOX_3,  0x0180018C)
//   MATH:   k = 1*4+3 = 7  (NEO_REGS_0 TRISC_MAILBOX_7,  0x0180019C)
//   PACK:   k = 2*4+3 = 11 (NEO_REGS_0 TRISC_MAILBOX_11, 0x018001AC)
// A DM core is not a TRISC, so it "impersonates" writer T3 (per the Quasar HW addressing model);
// each receiving thread drains its queue with ckernel::mailbox_read(IsolateSfpuThreadId) and records
// what arrived into its own L1 result slot. The host verifies all three values round-tripped.
//
// The DM kernel brackets the mailbox stores with sentinel writes to the cluster-control scratch
// register SCRATCH_16 (0x03000080): 0x00AAAAAA before, 0x0BBBBBBB after -- waveform markers for
// localizing a hang/fault of the mailbox MMIO accesses on Zebu.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarDmToTriscMailbox) {
    constexpr CoreCoord WORKER_CORE = {0, 0};
    // One value per receiving TRISC thread, handed to the DM writer kernel as compile-time args.
    // Distinct per receiver so a queue mix-up is detectable.
    constexpr std::uint32_t DM_MBX_VAL_UNPACK = 0xC0FFEE01u;
    constexpr std::uint32_t DM_MBX_VAL_MATH = 0xC0FFEE02u;
    constexpr std::uint32_t DM_MBX_VAL_PACK = 0xC0FFEE03u;

    const std::vector<std::uint32_t> expected_result = {DM_MBX_VAL_UNPACK, DM_MBX_VAL_MATH, DM_MBX_VAL_PACK};

    const experimental::NodeCoord node{WORKER_CORE.x, WORKER_CORE.y};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    // num_threads = 1: the DM allocator skips reserved DM0/DM1 (cluster orchestrator / DFB init)
    // and hands out the lowest free DM core, i.e. DM2. The kernel gates on hartid == 2 so exactly
    // that one core performs the writes.
    const experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_dm_mailbox_scratch_writer.cpp",
        .num_threads = 1,
        .compile_time_args =
            {{"value_unpack", DM_MBX_VAL_UNPACK}, {"value_math", DM_MBX_VAL_MATH}, {"value_pack", DM_MBX_VAL_PACK}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    // num_threads = 1 places the compute kernel on Tensix engine 0 (NEO0) of the cluster -- the
    // same NEO whose mailboxes the DM kernel writes.
    const experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_dm_mailbox_scratch_compute.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    const experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {WRITER, COMPUTE},
        .target_nodes = node,
    };

    const experimental::ProgramSpec spec{
        .name = "quasar_dm_to_trisc_mailbox",
        .kernels = {writer_spec, compute_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    // Each TRISC thread writes the value it read here; host reads it back after the run.
    const std::uint32_t result_size_bytes = expected_result.size() * sizeof(std::uint32_t);
    const std::uint32_t result_l1_addr = top_of_l1_result_addr(this->device(), result_size_bytes);
    std::vector<std::uint32_t> result_init(expected_result.size(), 0);
    slow_dispatch::WriteToL1(this->device(), WORKER_CORE, result_l1_addr, result_init);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = COMPUTE,
            .runtime_arg_values =
                experimental::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_l1_addr}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(this->device(), std::move(program));

    std::vector<std::uint32_t> host_buffer;
    slow_dispatch::ReadFromL1(this->device(), WORKER_CORE, result_l1_addr, result_size_bytes, host_buffer);

    EXPECT_EQ(host_buffer, expected_result);
}

// Validates ckernel::read_tile_value and ckernel::get_tile_address on Quasar (cb_api.h). Both are
// implemented as an UNPACK -> mailbox -> MATH/PACK broadcast, so this is the same handshake as
// QuasarMailboxMinimal above, exercised through the real compute API instead of raw mailbox calls.
//
// The host preloads two known tiles into the DFB's L1 ring; all three compute threads read them
// back through both APIs and each records what it observed into its own slice of the result
// buffer. Checking all three slices (rather than only UNPACK's) is what makes this cover the
// mailbox delivery -- verifying UNPACK alone would pass even if MATH/PACK received nothing.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarCbL1ReadApi) {
    constexpr CoreCoord WORKER_CORE = {0, 0};
    using CbApiDataT = std::uint32_t;
    constexpr auto CB_API_DATA_FORMAT = DataFormat::Int32;

    constexpr CbApiDataT CB_API_VAL0 = 0xA5A5A5A5u;
    constexpr CbApiDataT CB_API_VAL1 = 0x11111111u;
    constexpr CbApiDataT CB_API_VAL2 = 0x22222222u;
    constexpr CbApiDataT CB_API_VAL3 = 0x33333333u;

    // What each participating thread should observe:
    // {tile0[0], tile0[1], tile1[0], tile1[1], *get_tile_address(1)}.
    const std::vector<CbApiDataT> CB_API_EXPECTED_PER_THREAD = {
        CB_API_VAL0, CB_API_VAL1, CB_API_VAL2, CB_API_VAL3, CB_API_VAL2};
    // UNPACK, MATH and PACK each record their own copy; ISOLATE_SFPU does not participate.
    constexpr std::uint32_t CB_API_NUM_READER_THREADS = 3;

    const std::uint32_t tile_page_size = tt::tile_size(CB_API_DATA_FORMAT);

    const experimental::NodeCoord node{WORKER_CORE.x, WORKER_CORE.y};
    const experimental::DFBSpecName IN_DFB{"in_dfb"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const experimental::DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = tile_page_size,
        .num_entries = 2,
        .data_format_metadata = CB_API_DATA_FORMAT,
    };

    // The DFB has no other toucher -- the host preloads it and the kernel only reads it -- so
    // compute binds it as both PRODUCER and CONSUMER (self-loop). That is what finalizes the DFB
    // config (base_addr, entry_size, ...) and gets it written to L1.
    const experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_cb_l1_read_api_compute.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
             },
             {
                 .dfb_spec_name = IN_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
             }},
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    const experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {COMPUTE},
        .target_nodes = node,
    };

    const experimental::ProgramSpec spec{
        .name = "quasar_cb_l1_read_api",
        .kernels = {compute_spec},
        .dataflow_buffers = {in_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    // Preload two known tiles into the DFB's L1 ring (single DFB -> ring base is the L1
    // allocator base). Tiles are entry_size apart for this 1-producer/1-consumer layout.
    const std::uint32_t dfb_l1_addr =
        static_cast<std::uint32_t>(this->device().allocator()->get_base_allocator_addr(HalMemType::L1));
    const std::uint32_t words_per_entry = tile_page_size / sizeof(CbApiDataT);
    std::vector<CbApiDataT> ring(2 * words_per_entry, 0);
    ring[0] = CB_API_VAL0;
    ring[1] = CB_API_VAL1;
    ring[words_per_entry + 0] = CB_API_VAL2;
    ring[words_per_entry + 1] = CB_API_VAL3;
    slow_dispatch::WriteToL1(this->device(), WORKER_CORE, dfb_l1_addr, ring);

    std::vector<CbApiDataT> expected_result;
    for (std::uint32_t thread = 0; thread < CB_API_NUM_READER_THREADS; ++thread) {
        expected_result.insert(
            expected_result.end(), CB_API_EXPECTED_PER_THREAD.begin(), CB_API_EXPECTED_PER_THREAD.end());
    }

    // Each thread writes its reads here; host reads this spot back after the run.
    const std::uint32_t result_size_bytes = expected_result.size() * sizeof(CbApiDataT);
    const std::uint32_t result_l1_addr = top_of_l1_result_addr(this->device(), result_size_bytes);

    std::vector<CbApiDataT> result_init(expected_result.size(), 0);
    slow_dispatch::WriteToL1(this->device(), WORKER_CORE, result_l1_addr, result_init);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = COMPUTE,
            .runtime_arg_values =
                experimental::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_l1_addr}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(this->device(), std::move(program));

    std::vector<CbApiDataT> host_buffer;
    slow_dispatch::ReadFromL1(this->device(), WORKER_CORE, result_l1_addr, result_size_bytes, host_buffer);

    EXPECT_EQ(host_buffer, expected_result);
}

}  // namespace tt::tt_metal
