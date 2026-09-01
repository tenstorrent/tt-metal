// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end tests for Noc::async_write_zeros, covering both overloads across every accepted
// local-L1 endpoint type (CircularBuffer / DataflowBuffer / CoreLocalMem / Scratchpad /
// LocalTensorAccessor).
//
// Device kernels driven from here:
//   zero_memory_api_l1_producer.cpp           overload (1) into a DFB; ZERO_NUM_CHUNKS gives the
//                                             batched variant (N zeros, one barrier)
//   zero_memory_api_consumer.cpp              overload (2) into DRAM, DFB as the zeros source
//   zero_memory_api_raw_l1.cpp                overload (1) into CoreLocalMem / Scratchpad /
//                                             LocalTensorAccessor, picked by a ZERO_TARGET define
//   zero_memory_api_dram_from_raw_l1.cpp      overload (2) into DRAM sourcing its zeros from
//                                             CoreLocalMem / Scratchpad / LocalTensorAccessor,
//                                             same ZERO_TARGET define, no CB/DFB in the program
//
// Every test is non-vacuous by construction: the target is pre-stamped with a known non-zero pattern
// and that stamp is verified before the zero, so a kernel that did nothing cannot pass. Kernels report
// an in-kernel verdict through an L1 status word (see StatusName below), and the host independently
// re-reads the zeroed memory rather than trusting that verdict alone.
//
// The raw-L1 tests additionally zero only a SUB-WINDOW and require the bytes outside it to be
// unchanged; that flank invariance is what pins offset resolution, since a zero-the-whole-region bug
// satisfies a window-only check.

#include "device_fixture.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers.hpp"

#include "gtest/gtest.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

using tt::tt_metal::experimental::test_helpers::MakeShardedTensorParameter;

const experimental::DFBSpecName SCRATCH_DFB{"scratch"};
const experimental::TensorParamName OUT_TENSOR{"out"};

const experimental::KernelSpecName L1_PRODUCER{"l1_producer"};
const experimental::KernelSpecName DRAM_CONSUMER{"dram_consumer"};
const experimental::KernelSpecName L1_BATCHED_PRODUCER{"l1_batched_producer"};
const experimental::KernelSpecName RAW_L1_KERNEL{"raw_l1"};

constexpr uint32_t kStatusOk = 0xCAFEBABEu;

// Must match zero_memory_api_raw_l1.cpp.
constexpr uint32_t kPatternBase = 0xA5A50000u;

// Reserves L1 the host reads and writes directly — the kernel status/report words, and the region
// under test for the CoreLocalMem target — so nothing else in the runtime can claim those bytes.
std::shared_ptr<distributed::MeshBuffer> AllocateL1Scratch(distributed::MeshDevice& mesh_device, uint32_t size_bytes) {
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = size_bytes},
        {.page_size = size_bytes, .buffer_type = BufferType::L1},
        &mesh_device);
}

// One status word plus one report word, kept in a single allocation; the report slot sits at
// kReportOffset bytes from its base.
constexpr uint32_t kScratchWordsBytes = 64;
constexpr uint32_t kReportOffset = 32;

// Allocator addresses come back 64B-aligned; nudging by 16 makes the CoreLocalMem region minimally
// aligned (16B, not 32/64B) so it exercises the WH/BH floor. Allocations are sized +nudge to fit.
constexpr uint32_t kMinAlignNudge = 16;

const char* StatusName(uint32_t status) {
    switch (status) {
        case 0xCAFEBABEu: return "OK";
        case 0xBAADF00Du: return "SENTINEL (kernel never reported — crashed or never ran)";
        case 0xDEAD0001u: return "STAMP_FAIL (CPU pattern did not land; test infra bug)";
        case 0xDEAD0002u: return "ZERO_FAIL (window not zeroed)";
        case 0xDEAD0003u: return "FLANK_FAIL (bytes outside the window were clobbered — offset ignored?)";
        case 0xDEAD0011u: return "SCRATCH_NOT_ZERO (overload 1 left the scratchpad non-zero)";
        default: return "UNKNOWN";
    }
}

// Flat 1D UINT32 page layout: one DRAM page per logical row, page_size_bytes each.
// num_pages rows, page_size_bytes / 4 words per row.
TensorSpec make_flat_dram_tensor_spec(uint32_t page_size_bytes, uint32_t num_pages) {
    const uint32_t page_size_words = page_size_bytes / sizeof(uint32_t);
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{num_pages, page_size_words}, tensor_layout);
}

experimental::DataMovementHardwareConfig make_dm_config(tt::ARCH arch, DataMovementProcessor processor, NOC noc) {
    if (arch == tt::ARCH::QUASAR) {
        return experimental::DataMovementHardwareConfig{
            .gen2_specific = experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true,
            }};
    }
    return experimental::DataMovementHardwareConfig{
        .gen1_specific =
            experimental::DataMovementHardwareConfig::DataMovement1XXConfig{.processor = processor, .noc = noc}};
}

// ---------------------------------------------------------------------------
// Raw-L1 handle coverage: CoreLocalMem / Scratchpad / LocalTensorAccessor as the destination of
// async_write_zeros' local-L1 overload.
// ---------------------------------------------------------------------------

enum class RawL1Target { CoreLocalMem, Scratchpad, LocalTensorAccessor };

struct RawL1Cfg {
    RawL1Target target;
    uint32_t region_bytes;
    uint32_t offset_bytes;
    uint32_t window_bytes;
    uint32_t num_chunks = 0;  // 0 = single zero call; >0 = that many disjoint zeros, one barrier
};

const char* TargetDefine(RawL1Target t) {
    switch (t) {
        case RawL1Target::CoreLocalMem: return "ZERO_TARGET_CORE_LOCAL_MEM";
        case RawL1Target::Scratchpad: return "ZERO_TARGET_SCRATCHPAD";
        case RawL1Target::LocalTensorAccessor: return "ZERO_TARGET_LOCAL_TENSOR_ACCESSOR";
    }
    return "";
}

// Drives zero_memory_api_raw_l1.cpp for one handle type and window geometry, then verifies the
// region's L1 from the host independently of the kernel's own in-kernel check.
void RunRawL1ZeroTest(distributed::MeshDevice& mesh_device, const RawL1Cfg& cfg) {
    const experimental::NodeCoord node{0, 0};

    ASSERT_EQ(cfg.offset_bytes % 16u, 0u) << "window must respect the 16B WH/BH NoC read alignment";
    ASSERT_EQ(cfg.window_bytes % 16u, 0u) << "window must respect the 16B WH/BH NoC read alignment";
    ASSERT_LE(cfg.offset_bytes + cfg.window_bytes, cfg.region_bytes);
    if (cfg.num_chunks > 0) {
        ASSERT_EQ(cfg.window_bytes % cfg.num_chunks, 0u) << "num_chunks must divide window_bytes exactly";
    }

    // Allocate the L1 the host needs to reach: the status/report words, and (CoreLocalMem target
    // only) the region under test. The framework allocates the region for the other two targets.
    auto scratch_words = AllocateL1Scratch(mesh_device, kScratchWordsBytes);
    const uint32_t flag_addr = static_cast<uint32_t>(scratch_words->address());
    const uint32_t report_addr = flag_addr + kReportOffset;

    std::shared_ptr<distributed::MeshBuffer> core_local_region;
    uint32_t core_local_addr = 0;
    if (cfg.target == RawL1Target::CoreLocalMem) {
        core_local_region = AllocateL1Scratch(mesh_device, cfg.region_bytes + kMinAlignNudge);
        core_local_addr = static_cast<uint32_t>(core_local_region->address()) + kMinAlignNudge;
    }

    // Sentinel both status slots, so a kernel that crashes before reporting is caught rather than
    // reading back a stale pass from a previous test in the same binary.
    std::vector<uint32_t> sentinel{0xBAADF00Du};
    slow_dispatch::WriteToL1(mesh_device, node, flag_addr, sentinel);
    slow_dispatch::WriteToL1(mesh_device, node, report_addr, sentinel);

    experimental::KernelSpec kernel{
        .unique_id = RAW_L1_KERNEL,
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_raw_l1.cpp"},
        .num_threads = 1,
        // region_addr is in the schema for every target but only read by the CoreLocalMem kernel
        // variant; the other two learn their region from a binding token. One shared schema avoids a
        // per-target arg list for a single uint32_t.
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"region_bytes", "offset_bytes", "window_bytes", "flag_addr", "report_addr", "region_addr"}},
        .hw_config = make_dm_config(mesh_device.arch(), DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };
    kernel.compiler_options.defines.emplace(TargetDefine(cfg.target), "1");
    if (cfg.num_chunks > 0) {
        kernel.compiler_options.defines.emplace("ZERO_NUM_CHUNKS", std::to_string(cfg.num_chunks));
    }

    experimental::ProgramSpec spec{
        .name = "zero_memory_api_raw_l1",
        .kernels = {kernel},
        .work_units = {{.name = "main", .kernels = {RAW_L1_KERNEL}, .target_nodes = node}},
    };

    // Per-target binding: the handle has to come from somewhere the framework allocated.
    std::optional<MeshTensor> local_tensor;
    if (cfg.target == RawL1Target::Scratchpad) {
        spec.scratchpads = {experimental::ScratchpadSpec{
            .unique_id = experimental::ScratchpadSpecName{"pad"}, .size_per_node = cfg.region_bytes}};
        spec.kernels[0].scratchpad_bindings.push_back(experimental::KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = experimental::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    } else if (cfg.target == RawL1Target::LocalTensorAccessor) {
        // Single-shard L1 tensor sized to region_bytes: BFLOAT16 tiles, so 2 bytes/element.
        const uint32_t elems = cfg.region_bytes / 2;
        ASSERT_EQ(elems % 32u, 0u) << "region must be a whole number of 32-wide tile rows";
        auto tensor_param = MakeShardedTensorParameter("region", Shape{elems / 32, 32}, {elems / 32, 32}, 1);
        spec.tensor_parameters = {tensor_param};
        spec.kernels[0].tensor_bindings = {
            {.tensor_parameter_name = experimental::TensorParamName{"region"}, .accessor_name = "region"}};
        local_tensor.emplace(MeshTensor::allocate_on_device(mesh_device, tensor_param.spec));
    }

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = RAW_L1_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"region_bytes", cfg.region_bytes},
             {"offset_bytes", cfg.offset_bytes},
             {"window_bytes", cfg.window_bytes},
             {"flag_addr", flag_addr},
             {"report_addr", report_addr},
             {"region_addr", core_local_addr}}),
    }};
    if (local_tensor.has_value()) {
        params.tensor_args = {
            {experimental::TensorParamName{"region"}, experimental::ProgramRunArgs::TensorArgument{*local_tensor}}};
    }
    experimental::SetProgramRunArgs(program, params);

    // For the CoreLocalMem target the host owns the region, so seed it to something non-zero that is
    // NOT the kernel's pattern — the kernel's own stamp check then proves it really wrote there.
    if (cfg.target == RawL1Target::CoreLocalMem) {
        std::vector<uint32_t> junk(cfg.region_bytes / sizeof(uint32_t), 0xEEEEEEEEu);
        slow_dispatch::WriteToL1(mesh_device, node, core_local_addr, junk);
    }

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    // ----- In-kernel verdict -----
    std::vector<uint32_t> flag_out;
    slow_dispatch::ReadFromL1(mesh_device, node, flag_addr, sizeof(uint32_t), flag_out);
    ASSERT_EQ(flag_out.size(), 1u);
    EXPECT_EQ(flag_out[0], kStatusOk) << "in-kernel status: " << StatusName(flag_out[0]) << " (0x" << std::hex
                                      << flag_out[0] << ")";

    // ----- Independent host verify of the same region -----
    std::vector<uint32_t> reported;
    slow_dispatch::ReadFromL1(mesh_device, node, report_addr, sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 1u);
    const uint32_t region_addr = reported[0];
    ASSERT_NE(region_addr, 0xBAADF00Du) << "kernel never reported its region address";
    ASSERT_NE(region_addr, 0u) << "kernel reported a 0 region address (binding token not delivered?)";
    if (cfg.target == RawL1Target::LocalTensorAccessor) {
        EXPECT_EQ(region_addr, static_cast<uint32_t>(local_tensor->address()))
            << "LocalTensorAccessor base address does not match the bound tensor";
    }

    std::vector<uint32_t> contents;
    slow_dispatch::ReadFromL1(mesh_device, node, region_addr, cfg.region_bytes, contents);
    ASSERT_EQ(contents.size(), cfg.region_bytes / sizeof(uint32_t));

    const uint32_t window_first = cfg.offset_bytes / sizeof(uint32_t);
    const uint32_t window_last = (cfg.offset_bytes + cfg.window_bytes) / sizeof(uint32_t);
    for (uint32_t i = 0; i < contents.size(); ++i) {
        const bool in_window = (i >= window_first && i < window_last);
        const uint32_t expected = in_window ? 0u : (kPatternBase + i);
        if (contents[i] != expected) {
            ADD_FAILURE() << "word " << i << (in_window ? " (in window)" : " (flank)") << ": expected 0x" << std::hex
                          << expected << ", got 0x" << contents[i];
            return;  // First mismatch is enough.
        }
    }
}

}  // namespace

namespace tt::tt_metal {

TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApi) {
    constexpr uint32_t scratch_bytes = 8 * 1024;
    constexpr uint32_t num_pages = 4;
    constexpr uint32_t page_size_bytes = 4 * 1024;
    constexpr uint32_t total_words = num_pages * (page_size_bytes / sizeof(uint32_t));
    constexpr uint32_t flag_addr = 100 * 1024;  // fixed L1 scratch addr for the status word
    const experimental::NodeCoord node{0, 0};

    // ----- Host stamps -----
    // L1 status flag: sentinel that the kernel demotes to kStatusOk on success.
    std::vector<uint32_t> flag_init{0xBAADF00Du};
    slow_dispatch::WriteToL1(this->device(), node, flag_addr, flag_init);

    // DRAM tensor: 0xFF everywhere, so a no-op kernel can't pass the post-zero check.
    auto tensor =
        MeshTensor::allocate_on_device(this->device(), make_flat_dram_tensor_spec(page_size_bytes, num_pages));
    std::vector<uint32_t> stamped(total_words, 0xFFFFFFFFu);
    slow_dispatch::WriteToBuffer(tensor.mesh_buffer(), stamped);

    // Pre-write verify: confirm the DRAM stamp landed so the post-zero check is meaningful.
    std::vector<uint32_t> stamp_check;
    slow_dispatch::ReadFromBuffer(tensor.mesh_buffer(), stamp_check);
    ASSERT_EQ(stamp_check.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        ASSERT_EQ(stamp_check[i], 0xFFFFFFFFu) << "Pre-write 0xFF stamp did not land at DRAM word " << i;
    }

    // ----- Program spec -----
    experimental::DataflowBufferSpec scratch_spec{
        .unique_id = SCRATCH_DFB,
        .entry_size = scratch_bytes,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Producer: tests overload (1) on the DFB, then push_backs the now-zero entry.
    experimental::KernelSpec producer_spec{
        .unique_id = L1_PRODUCER,
        .source =
            std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_l1_producer.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .runtime_arg_schema = {.runtime_arg_names = {"total_bytes", "flag_addr"}},
        .hw_config = make_dm_config(this->device().arch(), DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    // Consumer: wait_fronts on the L1-zeroed DFB entry, uses it as DRAM scratch for overload (2).
    experimental::KernelSpec consumer_spec{
        .unique_id = DRAM_CONSUMER,
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_consumer.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"ZERO_DRAM", "1"}}},
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"page_start", "page_end", "page_size"}},
        .hw_config = make_dm_config(this->device().arch(), DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };

    experimental::ProgramSpec spec{
        .name = "zero_memory_api_end_to_end",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {scratch_spec},
        .tensor_parameters = {{.unique_id = OUT_TENSOR, .spec = tensor.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {L1_PRODUCER, DRAM_CONSUMER}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = L1_PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"total_bytes", scratch_bytes}, {"flag_addr", flag_addr}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DRAM_CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"page_start", 0u}, {"page_end", num_pages}, {"page_size", page_size_bytes}}),
        },
    };
    params.tensor_args = {{OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{tensor}}};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), workload, /*blocking=*/true);

    // ----- Host verifies -----
    // L1: kernel reports its in-kernel verify result via the flag word.
    std::vector<uint32_t> flag_out;
    slow_dispatch::ReadFromL1(this->device(), node, flag_addr, sizeof(uint32_t), flag_out);
    ASSERT_EQ(flag_out.size(), 1u);
    EXPECT_EQ(flag_out[0], kStatusOk) << "L1 zero test status word was 0x" << std::hex << flag_out[0] << " (expected 0x"
                                      << kStatusOk << ")";

    // DRAM: every word should be zero after overload (2).
    std::vector<uint32_t> result;
    slow_dispatch::ReadFromBuffer(tensor.mesh_buffer(), result);
    ASSERT_EQ(result.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        EXPECT_EQ(result[i], 0u) << "DRAM word " << i << " not zeroed; got 0x" << std::hex << result[i];
        if (result[i] != 0u) {
            return;  // First mismatch is enough; bail to avoid spamming the log.
        }
    }
}

// Batched L1 zeroing: a kernel issues several noc.async_write_zeros() calls into disjoint
// chunks of one DFB entry and then barriers once.
TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiBatchedL1) {
    constexpr uint32_t scratch_bytes = 32 * 1024;
    constexpr uint32_t num_chunks = 4;  // 4 disjoint 8 KB L1 zeros, then a single barrier
    constexpr uint32_t num_pages = 4;
    constexpr uint32_t page_size_bytes = 4 * 1024;
    constexpr uint32_t total_words = num_pages * (page_size_bytes / sizeof(uint32_t));
    constexpr uint32_t flag_addr = 100 * 1024;
    const experimental::NodeCoord node{0, 0};

    // L1 status flag: sentinel that the batched producer demotes to kStatusOk on success.
    std::vector<uint32_t> flag_init{0xBAADF00Du};
    slow_dispatch::WriteToL1(this->device(), node, flag_addr, flag_init);

    auto tensor =
        MeshTensor::allocate_on_device(this->device(), make_flat_dram_tensor_spec(page_size_bytes, num_pages));
    std::vector<uint32_t> stamped(total_words, 0xFFFFFFFFu);
    slow_dispatch::WriteToBuffer(tensor.mesh_buffer(), stamped);

    experimental::DataflowBufferSpec scratch_spec{
        .unique_id = SCRATCH_DFB,
        .entry_size = scratch_bytes,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::KernelSpec producer_spec{
        .unique_id = L1_BATCHED_PRODUCER,
        .source =
            std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_l1_producer.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"ZERO_NUM_CHUNKS", std::to_string(num_chunks)}}},
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .runtime_arg_schema = {.runtime_arg_names = {"total_bytes", "flag_addr"}},
        .hw_config = make_dm_config(this->device().arch(), DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };

    experimental::KernelSpec consumer_spec{
        .unique_id = DRAM_CONSUMER,
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_consumer.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"ZERO_DRAM", "1"}}},
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"page_start", "page_end", "page_size"}},
        .hw_config = make_dm_config(this->device().arch(), DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };

    experimental::ProgramSpec spec{
        .name = "zero_memory_api_batched_l1",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {scratch_spec},
        .tensor_parameters = {{.unique_id = OUT_TENSOR, .spec = tensor.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {L1_BATCHED_PRODUCER, DRAM_CONSUMER}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = L1_BATCHED_PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"total_bytes", scratch_bytes}, {"flag_addr", flag_addr}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DRAM_CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"page_start", 0u}, {"page_end", num_pages}, {"page_size", page_size_bytes}}),
        },
    };
    params.tensor_args = {{OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{tensor}}};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), workload, /*blocking=*/true);

    // The batched producer's in-kernel verify is the primary signal: kStatusOk only if
    // every byte across all chunks is zero after the single barrier.
    std::vector<uint32_t> flag_out;
    slow_dispatch::ReadFromL1(this->device(), node, flag_addr, sizeof(uint32_t), flag_out);
    ASSERT_EQ(flag_out.size(), 1u);
    EXPECT_EQ(flag_out[0], kStatusOk) << "Batched L1 zero status word was 0x" << std::hex << flag_out[0]
                                      << " (expected 0x" << kStatusOk << "); a non-OK value means a batched zero was "
                                      << "lost (stale bytes remained after the single barrier).";
}

// ---------------------------------------------------------------------------
// Raw-L1 handles as the local-L1 destination.
// ---------------------------------------------------------------------------

TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiCoreLocalMemL1) {
    RunRawL1ZeroTest(
        this->device(),
        {.target = RawL1Target::CoreLocalMem, .region_bytes = 256, .offset_bytes = 32, .window_bytes = 80});
}

TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiScratchpadL1) {
    RunRawL1ZeroTest(
        this->device(),
        {.target = RawL1Target::Scratchpad, .region_bytes = 256, .offset_bytes = 32, .window_bytes = 80});
}

TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiLocalTensorAccessorL1) {
    // 2048 B = one 32x32 BFLOAT16 tile.
    RunRawL1ZeroTest(
        this->device(),
        {.target = RawL1Target::LocalTensorAccessor, .region_bytes = 2048, .offset_bytes = 64, .window_bytes = 512});
}

// Window larger than MEM_ZEROS_SIZE (512 B), so the WH/BH implementation takes its chunked
// set_async_read_state / async_read_with_state loop rather than the single-read tail.
TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiRawL1LargeWindow) {
    RunRawL1ZeroTest(
        this->device(),
        {.target = RawL1Target::Scratchpad, .region_bytes = 8192, .offset_bytes = 1024, .window_bytes = 4096});
}

// N disjoint zeros into one region, then a single barrier.
TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiRawL1Batched) {
    RunRawL1ZeroTest(
        this->device(),
        {.target = RawL1Target::Scratchpad,
         .region_bytes = 8192,
         .offset_bytes = 1024,
         .window_bytes = 4096,
         .num_chunks = 4});
}

// Zeros streamed to DRAM from a raw L1 handle, with no CB/DFB in the program.
static void RunDramFromRawL1Test(distributed::MeshDevice& mesh_device, RawL1Target target) {
    // The pre-zeroed prefix must cover min(page_size, NOC_MAX_BURST_SIZE) -- the largest single burst
    // overload (2) issues from the (never-advancing) scratch address. NOC_MAX_BURST_SIZE is 512 B on
    // WH but 16 KB on BH and 64 KB on Quasar, so on those two that minimum is the WHOLE page: size the
    // region to a full page, or the burst reads past the region and streams whatever follows it in L1
    // to DRAM. A smaller region can still pass by luck when the bytes after it happen to be zero.
    // Also >= one 32x32 BFLOAT16 tile so the LocalTensorAccessor target is a legal TILE-layout tensor.
    constexpr uint32_t region_bytes = 4096;
    constexpr uint32_t num_pages = 4;
    constexpr uint32_t page_size_bytes = 4 * 1024;
    constexpr uint32_t total_words = num_pages * (page_size_bytes / sizeof(uint32_t));
    const experimental::NodeCoord node{0, 0};

    auto scratch_words = AllocateL1Scratch(mesh_device, kScratchWordsBytes);
    const uint32_t flag_addr = static_cast<uint32_t>(scratch_words->address());

    std::shared_ptr<distributed::MeshBuffer> core_local_region;
    uint32_t core_local_addr = 0;
    if (target == RawL1Target::CoreLocalMem) {
        core_local_region = AllocateL1Scratch(mesh_device, region_bytes + kMinAlignNudge);
        core_local_addr = static_cast<uint32_t>(core_local_region->address()) + kMinAlignNudge;
    }

    std::vector<uint32_t> sentinel{0xBAADF00Du};
    slow_dispatch::WriteToL1(mesh_device, node, flag_addr, sentinel);

    // No host-side seeding needed: the kernel stamps the scratch non-zero and verifies the stamp
    // before zeroing it, so the all-zero check proves overload (1) ran for every target type.

    auto tensor = MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(page_size_bytes, num_pages));
    std::vector<uint32_t> stamped(total_words, 0xFFFFFFFFu);
    slow_dispatch::WriteToBuffer(tensor.mesh_buffer(), stamped);

    // Confirm the stamp landed, so an all-zeros result afterwards is meaningful.
    std::vector<uint32_t> stamp_check;
    slow_dispatch::ReadFromBuffer(tensor.mesh_buffer(), stamp_check);
    ASSERT_EQ(stamp_check.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        ASSERT_EQ(stamp_check[i], 0xFFFFFFFFu) << "Pre-write 0xFF stamp did not land at DRAM word " << i;
    }

    experimental::KernelSpec kernel{
        .unique_id = RAW_L1_KERNEL,
        .source =
            std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_dram_from_raw_l1.cpp"},
        .num_threads = 1,
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"}},
        // region_addr is in the schema for every target but only read by the CoreLocalMem kernel
        // variant; the other two learn their region from a binding token. One shared schema avoids a
        // per-target arg list for a single uint32_t.
        .runtime_arg_schema =
            {.runtime_arg_names = {"page_start", "page_end", "page_size", "flag_addr", "region_bytes", "region_addr"}},
        .hw_config = make_dm_config(mesh_device.arch(), DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };
    kernel.compiler_options.defines.emplace(TargetDefine(target), "1");

    experimental::ProgramSpec spec{
        .name = "zero_memory_api_dram_from_raw_l1",
        .kernels = {kernel},
        .tensor_parameters = {{.unique_id = OUT_TENSOR, .spec = tensor.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {RAW_L1_KERNEL}, .target_nodes = node}},
    };

    // Per-target binding for the scratch region.
    std::optional<MeshTensor> local_tensor;
    if (target == RawL1Target::Scratchpad) {
        spec.scratchpads = {experimental::ScratchpadSpec{
            .unique_id = experimental::ScratchpadSpecName{"pad"}, .size_per_node = region_bytes}};
        spec.kernels[0].scratchpad_bindings.push_back(experimental::KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = experimental::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    } else if (target == RawL1Target::LocalTensorAccessor) {
        // region_bytes of BFLOAT16 (2 B/elem) as whole 32-wide tile rows.
        const uint32_t elems = region_bytes / 2;
        auto region_param = MakeShardedTensorParameter("region", Shape{elems / 32, 32}, {elems / 32, 32}, 1);
        spec.tensor_parameters.push_back(region_param);
        spec.kernels[0].tensor_bindings.push_back(
            {.tensor_parameter_name = experimental::TensorParamName{"region"}, .accessor_name = "region"});
        local_tensor.emplace(MeshTensor::allocate_on_device(mesh_device, region_param.spec));
    }

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = RAW_L1_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"page_start", 0u},
             {"page_end", num_pages},
             {"page_size", page_size_bytes},
             {"flag_addr", flag_addr},
             {"region_bytes", region_bytes},
             {"region_addr", core_local_addr}}),
    }};
    params.tensor_args = {{OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{tensor}}};
    if (local_tensor.has_value()) {
        params.tensor_args.emplace(
            experimental::TensorParamName{"region"}, experimental::ProgramRunArgs::TensorArgument{*local_tensor});
    }
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> flag_out;
    slow_dispatch::ReadFromL1(mesh_device, node, flag_addr, sizeof(uint32_t), flag_out);
    ASSERT_EQ(flag_out.size(), 1u);
    EXPECT_EQ(flag_out[0], kStatusOk) << "in-kernel status: " << StatusName(flag_out[0]) << " (0x" << std::hex
                                      << flag_out[0] << ")";

    std::vector<uint32_t> result;
    slow_dispatch::ReadFromBuffer(tensor.mesh_buffer(), result);
    ASSERT_EQ(result.size(), total_words);
    for (uint32_t i = 0; i < total_words; ++i) {
        if (result[i] != 0u) {
            ADD_FAILURE() << "DRAM word " << i << " not zeroed; got 0x" << std::hex << result[i];
            return;
        }
    }
}

TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiDramFromScratchpad) {
    RunDramFromRawL1Test(this->device(), RawL1Target::Scratchpad);
}

TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiDramFromCoreLocalMem) {
    RunDramFromRawL1Test(this->device(), RawL1Target::CoreLocalMem);
}

TEST_F(UnitMeshAnyDispatchFixture, ZeroMemoryApiDramFromLocalTensorAccessor) {
    RunDramFromRawL1Test(this->device(), RawL1Target::LocalTensorAccessor);
}

}  // namespace tt::tt_metal
