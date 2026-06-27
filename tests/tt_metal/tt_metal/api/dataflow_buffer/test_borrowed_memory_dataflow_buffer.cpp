// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include "device_fixture.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"

namespace tt::tt_metal {
namespace {

using namespace experimental;
using test_helpers::MakeMinimalComputeKernel;
using test_helpers::MakeMinimalDMKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

// Kernel paths shared with the standard DFB tests.
constexpr const char* DFB_PRODUCER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp";
constexpr const char* DFB_DM_CONSUMER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp";
constexpr const char* DFB_TENSIX_CONSUMER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp";

inline TensorSpec make_flat_dram_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    const uint32_t entry_size_words = entry_size / sizeof(uint32_t);
    auto page_config   = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{total_entries, entry_size_words}, tensor_layout);
}

inline TensorSpec make_flat_l1_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    // Use a single page covering the entire allocation so that aligned_size_per_bank()
    // equals the total size.  A multi-page INTERLEAVED buffer distributes pages across
    // distinct L1 banks (cores), leaving only one page's worth of bytes per bank and
    // failing the borrowed-DFB size check in AttachBorrowedDFBBuffers.
    const uint32_t total_words = total_entries * entry_size / sizeof(uint32_t);
    auto page_config   = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
    auto tensor_layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{1, total_words}, tensor_layout);
}

struct BorrowedDFBTestConfig {
    uint32_t num_entries      = 16;
    uint32_t entry_size       = 256;   // bytes
    uint32_t num_producers    = 1;
    uint32_t num_consumers    = 1;
    DFBAccessPattern cap      = DFBAccessPattern::STRIDED;  // producer is always STRIDED
    bool tensix_consumer      = false;
    bool verify_data          = false;
    std::optional<uint32_t> num_entries_override = std::nullopt;
    bool expect_attach_fatal = false;
};

// Runs a borrowed-memory DFB program and asserts:
//   1. The DFB ring was allocated at the borrowed MeshTensor's L1 address.
//   2. (Optional) DRAM output == DRAM input.
void run_borrowed_memory_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const NodeCoord& node,
    const BorrowedDFBTestConfig& cfg) {

    IDevice* device = mesh_device->get_devices()[0];
    const ARCH arch = device->arch();
    const bool is_all = (cfg.cap == DFBAccessPattern::ALL);

    constexpr uint32_t implicit_sync = 0;

    const uint32_t entries_per_producer =
        (cfg.num_entries + cfg.num_producers - 1) / cfg.num_producers;
    const uint32_t entries_per_consumer =
        is_all ? cfg.num_entries
               : (cfg.num_entries + cfg.num_consumers - 1) / cfg.num_consumers;

    // For a single-core run: chunk_offset=0, entries_per_core=num_entries.
    const uint32_t entries_per_core = cfg.num_entries;

    // -----------------------------------------------------------------------
    // Build ProgramSpec
    // -----------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "borrowed_memory_dfb_test";

    // --- Producer kernel (dfb_producer.cpp) ---
    KernelSpec producer_spec = (arch == ARCH::QUASAR)
        ? MakeMinimalDMKernel("producer", static_cast<uint8_t>(cfg.num_producers))
        : MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer_spec.source = DFB_PRODUCER_KERNEL;
    producer_spec.compile_time_args = {
        {"num_entries_per_producer", entries_per_producer},
        {"implicit_sync",            implicit_sync},
        {"num_producers",            cfg.num_producers},
    };
    producer_spec.runtime_arg_schema.runtime_arg_names = {"chunk_offset", "entries_per_core"};
    producer_spec.tensor_bindings = {
        {.tensor_parameter_name = experimental::TensorParamName{"src_tensor"}, .accessor_name = "src_tensor"},
        // dfb_ring_tensor backs the borrowed DFB; the kernel does not access it directly,
        // but every TensorParameter must be bound to at least one kernel.
        {.tensor_parameter_name = experimental::TensorParamName{"dfb_ring_tensor"}, .accessor_name = "dfb_ring"},
    };
    producer_spec.dfb_bindings.push_back(ProducerOf(experimental::DFBSpecName{"borrowed_dfb"}, "out"));

    // --- Consumer kernel ---
    KernelSpec consumer_spec;
    if (cfg.tensix_consumer) {
        // dfb_t6_consumer.cpp: drains the DFB, does not write to DRAM.
        consumer_spec = (arch == ARCH::QUASAR)
            ? MakeMinimalComputeKernel("consumer", static_cast<uint8_t>(cfg.num_consumers))
            : MakeMinimalComputeKernel("consumer");
        consumer_spec.source = DFB_TENSIX_CONSUMER_KERNEL;
        consumer_spec.compile_time_args = {
            {"num_entries_per_consumer", entries_per_consumer},
        };
    } else {
        // dfb_consumer.cpp: reads DFB and writes to dst_tensor.
        consumer_spec = (arch == ARCH::QUASAR)
            ? MakeMinimalDMKernel("consumer", static_cast<uint8_t>(cfg.num_consumers))
            : MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
        consumer_spec.source = DFB_DM_CONSUMER_KERNEL;
        consumer_spec.compile_time_args = {
            {"num_entries_per_consumer", entries_per_consumer},
            {"blocked_consumer",         static_cast<uint32_t>(is_all ? 1u : 0u)},
            {"implicit_sync",            implicit_sync},
            {"num_consumers",            cfg.num_consumers},
        };
        consumer_spec.runtime_arg_schema.runtime_arg_names = {"chunk_offset", "entries_per_core"};
        consumer_spec.tensor_bindings = {{
            .tensor_parameter_name = experimental::TensorParamName{"dst_tensor"},
            .accessor_name = "dst_tensor",
        }};
    }
    consumer_spec.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = experimental::DFBSpecName{"borrowed_dfb"},
        .accessor_name = "in",
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = cfg.cap,
    });

    // Disable implicit sync on the borrowed DFB for every DM endpoint (Gen2 only;
    // Gen1 has no ISR-based implicit sync to opt out of).
    if (arch == ARCH::QUASAR) {
        std::get<DataMovementHardwareConfig>(producer_spec.hw_config)
            .gen2_config->disable_implicit_sync_for.push_back(experimental::DFBSpecName{"borrowed_dfb"});
        if (!cfg.tensix_consumer) {
            std::get<DataMovementHardwareConfig>(consumer_spec.hw_config)
                .gen2_config->disable_implicit_sync_for.push_back(experimental::DFBSpecName{"borrowed_dfb"});
        }
    }

    // --- Borrowed DFB spec ---
    DataflowBufferSpec dfb_spec{
        .unique_id = experimental::DFBSpecName{"borrowed_dfb"},
        .entry_size = cfg.entry_size,
        .num_entries = cfg.num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .borrowed_from = experimental::TensorParamName{"dfb_ring_tensor"},
    };

    // --- TensorParameters ---
    const TensorSpec src_spec  = make_flat_dram_tensor_spec(cfg.entry_size, cfg.num_entries);
    const TensorSpec dst_spec  = make_flat_dram_tensor_spec(cfg.entry_size, cfg.num_entries);
    const TensorSpec ring_spec = make_flat_l1_tensor_spec(cfg.entry_size, cfg.num_entries);

    spec.tensor_parameters.push_back({.unique_id = experimental::TensorParamName{"src_tensor"}, .spec = src_spec});
    if (!cfg.tensix_consumer) {
        spec.tensor_parameters.push_back({.unique_id = experimental::TensorParamName{"dst_tensor"}, .spec = dst_spec});
    }
    spec.tensor_parameters.push_back(
        {.unique_id = experimental::TensorParamName{"dfb_ring_tensor"}, .spec = ring_spec});

    spec.kernels          = {producer_spec, consumer_spec};
    spec.dataflow_buffers = {dfb_spec};
    spec.work_units       = {MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    // -----------------------------------------------------------------------
    // Create program and allocate tensors
    // -----------------------------------------------------------------------
    Program program = MakeProgramFromSpec(*mesh_device, spec);

    MeshTensor src_tensor =
        MeshTensor::allocate_on_device(*mesh_device, src_spec, TensorTopology{});
    std::optional<MeshTensor> dst_tensor;
    if (!cfg.tensix_consumer) {
        dst_tensor.emplace(
            MeshTensor::allocate_on_device(*mesh_device, dst_spec, TensorTopology{}));
    }
    MeshTensor ring_tensor =
        MeshTensor::allocate_on_device(*mesh_device, ring_spec, TensorTopology{});

    // -----------------------------------------------------------------------
    // Build and apply run params
    // -----------------------------------------------------------------------
    using NodeRuntimeArgs = decltype(ProgramRunArgs::KernelRunArgs::runtime_arg_values)::value_type;
    const NodeRuntimeArgs dm_rtas{node, {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}};

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = experimental::KernelSpecName{"producer"},
        .runtime_arg_values = {dm_rtas},
    });
    if (cfg.tensix_consumer) {
        params.kernel_run_args.push_back(
            ProgramRunArgs::KernelRunArgs{.kernel = experimental::KernelSpecName{"consumer"}});
    } else {
        params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"consumer"},
            .runtime_arg_values = {dm_rtas},
        });
    }
    params.tensor_args.emplace(experimental::TensorParamName{"src_tensor"}, TensorArgument{src_tensor});
    if (!cfg.tensix_consumer) {
        params.tensor_args.emplace(experimental::TensorParamName{"dst_tensor"}, TensorArgument{*dst_tensor});
    }
    params.tensor_args.emplace(experimental::TensorParamName{"dfb_ring_tensor"}, TensorArgument{ring_tensor});

    // Issue #45925: optional mutable-size override on the borrowed DFB (applied inside
    // SetProgramRunArgs before the borrowed buffer is attached).
    if (cfg.num_entries_override.has_value()) {
        params.dfb_run_overrides.push_back(
            {.dfb = experimental::DFBSpecName{"borrowed_dfb"}, .num_entries = cfg.num_entries_override});
    }
    if (cfg.expect_attach_fatal) {
        // Oversized borrowed resize: rejected by the per-bank fit check in AttachBorrowedDFBBuffers
        // (no launch). This check is unique to borrowed memory.
        EXPECT_THAT(
            [&] { SetProgramRunArgs(program, params); },
            ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("exceeds the borrowed Buffer's per-bank size")));
        return;
    }
    SetProgramRunArgs(program, params);

    // -----------------------------------------------------------------------
    // Write input, launch, verify
    // -----------------------------------------------------------------------
    std::vector<uint32_t> input(cfg.num_entries * cfg.entry_size / sizeof(uint32_t));
    std::iota(input.begin(), input.end(), 0u);
    detail::WriteToBuffer(*src_tensor.mesh_buffer().get_reference_buffer(), input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    // Assert the borrowed tensor's L1 address was used for the DFB ring. For a borrowed DFB this
    // stays PINNED across a size override (no reallocation).
    EXPECT_EQ(
        program.impl().dataflow_buffers()[0]->uniform_alloc_addr(),
        static_cast<uint32_t>(ring_tensor.address()));

    if (cfg.num_entries_override.has_value()) {
        EXPECT_EQ(program.impl().dataflow_buffers()[0]->config.num_entries, *cfg.num_entries_override)
            << "borrowed DFB num_entries override was not applied";
    }

    if (cfg.verify_data && !cfg.tensix_consumer) {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*dst_tensor->mesh_buffer().get_reference_buffer(), output);
        EXPECT_EQ(input, output);
    }
}

// Verifies that UpdateTensorArgs can redirect a borrowed DFB ring to a different
// L1 tensor between runs on the same compiled program.
void run_update_address_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const NodeCoord& node,
    std::optional<uint32_t> reentry_num_entries_override = std::nullopt) {
    IDevice* device = mesh_device->get_devices()[0];
    const ARCH arch = device->arch();

    constexpr uint32_t num_entries  = 16;
    constexpr uint32_t entry_size   = 256;
    constexpr uint32_t total_words  = num_entries * entry_size / sizeof(uint32_t);
    constexpr uint32_t implicit_sync = 0;

    // -----------------------------------------------------------------------
    // Build ProgramSpec
    // -----------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "borrowed_dfb_update_address";

    KernelSpec producer_spec = (arch == ARCH::QUASAR)
        ? MakeMinimalDMKernel("producer")
        : MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer_spec.source = DFB_PRODUCER_KERNEL;
    producer_spec.compile_time_args = {
        {"num_entries_per_producer", num_entries},
        {"implicit_sync",            implicit_sync},
        {"num_producers",            1u},
    };
    producer_spec.runtime_arg_schema.runtime_arg_names = {"chunk_offset", "entries_per_core"};
    producer_spec.tensor_bindings = {
        {.tensor_parameter_name = experimental::TensorParamName{"src_tensor"}, .accessor_name = "src_tensor"},
        {.tensor_parameter_name = experimental::TensorParamName{"dfb_ring_tensor"}, .accessor_name = "dfb_ring"},
    };
    producer_spec.dfb_bindings.push_back(ProducerOf(experimental::DFBSpecName{"borrowed_dfb"}, "out"));

    KernelSpec consumer_spec = (arch == ARCH::QUASAR)
        ? MakeMinimalDMKernel("consumer")
        : MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer_spec.source = DFB_DM_CONSUMER_KERNEL;
    consumer_spec.compile_time_args = {
        {"num_entries_per_consumer", num_entries},
        {"blocked_consumer",         0u},
        {"implicit_sync",            implicit_sync},
        {"num_consumers",            1u},
    };
    consumer_spec.runtime_arg_schema.runtime_arg_names = {"chunk_offset", "entries_per_core"};
    consumer_spec.tensor_bindings = {{
        .tensor_parameter_name = experimental::TensorParamName{"dst_tensor"},
        .accessor_name = "dst_tensor",
    }};
    consumer_spec.dfb_bindings.push_back(ConsumerOf(experimental::DFBSpecName{"borrowed_dfb"}, "in"));

    // Disable implicit sync on the borrowed DFB for both DM endpoints (Gen2 only;
    // Gen1 has no ISR-based implicit sync to opt out of).
    if (arch == ARCH::QUASAR) {
        std::get<DataMovementHardwareConfig>(producer_spec.hw_config)
            .gen2_config->disable_implicit_sync_for.push_back(experimental::DFBSpecName{"borrowed_dfb"});
        std::get<DataMovementHardwareConfig>(consumer_spec.hw_config)
            .gen2_config->disable_implicit_sync_for.push_back(experimental::DFBSpecName{"borrowed_dfb"});
    }

    DataflowBufferSpec dfb_spec{
        .unique_id = experimental::DFBSpecName{"borrowed_dfb"},
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .borrowed_from = experimental::TensorParamName{"dfb_ring_tensor"},
    };

    const TensorSpec src_spec  = make_flat_dram_tensor_spec(entry_size, num_entries);
    const TensorSpec dst_spec  = make_flat_dram_tensor_spec(entry_size, num_entries);
    const TensorSpec ring_spec = make_flat_l1_tensor_spec(entry_size, num_entries);

    spec.tensor_parameters = {
        {.unique_id = experimental::TensorParamName{"src_tensor"}, .spec = src_spec},
        {.unique_id = experimental::TensorParamName{"dst_tensor"}, .spec = dst_spec},
        {.unique_id = experimental::TensorParamName{"dfb_ring_tensor"}, .spec = ring_spec},
    };
    spec.kernels          = {producer_spec, consumer_spec};
    spec.dataflow_buffers = {dfb_spec};
    spec.work_units       = {MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    MeshTensor src_tensor = MeshTensor::allocate_on_device(*mesh_device, src_spec, TensorTopology{});
    MeshTensor dst_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_spec, TensorTopology{});

    // Two distinct L1 ring tensors - swapped between runs.
    MeshTensor ring_tensor_a = MeshTensor::allocate_on_device(*mesh_device, ring_spec, TensorTopology{});
    MeshTensor ring_tensor_b = MeshTensor::allocate_on_device(*mesh_device, ring_spec, TensorTopology{});
    ASSERT_NE(ring_tensor_a.address(), ring_tensor_b.address())
        << "Test pre-condition: two separate L1 allocations must have distinct addresses";

    using NodeRuntimeArgs = decltype(ProgramRunArgs::KernelRunArgs::runtime_arg_values)::value_type;
    const NodeRuntimeArgs dm_rtas{node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}};

    // --- Run 1: ring at ring_tensor_a ---
    std::vector<uint32_t> input_a(total_words);
    std::iota(input_a.begin(), input_a.end(), 0u);
    detail::WriteToBuffer(*src_tensor.mesh_buffer().get_reference_buffer(), input_a);

    ProgramRunArgs params1;
    params1.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"producer"},
            .runtime_arg_values = {dm_rtas},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"consumer"},
            .runtime_arg_values = {dm_rtas},
        },
    };
    params1.tensor_args = {
        {experimental::TensorParamName{"src_tensor"}, TensorArgument{src_tensor}},
        {experimental::TensorParamName{"dst_tensor"}, TensorArgument{dst_tensor}},
        {experimental::TensorParamName{"dfb_ring_tensor"}, TensorArgument{ring_tensor_a}},
    };
    SetProgramRunArgs(program, params1);
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    EXPECT_EQ(
        program.impl().dataflow_buffers()[0]->uniform_alloc_addr(),
        static_cast<uint32_t>(ring_tensor_a.address()));
    {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*dst_tensor.mesh_buffer().get_reference_buffer(), output);
        EXPECT_EQ(input_a, output);
    }

    // --- Run 2: ring redirected to ring_tensor_b ---
    // When reentry_num_entries_override is set, Run 2 re-binds the ring to a different tensor AND
    // applies a num_entries override in the same SetProgramRunArgs. When unset, Run 2 uses the
    // UpdateTensorArgs fast-path (address only).
    std::vector<uint32_t> input_b(total_words);
    std::iota(input_b.begin(), input_b.end(), total_words);  // distinct from run 1
    detail::WriteToBuffer(*src_tensor.mesh_buffer().get_reference_buffer(), input_b);

    if (reentry_num_entries_override.has_value()) {
        // Combined re-bind + resize in ONE SetProgramRunArgs: point the borrowed ring at a different
        // L1 tensor AND override num_entries together (the sharded program-cache-hit analog). The
        // per-bank fit check in AttachBorrowedDFBBuffers validates the new size against ring_tensor_b.
        const NodeRuntimeArgs dm_rtas2{node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}};
        ProgramRunArgs params2;
        params2.kernel_run_args = {
            {.kernel = experimental::KernelSpecName{"producer"}, .runtime_arg_values = {dm_rtas2}},
            {.kernel = experimental::KernelSpecName{"consumer"}, .runtime_arg_values = {dm_rtas2}},
        };
        params2.tensor_args = {
            {experimental::TensorParamName{"src_tensor"}, TensorArgument{src_tensor}},
            {experimental::TensorParamName{"dst_tensor"}, TensorArgument{dst_tensor}},
            {experimental::TensorParamName{"dfb_ring_tensor"}, TensorArgument{ring_tensor_b}},
        };
        params2.dfb_run_overrides.push_back(
            {.dfb = experimental::DFBSpecName{"borrowed_dfb"}, .num_entries = reentry_num_entries_override});
        SetProgramRunArgs(program, params2);
    } else {
        UpdateTensorArgs(
            program,
            Table<experimental::TensorParamName, TensorArgument>{
                {experimental::TensorParamName{"src_tensor"}, TensorArgument{src_tensor}},
                {experimental::TensorParamName{"dst_tensor"}, TensorArgument{dst_tensor}},
                {experimental::TensorParamName{"dfb_ring_tensor"}, TensorArgument{ring_tensor_b}},
            });
    }
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    EXPECT_EQ(
        program.impl().dataflow_buffers()[0]->uniform_alloc_addr(),
        static_cast<uint32_t>(ring_tensor_b.address()));
    if (reentry_num_entries_override.has_value()) {
        EXPECT_EQ(program.impl().dataflow_buffers()[0]->config.num_entries, *reentry_num_entries_override)
            << "combined re-bind + resize: num_entries override was not applied";
    }
    {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*dst_tensor.mesh_buffer().get_reference_buffer(), output);
        EXPECT_EQ(input_b, output);
    }
}

}  // anonymous namespace

// =============================================================================
// All-architecture tests (Quasar, Wormhole, Blackhole)
// =============================================================================

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM1Sx1S) {
    run_borrowed_memory_dfb_program(devices_.at(0), NodeCoord{0, 0}, {
        .num_entries     = 16,
        .entry_size      = 256,
        .num_producers   = 1,
        .num_consumers   = 1,
        .cap             = DFBAccessPattern::STRIDED,
        .tensix_consumer = false,
        .verify_data     = true,
    });
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM1Sx1S_UpdateAddress) {
    run_update_address_test(devices_.at(0), NodeCoord{0, 0});
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM1Sx1S_UpdateAddressAndSize) {
    run_update_address_test(devices_.at(0), NodeCoord{0, 0}, /*reentry_num_entries_override=*/8);
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM1Sx1S_NumEntriesOverride) {
    run_borrowed_memory_dfb_program(
        devices_.at(0),
        NodeCoord{0, 0},
        {
            .num_entries = 16,
            .entry_size = 256,
            .num_producers = 1,
            .num_consumers = 1,
            .cap = DFBAccessPattern::STRIDED,
            .tensix_consumer = false,
            .verify_data = true,
            .num_entries_override = 8,  // 8*256 = 2048 B fits the 16*256 = 4096 B backing tensor
        });
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM1Sx1S_SizeOverrideExceedsBackingFails) {
    run_borrowed_memory_dfb_program(
        devices_.at(0),
        NodeCoord{0, 0},
        {
            .num_entries = 16,
            .entry_size = 256,
            .num_producers = 1,
            .num_consumers = 1,
            .cap = DFBAccessPattern::STRIDED,
            .tensix_consumer = false,
            .verify_data = false,
            .num_entries_override = 32,  // 32*256 = 8192 B > 16*256 = 4096 B backing -> FATAL
            .expect_attach_fatal = true,
        });
}

// =============================================================================
// Quasar-only tests (multi-producer / multi-consumer, explicit sync)
// =============================================================================

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB requires Quasar";
    }
    run_borrowed_memory_dfb_program(devices_.at(0), NodeCoord{0, 0}, {
        .num_entries     = 16,
        .entry_size      = 256,
        .num_producers   = 2,
        .num_consumers   = 4,
        .cap             = DFBAccessPattern::STRIDED,
        .tensix_consumer = false,
        .verify_data     = true,
    });
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM4Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB requires Quasar";
    }
    run_borrowed_memory_dfb_program(devices_.at(0), NodeCoord{0, 0}, {
        .num_entries     = 16,
        .entry_size      = 256,
        .num_producers   = 4,
        .num_consumers   = 2,
        .cap             = DFBAccessPattern::STRIDED,
        .tensix_consumer = false,
        .verify_data     = true,
    });
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMTensix2Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB requires Quasar";
    }
    run_borrowed_memory_dfb_program(devices_.at(0), NodeCoord{0, 0}, {
        .num_entries     = 16,
        .entry_size      = 256,
        .num_producers   = 2,
        .num_consumers   = 4,
        .cap             = DFBAccessPattern::ALL,
        .tensix_consumer = true,
        .verify_data     = false,
    });
}

}  // namespace tt::tt_metal
