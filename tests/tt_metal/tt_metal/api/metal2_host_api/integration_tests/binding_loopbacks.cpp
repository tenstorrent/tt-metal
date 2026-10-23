// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end resource bindings on WH/BH silicon: DFB, semaphore and tensor accessor names.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/bfloat16.hpp>

#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ProgramSpecHWTest;

// ============================================================================
// DFB Local Accessor Name Loopback Test
// ============================================================================
//
// Proves that DFB local accessor names work end-to-end on real WH/BH hardware:
//   1. kernel_bindings_generated.h is emitted correctly (dfb::buf resolves at compile time)
//   2. The DFBBindingToken mechanism works (DFB ID maps to the correct underlying CB)
//   3. Data flows correctly through the DFB from producer to consumer
//
// Pipeline:
//   Host writes random data → DRAM input buffer (single page = one bank)
//   Producer DM kernel (BRISC) reads DRAM → DFB (using dfb::buf)
//   Consumer DM kernel (NCRISC) reads DFB → DRAM (using dfb::buf)
//   Host reads DRAM output buffer and verifies match

TEST_F(ProgramSpecHWTest, DFBAccessorNameLoopback) {
    auto mesh_device = devices_.at(0);

    // Test parameters
    constexpr uint32_t entry_size = 1024;  // bytes per DFB entry
    constexpr uint32_t num_entries = 4;    // DFB depth (double-buffer + margin)
    constexpr uint32_t num_transfers = 8;  // total entries to move through the DFB
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    // Use a single core for simplicity
    const NodeCoord node{0, 0};

    // -------------------------------------------------------
    // Create DRAM buffers (single-page so all data is on one bank)
    // -------------------------------------------------------
    auto input_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto output_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto& cq = mesh_device->mesh_command_queue();

    // -------------------------------------------------------
    // Build ProgramSpec
    // -------------------------------------------------------
    ProgramSpec spec;
    spec.name = "dfb_accessor_loopback";

    // Producer: BRISC reads from DRAM → DFB
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 3;

    // Consumer: NCRISC reads DFB → DRAM
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 3;

    // DFB: both kernels bind it, with different local accessor names
    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, num_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "a_dfb_named_bob"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    // -------------------------------------------------------
    // Create Program
    // -------------------------------------------------------
    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // -------------------------------------------------------
    // Set runtime args
    // -------------------------------------------------------
    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              input_buffer->address(),
                              0u,  // bank_id (single-page buffer → bank 0)
                              num_transfers,
                          }}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              output_buffer->address(),
                              0u,  // bank_id
                              num_transfers,
                          }}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    // -------------------------------------------------------
    // Fill input buffer with known data
    // -------------------------------------------------------
    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data, /*blocking=*/true);

    // -------------------------------------------------------
    // Dispatch
    // -------------------------------------------------------
    LaunchProgram(*mesh_device, std::move(program));

    // -------------------------------------------------------
    // Verify
    // -------------------------------------------------------
    std::vector<uint32_t> output_data;
    distributed::EnqueueReadMeshBuffer(cq, output_data, output_buffer, /*blocking=*/true);

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// Semaphore Accessor Name Loopback Test
// ============================================================================
//
// Targeted test for the semaphore accessor name plumbing.
// Very minimal: just one semaphore and two kernels that sync on it via
// different local accessor names.
//
//   - Producer (BRISC) and consumer (NCRISC) each resolve their accessor name
//     to a sem ID. Test only completes if both land on the same underlying ID.
//   - If producer's sem::signal ID != consumer's sem::waiter ID, consumer hangs
//     forever on wait(1).
//
// Proves: kernel_bindings_generated.h emits the sem:: namespace correctly, both
// kernels' views agree on the sem ID, Metal 2.0 allocates the sem (on Gen1).

TEST_F(ProgramSpecHWTest, SemaphoreAccessorNameLoopback) {
    auto mesh_device = devices_.at(0);

    const NodeCoord node{0, 0};

    // A SemaphoreSpec describes a Program-scope semaphore: it identifies the sem by name and
    // declares which nodes will see it. Initial value defaults to 0.
    SemaphoreSpec sem{
        .unique_id = SemaphoreSpecName{"only_sem"},
        .target_nodes = node,
    };

    // A KernelSpec binds the semaphore by its `unique_id` and gives it a kernel-local
    // `accessor_name` — the name the kernel source uses to refer to it. The runtime emits
    // `sem::<accessor_name>` constants in `kernel_bindings_generated.h` for the kernel to
    // consume. The producer and consumer below choose different accessor names for the same
    // semaphore.
    // These two DM kernels only need to land on distinct DM processors. The READER/WRITER role
    // hints are the idiomatic way to get that — the producer writes the semaphore signal, the
    // consumer reads it — with no need to hand-pick a processor/NOC via an explicit DataMovement1XXConfig.
    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_accessor_loopback_producer.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"only_sem"}, .accessor_name = "signal"}},
        .hw_config = CreateWriterDataMovementConfig(),
    };
    KernelSpec consumer{
        .unique_id = KernelSpecName{"consumer"},
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_accessor_loopback_consumer.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"only_sem"}, .accessor_name = "waiter"}},
        .hw_config = CreateReaderDataMovementConfig(),
    };

    // A WorkUnitSpec describes the kernels that run on a shared set of nodes.
    WorkUnitSpec work_unit{
        .name = "work_unit_0",
        .kernels = {KernelSpecName{"producer"}, KernelSpecName{"consumer"}},
        .target_nodes = node,
    };

    // The ProgramSpec aggregates everything and is consumed by `MakeProgramFromSpec`.
    ProgramSpec spec{
        .name = "semaphore_accessor_loopback",
        .kernels = {producer, consumer},
        .semaphores = {sem},
        .work_units = std::vector<WorkUnitSpec>{work_unit},
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);
    LaunchProgram(*mesh_device, std::move(program));
    // If we got here, both kernels resolved their sem accessors to the same ID.
}

// ============================================================================
// TensorAccessor Binding End-to-End Loopback Test
// ============================================================================
//
// Proves that the Metal 2.0 TensorAccessor binding feature works end-to-end on real WH/BH:
//   1. Spec → MakeProgramFromSpec resolves the binding's TensorSpec into a correct CTA payload
//      (page size, args_config, bank coords, alignment).
//   2. Each binding's slot in the kernel's TensorBinding address section is filled with
//      MeshTensor::address() at enqueue.
//   3. kernel_bindings_generated.h emits a `tensor::` namespace with a working type alias + token.
//   4. TensorAccessor(tensor::name) constructs an accessor whose get_noc_addr returns
//      addresses that NoC reads/writes actually use correctly.
//
// Pipeline:
//   Host writes known data → input MeshTensor (DRAM)
//   Producer DM kernel (BRISC):  input MeshTensor → DFB,  via TensorAccessor(tensor::input_tensor)
//   Consumer DM kernel (NCRISC): DFB → output MeshTensor, via TensorAccessor(tensor::output_tensor)
//   Host reads output MeshTensor and verifies match
//
// DM-only on purpose: TensorAccessor is the NOC-capable accessor and only compiles on DM builds.
// The compute (TRISC) path uses LocalTensorAccessor instead — proven by
// LocalTensorAccessorBindingCompileComputeKernel below.

TEST_F(ProgramSpecHWTest, TensorAccessorBindingLoopback) {
    auto mesh_device = devices_.at(0);

    // Tensor: 8 pages × 1024 bytes (BFLOAT16, ROW_MAJOR, shape {8, 512} → page = row = 1024 B)
    constexpr uint32_t num_pages = 8;
    constexpr uint32_t page_size = 1024;
    constexpr uint32_t num_dfb_entries = 4;

    const NodeCoord node{0, 0};

    // -------------------------------------------------------
    // Allocate input + output MeshTensors (DRAM, interleaved)
    // -------------------------------------------------------
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto tensor_spec = TensorSpec(Shape{num_pages, 512}, tensor_layout);

    MeshTensor input_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec);
    MeshTensor output_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec);

    // -------------------------------------------------------
    // Build ProgramSpec: 2 DM kernels + 1 DFB + 2 TensorParameters
    // -------------------------------------------------------
    ProgramSpec spec;
    spec.name = "ta_binding_loopback";

    // Producer (BRISC): reads input tensor via TA binding, pushes to DFB
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 1;

    // Consumer (NCRISC): pops from DFB, writes output tensor via TA binding
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 1;

    // DFB connecting the two kernels
    auto dfb = MakeMinimalDFB("input_dfb", page_size, num_dfb_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"input_dfb"}, "input_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"input_dfb"}, "input_dfb"));

    // TensorAccessor bindings: each kernel sees its own tensor under its accessor name
    BindTensorParameterToKernel(producer, "input_tensor", "input_tensor");
    BindTensorParameterToKernel(consumer, "output_tensor", "output_tensor");

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{"input_tensor"}, .spec = tensor_spec},
        TensorParameter{.unique_id = TensorParamName{"output_tensor"}, .spec = tensor_spec},
    };
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    // -------------------------------------------------------
    // Create Program
    // -------------------------------------------------------
    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // -------------------------------------------------------
    // Set runtime args
    // -------------------------------------------------------
    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {num_pages}}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {num_pages}}},
                },
        },
    };
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{input_tensor}},
        {TensorParamName{"output_tensor"}, TensorArgument{output_tensor}},
    };
    SetProgramRunArgs(program, params);

    // -------------------------------------------------------
    // Fill input tensor with known data
    // -------------------------------------------------------
    std::vector<bfloat16> input_data(tensor_spec.logical_shape().volume());
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = bfloat16(static_cast<float>(i));
    }
    auto& cq = mesh_device->mesh_command_queue();
    auto input_host = HostTensor::from_vector(input_data, tensor_spec);
    cq.enqueue_write_tensor(input_host, input_tensor);

    // -------------------------------------------------------
    // Dispatch
    // -------------------------------------------------------
    LaunchProgram(*mesh_device, std::move(program));

    // -------------------------------------------------------
    // Verify
    // -------------------------------------------------------
    auto output_data = cq.enqueue_read_tensor(output_tensor).to_vector<bfloat16>();

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// LocalTensorAccessor Binding — Compute (TRISC) Compile + Token-Wiring Proof
// ============================================================================
//
// Proves the compute-kernel path for tensor bindings, which previously could not compile:
//   1. A compute (TRISC) kernel binds a tensor and constructs LocalTensorAccessor<uint32_t> from the
//      binding token. The generated header emits only the NOC-free token header on the TRISC build, so
//      this compiles (tensor_accessor.h would not — it needs NOC_INDEX, absent on compute builds).
//   2. ValidateProgramSpec accepts the compute-kernel tensor binding (the old guard is gone).
//   3. The binding's base-address CRTA is broadcast to the compute kernel and resolves to the local
//      L1 shard address — verified by comparing the reported address to the bound tensor's address.
//
// Pipeline (compute-only; no DFB / DM consumer):
//   Compute kernel (TRISC) — constructs LocalTensorAccessor (token ctor) and a second via the legacy
//       base-address ctor; PACK writes {base_address, get_unsafe_ptr, &operator[], legacy-ctor base}
//       into a host-known L1 report buffer (named RTA). All four should equal the bound tensor's
//       address. Host reads the report via ReadFromDeviceL1 after LaunchProgram.
//
// The tensor is a single-shard L1 tensor on core (0,0) (the compute kernel's core), so its shard base
// address equals MeshTensor::address(). No dereference of the shard occurs (address-of only), so the
// proof does not depend on the shard's contents.

TEST_F(ProgramSpecHWTest, LocalTensorAccessorBindingCompileComputeKernel) {
    auto mesh_device = devices_.at(0);

    constexpr uint32_t kReportAddr = 100 * 1024;  // host-known fixed L1 addr (same idiom as ScratchpadWriteReadback)
    constexpr uint32_t kNumReportWords = 4;

    const NodeCoord node{0, 0};

    // Single-shard L1 tensor on core (0,0): one 32x32 BFLOAT16 tile.
    auto tensor_param = MakeShardedTensorParameter("local_t", Shape{32, 32}, {32, 32}, /*num_cores=*/1);
    MeshTensor local_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_param.spec);

    ProgramSpec spec;
    spec.name = "local_tensor_accessor_compute";

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/local_tensor_accessor_compute.cpp";
    compute.runtime_arg_schema.runtime_arg_names = {"report_addr"};
    BindTensorParameterToKernel(compute, "local_t", "local_t");

    spec.kernels = {compute};
    spec.tensor_parameters = {tensor_param};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"compute"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"compute"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"report_addr", kReportAddr}}),
    }};
    params.tensor_args = {
        {TensorParamName{"local_t"}, TensorArgument{local_tensor}},
    };
    SetProgramRunArgs(program, params);

    std::vector<uint32_t> zero_report(kNumReportWords, 0u);
    slow_dispatch::WriteToL1(*mesh_device, node, kReportAddr, zero_report);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<uint32_t> reported;
    slow_dispatch::ReadFromL1(*mesh_device, node, kReportAddr, kNumReportWords * sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), kNumReportWords);

    const uint32_t expected_address = static_cast<uint32_t>(local_tensor.address());
    EXPECT_EQ(reported[0], expected_address) << "get_bank_base_address mismatch";
    EXPECT_EQ(reported[1], expected_address) << "get_unsafe_ptr mismatch";
    EXPECT_EQ(reported[2], expected_address) << "&operator[] mismatch";
    EXPECT_EQ(reported[3], expected_address) << "legacy base-address ctor mismatch";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
