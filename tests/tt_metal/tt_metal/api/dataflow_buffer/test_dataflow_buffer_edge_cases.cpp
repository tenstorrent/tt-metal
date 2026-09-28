// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Edge-case coverage: counter-wrap, ring-pressure, decoy, long-run (Metal 2.0).

#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {


// A1: DM->Tensix->DM decoy pipeline
enum class A1Transform { Identity, Relu };

static void run_a1_pipeline(distributed::MeshDevice& mesh_device, A1Transform transform) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }

    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    constexpr uint32_t num_entries = 4;
    const m2::NodeCoord node{0, 0};

    // Tensors
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    const m2::DFBSpecName DFB_IN{"dfb_in"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    // DFBs — disable_implicit_sync=true matches kernels' explicit credit-flow path.
    m2::DataflowBufferSpec dfb_in{
        .unique_id = DFB_IN,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_out{
        .unique_id = DFB_OUT,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Producer kernel: writes input → DFB_IN
    auto producer = make_dm_dfb_producer(PRODUCER, DFB_IN, IN_TENSOR, num_entries, /*implicit_sync=*/false);

    // Compute kernel: dfb_in → (relu or identity) → dfb_out
    const std::string compute_source = (transform == A1Transform::Relu)
                                           ? "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_relu_2_0.cpp"
                                           : "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy_2_0.cpp";
    auto compute = make_compute_kernel(COMPUTE, compute_source);
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"per_core_tile_cnt", num_entries}};

    // Consumer kernel: DFB_OUT → output tensor
    auto consumer = make_dm_dfb_consumer(
        CONSUMER, DFB_OUT, OUT_TENSOR, num_entries, /*blocked_consumer=*/false, /*implicit_sync=*/false);

    // All-pass set dfb_in/dfb_out .disable_implicit_sync = true; #45160 moved that onto the
    // Gen2 DM config, so disable per DM endpoint (the compute stage is Tensix → no DM side).
    disable_implicit_sync_for(producer, DFB_IN);
    disable_implicit_sync_for(consumer, DFB_OUT);

    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER, CONSUMER, COMPUTE},
        .target_nodes = node,
    };

    m2::ProgramSpec spec{
        .name = "a1_2_0",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in, dfb_out},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    // Stimulus
    const uint32_t total_bytes = entry_size * num_entries;
    auto input = (transform == A1Transform::Relu)
                     ? create_random_vector_of_bfloat16(total_bytes, 1.0f, 0xA1A1)   // positive only
                     : create_random_vector_of_bfloat16(total_bytes, 2.0f, 0xA1A1);  // [-1,1]
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(mesh_device, in_tensor, input);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);

    if (transform == A1Transform::Relu) {
        // Positive bf16 inputs → relu identity, allow bf16 tolerance.
        EXPECT_TRUE(
            packed_uint32_t_vector_comparison(output, input, [](float a, float b) { return std::abs(a - b) < 0.01f; }));
    } else {
        EXPECT_EQ(input, output);
    }
}

TEST_F(UnitMeshFixture, A1_2_0_DMTensixDMTest2xDFB1Sx1S) { run_a1_pipeline(this->device(), A1Transform::Identity); }

TEST_F(UnitMeshFixture, A1_2_0_DMTensixDMTest2xDFB1Sx1S_Relu) { run_a1_pipeline(this->device(), A1Transform::Relu); }

// B: implicit-sync edge-case regressions
static void run_dm_dfb_dm_implicit_sync_2_0(
    distributed::MeshDevice& mesh_device,
    uint32_t num_iterations,
    bool implicit_sync,
    uint32_t entry_size = 1024,
    uint32_t num_entries = 16,
    uint32_t total_tiles = 16) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only";
    }

    const m2::NodeCoord node{0, 0};

    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, total_tiles, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    // DFB-level implicit_sync must match the kernels' CTA (inverted polarity).
    m2::DataflowBufferSpec dfb{
        .unique_id = DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer = make_dm_dfb_producer(PRODUCER, DFB, IN_TENSOR, total_tiles, implicit_sync);

    auto consumer =
        make_dm_dfb_consumer(CONSUMER, DFB, OUT_TENSOR, total_tiles, /*blocked_consumer=*/false, implicit_sync);

    // All-pass: dfb.disable_implicit_sync = !implicit_sync (now per-DM-endpoint, post-#45160).
    maybe_disable_implicit_sync(producer, implicit_sync, DFB);
    maybe_disable_implicit_sync(consumer, implicit_sync, DFB);

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "dm_dfb_dm",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", total_tiles}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", total_tiles}}),
        },
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_words = entry_size * total_tiles / sizeof(uint32_t);
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 1000000, total_words);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(mesh_device, in_tensor, input);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
    }

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);
    EXPECT_EQ(input, output) << "M2 DM→DFB→DM identity mismatch";
}

TEST_F(UnitMeshFixture, B1_2_0_DM0NoKernel_TensixDMImplicitSync) {
    // B1 = single-iteration implicit-sync DM→DFB→DM. DM0-no-kernel coverage is
    // an internal-state regression that fires whether or not we explicitly
    // skip DM0; the helper produces the same wire-level traffic.
    run_dm_dfb_dm_implicit_sync_2_0(this->device(), /*num_iterations=*/1, /*implicit_sync=*/true);
}

TEST_F(UnitMeshFixture, B1b_2_0_DM0IdleSubordinateRuns_TensixDMImplicitSync) {
    // B1b same shape as B1 with an extra iter to expose stale-credit edge.
    run_dm_dfb_dm_implicit_sync_2_0(this->device(), /*num_iterations=*/2, /*implicit_sync=*/true);
}

TEST_F(UnitMeshFixture, B3_2_0_TailCreditRace_RepeatedImplicitSync_DMDM) {
    // B3: 3 repeated implicit-sync iterations exercise the tail-credit race.
    run_dm_dfb_dm_implicit_sync_2_0(this->device(), /*num_iterations=*/3, /*implicit_sync=*/true);
}

// Implicit-sync availability guards.
//
// Both guards drive one endpoint through the real implicit-sync path
// (async_read/async_write with TXN_ID) while the opposite endpoint acts as a
// credit controller using the explicit push_back / pop_front APIs. The
// controller withholds the credit the second implicit transfer needs and
// records, into L1, whether that transfer returned anyway. A pending transfer
// is invisible to the tile-counter occupancy / free-space registers, so a
// guard that consults only those registers lets the second transfer reuse a
// slot that is not actually available.
namespace {

constexpr uint32_t kGuardEntrySize = 1024;
constexpr uint32_t kGuardRingEntries = 16;
constexpr uint32_t kGuardSentinel = 0xBAADF00Du;

const m2::DFBSpecName GUARD_DFB{"dfb"};
const m2::KernelSpecName GUARD_PRODUCER{"producer"};
const m2::KernelSpecName GUARD_CONSUMER{"consumer"};
const m2::TensorParamName GUARD_TENSOR{"guard_tensor"};
const m2::SemaphoreSpecName GUARD_SEM_PRODUCER_READY{"producer_ready"};
const m2::SemaphoreSpecName GUARD_SEM_CONSUMER_READY{"consumer_ready"};
const m2::SemaphoreSpecName GUARD_SEM_SECOND_ATTEMPT{"second_attempt"};
const m2::SemaphoreSpecName GUARD_SEM_SECOND_RETURNED{"second_returned"};
const m2::SemaphoreSpecName GUARD_SEM_CREDIT_RELEASED{"credit_released"};
const m2::SemaphoreSpecName GUARD_SEM_PEER_PRELOADED{"peer_preloaded"};

std::vector<m2::SemaphoreBinding> guard_semaphore_bindings() {
    return {
        {.semaphore_spec_name = GUARD_SEM_PRODUCER_READY, .accessor_name = "producer_ready"},
        {.semaphore_spec_name = GUARD_SEM_CONSUMER_READY, .accessor_name = "consumer_ready"},
        {.semaphore_spec_name = GUARD_SEM_SECOND_ATTEMPT, .accessor_name = "second_attempt"},
        {.semaphore_spec_name = GUARD_SEM_SECOND_RETURNED, .accessor_name = "second_returned"},
        {.semaphore_spec_name = GUARD_SEM_CREDIT_RELEASED, .accessor_name = "credit_released"},
    };
}

std::vector<m2::SemaphoreSpec> guard_semaphore_specs(const m2::NodeCoord& node) {
    return {
        {.unique_id = GUARD_SEM_PRODUCER_READY, .target_nodes = node},
        {.unique_id = GUARD_SEM_CONSUMER_READY, .target_nodes = node},
        {.unique_id = GUARD_SEM_SECOND_ATTEMPT, .target_nodes = node},
        {.unique_id = GUARD_SEM_SECOND_RETURNED, .target_nodes = node},
        {.unique_id = GUARD_SEM_CREDIT_RELEASED, .target_nodes = node},
    };
}

void check_guard_result(distributed::MeshDevice& mesh_device, uint32_t result_l1_addr, const char* what) {
    std::vector<uint32_t> result;
    slow_dispatch::ReadFromL1(mesh_device, CoreCoord(0, 0), result_l1_addr, sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 1u);
    ASSERT_NE(result[0], kGuardSentinel) << "credit controller never reported a result";
    EXPECT_EQ(result[0], 0u) << "the second implicit " << what << " returned before its credit was released";
}

}  // namespace

// Implicit writes: the DM consumer drains one entry per tile counter, then
// wraps back to the first counter whose entry is claimed but not yet acked.
static void run_implicit_write_availability_guard(distributed::MeshDevice& mesh_device, uint32_t num_tile_counters) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only";
    }

    const m2::NodeCoord node{0, 0};
    const uint32_t num_pages = num_tile_counters + 1;
    const auto tensor_spec = make_flat_dram_tensor_spec(kGuardEntrySize, num_pages, DataType::UINT32);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    m2::DataflowBufferSpec dfb{
        .unique_id = GUARD_DFB,
        .entry_size = kGuardEntrySize,
        .num_entries = kGuardRingEntries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // One producer thread per tile counter; each posts to the counter it owns.
    auto producer = make_dm_kernel(
        GUARD_PRODUCER,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_implicit_write_guard_producer.cpp",
        static_cast<uint8_t>(num_tile_counters));
    producer.dfb_bindings = {
        {.dfb_spec_name = GUARD_DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}};
    producer.semaphore_bindings = guard_semaphore_bindings();
    disable_implicit_sync_for(producer, GUARD_DFB);

    auto consumer = make_dm_kernel(
        GUARD_CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_implicit_write_guard_consumer.cpp");
    consumer.dfb_bindings = {
        {.dfb_spec_name = GUARD_DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = GUARD_TENSOR, .accessor_name = "dst_tensor"}};
    consumer.compile_time_args = {{"num_tile_counters", num_tile_counters}};
    consumer.semaphore_bindings = guard_semaphore_bindings();

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {GUARD_PRODUCER, GUARD_CONSUMER}, .target_nodes = node};
    m2::ProgramSpec spec{
        .name = "implicit_write_availability_guard",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .semaphores = guard_semaphore_specs(node),
        .tensor_parameters = {{.unique_id = GUARD_TENSOR, .spec = out_tensor.tensor_spec()}},
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    const uint32_t result_l1_addr = top_of_l1_scratch_addr(mesh_device, sizeof(uint32_t));
    std::vector<uint32_t> sentinel{kGuardSentinel};
    slow_dispatch::WriteToL1(mesh_device, CoreCoord(0, 0), result_l1_addr, sentinel);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = GUARD_PRODUCER,
         .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_l1_addr}})},
        {.kernel = GUARD_CONSUMER},
    };
    params.tensor_args = {{GUARD_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    check_guard_result(mesh_device, result_l1_addr, "write");
}

// Implicit reads: one producer round-robins num_tile_counters. POSTED/ACKED are
// preloaded so each counter has exactly one free slot. The producer reserves
// every one of those slots, then the next read revisits the first counter and
// must wait for that counter's consumer to free it.
static void run_implicit_read_availability_guard(distributed::MeshDevice& mesh_device, uint32_t num_tile_counters) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only";
    }

    const m2::NodeCoord node{0, 0};
    // STRIDED splits the ring across the consumers, so each tile counter holds
    // ring / num_tile_counters entries. Preloading POSTED to that capacity with
    // ACKED at 1 leaves exactly one free slot on every counter: the producer can
    // reserve one read per counter, and the read that wraps back to the first
    // counter must then wait for that counter's consumer to free a slot.
    const uint32_t ring_entries = kGuardRingEntries;
    const uint32_t preload_posted = ring_entries / num_tile_counters;
    constexpr uint32_t preload_acked = 1;

    const auto tensor_spec = make_flat_dram_tensor_spec(kGuardEntrySize, num_tile_counters + 1, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    m2::DataflowBufferSpec dfb{
        .unique_id = GUARD_DFB,
        .entry_size = kGuardEntrySize,
        .num_entries = ring_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer = make_dm_kernel(
        GUARD_PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_implicit_read_guard_producer.cpp");
    producer.dfb_bindings = {
        {.dfb_spec_name = GUARD_DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.tensor_bindings = {{.tensor_parameter_name = GUARD_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {{"preload_posted", preload_posted}, {"num_tile_counters", num_tile_counters}};
    producer.semaphore_bindings = guard_semaphore_bindings();

    auto consumer = make_dm_kernel(
        GUARD_CONSUMER,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_implicit_read_guard_consumer.cpp",
        static_cast<uint8_t>(num_tile_counters));
    consumer.dfb_bindings = {
        {.dfb_spec_name = GUARD_DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.compile_time_args = {{"preload_acked", preload_acked}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}};
    consumer.semaphore_bindings = guard_semaphore_bindings();
    consumer.semaphore_bindings.push_back(
        {.semaphore_spec_name = GUARD_SEM_PEER_PRELOADED, .accessor_name = "peer_preloaded"});
    disable_implicit_sync_for(consumer, GUARD_DFB);

    auto semaphores = guard_semaphore_specs(node);
    semaphores.push_back({.unique_id = GUARD_SEM_PEER_PRELOADED, .target_nodes = node});
    m2::WorkUnitSpec wu{.name = "wu", .kernels = {GUARD_PRODUCER, GUARD_CONSUMER}, .target_nodes = node};
    m2::ProgramSpec spec{
        .name = "implicit_read_availability_guard",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .semaphores = std::move(semaphores),
        .tensor_parameters = {{.unique_id = GUARD_TENSOR, .spec = in_tensor.tensor_spec()}},
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    const uint32_t result_l1_addr = top_of_l1_scratch_addr(mesh_device, sizeof(uint32_t));
    std::vector<uint32_t> sentinel{kGuardSentinel};
    slow_dispatch::WriteToL1(mesh_device, CoreCoord(0, 0), result_l1_addr, sentinel);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = GUARD_PRODUCER},
        {.kernel = GUARD_CONSUMER,
         .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_l1_addr}})},
    };
    params.tensor_args = {{GUARD_TENSOR, std::cref(in_tensor)}};
    m2::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    check_guard_result(mesh_device, result_l1_addr, "read");
}

TEST_F(UnitMeshFixture, ImplicitReadPendingPostReservesFreeSlot) {
    run_implicit_read_availability_guard(this->device(), /*num_tile_counters=*/1);
}

TEST_F(UnitMeshFixture, ImplicitReadAvailability_1Producer2Consumer_2TC) {
    run_implicit_read_availability_guard(this->device(), /*num_tile_counters=*/2);
}

TEST_F(UnitMeshFixture, ImplicitWritePendingAckIsNotNewData) {
    run_implicit_write_availability_guard(this->device(), /*num_tile_counters=*/1);
}

TEST_F(UnitMeshFixture, ImplicitWriteAvailability_2Producer1Consumer_2TC) {
    run_implicit_write_availability_guard(this->device(), /*num_tile_counters=*/2);
}

// D1: long implicit-sync run past counter wrap
TEST_F(UnitMeshFixture, D1_2_0_LongImplicitSync_PostCounterWrap) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only";
    }

    constexpr uint32_t kPreloadValue = 65528;
    constexpr uint32_t kPushTiles = 32;
    constexpr uint32_t kEntrySize = 1024;
    constexpr uint32_t kRingEntries = 16;
    const m2::NodeCoord node{0, 0};

    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};
    // Two single-writer semaphores form a producer<->consumer rendezvous so that
    // neither DM kernel enters its data loop until BOTH the producer's posted and
    // the consumer's acked counters have been preloaded (occupancy provably 0).
    const m2::SemaphoreSpecName SEM_PROD_READY{"sem_prod_ready"};
    const m2::SemaphoreSpecName SEM_CONS_READY{"sem_cons_ready"};

    const auto tensor_spec = make_flat_dram_tensor_spec(kEntrySize, kPushTiles, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);

    m2::DataflowBufferSpec dfb{
        .unique_id = DFB,
        .entry_size = kEntrySize,
        .num_entries = kRingEntries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer =
        make_dm_kernel(PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_with_tc_preload_2_0.cpp");
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {
        {"num_entries_per_producer", kPushTiles}, {"implicit_sync", 1u}, {"kPreloadPostedValue", kPreloadValue}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    producer.semaphore_bindings = {
        {.semaphore_spec_name = SEM_PROD_READY, .accessor_name = "prod_ready"},
        {.semaphore_spec_name = SEM_CONS_READY, .accessor_name = "cons_ready"}};

    auto consumer =
        make_dm_kernel(CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_with_tc_preload_2_0.cpp");
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    consumer.compile_time_args = {
        {"num_entries_per_consumer", kPushTiles},
        {"blocked_consumer", 0u},
        {"implicit_sync", 1u},
        {"kPreloadAckedValue", kPreloadValue}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    consumer.semaphore_bindings = {
        {.semaphore_spec_name = SEM_PROD_READY, .accessor_name = "prod_ready"},
        {.semaphore_spec_name = SEM_CONS_READY, .accessor_name = "cons_ready"}};

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "d1_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .semaphores =
            {
                m2::SemaphoreSpec{.unique_id = SEM_PROD_READY, .target_nodes = node},
                m2::SemaphoreSpec{.unique_id = SEM_CONS_READY, .target_nodes = node},
            },
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", kPushTiles}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", kPushTiles}}),
        },
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    auto input = create_random_vector_of_bfloat16(kPushTiles * kEntrySize, 1.0f, 0xD1D1);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(this->device(), in_tensor, input);

    LaunchProgram(this->device(), std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);

    // Diagnostic: on mismatch, dump first divergent tile + per-tile histogram so
    // we can characterize the failure mode (all-zeros from start, wrap-point only, etc.).
    if (input != output) {
        constexpr size_t kU32PerTile = kEntrySize / sizeof(uint32_t);
        auto mm = std::mismatch(input.begin(), input.end(), output.begin());
        size_t first_diff = mm.first - input.begin();
        size_t mismatch_count = std::transform_reduce(
            input.begin(), input.end(), output.begin(), size_t{0}, std::plus<>{}, std::not_equal_to<>{});
        if (first_diff < input.size()) {
            std::vector<size_t> per_tile_mismatches(kPushTiles, 0);
            for (size_t i = 0; i < input.size(); ++i) {
                if (input[i] != output[i]) {
                    per_tile_mismatches[i / kU32PerTile]++;
                }
            }
            log_info(
                tt::LogTest,
                "D1_2_0 first mismatch at idx {} (tile {}, word {}): input=0x{:x} output=0x{:x}. Total {}/{}.",
                first_diff,
                first_diff / kU32PerTile,
                first_diff % kU32PerTile,
                input[first_diff],
                output[first_diff],
                mismatch_count,
                input.size());
            // Wrap is at tile 8 (posted=65528+8 = 65536 wraps to 0). Print per-tile state.
            for (size_t t = 0; t < kPushTiles; ++t) {
                if (per_tile_mismatches[t] > 0) {
                    log_info(
                        tt::LogTest,
                        "D1_2_0 tile {}: {}/{} words mismatched (input[0]=0x{:x} output[0]=0x{:x}){}",
                        t,
                        per_tile_mismatches[t],
                        kU32PerTile,
                        input[t * kU32PerTile],
                        output[t * kU32PerTile],
                        t == 8 ? "  <-- wrap point" : "");
                }
            }
        }
    }
    EXPECT_EQ(input, output) << "M2 D1: identity copy across uint16 TC-counter wrap point failed";
}

// D2: all-DMs-concurrent ring saturation
static void run_d2_all_dms_concurrent_2_0(distributed::MeshDevice& mesh_device, bool implicit_sync) {
    run_dm_dfb_dm_implicit_sync_2_0(
        mesh_device,
        /*num_iterations=*/1,
        implicit_sync,
        /*entry_size=*/1024,
        /*num_entries=*/24,
        /*total_tiles=*/96);
}

TEST_F(UnitMeshFixture, D2_2_0_AllDMsConcurrent_6Sx2S_ImplicitOff) {
    run_d2_all_dms_concurrent_2_0(this->device(), /*implicit_sync=*/false);
}

TEST_F(UnitMeshFixture, D2_2_0_AllDMsConcurrent_6Sx2S_ImplicitOn) {
    run_d2_all_dms_concurrent_2_0(this->device(), /*implicit_sync=*/true);
}

// D3: multi-core two-groups-via-decoy
TEST_F(UnitMeshFixture, D3_2_0_MultiCoreDFB_TwoGroupsViaDecoy) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "TC-based grouping is Quasar-only";
    }

    CoreCoord grid = this->device().compute_with_storage_grid_size();
    const uint32_t num_workers = grid.x * grid.y;
    if (num_workers < 2) {
        GTEST_SKIP() << "Need >= 2 Tensix cores; device has " << num_workers
                     << " (single-Tensix emulator?). Run on silicon or a multi-Tensix sim.";
    }

    constexpr uint32_t entries_per_core = 16;
    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 16;
    const m2::NodeCoord core_a{0, 0};
    const m2::NodeCoord core_b{1, 0};

    const m2::DFBSpecName DECOY_DFB{"decoy_dfb"};
    const m2::DFBSpecName SHARED_DFB{"shared_dfb"};
    const m2::KernelSpecName DECOY_PRODUCER{"decoy_producer"};
    const m2::KernelSpecName DECOY_CONSUMER{"decoy_consumer"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, 2 * entries_per_core, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);

    // Decoy DFB: lives on core A only.
    m2::DataflowBufferSpec decoy_dfb{
        .unique_id = DECOY_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    // Shared DFB: lives on cores A and B.
    m2::DataflowBufferSpec shared_dfb{
        .unique_id = SHARED_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Decoy producer/consumer on core A only — no-ops, just claim TC slots.
    auto decoy_producer =
        make_dm_kernel(DECOY_PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp");
    decoy_producer.dfb_bindings = {
        {.dfb_spec_name = DECOY_DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    decoy_producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    decoy_producer.compile_time_args = {{"num_entries_per_producer", 0u}, {"implicit_sync", 0u}};
    decoy_producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    auto decoy_consumer =
        make_dm_kernel(DECOY_CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp");
    decoy_consumer.dfb_bindings = {
        {.dfb_spec_name = DECOY_DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    decoy_consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    decoy_consumer.compile_time_args = {
        {"num_entries_per_consumer", 0u}, {"blocked_consumer", 0u}, {"implicit_sync", 0u}};
    decoy_consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // Real shared producer/consumer across A and B (uses dfb::shared kernel variant).
    auto producer =
        make_dm_kernel(PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_with_id_2_0.cpp");
    producer.dfb_bindings = {
        {.dfb_spec_name = SHARED_DFB,
         .accessor_name = "shared",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {{"num_entries_per_producer", entries_per_core}, {"implicit_sync", 0u}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    auto consumer = make_dm_kernel(CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp");
    // NOTE: dfb_consumer.cpp uses dfb::in — rebinding it to the SHARED_DFB by
    // accessor_name "in" is valid; the kernel doesn't care about the host
    // DFB's spec name.
    consumer.dfb_bindings = {
        {.dfb_spec_name = SHARED_DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    consumer.compile_time_args = {
        {"num_entries_per_consumer", entries_per_core}, {"blocked_consumer", 0u}, {"implicit_sync", 0u}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // All-pass disabled decoy_dfb/shared_dfb implicit sync (now per-DM-endpoint, post-#45160).
    disable_implicit_sync_for(decoy_producer, DECOY_DFB);
    disable_implicit_sync_for(decoy_consumer, DECOY_DFB);
    disable_implicit_sync_for(producer, SHARED_DFB);
    disable_implicit_sync_for(consumer, SHARED_DFB);

    // WUs: decoy on core A only; shared on both. WUs cannot overlap target_nodes,
    // so we put decoy on a single-core WU and shared on a disjoint range.
    m2::WorkUnitSpec decoy_wu{
        .name = "decoy_wu",
        .kernels = {DECOY_PRODUCER, DECOY_CONSUMER},
        .target_nodes = core_a,
    };
    m2::WorkUnitSpec shared_wu{
        .name = "shared_wu",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = core_b,  // start with core_b only, since decoy claims core_a
    };

    m2::ProgramSpec spec{
        .name = "d3_2_0",
        .kernels = {decoy_producer, decoy_consumer, producer, consumer},
        .dataflow_buffers = {decoy_dfb, shared_dfb},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {decoy_wu, shared_wu},
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = DECOY_PRODUCER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(core_a, {{"chunk_offset", 0u}, {"entries_per_core", 0u}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = DECOY_CONSUMER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(core_a, {{"chunk_offset", 0u}, {"entries_per_core", 0u}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
                core_b, {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
                core_b, {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}),
        },
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    auto input = create_constant_vector_of_bfloat16(2 * entries_per_core * entry_size, 1.0f);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(this->device(), in_tensor, input);

    LaunchProgram(this->device(), std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);
    // For the m2 version we just verify the shared DFB ran end-to-end on core B.
    // The "two DfbGroups" assertion from the legacy test depends on inspecting
    // program.impl() before LaunchProgram; we keep that for the legacy test and
    // make this m2 variant a simpler end-to-end correctness check.
    //
    // The shared pipeline runs only on core_b (the decoy claims core_a and is a no-op), so it produces
    // only the first entries_per_core of the 2*entries_per_core tensor; the out_tensor's second half is
    // never written. Compare only the produced first half.
    ASSERT_EQ(input.size(), output.size());
    const size_t produced = output.size() / 2;
    input.resize(produced);
    output.resize(produced);
    EXPECT_EQ(input, output) << "M2 D3: shared DFB pipeline mismatch (core_b produced half)";
}

// ring-pressure scenarios (tight rings, heavy wraparound)
TEST_P(DFBImplicitSyncParamFixture_2_0, DMTest1xDFB_RingPressure_1Sx1S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .implicit_sync = GetParam(),
        .num_entries = 16,
        .num_entries_in_buffer = 32,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

TEST_P(DFBImplicitSyncParamFixture_2_0, DMTest1xDFB_RingPressure_3Sx3S_2_0) {
    // M2 caps user DM cores per WU at 6 (legacy 4Sx4S=8 doesn't fit on Gen2).
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 3,
        .num_consumers = 3,
        .implicit_sync = GetParam(),
        .num_entries = default_num_entries(3, 3),
        .num_entries_in_buffer = 27,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

TEST_P(DFBImplicitSyncParamFixture_2_0, TensixDMTest1xDFB_RingPressure_2Sx4S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::TENSIX,
        .consumer_type = M2PorCType::DM,
        .num_producers = 2,
        .num_consumers = 4,
        .implicit_sync = GetParam(),
        .num_entries = 16,
        .num_entries_in_buffer = 32,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

// 4 DM producers + 4 Tensix consumers ALL, num_entries=4 → capacity=1: maximum
// ring pressure on the remapper fan-out path (1 DM post → 4 UNPACK TC acks).
TEST_P(DFBImplicitSyncParamFixture_2_0, DMTensixTest1xDFB_RingPressure_4Sx4A_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::TENSIX,
        .num_producers = 4,
        .num_consumers = 4,
        .pap = m2::DFBAccessPattern::STRIDED,
        .cap = m2::DFBAccessPattern::ALL,
        .implicit_sync = GetParam(),
        .num_entries = 4,
        .num_entries_in_buffer = 64,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

// ---------------------------------------------------------------------------------------------
// Tensix→DM ring pressure with a mapping-independent oracle.
//
// The RingPressure_2Sx4S case above is the only test in the tree that combines ring wraparound,
// data verification, and a DM side wider than the Tensix side -- and its expected values were
// re-derived by mapping observed output tiles back to input pages, so it cannot distinguish a
// slot-mapping defect from intended behaviour. These three use M2Oracle::MULTISET, which asserts
// only what the DFB contract guarantees: every ring slot is delivered exactly once per ring-fill,
// in any order.
//
// The 2Sx2S case is the control. It must pass, and it is what makes a failure in the other two
// evidence about the DFB rather than about the new oracle.
TEST_F(UnitMeshFixture, TensixDMTest1xDFB_RingPressure_2Sx2S_Multiset_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::TENSIX,
        .consumer_type = M2PorCType::DM,
        .num_producers = 2,
        .num_consumers = 2,
        .implicit_sync = false,  // explicit sync only; see note above
        .num_entries = 16,
        .num_entries_in_buffer = 32,
        .oracle = M2Oracle::MULTISET,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

TEST_F(UnitMeshFixture, TensixDMTest1xDFB_RingPressure_1Sx2S_Multiset_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::TENSIX,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 2,
        .implicit_sync = false,  // explicit sync only; see note above
        .num_entries = 16,
        .num_entries_in_buffer = 32,
        .oracle = M2Oracle::MULTISET,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

TEST_F(UnitMeshFixture, TensixDMTest1xDFB_RingPressure_2Sx4S_Multiset_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::TENSIX,
        .consumer_type = M2PorCType::DM,
        .num_producers = 2,
        .num_consumers = 4,
        .implicit_sync = false,  // explicit sync only; see note above
        .num_entries = 16,
        .num_entries_in_buffer = 32,
        .oracle = M2Oracle::MULTISET,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

// ---------------------------------------------------------------------------------------------
// DM→DM asymmetric under ring pressure.
//
// This is the configuration closest to the op-level corruption that the harness can verify END TO END:
// the DM producers NoC-read from DRAM and write their OWN ring slots (unlike the Tensix-producer cases,
// where the host prefills the ring and the producer only posts credits), and the DM consumers NoC-write
// to DRAM, so the default identity oracle applies. Existing DM→DM ring-pressure coverage is symmetric
// only (1Sx1S, 3Sx3S), and existing asymmetric DM→DM coverage moves exactly one ring-fill -- so
// asymmetric-plus-wrap is untested, and it is where per-thread producer slot assignment would show up.
//
// Shapes mirror the op configs that corrupt: 4Sx1S/2Sx1S are the in0/in1 (R, C) shapes at C=1,
// 1Sx2S/1Sx4S are the out (C, W) shapes at C=1.
TEST_F(UnitMeshFixture, DMTest1xDFB_RingPressure_4Sx1S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 4,
        .num_consumers = 1,
        .implicit_sync = false,  // explicit sync only; see note above
        .num_entries = default_num_entries(4, 1),
        .num_entries_in_buffer = 2 * default_num_entries(4, 1),
    };
    run_single_dfb_program_2_0(this->device(), params);
}

TEST_F(UnitMeshFixture, DMTest1xDFB_RingPressure_2Sx1S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 2,
        .num_consumers = 1,
        .implicit_sync = false,  // explicit sync only; see note above
        .num_entries = default_num_entries(2, 1),
        .num_entries_in_buffer = 2 * default_num_entries(2, 1),
    };
    run_single_dfb_program_2_0(this->device(), params);
}

TEST_F(UnitMeshFixture, DMTest1xDFB_RingPressure_1Sx2S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 2,
        .implicit_sync = false,  // explicit sync only; see note above
        .num_entries = default_num_entries(1, 2),
        .num_entries_in_buffer = 2 * default_num_entries(1, 2),
    };
    run_single_dfb_program_2_0(this->device(), params);
}

TEST_F(UnitMeshFixture, DMTest1xDFB_RingPressure_1Sx4S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 4,
        .implicit_sync = false,  // explicit sync only; see note above
        .num_entries = default_num_entries(1, 4),
        .num_entries_in_buffer = 2 * default_num_entries(1, 4),
    };
    run_single_dfb_program_2_0(this->device(), params);
}

// ---------------------------------------------------------------------------------------------
// DM -> DFB -> Tensix -> DFB -> DM, multi-threaded, under ring pressure.
//
// The single-DFB sweeps all pass at the asymmetric shapes an op-level binary_ng corrupts, so the
// remaining structural difference is that a real op chains TWO DFBs through a Tensix stage that is
// simultaneously a consumer and a producer. Nothing else in the tree covers that with multiple threads:
// A1 above is the right topology but hardcodes one thread per stage and one ring-fill.
//
// (R, C, W) are the producer / compute / consumer thread counts, i.e. exactly the KernelSpec::num_threads
// triple a Quasar op sets. Rings are sized 2 x max(endpoints) so the tile stream wraps many times. The
// oracle is the end-to-end identity, since the copy kernel is an identity.
static void run_a1_threaded_pipeline(
    distributed::MeshDevice& mesh_device, uint32_t r, uint32_t c, uint32_t w, uint32_t total_tiles) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    ASSERT_EQ(total_tiles % r, 0u);
    ASSERT_EQ(total_tiles % c, 0u);
    ASSERT_EQ(total_tiles % w, 0u);

    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    const m2::NodeCoord node{0, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, total_tiles, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    const m2::DFBSpecName DFB_IN{"dfb_in"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    m2::DataflowBufferSpec dfb_in{
        .unique_id = DFB_IN,
        .entry_size = entry_size,
        .num_entries = 2 * std::max(r, c),
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_out{
        .unique_id = DFB_OUT,
        .entry_size = entry_size,
        .num_entries = 2 * std::max(c, w),
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer = make_dm_dfb_producer(
        PRODUCER,
        DFB_IN,
        IN_TENSOR,
        total_tiles / r,
        /*implicit_sync=*/false,
        m2::DFBAccessPattern::STRIDED,
        static_cast<uint8_t>(r));

    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy_2_0.cpp", static_cast<uint8_t>(c));
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    // Per-THREAD count: the kernel loop is not strided, the STRIDED binding is what hands each thread
    // its own sub-stream. Same convention the single-DFB helper uses.
    compute.compile_time_args = {{"per_core_tile_cnt", total_tiles / c}};

    auto consumer = make_dm_dfb_consumer(
        CONSUMER,
        DFB_OUT,
        OUT_TENSOR,
        total_tiles / w,
        /*blocked_consumer=*/false,
        /*implicit_sync=*/false,
        m2::DFBAccessPattern::STRIDED,
        static_cast<uint8_t>(w));

    disable_implicit_sync_for(producer, DFB_IN);
    disable_implicit_sync_for(consumer, DFB_OUT);

    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER, CONSUMER, COMPUTE},
        .target_nodes = node,
    };
    m2::ProgramSpec spec{
        .name = "a1_threaded_2_0",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in, dfb_out},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", total_tiles}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", total_tiles}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    auto input = create_random_vector_of_bfloat16(entry_size * total_tiles, 2.0f, 0xA1A1);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(mesh_device, in_tensor, input);

    // Poison the output so a pass cannot come from a buffer that happened to already hold the answer,
    // and so pages nobody wrote are distinguishable from pages written with wrong data.
    constexpr uint32_t kPoison = 0xDEADBEEFu;
    const std::vector<uint32_t> poison(input.size(), kPoison);
    slow_dispatch::WriteToBuffer(out_tensor.mesh_buffer(), poison);
    m2_writeshard_barrier_uint32(mesh_device, out_tensor, poison);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);
    ASSERT_EQ(input.size(), output.size());

    // Second read of the same buffer: a difference would mean the readback is unstable, which would
    // make every mismatch count below meaningless.
    std::vector<uint32_t> output_reread;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output_reread);
    ASSERT_EQ(output, output_reread) << "output buffer readback is not stable across two reads";

    size_t bad = 0;
    size_t untouched = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] == output[i]) {
            continue;
        }
        ++bad;
        untouched += (output[i] == kPoison);
    }
    EXPECT_EQ(bad, 0u) << "A1 threaded pipeline R=" << r << " C=" << c << " W=" << w << ": " << bad << " of "
                       << input.size() << " words wrong (" << (100.0 * bad / input.size()) << "%), of which "
                       << untouched << " still hold the poison value (never written)";
}

// 48 tiles is divisible by every thread count used below, so no thread is ever handed a short share.
#define A1_THREADED_TEST(r, c, w)                                                    \
    TEST_F(UnitMeshFixture, DMTensixDMTest2xDFB_Threaded_R##r##C##c##W##w##_2_0) {   \
        run_a1_threaded_pipeline(this->device(), (r), (c), (w), /*total_tiles=*/48); \
    }

A1_THREADED_TEST(1, 1, 1)  // control: op-level PASS
A1_THREADED_TEST(2, 2, 1)  // op-level PASS
A1_THREADED_TEST(4, 4, 2)  // op-level PASS, the optimum
A1_THREADED_TEST(2, 1, 1)  // op-level FAIL 46.6%  (R > C)
A1_THREADED_TEST(4, 1, 1)  // op-level FAIL 69.9%  (R > C)
A1_THREADED_TEST(1, 1, 2)  // op-level FAIL 50.0%  (W > C)
A1_THREADED_TEST(1, 1, 4)  // op-level FAIL 74.9%  (W > C)

#undef A1_THREADED_TEST

}  // namespace tt::tt_metal
