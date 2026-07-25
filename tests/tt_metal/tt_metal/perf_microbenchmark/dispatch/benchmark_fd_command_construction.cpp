// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//////////////////////////////////////////////////////////////////////////////////////////
// Fast Dispatch host-side command construction microbenchmark (Metal 2.0 host API)
//
// Unlike test_pgm_dispatch (which measures end-to-end dispatch + device execution), this
// benchmark isolates the *host-side* cost of the fast-dispatch command path, expressed
// through the Metal 2.0 descriptor API (ProgramSpec / ProgramRunArgs):
//
//   * Host program building (#5)  - assembling a ProgramSpec descriptor (kernels/DFBs/sems/
//                                    RTA schema) before it is compiled into a Program.
//   * Enqueue program      (#6)   - EnqueueMeshWorkload command construction.
//   * Enqueue r/w buffer   (#6)   - EnqueueWriteMeshBuffer / EnqueueReadMeshBuffer construction.
//   * Trace                (#6)   - trace capture construction + replay host cost.
//   * Mutable updates      (#7)   - SetProgramRunArgs / UpdateProgramRunArgs (RTAs, common
//                                    RTAs) and DFB size overrides (entry_size / num_entries).
//
// Each family runs a "common" case (small representative program) and a "worst" case (many
// DFBs, max RTAs, common RTAs, semaphores, multiple kernel groups) over the full worker grid
// so regressions in the host command-generation path are caught over time in CI.
//
// Timing note: benchmarks use UseManualTime() and time only the host build/enqueue call.
// The device is drained with Finish() *outside* the timed region so queue back-pressure
// does not contaminate the host-side measurement.
//////////////////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

// Vetted minimal spec builders shared with the Metal 2.0 host-API unit tests.
#include "tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace distributed = tt::tt_metal::distributed;

namespace {

constexpr uint32_t DEFAULT_ITERATIONS = 1000;
constexpr uint32_t TRACE_ITERATIONS = 200;  // trace capture/replay allocate device buffers; keep modest
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 50;

// Small (sub-~20us) host-latency benchmarks show meaningful run-to-run variance (allocator/TLB
// state, background load). To keep the golden non-flaky, those are run with multiple repetitions
// and the compare uses the median aggregate. Iteration count is reduced for repeated benchmarks so
// total runtime stays comparable. The compare script strips the "_median" suffix automatically.
constexpr uint32_t REPETITIONS = 7;
constexpr uint32_t REP_ITERATIONS = 300;

// Describes a program shape as a set of Metal 2.0 descriptor knobs. `common` is a small,
// representative workload; `worst` maxes out the per-program mutable state.
struct ProgramShape {
    std::string name;
    uint32_t n_dfbs = 0;         // producer/consumer dataflow buffers (0 => single DM kernel only)
    uint32_t n_args = 0;         // named per-node runtime args on the primary producer kernel
    uint32_t n_common_args = 0;  // named common (broadcast) runtime args
    uint32_t n_sems = 0;         // program-scope semaphores
    uint32_t n_kgs = 1;          // kernel groups: distinct kernels placed on disjoint node columns
};

const ProgramShape COMMON_SHAPE{
    .name = "common", .n_dfbs = 0, .n_args = 8, .n_common_args = 0, .n_sems = 1, .n_kgs = 1};
const ProgramShape WORST_SHAPE{
    .name = "worst", .n_dfbs = 8, .n_args = 64, .n_common_args = 16, .n_sems = 4, .n_kgs = 8};

constexpr const char* kProducer = "producer";
constexpr const char* kConsumer = "consumer";

std::string arg_name(uint32_t i) { return "arg" + std::to_string(i); }
std::string common_arg_name(uint32_t i) { return "carg" + std::to_string(i); }
std::string dfb_name(uint32_t i) { return "dfb" + std::to_string(i); }

// Worker grid selection mirrors test_pgm_dispatch::get_core_count so worst-case programs span a
// realistic, harvesting-safe full-chip grid.
NodeRange worker_range() {
    std::string arch_name = tt::tt_metal::hal::get_arch_name();
    if (arch_name == "wormhole_b0") {
        return NodeRange({0, 0}, {7, 6});
    }
    if (arch_name == "blackhole") {
        return NodeRange({0, 0}, {10, 8});
    }
    log_fatal(tt::LogTest, "Unexpected ARCH_NAME {}", arch_name);
    exit(1);
}

std::vector<NodeCoord> nodes_in(const NodeRange& range) {
    std::vector<NodeCoord> nodes;
    for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
        for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
            nodes.push_back(NodeCoord{x, y});
        }
    }
    return nodes;
}

// Partition the grid columns into n_kgs kernel-group ranges. Following test_pgm_dispatch, the
// first group is wide (the remaining columns) and each subsequent group is a single column.
std::vector<NodeRange> kernel_group_ranges(const NodeRange& grid, uint32_t n_kgs) {
    n_kgs = std::max<uint32_t>(1, std::min<uint32_t>(n_kgs, grid.end_coord.x - grid.start_coord.x + 1));
    std::vector<NodeRange> ranges;
    const uint32_t g0_end_x = grid.end_coord.x - (n_kgs - 1);
    ranges.emplace_back(NodeCoord{grid.start_coord.x, grid.start_coord.y}, NodeCoord{g0_end_x, grid.end_coord.y});
    for (uint32_t i = 1; i < n_kgs; i++) {
        const uint32_t x = g0_end_x + i;
        ranges.emplace_back(NodeCoord{x, grid.start_coord.y}, NodeCoord{x, grid.end_coord.y});
    }
    return ranges;
}

// The immutable spec plus the derived info the mutable-update paths need.
struct SpecBundle {
    ProgramSpec spec;
    std::vector<NodeCoord> producer_nodes;  // nodes the RTA-carrying producer runs on
    std::vector<std::string> dfb_names;     // DFBs available for size overrides
};

//////////////////////////////////////////////////////////////////////////////////////////
// #5 Host program building: assemble the immutable ProgramSpec descriptor.
//////////////////////////////////////////////////////////////////////////////////////////
SpecBundle build_spec(const ProgramShape& shape) {
    const NodeRange grid = worker_range();
    const auto kg_ranges = kernel_group_ranges(grid, shape.n_kgs);
    const NodeRange& group0 = kg_ranges.front();

    SpecBundle bundle;
    ProgramSpec& spec = bundle.spec;
    spec.name = "fd_cmd_" + shape.name;
    bundle.producer_nodes = nodes_in(group0);

    // Primary producer (group 0): carries the RTA schema; also the DFB producer endpoint.
    auto producer = test_helpers::MakeMinimalGen1DMKernel(kProducer, DataMovementProcessor::RISCV_0);
    for (uint32_t i = 0; i < shape.n_args; i++) {
        producer.runtime_arg_schema.runtime_arg_names.push_back(arg_name(i));
    }
    for (uint32_t i = 0; i < shape.n_common_args; i++) {
        producer.runtime_arg_schema.common_runtime_arg_names.push_back(common_arg_name(i));
    }

    std::vector<KernelSpec> kernels{producer};
    std::vector<WorkUnitSpec> work_units;

    if (shape.n_dfbs > 0) {
        // DFBs are local: producer and consumer endpoints must be co-located on group 0's nodes.
        auto consumer = test_helpers::MakeMinimalGen1ComputeKernel(kConsumer);
        for (uint32_t i = 0; i < shape.n_dfbs; i++) {
            auto dfb = test_helpers::MakeMinimalDFB(dfb_name(i), /*entry_size=*/1024, /*num_entries=*/2);
            dfb.data_format_metadata = tt::DataFormat::Float16_b;
            producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{dfb_name(i)}, "p" + std::to_string(i)));
            consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{dfb_name(i)}, "c" + std::to_string(i)));
            spec.dataflow_buffers.push_back(dfb);
            bundle.dfb_names.push_back(dfb_name(i));
        }
        kernels = {producer, consumer};
        work_units.push_back(test_helpers::MakeMinimalWorkUnit("kg0", group0, {kProducer, kConsumer}));
    } else {
        kernels = {producer};
        work_units.push_back(test_helpers::MakeMinimalWorkUnit("kg0", group0, {kProducer}));
    }

    // Additional kernel groups: one distinct DM kernel per remaining column range. These add
    // structural placement/binary variety (the "different kernel groups" worst-case dimension).
    for (uint32_t i = 1; i < kg_ranges.size(); i++) {
        const std::string kname = "kg_dm" + std::to_string(i);
        kernels.push_back(test_helpers::MakeMinimalGen1DMKernel(kname, DataMovementProcessor::RISCV_0));
        work_units.push_back(test_helpers::MakeMinimalWorkUnit("kg" + std::to_string(i), kg_ranges[i], {kname}));
    }

    for (uint32_t i = 0; i < shape.n_sems; i++) {
        spec.semaphores.push_back(
            SemaphoreSpec{.unique_id = SemaphoreSpecName{"sem" + std::to_string(i)}, .target_nodes = grid});
    }

    spec.kernels = std::move(kernels);
    spec.work_units = std::move(work_units);
    return bundle;
}

// Build the full ProgramRunArgs (mutable state) for the producer kernel: sets every declared
// named RTA on every node it runs on, plus every common RTA. Worst-case host cost path (#7).
ProgramRunArgs make_full_run_args(const ProgramShape& shape, const std::vector<NodeCoord>& nodes, uint32_t seed) {
    ProgramRunArgs params;
    KernelRunArgs kra{.kernel = KernelSpecName{kProducer}};
    for (const auto& node : nodes) {
        for (uint32_t i = 0; i < shape.n_args; i++) {
            kra.runtime_arg_values[arg_name(i)][node] = seed + i;
        }
    }
    for (uint32_t i = 0; i < shape.n_common_args; i++) {
        kra.common_runtime_arg_values[common_arg_name(i)] = seed + i;
    }
    params.kernel_run_args = {kra};
    return params;
}

// Build a DFB size-override update (#7): overrides entry_size / num_entries for every DFB.
ProgramRunArgs make_dfb_overrides(
    const std::vector<std::string>& dfb_names, uint32_t entry_size, uint32_t num_entries) {
    ProgramRunArgs params;
    for (const auto& name : dfb_names) {
        params.dfb_run_overrides.push_back(
            DFBRunOverrides{.dfb = DFBSpecName{name}, .entry_size = entry_size, .num_entries = num_entries});
    }
    return params;
}

std::shared_ptr<distributed::MeshDevice> open_device() {
    constexpr ChipId device_id = 0;
    return distributed::MeshDevice::create_unit_mesh(device_id);
}

// Times only `body()` per iteration; `drain()` (if any) runs after the timer stops so device
// back-pressure is excluded from the host-side measurement.
void run_manual_timed(benchmark::State& state, const std::function<void()>& body, const std::function<void()>& drain) {
    for ([[maybe_unused]] auto _ : state) {
        auto start = std::chrono::steady_clock::now();
        body();
        auto end = std::chrono::steady_clock::now();
        if (drain) {
            drain();
        }
        state.SetIterationTime(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    }
}

// Build a workload with its run args configured, ready to enqueue.
distributed::MeshWorkload make_configured_workload(
    distributed::MeshDevice& device, const ProgramShape& shape, const SpecBundle& bundle) {
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(device, bundle.spec);
    Program& program = workload.get_programs().begin()->second;
    SetProgramRunArgs(program, make_full_run_args(shape, bundle.producer_nodes, /*seed=*/1));
    return workload;
}

//////////////////////////////////////////////////////////////////////////////////////////
// #5 Host program building: time ProgramSpec descriptor assembly (pure host, no compile).
//////////////////////////////////////////////////////////////////////////////////////////
void BM_program_spec_build(benchmark::State& state, ProgramShape shape) {
    // Device is opened only so HAL (arch name / worker grid) is initialized; the spec build
    // itself is pure host work and does not touch the device.
    auto device = open_device();
    run_manual_timed(state, [&]() { benchmark::DoNotOptimize(build_spec(shape)); }, nullptr);
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #6 Enqueue program: time the host command construction of EnqueueMeshWorkload.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_enqueue_program(benchmark::State& state, ProgramShape shape) {
    auto device = open_device();
    auto& cq = device->mesh_command_queue();
    distributed::MeshWorkload workload = make_configured_workload(*device, shape, build_spec(shape));

    for (uint32_t i = 0; i < DEFAULT_WARMUP_ITERATIONS; i++) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    Finish(cq);

    run_manual_timed(state, [&]() { distributed::EnqueueMeshWorkload(cq, workload, false); }, [&]() { Finish(cq); });
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #7 Mutable updates: time UpdateProgramRunArgs (RTA + common RTA refresh) on a built Program.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_run_args_update(benchmark::State& state, ProgramShape shape) {
    auto device = open_device();
    const SpecBundle bundle = build_spec(shape);
    distributed::MeshWorkload workload = make_configured_workload(*device, shape, bundle);
    Program& program = workload.get_programs().begin()->second;

    uint32_t seed = 1;
    run_manual_timed(
        state,
        [&]() {
            ProgramRunArgs params = make_full_run_args(shape, bundle.producer_nodes, ++seed);
            UpdateProgramRunArgs(program, params);
        },
        nullptr);
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #7 Mutable updates: time UpdateProgramRunArgs DFB size overrides (entry_size / num_entries).
//////////////////////////////////////////////////////////////////////////////////////////
void BM_dfb_override(benchmark::State& state, ProgramShape shape) {
    auto device = open_device();
    const SpecBundle bundle = build_spec(shape);
    distributed::MeshWorkload workload = make_configured_workload(*device, shape, bundle);
    Program& program = workload.get_programs().begin()->second;

    bool toggle = false;
    run_manual_timed(
        state,
        [&]() {
            toggle = !toggle;
            // Alternate size each iteration so every update is a real state change.
            ProgramRunArgs params = make_dfb_overrides(bundle.dfb_names, toggle ? 1024 : 2048, toggle ? 2 : 4);
            UpdateProgramRunArgs(program, params);
        },
        nullptr);
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #6 Enqueue write buffer: time host command construction of EnqueueWriteMeshBuffer.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_enqueue_write_buffer(benchmark::State& state, BufferType buffer_type) {
    auto device = open_device();
    auto& cq = device->mesh_command_queue();

    const uint32_t page_size = state.range(0);
    constexpr uint64_t transfer_size = 8 * 1024 * 1024;  // 8 MB
    std::vector<uint32_t> host_buffer(transfer_size / sizeof(uint32_t), 0);

    auto device_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{transfer_size},
        distributed::DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        device.get());

    run_manual_timed(
        state,
        [&]() { distributed::EnqueueWriteMeshBuffer(cq, device_buffer, host_buffer, false); },
        [&]() { Finish(cq); });
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #6 Enqueue read buffer: time a blocking EnqueueReadMeshBuffer (read command construction +
// device readback). enqueue_read_mesh_buffer only supports blocking, and the non-blocking
// shard path builds a single tiny descriptor whose host cost is pure noise; the blocking
// end-to-end read is bandwidth-bound and stable, mirroring benchmark_rw_buffer's read.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_enqueue_read_buffer(benchmark::State& state, BufferType buffer_type) {
    auto device = open_device();
    auto& cq = device->mesh_command_queue();

    const uint32_t page_size = state.range(0);
    constexpr uint64_t transfer_size = 8 * 1024 * 1024;  // 8 MB
    std::vector<uint32_t> host_buffer(transfer_size / sizeof(uint32_t), 0);

    auto device_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{transfer_size},
        distributed::DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
        device.get());
    // Prime the device buffer so the read has valid data to move.
    distributed::EnqueueWriteMeshBuffer(cq, device_buffer, host_buffer, true);

    run_manual_timed(
        state,
        [&]() { distributed::EnqueueReadMeshBuffer(cq, host_buffer, device_buffer, /*blocking=*/true); },
        nullptr);
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #6 Trace: time trace capture command construction (BeginTraceCapture -> enqueue -> end).
//////////////////////////////////////////////////////////////////////////////////////////
void BM_trace_capture(benchmark::State& state, ProgramShape shape) {
    auto device = open_device();
    constexpr std::size_t cq_id = 0;
    auto& cq = device->mesh_command_queue(cq_id);
    distributed::MeshWorkload workload = make_configured_workload(*device, shape, build_spec(shape));

    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    // Shared between the timed body and the (untimed) drain so the captured trace can be
    // released after each iteration without timing the release.
    std::optional<distributed::MeshTraceId> captured;
    run_manual_timed(
        state,
        [&]() {
            distributed::MeshTraceId tid = distributed::BeginTraceCapture(device.get(), cq_id);
            distributed::EnqueueMeshWorkload(cq, workload, false);
            device->end_mesh_trace(cq_id, tid);
            captured = tid;
        },
        [&]() {
            Finish(cq);
            if (captured) {
                device->release_mesh_trace(*captured);
                captured.reset();
            }
        });
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #6 Trace: time replay_mesh_trace host cost for a captured workload.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_trace_replay(benchmark::State& state, ProgramShape shape) {
    auto device = open_device();
    constexpr std::size_t cq_id = 0;
    auto& cq = device->mesh_command_queue(cq_id);
    distributed::MeshWorkload workload = make_configured_workload(*device, shape, build_spec(shape));

    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    distributed::MeshTraceId tid = distributed::BeginTraceCapture(device.get(), cq_id);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    device->end_mesh_trace(cq_id, tid);
    Finish(cq);

    run_manual_timed(state, [&]() { device->replay_mesh_trace(cq_id, tid, false); }, [&]() { Finish(cq); });
    device->close();
}

void PageSizeArgs(benchmark::internal::Benchmark* b) { b->Arg(32)->Arg(256)->Arg(1024)->Arg(2048); }

// Config for small host-latency benchmarks: run several repetitions and report the MIN across
// them (a slow flake from a contended host only ever inflates a sample, so the minimum is the
// most representative measure of true host cost; a genuine regression still raises the minimum).
// The compare script consumes the "min" aggregate. Applied via ->Apply() at registration.
void RepeatedConfig(benchmark::internal::Benchmark* b) {
    b->Iterations(REP_ITERATIONS)
        ->Repetitions(REPETITIONS)
        ->ComputeStatistics("min", [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); })
        ->ReportAggregatesOnly(true)
        ->UseManualTime();
}

//////////////////////////////////////////////////////////////////////////////////////////
// #7 Borrowed-memory DFB updates.
//
// A borrowed-memory DFB has no Program-owned L1 storage; its backing ring lives in a
// user-managed MeshTensor supplied at runtime via a TensorParameter. The mutable host paths
// this exercises are:
//   * rebind  - point the borrowed DFB at a different L1 tensor (UpdateTensorArgs, address only).
//   * resize  - re-attach with a new num_entries override (SetProgramRunArgs, runs the per-bank
//               fit check against the backing tensor). This is the borrowed analog of a
//               program-cache-hit resize.
// Kernel sources are the standard DFB producer/consumer test kernels.
//////////////////////////////////////////////////////////////////////////////////////////
constexpr const char* kDfbProducerKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp";
constexpr const char* kDfbConsumerKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp";

// A single-page INTERLEAVED L1 tensor so aligned_size_per_bank() covers the whole allocation
// (a multi-page L1 tensor would fail the borrowed-DFB per-bank size check).
TensorSpec make_flat_l1_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    const uint32_t total_words = total_entries * entry_size / sizeof(uint32_t);
    auto layout = TensorLayout(
        DataType::UINT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
    return TensorSpec(Shape{1, total_words}, layout);
}

TensorSpec make_flat_dram_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    const uint32_t entry_size_words = entry_size / sizeof(uint32_t);
    auto layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    return TensorSpec(Shape{total_entries, entry_size_words}, layout);
}

// Everything required to keep a borrowed-memory program alive across a benchmark loop. The
// MeshTensors are RAII and must outlive the Program's last use, so they are held here.
struct BorrowedProgram {
    std::shared_ptr<distributed::MeshDevice> device;
    Program program;
    std::optional<MeshTensor> src;
    std::optional<MeshTensor> dst;
    std::optional<MeshTensor> ring_a;
    std::optional<MeshTensor> ring_b;
    NodeCoord node{0, 0};
    uint32_t num_entries = 0;
};

constexpr const char* kBorrowedDfb = "borrowed_dfb";

// Build the (single-node, Gen1) borrowed-memory ProgramSpec used by both borrowed benchmarks.
ProgramSpec build_borrowed_spec(uint32_t entry_size, uint32_t num_entries, const NodeCoord& node) {
    ProgramSpec spec;
    spec.name = "fd_cmd_borrowed";

    auto producer = test_helpers::MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = kDfbProducerKernel;
    producer.compile_time_args = {
        {"num_entries_per_producer", num_entries}, {"implicit_sync", 0u}, {"num_producers", 1u}};
    producer.runtime_arg_schema.runtime_arg_names = {"chunk_offset", "entries_per_core"};
    producer.tensor_bindings = {
        {.tensor_parameter_name = TensorParamName{"src_tensor"}, .accessor_name = "src_tensor"},
        {.tensor_parameter_name = TensorParamName{"dfb_ring_tensor"}, .accessor_name = "dfb_ring"}};
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{kBorrowedDfb}, "out"));

    auto consumer = test_helpers::MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = kDfbConsumerKernel;
    consumer.compile_time_args = {
        {"num_entries_per_consumer", num_entries},
        {"blocked_consumer", 0u},
        {"implicit_sync", 0u},
        {"num_consumers", 1u}};
    consumer.runtime_arg_schema.runtime_arg_names = {"chunk_offset", "entries_per_core"};
    consumer.tensor_bindings = {
        {.tensor_parameter_name = TensorParamName{"dst_tensor"}, .accessor_name = "dst_tensor"}};
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{kBorrowedDfb}, "in"));

    DataflowBufferSpec dfb{
        .unique_id = DFBSpecName{kBorrowedDfb},
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .borrowed_from = TensorParamName{"dfb_ring_tensor"}};

    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"src_tensor"}, .spec = make_flat_dram_tensor_spec(entry_size, num_entries)},
        {.unique_id = TensorParamName{"dst_tensor"}, .spec = make_flat_dram_tensor_spec(entry_size, num_entries)},
        {.unique_id = TensorParamName{"dfb_ring_tensor"}, .spec = make_flat_l1_tensor_spec(entry_size, num_entries)}};
    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("wu", node, {"producer", "consumer"})};
    return spec;
}

// Compile the borrowed program and allocate its backing tensors (two swappable L1 rings).
BorrowedProgram make_borrowed_program(uint32_t entry_size, uint32_t num_entries) {
    BorrowedProgram bp;
    bp.device = open_device();
    bp.num_entries = num_entries;

    const ProgramSpec spec = build_borrowed_spec(entry_size, num_entries, bp.node);
    const TensorSpec src_spec = make_flat_dram_tensor_spec(entry_size, num_entries);
    const TensorSpec dst_spec = make_flat_dram_tensor_spec(entry_size, num_entries);
    const TensorSpec ring_spec = make_flat_l1_tensor_spec(entry_size, num_entries);

    bp.program = MakeProgramFromSpec(*bp.device, spec);
    bp.src.emplace(MeshTensor::allocate_on_device(*bp.device, src_spec, TensorTopology{}));
    bp.dst.emplace(MeshTensor::allocate_on_device(*bp.device, dst_spec, TensorTopology{}));
    bp.ring_a.emplace(MeshTensor::allocate_on_device(*bp.device, ring_spec, TensorTopology{}));
    bp.ring_b.emplace(MeshTensor::allocate_on_device(*bp.device, ring_spec, TensorTopology{}));
    return bp;
}

Table<TensorParamName, TensorArgument> borrowed_tensor_args(const BorrowedProgram& bp, const MeshTensor& ring) {
    return Table<TensorParamName, TensorArgument>{
        {TensorParamName{"src_tensor"}, TensorArgument{*bp.src}},
        {TensorParamName{"dst_tensor"}, TensorArgument{*bp.dst}},
        {TensorParamName{"dfb_ring_tensor"}, TensorArgument{ring}}};
}

ProgramRunArgs borrowed_run_args(const BorrowedProgram& bp, const MeshTensor& ring) {
    const auto rtas =
        MakeRuntimeArgsForSingleNode(bp.node, {{"chunk_offset", 0u}, {"entries_per_core", bp.num_entries}});
    ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = KernelSpecName{"producer"}, .runtime_arg_values = rtas},
        {.kernel = KernelSpecName{"consumer"}, .runtime_arg_values = rtas}};
    params.tensor_args = borrowed_tensor_args(bp, ring);
    return params;
}

// #7 borrowed rebind: time UpdateTensorArgs swapping the borrowed ring between two L1 tensors.
void BM_borrowed_rebind(benchmark::State& state) {
    BorrowedProgram bp = make_borrowed_program(/*entry_size=*/256, /*num_entries=*/16);
    SetProgramRunArgs(bp.program, borrowed_run_args(bp, *bp.ring_a));

    bool toggle = false;
    run_manual_timed(
        state,
        [&]() {
            toggle = !toggle;
            UpdateTensorArgs(bp.program, borrowed_tensor_args(bp, toggle ? *bp.ring_b : *bp.ring_a));
        },
        nullptr);
    bp.device->close();
}

// #7 borrowed resize: time SetProgramRunArgs applying a num_entries override on the borrowed DFB
// (re-attach + per-bank fit check). Sizes alternate but always fit the 16-entry backing tensor.
void BM_borrowed_resize(benchmark::State& state) {
    BorrowedProgram bp = make_borrowed_program(/*entry_size=*/256, /*num_entries=*/16);
    SetProgramRunArgs(bp.program, borrowed_run_args(bp, *bp.ring_a));

    bool toggle = false;
    run_manual_timed(
        state,
        [&]() {
            toggle = !toggle;
            ProgramRunArgs params = borrowed_run_args(bp, *bp.ring_a);
            params.dfb_run_overrides.push_back(
                DFBRunOverrides{.dfb = DFBSpecName{kBorrowedDfb}, .num_entries = toggle ? 8u : 16u});
            SetProgramRunArgs(bp.program, params);
        },
        nullptr);
    bp.device->close();
}

}  // namespace

// --- #5 host program (spec) building ---
BENCHMARK_CAPTURE(BM_program_spec_build, common, COMMON_SHAPE)->Apply(RepeatedConfig);
BENCHMARK_CAPTURE(BM_program_spec_build, worst, WORST_SHAPE)->Apply(RepeatedConfig);

// --- #6 enqueue program ---
BENCHMARK_CAPTURE(BM_enqueue_program, common, COMMON_SHAPE)->Apply(RepeatedConfig);
BENCHMARK_CAPTURE(BM_enqueue_program, worst, WORST_SHAPE)->Apply(RepeatedConfig);

// --- #7 mutable run-arg updates (stable, sub-ms; single run) ---
BENCHMARK_CAPTURE(BM_run_args_update, common, COMMON_SHAPE)->Iterations(DEFAULT_ITERATIONS)->UseManualTime();
BENCHMARK_CAPTURE(BM_run_args_update, worst, WORST_SHAPE)->Iterations(DEFAULT_ITERATIONS)->UseManualTime();

// --- #7 mutable DFB size overrides (worst case only; common case has no DFBs) ---
BENCHMARK_CAPTURE(BM_dfb_override, worst, WORST_SHAPE)->Apply(RepeatedConfig);

// --- #7 borrowed-memory DFB updates (rebind + resize) ---
BENCHMARK(BM_borrowed_rebind)->Apply(RepeatedConfig);
BENCHMARK(BM_borrowed_resize)->Apply(RepeatedConfig);

// --- #6 enqueue write buffer (device-transfer dominated, stable; single run) ---
BENCHMARK_CAPTURE(BM_enqueue_write_buffer, dram, BufferType::DRAM)
    ->Apply(PageSizeArgs)
    ->Iterations(DEFAULT_ITERATIONS)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_enqueue_write_buffer, l1, BufferType::L1)
    ->Apply(PageSizeArgs)
    ->Iterations(DEFAULT_ITERATIONS)
    ->UseManualTime();

// --- #6 enqueue read buffer (blocking, bandwidth-dominated, stable; single run) ---
BENCHMARK_CAPTURE(BM_enqueue_read_buffer, dram, BufferType::DRAM)
    ->Apply(PageSizeArgs)
    ->Iterations(DEFAULT_ITERATIONS)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_enqueue_read_buffer, l1, BufferType::L1)
    ->Apply(PageSizeArgs)
    ->Iterations(DEFAULT_ITERATIONS)
    ->UseManualTime();

// --- #6 trace capture (allocates device buffers each iter; single run) + replay (repeated + median) ---
BENCHMARK_CAPTURE(BM_trace_capture, common, COMMON_SHAPE)->Iterations(TRACE_ITERATIONS)->UseManualTime();
BENCHMARK_CAPTURE(BM_trace_capture, worst, WORST_SHAPE)->Iterations(TRACE_ITERATIONS)->UseManualTime();
BENCHMARK_CAPTURE(BM_trace_replay, common, COMMON_SHAPE)->Apply(RepeatedConfig);
BENCHMARK_CAPTURE(BM_trace_replay, worst, WORST_SHAPE)->Apply(RepeatedConfig);

BENCHMARK_MAIN();
