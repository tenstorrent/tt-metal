// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar-only DFB init timing benchmarks
// Device rdcycle instrumentation is gated by TT_METAL_MEASURE_DFB_INIT_TIME=1.
//
// Usage:
//   export TT_METAL_SLOW_DISPATCH_MODE=1
//   export TT_METAL_MEASURE_DFB_INIT_TIME=1
//   ./build/test/tt_metal/dfb_init_timing_bench [--case base|two|...|all]

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "tt_metal/impl/host_api/temp_quasar_api.hpp"

namespace tt::tt_metal {

struct DfbInitTimingBenchContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    IDevice* device = nullptr;
};

DfbInitTimingBenchContext create_dfb_init_timing_bench_context() {
    if (!std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        TT_FATAL(false, "Set TT_METAL_SLOW_DISPATCH_MODE=1 to run dfb_init_timing_bench");
    }
    if (MetalContext::instance().get_cluster().arch() != ARCH::QUASAR) {
        TT_FATAL(false, "dfb_init_timing_bench requires Quasar");
    }
    if (!MetalContext::instance().rtoptions().get_measure_dfb_init_time_enabled()) {
        TT_FATAL(false, "Set TT_METAL_MEASURE_DFB_INIT_TIME=1 to run dfb_init_timing_bench");
    }

    std::vector<ChipId> ids;
    for (ChipId id : MetalContext::instance().get_cluster().mmio_chip_ids()) {
        ids.push_back(id);
    }
    TT_FATAL(!ids.empty(), "No MMIO devices available");

    const auto& dispatch_core_config = MetalContext::instance().rtoptions().get_dispatch_core_config();
    auto id_to_device = distributed::MeshDevice::create_unit_meshes(
        ids,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        dispatch_core_config);

    DfbInitTimingBenchContext ctx;
    ctx.mesh_device = id_to_device.begin()->second;
    ctx.device = ctx.mesh_device->get_devices()[0];
    return ctx;
}

inline tt::tt_metal::TensorSpec make_flat_dram_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    const uint32_t entry_size_words = entry_size / sizeof(uint32_t);
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    return tt::tt_metal::TensorSpec(tt::tt_metal::Shape{total_entries, entry_size_words}, tensor_layout);
}

namespace {
constexpr tt::DataFormat kDfbBenchDataFormat = tt::DataFormat::Float16_b;

experimental::DataflowBufferSpec MakeBenchDfbSpec(
    experimental::DFBSpecName id, uint32_t entry_size, uint32_t num_entries) {
    return experimental::DataflowBufferSpec{
        .unique_id = std::move(id),
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = kDfbBenchDataFormat,
    };
}

constexpr const char* kDfbInitTimingSlotNames[::dfb::DFB_INIT_TIMING_NUM_SLOTS] = {
    "DM0",
    "DM1",
    "DM2",
    "DM3",
    "DM4",
    "DM5",
    "DM6",
    "DM7",
    "Neo0 unpack",
    "Neo0 pack",
    "Neo1 unpack",
    "Neo1 pack",
    "Neo2 unpack",
    "Neo2 pack",
    "Neo3 unpack",
    "Neo3 pack",
};

void ClearDfbInitTimingL1(IDevice* device, const CoreCoord& core) {
    std::vector<uint32_t> zeros(::dfb::DFB_INIT_TIMING_REGION_BYTES / sizeof(uint32_t), 0u);
    detail::WriteToDeviceL1(device, core, ::dfb::DFB_INIT_TIMING_L1_BYTE_OFFSET, zeros);
}

// Slots that this program's DFB participants are expected to write.
// DM0 (ISR) and DM1 (remapper) always run coordinator paths for any DFB program.
// DM2-7 / Neo unpack+pack are included only when their risc bit appears in a DFB mask.
uint16_t DfbInitTimingUsedSlotsMask(Program& program, const CoreCoord& core) {
    uint16_t mask = (1u << 0) | (1u << 1);
    for (const auto& dfb : program.impl().dataflow_buffers_on_core(core)) {
        const uint16_t combined =
            static_cast<uint16_t>(dfb->config.producer_risc_mask | dfb->config.consumer_risc_mask);
        for (uint8_t dm = 0; dm < 8; ++dm) {
            if (combined & (1u << dm)) {
                mask |= static_cast<uint16_t>(1u << dm);
            }
        }
        for (uint8_t neo = 0; neo < 4; ++neo) {
            if (combined & (1u << (::dfb::TENSIX_RISC_OFFSET + neo))) {
                mask |= static_cast<uint16_t>(1u << (8u + neo * 2u));      // unpack
                mask |= static_cast<uint16_t>(1u << (8u + neo * 2u + 1u));  // pack
            }
        }
    }
    return mask;
}

void LogDfbInitTimingFromL1(
    IDevice* device, const CoreCoord& core, const char* benchmark_name, uint16_t used_slots_mask) {
    if (!MetalContext::instance().rtoptions().get_measure_dfb_init_time_enabled()) {
        log_info(
            tt::LogTest,
            "DFB init timing [{}]: disabled (set TT_METAL_MEASURE_DFB_INIT_TIME=1)",
            benchmark_name);
        return;
    }

    log_info(
        tt::LogTest,
        "DFB init timing [{}] @ L1 0x{:x} (used_slots_mask=0x{:x}):",
        benchmark_name,
        ::dfb::DFB_INIT_TIMING_L1_BYTE_OFFSET,
        used_slots_mask);

    const uint32_t dfb_init_timing_slot_bytes =
        static_cast<uint32_t>(::dfb::DFB_INIT_TIMING_WORDS_PER_SLOT) * sizeof(uint32_t);

    for (uint8_t slot = 0; slot < ::dfb::DFB_INIT_TIMING_NUM_SLOTS; ++slot) {
        if ((used_slots_mask & (1u << slot)) == 0u) {
            continue;
        }

        const uint32_t slot_l1_addr =
            ::dfb::DFB_INIT_TIMING_L1_BYTE_OFFSET + static_cast<uint32_t>(slot) * dfb_init_timing_slot_bytes;
        std::vector<uint32_t> timing_words;
        detail::ReadFromDeviceL1(device, core, slot_l1_addr, dfb_init_timing_slot_bytes, timing_words);
        const uint32_t* words = timing_words.data();

        if (words[::dfb::DFB_INIT_TIMING_W_VALID] != 1u ||
            words[::dfb::DFB_INIT_TIMING_W_MAGIC] != ::dfb::DFB_INIT_TIMING_MAGIC) {
            log_info(tt::LogTest, "  {}: (not written)", kDfbInitTimingSlotNames[slot]);
            continue;
        }

        const uint8_t role = static_cast<uint8_t>(words[::dfb::DFB_INIT_TIMING_W_ROLE]);
        const uint32_t e2e = words[::dfb::DFB_INIT_TIMING_W_E2E];
        const uint32_t metric_a = words[::dfb::DFB_INIT_TIMING_W_METRIC_A];
        const uint32_t metric_b = words[::dfb::DFB_INIT_TIMING_W_METRIC_B];
        const uint32_t metric_c = words[::dfb::DFB_INIT_TIMING_W_METRIC_C];
        const uint32_t metric_d = words[::dfb::DFB_INIT_TIMING_W_METRIC_D];
        const uint32_t metric_e = words[::dfb::DFB_INIT_TIMING_W_METRIC_E];
        const uint32_t metric_f = words[::dfb::DFB_INIT_TIMING_W_METRIC_F];
        const uint32_t metric_g = words[::dfb::DFB_INIT_TIMING_W_METRIC_G];
        const uint32_t metric_h = words[::dfb::DFB_INIT_TIMING_W_METRIC_H];
        const uint32_t metric_i = words[::dfb::DFB_INIT_TIMING_W_METRIC_I];
        const uint32_t metric_j = words[::dfb::DFB_INIT_TIMING_W_METRIC_J];
        const uint32_t start_time = words[::dfb::DFB_INIT_TIMING_W_START];
        const uint32_t end_time = words[::dfb::DFB_INIT_TIMING_W_END];

        if (role == ::dfb::DFB_INIT_TIMING_ROLE_DM0_ISR) {
            log_info(
                tt::LogTest,
                "  {}: e2e={} pre_loop_sw={} subpassB_desc={} between_dfb_sw={} subpassB_l1_read={} "
                "subpassB_rocc_issue={} subpassB_hw={} first_ie_rmw={} second_ie_rmw={} isr_enable={} "
                "start={} end={}",
                kDfbInitTimingSlotNames[slot],
                e2e,
                metric_a,
                metric_b,
                metric_c,
                metric_d,
                metric_e,
                metric_j,
                metric_f,
                metric_g,
                metric_h,
                start_time,
                end_time);
        } else if (role == ::dfb::DFB_INIT_TIMING_ROLE_DM1_RMP) {
            log_info(
                tt::LogTest,
                "  {}: e2e={} blob_l1_read_sw={} blob_loop_ovhd={} pairs_reg_hw={} enable_remapper_hw={} "
                "first_pair_clientR_hw={} first_pair_clientL_hw={} last_pair_hw={} pairs_slots_written={} "
                "start={} end={}",
                kDfbInitTimingSlotNames[slot],
                e2e,
                metric_a,
                metric_b,
                metric_c,
                metric_d,
                metric_e,
                metric_f,
                metric_g,
                metric_j,
                start_time,
                end_time);
        } else {
            log_info(
                tt::LogTest,
                "  {}: e2e={} merged_sw={} remapper_spin={} tc_hw={} wait_all={} tc_reset_hw={} "
                "tc_capacity_hw={} pre_loop={} entry_hdr={} tc_slots={} sig_write={} start={} end={}",
                kDfbInitTimingSlotNames[slot],
                e2e,
                metric_a,
                metric_b,
                metric_c,
                metric_d,
                metric_e,
                metric_f,
                metric_g,
                metric_h,
                metric_i,
                metric_j,
                start_time,
                end_time);
        }
    }
}

void LaunchAndLogDfbInitTiming(
    IDevice* device, Program& program, const CoreCoord& core, const char* benchmark_name) {
    ClearDfbInitTimingL1(device, core);
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);
    LogDfbInitTimingFromL1(device, core, benchmark_name, DfbInitTimingUsedSlotsMask(program, core));
}
}  // namespace

// Base-case DFB init benchmark.
//
// One 1Sx1S DM→Tensix DFB on core (0,0) — same configuration as
// DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx1S/ImplicitSyncTrue:
//   entry_size=1024, num_entries=16, 1 STRIDED producer, 1 STRIDED consumer,
//   implicit sync enabled, no remapper.
//
// Reuses dfb_producer.cpp and dfb_t6_consumer.cpp (same kernels as DMTensixTest1xDFB1Sx1S).
// Data movement is commented out in those kernels so measured time reflects DFB init overhead.
void run_benchmark_case_base(DfbInitTimingBenchContext& ctx) {
    auto mesh_device = ctx.mesh_device;
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE = 1024;
    constexpr uint32_t NUM_ENTRIES = 16;
    constexpr uint8_t NUM_PRODUCERS = 1;
    constexpr uint8_t NUM_CONSUMERS = 1;
    constexpr uint32_t NUM_ENTRIES_PER_PRODUCER = NUM_ENTRIES;
    constexpr uint32_t NUM_ENTRIES_PER_CONSUMER = NUM_ENTRIES;

    const experimental::DFBSpecName DFB{"dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};
    const experimental::TensorParamName IN_TENSOR{"in_tensor"};

    const experimental::DataMovementHardwareConfig dm_producer_cfg = experimental::DataMovementGen2Config{};

    auto in_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(ENTRY_SIZE, NUM_ENTRIES), TensorTopology{});

    experimental::DataflowBufferSpec dfb_spec = MakeBenchDfbSpec(DFB, ENTRY_SIZE, NUM_ENTRIES);

    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        .num_threads = NUM_PRODUCERS,
        .dfb_bindings = {experimental::ProducerOf(DFB, "out")},
        .tensor_bindings = {{
            .tensor_parameter_name = IN_TENSOR,
            .accessor_name = "src_tensor",
        }},
        .compile_time_args =
            {
                {"num_entries_per_producer", NUM_ENTRIES_PER_PRODUCER},
                {"implicit_sync", 1u},
                {"num_producers", static_cast<uint32_t>(NUM_PRODUCERS)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}},
        .hw_config = dm_producer_cfg,
    };

    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp",
        .num_threads = NUM_CONSUMERS,
        .dfb_bindings = {experimental::StridedConsumerOf(DFB, "in")},
        .compile_time_args = {{"num_entries_per_consumer", NUM_ENTRIES_PER_CONSUMER}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "bench_base_wu",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = core_range_set,
    };

    experimental::ProgramSpec spec{
        .name = "bench_base",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()}},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    experimental::ProgramRunArgs::KernelRunArgs producer_params{};
    producer_params.kernel = PRODUCER;
    producer_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        experimental::NodeCoord{0, 0},
        {{"chunk_offset", 0u}, {"entries_per_core", NUM_ENTRIES}});
    experimental::ProgramRunArgs::KernelRunArgs consumer_params{};
    consumer_params.kernel = CONSUMER;
    run_args.kernel_run_args = {producer_params, consumer_params};
    run_args.tensor_args.emplace(IN_TENSOR, experimental::TensorArgument{in_tensor});
    experimental::SetProgramRunArgs(program, run_args);

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseBase");
}

void run_benchmark_case_two(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE  = 1024;
    constexpr uint32_t NUM_ENTRIES = 16;
    constexpr uint8_t  NUM_IN_THREADS = 4;
    constexpr uint8_t  NUM_OUT_THREADS = 2;

    const experimental::DFBSpecName DFB_SS{"dfb_ss"};   // 4Sx4S DM→Tensix
    const experimental::DFBSpecName DFB_SA{"dfb_sa"};   // 4Sx4A DM→Tensix
    const experimental::DFBSpecName DFB_T6{"dfb_t6"};   // 4Sx4S Tensix→DM
    const experimental::KernelSpecName READER{"reader_dm"};
    const experimental::KernelSpecName COMPUTE{"compute"};
    const experimental::KernelSpecName WRITER{"writer_dm"};

    const experimental::DataMovementHardwareConfig gen2_dm_hw = experimental::DataMovementGen2Config{};

    experimental::DataflowBufferSpec dfb_ss_spec = MakeBenchDfbSpec(DFB_SS, ENTRY_SIZE, NUM_ENTRIES);
    experimental::DataflowBufferSpec dfb_sa_spec = MakeBenchDfbSpec(DFB_SA, ENTRY_SIZE, NUM_ENTRIES);
    experimental::DataflowBufferSpec dfb_t6_spec = MakeBenchDfbSpec(DFB_T6, ENTRY_SIZE, NUM_ENTRIES);

    // Reader DM: producer on DFB_SS (STRIDED) and DFB_SA (STRIDED)
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_avg_reader_dm.cpp",
        .num_threads = NUM_IN_THREADS,
        .dfb_bindings = {
            experimental::ProducerOf(DFB_SS, "ss_out"),
            experimental::ProducerOf(DFB_SA, "sa_out"),
        },
        .hw_config = gen2_dm_hw,
    };

    // Compute: STRIDED consumer on DFB_SS, ALL consumer on DFB_SA, STRIDED producer on DFB_T6
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_avg_compute.cpp",
        .num_threads = NUM_IN_THREADS,
        .dfb_bindings = {
            experimental::StridedConsumerOf(DFB_SS, "ss_in"),
            experimental::AllConsumerOf(DFB_SA, "sa_in"),
            experimental::ProducerOf(DFB_T6, "t6_out"),
        },
        .hw_config = experimental::ComputeGen2Config{},
    };

    // Writer DM: STRIDED consumer on DFB_T6
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_avg_writer_dm.cpp",
        .num_threads = NUM_OUT_THREADS,
        .dfb_bindings = {experimental::StridedConsumerOf(DFB_T6, "t6_in")},
        .hw_config = gen2_dm_hw,
    };

    experimental::WorkUnitSpec wu{
        .name = "bench_avg_wu",
        .kernels = {READER, COMPUTE, WRITER},
        .target_nodes = core_range_set,
    };

    experimental::ProgramSpec spec{
        .name = "bench_avg",
        .kernels = {reader_spec, compute_spec, writer_spec},
        .dataflow_buffers = {dfb_ss_spec, dfb_sa_spec, dfb_t6_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*ctx.mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = READER},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = WRITER},
    };
    experimental::SetProgramRunArgs(program, run_args);

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseTwo");
}

// Worst-case DFB init benchmark.
//
// Three concurrent 4Sx4A DM→Tensix DFBs are the hardware worst case for
// DM0's setup_local_dfb_interfaces.
//
// Per-tensix TC budget (16 DM-visible TCs per tensix):
//   Each DFB allocates 5 TCs on each of the consumer tensix IDs:
//     - 1 producer TC (DM k → tensix k)
//     - 4 consumer TCs (Neo k gets 1 TC per DM producer × 4 producers, all on tensix k)
//   So each DFB uses 5 TCs on tensix 0–3 → max DFBs = floor(16/5) = 3.
//   3 DFBs × 5 = 15/16 TCs used per tensix.
//
// Remapper: 3 DFBs × 4 producers (each 1-to-4 fan-out) = 12/16 1-to-many entries used.
// This yields 12 remapper write_all_configs() calls with 4 set_clientR_slot writes
// each = 48 total clientR writes — the maximum achievable given the TC constraint.
//
// Drains the full 16-entry ring per DFB (num_entries=16, 4 producers / 4 ALL consumers):
//   reader: 4 implicit reads per DFB per DM
//   compute: 16 copy_tile+pop_front per Neo per DFB (4 ALL TCs × 4 entries)
void run_benchmark_case_four(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE    = 1024;
    constexpr uint32_t NUM_ENTRIES   = 16;
    constexpr uint8_t  NUM_PRODUCERS = 4;  // DM STRIDED producers per DFB
    constexpr uint8_t  NUM_CONSUMERS = 4;  // Tensix ALL consumers per DFB

    const experimental::DFBSpecName DFB0{"dfb0"};
    const experimental::DFBSpecName DFB1{"dfb1"};
    const experimental::DFBSpecName DFB2{"dfb2"};
    const experimental::KernelSpecName READER{"reader_dm"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const experimental::DataMovementHardwareConfig gen2_dm_hw = experimental::DataMovementGen2Config{};

    // Reader DM: 4 STRIDED producers on all three DFBs
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_worst_reader_dm.cpp",
        .num_threads = NUM_PRODUCERS,
        .dfb_bindings = {
            experimental::ProducerOf(DFB0, "out0"),
            experimental::ProducerOf(DFB1, "out1"),
            experimental::ProducerOf(DFB2, "out2"),
        },
        .hw_config = gen2_dm_hw,
    };

    // Compute: 4 ALL consumers on all three DFBs (each Neo gets a full copy via remapper)
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_worst_compute.cpp",
        .num_threads = NUM_CONSUMERS,
        .dfb_bindings = {
            experimental::AllConsumerOf(DFB0, "in0"),
            experimental::AllConsumerOf(DFB1, "in1"),
            experimental::AllConsumerOf(DFB2, "in2"),
        },
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "bench_worst_wu",
        .kernels = {READER, COMPUTE},
        .target_nodes = core_range_set,
    };

    experimental::ProgramSpec spec{
        .name = "bench_worst",
        .kernels = {reader_spec, compute_spec},
        .dataflow_buffers = {
            MakeBenchDfbSpec(DFB0, ENTRY_SIZE, NUM_ENTRIES),
            MakeBenchDfbSpec(DFB1, ENTRY_SIZE, NUM_ENTRIES),
            MakeBenchDfbSpec(DFB2, ENTRY_SIZE, NUM_ENTRIES),
        },
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*ctx.mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = READER},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, run_args);

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseFour");
}

// Average-case-two DFB init benchmark.
//
// Three concurrent 4Sx4S DM→Tensix DFBs: 4 STRIDED DM producers, 4 STRIDED
// Tensix consumers. This is a STRIDED-STRIDED variant of BenchmarkCaseWorst
// (which uses ALL consumers). No remapper is used, making this a clean mid-point
// between BenchmarkCaseBase (1 DFB, no remapper) and BenchmarkCaseWorst (remapper).
//
// Per-tensix TC budget:
//   Each 4Sx4S DFB uses 1 TC per (producer, consumer) pair, all on the same tensix.
//   4 pairs × 3 DFBs = 12 TCs total, 3 per tensix (12/16 used).
//   No remapper entries.
//
// Drains the full 16-entry ring per DFB (num_entries=16, 4 producers / 4 consumers):
//   reader: 4 implicit reads per DFB per DM
//   compute: 4 copy_tile+pop_front per Neo per DFB (eltwise_copy tile_regs pattern)
void run_benchmark_case_three(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE    = 1024;
    constexpr uint32_t NUM_ENTRIES   = 16;
    constexpr uint8_t  NUM_PRODUCERS = 4;  // DM STRIDED producers per DFB
    constexpr uint8_t  NUM_CONSUMERS = 4;  // Tensix STRIDED consumers per DFB

    const experimental::DFBSpecName DFB0{"avg2_dfb0"};
    const experimental::DFBSpecName DFB1{"avg2_dfb1"};
    const experimental::DFBSpecName DFB2{"avg2_dfb2"};
    const experimental::KernelSpecName READER{"reader_dm"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const experimental::DataMovementHardwareConfig gen2_dm_hw = experimental::DataMovementGen2Config{};

    // Reader DM: 4 STRIDED producers on all three DFBs.
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_avg2_reader_dm.cpp",
        .num_threads = NUM_PRODUCERS,
        .dfb_bindings = {
            experimental::ProducerOf(DFB0, "out0"),
            experimental::ProducerOf(DFB1, "out1"),
            experimental::ProducerOf(DFB2, "out2"),
        },
        .hw_config = gen2_dm_hw,
    };

    // Compute: 4 STRIDED consumers on all three DFBs.
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_avg2_compute.cpp",
        .num_threads = NUM_CONSUMERS,
        .dfb_bindings = {
            experimental::StridedConsumerOf(DFB0, "in0"),
            experimental::StridedConsumerOf(DFB1, "in1"),
            experimental::StridedConsumerOf(DFB2, "in2"),
        },
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "bench_avg2_wu",
        .kernels = {READER, COMPUTE},
        .target_nodes = core_range_set,
    };

    experimental::ProgramSpec spec{
        .name = "bench_avg2",
        .kernels = {reader_spec, compute_spec},
        .dataflow_buffers = {
            MakeBenchDfbSpec(DFB0, ENTRY_SIZE, NUM_ENTRIES),
            MakeBenchDfbSpec(DFB1, ENTRY_SIZE, NUM_ENTRIES),
            MakeBenchDfbSpec(DFB2, ENTRY_SIZE, NUM_ENTRIES),
        },
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*ctx.mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = READER},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, run_args);

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseThree");
}

// Worst-case-two DFB init benchmark.
//
// Stresses the DFB init loop with the maximum number of DFBs achievable under
// the TC budget when a single 4-thread compute kernel consumes all of them.
//
// With a single compute kernel (num_threads=4), all 4 Neo threads consume EVERY
// DFB via the ALL pattern. This forces every DFB to be 1Sx4A (1 DM producer,
// 4 Neo consumers). TC cost per DFB:
//   - 1 producer TC on producer tensix
//   - 1 consumer TC per Neo tensix (Neo k shares tensix k with DM k → 2 TCs
//     on the producer tensix, 1 TC on each other tensix) = 5 TCs total
//
// Cycling 4 DMs with 3 DFBs each: each tensix accumulates 3×2 + 3×1×3 = 15 TCs.
// Maximum: 4 DMs × 3 DFBs = 12 DFBs (60/64 TCs used, 15 per tensix).
//
// Contrast with BenchmarkCaseWorst (3×4Sx4A):
//   - Same 12 remapper entries (1-to-4), same 48 set_clientR_slot writes.
//   - 4× more DFBs (12 vs 3) → 4× more init loop iterations per RISC.
//   - 4 separate single-thread DM readers vs 1 four-thread DM reader.
//
// Remapper: 12 entries (all 1-to-4), 48 set_clientR_slot writes.
//
// Drains the full 16-entry ring per DFB (num_entries=16, 1 producer / 4 ALL consumers):
//   reader: 16 implicit reads per DFB per DM (single producer)
//   compute: 16 copy_tile+pop_front per Neo per DFB across all 12 DFBs
void run_benchmark_case_five(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE  = 1024;
    constexpr uint32_t NUM_ENTRIES = 16;

    auto dfb_id = [](char group, int i) -> experimental::DFBSpecName {
        return experimental::DFBSpecName{std::string("dfb_") + group + std::to_string(i)};
    };

    const experimental::DataMovementHardwareConfig gen2_dm_hw = experimental::DataMovementGen2Config{};

    // Each reader: single DM, 1Sx4A, 16 reads per DFB (full ring).
    const char* READER_SRC =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_worst2_reader_dm.cpp";
    const char* COMPUTE_SRC =
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_worst2_compute.cpp";

    // Each reader: 1 DM thread, STRIDED producer on 3 DFBs.
    auto make_reader_bindings = [&](char group) -> std::vector<experimental::DFBBinding> {
        std::vector<experimental::DFBBinding> bindings;
        bindings.reserve(3);
        for (int i = 0; i < 3; i++) {
            bindings.push_back(experimental::ProducerOf(dfb_id(group, i), "out" + std::to_string(i)));
        }
        return bindings;
    };

    // Single compute kernel: 4 Neo threads, ALL consumer on all 12 DFBs.
    std::vector<experimental::DFBBinding> compute_bindings;
    {
        int in_idx = 0;
        for (char g : {'a', 'b', 'c', 'd'}) {
            for (int i = 0; i < 3; i++) {
                compute_bindings.push_back(
                    experimental::AllConsumerOf(dfb_id(g, i), "in" + std::to_string(in_idx++)));
            }
        }
    }

    std::vector<experimental::KernelSpec> kernels;
    for (char g : {'a', 'b', 'c', 'd'}) {
        kernels.push_back({
            .unique_id = experimental::KernelSpecName{std::string("reader_") + g},
            .source = READER_SRC,
            .num_threads = 1,
            .dfb_bindings = make_reader_bindings(g),
            .hw_config = gen2_dm_hw,
        });
    }
    kernels.push_back({
        .unique_id = experimental::KernelSpecName{"compute_all"},
        .source = COMPUTE_SRC,
        .num_threads = 4,
        .dfb_bindings = compute_bindings,
        .hw_config = experimental::ComputeGen2Config{},
    });

    // 12 DFB specs: 3 per group × 4 groups.
    std::vector<experimental::DataflowBufferSpec> dfb_specs;
    for (char g : {'a', 'b', 'c', 'd'}) {
        for (int i = 0; i < 3; i++) {
            dfb_specs.push_back(MakeBenchDfbSpec(dfb_id(g, i), ENTRY_SIZE, NUM_ENTRIES));
        }
    }

    std::vector<experimental::KernelSpecName> all_kernel_ids = {
        experimental::KernelSpecName{"reader_a"},
        experimental::KernelSpecName{"reader_b"},
        experimental::KernelSpecName{"reader_c"},
        experimental::KernelSpecName{"reader_d"},
        experimental::KernelSpecName{"compute_all"},
    };

    experimental::WorkUnitSpec wu{
        .name = "bench_worst2_wu",
        .kernels = all_kernel_ids,
        .target_nodes = core_range_set,
    };

    experimental::ProgramSpec spec{
        .name = "bench_worst2",
        .kernels = kernels,
        .dataflow_buffers = dfb_specs,
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*ctx.mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    for (const auto& kid : all_kernel_ids) {
        run_args.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{.kernel = kid});
    }
    experimental::SetProgramRunArgs(program, run_args);

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseFive");
}

// Worst-case-three DFB init benchmark.
//
// Uses the experimental explicit producer_risc_mask / consumer_risc_mask APIs
// to mix 6 independent DM producers with a single 4-thread compute consumer,
// creating 32 DFBs that are impossible to express through the standard binding
// API (which can only select the first N DMs for an N-thread kernel).
//
// Configuration: 16 × 1Sx1S + 16 × 1Sx2A = 32 DFBs, 64/64 TCs, 6 DMs active
//
//   1Sx1S DFBs (0–15): one DM producer, one Neo STRIDED consumer (no remapper)
//     DFB  0– 3 : DM4 (t0) → Neo0 (t0)   [4 TCs on t0]
//     DFB  4– 7 : DM5 (t1) → Neo1 (t1)   [4 TCs on t1]
//     DFB  8–11 : DM4 (t0) → Neo2 (t2)   [4 TCs on t2]
//     DFB 12–15 : DM5 (t1) → Neo3 (t3)   [4 TCs on t3]
//
//   1Sx2A DFBs (16–31): one DM producer, two Neo ALL consumers (remapper 1-to-2)
//     DFB 16–19 : DM2 (t2) → {Neo0, Neo2}  [t0:+4, t2:+8]
//     DFB 20–23 : DM3 (t3) → {Neo0, Neo2}  [t0:+4, t2:+4, t3:+4]
//     DFB 24–27 : DM4 (t0) → {Neo1, Neo3}  [t0:+4, t1:+4, t3:+4]
//     DFB 28–31 : DM5 (t1) → {Neo1, Neo3}  [t1:+8, t3:+4]
//
// TC totals: t0=16, t1=16, t2=16, t3=16 (64/64 used).
// Remapper:  16 × 1-to-2 entries — exhausts all 16 one-to-many remapper slots.
//            32 set_clientR_slot writes.
//
// Implicit sync is enabled to exercise ISR setup. num_entries=1 is required to
// stay within the 32-slot TxnIdAllocator budget: with num_entries=1 the
// compute_optimal_txn_id_count function returns 1 (no n≥2 divides 1), so each
// DFB consumes exactly 1 txn ID. Any even num_entries would return n=2 (budget
// exhausted at DFB 17).
//
// Cases 6/7 use entry_size=2048 (not 1024): num_entries=1 cannot use the Cases 2–5
// workaround where a default 2048 B unpack over-reads into the next ring slot.
// Quasar copy_tile requires standard 32×32 tile geometry (narrow/partial tiles fault).
void run_benchmark_case_seven(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE  = 2048;
    // num_entries=1 is intentional: compute_optimal_txn_id_count iterates n=2..32
    // looking for num_entries % (n * num_producers * num_tcs_per_risc) == 0.
    // With num_entries=1 no n≥2 divides 1, so the function returns 1 txn ID per DFB.
    // 32 DFBs × 1 txn ID = 32, fitting exactly in the 32-slot TxnIdAllocator budget.
    // Any even num_entries (e.g. 4) would return n=2, exhausting the budget at DFB 17.
    constexpr uint32_t NUM_ENTRIES = 1;

    // risc_mask bit positions
    // DM k  → bit k        (bits 0–7)
    // Neo k → bit (8 + k)  (bits 8–11)
    constexpr uint16_t DM2  = (1u << 2);
    constexpr uint16_t DM3  = (1u << 3);
    constexpr uint16_t DM4  = (1u << 4);
    constexpr uint16_t DM5  = (1u << 5);
    constexpr uint16_t NEO0 = (1u << 8);
    constexpr uint16_t NEO1 = (1u << 9);
    constexpr uint16_t NEO2 = (1u << 10);
    constexpr uint16_t NEO3 = (1u << 11);

    Program program = CreateProgram();

    // -----------------------------------------------------------------------
    // 1Sx1S DFBs (IDs 0–15): STRIDED producer, STRIDED consumer, no remapper.
    // TC is shared on the Neo consumer's tensix.
    // -----------------------------------------------------------------------
    auto make_1sx1s = [&](uint16_t producer_dm, uint16_t consumer_neo) {
        return experimental::dfb::DataflowBufferConfig{
            .entry_size          = ENTRY_SIZE,
            .num_entries         = NUM_ENTRIES,
            .producer_risc_mask  = producer_dm,
            .num_producers       = 1,
            .pap                 = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask  = consumer_neo,
            .num_consumers       = 1,
            .cap                 = dfb::AccessPattern::STRIDED,
            .enable_producer_implicit_sync = true,
            .enable_consumer_implicit_sync = true,
            .data_format         = kDfbBenchDataFormat,
        };
    };

    // DFBs 0–3: DM4 → Neo0
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1s(DM4, NEO0));
    }
    // DFBs 4–7: DM5 → Neo1
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1s(DM5, NEO1));
    }
    // DFBs 8–11: DM4 → Neo2
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1s(DM4, NEO2));
    }
    // DFBs 12–15: DM5 → Neo3
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1s(DM5, NEO3));
    }

    // -----------------------------------------------------------------------
    // 1Sx2A DFBs (IDs 16–31): STRIDED producer, ALL consumer, remapper 1-to-2.
    // 1 prod TC on DM's tensix + 1 cons TC on each Neo consumer's tensix.
    // -----------------------------------------------------------------------
    auto make_1sx2a = [&](uint16_t producer_dm, uint16_t consumer_neos) {
        return experimental::dfb::DataflowBufferConfig{
            .entry_size          = ENTRY_SIZE,
            .num_entries         = NUM_ENTRIES,
            .producer_risc_mask  = producer_dm,
            .num_producers       = 1,
            .pap                 = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask  = consumer_neos,
            .num_consumers       = 2,
            .cap                 = dfb::AccessPattern::ALL,
            .enable_producer_implicit_sync = true,
            .enable_consumer_implicit_sync = true,
            .data_format         = kDfbBenchDataFormat,
        };
    };

    // DFBs 16–19: DM2 (t2) → {Neo0 (t0), Neo2 (t2)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM2, NEO0 | NEO2));
    }
    // DFBs 20–23: DM3 (t3) → {Neo0 (t0), Neo2 (t2)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM3, NEO0 | NEO2));
    }
    // DFBs 24–27: DM4 (t0) → {Neo1 (t1), Neo3 (t3)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM4, NEO1 | NEO3));
    }
    // DFBs 28–31: DM5 (t1) → {Neo1 (t1), Neo3 (t3)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM5, NEO1 | NEO3));
    }

    // -----------------------------------------------------------------------
    // Kernels: created with the low-level experimental Quasar API.
    // DFB risc masks are already set above; no BindDataflowBufferToProducerConsumerKernels call needed.
    // -----------------------------------------------------------------------
    constexpr const char* DM_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_worst3_dm.cpp";
    constexpr const char* COMPUTE_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_worst3_compute.cpp";

    // 6-thread DM kernel (DM0–DM5 launched; DM0/DM1 coordinators only).
    // Producers DM2–DM5; each checks mhartid for DFBs in producer_risc_mask.
    // DM4: DFBs 0-3, 8-11 (1Sx1S), 24-27 (1Sx2A). DM5: 4-7, 12-15 (1Sx1S), 28-31 (1Sx2A).
    experimental::quasar::CreateKernel(
        program,
        DM_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 6,
        });

    // 4-thread compute kernel: Neo0–Neo3 all run the same source; each checks
    // NEO_ID and operates only on DFBs for which its bit is set in consumer_risc_mask.
    experimental::quasar::CreateKernel(
        program,
        COMPUTE_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 4,
        });

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseSeven");
}

// BenchmarkWorstCaseFour — exhausts all 16 one-to-many remapper slots AND exercises
// 8 of the 48 one-to-one remapper slots simultaneously, for 24 total remapper entries.
//
// This is the first test to use both remapper slot types at once.
//
// DFB layout (24 DFBs total):
//   1Sx2A (DFBs  0–15): 16 × 1-to-2 remapper entries (all 16 one-to-many slots)
//     DFB  0– 3: DM2 → {Neo0, Neo2}
//     DFB  4– 7: DM3 → {Neo0, Neo2}
//     DFB  8–11: DM4 → {Neo1, Neo3}
//     DFB 12–15: DM5 → {Neo1, Neo3}
//
//   1Sx1A (DFBs 16–23): 8 × 1-to-1 remapper entries (from the 48 one-to-one slots)
//     DFB 16–17: DM2 → Neo1
//     DFB 18–19: DM3 → Neo1
//     DFB 20–21: DM2 → Neo3
//     DFB 22–23: DM4 → Neo0
//
// TC budget: 64/64 (exactly 16 per tensix):
//   t0: 4(DM4 prod 8-11) + 4(Neo0 cons DM2) + 4(Neo0 cons DM3) + 2(DM4 prod 22-23) + 2(Neo0 cons 22-23) = 16
//   t1: 4(DM5 prod 12-15)+ 4(Neo1 cons DM4) + 4(Neo1 cons DM5) + 2(Neo1 cons 16-17) + 2(Neo1 cons 18-19) = 16
//   t2: 4(DM2 prod 0-3)  + 4(Neo2 cons DM2) + 4(Neo2 cons DM3) + 2(DM2 prod 16-17)  + 2(DM2 prod 20-21)  = 16
//   t3: 4(DM3 prod 4-7)  + 4(Neo3 cons DM4) + 4(Neo3 cons DM5) + 2(DM3 prod 18-19)  + 2(Neo3 cons 20-21) = 16
//
// Remapper: 16 × 1-to-2 (all 16 one-to-many slots) + 8 × 1-to-1 (8 of 48 one-to-one slots)
//           = 24 total remapper entries, 16×2 + 8×1 = 40 set_clientR_slot writes.
// TxnId budget: 24 DFBs × 1 txn ID = 24 ≤ 32 (num_entries=1 required; 16 entries would
// need 2 txn IDs/DFB and exceed the 32-slot budget).
//
// Traffic per DFB (num_entries=1): DM issues 1 implicit read + finish; Neo drains via
// copy_tile. entry_size=2048 matches default 32×32 Float16_b JIT unpack (see Case Seven comment).
void run_benchmark_case_six(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE  = 2048;
    // num_entries=1: ensures each DFB consumes exactly 1 TxnId (24 DFBs × 1 = 24 ≤ 32 budget)
    // while keeping implicit sync enabled to exercise ISR setup.
    constexpr uint32_t NUM_ENTRIES = 1;

    // risc_mask bit positions: DM k → bit k (bits 0–7), Neo k → bit (8+k) (bits 8–11)
    constexpr uint16_t DM2  = (1u << 2);
    constexpr uint16_t DM3  = (1u << 3);
    constexpr uint16_t DM4  = (1u << 4);
    constexpr uint16_t DM5  = (1u << 5);
    constexpr uint16_t NEO0 = (1u << 8);
    constexpr uint16_t NEO1 = (1u << 9);
    constexpr uint16_t NEO2 = (1u << 10);
    constexpr uint16_t NEO3 = (1u << 11);

    Program program = CreateProgram();

    // -----------------------------------------------------------------------
    // 1Sx2A DFBs (IDs 0–15): STRIDED producer, ALL consumer — 16 × 1-to-2 remapper
    // entries, consuming all 16 one-to-many remapper slots.
    // Each DFB: 1 prod TC on DM's tensix + 1 cons TC on each of 2 Neo tensixes = 3 TCs.
    // -----------------------------------------------------------------------
    auto make_1sx2a = [&](uint16_t producer_dm, uint16_t consumer_neos) {
        return experimental::dfb::DataflowBufferConfig{
            .entry_size          = ENTRY_SIZE,
            .num_entries         = NUM_ENTRIES,
            .producer_risc_mask  = producer_dm,
            .num_producers       = 1,
            .pap                 = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask  = consumer_neos,
            .num_consumers       = 2,
            .cap                 = dfb::AccessPattern::ALL,
            .enable_producer_implicit_sync = true,
            .enable_consumer_implicit_sync = true,
            .data_format         = kDfbBenchDataFormat,
        };
    };

    // DFBs 0–3: DM2(t2) → {Neo0(t0), Neo2(t2)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM2, NEO0 | NEO2));
    }
    // DFBs 4–7: DM3(t3) → {Neo0(t0), Neo2(t2)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM3, NEO0 | NEO2));
    }
    // DFBs 8–11: DM4(t0) → {Neo1(t1), Neo3(t3)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM4, NEO1 | NEO3));
    }
    // DFBs 12–15: DM5(t1) → {Neo1(t1), Neo3(t3)}
    for (int i = 0; i < 4; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx2a(DM5, NEO1 | NEO3));
    }

    // -----------------------------------------------------------------------
    // 1Sx1A DFBs (IDs 16–23): STRIDED producer, ALL consumer (1 Neo) — 8 × 1-to-1
    // remapper entries, exercising 8 of the 48 one-to-one remapper slots.
    // Each DFB: 1 prod TC on DM's tensix + 1 cons TC on Neo's tensix = 2 TCs.
    // -----------------------------------------------------------------------
    auto make_1sx1a = [&](uint16_t producer_dm, uint16_t consumer_neo) {
        return experimental::dfb::DataflowBufferConfig{
            .entry_size          = ENTRY_SIZE,
            .num_entries         = NUM_ENTRIES,
            .producer_risc_mask  = producer_dm,
            .num_producers       = 1,
            .pap                 = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask  = consumer_neo,
            .num_consumers       = 1,
            .cap                 = dfb::AccessPattern::ALL,
            .enable_producer_implicit_sync = true,
            .enable_consumer_implicit_sync = true,
            .data_format         = kDfbBenchDataFormat,
        };
    };

    // DFBs 16–17: DM2(t2) → Neo1(t1)  [+2 prod on t2, +2 cons on t1]
    for (int i = 0; i < 2; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1a(DM2, NEO1));
    }
    // DFBs 18–19: DM3(t3) → Neo1(t1)  [+2 prod on t3, +2 cons on t1]
    for (int i = 0; i < 2; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1a(DM3, NEO1));
    }
    // DFBs 20–21: DM2(t2) → Neo3(t3)  [+2 prod on t2, +2 cons on t3]
    for (int i = 0; i < 2; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1a(DM2, NEO3));
    }
    // DFBs 22–23: DM4(t0) → Neo0(t0)  [+2 prod on t0, +2 cons on t0]
    for (int i = 0; i < 2; i++) {
        experimental::dfb::CreateDataflowBuffer(program, core_range_set, make_1sx1a(DM4, NEO0));
    }

    // -----------------------------------------------------------------------
    // Kernels: 6-thread DM kernel (DM0–DM5 launched; DM0/DM1 coordinators only;
    // producers DM2–DM5) and 4-thread compute kernel (Neo0–Neo3).
    // -----------------------------------------------------------------------
    constexpr const char* DM_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_worst4_dm.cpp";
    constexpr const char* COMPUTE_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_worst4_compute.cpp";

    experimental::quasar::CreateKernel(
        program,
        DM_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 6,
        });

    experimental::quasar::CreateKernel(
        program,
        COMPUTE_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 4,
        });

    log_info(tt::LogTest, "BenchmarkCaseSix: ENTRY_SIZE={} NUM_ENTRIES={}", ENTRY_SIZE, NUM_ENTRIES);

    // Host preflight: catch stale builds where blobs still carry entry_size=1024.
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);
    const CoreCoord bench_core(0, 0);
    const auto dfbs = program.impl().dataflow_buffers_on_core(bench_core);
    TT_FATAL(dfbs.size() == 24u, "Expected 24 DFBs, got {}", dfbs.size());
    for (const auto& dfb : dfbs) {
        TT_FATAL(dfb->config.entry_size == ENTRY_SIZE, "DFB {} entry_size mismatch", dfb->id);
    }
    std::vector<uint8_t> cfg_buf(256 * 1024, 0);
    const size_t cfg_bytes =
        experimental::dfb::detail::serialize_dfb_config_for_core(bench_core, dfbs, cfg_buf);
    TT_FATAL(cfg_bytes > 0u, "serialize_dfb_config_for_core returned 0 bytes");
    const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(cfg_buf.data());
    constexpr uint8_t kNeo0Hart = ::dfb::TENSIX_RISC_OFFSET;  // Neo0 unpack hart
    const auto* neo0_e0 = reinterpret_cast<const dfb_hart_init_entry_t*>(
        cfg_buf.data() + ghdr->hart_blob_offset[kNeo0Hart]);
    const auto* neo0_tc0 = reinterpret_cast<const dfb_blob_tc_pair_t*>(
        cfg_buf.data() + ghdr->hart_blob_offset[kNeo0Hart] + sizeof(dfb_hart_init_entry_t));
    log_info(
        tt::LogTest,
        "BenchmarkCaseSix preflight: Neo0 dfb{} entry_size={} tc0 base=0x{:x} limit=0x{:x} ring_tiles={}",
        neo0_e0->logical_dfb_id,
        neo0_e0->entry_size,
        neo0_tc0->base_addr,
        neo0_tc0->limit,
        neo0_tc0->limit - neo0_tc0->base_addr);
    TT_FATAL(neo0_e0->entry_size == ENTRY_SIZE, "Neo0 entry_size preflight mismatch");
    TT_FATAL(neo0_tc0->limit - neo0_tc0->base_addr == ENTRY_SIZE / 16u, "Neo0 ring_tiles preflight mismatch");

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseSix");
}

// Minimal 1Sx1A isolation variant of BenchmarkCaseSix for remapper debugging.
// Single DFB via CreateDataflowBuffer: DM4 (STRIDED producer) → Neo0 (ALL consumer),
// 1-to-1 remapper, num_entries=1, entry_size=2048, copy_tile drain.
// Producer uses explicit reserve_back/push_back (implicit_sync=false), matching
// DMTensixTest1xDFB1Sx1S. With implicit_sync=true the DM kernel hangs in finish()
// at handle_final_credits WTP1 waiting for ISR-posted producer TC credits.
// Pass/fail here isolates ALL+remapper consumer TC init from multi-DFB TC pressure.
void run_benchmark_case_six_debug(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE  = 2048;
    constexpr uint32_t NUM_ENTRIES = 1;

    constexpr uint16_t DM4  = (1u << 4);
    constexpr uint16_t NEO0 = (1u << 8);

    Program program = CreateProgram();

    experimental::dfb::CreateDataflowBuffer(
        program,
        core_range_set,
        experimental::dfb::DataflowBufferConfig{
            .entry_size          = ENTRY_SIZE,
            .num_entries         = NUM_ENTRIES,
            .producer_risc_mask  = DM4,
            .num_producers       = 1,
            .pap                 = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask  = NEO0,
            .num_consumers       = 1,
            .cap                 = dfb::AccessPattern::ALL,
            .enable_producer_implicit_sync = false,
            .enable_consumer_implicit_sync = false,
            .data_format         = kDfbBenchDataFormat,
        });

    constexpr const char* DM_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_case6_debug_dm.cpp";
    constexpr const char* COMPUTE_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_case6_debug_compute.cpp";

    experimental::quasar::CreateKernel(
        program,
        DM_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 6,
        });

    experimental::quasar::CreateKernel(
        program,
        COMPUTE_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 4,
        });

    log_info(
        tt::LogTest,
        "BenchmarkCaseSixDebug: 1×1Sx1A DFB (DM4→Neo0) ENTRY_SIZE={} NUM_ENTRIES={}",
        ENTRY_SIZE,
        NUM_ENTRIES);

    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);
    const CoreCoord bench_core(0, 0);
    const auto dfbs = program.impl().dataflow_buffers_on_core(bench_core);
    TT_FATAL(dfbs.size() == 1u, "Expected 1 DFB");
    TT_FATAL(dfbs[0]->config.entry_size == ENTRY_SIZE, "DFB entry_size mismatch");

    std::vector<uint8_t> cfg_buf(256 * 1024, 0);
    const size_t cfg_bytes =
        experimental::dfb::detail::serialize_dfb_config_for_core(bench_core, dfbs, cfg_buf);
    TT_FATAL(cfg_bytes > 0u, "serialize_dfb_config_for_core returned 0 bytes");
    const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(cfg_buf.data());
    constexpr uint8_t kNeo0Hart = ::dfb::TENSIX_RISC_OFFSET;
    const auto* neo0_e0 = reinterpret_cast<const dfb_hart_init_entry_t*>(
        cfg_buf.data() + ghdr->hart_blob_offset[kNeo0Hart]);
    const auto* neo0_tc0 = reinterpret_cast<const dfb_blob_tc_pair_t*>(
        cfg_buf.data() + ghdr->hart_blob_offset[kNeo0Hart] + sizeof(dfb_hart_init_entry_t));
    log_info(
        tt::LogTest,
        "BenchmarkCaseSixDebug preflight: Neo0 dfb{} entry_size={} tc0 base=0x{:x} limit=0x{:x} ring_tiles={}",
        neo0_e0->logical_dfb_id,
        neo0_e0->entry_size,
        neo0_tc0->base_addr,
        neo0_tc0->limit,
        neo0_tc0->limit - neo0_tc0->base_addr);
    TT_FATAL(neo0_e0->logical_dfb_id == 0u, "Neo0 logical_dfb_id preflight mismatch");
    TT_FATAL(neo0_e0->entry_size == ENTRY_SIZE, "Neo0 entry_size preflight mismatch");
    TT_FATAL(neo0_tc0->limit - neo0_tc0->base_addr == ENTRY_SIZE / 16u, "Neo0 ring_tiles preflight mismatch");

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseSixDebug");
}

// Same 1Sx1A topology as BenchmarkCaseSixDebug but with implicit_sync enabled.
// Uses a single-producer DM kernel that forces num_sw_threads=1 before finish()
// (see dfb_bench_case6_debug_implicit_dm.cpp). Without that, finish()'s internal
// sync_threads(get_num_threads()) waits for all 6 launched DM harts even though
// only DM4 issues the implicit read.
void run_benchmark_case_six_debug_implicit_sync(DfbInitTimingBenchContext& ctx) {
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE  = 2048;
    constexpr uint32_t NUM_ENTRIES = 1;

    constexpr uint16_t DM4  = (1u << 4);
    constexpr uint16_t NEO0 = (1u << 8);

    Program program = CreateProgram();

    experimental::dfb::CreateDataflowBuffer(
        program,
        core_range_set,
        experimental::dfb::DataflowBufferConfig{
            .entry_size          = ENTRY_SIZE,
            .num_entries         = NUM_ENTRIES,
            .producer_risc_mask  = DM4,
            .num_producers       = 1,
            .pap                 = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask  = NEO0,
            .num_consumers       = 1,
            .cap                 = dfb::AccessPattern::ALL,
            .enable_producer_implicit_sync = true,
            .enable_consumer_implicit_sync = true,
            .data_format         = kDfbBenchDataFormat,
        });

    constexpr const char* DM_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_bench_case6_debug_implicit_dm.cpp";
    constexpr const char* COMPUTE_KERNEL_SRC =
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_bench_case6_debug_compute.cpp";

    experimental::quasar::CreateKernel(
        program,
        DM_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 6,
        });

    experimental::quasar::CreateKernel(
        program,
        COMPUTE_KERNEL_SRC,
        core_range_set,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 4,
        });

    log_info(
        tt::LogTest,
        "BenchmarkCaseSixDebugImplicitSync: 1×1Sx1A DFB (DM4→Neo0) implicit_sync ENTRY_SIZE={} NUM_ENTRIES={}",
        ENTRY_SIZE,
        NUM_ENTRIES);

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseSixDebugImplicitSync");
}

void run_benchmark_case_six_debug_implicit_sync_program_spec(DfbInitTimingBenchContext& ctx) {
    auto mesh_device = ctx.mesh_device;
    IDevice* device = ctx.device;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    constexpr uint32_t ENTRY_SIZE = 2048;
    constexpr uint32_t NUM_ENTRIES = 1;
    constexpr uint8_t NUM_PRODUCERS = 1;
    constexpr uint8_t NUM_CONSUMERS = 1;

    const experimental::DFBSpecName DFB{"dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};
    const experimental::TensorParamName IN_TENSOR{"in_tensor"};

    const experimental::DataMovementHardwareConfig dm_producer_cfg = experimental::DataMovementGen2Config{};

    auto in_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(ENTRY_SIZE, NUM_ENTRIES), TensorTopology{});

    experimental::DataflowBufferSpec dfb_spec = MakeBenchDfbSpec(DFB, ENTRY_SIZE, NUM_ENTRIES);

    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        .num_threads = NUM_PRODUCERS,
        .dfb_bindings = {experimental::ProducerOf(DFB, "out")},
        .tensor_bindings = {{
            .tensor_parameter_name = IN_TENSOR,
            .accessor_name = "src_tensor",
        }},
        .compile_time_args =
            {
                {"num_entries_per_producer", NUM_ENTRIES},
                {"implicit_sync", 1u},
                {"num_producers", static_cast<uint32_t>(NUM_PRODUCERS)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}},
        .hw_config = dm_producer_cfg,
    };

    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp",
        .num_threads = NUM_CONSUMERS,
        .dfb_bindings = {experimental::AllConsumerOf(DFB, "in")},
        .compile_time_args = {{"num_entries_per_consumer", NUM_ENTRIES}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "bench_case6_program_spec_wu",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = core_range_set,
    };

    experimental::ProgramSpec spec{
        .name = "bench_case6_program_spec",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()}},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    experimental::ProgramRunArgs::KernelRunArgs producer_params{};
    producer_params.kernel = PRODUCER;
    producer_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        experimental::NodeCoord{0, 0},
        {{"chunk_offset", 0u}, {"entries_per_core", NUM_ENTRIES}});
    experimental::ProgramRunArgs::KernelRunArgs consumer_params{};
    consumer_params.kernel = CONSUMER;
    run_args.kernel_run_args = {producer_params, consumer_params};
    run_args.tensor_args.emplace(IN_TENSOR, experimental::TensorArgument{in_tensor});
    experimental::SetProgramRunArgs(program, run_args);

    LaunchAndLogDfbInitTiming(device, program, CoreCoord(0, 0), "BenchmarkCaseSixDebugImplicitSyncProgramSpec");
}

struct DfbInitTimingBenchCase {
    const char* cli_name;
    void (*run)(DfbInitTimingBenchContext&);
};

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [--case NAME]\n"
        << "  NAME: base, two, three, four, five, six, seven,\n"
        << "        six-debug, six-debug-implicit-sync, six-debug-implicit-sync-program-spec, all\n"
        << "\nRequires TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_MEASURE_DFB_INIT_TIME=1 on Quasar.\n";
}

}  // namespace tt::tt_metal

int main(int argc, char** argv) {
    using namespace tt::tt_metal;

    std::string case_name = "all";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--case" && i + 1 < argc) {
            case_name = argv[++i];
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 1;
    }

    static const DfbInitTimingBenchCase kCases[] = {
        {"base", run_benchmark_case_base},
        {"two", run_benchmark_case_two},
        {"three", run_benchmark_case_three},
        {"four", run_benchmark_case_four},
        {"five", run_benchmark_case_five},
        {"six", run_benchmark_case_six},
        {"seven", run_benchmark_case_seven},
        // {"six-debug", run_benchmark_case_six_debug},
        // {"six-debug-implicit-sync", run_benchmark_case_six_debug_implicit_sync},
        // {"six-debug-implicit-sync-program-spec", run_benchmark_case_six_debug_implicit_sync_program_spec},
    };

    std::unordered_set<std::string> selected;
    if (case_name == "all") {
        for (const auto& bench_case : kCases) {
            selected.insert(bench_case.cli_name);
        }
    } else {
        selected.insert(case_name);
    }

    // Reuse one MeshDevice across cases so we can catch residual DFB HW state.
    // Prefer fixing that over open/close (which masks the bug). Enable
    // TT_METAL_DPRINT_CORES / watcher dprint to compare cold vs warm case-two init.
    auto ctx = create_dfb_init_timing_bench_context();
    bool ran_any = false;
    for (const auto& bench_case : kCases) {
        if (!selected.contains(bench_case.cli_name)) {
            continue;
        }
        log_info(tt::LogTest, "Running DFB init timing benchmark case: {}", bench_case.cli_name);
        bench_case.run(ctx);
        ran_any = true;
    }

    if (!ran_any) {
        std::cerr << "Unknown benchmark case: " << case_name << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
