// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-side DataflowBuffer API tests:
// - read_tile_value / get_tile_address (DM → Tensix, 1Sx1S, WH/BH; Quasar skipped)
// - Static DFB extent getters (no DRAM); WH/BH 1Sx1S; Quasar multi-TC layout probes

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include "dfb_test_common.hpp"
#include "device_fixture.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "umd/device/driver_atomics.hpp"

namespace tt::tt_metal {

namespace m2 = experimental;

TEST_F(MeshDeviceFixture, DataflowBufferReadTileValue) {
    using DataT = std::uint32_t;

    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    if (device->arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar read_tile_value / get_tile_address on DFB is under debug; run on WH/BH";
    }
    if (MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "Skipping DataflowBufferReadTileValue for tt-sim until GH#50135 is resolved";
    }

    constexpr uint32_t num_producers = 1;
    constexpr uint32_t num_consumers = 1;
    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 2;

    constexpr uint32_t num_results_per_thread = 7;
    constexpr uint32_t num_trisc_threads = 3;

    // Tile 0 / tile 1 scalars at element offsets 0 and 1 within each entry.
    constexpr DataT tile0_val0 = 0xA5A5A5A5u;
    constexpr DataT tile0_val1 = 0x11111111u;
    // Distinct low/high halfwords so uint16 reads can distinguish T-indexing from uint32-indexing + truncate.
    constexpr DataT tile1_val0 = 0xABCD1234u;
    constexpr DataT tile1_val1 = 0x33333333u;
    constexpr uint16_t tile1_val0_lo = 0x1234u;
    constexpr uint16_t tile1_val0_hi = 0xABCDu;
    // Both entries stay at the front; tile_index 0/1 address fifo_rd_ptr + {0, fifo_page_size}.
    // Per thread: {tile0[0], tile0[1], tile1[0], tile1[1], get_tile_address(1)[0],
    //             read_tile_value<uint16_t>(1)[0], read_tile_value<uint16_t>(1)[1]}
    const std::vector<DataT> expected_per_thread = {
        tile0_val0,
        tile0_val1,
        tile1_val0,
        tile1_val1,
        tile1_val0,
        tile1_val0_lo,
        tile1_val0_hi};
    // UNPACK, MATH, and PACK each write expected_per_thread to a distinct L1 slot.
    std::vector<DataT> expected_scalar_reads;
    expected_scalar_reads.reserve(num_results_per_thread * num_trisc_threads);
    for (uint32_t thread = 0; thread < num_trisc_threads; ++thread) {
        expected_scalar_reads.insert(
            expected_scalar_reads.end(), expected_per_thread.begin(), expected_per_thread.end());
    }

    const m2::NodeCoord node{0, 0};
    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};

    const uint32_t words_per_entry = entry_size / sizeof(DataT);

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = DataFormat::Float16_b,
    };

    m2::KernelSpec producer{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        .num_threads = num_producers,
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .accessor_name = "out",
              .endpoint_type = m2::DFBEndpointType::PRODUCER,
              .access_pattern = m2::DFBAccessPattern::STRIDED}},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .compile_time_args =
            {
                {"num_entries_per_producer", num_entries},
                {"implicit_sync", 0u},
                {"num_producers", num_producers},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}},
        .hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0},
    };

    m2::KernelSpec consumer{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_read_tile_value_compute.cpp",
        .num_threads = num_consumers,
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .accessor_name = "in",
              .endpoint_type = m2::DFBEndpointType::CONSUMER,
              .access_pattern = m2::DFBAccessPattern::STRIDED}},
        .compile_time_args = {{"num_entries_per_consumer", num_entries}},
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config = m2::ComputeGen1Config{},
    };

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "dfb_read_tile_value_2tiles",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()}},
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    const uint32_t result_size_bytes = static_cast<uint32_t>(expected_scalar_reads.size() * sizeof(DataT));
    const uint32_t l1_alignment = device->allocator()->get_alignment(BufferType::L1);
    const uint32_t aligned_result_size = (result_size_bytes + l1_alignment - 1) / l1_alignment * l1_alignment;
    const uint32_t result_l1_addr = static_cast<uint32_t>(device->l1_size_per_core()) - aligned_result_size;

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_l1_addr}}),
        },
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}};
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_words = num_entries * words_per_entry;
    auto input = tt::test_utils::generate_uniform_random_vector<DataT>(0, 1000000, total_words);
    input[0] = tile0_val0;
    input[1] = tile0_val1;
    input[words_per_entry + 0] = tile1_val0;
    input[words_per_entry + 1] = tile1_val1;
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);

    std::vector<DataT> result_init(expected_scalar_reads.size(), 0u);
    detail::WriteToDeviceL1(device, CoreCoord(0, 0), result_l1_addr, result_init);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    tt_driver_atomics::mfence();
    std::vector<DataT> scalar_results;
    detail::ReadFromDeviceL1(device, CoreCoord(0, 0), result_l1_addr, result_size_bytes, scalar_results);
    ASSERT_EQ(scalar_results.size(), expected_scalar_reads.size());
    for (uint32_t thread = 0; thread < num_trisc_threads; ++thread) {
        const auto begin = scalar_results.begin() + thread * num_results_per_thread;
        EXPECT_EQ(
            std::vector<DataT>(begin, begin + num_results_per_thread),
            expected_per_thread)
            << "TRISC thread slot " << thread;
    }
}

namespace {

using ExtentRecord = std::array<uint32_t, 8>;

enum ExtentField : uint32_t {
    EntrySize = 0,
    StrideSize,
    TotalNumEntries,
    TotalSizeBytes,
    LocalNumEntries,
    LocalSizeBytes,
    RingSpanBytes,
    RingSpanNumEntries,
};

inline uint32_t strided_num_tcs(bool is_producer, uint32_t num_producers, uint32_t num_consumers) {
    if (is_producer) {
        return num_consumers >= num_producers ? num_consumers / num_producers : 1u;
    }
    return num_producers >= num_consumers ? num_producers / num_consumers : 1u;
}

struct ExtentLayoutParams {
    uint32_t capacity = 0;
    uint32_t stride_in_entries = 0;
    uint32_t ring_bytes = 0;
    uint32_t num_tcs_on_risc = 0;
};

inline ExtentLayoutParams compute_strided_layout(
    uint32_t num_entries, uint32_t entry_size, uint32_t num_producers, uint32_t num_consumers, bool is_producer) {
    const uint32_t max_pc = std::max(num_producers, num_consumers);
    return ExtentLayoutParams{
        .capacity = num_entries / max_pc,
        .stride_in_entries = max_pc,
        .ring_bytes = entry_size * (max_pc * (num_entries / max_pc - 1u) + 1u),
        .num_tcs_on_risc = strided_num_tcs(is_producer, num_producers, num_consumers),
    };
}

inline uint32_t expected_strided_ring_span_bytes(
    uint32_t entry_size, uint32_t local_ring_bytes, uint32_t num_tcs_on_risc, uint32_t num_endpoint_threads) {
    if (num_tcs_on_risc <= 1) {
        return local_ring_bytes;
    }
    // Host assigns bases per tc slot across all same-role RISCs on the node (base += entry_size
    // per RISC). TC[t] on RISC p is at alloc + (t * num_endpoint_threads + p) * entry_size, so the
    // bounding span on one RISC is local_ring + (num_tcs - 1) * num_endpoint_threads * entry_size.
    return local_ring_bytes + (num_tcs_on_risc - 1u) * num_endpoint_threads * entry_size;
}

inline ExtentRecord expected_strided_extent(
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_producers,
    uint32_t num_consumers,
    bool is_producer) {
    const auto layout = compute_strided_layout(num_entries, entry_size, num_producers, num_consumers, is_producer);
    const uint32_t total_bytes = num_entries * entry_size;
    const uint32_t num_endpoint_threads = is_producer ? num_producers : num_consumers;
    const uint32_t ring_span_bytes = expected_strided_ring_span_bytes(
        entry_size, layout.ring_bytes, layout.num_tcs_on_risc, num_endpoint_threads);
    return ExtentRecord{
        entry_size,
        entry_size * layout.stride_in_entries,
        num_entries,
        total_bytes,
        layout.capacity,
        layout.ring_bytes,
        ring_span_bytes,
        ring_span_bytes / entry_size,
    };
}

// ALL consumer (cap=ALL): capacity/stride follow num_producers; TC counts match DMTensix ALL.
inline ExtentRecord expected_all_extent(
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_producers,
    uint32_t num_consumers,
    bool is_producer) {
    (void)num_consumers;
    const uint32_t capacity = num_entries / num_producers;
    const uint32_t stride_in_entries = 1;
    const uint32_t total_bytes = num_entries * entry_size;
    const uint32_t ring_bytes = entry_size * (stride_in_entries * (capacity - 1u) + 1u);
    const uint32_t num_tcs_on_risc = is_producer ? 1u : num_producers;
    const uint32_t ring_span_bytes = num_tcs_on_risc <= 1 ? ring_bytes : total_bytes;
    return ExtentRecord{
        entry_size,
        entry_size * stride_in_entries,
        num_entries,
        total_bytes,
        capacity,
        ring_bytes,
        ring_span_bytes,
        ring_span_bytes / entry_size,
    };
}

inline ExtentRecord expected_extent(
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_producers,
    uint32_t num_consumers,
    bool is_producer,
    m2::DFBAccessPattern consumer_access_pattern) {
    if (consumer_access_pattern == m2::DFBAccessPattern::ALL) {
        return expected_all_extent(entry_size, num_entries, num_producers, num_consumers, is_producer);
    }
    return expected_strided_extent(entry_size, num_entries, num_producers, num_consumers, is_producer);
}

inline void expect_extent_record(const ExtentRecord& actual, const ExtentRecord& expected) {
    EXPECT_EQ(actual[EntrySize], expected[EntrySize]) << "entry_size";
    EXPECT_EQ(actual[StrideSize], expected[StrideSize]) << "stride_size";
    EXPECT_EQ(actual[TotalNumEntries], expected[TotalNumEntries]) << "total_num_entries";
    EXPECT_EQ(actual[TotalSizeBytes], expected[TotalSizeBytes]) << "total_size_bytes";
    EXPECT_EQ(actual[LocalNumEntries], expected[LocalNumEntries]) << "local_num_entries";
    EXPECT_EQ(actual[LocalSizeBytes], expected[LocalSizeBytes]) << "local_size_bytes";
    EXPECT_EQ(actual[RingSpanBytes], expected[RingSpanBytes]) << "ring_span_bytes";
    EXPECT_EQ(actual[RingSpanNumEntries], expected[RingSpanNumEntries]) << "ring_span_num_entries";
}

inline void expect_wh_bh_aliases(const ExtentRecord& rec) {
    EXPECT_EQ(rec[LocalNumEntries], rec[TotalNumEntries]);
    EXPECT_EQ(rec[LocalSizeBytes], rec[TotalSizeBytes]);
    EXPECT_EQ(rec[RingSpanBytes], rec[TotalSizeBytes]);
    EXPECT_EQ(rec[RingSpanNumEntries], rec[TotalNumEntries]);
    EXPECT_EQ(rec[StrideSize], rec[EntrySize]);
}

inline uint32_t allocate_l1_result_region(IDevice* device, uint32_t bytes) {
    const uint32_t alignment = device->allocator()->get_alignment(BufferType::L1);
    const uint32_t aligned = (bytes + alignment - 1u) / alignment * alignment;
    return static_cast<uint32_t>(device->l1_size_per_core()) - aligned;
}

inline ExtentRecord read_extent_record(IDevice* device, CoreCoord core, uint32_t l1_addr) {
    constexpr uint32_t num_fields = 8;
    constexpr uint32_t record_bytes = num_fields * sizeof(uint32_t);
    std::vector<uint32_t> words(num_fields, 0u);
    detail::ReadFromDeviceL1(device, core, l1_addr, record_bytes, words);
    ExtentRecord rec{};
    for (uint32_t i = 0; i < num_fields; ++i) {
        rec[i] = words[i];
    }
    return rec;
}

struct ProducerProbeConfig {
    uint32_t num_tc_snapshots = 1;
    bool rotate_tc = false;
    uint32_t credits_to_post = 0;
};

struct ConsumerProbeConfig {
    uint32_t num_tc_snapshots = 1;
    bool rotate_tc = false;
    bool drain_producer_rotate_credits = false;
    bool drain_last_tc_credit = false;
};

struct ExtentProbeParams {
    uint32_t num_producers = 1;
    uint32_t num_consumers = 1;
    uint32_t entry_size = 1024;
    uint32_t num_entries = 16;
    m2::DFBAccessPattern consumer_access_pattern = m2::DFBAccessPattern::STRIDED;
    ProducerProbeConfig producer{};
    ConsumerProbeConfig consumer{};
};

void run_extent_probe(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ExtentProbeParams& params) {
    constexpr uint32_t extent_record_bytes = 8 * sizeof(uint32_t);

    IDevice* device = mesh_device->get_devices()[0];
    const bool is_quasar = device->arch() == ARCH::QUASAR;

    if (!is_quasar && (params.num_producers > 1 || params.num_consumers > 1)) {
        GTEST_SKIP() << "WH/BH supports 1Sx1S only";
    }

    const m2::NodeCoord node{0, 0};
    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = params.entry_size,
        .num_entries = params.num_entries,
        .data_format_metadata = DataFormat::Float16_b,
    };

    m2::DataMovementHardwareConfig producer_hw;
    m2::ComputeHardwareConfig consumer_hw;
    if (is_quasar) {
        producer_hw = m2::DataMovementGen2Config{.disable_dfb_implicit_sync_for = {DFB}};
        consumer_hw = m2::ComputeGen2Config{};
    } else {
        producer_hw = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0};
        consumer_hw = m2::ComputeGen1Config{};
    }

    m2::KernelSpec producer{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_extent_probe_dm.cpp",
        .num_threads = static_cast<uint8_t>(params.num_producers),
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .accessor_name = "out",
              .endpoint_type = m2::DFBEndpointType::PRODUCER,
              .access_pattern = m2::DFBAccessPattern::STRIDED}},
        .compile_time_args =
            {
                {"num_tc_snapshots", params.producer.num_tc_snapshots},
                {"rotate_tc", params.producer.rotate_tc ? 1u : 0u},
                {"credits_to_post", params.producer.credits_to_post},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config = producer_hw,
    };

    m2::KernelSpec consumer{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_extent_probe_compute.cpp",
        .num_threads = static_cast<uint8_t>(params.num_consumers),
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .accessor_name = "in",
              .endpoint_type = m2::DFBEndpointType::CONSUMER,
              .access_pattern = params.consumer_access_pattern}},
        .compile_time_args =
            {
                {"num_tc_snapshots", params.consumer.num_tc_snapshots},
                {"rotate_tc", params.consumer.rotate_tc ? 1u : 0u},
                {"drain_producer_rotate_credits", params.consumer.drain_producer_rotate_credits ? 1u : 0u},
                {"drain_last_tc_credit", params.consumer.drain_last_tc_credit ? 1u : 0u},
                {"num_producers", params.num_producers},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config = consumer_hw,
    };

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "dfb_extent_probe",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_spec},
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    const uint32_t producer_records =
        params.num_producers * params.producer.num_tc_snapshots;
    const uint32_t consumer_records = params.consumer.num_tc_snapshots;
    const uint32_t producer_result_l1 =
        allocate_l1_result_region(device, (producer_records + consumer_records) * extent_record_bytes);
    const uint32_t consumer_result_l1 = producer_result_l1 + producer_records * extent_record_bytes;

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", producer_result_l1}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", consumer_result_l1}}),
        },
    };
    m2::SetProgramRunArgs(program, run_args);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    tt_driver_atomics::mfence();

    const CoreCoord core{0, 0};
    const ExtentRecord expected_producer = expected_extent(
        params.entry_size,
        params.num_entries,
        params.num_producers,
        params.num_consumers,
        true,
        params.consumer_access_pattern);
    const ExtentRecord expected_consumer = expected_extent(
        params.entry_size,
        params.num_entries,
        params.num_producers,
        params.num_consumers,
        false,
        params.consumer_access_pattern);

    for (uint32_t t = 0; t < params.num_producers; ++t) {
        for (uint32_t s = 0; s < params.producer.num_tc_snapshots; ++s) {
            const uint32_t l1_addr =
                producer_result_l1 + (t * params.producer.num_tc_snapshots + s) * extent_record_bytes;
            const ExtentRecord rec = read_extent_record(device, core, l1_addr);
            if (is_quasar) {
                expect_extent_record(rec, expected_producer);
            } else {
                expect_wh_bh_aliases(rec);
                EXPECT_EQ(rec[EntrySize], params.entry_size);
                EXPECT_EQ(rec[TotalNumEntries], params.num_entries);
                EXPECT_EQ(rec[TotalSizeBytes], params.entry_size * params.num_entries);
            }
        }
    }

    for (uint32_t s = 0; s < params.consumer.num_tc_snapshots; ++s) {
        const ExtentRecord rec = read_extent_record(device, core, consumer_result_l1 + s * extent_record_bytes);
        if (is_quasar) {
            expect_extent_record(rec, expected_consumer);
        } else {
            expect_wh_bh_aliases(rec);
            EXPECT_EQ(rec[EntrySize], params.entry_size);
            EXPECT_EQ(rec[TotalNumEntries], params.num_entries);
            EXPECT_EQ(rec[TotalSizeBytes], params.entry_size * params.num_entries);
        }
    }

    if (is_quasar && params.producer.num_tc_snapshots == 1 && params.num_producers > params.num_consumers) {
        const ExtentRecord consumer_rec = read_extent_record(device, core, consumer_result_l1);
        EXPECT_LT(consumer_rec[LocalSizeBytes], consumer_rec[RingSpanBytes])
            << "multi-TC consumer RISC: local ring < bounding span";
    }
    if (is_quasar && params.producer.num_tc_snapshots == 1 && params.num_consumers > params.num_producers &&
        params.consumer_access_pattern == m2::DFBAccessPattern::STRIDED) {
        const ExtentRecord producer_rec = read_extent_record(device, core, producer_result_l1);
        EXPECT_LT(producer_rec[LocalSizeBytes], producer_rec[RingSpanBytes])
            << "multi-TC producer RISC: local ring < bounding span";
    }
    if (is_quasar && params.producer.num_tc_snapshots == 1 &&
        params.consumer_access_pattern == m2::DFBAccessPattern::ALL && params.num_producers > 1) {
        const ExtentRecord consumer_rec = read_extent_record(device, core, consumer_result_l1);
        EXPECT_LT(consumer_rec[LocalSizeBytes], consumer_rec[RingSpanBytes])
            << "ALL consumer multi-TC RISC: local ring < bounding span";
    }
}

}  // namespace

TEST_F(MeshDeviceFixture, DataflowBufferExtentApis_1Sx1S) {
    run_extent_probe(devices_.at(0), {});
}

TEST_F(MeshDeviceFixture, DataflowBufferExtentApis_4Sx1S) {
    run_extent_probe(
        devices_.at(0),
        {
            .num_producers = 4,
            .num_consumers = 1,
        });
}

// 2Sx4S: each producer RISC round-robins 2 TCs (push advances tc_idx between snapshots).
TEST_F(MeshDeviceFixture, DataflowBufferExtentApis_2Sx4S_ProducerEachTC) {
    run_extent_probe(
        devices_.at(0),
        {
            .num_producers = 2,
            .num_consumers = 4,
            .producer =
                {
                    .num_tc_snapshots = 2,
                    .rotate_tc = true,
                },
            .consumer =
                {
                    .num_tc_snapshots = 1,
                    .drain_producer_rotate_credits = true,
                },
        });
}

// 2Sx4A: each consumer RISC round-robins 2 TCs (pop advances tc_idx between snapshots).
TEST_F(MeshDeviceFixture, DataflowBufferExtentApis_2Sx4A_ConsumerEachTC) {
    run_extent_probe(
        devices_.at(0),
        {
            .num_producers = 2,
            .num_consumers = 4,
            .consumer_access_pattern = m2::DFBAccessPattern::ALL,
            .producer =
                {
                    .credits_to_post = 1,
                },
            .consumer =
                {
                    .num_tc_snapshots = 2,
                    .rotate_tc = true,
                    .drain_last_tc_credit = true,
                },
        });
}

}  // namespace tt::tt_metal
