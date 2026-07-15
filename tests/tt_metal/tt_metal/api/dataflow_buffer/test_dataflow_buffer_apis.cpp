// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-side DataflowBuffer API tests (read_tile_value / get_tile_address).
// DM → Tensix, 1 producer × 1 consumer, 2-entry DFB, explicit sync (WH/BH).
//
// Quasar is skipped until read_tile_value / get_tile_address on 2xx DFB is debugged.

#include <cstdint>
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

}  // namespace tt::tt_metal
