// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "host_api.hpp"
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/host_api/temp_quasar_api.hpp"
#include "llk_device_fixture.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal {

namespace {

constexpr CoreCoord WORKER_CORE = {0, 0};

using DataT = std::uint32_t;
constexpr auto DATA_FORMAT = DataFormat::UInt32;

constexpr DataT VAL0 = 0xA5A5A5A5u;
constexpr DataT VAL1 = 0x11111111u;
constexpr DataT VAL2 = 0x22222222u;
constexpr DataT VAL3 = 0x33333333u;
// {read tile0[0], tile0[1], tile1[0], tile1[1], get_tile_address(1)[0]}
const std::vector<DataT> EXPECTED_RESULT = {VAL0, VAL1, VAL2, VAL3, VAL2};

}  // namespace

// Validates ckernel::read_tile_value and ckernel::get_tile_address on Quasar (cb_api.h).
// The host preloads two known tiles into the DFB's L1 ring, then the compute kernel reads
// them back through both APIs. Reading tile_index 1 exercises the per-tile stride term.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarCbL1ReadApi) {
    auto mesh_device = devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const uint32_t tile_page_size = tt::tile_size(DATA_FORMAT);

    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    const uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(
        program_,
        WORKER_CORE,
        experimental::dfb::DataflowBufferConfig{
            .entry_size = tile_page_size,
            .num_entries = 2,
            .data_format = DATA_FORMAT,
            .tensix_scope = experimental::dfb::TensixScope::INTRA,
        });

    auto compute_kernel = experimental::quasar::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_cb_l1_read_api_compute.cpp",
        WORKER_CORE,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {dfb_id},
        });

    // Bind the compute kernel as both producer and consumer so the DFB config (base_addr,
    // entry_size, ...) is finalized and written to L1; the kernel only reads from it.
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program_, dfb_id, compute_kernel, compute_kernel);

    // Preload two known tiles into the DFB's L1 ring (single DFB → ring base is the L1
    // allocator base). Tiles are entry_size apart for this 1-producer/1-consumer layout.
    const uint32_t dfb_l1_addr = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t words_per_entry = tile_page_size / sizeof(DataT);
    std::vector<DataT> ring(2 * words_per_entry, 0);
    ring[0] = VAL0;
    ring[1] = VAL1;
    ring[words_per_entry + 0] = VAL2;
    ring[words_per_entry + 1] = VAL3;
    detail::WriteToDeviceL1(device, WORKER_CORE, dfb_l1_addr, ring);

    // Kernel writes its reads here; host reads this spot back after the run.
    const uint32_t result_size_bytes = EXPECTED_RESULT.size() * sizeof(DataT);
    const uint32_t l1_alignment = device->allocator()->get_alignment(BufferType::L1);
    const uint32_t aligned_result_size = (result_size_bytes + l1_alignment - 1) / l1_alignment * l1_alignment;
    const uint32_t result_l1_addr = static_cast<uint32_t>(device->l1_size_per_core()) - aligned_result_size;

    std::vector<DataT> result_init(EXPECTED_RESULT.size(), 0);
    detail::WriteToDeviceL1(device, WORKER_CORE, result_l1_addr, result_init);

    SetRuntimeArgs(program_, compute_kernel, WORKER_CORE, {result_l1_addr});

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::vector<DataT> host_buffer;
    detail::ReadFromDeviceL1(device, WORKER_CORE, result_l1_addr, result_size_bytes, host_buffer);

    EXPECT_EQ(host_buffer, EXPECTED_RESULT);
}

}  // namespace tt::tt_metal
