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
// Must match kValue in quasar_mailbox_minimal_compute.cpp -- the kernel and host are separate TUs,
// so nothing enforces this at compile time.
constexpr std::uint32_t MAILBOX_MIN_EXPECTED_VALUE = 0xfacefaceu;

using CbApiDataT = std::uint32_t;
constexpr auto CB_API_DATA_FORMAT = DataFormat::UInt32;

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

}  // namespace

// Minimal reproducer for the QuasarCbL1ReadApi fault: a single mailbox_write (UNPACK ->
// MathThreadId) matched by a single mailbox_read (MATH <- UnpackThreadId). No CB/DFB is
// involved, isolating whether the mailbox mechanism itself faults independent of the
// dataflow-buffer address computation in QuasarCbL1ReadApi.
TEST_F(LLKQuasarUnitMeshFixture, QuasarMailboxMinimal) {
    auto& device = this->device();
    auto& cq = device.mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto compute_kernel = experimental::quasar::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_mailbox_minimal_compute.cpp",
        WORKER_CORE,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
        });

    // Kernel (MATH thread) writes the mailbox-read value here; host reads it back after the run.
    // Round up to the L1 allocation alignment, matching the CB API test below.
    const std::uint32_t l1_alignment = device.allocator()->get_alignment(BufferType::L1);
    const std::uint32_t aligned_result_size = (sizeof(std::uint32_t) + l1_alignment - 1) / l1_alignment * l1_alignment;
    const std::uint32_t result_l1_addr = static_cast<std::uint32_t>(device.l1_size_per_core()) - aligned_result_size;
    std::vector<std::uint32_t> result_init(1, 0);
    this->WriteToL1(WORKER_CORE, result_l1_addr, result_init);

    SetRuntimeArgs(program_, compute_kernel, WORKER_CORE, {result_l1_addr});

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::vector<std::uint32_t> host_buffer;
    this->ReadFromL1(WORKER_CORE, result_l1_addr, sizeof(std::uint32_t), host_buffer);

    ASSERT_EQ(host_buffer.size(), 1u);
    EXPECT_EQ(host_buffer[0], MAILBOX_MIN_EXPECTED_VALUE);
}

// Validates ckernel::read_tile_value and ckernel::get_tile_address on Quasar (cb_api.h). Both are
// implemented as an UNPACK -> mailbox -> MATH/PACK broadcast, so this is the same handshake as
// QuasarMailboxMinimal above, exercised through the real compute API instead of raw mailbox calls.
//
// The host preloads two known tiles into the DFB's L1 ring; all three compute threads read them
// back through both APIs and each records what it observed into its own slice of the result
// buffer. Checking all three slices (rather than only UNPACK's) is what makes this cover the
// mailbox delivery -- verifying UNPACK alone would pass even if MATH/PACK received nothing.
TEST_F(LLKQuasarUnitMeshFixture, QuasarCbL1ReadApi) {
    auto& device = this->device();

    const std::uint32_t tile_page_size = tt::tile_size(CB_API_DATA_FORMAT);

    Program program = CreateProgram();

    const std::uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(
        program,
        WORKER_CORE,
        experimental::dfb::DataflowBufferConfig{
            .entry_size = tile_page_size,
            .num_entries = 2,
            .data_format = CB_API_DATA_FORMAT,
            .tensix_scope = experimental::dfb::TensixScope::INTRA,
        });

    auto compute_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_cb_l1_read_api_compute.cpp",
        WORKER_CORE,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {dfb_id},
        });

    // Bind the compute kernel as both producer and consumer so the DFB config (base_addr,
    // entry_size, ...) is finalized and written to L1; the kernel only reads from it.
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, dfb_id, compute_kernel, compute_kernel);

    // Preload two known tiles into the DFB's L1 ring (single DFB -> ring base is the L1
    // allocator base). Tiles are entry_size apart for this 1-producer/1-consumer layout.
    const std::uint32_t dfb_l1_addr =
        static_cast<std::uint32_t>(device.allocator()->get_base_allocator_addr(HalMemType::L1));
    const std::uint32_t words_per_entry = tile_page_size / sizeof(CbApiDataT);
    std::vector<CbApiDataT> ring(2 * words_per_entry, 0);
    ring[0] = CB_API_VAL0;
    ring[1] = CB_API_VAL1;
    ring[words_per_entry + 0] = CB_API_VAL2;
    ring[words_per_entry + 1] = CB_API_VAL3;
    this->WriteToL1(WORKER_CORE, dfb_l1_addr, ring);

    std::vector<CbApiDataT> expected_result;
    for (std::uint32_t thread = 0; thread < CB_API_NUM_READER_THREADS; ++thread) {
        expected_result.insert(
            expected_result.end(), CB_API_EXPECTED_PER_THREAD.begin(), CB_API_EXPECTED_PER_THREAD.end());
    }

    // Each thread writes its reads here; host reads this spot back after the run.
    const std::uint32_t result_size_bytes = expected_result.size() * sizeof(CbApiDataT);
    const std::uint32_t l1_alignment = device.allocator()->get_alignment(BufferType::L1);
    const std::uint32_t aligned_result_size = (result_size_bytes + l1_alignment - 1) / l1_alignment * l1_alignment;
    const std::uint32_t result_l1_addr = static_cast<std::uint32_t>(device.l1_size_per_core()) - aligned_result_size;

    std::vector<CbApiDataT> result_init(expected_result.size(), 0);
    this->WriteToL1(WORKER_CORE, result_l1_addr, result_init);

    SetRuntimeArgs(program, compute_kernel, WORKER_CORE, {result_l1_addr});

    this->RunProgram(std::move(program));

    std::vector<CbApiDataT> host_buffer;
    this->ReadFromL1(WORKER_CORE, result_l1_addr, result_size_bytes, host_buffer);

    EXPECT_EQ(host_buffer, expected_result);
}

}  // namespace tt::tt_metal
