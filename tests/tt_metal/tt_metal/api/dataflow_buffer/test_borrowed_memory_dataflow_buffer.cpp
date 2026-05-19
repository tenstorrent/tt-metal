// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

#include "device_fixture.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

namespace {

static constexpr const char* PRODUCER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/borrowed_dfb_producer.cpp";
static constexpr const char* DM_CONSUMER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/borrowed_dfb_consumer.cpp";
static constexpr const char* TENSIX_CONSUMER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/compute/borrowed_dfb_tensix_consumer.cpp";

struct BorrowedDFBTestConfig {
    uint32_t num_entries        = 16;
    uint32_t entry_size         = 256;  // bytes
    uint32_t num_producers      = 1;
    uint32_t num_consumers      = 1;
    dfb::AccessPattern pap      = dfb::AccessPattern::STRIDED;
    dfb::AccessPattern cap      = dfb::AccessPattern::STRIDED;
    bool enable_implicit_sync   = false;
    bool tensix_consumer        = false;
    // When true, verify that output data == input data
    bool verify_data            = false;
};

// Fills a vector with 0,1,2,...,n-1 (uint32_t).
static std::vector<uint32_t> make_sequential(uint32_t n_words) {
    std::vector<uint32_t> v(n_words);
    std::iota(v.begin(), v.end(), 0u);
    return v;
}

// Runs a single borrowed-memory DFB program on one core and asserts:
//   1. The DFB was allocated at the borrowed L1 buffer address.
//   2. (Optionally) output data matches input data.
void run_borrowed_memory_dfb_program(
    IDevice* device,
    const CoreCoord& core,
    const BorrowedDFBTestConfig& test_cfg) {

    const bool is_all               = (test_cfg.cap == dfb::AccessPattern::ALL);
    const uint32_t dfb_ring_size    = test_cfg.num_entries * test_cfg.entry_size;

    // Per-RISC entry counts passed as compile-time args.
    const uint32_t entries_per_producer =
        (test_cfg.num_entries + test_cfg.num_producers - 1) / test_cfg.num_producers;
    const uint32_t entries_per_consumer =
        is_all ? test_cfg.num_entries
               : (test_cfg.num_entries + test_cfg.num_consumers - 1) / test_cfg.num_consumers;

    // Total bytes of input/output regions (rounded up per-RISC).
    const uint32_t input_size  = test_cfg.num_producers * entries_per_producer * test_cfg.entry_size;
    const uint32_t output_size = test_cfg.num_consumers * entries_per_consumer * test_cfg.entry_size;

    // -----------------------------------------------------------------------
    // 1. Allocate L1 buffers: DFB ring (borrowed), input, output.
    // -----------------------------------------------------------------------
    auto dfb_ring_buf = CreateBuffer(InterleavedBufferConfig{
        .device    = device,
        .size      = dfb_ring_size,
        .page_size = dfb_ring_size,
        .buffer_type = BufferType::L1});

    auto input_buf = CreateBuffer(InterleavedBufferConfig{
        .device    = device,
        .size      = input_size,
        .page_size = input_size,
        .buffer_type = BufferType::L1});

    std::shared_ptr<Buffer> output_buf;
    if (!test_cfg.tensix_consumer) {
        output_buf = CreateBuffer(InterleavedBufferConfig{
            .device    = device,
            .size      = output_size,
            .page_size = output_size,
            .buffer_type = BufferType::L1});
    }

    // -----------------------------------------------------------------------
    // 2. Build DFB config with borrowed_buffer set BEFORE CreateDataflowBuffer.
    // -----------------------------------------------------------------------
    experimental::dfb::DataflowBufferConfig dfb_config{
        .entry_size          = test_cfg.entry_size,
        .num_entries         = test_cfg.num_entries,
        .num_producers       = test_cfg.num_producers,
        .pap                 = test_cfg.pap,
        .num_consumers       = test_cfg.num_consumers,
        .cap                 = test_cfg.cap,
        .enable_implicit_sync = test_cfg.enable_implicit_sync,
        .borrows_memory      = true,
    };

    // -----------------------------------------------------------------------
    // 3. Create program and DFB.
    // -----------------------------------------------------------------------
    Program program = CreateProgram();
    uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(program, core, dfb_config);

    // -----------------------------------------------------------------------
    // 4. Create kernels
    // -----------------------------------------------------------------------
    KernelHandle producer_handle{};
    KernelHandle consumer_handle{};

    const ARCH arch = device->arch();

    if (arch == ARCH::QUASAR) {
        producer_handle = experimental::quasar::CreateKernel(
            program,
            PRODUCER_KERNEL,
            core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = test_cfg.num_producers,
                .compile_args = {entries_per_producer, test_cfg.entry_size}});

        if (test_cfg.tensix_consumer) {
            consumer_handle = experimental::quasar::CreateKernel(
                program,
                TENSIX_CONSUMER_KERNEL,
                core,
                experimental::quasar::QuasarComputeConfig{
                    .num_threads_per_cluster = test_cfg.num_consumers,
                    .compile_args = {entries_per_consumer}});
        } else {
            consumer_handle = experimental::quasar::CreateKernel(
                program,
                DM_CONSUMER_KERNEL,
                core,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = test_cfg.num_consumers,
                    .compile_args = {entries_per_consumer, test_cfg.entry_size}});
        }
    } else {
        // WH / BH: BRISC = producer (RISCV_0), NCRISC / Compute = consumer.
        producer_handle = CreateKernel(
            program,
            PRODUCER_KERNEL,
            core,
            DataMovementConfig{
                .processor  = DataMovementProcessor::RISCV_0,
                .compile_args = {entries_per_producer, test_cfg.entry_size}});

        if (test_cfg.tensix_consumer) {
            consumer_handle = CreateKernel(
                program,
                TENSIX_CONSUMER_KERNEL,
                core,
                ComputeConfig{.compile_args = {entries_per_consumer}});
        } else {
            consumer_handle = CreateKernel(
                program,
                DM_CONSUMER_KERNEL,
                core,
                DataMovementConfig{
                    .processor  = DataMovementProcessor::RISCV_1,
                    .compile_args = {entries_per_consumer, test_cfg.entry_size}});
        }
    }

    // -----------------------------------------------------------------------
    // 5. Bind DFB to producer and consumer kernels.
    // -----------------------------------------------------------------------
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, dfb_id, producer_handle, consumer_handle);

    // -----------------------------------------------------------------------
    // 6. Pre-fill input L1 region and set runtime args.
    // -----------------------------------------------------------------------
    auto input_words = make_sequential(input_size / sizeof(uint32_t));
    detail::WriteToDeviceL1(device, core, static_cast<uint32_t>(input_buf->address()), input_words);

    SetRuntimeArgs(program, producer_handle, core,
                   {static_cast<uint32_t>(input_buf->address())});

    if (!test_cfg.tensix_consumer) {
        SetRuntimeArgs(program, consumer_handle, core,
                       {static_cast<uint32_t>(output_buf->address())});
    }

    // -----------------------------------------------------------------------
    // 7. Set the borrowed buffer's address, then launch and wait.
    // -----------------------------------------------------------------------
    program.impl().get_dataflow_buffer(dfb_id)->set_borrowed_memory_base_addr(
        static_cast<uint32_t>(dfb_ring_buf->address()));

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    // -----------------------------------------------------------------------
    // 8. Assert the borrowed buffer address was used for the DFB ring.
    // -----------------------------------------------------------------------
    EXPECT_EQ(
        program.impl().dataflow_buffers()[0]->uniform_alloc_addr(),
        static_cast<uint32_t>(dfb_ring_buf->address()));

    // -----------------------------------------------------------------------
    // 9. Optionally verify round-trip data correctness
    // -----------------------------------------------------------------------
    if (test_cfg.verify_data && !test_cfg.tensix_consumer) {
        std::vector<uint32_t> output;
        detail::ReadFromDeviceL1(
            device, core,
            static_cast<uint32_t>(output_buf->address()),
            output_size,
            output);
        EXPECT_EQ(input_words, output);
    }
}

}  // anonymous namespace

// =============================================================================
// All-architecture tests (Quasar, Wormhole, Blackhole)
// =============================================================================

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM1Sx1S) {
    IDevice* device = devices_.at(0)->get_devices()[0];

    BorrowedDFBTestConfig cfg{
        .num_entries         = 16,
        .entry_size          = 256,
        .num_producers       = 1,
        .num_consumers       = 1,
        .pap                 = dfb::AccessPattern::STRIDED,
        .cap                 = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_consumer     = false,
        .verify_data         = true,
    };

    run_borrowed_memory_dfb_program(device, CoreCoord(0, 0), cfg);
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM1Sx1S_UpdateAddress) {
    IDevice* device = devices_.at(0)->get_devices()[0];
    const CoreCoord core(0, 0);
    const ARCH arch = device->arch();

    constexpr uint32_t num_entries = 16;
    constexpr uint32_t entry_size  = 256;
    constexpr uint32_t dfb_size    = num_entries * entry_size;
    constexpr uint32_t data_words  = dfb_size / sizeof(uint32_t);

    auto ring_buf_a = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dfb_size, .page_size = dfb_size, .buffer_type = BufferType::L1});
    auto ring_buf_b = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dfb_size, .page_size = dfb_size, .buffer_type = BufferType::L1});
    auto input_buf = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dfb_size, .page_size = dfb_size, .buffer_type = BufferType::L1});
    auto output_buf = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dfb_size, .page_size = dfb_size, .buffer_type = BufferType::L1});

    experimental::dfb::DataflowBufferConfig dfb_cfg{
        .entry_size    = entry_size,
        .num_entries   = num_entries,
        .num_producers = 1,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap           = dfb::AccessPattern::STRIDED,
        .borrows_memory = true,
    };

    Program program = CreateProgram();
    uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(program, core, dfb_cfg);

    KernelHandle producer_h{};
    KernelHandle consumer_h{};
    if (arch == ARCH::QUASAR) {
        producer_h = experimental::quasar::CreateKernel(
            program, PRODUCER_KERNEL, core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1,
                .compile_args = {num_entries, entry_size}});
        consumer_h = experimental::quasar::CreateKernel(
            program, DM_CONSUMER_KERNEL, core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1,
                .compile_args = {num_entries, entry_size}});
    } else {
        producer_h = CreateKernel(
            program, PRODUCER_KERNEL, core,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_0,
                .compile_args = {num_entries, entry_size}});
        consumer_h = CreateKernel(
            program, DM_CONSUMER_KERNEL, core,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_1,
                .compile_args = {num_entries, entry_size}});
    }

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, dfb_id, producer_h, consumer_h);

    SetRuntimeArgs(program, producer_h, core, {static_cast<uint32_t>(input_buf->address())});
    SetRuntimeArgs(program, consumer_h, core, {static_cast<uint32_t>(output_buf->address())});

    auto dfb = program.impl().get_dataflow_buffer(dfb_id);

    // --- Run 1: DFB ring at ring_buf_a ---
    std::vector<uint32_t> input_a(data_words);
    std::iota(input_a.begin(), input_a.end(), 0u);
    detail::WriteToDeviceL1(device, core, static_cast<uint32_t>(input_buf->address()), input_a);
    dfb->set_borrowed_memory_base_addr(static_cast<uint32_t>(ring_buf_a->address()));
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    EXPECT_EQ(dfb->uniform_alloc_addr(), static_cast<uint32_t>(ring_buf_a->address()));
    {
        std::vector<uint32_t> output;
        detail::ReadFromDeviceL1(
            device, core, static_cast<uint32_t>(output_buf->address()), dfb_size, output);
        EXPECT_EQ(input_a, output);
    }

    // --- Run 2: DFB ring redirected to ring_buf_b ---
    std::vector<uint32_t> input_b(data_words);
    std::iota(input_b.begin(), input_b.end(), data_words);  // distinct values from run 1
    detail::WriteToDeviceL1(device, core, static_cast<uint32_t>(input_buf->address()), input_b);
    dfb->set_borrowed_memory_base_addr(static_cast<uint32_t>(ring_buf_b->address()));
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    EXPECT_EQ(dfb->uniform_alloc_addr(), static_cast<uint32_t>(ring_buf_b->address()));
    {
        std::vector<uint32_t> output;
        detail::ReadFromDeviceL1(
            device, core, static_cast<uint32_t>(output_buf->address()), dfb_size, output);
        EXPECT_EQ(input_b, output);
    }
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMTensix1Sx1S) {
    IDevice* device = devices_.at(0)->get_devices()[0];

    BorrowedDFBTestConfig cfg{
        .num_entries         = 16,
        .entry_size          = 256,
        .num_producers       = 1,
        .num_consumers       = 1,
        .pap                 = dfb::AccessPattern::STRIDED,
        .cap                 = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_consumer     = true,
        .verify_data         = true,
    };

    run_borrowed_memory_dfb_program(device, CoreCoord(0, 0), cfg);
}

// =============================================================================
// Quasar-only tests (multi-producer / multi-consumer with implicit sync)
// =============================================================================

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB with implicit sync requires Quasar";
    }

    IDevice* device = devices_.at(0)->get_devices()[0];

    BorrowedDFBTestConfig cfg{
        .num_entries         = 16,
        .entry_size          = 256,
        .num_producers       = 2,
        .num_consumers       = 4,
        .pap                 = dfb::AccessPattern::STRIDED,
        .cap                 = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true,
        .tensix_consumer     = false,
        .verify_data         = true,
    };

    run_borrowed_memory_dfb_program(device, CoreCoord(0, 0), cfg);
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMDM4Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB with implicit sync requires Quasar";
    }

    IDevice* device = devices_.at(0)->get_devices()[0];

    BorrowedDFBTestConfig cfg{
        .num_entries         = 16,
        .entry_size          = 256,
        .num_producers       = 4,
        .num_consumers       = 2,
        .pap                 = dfb::AccessPattern::STRIDED,
        .cap                 = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true,
        .tensix_consumer     = false,
        .verify_data         = true,
    };

    run_borrowed_memory_dfb_program(device, CoreCoord(0, 0), cfg);
}

TEST_F(MeshDeviceFixture, BorrowedMemoryDMTensix2Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB with implicit sync requires Quasar";
    }

    IDevice* device = devices_.at(0)->get_devices()[0];

    BorrowedDFBTestConfig cfg{
        .num_entries         = 16,
        .entry_size          = 256,
        .num_producers       = 2,
        .num_consumers       = 4,
        .pap                 = dfb::AccessPattern::STRIDED,
        .cap                 = dfb::AccessPattern::ALL,
        .enable_implicit_sync = true,
        .tensix_consumer     = true,
        .verify_data         = false,
    };

    run_borrowed_memory_dfb_program(device, CoreCoord(0, 0), cfg);
}

}  // namespace tt::tt_metal
