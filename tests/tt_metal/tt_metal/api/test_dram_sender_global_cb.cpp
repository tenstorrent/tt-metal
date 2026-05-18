// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end smoke test for DramSenderGlobalCircularBuffer:
// one DRISC sender on bank 0's unused subchannel pushes a known per-receiver
// pattern to a CoreRangeSet of worker receivers via remote_cb_*. After Finish,
// verify each receiver's L1 slice matches the expected stripe and that the
// sender's pages_acked counters all caught up to pages_sent.

#include <gtest/gtest.h>
#include <cstdint>
#include <exception>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/dram_sender_global_circular_buffer.hpp>
#include <tt-metalium/dram_subchannel.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <tt-metalium/experimental/dispatch_context.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

class DramSenderGCBFixture : public BlackholeSingleCardFixture {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;
        }
        const auto& hal = MetalContext::instance().hal();
        if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
        mesh_device_ = devices_[0].get();
        device_ = mesh_device_->get_devices()[0];
    }

    distributed::MeshDevice* mesh_device_{};
    IDevice* device_{};
};

TEST_F(DramSenderGCBFixture, SmokeOneSenderFourReceivers) {
    // Layout: 4 receivers, each receives one 64-byte page.
    constexpr uint32_t kNumReceivers = 4;
    constexpr uint32_t kPageSize = 64;  // multiple of L1_ALIGNMENT (16 on BH)
    constexpr uint32_t kNumPages = 1;
    constexpr uint32_t kRemoteCBId = 31;

    // Sender: bank 0, unused subchannel
    const uint32_t bank_id = 0;
    const uint32_t unused_sub = experimental::pick_unused_dram_subchannel(device_, bank_id);
    CoreCoord sender_logical{bank_id, unused_sub};

    // Receivers
    CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_logical, receiver_cores}};

    // Size: per-receiver fifo. Use 1KB.
    constexpr uint32_t kGcbSize = 1024;
    auto gcb = experimental::CreateDramSenderGlobalCircularBuffer(mesh_device_, mapping, kGcbSize, BufferType::L1);

    // Pre-load DRISC L1 with per-receiver data pattern starting at DRISC L1 UNRESERVED + offset
    // far enough above pages_sent_drisc_l1_base.
    const auto& hal = MetalContext::instance().hal();
    const uint32_t drisc_l1_unreserved = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    const uint32_t pages_sent_size = 2 * l1_alignment * kNumReceivers;
    const uint32_t noc_xy_size = 2 * sizeof(uint32_t) * kNumReceivers;
    const uint32_t config_size = 16;  // 4 uint32 words
    uint32_t cursor = drisc_l1_unreserved;
    const uint32_t pages_sent_addr = cursor;
    cursor += pages_sent_size;
    cursor = (cursor + l1_alignment - 1) & ~(l1_alignment - 1);
    const uint32_t noc_xy_addr = cursor;
    cursor += noc_xy_size;
    cursor = (cursor + l1_alignment - 1) & ~(l1_alignment - 1);
    const uint32_t config_addr = cursor;
    cursor += config_size;
    cursor = (cursor + l1_alignment - 1) & ~(l1_alignment - 1);
    const uint32_t data_addr = cursor;

    // Sanity: our DramSenderGlobalCircularBuffer agreed to plant pages_sent at drisc_l1_unreserved.
    ASSERT_EQ(gcb.pages_sent_drisc_l1_base(), pages_sent_addr);

    // Per-receiver pattern: pattern[r] = 0xABCD0000 + r*page_index*256
    std::vector<uint32_t> pattern(kNumReceivers * kPageSize / sizeof(uint32_t));
    for (uint32_t r = 0; r < kNumReceivers; ++r) {
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            pattern[r * kPageSize / sizeof(uint32_t) + w] = 0xABCD0000u + r * 0x100u + w;
        }
    }
    auto sender_virtual = device_->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
    const uint64_t drisc_l1_noc_addr_base =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t data_noc_addr = drisc_l1_noc_addr_base + (data_addr - drisc_l1_unreserved);
    MetalContext::instance().get_cluster().write_core(
        pattern.data(),
        pattern.size() * sizeof(uint32_t),
        tt_cxy_pair(mesh_device_->build_id(), sender_virtual),
        data_noc_addr);

    // Build a single program with both sender (DRISC) and receiver (worker) kernels.
    distributed::MeshCoordinateRange device_range(distributed::MeshCoordinate(0, 0));
    Program program = CreateProgram();

    std::vector<uint32_t> sender_compile_args = {
        kRemoteCBId,
        kNumPages,
        kPageSize,
        kNumReceivers,
        pages_sent_addr,
        noc_xy_addr,
        config_addr,
        data_addr,
        kGcbSize,
        static_cast<uint32_t>(gcb.buffer_address()),
        static_cast<uint32_t>(gcb.pages_sent_worker_l1_base()),
    };
    KernelHandle sender_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_sender.cpp",
        sender_logical,
        DramConfig{.noc = NOC::NOC_0, .compile_args = sender_compile_args});

    std::vector<uint32_t> sender_rt_args;
    sender_rt_args.reserve(2 * kNumReceivers);
    const auto& receiver_phys = gcb.receiver_coords_per_sender().at(0);
    for (const auto& c : receiver_phys) {
        sender_rt_args.push_back(c.x);
        sender_rt_args.push_back(c.y);
    }
    SetRuntimeArgs(program, sender_kernel_id, sender_logical, sender_rt_args);

    CircularBufferConfig cb_config(kPageSize);
    cb_config.remote_index(kRemoteCBId).set_page_size(kPageSize).set_data_format(tt::DataFormat::Float16_b);
    experimental::CreateCircularBuffer(program, receiver_cores, cb_config, gcb);

    std::vector<uint32_t> receiver_compile_args = {kRemoteCBId, kNumPages};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = receiver_compile_args});

    distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device_->mesh_command_queue());

    // Verify each receiver's L1 slice
    auto receivers_vec = corerange_to_cores(receiver_cores);
    for (uint32_t r = 0; r < receivers_vec.size(); ++r) {
        std::vector<uint32_t> result;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device_, receivers_vec[r], gcb.buffer_address(), kPageSize, result, CoreType::WORKER);
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            uint32_t expected = 0xABCD0000u + r * 0x100u + w;
            EXPECT_EQ(result[w], expected) << "Receiver " << r << " word " << w << " mismatch (expected 0x" << std::hex
                                           << expected << ", got 0x" << result[w] << std::dec << ")";
        }
    }

    // Verify pages_sent == pages_acked on DRISC (the barrier in the sender kernel waits for this)
    const uint64_t pages_sent_noc_addr = drisc_l1_noc_addr_base + (pages_sent_addr - drisc_l1_unreserved);
    std::vector<uint32_t> pages_buf(2 * l1_alignment * kNumReceivers / sizeof(uint32_t), 0);
    MetalContext::instance().get_cluster().read_core(
        pages_buf.data(),
        pages_buf.size() * sizeof(uint32_t),
        tt_cxy_pair(mesh_device_->build_id(), sender_virtual),
        pages_sent_noc_addr);
    const uint32_t stride_uints = 2 * l1_alignment / sizeof(uint32_t);
    for (uint32_t r = 0; r < kNumReceivers; ++r) {
        uint32_t sent = pages_buf[r * stride_uints];
        uint32_t acked = pages_buf[r * stride_uints + l1_alignment / sizeof(uint32_t)];
        EXPECT_EQ(sent, acked) << "Pages sent/acked mismatch for receiver " << r << " (sent=" << sent
                               << ", acked=" << acked << ")";
        EXPECT_GT(sent, 0u) << "Sender did not push any pages to receiver " << r;
    }
}

// Same data flow as SmokeOneSenderFourReceivers, but the sender (DRISC) and receiver
// (workers) live in TWO SEPARATE Programs and we rely on async slow dispatch to launch
// them concurrently. This mirrors how the ttnn prefetcher op + matmul op flow works
// (each op enqueues its own Program). If this passes, async SD is a viable substitute
// for fast dispatch for the DRAM-core mode.
TEST_F(DramSenderGCBFixture, SmokeTwoProgramsAsyncSlowDispatch) {
    constexpr uint32_t kNumReceivers = 4;
    constexpr uint32_t kPageSize = 64;
    constexpr uint32_t kNumPages = 1;
    constexpr uint32_t kRemoteCBId = 31;
    constexpr uint32_t kGcbSize = 1024;

    const uint32_t bank_id = 0;
    const uint32_t unused_sub = experimental::pick_unused_dram_subchannel(device_, bank_id);
    CoreCoord sender_logical{bank_id, unused_sub};
    CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_logical, receiver_cores}};
    auto gcb = experimental::CreateDramSenderGlobalCircularBuffer(mesh_device_, mapping, kGcbSize, BufferType::L1);

    const auto& hal = MetalContext::instance().hal();
    const uint32_t drisc_l1_unreserved = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    auto align_up = [&](uint32_t a) { return (a + l1_alignment - 1) & ~(l1_alignment - 1); };

    const uint32_t pages_sent_addr = drisc_l1_unreserved;
    uint32_t cursor = pages_sent_addr + 2 * l1_alignment * kNumReceivers;
    cursor = align_up(cursor);
    const uint32_t noc_xy_addr = cursor;
    cursor += 2 * sizeof(uint32_t) * kNumReceivers;
    cursor = align_up(cursor);
    const uint32_t config_addr = cursor;
    cursor += 16;
    cursor = align_up(cursor);
    const uint32_t data_addr = cursor;

    // Pre-load DRISC L1 with a per-receiver pattern.
    std::vector<uint32_t> pattern(kNumReceivers * kPageSize / sizeof(uint32_t));
    for (uint32_t r = 0; r < kNumReceivers; ++r) {
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            pattern[r * kPageSize / sizeof(uint32_t) + w] = 0x55AA0000u + r * 0x100u + w;
        }
    }
    auto sender_virtual = device_->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
    const uint64_t drisc_l1_noc_addr_base =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    MetalContext::instance().get_cluster().write_core(
        pattern.data(),
        pattern.size() * sizeof(uint32_t),
        tt_cxy_pair(mesh_device_->build_id(), sender_virtual),
        drisc_l1_noc_addr_base + (data_addr - drisc_l1_unreserved));

    // --- Sender program: DRISC kernel only ---
    Program sender_program = CreateProgram();
    std::vector<uint32_t> sender_compile_args = {
        kRemoteCBId,
        kNumPages,
        kPageSize,
        kNumReceivers,
        pages_sent_addr,
        noc_xy_addr,
        config_addr,
        data_addr,
        kGcbSize,
        static_cast<uint32_t>(gcb.buffer_address()),
        static_cast<uint32_t>(gcb.pages_sent_worker_l1_base()),
    };
    KernelHandle sender_kernel_id = CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_sender.cpp",
        sender_logical,
        DramConfig{.noc = NOC::NOC_0, .compile_args = sender_compile_args});
    std::vector<uint32_t> sender_rt_args;
    const auto& receiver_phys = gcb.receiver_coords_per_sender().at(0);
    for (const auto& c : receiver_phys) {
        sender_rt_args.push_back(c.x);
        sender_rt_args.push_back(c.y);
    }
    SetRuntimeArgs(sender_program, sender_kernel_id, sender_logical, sender_rt_args);

    // --- Receiver program: worker kernel only, with c_31 attached to the GCB ---
    Program receiver_program = CreateProgram();
    CircularBufferConfig cb_config(kPageSize);
    cb_config.remote_index(kRemoteCBId).set_page_size(kPageSize).set_data_format(tt::DataFormat::Float16_b);
    experimental::CreateCircularBuffer(receiver_program, receiver_cores, cb_config, gcb);
    std::vector<uint32_t> receiver_compile_args = {kRemoteCBId, kNumPages};
    CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = receiver_compile_args});

    // Enable async SD and enqueue the two programs as separate workloads.
    experimental::DispatchContext::get().enable_asynchronous_slow_dispatch(mesh_device_);
    distributed::MeshCoordinateRange device_range(distributed::MeshCoordinate(0, 0));
    {
        distributed::MeshWorkload sender_workload;
        sender_workload.add_program(device_range, std::move(sender_program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), sender_workload, /*blocking=*/false);
        distributed::MeshWorkload receiver_workload;
        receiver_workload.add_program(device_range, std::move(receiver_program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), receiver_workload, /*blocking=*/false);
        distributed::Finish(mesh_device_->mesh_command_queue());
    }
    experimental::DispatchContext::get().disable_asynchronous_slow_dispatch(mesh_device_);

    // Verify each receiver's L1 slice matches the per-receiver expected stripe.
    auto receivers_vec = corerange_to_cores(receiver_cores);
    for (uint32_t r = 0; r < receivers_vec.size(); ++r) {
        std::vector<uint32_t> result;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device_, receivers_vec[r], gcb.buffer_address(), kPageSize, result, CoreType::WORKER);
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            uint32_t expected = 0x55AA0000u + r * 0x100u + w;
            EXPECT_EQ(result[w], expected) << "Receiver " << r << " word " << w;
        }
    }
}

TEST_F(DramSenderGCBFixture, RejectsDuplicateSender) {
    CoreCoord sender_logical{0, experimental::pick_unused_dram_subchannel(device_, 0)};
    CoreRangeSet recv0(CoreRange({0, 0}, {0, 0}));
    CoreRangeSet recv1(CoreRange({1, 0}, {1, 0}));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_logical, recv0}, {sender_logical, recv1}};
    EXPECT_ANY_THROW(experimental::CreateDramSenderGlobalCircularBuffer(mesh_device_, mapping, 1024, BufferType::L1));
}

}  // namespace tt::tt_metal
