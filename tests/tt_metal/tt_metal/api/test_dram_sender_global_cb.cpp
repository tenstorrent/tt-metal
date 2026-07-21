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
#include <cstdlib>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/dram_sender_state_block.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/experimental/global_circular_buffer.hpp>

#include "impl/kernels/kernel.hpp"  // DramConfig
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

class DramSenderGCBMultiDeviceFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
            GTEST_SKIP() << "Requires TT_METAL_SLOW_DISPATCH_MODE=1";
        }
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Requires Blackhole";
        }

        const auto& cluster = MetalContext::instance().get_cluster();
        const auto& available_ids = cluster.user_exposed_chip_ids();
        if (available_ids.size() < 2) {
            GTEST_SKIP() << "Requires at least two devices";
        }
        std::vector<ChipId> device_ids(available_ids.begin(), std::next(available_ids.begin(), 2));
        mesh_device_ = distributed::MeshDevice::create(
            distributed::MeshDeviceConfig(distributed::MeshShape{1, 2}, std::nullopt, device_ids));
        if (!MetalContext::instance(mesh_device_->impl().get_context_id())
                 .hal()
                 .has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

TEST_F(DramSenderGCBMultiDeviceFixture, ConfigAndSenderStateUsePerDeviceDramTopology) {
    constexpr uint32_t kGcbSize = 1024;
    constexpr uint32_t kBankId = 0;
    constexpr uint32_t kNumReceivers = 2;
    CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));
    auto gcb = experimental::CreateGlobalCircularBufferWithDramSenders(
        *mesh_device_, {{kBankId, receiver_cores}}, kGcbSize, BufferType::L1, /*dual_senders_per_bank=*/true);
    ASSERT_EQ(gcb.sender_receiver_core_mapping().size(), kNumReceivers);

    const auto& hal = MetalContext::instance(mesh_device_->impl().get_context_id()).hal();
    const uint64_t dram_l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint64_t sender_state_addr =
        dram_l1_noc_offset + static_cast<uint64_t>(experimental::sender_state_drisc_l1_base(gcb));
    const auto receiver_logical_cores =
        corerange_to_cores(receiver_cores, /*max_cores=*/std::nullopt, /*row_wise=*/true);

    for (IDevice* device : mesh_device_->get_devices()) {
        const std::vector<CoreCoord> device_senders = mesh_device_->impl().dram_sender_logical_cores(device, kBankId);
        ASSERT_EQ(device_senders.size(), kNumReceivers);

        for (uint32_t sender_role = 0; sender_role < kNumReceivers; ++sender_role) {
            const CoreCoord expected_sender_virtual =
                device->virtual_core_from_logical_core(device_senders[sender_role], CoreType::DRAM);

            // Each receiver's config page stores the NOC XY to increment when returning
            // pages_acked credits. With dual senders and two receivers, role s owns receiver s.
            std::vector<uint32_t> receiver_config;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device,
                receiver_logical_cores[sender_role],
                gcb.config_address(),
                10 * sizeof(uint32_t),
                receiver_config,
                CoreType::WORKER);
            ASSERT_GE(receiver_config.size(), 10u);
            EXPECT_EQ(receiver_config[8], expected_sender_virtual.x)
                << "device " << device->id() << ", sender role " << sender_role;
            EXPECT_EQ(receiver_config[9], expected_sender_virtual.y)
                << "device " << device->id() << ", sender role " << sender_role;

            const CoreCoord expected_receiver_phys =
                device->worker_core_from_logical_core(receiver_logical_cores[sender_role]);
            const size_t sender_state_size = sizeof(DramSenderStateBlock) + 2 * sizeof(uint32_t);
            std::vector<uint8_t> sender_state_bytes(sender_state_size, 0);
            MetalContext::instance(mesh_device_->impl().get_context_id())
                .get_cluster()
                .read_core(
                    sender_state_bytes.data(),
                    sender_state_bytes.size(),
                    tt_cxy_pair(device->id(), expected_sender_virtual),
                    sender_state_addr);
            const auto* sender_state = reinterpret_cast<const DramSenderStateBlock*>(sender_state_bytes.data());
            EXPECT_EQ(sender_state->num_receivers, 1u);
            const auto* receiver_xy =
                reinterpret_cast<const uint32_t*>(sender_state_bytes.data() + sizeof(DramSenderStateBlock));
            EXPECT_EQ(receiver_xy[0], expected_receiver_phys.x)
                << "device " << device->id() << ", sender role " << sender_role;
            EXPECT_EQ(receiver_xy[1], expected_receiver_phys.y)
                << "device " << device->id() << ", sender role " << sender_role;
        }
    }
}

TEST_F(DramSenderGCBFixture, SmokeOneSenderFourReceivers) {
    // Layout: 4 receivers, each receives one 64-byte page.
    constexpr uint32_t kNumReceivers = 4;
    constexpr uint32_t kPageSize = 64;  // multiple of L1_ALIGNMENT (16 on BH)
    constexpr uint32_t kNumPages = 1;
    constexpr uint32_t kRemoteCBId = 31;

    // Sender: bank 0
    const uint32_t bank_id = 0;
    // Receivers
    CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers = {{bank_id, receiver_cores}};

    // Size: per-receiver fifo. Use 1KB.
    constexpr uint32_t kGcbSize = 1024;
    auto gcb = experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device_, bank_to_receivers, kGcbSize, BufferType::L1, /*support_multi_receiver_shards=*/true);
    // Use the sender coord the factory resolved; recomputing via pick_unused_dram_logical_core
    // would couple this test to the picker's current strategy.
    const CoreCoord sender_logical = gcb.sender_receiver_core_mapping().at(0).first;

    // Pre-load DRISC L1 with per-receiver data pattern starting at DRISC L1 UNRESERVED + offset
    // far enough above pages_sent_drisc_l1_base.
    const auto& hal = MetalContext::instance().hal();
    const uint32_t drisc_l1_unreserved = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    // DRISC slots are packed at uint32 stride (see REMOTE_CB_LOCAL_PAGES_STRIDE).
    constexpr uint32_t kDriscSlotBytes = sizeof(uint32_t);
    const uint32_t pages_sent_size = 2 * kDriscSlotBytes * kNumReceivers;
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
    ASSERT_EQ(experimental::pages_sent_drisc_l1_base(gcb), pages_sent_addr);

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
        static_cast<uint32_t>(experimental::pages_sent_worker_l1_base(gcb)),
    };
    KernelHandle sender_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_sender.cpp",
        sender_logical,
        DramConfig{.noc = NOC::NOC_0, .compile_args = sender_compile_args});

    std::vector<uint32_t> sender_rt_args;
    sender_rt_args.reserve(2 * kNumReceivers);
    const auto& receiver_phys = experimental::receiver_coords_per_sender(gcb).at(0);
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

    // Verify pages_sent == pages_acked on DRISC (the barrier in the sender kernel waits for
    // this). Slots are packed: per receiver, pages_sent at +0 uint32, pages_acked at +1 uint32.
    const uint64_t pages_sent_noc_addr = drisc_l1_noc_addr_base + (pages_sent_addr - drisc_l1_unreserved);
    std::vector<uint32_t> pages_buf(2 * kNumReceivers, 0);
    MetalContext::instance().get_cluster().read_core(
        pages_buf.data(),
        pages_buf.size() * sizeof(uint32_t),
        tt_cxy_pair(mesh_device_->build_id(), sender_virtual),
        pages_sent_noc_addr);
    for (uint32_t r = 0; r < kNumReceivers; ++r) {
        uint32_t sent = pages_buf[2 * r];
        uint32_t acked = pages_buf[2 * r + 1];
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
    CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers = {{bank_id, receiver_cores}};
    auto gcb = experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device_, bank_to_receivers, kGcbSize, BufferType::L1, /*support_multi_receiver_shards=*/true);
    const CoreCoord sender_logical = gcb.sender_receiver_core_mapping().at(0).first;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t drisc_l1_unreserved = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    auto align_up = [&](uint32_t a) { return (a + l1_alignment - 1) & ~(l1_alignment - 1); };

    const uint32_t pages_sent_addr = drisc_l1_unreserved;
    // DRISC slots are packed at uint32 stride (see REMOTE_CB_LOCAL_PAGES_STRIDE).
    uint32_t cursor = pages_sent_addr + 2 * sizeof(uint32_t) * kNumReceivers;
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
        static_cast<uint32_t>(experimental::pages_sent_worker_l1_base(gcb)),
    };
    KernelHandle sender_kernel_id = CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_sender.cpp",
        sender_logical,
        DramConfig{.noc = NOC::NOC_0, .compile_args = sender_compile_args});
    std::vector<uint32_t> sender_rt_args;
    const auto& receiver_phys = experimental::receiver_coords_per_sender(gcb).at(0);
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

// Two DRAM-sender GCBs sharing bank 0 with disjoint receivers. Validates the DRISC L1
// arena gives them disjoint pages_sent regions and that each GCB's data flow is
// uncorrupted by the other's presence.
//
// Pre-arena, both GCBs would have hardcoded pages_sent_drisc_l1_base_ = UNRESERVED,
// so their per-receiver counter slots overlapped: receivers from GCB A NoC-inc'd into
// the same DRISC L1 words that GCB B's receivers also targeted. With the arena, GCB A
// lands at UNRESERVED and GCB B lands at UNRESERVED + sizeof(GCB A's pages_sent), and
// neither kernel touches the other's bookkeeping.
//
// Programs run sequentially because only one DRISC kernel can occupy bank 0's DRISC at
// a time. Both GCBs are live across both program runs (the arena allocations outlive
// each individual program); each run targets one GCB.
TEST_F(DramSenderGCBFixture, MultiGcbDisjointPagesSent) {
    constexpr uint32_t kPageSize = 64;
    constexpr uint32_t kNumPages = 1;
    constexpr uint32_t kRemoteCBId = 31;
    constexpr uint32_t kGcbSize = 1024;
    const uint32_t bank_id = 0;

    // GCB A: receiver at worker (0, 0). GCB B: receiver at worker (1, 0). Same bank.
    CoreRangeSet recv_a(CoreRange({0, 0}, {0, 0}));
    CoreRangeSet recv_b(CoreRange({1, 0}, {1, 0}));
    auto gcb_a = experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device_, {{bank_id, recv_a}}, kGcbSize, BufferType::L1, /*support_multi_receiver_shards=*/true);
    auto gcb_b = experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device_, {{bank_id, recv_b}}, kGcbSize, BufferType::L1, /*support_multi_receiver_shards=*/true);

    const DeviceAddr pa = experimental::pages_sent_drisc_l1_base(gcb_a);
    const DeviceAddr pb = experimental::pages_sent_drisc_l1_base(gcb_b);
    ASSERT_NE(pa, pb) << "Arena handed both GCBs the same pages_sent base (0x" << std::hex << pa
                      << "); this is the corruption the arena was meant to fix.";
    // Per-GCB pages_sent footprint here is 2 * sizeof(uint32_t) * num_receivers(=1) (DRISC slots
    // are packed at uint32 stride; see REMOTE_CB_LOCAL_PAGES_STRIDE).
    const auto& hal = MetalContext::instance().hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    EXPECT_GE(pb, pa + 2 * sizeof(uint32_t)) << "GCB B's pages_sent overlaps GCB A's range";

    // Both GCBs are on the same bank, so they share a sender core; pick either one's.
    const CoreCoord sender_logical = gcb_a.sender_receiver_core_mapping().at(0).first;
    const uint32_t drisc_l1_unreserved = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t drisc_l1_noc_addr_base =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    auto sender_virtual = device_->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
    auto align_up = [&](uint32_t a) { return (a + l1_alignment - 1) & ~(l1_alignment - 1); };

    // Run one GCB's data flow end-to-end. The pages_sent address comes from the GCB
    // (i.e., from the arena). Working state (noc_xy / config / data) is placed above
    // the arena's kernel_working_region_base so that the *other* GCB's pages_sent
    // (also inside the fixed zone) is never touched.
    auto run_one = [&](const tt::tt_metal::experimental::GlobalCircularBuffer& gcb,
                       const CoreRangeSet& receivers,
                       uint32_t pattern_seed) {
        constexpr uint32_t kNumReceivers = 1;
        const uint32_t pages_sent_addr = static_cast<uint32_t>(experimental::pages_sent_drisc_l1_base(gcb));

        // Kernel-local layout: noc_xy / config / data sit above the arena's GCB zone,
        // i.e. at the same offset as arena.kernel_working_region_base().
        uint32_t cursor = align_up(drisc_l1_unreserved + DriscL1Arena::kGcbZoneSize);
        const uint32_t noc_xy_addr = cursor;
        cursor += 2 * sizeof(uint32_t) * kNumReceivers;
        cursor = align_up(cursor);
        const uint32_t config_addr = cursor;
        cursor += 16;
        cursor = align_up(cursor);
        const uint32_t data_addr = cursor;

        // Per-receiver pattern.
        std::vector<uint32_t> pattern(kNumReceivers * kPageSize / sizeof(uint32_t));
        for (uint32_t r = 0; r < kNumReceivers; ++r) {
            for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
                pattern[r * kPageSize / sizeof(uint32_t) + w] = pattern_seed + r * 0x100u + w;
            }
        }
        MetalContext::instance().get_cluster().write_core(
            pattern.data(),
            pattern.size() * sizeof(uint32_t),
            tt_cxy_pair(mesh_device_->build_id(), sender_virtual),
            drisc_l1_noc_addr_base + (data_addr - drisc_l1_unreserved));

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
            static_cast<uint32_t>(experimental::pages_sent_worker_l1_base(gcb)),
        };
        KernelHandle sender_kernel_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_sender.cpp",
            sender_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = sender_compile_args});
        std::vector<uint32_t> sender_rt_args;
        const auto& receiver_phys = experimental::receiver_coords_per_sender(gcb).at(0);
        for (const auto& c : receiver_phys) {
            sender_rt_args.push_back(c.x);
            sender_rt_args.push_back(c.y);
        }
        SetRuntimeArgs(program, sender_kernel_id, sender_logical, sender_rt_args);

        CircularBufferConfig cb_config(kPageSize);
        cb_config.remote_index(kRemoteCBId).set_page_size(kPageSize).set_data_format(tt::DataFormat::Float16_b);
        experimental::CreateCircularBuffer(program, receivers, cb_config, gcb);
        std::vector<uint32_t> receiver_compile_args = {kRemoteCBId, kNumPages};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_receiver.cpp",
            receivers,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = receiver_compile_args});

        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device_->mesh_command_queue());
    };

    // Run A then B. GCB B is alive (host-side) the whole time; its pages_sent region
    // sits in the arena zone above GCB A's but is untouched by A's kernel.
    run_one(gcb_a, recv_a, 0xAAAA0000u);
    run_one(gcb_b, recv_b, 0xBBBB0000u);

    // Verify receiver A got pattern A.
    {
        std::vector<uint32_t> result;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device_, CoreCoord{0, 0}, gcb_a.buffer_address(), kPageSize, result, CoreType::WORKER);
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            uint32_t expected = 0xAAAA0000u + w;
            EXPECT_EQ(result[w], expected) << "Receiver A word " << w;
        }
    }
    // Verify receiver B got pattern B.
    {
        std::vector<uint32_t> result;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device_, CoreCoord{1, 0}, gcb_b.buffer_address(), kPageSize, result, CoreType::WORKER);
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            uint32_t expected = 0xBBBB0000u + w;
            EXPECT_EQ(result[w], expected) << "Receiver B word " << w;
        }
    }

    // Verify pages_sent==pages_acked at BOTH GCB regions on the DRISC. Each GCB here has 1
    // receiver, so its DRISC footprint is 2 uint32_t (pages_sent, pages_acked).
    auto check_pages = [&](DeviceAddr pages_sent_addr, const char* tag) {
        const uint64_t noc_addr = drisc_l1_noc_addr_base + (pages_sent_addr - drisc_l1_unreserved);
        std::vector<uint32_t> buf(2, 0);
        MetalContext::instance().get_cluster().read_core(
            buf.data(), buf.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), sender_virtual), noc_addr);
        uint32_t sent = buf[0];
        uint32_t acked = buf[1];
        EXPECT_EQ(sent, acked) << tag << " pages_sent != pages_acked (sent=" << sent << " acked=" << acked << ")";
        EXPECT_GT(sent, 0u) << tag << " no pages were pushed";
    };
    check_pages(pa, "GCB A");
    check_pages(pb, "GCB B");
}

// Allocating a GCB *after* the prefetcher kernel has started must not move the kernel's
// L1 layout. We don't drive the prefetcher here (the user asked for low-level kernels),
// but the equivalent invariant is checkable via the arena's contract directly: the
// kernel_working_region_base value is fixed for the device's lifetime regardless of
// arena allocations. The GCB-A path above is the smoke test for that invariant; the
// concrete address-stability check belongs in test_drisc_l1_arena.cpp once that lands.

TEST_F(DramSenderGCBFixture, RejectsDuplicateSender) {
    CoreRangeSet recv0(CoreRange({0, 0}, {0, 0}));
    CoreRangeSet recv1(CoreRange({1, 0}, {1, 0}));
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers = {{0, recv0}, {0, recv1}};
    EXPECT_ANY_THROW(experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device_, bank_to_receivers, 1024, BufferType::L1, /*support_multi_receiver_shards=*/true));
}

}  // namespace tt::tt_metal
