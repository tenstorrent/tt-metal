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
#include <optional>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/global_circular_buffer_dram_sender_internal.hpp"
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
#include "multi_device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
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
    }

    distributed::MeshDevice* mesh_device_{};
};

// Sender 0's receivers as the physical NOC XY runtime args the smoke sender kernel expects,
// translated through the device the program will run on.
std::vector<uint32_t> receiver_noc_xy_rt_args(const experimental::GlobalCircularBuffer& gcb, IDevice* device) {
    const auto& receivers = experimental::receiver_logical_cores_per_sender(gcb).at(0);
    std::vector<uint32_t> rt_args;
    rt_args.reserve(2 * receivers.size());
    for (const auto& receiver_logical : receivers) {
        const CoreCoord phys = device->worker_core_from_logical_core(receiver_logical);
        rt_args.push_back(phys.x);
        rt_args.push_back(phys.y);
    }
    return rt_args;
}

// One MeshDevice spanning two chips, which is what makes the per-device fan-out inside one GCB
// observable. A MeshDispatchFixture would hand back unit meshes (one 1x1 MeshDevice per chip),
// collapsing every per-device loop below to a single iteration, and its shared-device suite setup
// would be torn down twice.
// Runs under fast dispatch (MeshDeviceFixtureBase's requirement, and what CI defaults to). Nothing
// here needs slow dispatch: the receiver config lands via a blocking shard write and the sender
// state block via cluster.write_core, so both have landed by the time the factory returns and the
// raw L1 reads below are safe.
class DramSenderGCBMultiDeviceFixture : public MeshDeviceFixtureBase {
protected:
    DramSenderGCBMultiDeviceFixture() :
        MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 2}, .arch = tt::ARCH::BLACKHOLE}) {}

    void SetUp() override {
        // Opens the 1x2 mesh, or skips on a non-Blackhole arch / a system mesh smaller than two devices.
        MeshDeviceFixtureBase::SetUp();
        if (mesh_device_ == nullptr) {
            return;
        }
        if (!MetalContext::instance(mesh_device_->impl().get_context_id())
                 .hal()
                 .has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }
};

TEST_F(DramSenderGCBMultiDeviceFixture, ConfigAndSenderStateUsePerDeviceDramTopology) {
    constexpr uint32_t kGcbSize = 1024;
    constexpr uint32_t kSendersPerBank = 2;

    // Sweep every bank rather than pinning bank 0. Logical bank ids are the compacted (harvested)
    // view index, so a pair of devices that harvest different late channels still agrees on bank 0
    // while diverging on some later bank; testing one bank can therefore miss the very divergence
    // this test exists to catch.
    const uint32_t num_banks = mesh_device_->dram_grid_size().x;
    ASSERT_GT(num_banks, 0u);

    // Bank b's two receivers are the column (b, 0)..(b, 1), so every bank gets a disjoint receiver
    // pair (one GCB cannot map the same worker to two senders) and the row-wise split inside
    // build_dram_sender_mapping hands (b, 0) to role 0 and (b, 1) to role 1.
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers;
    std::vector<std::vector<CoreCoord>> receivers_per_bank;
    bank_to_receivers.reserve(num_banks);
    receivers_per_bank.reserve(num_banks);
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        bank_to_receivers.emplace_back(bank, CoreRangeSet(CoreRange({bank, 0}, {bank, kSendersPerBank - 1})));
        receivers_per_bank.push_back(
            corerange_to_cores(bank_to_receivers.back().second, /*max_cores=*/std::nullopt, /*row_wise=*/true));
    }

    auto gcb = experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device_, bank_to_receivers, kGcbSize, BufferType::L1, /*support_multi_receiver_shards=*/false);
    ASSERT_EQ(gcb.sender_receiver_core_mapping().size(), num_banks * kSendersPerBank);

    const auto& hal = MetalContext::instance(mesh_device_->impl().get_context_id()).hal();
    const uint64_t dram_l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint64_t sender_state_addr =
        dram_l1_noc_offset + static_cast<uint64_t>(experimental::sender_state_drisc_l1_base(gcb));

    // Tracks whether this device pair actually places any sender differently. Harvest masks are a
    // property of the silicon, so a homogeneous pair exercises the per-device path without
    // distinguishing it from the old mesh-wide reference translation; say so rather than let a
    // green run read as proof.
    bool any_sender_placement_differs = false;

    IDevice* reference_device = mesh_device_->get_devices().front();
    auto& cluster = MetalContext::instance(mesh_device_->impl().get_context_id()).get_cluster();
    std::vector<uint8_t> sender_state_bytes(sizeof(DramSenderStateBlock) + 2 * sizeof(uint32_t));

    for (IDevice* device : mesh_device_->get_devices()) {
        for (uint32_t bank = 0; bank < num_banks; ++bank) {
            const std::vector<CoreCoord> device_senders = mesh_device_->impl().dram_sender_logical_cores(device, bank);
            ASSERT_EQ(device_senders.size(), kSendersPerBank);

            for (uint32_t sender_role = 0; sender_role < kSendersPerBank; ++sender_role) {
                SCOPED_TRACE(fmt::format("device {}, bank {}, sender role {}", device->id(), bank, sender_role));
                const size_t mapping_idx = (bank * kSendersPerBank) + sender_role;

                // Logical sender coords name an endpoint role, so they are the same on every device;
                // only the physical subchannel they resolve to tracks that device's DRAM harvest
                // mask. The GCB's one mapping is only valid for the whole mesh because of this.
                EXPECT_EQ(device_senders[sender_role], gcb.sender_receiver_core_mapping()[mapping_idx].first);

                const CoreCoord expected_sender_virtual =
                    device->virtual_core_from_logical_core(device_senders[sender_role], CoreType::DRAM);
                any_sender_placement_differs |=
                    (expected_sender_virtual !=
                     reference_device->virtual_core_from_logical_core(device_senders[sender_role], CoreType::DRAM));

                // Each receiver's config page stores the NOC XY to increment when returning
                // pages_acked credits. With dual senders and two receivers, role s owns receiver s.
                //
                // Reads the physical IDevice, not the mesh: slow_dispatch::ReadFromL1 takes a
                // MeshDevice and TT_FATALs unless it is a unit mesh, because it has no way to say
                // which device to read. This mesh spans two devices and the whole point here is to
                // read each one separately, so resolve the device ourselves -- which is what that
                // helper does internally for the unit-mesh case anyway.
                std::vector<uint32_t> receiver_config;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    receivers_per_bank[bank][sender_role],
                    gcb.config_address(),
                    10 * sizeof(uint32_t),
                    receiver_config,
                    CoreType::WORKER);
                ASSERT_GE(receiver_config.size(), 10u);
                EXPECT_EQ(receiver_config[8], expected_sender_virtual.x);
                EXPECT_EQ(receiver_config[9], expected_sender_virtual.y);

                const CoreCoord expected_receiver_phys =
                    device->worker_core_from_logical_core(receivers_per_bank[bank][sender_role]);
                cluster.read_core(
                    sender_state_bytes.data(),
                    sender_state_bytes.size(),
                    tt_cxy_pair(device->id(), expected_sender_virtual),
                    sender_state_addr);
                const auto* sender_state = reinterpret_cast<const DramSenderStateBlock*>(sender_state_bytes.data());
                EXPECT_EQ(sender_state->num_receivers, 1u);
                const auto* receiver_xy =
                    reinterpret_cast<const uint32_t*>(sender_state_bytes.data() + sizeof(DramSenderStateBlock));
                EXPECT_EQ(receiver_xy[0], expected_receiver_phys.x);
                EXPECT_EQ(receiver_xy[1], expected_receiver_phys.y);
            }
        }
    }

    if (!any_sender_placement_differs) {
        log_warning(
            tt::LogTest,
            "This device pair harvests DRAM identically, so no bank resolves to a different physical sender "
            "across the mesh. The per-device assertions above still hold, but this run cannot distinguish "
            "per-device translation from the mesh-wide reference translation it replaced.");
    }
}

// One DRISC sender pushes a per-receiver page to every core in `receiver_cores` and checks each
// landed its own stripe. Parameterized on the receiver set: a receiver's credit slot is its index
// in a row-wise flatten of that set, so a set spanning both rows and columns is what catches a
// sender NOC XY table ordered differently than the receivers' config pages.
void run_one_sender_smoke(distributed::MeshDevice* mesh_device, const CoreRangeSet& receiver_cores) {
    const uint32_t kNumReceivers = receiver_cores.num_cores();
    constexpr uint32_t kPageSize = 64;  // multiple of L1_ALIGNMENT (16 on BH)
    constexpr uint32_t kNumPages = 1;
    constexpr uint32_t kRemoteCBId = 31;

    // Sender: bank 0
    const uint32_t bank_id = 0;
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers = {{bank_id, receiver_cores}};

    // Size: per-receiver fifo. Use 1KB.
    constexpr uint32_t kGcbSize = 1024;
    auto gcb = experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, bank_to_receivers, kGcbSize, BufferType::L1, /*support_multi_receiver_shards=*/true);
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
    auto sender_virtual = mesh_device->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
    const uint64_t drisc_l1_noc_addr_base =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t data_noc_addr = drisc_l1_noc_addr_base + (data_addr - drisc_l1_unreserved);
    MetalContext::instance().get_cluster().write_core(
        pattern.data(),
        pattern.size() * sizeof(uint32_t),
        tt_cxy_pair(mesh_device->build_id(), sender_virtual),
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

    const std::vector<uint32_t> sender_rt_args = receiver_noc_xy_rt_args(gcb, mesh_device);
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
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    // Verify each receiver's L1 slice. Walk the GCB's own receiver order (the same order the
    // sender's NOC XY table and each receiver's credit slot are built from), not a fresh flatten
    // of receiver_cores -- reusing the contract's order is what makes slot r mean receiver r.
    const auto& receivers_vec = experimental::receiver_logical_cores_per_sender(gcb).at(0);
    ASSERT_EQ(receivers_vec.size(), kNumReceivers);
    for (uint32_t r = 0; r < receivers_vec.size(); ++r) {
        std::vector<uint32_t> result;
        slow_dispatch::ReadFromL1(
            *mesh_device, receivers_vec[r], gcb.buffer_address(), kPageSize, result, CoreType::WORKER);
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            uint32_t expected = 0xABCD0000u + r * 0x100u + w;
            EXPECT_EQ(result[w], expected)
                << "Receiver " << r << " (" << receivers_vec[r].str() << ") word " << w << " mismatch (expected 0x"
                << std::hex << expected << ", got 0x" << result[w] << std::dec << ")";
        }
    }

    // Verify pages_sent == pages_acked on DRISC (the barrier in the sender kernel waits for
    // this). Slots are packed: per receiver, pages_sent at +0 uint32, pages_acked at +1 uint32.
    const uint64_t pages_sent_noc_addr = drisc_l1_noc_addr_base + (pages_sent_addr - drisc_l1_unreserved);
    std::vector<uint32_t> pages_buf(2 * kNumReceivers, 0);
    MetalContext::instance().get_cluster().read_core(
        pages_buf.data(),
        pages_buf.size() * sizeof(uint32_t),
        tt_cxy_pair(mesh_device->build_id(), sender_virtual),
        pages_sent_noc_addr);
    for (uint32_t r = 0; r < kNumReceivers; ++r) {
        uint32_t sent = pages_buf[2 * r];
        uint32_t acked = pages_buf[2 * r + 1];
        EXPECT_EQ(sent, acked) << "Pages sent/acked mismatch for receiver " << r << " (sent=" << sent
                               << ", acked=" << acked << ")";
        EXPECT_GT(sent, 0u) << "Sender did not push any pages to receiver " << r;
    }
}

TEST_F(DramSenderGCBFixture, SmokeOneSenderFourReceivers) {
    // 4 receivers in one row, each receiving one 64-byte page.
    run_one_sender_smoke(mesh_device_, CoreRangeSet(CoreRange({0, 0}, {3, 0})));
}

TEST_F(DramSenderGCBFixture, SmokeReceiverGridSpansRowsAndColumns) {
    // Same flow over a 2x2 receiver grid, where row-wise and column-wise flattening disagree:
    // (1,0) and (0,1) trade places. A sender whose NOC XY table is ordered one way while the
    // receivers' pages_sent/pages_acked slots are assigned the other way credits the wrong
    // receiver -- each side then waits on a counter nobody advances, so this hangs rather than
    // returning bad data.
    run_one_sender_smoke(mesh_device_, CoreRangeSet(CoreRange({0, 0}, {1, 1})));
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
    auto sender_virtual = mesh_device_->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
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
    const std::vector<uint32_t> sender_rt_args = receiver_noc_xy_rt_args(gcb, mesh_device_);
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
        slow_dispatch::ReadFromL1(
            *mesh_device_, receivers_vec[r], gcb.buffer_address(), kPageSize, result, CoreType::WORKER);
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
    auto sender_virtual = mesh_device_->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
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
        const std::vector<uint32_t> sender_rt_args = receiver_noc_xy_rt_args(gcb, mesh_device_);
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
        slow_dispatch::ReadFromL1(
            *mesh_device_, CoreCoord{0, 0}, gcb_a.buffer_address(), kPageSize, result, CoreType::WORKER);
        for (uint32_t w = 0; w < kPageSize / sizeof(uint32_t); ++w) {
            uint32_t expected = 0xAAAA0000u + w;
            EXPECT_EQ(result[w], expected) << "Receiver A word " << w;
        }
    }
    // Verify receiver B got pattern B.
    {
        std::vector<uint32_t> result;
        slow_dispatch::ReadFromL1(
            *mesh_device_, CoreCoord{1, 0}, gcb_b.buffer_address(), kPageSize, result, CoreType::WORKER);
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
