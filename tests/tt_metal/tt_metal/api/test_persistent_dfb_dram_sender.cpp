// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-sender PersistentDFB: a programmable DRAM core (Blackhole DRISC) produces into a durable
// remote dataflow buffer that ordinary worker cores consume through the device PersistentDFB class.
//
// These tests exercise the plumbing the Tensor prefetcher's PersistentDFB delivery path is built
// on, without involving the prefetcher itself:
//   * the host stamping a real PersistentDFB sender config page into DRISC L1,
//   * credits crossing L1 address spaces in both directions (sender credit -> worker L1,
//     receiver ack -> DRISC L1),
//   * the sender deriving each receiver's write cursor from that receiver's durable counter, so
//     the cursor survives across programs.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/persistent_dfb.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/persistent_dfb_dram_sender_internal.hpp"
#include "impl/buffers/persistent_dfb_dram_sender_state.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/persistent_dfb.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

class PersistentDfbDramSenderFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (devices_.empty()) {
            GTEST_SKIP() << "No devices available";
        }
        if (!MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
        if (this->slow_dispatch_) {
            // Receivers Attach and read their config slot out of the launch message, which only
            // the fast-dispatch path populates. The DRISC senders are launched slow-dispatch-style
            // regardless (see launch_dram_senders), matching how the Tensor prefetcher does it.
            GTEST_SKIP() << "PersistentDFB receivers require fast dispatch";
        }
    }
};

namespace {

constexpr const char* kSenderKernel = "tests/tt_metal/tt_metal/test_kernels/misc/pdfb_smoke_sender.cpp";
constexpr const char* kReceiverKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_receiver.cpp";

constexpr uint32_t kEntrySize = 256;  // multiple of L1_ALIGNMENT (16 on Blackhole)
constexpr uint32_t kRingDepth = 4;

// Distinct bytes per (receiver, entry, word) so a mis-addressed write shows up as the wrong
// receiver's or wrong slot's data rather than as a hang.
uint32_t pattern_word(uint32_t receiver, uint32_t entry, uint32_t word) {
    return 0xD0FB0000u | (receiver << 12) | (entry << 8) | (word & 0xFFu);
}

// Where the host parks the sender's payload pattern in DRISC L1: above the arena's fixed zone, in
// the region the arena reports as free for a co-resident DRISC kernel.
DeviceAddr drisc_pattern_base(distributed::MeshDevice& mesh_device) {
    return mesh_device.impl().drisc_l1_arena().kernel_working_region_base();
}

uint64_t drisc_noc_addr(DeviceAddr local_addr) {
    return MetalContext::instance().hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM) + local_addr;
}

tt_cxy_pair drisc_cxy(distributed::MeshDevice& mesh_device, const CoreCoord& sender_logical) {
    IDevice* device = mesh_device.get_devices().at(0);
    return tt_cxy_pair(device->id(), device->virtual_core_from_logical_core(sender_logical, CoreType::DRAM));
}

void write_drisc_l1(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& sender_logical,
    DeviceAddr local_addr,
    const std::vector<uint32_t>& words) {
    MetalContext::instance().get_cluster().write_core(
        words.data(),
        words.size() * sizeof(uint32_t),
        drisc_cxy(mesh_device, sender_logical),
        drisc_noc_addr(local_addr));
}

std::vector<uint32_t> read_drisc_l1(
    distributed::MeshDevice& mesh_device, const CoreCoord& sender_logical, DeviceAddr local_addr, uint32_t num_words) {
    std::vector<uint32_t> out(num_words, 0);
    MetalContext::instance().get_cluster().read_core(
        out.data(), num_words * sizeof(uint32_t), drisc_cxy(mesh_device, sender_logical), drisc_noc_addr(local_addr));
    return out;
}

// Preload the payload the sender pushes: receiver r's entry i at
// pattern_base + (r * num_entries + i) * entry_size, matching the smoke kernel's addressing.
// `entry_label` distinguishes successive programs' batches in the pattern bytes.
void preload_pattern(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& sender_logical,
    uint32_t num_entries,
    uint32_t num_receivers,
    uint32_t entry_label) {
    constexpr uint32_t words_per_entry = kEntrySize / sizeof(uint32_t);
    std::vector<uint32_t> pattern(static_cast<size_t>(num_receivers) * num_entries * words_per_entry, 0);
    for (uint32_t r = 0; r < num_receivers; ++r) {
        for (uint32_t i = 0; i < num_entries; ++i) {
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                pattern[(r * num_entries + i) * words_per_entry + w] = pattern_word(r, entry_label + i, w);
            }
        }
    }
    write_drisc_l1(mesh_device, sender_logical, drisc_pattern_base(mesh_device), pattern);
}

// Build and launch the DRISC sender program. DRAM cores are not reachable through the fast-dispatch
// command queue, so the program goes out through the slow-dispatch path even though the rest of the
// test runs fast dispatch -- the same split the Tensor prefetcher uses (TensorPrefetcherManager
// launches its DRISC kernels with force_slow_dispatch while consumers are dispatched normally).
//
// Returned by value so the caller keeps the Program alive until WaitProgramDone; launched
// non-blocking because the senders will park waiting for receiver acks.
Program launch_dram_senders(
    distributed::MeshDevice& mesh_device, experimental::PersistentDFB& pdfb, uint32_t num_entries) {
    Program program = CreateProgram();

    const uint32_t config_page_addr =
        static_cast<uint32_t>(experimental::persistent_dfb_sender_state_drisc_l1_base(pdfb)) +
        persistent_dfb_config_page_offset();
    const uint32_t pattern_base = static_cast<uint32_t>(drisc_pattern_base(mesh_device));

    for (const auto& [sender_logical, _receivers] : experimental::persistent_dfb_sender_receiver_core_mapping(pdfb)) {
        CreateKernel(
            program,
            kSenderKernel,
            sender_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {config_page_addr, num_entries, pattern_base}});
    }

    IDevice* device = mesh_device.get_devices().at(0);
    detail::CompileProgram(device, program, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(device, program, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    return program;
}

// Receiver program: Attach and pop `num_entries`, which acks the senders and lets their drain
// barriers complete.
void run_receivers(distributed::MeshDevice& mesh_device, experimental::PersistentDFB& pdfb, uint32_t num_entries) {
    Program program = CreateProgram();

    const CoreRangeSet receiver_cores = experimental::persistent_dfb_receiver_cores(pdfb);
    const uint8_t persistent_dfb_id = AttachPersistentDFB(program, pdfb, receiver_cores, std::nullopt);
    CreateKernel(
        program,
        kReceiverKernel,
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {persistent_dfb_id, experimental::persistent_dfb_entry_size(pdfb), num_entries, 0u}});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange({0, 0}, {0, 0}), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/false);
    distributed::Finish(mesh_device.mesh_command_queue());
}

// One push/pop cycle: senders fill the ring, receivers drain it, senders' barriers retire.
// num_entries must fit the ring so the senders can publish everything before any receiver runs.
void run_push_and_pop(distributed::MeshDevice& mesh_device, experimental::PersistentDFB& pdfb, uint32_t num_entries) {
    Program sender_program = launch_dram_senders(mesh_device, pdfb, num_entries);
    run_receivers(mesh_device, pdfb, num_entries);
    detail::WaitProgramDone(mesh_device.get_devices().at(0), sender_program, /*read_device_profiler_results=*/false);
}

// Read one entry-sized slot out of a receiver's ring.
std::vector<uint32_t> read_ring_slot(
    distributed::MeshDevice& mesh_device,
    experimental::PersistentDFB& pdfb,
    const CoreCoord& receiver_logical,
    uint32_t slot) {
    std::vector<uint32_t> out;
    detail::ReadFromDeviceL1(
        mesh_device.get_devices().at(0),
        receiver_logical,
        experimental::persistent_dfb_buffer_address(pdfb) + slot * kEntrySize,
        kEntrySize,
        out,
        CoreType::WORKER);
    return out;
}

void expect_ring_slot(
    distributed::MeshDevice& mesh_device,
    experimental::PersistentDFB& pdfb,
    const CoreCoord& receiver_logical,
    uint32_t slot,
    uint32_t expected_receiver_label,
    uint32_t expected_entry_label) {
    const auto got = read_ring_slot(mesh_device, pdfb, receiver_logical, slot);
    ASSERT_EQ(got.size(), kEntrySize / sizeof(uint32_t));
    for (uint32_t w = 0; w < got.size(); ++w) {
        const uint32_t expected = pattern_word(expected_receiver_label, expected_entry_label, w);
        ASSERT_EQ(got[w], expected) << "receiver " << receiver_logical.str() << " ring slot " << slot << " word " << w
                                    << ": expected 0x" << std::hex << expected << ", got 0x" << got[w] << std::dec;
    }
}

// Every receiver acked everything its sender published, read back from that sender's DRISC-L1
// counters. Pairs are strided by 2 * L1_ALIGNMENT with entries_sent first, entries_acked next.
void expect_credits_drained(
    distributed::MeshDevice& mesh_device, experimental::PersistentDFB& pdfb, uint32_t expected_units_per_receiver) {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t stride_words = 2 * l1_alignment / sizeof(uint32_t);
    const uint32_t acked_word = l1_alignment / sizeof(uint32_t);
    const DeviceAddr counters_base = experimental::persistent_dfb_sender_state_drisc_l1_base(pdfb) +
                                     persistent_dfb_config_page_offset() + pdfb.credit_reset_offset();

    const auto& mapping = experimental::persistent_dfb_sender_receiver_core_mapping(pdfb);
    for (const auto& [sender_logical, receivers] : mapping) {
        const uint32_t num_receivers = receivers.num_cores();
        const auto counters = read_drisc_l1(mesh_device, sender_logical, counters_base, num_receivers * stride_words);
        for (uint32_t r = 0; r < num_receivers; ++r) {
            const uint32_t sent = counters[r * stride_words];
            const uint32_t acked = counters[r * stride_words + acked_word];
            EXPECT_EQ(sent, expected_units_per_receiver)
                << "sender " << sender_logical.str() << " receiver " << r << " published " << sent << " credit units";
            EXPECT_EQ(sent, acked) << "sender " << sender_logical.str() << " receiver " << r
                                   << " left credits outstanding (sent=" << sent << ", acked=" << acked << ")";
        }
    }
}

uint32_t credit_units(uint32_t num_entries) {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    return num_entries * kEntrySize / l1_alignment;
}

// Receivers in bank-local slab order, which is the order the sender's NOC XY table uses and hence
// the order pattern_word's receiver label follows.
std::vector<CoreCoord> receivers_in_slab_order(const CoreRangeSet& receivers) {
    return corerange_to_cores(receivers, /*max_cores=*/std::nullopt, /*row_wise=*/true);
}

}  // namespace

TEST_F(PersistentDfbDramSenderFixture, SmokeOneSenderFourReceivers) {
    auto mesh_device = devices_[0];
    constexpr uint32_t kNumReceivers = 4;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    // support_multi_receiver_shards=true forces a single sender for the bank, so all four
    // receivers hang off one DRISC core.
    auto pdfb = experimental::CreatePersistentDFBForTensorPrefetcher(
        *mesh_device,
        {{/*bank_id=*/0, receiver_cores}},
        kEntrySize,
        kRingDepth,
        BufferType::L1,
        /*support_multi_receiver_shards=*/true);
    ASSERT_EQ(experimental::persistent_dfb_sender_core_type(*pdfb), experimental::SenderCoreType::Dram);
    ASSERT_EQ(experimental::persistent_dfb_sender_receiver_core_mapping(*pdfb).size(), 1u);

    const CoreCoord sender_logical = experimental::persistent_dfb_sender_receiver_core_mapping(*pdfb).at(0).first;
    preload_pattern(*mesh_device, sender_logical, kRingDepth, kNumReceivers, /*entry_label=*/0);
    run_push_and_pop(*mesh_device, *pdfb, kRingDepth);

    const auto receivers = receivers_in_slab_order(receiver_cores);
    for (uint32_t r = 0; r < receivers.size(); ++r) {
        for (uint32_t i = 0; i < kRingDepth; ++i) {
            expect_ring_slot(*mesh_device, *pdfb, receivers[r], /*slot=*/i, /*receiver_label=*/r, /*entry_label=*/i);
        }
    }
    expect_credits_drained(*mesh_device, *pdfb, credit_units(kRingDepth));
}

TEST_F(PersistentDfbDramSenderFixture, CursorPersistsAcrossPrograms) {
    // The sender stores no write cursor: it derives each receiver's position from that receiver's
    // durable entries_sent counter. A second program must therefore resume mid-ring rather than
    // restart at slot 0.
    auto mesh_device = devices_[0];
    constexpr uint32_t kNumReceivers = 2;
    constexpr uint32_t kBatch = 2;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    auto pdfb = experimental::CreatePersistentDFBForTensorPrefetcher(
        *mesh_device,
        {{/*bank_id=*/0, receiver_cores}},
        kEntrySize,
        kRingDepth,
        BufferType::L1,
        /*support_multi_receiver_shards=*/true);
    const CoreCoord sender_logical = experimental::persistent_dfb_sender_receiver_core_mapping(*pdfb).at(0).first;

    preload_pattern(*mesh_device, sender_logical, kBatch, kNumReceivers, /*entry_label=*/0);
    run_push_and_pop(*mesh_device, *pdfb, kBatch);

    // Second batch carries different bytes so landing on slots 0-1 again would be visible.
    preload_pattern(*mesh_device, sender_logical, kBatch, kNumReceivers, /*entry_label=*/kBatch);
    run_push_and_pop(*mesh_device, *pdfb, kBatch);

    const auto receivers = receivers_in_slab_order(receiver_cores);
    for (uint32_t r = 0; r < receivers.size(); ++r) {
        for (uint32_t i = 0; i < 2 * kBatch; ++i) {
            expect_ring_slot(*mesh_device, *pdfb, receivers[r], /*slot=*/i, /*receiver_label=*/r, /*entry_label=*/i);
        }
    }
    expect_credits_drained(*mesh_device, *pdfb, credit_units(2 * kBatch));
}

TEST_F(PersistentDfbDramSenderFixture, DualSendersSplitBankReceivers) {
    // Receiver-contiguous mode lets one bank be driven by two DRISC cores, each owning a disjoint
    // half of the bank's receivers -- which is exactly what PersistentDFB requires of a sender's
    // receiver set.
    auto mesh_device = devices_[0];
    constexpr uint32_t kNumReceivers = 4;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    auto pdfb = experimental::CreatePersistentDFBForTensorPrefetcher(
        *mesh_device,
        {{/*bank_id=*/0, receiver_cores}},
        kEntrySize,
        kRingDepth,
        BufferType::L1,
        /*support_multi_receiver_shards=*/false);
    const auto& mapping = experimental::persistent_dfb_sender_receiver_core_mapping(*pdfb);
    ASSERT_EQ(mapping.size(), 2u) << "expected the bank's receivers to be split across two DRISC senders";
    ASSERT_EQ(mapping.at(0).second.num_cores(), 2u);
    ASSERT_EQ(mapping.at(1).second.num_cores(), 2u);

    // Each sender addresses its own receivers as local indices 0..n-1, so the pattern is preloaded
    // per sender with labels restarting at 0.
    for (const auto& [sender_logical, receivers] : mapping) {
        preload_pattern(*mesh_device, sender_logical, kRingDepth, receivers.num_cores(), /*entry_label=*/0);
    }
    run_push_and_pop(*mesh_device, *pdfb, kRingDepth);

    for (const auto& [_sender_logical, receivers] : mapping) {
        const auto local_receivers = receivers_in_slab_order(receivers);
        for (uint32_t r = 0; r < local_receivers.size(); ++r) {
            for (uint32_t i = 0; i < kRingDepth; ++i) {
                expect_ring_slot(
                    *mesh_device, *pdfb, local_receivers[r], /*slot=*/i, /*receiver_label=*/r, /*entry_label=*/i);
            }
        }
    }
    expect_credits_drained(*mesh_device, *pdfb, credit_units(kRingDepth));
}

TEST_F(PersistentDfbDramSenderFixture, RejectsDuplicateBank) {
    auto mesh_device = devices_[0];
    const CoreRangeSet first(CoreRange({0, 0}, {0, 0}));
    const CoreRangeSet second(CoreRange({1, 0}, {1, 0}));
    EXPECT_ANY_THROW(experimental::CreatePersistentDFBForTensorPrefetcher(
        *mesh_device, {{0, first}, {0, second}}, kEntrySize, kRingDepth));
}

}  // namespace tt::tt_metal
