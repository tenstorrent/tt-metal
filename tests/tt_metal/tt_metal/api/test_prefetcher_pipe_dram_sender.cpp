// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-sender PrefetcherPipes: a programmable DRAM core (Blackhole DRISC) produces into a durable
// remote dataflow buffer that ordinary worker cores consume through the device PrefetcherPipe
// class.
//
// These tests exercise the plumbing the Tensor prefetcher's PrefetcherPipe delivery path is built
// on, without involving the prefetcher itself:
//   * the host stamping a real PrefetcherPipe sender config page into DRISC L1,
//   * credits crossing L1 address spaces in both directions (sender credit -> worker L1,
//     receiver ack -> DRISC L1),
//   * the sender deriving each receiver's write cursor from that receiver's durable counter, so
//     the cursor survives across programs,
//   * per-sender DRISC L1 placement: a pipe reserves its config page on its own sender core, so a
//     whole set of pipes costs the small DRISC zone one offset.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

// Slow dispatch, like DramSenderGCBFixture: BlackholeSingleCardFixture is slow-dispatch-only.
// Slow dispatch materializes a program's PrefetcherPipe dense index per core in
// ConfigureDeviceWithProgram, so the DRISC sender and the worker receivers can share one Program.
class PrefetcherPipeDramSenderFixture : public BlackholeSingleCardFixture {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;
        }
        mesh_device_ = devices_[0].get();
        if (!MetalContext::instance(mesh_device_->impl().get_context_id())
                 .hal()
                 .has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }

    distributed::MeshDevice* mesh_device_{};
};

namespace {

constexpr const char* kSenderKernel = "tests/tt_metal/tt_metal/test_kernels/misc/prefetcher_pipe_dram_smoke_sender.cpp";
constexpr const char* kReceiverKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp";

constexpr uint32_t kEntrySize = 256;  // multiple of L1_ALIGNMENT (16 on Blackhole)
constexpr uint32_t kRingDepth = 4;

// A Tensor-prefetcher delivery target: the per-bank pipe groups the factory returns, plus the
// bank-major flattening the rest of this file drives the senders through. The
// TensorPrefetcherPipes wrapper one layer up derives the same flattening from the same groups.
struct PipeSet {
    std::vector<experimental::TensorPrefetcherBankPipes> banks;
    // One entry per pipe, bank-major: that pipe's sender core and its receivers.
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    std::vector<std::shared_ptr<experimental::PrefetcherPipe>> pipes;
};

PipeSet make_pipe_set(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool dual_senders_per_bank,
    uint32_t entry_size = kEntrySize,
    uint32_t num_entries = kRingDepth) {
    PipeSet set;
    set.banks = experimental::CreatePrefetcherPipesForTensorPrefetcher(
        mesh_device,
        bank_to_receivers,
        entry_size,
        num_entries,
        BufferType::L1,
        /*support_multi_receiver_shards=*/!dual_senders_per_bank);
    for (const auto& bank : set.banks) {
        for (const auto& pipe : bank.pipes) {
            set.mapping.emplace_back(
                experimental::prefetcher_pipe_sender_core(*pipe), experimental::prefetcher_pipe_receiver_cores(*pipe));
            set.pipes.push_back(pipe);
        }
    }
    return set;
}

// Distinct bytes per (receiver, entry, word) so a mis-addressed write shows up as the wrong
// receiver's or wrong slot's data rather than as a hang.
uint32_t pattern_word(uint32_t receiver, uint32_t entry, uint32_t word) {
    return 0xD0FB0000u | (receiver << 12) | (entry << 8) | (word & 0xFFu);
}

ContextId context_id_of(distributed::MeshDevice& mesh_device) { return mesh_device.impl().get_context_id(); }

// Where the host parks the sender's payload pattern in DRISC L1: above the arena's fixed zone, in
// the region the arena reports as free for a co-resident DRISC kernel.
DeviceAddr drisc_pattern_base(distributed::MeshDevice& mesh_device) {
    return mesh_device.impl().drisc_l1_arena().kernel_working_region_base();
}

uint64_t drisc_noc_addr(distributed::MeshDevice& mesh_device, DeviceAddr local_addr) {
    return MetalContext::instance(context_id_of(mesh_device)).hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM) +
           local_addr;
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
    MetalContext::instance(context_id_of(mesh_device))
        .get_cluster()
        .write_core(
            words.data(),
            words.size() * sizeof(uint32_t),
            drisc_cxy(mesh_device, sender_logical),
            drisc_noc_addr(mesh_device, local_addr));
}

std::vector<uint32_t> read_drisc_l1(
    distributed::MeshDevice& mesh_device, const CoreCoord& sender_logical, DeviceAddr local_addr, uint32_t num_words) {
    std::vector<uint32_t> out(num_words, 0);
    MetalContext::instance(context_id_of(mesh_device))
        .get_cluster()
        .read_core(
            out.data(),
            num_words * sizeof(uint32_t),
            drisc_cxy(mesh_device, sender_logical),
            drisc_noc_addr(mesh_device, local_addr));
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
    uint32_t entry_label,
    uint32_t entry_size = kEntrySize) {
    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
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

// One push/pop cycle. The DRISC senders and the worker receivers go out in a single Program: a
// receiver reads its Attach slot from the per-core dense index, which slow dispatch writes in
// ConfigureDeviceWithProgram, while the senders take their config page address from DRISC L1 and
// never Attach at all.
//
// num_entries must fit the ring so a sender can publish its whole batch even if its receivers
// start late.
void run_push_and_pop(
    distributed::MeshDevice& mesh_device, const PipeSet& set, uint32_t num_entries, uint32_t entry_size = kEntrySize) {
    Program program = CreateProgram();

    const uint32_t pattern_base = static_cast<uint32_t>(drisc_pattern_base(mesh_device));

    for (size_t s = 0; s < set.pipes.size(); ++s) {
        experimental::PrefetcherPipe& pipe = *set.pipes[s];
        const auto config_page_addr = static_cast<uint32_t>(experimental::sender_state_drisc_l1_base(pipe));
        const uint8_t pipe_id = experimental::AttachPrefetcherPipe(program, pipe, set.mapping[s].second, entry_size);
        CreateKernel(
            program,
            kSenderKernel,
            set.mapping[s].first,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {config_page_addr, num_entries, pattern_base, entry_size}});
        // One kernel per pipe rather than one for all receivers: the pipe id is a compile-time arg
        // of the shared receiver kernel, and each pipe's receivers hold a different id.
        CreateKernel(
            program,
            kReceiverKernel,
            set.mapping[s].second,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {pipe_id, entry_size, num_entries, 0u}});
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange({0, 0}, {0, 0}), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/false);
    distributed::Finish(mesh_device.mesh_command_queue());
}

// Read one entry-sized slot out of a receiver's ring.
std::vector<uint32_t> read_ring_slot(
    distributed::MeshDevice& mesh_device,
    experimental::PrefetcherPipe& pipe,
    const CoreCoord& receiver_logical,
    uint32_t slot,
    uint32_t entry_size) {
    std::vector<uint32_t> out;
    detail::ReadFromDeviceL1(
        mesh_device.get_devices().at(0),
        receiver_logical,
        pipe.buffer_address() + slot * entry_size,
        entry_size,
        out,
        CoreType::WORKER);
    return out;
}

void expect_ring_slot(
    distributed::MeshDevice& mesh_device,
    experimental::PrefetcherPipe& pipe,
    const CoreCoord& receiver_logical,
    uint32_t slot,
    uint32_t expected_receiver_label,
    uint32_t expected_entry_label,
    uint32_t entry_size = kEntrySize) {
    const auto got = read_ring_slot(mesh_device, pipe, receiver_logical, slot, entry_size);
    ASSERT_EQ(got.size(), entry_size / sizeof(uint32_t));
    for (uint32_t w = 0; w < got.size(); ++w) {
        const uint32_t expected = pattern_word(expected_receiver_label, expected_entry_label, w);
        ASSERT_EQ(got[w], expected) << "receiver " << receiver_logical.str() << " ring slot " << slot << " word " << w
                                    << ": expected 0x" << std::hex << expected << ", got 0x" << got[w] << std::dec;
    }
}

// Every receiver acked everything its sender published, read back from that sender's DRISC-L1
// counters. Pairs are strided by 2 * L1_ALIGNMENT with entries_sent first, entries_acked next.
void expect_credits_drained(
    distributed::MeshDevice& mesh_device, const PipeSet& set, uint32_t expected_units_per_receiver) {
    const uint32_t l1_alignment =
        MetalContext::instance(context_id_of(mesh_device)).hal().get_alignment(HalMemType::L1);
    const uint32_t stride_words = 2 * l1_alignment / sizeof(uint32_t);
    const uint32_t acked_word = l1_alignment / sizeof(uint32_t);

    for (size_t s = 0; s < set.pipes.size(); ++s) {
        const auto& [sender_logical, receivers] = set.mapping[s];
        const uint32_t num_receivers = receivers.num_cores();
        const DeviceAddr counters_base =
            experimental::sender_state_drisc_l1_base(*set.pipes[s]) + set.pipes[s]->credit_reset_offset();
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

// Credit is counted in L1_ALIGNMENT-byte units, so a run's expected credit is the bytes it moved --
// payload plus any pad the sender published to snap onto a new entry grid.
uint32_t credit_units(distributed::MeshDevice& mesh_device, uint32_t bytes) {
    const uint32_t l1_alignment =
        MetalContext::instance(context_id_of(mesh_device)).hal().get_alignment(HalMemType::L1);
    return bytes / l1_alignment;
}

// Receivers in bank-local slab order, which is the order the sender's NOC XY table uses and hence
// the order pattern_word's receiver label follows.
std::vector<CoreCoord> receivers_in_slab_order(const CoreRangeSet& receivers) {
    return corerange_to_cores(receivers, /*max_cores=*/std::nullopt, /*row_wise=*/true);
}

}  // namespace

TEST_F(PrefetcherPipeDramSenderFixture, SmokeOneSenderFourReceivers) {
    constexpr uint32_t kNumReceivers = 4;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    // dual_senders_per_bank=false forces a single sender for the bank, so all four receivers hang
    // off one DRISC core and the set collapses to one pipe.
    const PipeSet set =
        make_pipe_set(*mesh_device_, {{/*bank_id=*/0, receiver_cores}}, /*dual_senders_per_bank=*/false);
    ASSERT_EQ(set.banks.size(), 1u);
    ASSERT_EQ(set.banks[0].bank_id, 0u);
    ASSERT_EQ(set.pipes.size(), 1u);
    ASSERT_EQ(experimental::prefetcher_pipe_sender_core_type(*set.pipes[0]), experimental::SenderCoreType::Dram);

    const CoreCoord sender_logical = set.mapping.at(0).first;
    preload_pattern(*mesh_device_, sender_logical, kRingDepth, kNumReceivers, /*entry_label=*/0);
    run_push_and_pop(*mesh_device_, set, kRingDepth);

    const auto receivers = receivers_in_slab_order(receiver_cores);
    for (uint32_t r = 0; r < receivers.size(); ++r) {
        for (uint32_t i = 0; i < kRingDepth; ++i) {
            expect_ring_slot(
                *mesh_device_, *set.pipes[0], receivers[r], /*slot=*/i, /*receiver_label=*/r, /*entry_label=*/i);
        }
    }
    expect_credits_drained(*mesh_device_, set, credit_units(*mesh_device_, kRingDepth * kEntrySize));
}

TEST_F(PrefetcherPipeDramSenderFixture, CursorPersistsAcrossPrograms) {
    // The sender stores no write cursor: it derives each receiver's position from that receiver's
    // durable entries_sent counter. A second program must therefore resume mid-ring rather than
    // restart at slot 0.
    constexpr uint32_t kNumReceivers = 2;
    constexpr uint32_t kBatch = 2;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    const PipeSet set =
        make_pipe_set(*mesh_device_, {{/*bank_id=*/0, receiver_cores}}, /*dual_senders_per_bank=*/false);
    const CoreCoord sender_logical = set.mapping.at(0).first;

    preload_pattern(*mesh_device_, sender_logical, kBatch, kNumReceivers, /*entry_label=*/0);
    run_push_and_pop(*mesh_device_, set, kBatch);

    // Second batch carries different bytes so landing on slots 0-1 again would be visible.
    preload_pattern(*mesh_device_, sender_logical, kBatch, kNumReceivers, /*entry_label=*/kBatch);
    run_push_and_pop(*mesh_device_, set, kBatch);

    const auto receivers = receivers_in_slab_order(receiver_cores);
    for (uint32_t r = 0; r < receivers.size(); ++r) {
        for (uint32_t i = 0; i < 2 * kBatch; ++i) {
            expect_ring_slot(
                *mesh_device_, *set.pipes[0], receivers[r], /*slot=*/i, /*receiver_label=*/r, /*entry_label=*/i);
        }
    }
    expect_credits_drained(*mesh_device_, set, credit_units(*mesh_device_, 2 * kBatch * kEntrySize));
}

TEST_F(PrefetcherPipeDramSenderFixture, DualSendersSplitBankReceivers) {
    // Receiver-contiguous mode lets one bank be driven by two DRISC cores, each owning a disjoint
    // half of the bank's receivers -- which is what makes them two independent one-sender pipes.
    constexpr uint32_t kNumReceivers = 4;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    const PipeSet set = make_pipe_set(*mesh_device_, {{/*bank_id=*/0, receiver_cores}}, /*dual_senders_per_bank=*/true);
    ASSERT_EQ(set.banks.size(), 1u);
    ASSERT_EQ(set.banks[0].pipes.size(), 2u) << "expected the bank's receivers to be split across two DRISC senders";
    ASSERT_EQ(set.mapping.at(0).second.num_cores(), 2u);
    ASSERT_EQ(set.mapping.at(1).second.num_cores(), 2u);
    ASSERT_NE(set.mapping.at(0).first, set.mapping.at(1).first);

    // Each sender addresses its own receivers as local indices 0..n-1, so the pattern is preloaded
    // per sender with labels restarting at 0.
    for (const auto& [sender_logical, receivers] : set.mapping) {
        preload_pattern(*mesh_device_, sender_logical, kRingDepth, receivers.num_cores(), /*entry_label=*/0);
    }
    run_push_and_pop(*mesh_device_, set, kRingDepth);

    for (size_t s = 0; s < set.mapping.size(); ++s) {
        const auto local_receivers = receivers_in_slab_order(set.mapping[s].second);
        for (uint32_t r = 0; r < local_receivers.size(); ++r) {
            for (uint32_t i = 0; i < kRingDepth; ++i) {
                expect_ring_slot(
                    *mesh_device_,
                    *set.pipes[s],
                    local_receivers[r],
                    /*slot=*/i,
                    /*receiver_label=*/r,
                    /*entry_label=*/i);
            }
        }
    }
    expect_credits_drained(*mesh_device_, set, credit_units(*mesh_device_, kRingDepth * kEntrySize));
}

TEST_F(PrefetcherPipeDramSenderFixture, PipesOnDistinctSendersShareOneDriscOffset) {
    // A pipe reserves its config page on its own sender core, so a whole set of one-sender pipes
    // costs the small DRISC zone one page rather than one page per pipe. Anything a given sender
    // core would also see -- a second range on that core, or a uniform GCB-style range every bank
    // sees -- still has to go somewhere else.
    const PipeSet split_bank = make_pipe_set(
        *mesh_device_,
        {{/*bank_id=*/0, CoreRangeSet(CoreRange({0, 0}, {1, 0}))}},
        /*dual_senders_per_bank=*/true);
    ASSERT_EQ(split_bank.pipes.size(), 2u) << "expected the bank's receivers to be split across two DRISC senders";
    const PipeSet other_bank = make_pipe_set(
        *mesh_device_,
        {{/*bank_id=*/1, CoreRangeSet(CoreRange({2, 0}, {3, 0}))}},
        /*dual_senders_per_bank=*/false);
    ASSERT_EQ(other_bank.pipes.size(), 1u);

    const DeviceAddr shared_base = experimental::sender_state_drisc_l1_base(*split_bank.pipes[0]);
    EXPECT_EQ(experimental::sender_state_drisc_l1_base(*split_bank.pipes[1]), shared_base)
        << "a bank's two senders are distinct DRISC cores and may hold the same offset";
    EXPECT_EQ(experimental::sender_state_drisc_l1_base(*other_bank.pipes[0]), shared_base);

    auto& arena = mesh_device_->impl().drisc_l1_arena();
    const uint32_t l1_alignment =
        MetalContext::instance(context_id_of(*mesh_device_)).hal().get_alignment(HalMemType::L1);
    const uint32_t page_size = split_bank.pipes[0]->config_page_size();
    EXPECT_NE(arena.allocate_on(split_bank.mapping[0].first, page_size, l1_alignment)->addr(), shared_base)
        << "a second range on a live sender's own core must not overlap its config page";
    EXPECT_NE(arena.allocate(page_size, l1_alignment)->addr(), shared_base)
        << "a uniform range is reserved on every bank, so it must clear every per-core page";
}

TEST_F(PrefetcherPipeDramSenderFixture, RejectsDuplicateBank) {
    const CoreRangeSet first(CoreRange({0, 0}, {0, 0}));
    const CoreRangeSet second(CoreRange({1, 0}, {1, 0}));
    EXPECT_ANY_THROW(make_pipe_set(*mesh_device_, {{0, first}, {0, second}}, /*dual_senders_per_bank=*/true));
}

TEST_F(PrefetcherPipeDramSenderFixture, AttachRejectsEntrySizeThatDoesNotDivideRing) {
    // A DRAM sender addresses its ring in whole entries, with no trailing-gap credit. An entry size
    // the ring is not a multiple of would put it on a different grid than the receivers after the
    // first wrap. Any other size is fine: the sender snaps its cursor and publishes pad credits.
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {1, 0}));
    const PipeSet set =
        make_pipe_set(*mesh_device_, {{/*bank_id=*/0, receiver_cores}}, /*dual_senders_per_bank=*/false);
    const uint32_t ring_size = set.pipes[0]->ring_size();

    Program program = CreateProgram();
    // L1-aligned and well inside the ring, so only the divisibility rule can reject it.
    constexpr uint32_t kRingIndivisibleEntrySize = 48;
    ASSERT_NE(ring_size % kRingIndivisibleEntrySize, 0u);
    EXPECT_ANY_THROW(
        experimental::AttachPrefetcherPipe(program, *set.pipes[0], receiver_cores, kRingIndivisibleEntrySize));
    EXPECT_NO_THROW(experimental::AttachPrefetcherPipe(program, *set.pipes[0], receiver_cores, kEntrySize / 2));
    EXPECT_NO_THROW(experimental::AttachPrefetcherPipe(program, *set.pipes[0], receiver_cores, kEntrySize));
}

TEST_F(PrefetcherPipeDramSenderFixture, BlockSizeChangeAcrossPrograms) {
    // Two consumers of one pipe set that read different block sizes. The ring size is fixed at
    // creation and both sizes divide it, so the DRAM sender snaps its derived cursor onto the new
    // grid and publishes the skipped bytes as pad credits, which the receivers' own resize
    // consumes. Neither endpoint stores a cursor, so the two snaps have to agree by arithmetic.
    constexpr uint32_t kNumReceivers = 2;
    constexpr uint32_t kFirstEntrySize = kEntrySize;
    constexpr uint32_t kSecondEntrySize = 2 * kEntrySize;
    // Leaves the cursor at 3/4 of the ring, which is not on the second size's grid: the snap has to
    // wrap it to zero and credit the quarter ring it skips.
    constexpr uint32_t kFirstBatch = 3;
    constexpr uint32_t kSecondBatch = 2;
    constexpr uint32_t kRingBytes = kEntrySize * kRingDepth;
    static_assert(kRingBytes % kFirstEntrySize == 0);
    static_assert(kRingBytes % kSecondEntrySize == 0);
    static_assert(kFirstBatch * kFirstEntrySize < kRingBytes);
    constexpr uint32_t kPadBytes = kRingBytes - kFirstBatch * kFirstEntrySize;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));

    const PipeSet set =
        make_pipe_set(*mesh_device_, {{/*bank_id=*/0, receiver_cores}}, /*dual_senders_per_bank=*/false);
    const CoreCoord sender_logical = set.mapping.at(0).first;

    preload_pattern(*mesh_device_, sender_logical, kFirstBatch, kNumReceivers, /*entry_label=*/0, kFirstEntrySize);
    run_push_and_pop(*mesh_device_, set, kFirstBatch, kFirstEntrySize);

    preload_pattern(
        *mesh_device_, sender_logical, kSecondBatch, kNumReceivers, /*entry_label=*/kFirstBatch, kSecondEntrySize);
    run_push_and_pop(*mesh_device_, set, kSecondBatch, kSecondEntrySize);

    const auto receivers = receivers_in_slab_order(receiver_cores);
    for (uint32_t r = 0; r < receivers.size(); ++r) {
        // The second batch starts over at ring offset 0: the snap wrapped the cursor rather than
        // leaving a partial entry at the end of the ring.
        for (uint32_t i = 0; i < kSecondBatch; ++i) {
            expect_ring_slot(
                *mesh_device_,
                *set.pipes[0],
                receivers[r],
                /*slot=*/i,
                /*receiver_label=*/r,
                /*entry_label=*/kFirstBatch + i,
                kSecondEntrySize);
        }
    }
    // Pad bytes are credited like payload, so sent == acked only if both endpoints snapped by the
    // same amount.
    expect_credits_drained(
        *mesh_device_,
        set,
        credit_units(*mesh_device_, kFirstBatch * kFirstEntrySize + kPadBytes + kSecondBatch * kSecondEntrySize));
}

}  // namespace tt::tt_metal
