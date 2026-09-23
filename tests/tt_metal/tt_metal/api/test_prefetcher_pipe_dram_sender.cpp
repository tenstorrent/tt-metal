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
//   * the sender keeping each receiver's write cursor in that receiver's SENT credit slot in the
//     config page, so the cursor survives across programs,
//   * per-sender DRISC L1 placement: a pipe reserves its config page on its own sender core, so a
//     whole set of pipes costs the small DRISC zone one offset.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "tests/tt_metal/tt_metal/api/dram_sender_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "hostdev/remote_dfb_config_layout.h"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

class PrefetcherPipeDramSenderFixture : public DramSenderFixture {};

namespace {

constexpr const char* kSenderKernel = "tests/tt_metal/tt_metal/test_kernels/misc/prefetcher_pipe_dram_smoke_sender.cpp";
constexpr const char* kReceiverKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp";

constexpr uint32_t kEntrySize = 256;  // multiple of L1_ALIGNMENT (16 on Blackhole)
constexpr uint32_t kRingDepth = 4;

// A Tensor-prefetcher delivery target: the bank-major pipe list the factory returns, plus the
// sender topology read back out of it through prefetcher_pipe_sender_receiver_mapping -- the same
// helper a consumer one layer up reads it through, so this test cannot disagree with it about pipe
// order. `pipes` and `mapping` index alongside each other.
struct PipeSet {
    // One entry per pipe, bank-major: that pipe's sender core and its receivers.
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    // Declared before the pipes so it is destroyed after them: a space must outlive its pipes.
    std::optional<experimental::PrefetcherPipeSpace> space;
    std::vector<std::shared_ptr<experimental::PrefetcherPipe>> pipes;
};

PipeSet make_pipe_set(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool dual_senders_per_bank,
    uint32_t entry_size = kEntrySize,
    uint32_t num_entries = kRingDepth) {
    PipeSet set;
    CoreRangeSet receiver_domain;
    uint32_t max_receivers = 0;
    for (const auto& [_bank, receivers] : bank_to_receivers) {
        receiver_domain = receiver_domain.merge(receivers);
        max_receivers = std::max(max_receivers, receivers.num_cores());
    }
    set.space.emplace(experimental::CreatePrefetcherPipeSpace(
        mesh_device,
        experimental::PrefetcherPipeSpaceConfig{
            .sender_cores = {},
            .num_dram_senders = static_cast<uint32_t>(2 * bank_to_receivers.size()),
            .receiver_domain = receiver_domain,
            .ring_size = entry_size * num_entries,
            .max_receivers_per_pipe = max_receivers,
            .buffer_type = BufferType::L1}));
    set.pipes = experimental::CreatePrefetcherPipesForTensorPrefetcher(
        *set.space,
        bank_to_receivers,
        /*support_multi_receiver_shards=*/!dual_senders_per_bank);
    set.mapping = experimental::prefetcher_pipe_sender_receiver_mapping(set.pipes);
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

// One push/pop cycle. The DRISC senders first fill at most one ring, then the worker receiver
// Program drains it. Receivers get their generated pipe slots through ProgramSpec/ProgramRunArgs;
// DRAM cores remain a legacy-program detail because Metal 2.0 WorkUnits target worker nodes.
//
// num_entries must fit the ring so a sender can publish its whole batch even if its receivers
// start late.
void run_push_and_pop(
    distributed::MeshDevice& mesh_device, const PipeSet& set, uint32_t num_entries, uint32_t entry_size = kEntrySize) {
    std::vector<experimental::PrefetcherPipeParamName> pipe_names;
    std::vector<experimental::PrefetcherPipeParameter> pipe_parameters;
    experimental::ProgramRunArgs run_args;
    for (size_t s = 0; s < set.pipes.size(); ++s) {
        experimental::PrefetcherPipeParamName name{fmt::format("pipe_{}", s)};
        pipe_names.push_back(name);
        pipe_parameters.push_back(experimental::PrefetcherPipeParameter{
            .unique_id = name,
            .receivers = set.pipes[s]->receiver_cores(),
            .ring_size = set.pipes[s]->ring_size(),
            .entry_size = entry_size});
        run_args.advanced_options.prefetcher_pipe_args.emplace(name, experimental::PrefetcherPipeArgument{*set.pipes[s]});
    }
    experimental::KernelSpec receiver{
        .unique_id = experimental::KernelSpecName{"receiver"}, .source = std::filesystem::path{kReceiverKernel}};
    receiver.advanced_options.prefetcher_pipe_bindings = {{.pipe_parameter_names = pipe_names, .accessor_name = "in"}};
    receiver.compile_time_args = {{"num_entries", num_entries}};
    receiver.hw_config =
        experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    experimental::ProgramSpec spec{
        .name = "dram_sender_pipe_receiver",
        .kernels = {std::move(receiver)},
        .work_units =
            {{.name = "receiver_wu",
              .kernels = {experimental::KernelSpecName{"receiver"}},
              .target_nodes = experimental::prefetcher_pipe_receiver_cores(set.pipes)}},
        .advanced_options = {.prefetcher_pipe_parameters = std::move(pipe_parameters)}};
    Program receiver_program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::SetProgramRunArgs(receiver_program, run_args);

    const uint32_t pattern_base = static_cast<uint32_t>(drisc_pattern_base(mesh_device));
    Program sender_program = CreateProgram();
    for (size_t s = 0; s < set.pipes.size(); ++s) {
        experimental::PrefetcherPipe& pipe = *set.pipes[s];
        const auto config_page_addr = static_cast<uint32_t>(experimental::sender_state_drisc_l1_base(pipe));
        CreateKernel(
            sender_program,
            kSenderKernel,
            set.mapping[s].first,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {config_page_addr, num_entries, pattern_base, entry_size}});
    }

    // The sender consumes receiver credits after the first entry, while the receiver waits for
    // sender data. Queue both programs before waiting so those two sides can make progress
    // concurrently under slow dispatch.
    experimental::DispatchContext::get().enable_asynchronous_slow_dispatch(&mesh_device);
    {
        distributed::MeshWorkload sender_workload;
        sender_workload.add_program(distributed::MeshCoordinateRange({0, 0}, {0, 0}), std::move(sender_program));
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), sender_workload, /*blocking=*/false);

        distributed::MeshWorkload receiver_workload;
        receiver_workload.add_program(distributed::MeshCoordinateRange({0, 0}, {0, 0}), std::move(receiver_program));
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), receiver_workload, /*blocking=*/false);
        distributed::Finish(mesh_device.mesh_command_queue());
    }
    experimental::DispatchContext::get().disable_asynchronous_slow_dispatch(&mesh_device);
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
    uint32_t receiver_label,
    uint32_t entry_label,
    uint32_t entry_size = kEntrySize) {
    const auto got = read_ring_slot(mesh_device, pipe, receiver_logical, slot, entry_size);
    ASSERT_EQ(got.size(), entry_size / sizeof(uint32_t));
    for (uint32_t w = 0; w < got.size(); ++w) {
        const uint32_t expected = pattern_word(receiver_label, entry_label, w);
        ASSERT_EQ(got[w], expected) << "receiver " << receiver_logical.str() << " ring slot " << slot << " word " << w
                                    << ": expected 0x" << std::hex << expected << ", got 0x" << got[w] << std::dec;
    }
}

// Every receiver acked everything its sender published, read back from that sender's DRISC-L1
// credit blocks. The page header words name both block bases, and within a block a receiver's slot
// is one L1_ALIGNMENT -- a DRAM-sender pipe reserves one credit lane per receiver.
void expect_credits_drained(
    distributed::MeshDevice& mesh_device, const PipeSet& set, uint32_t expected_units_per_receiver) {
    const uint32_t l1_alignment =
        MetalContext::instance(context_id_of(mesh_device)).hal().get_alignment(HalMemType::L1);
    const uint32_t stride_words = l1_alignment / sizeof(uint32_t);

    for (size_t s = 0; s < set.pipes.size(); ++s) {
        const auto& [sender_logical, receivers] = set.mapping[s];
        const uint32_t num_receivers = receivers.num_cores();
        const DeviceAddr page_base = experimental::sender_state_drisc_l1_base(*set.pipes[s]);
        const auto header = read_drisc_l1(mesh_device, sender_logical, page_base, PREFETCHER_PIPE_CONFIG_HEADER_WORDS);
        const auto read_block = [&](uint32_t base_word) {
            return read_drisc_l1(
                mesh_device, sender_logical, page_base + header[base_word], num_receivers * stride_words);
        };
        const auto sent_block = read_block(PREFETCHER_PIPE_CFG_PAGES_SENT_OFFSET);
        const auto acked_block = read_block(PREFETCHER_PIPE_CFG_PAGES_ACKED_OFFSET);
        for (uint32_t r = 0; r < num_receivers; ++r) {
            const uint32_t sent = sent_block[r * stride_words];
            const uint32_t acked = acked_block[r * stride_words];
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
    ASSERT_EQ(set.pipes.size(), 1u);
    ASSERT_EQ(set.pipes[0]->sender_core().x, 0u) << "a pipe's DRAM-logical sender x is the bank it is fed from";
    ASSERT_EQ(set.pipes[0]->sender_core_type(), experimental::SenderCoreType::Dram);

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
    // The sender's write cursors live in its config page, beside the credit counters, and no
    // Attach resets either. A second program must therefore resume mid-ring rather than restart at
    // slot 0.
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
    ASSERT_EQ(set.pipes.size(), 2u) << "expected the bank's receivers to be split across two DRISC senders";
    ASSERT_EQ(set.mapping.at(0).second.num_cores(), 2u);
    ASSERT_EQ(set.mapping.at(1).second.num_cores(), 2u);
    ASSERT_NE(set.mapping.at(0).first, set.mapping.at(1).first);
    // Both senders drive this bank, and the leading pipe owns ceil(n/2) of its receivers. Each
    // pipe carries the bank-local slab base that split gives it, so a caller passing the pipes on
    // is free to reorder them.
    EXPECT_EQ(set.pipes.at(0)->sender_core().x, set.pipes.at(1)->sender_core().x);
    EXPECT_EQ(set.pipes.at(0)->receiver_cores().num_cores(), (kNumReceivers + 1) / 2);
    EXPECT_EQ(set.pipes.at(0)->impl().recv_index_base(), 0u);
    EXPECT_EQ(set.pipes.at(1)->impl().recv_index_base(), (kNumReceivers + 1) / 2);

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
    // A config page is placed at the credit-block alignment so its page-relative block offsets are
    // line-aligned too; probe with the same alignment the live pages were allocated with.
    const uint32_t page_alignment = std::max(l1_alignment, PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN);
    const uint32_t page_size = split_bank.pipes[0]->config_page_size();
    EXPECT_NE(arena.allocate_on(split_bank.mapping[0].first, page_size, page_alignment)->addr(), shared_base)
        << "a second range on a live sender's own core must not overlap its config page";
    EXPECT_NE(arena.allocate(page_size, page_alignment)->addr(), shared_base)
        << "a uniform range is reserved on every bank, so it must clear every per-core page";
}

TEST_F(PrefetcherPipeDramSenderFixture, RejectsDuplicateBank) {
    const CoreRangeSet first(CoreRange({0, 0}, {0, 0}));
    const CoreRangeSet second(CoreRange({1, 0}, {1, 0}));
    EXPECT_ANY_THROW(make_pipe_set(*mesh_device_, {{0, first}, {0, second}}, /*dual_senders_per_bank=*/true));
}

TEST_F(PrefetcherPipeDramSenderFixture, InvalidBatchLeavesSpaceReusable) {
    const CoreRangeSet receiver_domain(CoreRange({0, 0}, {1, 0}));
    auto space = experimental::CreatePrefetcherPipeSpace(
        *mesh_device_,
        experimental::PrefetcherPipeSpaceConfig{
            .sender_cores = {},
            .num_dram_senders = 2,
            .receiver_domain = receiver_domain,
            .ring_size = kEntrySize * kRingDepth,
            .max_receivers_per_pipe = 1,
        });

    const CoreRangeSet duplicated_receiver(CoreRange({0, 0}));
    EXPECT_ANY_THROW(experimental::CreatePrefetcherPipesForTensorPrefetcher(
        space,
        {{0, duplicated_receiver}, {1, duplicated_receiver}},
        /*support_multi_receiver_shards=*/true));

    // The rejected two-bank plan must not have fixed this space to its sender set or claimed the
    // receiver. A different one-bank plan can still reserve and carve it.
    auto pipes = experimental::CreatePrefetcherPipesForTensorPrefetcher(
        space,
        {{0, CoreRangeSet(CoreRange({1, 0}))}},
        /*support_multi_receiver_shards=*/true);
    ASSERT_EQ(pipes.size(), 1u);
    EXPECT_EQ(pipes[0]->receiver_cores(), CoreRangeSet(CoreRange({1, 0})));
}

TEST_F(PrefetcherPipeDramSenderFixture, DirectSenderReservationRejectsInvalidDramCoordinate) {
    auto space = experimental::CreatePrefetcherPipeSpace(
        *mesh_device_,
        experimental::PrefetcherPipeSpaceConfig{
            .sender_cores = {},
            .num_dram_senders = 1,
            .receiver_domain = CoreRangeSet(CoreRange({0, 0})),
            .ring_size = kEntrySize * kRingDepth,
            .max_receivers_per_pipe = 1,
        });
    const std::vector<CoreCoord> invalid_senders = {{999, 999}};
    EXPECT_ANY_THROW(experimental::set_dram_sender_cores(space, invalid_senders));
}

TEST_F(PrefetcherPipeDramSenderFixture, DroppedPipesRecarveSameSpaceWithoutAllocation) {
    const CoreRangeSet receivers(CoreRange({0, 0}, {1, 0}));
    auto set = make_pipe_set(*mesh_device_, {{0, receivers}}, /*dual_senders_per_bank=*/false);
    ASSERT_EQ(set.pipes.size(), 1u);
    const uint32_t buffer_address = set.pipes[0]->buffer_address();
    const uint32_t config_address = set.pipes[0]->config_address();
    const DeviceAddr sender_state_address = experimental::sender_state_drisc_l1_base(*set.pipes[0]);
    const uint64_t first_identity = set.pipes[0]->identity();

    set.pipes.clear();
    auto replacement = experimental::CreatePrefetcherPipesForTensorPrefetcher(
        *set.space, {{0, receivers}}, /*support_multi_receiver_shards=*/true);
    ASSERT_EQ(replacement.size(), 1u);
    EXPECT_EQ(replacement[0]->buffer_address(), buffer_address);
    EXPECT_EQ(replacement[0]->config_address(), config_address);
    EXPECT_EQ(experimental::sender_state_drisc_l1_base(*replacement[0]), sender_state_address);
    EXPECT_NE(replacement[0]->identity(), first_identity);
}

TEST_F(PrefetcherPipeDramSenderFixture, RecarveSwitchesBetweenOneAndTwoSendersPerBank) {
    const CoreRangeSet receivers(CoreRange({0, 0}, {1, 0}));
    auto set = make_pipe_set(*mesh_device_, {{0, receivers}}, /*dual_senders_per_bank=*/false);
    ASSERT_EQ(set.pipes.size(), 1u);
    const CoreCoord primary_sender = set.pipes[0]->sender_core();
    const DeviceAddr primary_state_address = experimental::sender_state_drisc_l1_base(*set.pipes[0]);

    // Growing to both of the bank's senders reserves only the second one.
    set.pipes.clear();
    auto dual = experimental::CreatePrefetcherPipesForTensorPrefetcher(
        *set.space, {{0, receivers}}, /*support_multi_receiver_shards=*/false);
    ASSERT_EQ(dual.size(), 2u);
    EXPECT_EQ(dual[0]->sender_core(), primary_sender);
    EXPECT_EQ(experimental::sender_state_drisc_l1_base(*dual[0]), primary_state_address);

    // Shrinking back to a subset of the reserved senders reuses the primary's state.
    dual.clear();
    auto single = experimental::CreatePrefetcherPipesForTensorPrefetcher(
        *set.space, {{0, receivers}}, /*support_multi_receiver_shards=*/true);
    ASSERT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0]->sender_core(), primary_sender);
    EXPECT_EQ(experimental::sender_state_drisc_l1_base(*single[0]), primary_state_address);
}

TEST_F(PrefetcherPipeDramSenderFixture, BindingAcceptsAnyEntrySizeTheRingHolds) {
    // An entry size the ring does not divide is legal: the remainder is a trailing gap holding no
    // entry, which both endpoints credit as padding at the wrap. Only a size the ring cannot hold
    // at all is rejected.
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {1, 0}));
    const PipeSet set =
        make_pipe_set(*mesh_device_, {{/*bank_id=*/0, receiver_cores}}, /*dual_senders_per_bank=*/false);
    const uint32_t ring_size = set.pipes[0]->ring_size();

    const auto make_consumer = [&](uint32_t entry_size) {
        const experimental::PrefetcherPipeParamName name{"pipe"};
        experimental::KernelSpec receiver{
            .unique_id = experimental::KernelSpecName{"receiver"}, .source = std::filesystem::path{kReceiverKernel}};
        receiver.advanced_options.prefetcher_pipe_bindings = {{{name}, "in"}};
        receiver.compile_time_args = {{"num_entries", 1u}};
        receiver.hw_config =
            experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        experimental::ProgramSpec spec{
            .name = "entry_size_consumer",
            .kernels = {std::move(receiver)},
            .work_units =
                {{.name = "receiver_wu",
                  .kernels = {experimental::KernelSpecName{"receiver"}},
                  .target_nodes = receiver_cores}},
            .advanced_options = {
                .prefetcher_pipe_parameters = {
                    {.unique_id = name,
                     .receivers = receiver_cores,
                     .ring_size = ring_size,
                     .entry_size = entry_size}}}};
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::ProgramRunArgs args;
        args.advanced_options.prefetcher_pipe_args.emplace(name, experimental::PrefetcherPipeArgument{*set.pipes[0]});
        experimental::SetProgramRunArgs(program, args);
    };
    // L1-aligned and well inside the ring, so only a divisibility rule could have rejected it.
    constexpr uint32_t kRingIndivisibleEntrySize = 48;
    ASSERT_NE(ring_size % kRingIndivisibleEntrySize, 0u);
    EXPECT_NO_THROW(make_consumer(kRingIndivisibleEntrySize));
    EXPECT_NO_THROW(make_consumer(kEntrySize / 2));
    EXPECT_NO_THROW(make_consumer(kEntrySize));
    EXPECT_ANY_THROW(make_consumer(ring_size + kEntrySize));
}

TEST_F(PrefetcherPipeDramSenderFixture, EntrySizeNotDividingRingWrapsOnTheGap) {
    // An entry size the ring does not divide leaves a trailing gap that holds no entry. The sender
    // lands back on slot 0 only if it credits that gap along with the entry reaching the usable
    // limit: the cursor advances by the units credited, so an uncredited gap would leave the next
    // lap starting inside the gap instead of at the ring base.
    constexpr uint32_t kNumReceivers = 2;
    const CoreRangeSet receiver_cores(CoreRange({0, 0}, {kNumReceivers - 1, 0}));
    const PipeSet set =
        make_pipe_set(*mesh_device_, {{/*bank_id=*/0, receiver_cores}}, /*dual_senders_per_bank=*/false);
    const CoreCoord sender_logical = set.mapping.at(0).first;
    const uint32_t ring_size = set.pipes[0]->ring_size();

    const uint32_t l1_alignment =
        MetalContext::instance(context_id_of(*mesh_device_)).hal().get_alignment(HalMemType::L1);
    // Two thirds of the creation entry size, rounded down to L1 alignment: divides neither it nor
    // the ring, so every lap ends on a gap.
    const uint32_t entry_size = ((kEntrySize * 2) / 3) / l1_alignment * l1_alignment;
    ASSERT_GT(entry_size, 0u);
    ASSERT_NE(ring_size % entry_size, 0u) << "this test is only meaningful with a trailing gap";
    const uint32_t entries_per_lap = ring_size / entry_size;

    // One full lap, then a second carrying different bytes: the second must overwrite slot 0
    // onward, which it does only if the first lap's wrap landed the cursor back on the ring base.
    preload_pattern(*mesh_device_, sender_logical, entries_per_lap, kNumReceivers, /*entry_label=*/0, entry_size);
    run_push_and_pop(*mesh_device_, set, entries_per_lap, entry_size);
    preload_pattern(
        *mesh_device_, sender_logical, entries_per_lap, kNumReceivers, /*entry_label=*/entries_per_lap, entry_size);
    run_push_and_pop(*mesh_device_, set, entries_per_lap, entry_size);

    const auto receivers = receivers_in_slab_order(receiver_cores);
    for (uint32_t r = 0; r < receivers.size(); ++r) {
        for (uint32_t i = 0; i < entries_per_lap; ++i) {
            expect_ring_slot(
                *mesh_device_,
                *set.pipes[0],
                receivers[r],
                /*slot=*/i,
                /*receiver_label=*/r,
                /*entry_label=*/entries_per_lap + i,
                entry_size);
        }
    }
    // Two laps credit exactly two rings' worth: the payload plus one gap per lap. This is the
    // assertion that fails if the gap term is dropped.
    expect_credits_drained(*mesh_device_, set, credit_units(*mesh_device_, 2 * ring_size));
}

TEST_F(PrefetcherPipeDramSenderFixture, EntryLargerThanNocBurstIsSplitIntoPackets) {
    // A block larger than one NoC packet has to go out as several packets; a single oversized
    // packet command would not deliver it intact. One receiver and a two-entry ring keep the
    // sender's DRISC-side pattern (receivers * entries * entry_size) inside its working region.
    constexpr uint32_t kNumReceivers = 1;
    constexpr uint32_t kNumEntries = 2;
    constexpr uint32_t kLargeEntrySize = 20 * 1024;
    const uint32_t max_packet_bytes =
        MetalContext::instance(context_id_of(*mesh_device_)).hal().get_noc_max_burst_size_bytes();
    ASSERT_GT(kLargeEntrySize, max_packet_bytes) << "this test needs an entry that spans several NoC packets";
    ASSERT_LE(
        kNumReceivers * kNumEntries * kLargeEntrySize,
        mesh_device_->impl().drisc_l1_arena().kernel_working_region_size());

    const CoreRangeSet receiver_cores(CoreRange({0, 0}));
    const PipeSet set = make_pipe_set(
        *mesh_device_,
        {{/*bank_id=*/0, receiver_cores}},
        /*dual_senders_per_bank=*/false,
        kLargeEntrySize,
        kNumEntries);
    const CoreCoord sender_logical = set.mapping.at(0).first;

    preload_pattern(*mesh_device_, sender_logical, kNumEntries, kNumReceivers, /*entry_label=*/0, kLargeEntrySize);
    run_push_and_pop(*mesh_device_, set, kNumEntries, kLargeEntrySize);

    for (uint32_t i = 0; i < kNumEntries; ++i) {
        expect_ring_slot(
            *mesh_device_,
            *set.pipes[0],
            receiver_cores.ranges().front().start_coord,
            /*slot=*/i,
            /*receiver_label=*/0,
            /*entry_label=*/i,
            kLargeEntrySize);
    }
    expect_credits_drained(*mesh_device_, set, credit_units(*mesh_device_, kNumEntries * kLargeEntrySize));
}

TEST_F(PrefetcherPipeDramSenderFixture, BlockSizeChangeAcrossPrograms) {
    // Two consumers of one pipe set that read different block sizes. The ring size is fixed at
    // creation and both sizes divide it, so the DRAM sender snaps each cursor onto the new grid and
    // publishes the skipped bytes as pad credits, which the receivers' own resize consumes. Each
    // endpoint snaps its own pointer, so the two snaps have to agree by arithmetic.
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
