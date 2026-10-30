// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM-7: the production Tensor Prefetcher on Mimir CCEs.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/dram_receiver_config_page.hpp"
#include "impl/buffers/dram_sender_state_block.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/global_circular_buffer_dram_sender_internal.hpp"
#include "impl/buffers/tensor_prefetcher_cce_abi.hpp"
#include "impl/buffers/tensor_prefetcher_manager.hpp"
#include "impl/buffers/tensor_prefetcher_manager_test_access.hpp"
#include "impl/buffers/tensor_prefetcher_transport.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_prefetcher_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_gcb_receiver_abi.h"

namespace tt::tt_metal::mimir_test::prefetcher {

void write_go_word(IDevice* device, uint32_t cce_index, uint32_t go_word) {
    write_cce_tl1(device, cce_index, go_message_tl1_offset(), {go_word}, Tl1Range::WholeTl1);
}

uint32_t with_init_signal(uint32_t go_word) {
    dev_msgs::go_msg_t go_message =
        MetalContext::instance().hal().get_dev_msgs_factory(HalProgrammableCoreType::DRAM).create<dev_msgs::go_msg_t>();
    std::memcpy(go_message.data(), &go_word, sizeof(go_word));
    go_message.view().signal() = dev_msgs::RUN_MSG_INIT;
    uint32_t updated_word = 0;
    std::memcpy(&updated_word, go_message.data(), sizeof(updated_word));
    return updated_word;
}

// The CCE index (Mimir-major, then location) of a translated DRAM core.
uint32_t cce_index_at(const metal_SocDescriptor& soc_desc, const CoreCoord& translated_core) {
    const uint32_t num_cces = num_cces_from_soc(soc_desc);
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        if (translated_dram_core(soc_desc, cce_index) == translated_core) {
            return cce_index;
        }
    }
    TT_THROW("Translated core {} is not one of the {} CCEs", translated_core.str(), num_cces);
}

uint32_t gcb_layout_sender_cce(IDevice* device, distributed::MeshDevice& mesh, uint32_t bank) {
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const CoreCoord sender_logical = mesh.impl().dram_sender_logical_cores(device, bank).front();
    return cce_index_at(soc_desc, soc_desc.get_physical_dram_core_from_logical(sender_logical));
}

std::optional<uint32_t> lifecycle_other_sender_bank(distributed::MeshDevice& mesh) {
    IDevice* device = mesh.get_devices().front();
    const uint32_t bank_zero_sender_cce = gcb_layout_sender_cce(device, mesh, 0);
    for (uint32_t bank = 1; bank < soc_desc_of(device).get_num_dram_views(); ++bank) {
        if (gcb_layout_sender_cce(device, mesh, bank) != bank_zero_sender_cce) {
            return bank;
        }
    }
    return std::nullopt;
}

// Sets one CCE's GO_MSG signal to RUN_MSG_INIT and writes the original word back on every exit path. A CCE left at
// RUN_MSG_INIT makes the next device open time out, and reopening does not reload firmware. Declare it after the
// ScopedDevice so the word is restored before the mesh closes.
class ScopedInitGoSignal {
public:
    ScopedInitGoSignal(IDevice* device, uint32_t cce_index) :
        device_(device), cce_index_(cce_index), original_word_(read_go_word(device, cce_index)) {
        write_go_word(device_, cce_index_, with_init_signal(original_word_));
    }
    ScopedInitGoSignal(const ScopedInitGoSignal&) = delete;
    ScopedInitGoSignal& operator=(const ScopedInitGoSignal&) = delete;

    ~ScopedInitGoSignal() {
        try {
            write_go_word(device_, cce_index_, original_word_);
        } catch (const std::exception& error) {
            ADD_FAILURE() << fmt::format(
                "CCE {}: restoring GO_MSG 0x{:08x} threw: {}", cce_index_, original_word_, error.what());
        }
    }

private:
    IDevice* device_;
    uint32_t cce_index_;
    uint32_t original_word_;
};

// SetUp runs the skip that every test starts with and opens no device: a test checks its own preconditions before it
// opens one.
class MimirPrefetcher : public ::testing::Test {
protected:
    void SetUp() override { skip_unless_programmable_dram_package(); }
};

TEST_F(MimirPrefetcher, MM7GatePreconditionsHold) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();

    const distributed::TensorPrefetcherPreconditions preconditions =
        distributed::TensorPrefetcherManagerTestAccess::preconditions(mesh);
    EXPECT_TRUE(preconditions.ok) << preconditions.reason;
    EXPECT_TRUE(experimental::IsTensorPrefetcherSupported(mesh));

    IDevice* device = mesh.get_devices().front();
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    for (uint32_t view = 0; view < soc_desc.get_num_dram_views(); ++view) {
        const std::vector<CoreCoord> senders = mesh.impl().dram_sender_logical_cores(device, view);
        ASSERT_EQ(senders, std::vector<CoreCoord>{soc_desc.get_logical_core_for_dram_view(view)})
            << "DRAM view " << view;
        const CoreCoord sender_translated = soc_desc.get_physical_dram_core_from_logical(senders.front());
        log_info(
            tt::LogTest,
            "MM7Gate DRAM view {} sender {} is CCE {} at translated {}",
            view,
            senders.front().str(),
            cce_index_at(soc_desc, sender_translated),
            sender_translated.str());
    }
}

TEST_F(MimirPrefetcher, MM7GateUninitializedSenderFails) {
    skip_unless_mimir_local_package();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();

    IDevice* device = mesh.get_devices().front();
    const uint32_t cce_index = gcb_layout_sender_cce(device, mesh, 0);
    ASSERT_EQ(read_go_signal(device, cce_index), dev_msgs::RUN_MSG_DONE) << "CCE " << cce_index;

    const ScopedInitGoSignal uninitialized_sender(device, cce_index);
    const distributed::TensorPrefetcherPreconditions preconditions =
        distributed::TensorPrefetcherManagerTestAccess::preconditions(mesh);
    log_info(tt::LogTest, "MM7Gate CCE {} at RUN_MSG_INIT: {}", cce_index, preconditions.reason);
    EXPECT_FALSE(preconditions.ok);
    EXPECT_NE(preconditions.reason.find(fmt::format("CCE {} ", cce_index)), std::string::npos) << preconditions.reason;
    EXPECT_NE(preconditions.reason.find(fmt::format("0x{:02x}", dev_msgs::RUN_MSG_INIT)), std::string::npos)
        << preconditions.reason;
}

inline constexpr uint32_t kGcbLayoutFifoBytes = 4 * kKiB;
inline constexpr uint32_t kGcbLayoutFillPattern = 0xA5C3'0000;

using BankToDramReceivers = std::vector<std::pair<uint32_t, std::vector<CoreCoord>>>;

// One sender bank of a GCB under test and its receivers, as CCE indices in receiver-table order.
struct GcbLayoutSender {
    uint32_t bank;
    std::vector<uint32_t> receiver_cces;
};

uint32_t gcb_layout_word(const std::vector<uint32_t>& words, uint32_t byte_offset) {
    return words.at(byte_offset / sizeof(uint32_t));
}

// Non-zero words, so a readback of zero proves the constructor wrote them.
void fill_gcb_layout_pattern(IDevice* device, uint32_t cce_index, uint64_t tl1_offset, uint32_t size_bytes) {
    std::vector<uint32_t> pattern(size_bytes / sizeof(uint32_t));
    for (uint32_t word = 0; word < pattern.size(); ++word) {
        pattern[word] = kGcbLayoutFillPattern | word;
    }
    write_cce_tl1(device, cce_index, tl1_offset, pattern);
}

uint32_t gcb_layout_max_receivers(const std::vector<GcbLayoutSender>& senders) {
    uint32_t max_receivers = 0;
    for (const GcbLayoutSender& sender : senders) {
        max_receivers = std::max(max_receivers, static_cast<uint32_t>(sender.receiver_cces.size()));
    }
    return max_receivers;
}

// Fills the sender state and receiver config pages the GCB will take with a pattern, creates the GCB on a fresh
// mesh, and checks every sender's state block and counters and every receiver's config page against
// dram_receiver_config_page.hpp and dram_sender_state_block.hpp. Returns the GCB so a caller can keep it alive.
experimental::GlobalCircularBuffer expect_dram_receiver_gcb_layout(
    distributed::MeshDevice& mesh, const std::vector<GcbLayoutSender>& senders) {
    IDevice* device = mesh.get_devices().front();
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const DriscL1Arena& arena = mesh.impl().drisc_l1_arena();
    const uint64_t unreserved_base = tl1_test_base();
    EXPECT_EQ(arena.sender_state_zone_size(), DriscL1Arena::kQuasarSenderStateZoneSize);
    EXPECT_EQ(arena.receiver_zone_base(), unreserved_base + DriscL1Arena::kQuasarSenderStateZoneSize);
    EXPECT_EQ(arena.receiver_zone_size(), DriscL1Arena::kQuasarReceiverZoneSize);
    EXPECT_EQ(
        arena.kernel_working_region_base(),
        unreserved_base + DriscL1Arena::kQuasarSenderStateZoneSize + DriscL1Arena::kQuasarReceiverZoneSize);

    // The mesh is fresh, so this GCB takes the first range of each zone.
    const uint32_t max_receivers = gcb_layout_max_receivers(senders);
    const uint32_t sender_region_bytes =
        dram_receiver_sender_counters_bytes(max_receivers) + dram_sender_state_block_size(max_receivers);
    std::vector<uint32_t> sender_cces;
    BankToDramReceivers bank_to_dram_receivers;
    for (const GcbLayoutSender& sender : senders) {
        sender_cces.push_back(gcb_layout_sender_cce(device, mesh, sender.bank));
        fill_gcb_layout_pattern(device, sender_cces.back(), unreserved_base, sender_region_bytes);
        std::vector<CoreCoord> receivers;
        for (const uint32_t receiver_cce : sender.receiver_cces) {
            receivers.push_back(logical_cce_core(soc_desc, receiver_cce));
            fill_gcb_layout_pattern(device, receiver_cce, arena.receiver_zone_base(), kDramReceiverConfigPageBytes);
        }
        bank_to_dram_receivers.emplace_back(sender.bank, std::move(receivers));
    }
    experimental::GlobalCircularBuffer gcb =
        experimental::CreateGlobalCircularBufferForTensorPrefetcher(mesh, bank_to_dram_receivers, kGcbLayoutFifoBytes);

    const uint64_t pages_sent_base = experimental::pages_sent_drisc_l1_base(gcb);
    const uint64_t state_base = experimental::sender_state_drisc_l1_base(gcb);
    const uint64_t config_address = gcb.config_address();
    const uint64_t buffer_address = gcb.buffer_address();
    log_info(
        tt::LogTest,
        "MM7GcbLayout {} senders: counters 0x{:x}, state block 0x{:x}, config page 0x{:x}, FIFO 0x{:x}",
        senders.size(),
        pages_sent_base,
        state_base,
        config_address,
        buffer_address);
    EXPECT_EQ(pages_sent_base, unreserved_base);
    EXPECT_EQ(state_base, pages_sent_base + dram_receiver_sender_counters_bytes(max_receivers));
    EXPECT_EQ(config_address, arena.receiver_zone_base());
    EXPECT_EQ(buffer_address, config_address + kDramReceiverConfigPageBytes);
    EXPECT_EQ(experimental::pages_sent_receiver_l1_base(gcb), config_address + kDramReceiverCreditLineOffset);
    std::vector<std::vector<CoreCoord>> expected_receivers_per_sender;
    for (const auto& [bank, receivers] : bank_to_dram_receivers) {
        expected_receivers_per_sender.push_back(receivers);
    }
    EXPECT_EQ(experimental::receiver_logical_cores_per_sender(gcb), expected_receivers_per_sender);

    for (size_t sender_slot = 0; sender_slot < senders.size(); ++sender_slot) {
        const uint32_t sender_cce = sender_cces[sender_slot];
        const std::vector<uint32_t>& receiver_cces = senders[sender_slot].receiver_cces;
        const uint32_t num_receivers = static_cast<uint32_t>(receiver_cces.size());
        const std::vector<uint32_t> counters =
            read_cce_tl1(device, sender_cce, pages_sent_base, dram_receiver_sender_counters_bytes(max_receivers));
        EXPECT_EQ(counters, std::vector<uint32_t>(counters.size(), 0u))
            << "sender CCE " << sender_cce << " counters are not zeroed";

        const std::vector<uint32_t> state_words =
            read_cce_tl1(device, sender_cce, state_base, dram_sender_state_block_size(max_receivers));
        DramSenderStateBlock state_block{};
        std::memcpy(&state_block, state_words.data(), sizeof(state_block));
        EXPECT_EQ(state_block.config_ptr, state_base + offsetof(DramSenderStateBlock, is_sender));
        EXPECT_EQ(state_block.fifo_start_addr, buffer_address);
        EXPECT_EQ(state_block.fifo_wr_ptr, buffer_address);
        EXPECT_EQ(state_block.receiver_noc_xy_ptr, state_base + sizeof(DramSenderStateBlock));
        EXPECT_EQ(state_block.aligned_pages_sent_ptr, pages_sent_base);
        EXPECT_EQ(
            state_block.num_receivers_and_remote_pages_sent_ptr >> dev_msgs::REMOTE_CB_PACKED_COUNT_SHIFT,
            num_receivers)
            << "sender CCE " << sender_cce;
        EXPECT_EQ(
            state_block.num_receivers_and_remote_pages_sent_ptr & dev_msgs::REMOTE_CB_PACKED_ADDR_MASK,
            config_address + kDramReceiverCreditLineOffset);
        EXPECT_EQ(state_block.is_sender, 1u);
        EXPECT_EQ(state_block.num_receivers, num_receivers) << "sender CCE " << sender_cce;
        EXPECT_EQ(state_block.buffer_address, buffer_address);
        EXPECT_EQ(state_block.fifo_size_per_receiver, kGcbLayoutFifoBytes);
        const size_t receiver_table_word = sizeof(DramSenderStateBlock) / sizeof(uint32_t);
        for (uint32_t receiver_slot = 0; receiver_slot < max_receivers; ++receiver_slot) {
            const uint32_t expected_cce = receiver_slot < num_receivers ? receiver_cces[receiver_slot] : 0u;
            EXPECT_EQ(state_words.at(receiver_table_word + (2 * receiver_slot)), expected_cce)
                << "sender CCE " << sender_cce << " receiver table slot " << receiver_slot;
            EXPECT_EQ(state_words.at(receiver_table_word + (2 * receiver_slot) + 1), 0u)
                << "sender CCE " << sender_cce << " receiver table slot " << receiver_slot;
        }

        // The acked block follows this sender's own receiver count, not the GCB-wide maximum.
        const uint64_t pages_acked_base = pages_sent_base + dram_receiver_sender_pages_acked_offset(num_receivers);
        for (uint32_t receiver_slot = 0; receiver_slot < num_receivers; ++receiver_slot) {
            const uint32_t receiver_cce = receiver_cces[receiver_slot];
            const std::vector<uint32_t> page =
                read_cce_tl1(device, receiver_cce, config_address, kDramReceiverConfigPageBytes);
            std::vector<uint32_t> expected_page(page.size(), 0u);
            expected_page[kDramReceiverFifoStartAddrOffset / sizeof(uint32_t)] = buffer_address;
            expected_page[kDramReceiverFifoSizeOffset / sizeof(uint32_t)] = kGcbLayoutFifoBytes;
            expected_page[kDramReceiverSenderIndexOffset / sizeof(uint32_t)] = sender_cce;
            expected_page[kDramReceiverRemotePagesAckedAddrOffset / sizeof(uint32_t)] =
                pages_acked_base + (receiver_slot * kDramReceiverCreditStrideBytes);
            expected_page[kDramReceiverFifoRdPtrOffset / sizeof(uint32_t)] = buffer_address;
            EXPECT_EQ(page, expected_page) << "receiver CCE " << receiver_cce;
            EXPECT_EQ(gcb_layout_word(page, kDramReceiverCreditLineOffset), 0u) << "receiver CCE " << receiver_cce;
            EXPECT_EQ(gcb_layout_word(page, kDramReceiverPagesAckedOffset), 0u) << "receiver CCE " << receiver_cce;
        }
    }
    return gcb;
}

// Checks that two live GCBs hold disjoint ranges in the sender-state zone and in the receiver zone.
void expect_gcb_layout_ranges_disjoint(
    const experimental::GlobalCircularBuffer& first, const experimental::GlobalCircularBuffer& second) {
    const std::shared_ptr<DriscL1Allocation> first_sender = experimental::sender_state_drisc_l1_allocation(first);
    const std::shared_ptr<DriscL1Allocation> second_sender = experimental::sender_state_drisc_l1_allocation(second);
    EXPECT_TRUE(
        first_sender->addr() + first_sender->size() <= second_sender->addr() ||
        second_sender->addr() + second_sender->size() <= first_sender->addr())
        << fmt::format(
               "sender-state ranges [0x{:x}, +{}) and [0x{:x}, +{}) overlap",
               first_sender->addr(),
               first_sender->size(),
               second_sender->addr(),
               second_sender->size());
    const uint64_t first_receiver_end = first.buffer_address() + first.size();
    const uint64_t second_receiver_end = second.buffer_address() + second.size();
    EXPECT_TRUE(first_receiver_end <= second.config_address() || second_receiver_end <= first.config_address())
        << fmt::format(
               "receiver ranges [0x{:x}, 0x{:x}) and [0x{:x}, 0x{:x}) overlap",
               first.config_address(),
               first_receiver_end,
               second.config_address(),
               second_receiver_end);
}

std::string gcb_layout_creation_error(
    distributed::MeshDevice& mesh, const BankToDramReceivers& bank_to_dram_receivers) {
    try {
        experimental::CreateGlobalCircularBufferForTensorPrefetcher(mesh, bank_to_dram_receivers, kGcbLayoutFifoBytes);
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}

TEST_F(MimirPrefetcher, MM7GcbLayoutLocalReceiver) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();

    IDevice* device = mesh.get_devices().front();
    const uint32_t sender_cce = gcb_layout_sender_cce(device, mesh, 0);
    const uint32_t local_receiver_cce = sibling_cce(sender_cce);
    const experimental::GlobalCircularBuffer first_gcb =
        expect_dram_receiver_gcb_layout(mesh, {{.bank = 0, .receiver_cces = {local_receiver_cce}}});

    const BankToDramReceivers bank_to_dram_receivers = {
        {0, {logical_cce_core(soc_desc_of(device), local_receiver_cce)}}};
    const experimental::GlobalCircularBuffer second_gcb =
        experimental::CreateGlobalCircularBufferForTensorPrefetcher(mesh, bank_to_dram_receivers, kGcbLayoutFifoBytes);
    expect_gcb_layout_ranges_disjoint(first_gcb, second_gcb);
}

TEST_F(MimirPrefetcher, MM7GcbLayoutRemoteReceiver) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_mimirs < 2) {
        GTEST_SKIP() << "Needs a receiver on another Mimir; this descriptor has one Mimir.";
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();

    // The remote receiver comes first, so the receiver table must keep this order, not sort it.
    const uint32_t sender_cce = gcb_layout_sender_cce(mesh.get_devices().front(), mesh, 0);
    const uint32_t remote_receiver_cce = next_mimir_cce(sender_cce);
    const uint32_t local_receiver_cce = sibling_cce(sender_cce);
    expect_dram_receiver_gcb_layout(mesh, {{.bank = 0, .receiver_cces = {remote_receiver_cce, local_receiver_cce}}});
}

TEST_F(MimirPrefetcher, MM7GcbLayoutTwoSenders) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    // Two senders with one receiver each, all four on different CCEs.
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (topology.num_mimirs < 2) {
        GTEST_SKIP() << "Needs four CCEs (two senders, two receivers); this descriptor has one Mimir.";
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();

    IDevice* device = mesh.get_devices().front();
    const uint32_t first_sender_cce = gcb_layout_sender_cce(device, mesh, 0);
    uint32_t second_bank = 0;
    for (uint32_t bank = 1; bank < topology.num_views; ++bank) {
        if (gcb_layout_sender_cce(device, mesh, bank) != first_sender_cce) {
            second_bank = bank;
            break;
        }
    }
    ASSERT_NE(second_bank, 0u) << "no DRAM view has a sender CCE other than bank 0's";
    const uint32_t second_sender_cce = gcb_layout_sender_cce(device, mesh, second_bank);
    std::vector<uint32_t> receiver_cces;
    for (uint32_t cce_index = 0; cce_index < topology.num_cces && receiver_cces.size() < 2; ++cce_index) {
        if (cce_index != first_sender_cce && cce_index != second_sender_cce) {
            receiver_cces.push_back(cce_index);
        }
    }
    // Equal receiver counts: four CCEs leave two receivers, and a receiver can serve only one sender of a GCB.
    const std::vector<GcbLayoutSender> senders = {
        {.bank = 0, .receiver_cces = {receiver_cces[0]}}, {.bank = second_bank, .receiver_cces = {receiver_cces[1]}}};
    const experimental::GlobalCircularBuffer first_gcb = expect_dram_receiver_gcb_layout(mesh, senders);

    BankToDramReceivers bank_to_dram_receivers;
    for (const GcbLayoutSender& sender : senders) {
        bank_to_dram_receivers.emplace_back(
            sender.bank, std::vector<CoreCoord>{logical_cce_core(soc_desc_of(device), sender.receiver_cces.front())});
    }
    const experimental::GlobalCircularBuffer second_gcb =
        experimental::CreateGlobalCircularBufferForTensorPrefetcher(mesh, bank_to_dram_receivers, kGcbLayoutFifoBytes);
    expect_gcb_layout_ranges_disjoint(first_gcb, second_gcb);
}

TEST_F(MimirPrefetcher, MM7GcbLayoutRejectsConflictingReceivers) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const CoreCoord sender_logical = mesh.impl().dram_sender_logical_cores(device, 0).front();
    const CoreCoord sender_translated = soc_desc.get_physical_dram_core_from_logical(sender_logical);

    std::map<std::pair<size_t, size_t>, std::vector<CoreCoord>> logical_by_translated;
    for (uint32_t view = 0; view < soc_desc.get_num_dram_views(); ++view) {
        for (uint32_t endpoint = 0; endpoint < locations_per_channel(soc_desc); ++endpoint) {
            const CoreCoord logical(view, endpoint);
            const CoreCoord translated = soc_desc.get_physical_dram_core_from_logical(logical);
            logical_by_translated[{translated.x, translated.y}].push_back(logical);
        }
    }
    std::vector<CoreCoord> alias_pair;
    for (const auto& [translated, logical_coords] : logical_by_translated) {
        if (translated != std::pair<size_t, size_t>{sender_translated.x, sender_translated.y} &&
            logical_coords.size() >= 2) {
            alias_pair = {logical_coords[0], logical_coords[1]};
            break;
        }
    }
    ASSERT_EQ(alias_pair.size(), 2u) << "no two logical DRAM coords name one non-sender CCE";
    std::vector<CoreCoord> sender_aliases;
    for (const CoreCoord& logical : logical_by_translated.at({sender_translated.x, sender_translated.y})) {
        if (logical != sender_logical) {
            sender_aliases.push_back(logical);
        }
    }
    ASSERT_FALSE(sender_aliases.empty()) << "no other logical DRAM coord names the sender CCE";
    log_info(
        tt::LogTest,
        "MM7GcbLayout aliases {} and {}; sender {} alias {}",
        alias_pair[0].str(),
        alias_pair[1].str(),
        sender_logical.str(),
        sender_aliases.front().str());

    const std::string alias_error = gcb_layout_creation_error(mesh, {{0, alias_pair}});
    EXPECT_NE(alias_error.find("logical aliases"), std::string::npos) << alias_error;

    const std::string sender_error = gcb_layout_creation_error(mesh, {{0, {alias_pair[0], sender_logical}}});
    EXPECT_NE(sender_error.find("also this GlobalCircularBuffer's sender"), std::string::npos) << sender_error;

    const std::string sender_alias_error =
        gcb_layout_creation_error(mesh, {{0, {alias_pair[0], sender_aliases.front()}}});
    EXPECT_NE(sender_alias_error.find("also this GlobalCircularBuffer's sender"), std::string::npos)
        << sender_alias_error;

    const std::string duplicate_error = gcb_layout_creation_error(mesh, {{0, {alias_pair[0], alias_pair[0]}}});
    EXPECT_NE(duplicate_error.find("listed more than once"), std::string::npos) << duplicate_error;
}

namespace gcb_abi = cce_gcb_receiver_abi;

const char* const kGcbSmokeSenderKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_gcb_smoke_sender.cpp";
const char* const kGcbDiscardReceiverKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_gcb_discard_receiver.cpp";
const char* const kGcbValidatorReceiverKernel =
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_gcb_validator_receiver.cpp";

inline constexpr uint32_t kGcbSmokeFifoBytes = 4 * kKiB;
// The bytes right after every receiver's FIFO, filled with a pattern before a run and checked with the FIFO state, so
// a sender that writes past the FIFO end fails the test instead of landing in whatever the receiver zone holds next.
inline constexpr uint32_t kGcbSmokeGuardBytes = 4 * gcb_abi::kLineBytes;
inline constexpr uint32_t kGcbSmokeGuardPattern = 0x6A4D'0000;
// rdcycle budgets per wait. S5 measured the emulated chip clock at ~85,000 cycles/s while the host polls it and
// ~90 cycles/s while the host is idle, so these count host activity rather than wall time. The wait budget bounds
// waits that should succeed (~200 s of polling, under TT_METAL_OPERATION_TIMEOUT_SECONDS=600); the stall budget is
// the sender's in the stall case (~12 s of polling), which must expire while the receiver holds its credits.
inline constexpr uint64_t kGcbSmokeWaitBudgetCycles = 1ull << 24;
inline constexpr uint64_t kGcbSmokeStallBudgetCycles = 1ull << 20;
inline constexpr std::chrono::seconds kGcbSmokeHostWaitLimit{120};
inline constexpr std::chrono::milliseconds kGcbSmokeHostPollInterval{50};

struct GcbSmokeRequest {
    uint32_t tensor_id = 0;
    uint32_t slab = 0;
    uint32_t page_bytes = 0;
    uint32_t num_pages = 0;
};

// The credits a sender publishes for a schedule and where its write pointer ends, from the FIFO start. Mirrors the
// page-aligned limit, wrap padding and resize padding of cce_remote_cb.h, so the host can check both ends' counters.
struct GcbFifoEndState {
    uint32_t credit_units = 0;
    uint32_t fifo_offset = 0;
};

GcbFifoEndState expected_gcb_fifo_end_state(const std::vector<GcbSmokeRequest>& schedule, uint32_t fifo_bytes) {
    GcbFifoEndState state;
    for (const GcbSmokeRequest& request : schedule) {
        const uint32_t limit = fifo_bytes - (fifo_bytes % request.page_bytes);
        const uint32_t aligned_offset =
            ((state.fifo_offset + request.page_bytes - 1) / request.page_bytes) * request.page_bytes;
        if (aligned_offset >= limit) {
            state.credit_units += (fifo_bytes - state.fifo_offset) / kDramReceiverCreditUnitBytes;
            state.fifo_offset = 0;
        } else {
            state.credit_units += (aligned_offset - state.fifo_offset) / kDramReceiverCreditUnitBytes;
            state.fifo_offset = aligned_offset;
        }
        for (uint32_t page = 0; page < request.num_pages; ++page) {
            uint32_t bytes = request.page_bytes;
            state.fifo_offset += request.page_bytes;
            if (state.fifo_offset >= limit) {
                bytes += fifo_bytes - limit;
                state.fifo_offset -= limit;
            }
            state.credit_units += bytes / kDramReceiverCreditUnitBytes;
        }
    }
    return state;
}

uint32_t gcb_smoke_total_pages(const std::vector<GcbSmokeRequest>& schedule) {
    uint32_t pages = 0;
    for (const GcbSmokeRequest& request : schedule) {
        pages += request.num_pages;
    }
    return pages;
}

std::string gcb_status_string(const std::vector<uint32_t>& status) {
    return fmt::format(
        "magic 0x{:08x} code {} requests {} pages {} mismatched words {} first mismatch request {} page {} word {} "
        "observed 0x{:08x} expected 0x{:08x} rechecked 0x{:08x}; waited on receiver {} sent {} acked {}",
        status[gcb_abi::kStatusMagic],
        status[gcb_abi::kStatusCode],
        status[gcb_abi::kStatusRequestsDone],
        status[gcb_abi::kStatusPagesDone],
        status[gcb_abi::kStatusMismatchedWords],
        status[gcb_abi::kStatusFirstMismatchRequest],
        status[gcb_abi::kStatusFirstMismatchPage],
        status[gcb_abi::kStatusFirstMismatchWord],
        status[gcb_abi::kStatusObserved],
        status[gcb_abi::kStatusExpected],
        status[gcb_abi::kStatusRechecked],
        status[gcb_abi::kStatusWaitReceiver],
        status[gcb_abi::kStatusWaitPagesSent],
        status[gcb_abi::kStatusWaitPagesAcked]);
}

// One DRAM-receiver GCB of `fifo_bytes` per receiver from `sender_bank`'s sender CCE to `receiver_cces` (in
// receiver-table order), plus a kernel block (cce_gcb_receiver_abi.h) carved from the receiver zone at one offset on
// every CCE. Declare it after the ScopedDevice and before any CceReceiverProgram.
class GcbSmoke {
public:
    GcbSmoke(
        distributed::MeshDevice& mesh,
        uint32_t sender_bank,
        std::vector<uint32_t> receiver_cces,
        uint32_t fifo_bytes = kGcbSmokeFifoBytes) :
        device_(mesh.get_devices().front()),
        sender_bank_(sender_bank),
        sender_cce_(gcb_layout_sender_cce(device_, mesh, sender_bank_)),
        receiver_cces_(std::move(receiver_cces)),
        fifo_bytes_(fifo_bytes),
        gcb_(create_gcb(mesh)),
        guard_(mesh.impl().drisc_l1_arena().allocate_receiver(kGcbSmokeGuardBytes, gcb_abi::kLineBytes)),
        block_(mesh.impl().drisc_l1_arena().allocate_receiver(gcb_abi::kBlockBytes, gcb_abi::kLineBytes)) {
        TT_FATAL(
            guard_->addr() == gcb_.buffer_address() + fifo_bytes_,
            "The FIFO guard landed at 0x{:x}, not at the FIFO end 0x{:x}",
            guard_->addr(),
            gcb_.buffer_address() + fifo_bytes_);
        for (const uint32_t receiver_cce : receiver_cces_) {
            write_cce_tl1(device_, receiver_cce, guard_->addr(), guard_pattern());
        }
        reset_kernel_blocks();
        log_info(
            tt::LogTest,
            "MM7GcbSmoke sender CCE {} receivers {}: config page 0x{:x}, FIFO 0x{:x} +{} B, guard 0x{:x}, kernel block "
            "0x{:x}",
            sender_cce_,
            fmt::join(receiver_cces_, ","),
            gcb_.config_address(),
            gcb_.buffer_address(),
            fifo_bytes_,
            guard_->addr(),
            block_->addr());
    }

    IDevice* device() const { return device_; }
    uint32_t fifo_bytes() const { return fifo_bytes_; }
    uint32_t receiver_cce(uint32_t slot) const { return receiver_cces_.at(slot); }
    const experimental::GlobalCircularBuffer& gcb() const { return gcb_; }

    // Zeroes the command and holding lines and poisons the status line on the sender and every receiver, so a status
    // read after the next run cannot see an earlier run's.
    void reset_kernel_blocks() const {
        std::vector<uint32_t> block_words(gcb_abi::kBlockBytes / sizeof(uint32_t), 0);
        std::fill(
            block_words.begin() + (gcb_abi::kStatusLineOffset / sizeof(uint32_t)),
            block_words.begin() + (gcb_abi::kHoldingLineOffset / sizeof(uint32_t)),
            gcb_abi::kStatusPoison);
        write_cce_tl1(device_, sender_cce_, block_->addr(), block_words);
        for (const uint32_t receiver_cce : receiver_cces_) {
            write_cce_tl1(device_, receiver_cce, block_->addr(), block_words);
        }
    }

    // The receiver in table slot `slot` consumes slab `slab + slot` of each request.
    CceKernelLaunch receiver_launch(
        uint32_t slot,
        const char* kernel_path,
        uint32_t seed,
        uint32_t mode,
        uint32_t stall_pages,
        uint64_t budget_cycles,
        const std::vector<GcbSmokeRequest>& schedule) const {
        std::vector<GcbSmokeRequest> receiver_schedule = schedule;
        for (GcbSmokeRequest& request : receiver_schedule) {
            request.slab += slot;
        }
        return CceKernelLaunch{
            .cce_index = receiver_cces_.at(slot),
            .kernel_path = kernel_path,
            .runtime_args = kernel_args(
                static_cast<uint32_t>(gcb_.config_address()),
                budget_cycles,
                seed,
                mode,
                stall_pages,
                receiver_schedule)};
    }

    std::vector<uint32_t> run_sender(uint64_t budget_cycles, const std::vector<GcbSmokeRequest>& schedule) {
        const CceKernelLaunch sender{
            .cce_index = sender_cce_,
            .kernel_path = kGcbSmokeSenderKernel,
            .runtime_args = kernel_args(
                static_cast<uint32_t>(experimental::sender_state_drisc_l1_base(gcb_)),
                budget_cycles,
                mimir_test_seed(),
                0,
                0,
                schedule)};
        std::unique_ptr<Program> program = launch_cce_program(device_, {sender});
        slow_dispatch::WaitProgramDone(*device_, *program);
        const std::vector<uint32_t> status = read_status(sender_cce_);
        log_info(tt::LogTest, "MM7GcbSmoke sender CCE {}: {}", sender_cce_, gcb_status_string(status));
        return status;
    }

    std::vector<uint32_t> read_status(uint32_t cce_index) const {
        return read_cce_tl1(
            device_, cce_index, block_->addr() + gcb_abi::kStatusLineOffset, gcb_abi::kStatusWords * sizeof(uint32_t));
    }

    std::vector<uint32_t> read_receiver_status(uint32_t slot) const {
        const std::vector<uint32_t> status = read_status(receiver_cces_.at(slot));
        log_info(tt::LogTest, "MM7GcbSmoke receiver CCE {}: {}", receiver_cces_.at(slot), gcb_status_string(status));
        return status;
    }

    ::testing::AssertionResult wait_for_holding(uint32_t slot, uint32_t pages) const {
        const uint32_t receiver_cce = receiver_cces_.at(slot);
        const uint64_t holding_address = block_->addr() + gcb_abi::kHoldingLineOffset;
        const std::chrono::steady_clock::time_point deadline =
            std::chrono::steady_clock::now() + kGcbSmokeHostWaitLimit;
        uint32_t observed = read_cce_tl1(device_, receiver_cce, holding_address, sizeof(uint32_t)).front();
        while (observed != pages) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return ::testing::AssertionFailure() << fmt::format(
                           "receiver CCE {} holds {} pages after {} s, expected {}",
                           receiver_cce,
                           observed,
                           kGcbSmokeHostWaitLimit.count(),
                           pages);
            }
            std::this_thread::sleep_for(kGcbSmokeHostPollInterval);
            observed = read_cce_tl1(device_, receiver_cce, holding_address, sizeof(uint32_t)).front();
        }
        return ::testing::AssertionSuccess();
    }

    // Credits `units` to receiver `slot` as a sender's finalize would, but from the host and with no data: the sender's
    // pages_sent slot and the receiver's credit line. The sender's wr_ptr stays where it is.
    void credit_from_host(uint32_t slot, uint32_t units) const {
        write_cce_tl1(
            device_,
            sender_cce_,
            experimental::pages_sent_drisc_l1_base(gcb_) + (slot * kDramReceiverCreditStrideBytes),
            {units});
        write_cce_tl1(device_, receiver_cces_.at(slot), gcb_.config_address() + kDramReceiverCreditLineOffset, {units});
    }

    // Acks `units` on the sender's pages_acked slot for receiver `slot`, as the receiver's remote ack store would.
    void ack_from_host(uint32_t slot, uint32_t units) const {
        const uint32_t num_receivers = static_cast<uint32_t>(receiver_cces_.size());
        write_cce_tl1(
            device_,
            sender_cce_,
            experimental::pages_sent_drisc_l1_base(gcb_) + dram_receiver_sender_pages_acked_offset(num_receivers) +
                (slot * kDramReceiverCreditStrideBytes),
            {units});
    }

    // Both ends' counters and pointers once every page of `schedule` is credited, and `acked_units` of them acked.
    void expect_fifo_state(uint32_t slot, const GcbFifoEndState& sent, uint32_t acked_units, uint32_t rd_offset) const {
        const uint32_t receiver_cce = receiver_cces_.at(slot);
        const uint32_t num_receivers = static_cast<uint32_t>(receiver_cces_.size());
        const uint64_t pages_sent_base = experimental::pages_sent_drisc_l1_base(gcb_);
        const uint64_t state_base = experimental::sender_state_drisc_l1_base(gcb_);
        const uint32_t fifo_start = static_cast<uint32_t>(gcb_.buffer_address());
        EXPECT_EQ(
            read_tl1_word(sender_cce_, pages_sent_base + (slot * kDramReceiverCreditStrideBytes)), sent.credit_units)
            << "sender pages_sent for receiver CCE " << receiver_cce;
        EXPECT_EQ(
            read_tl1_word(
                sender_cce_,
                pages_sent_base + dram_receiver_sender_pages_acked_offset(num_receivers) +
                    (slot * kDramReceiverCreditStrideBytes)),
            acked_units)
            << "sender pages_acked for receiver CCE " << receiver_cce;
        EXPECT_EQ(
            read_tl1_word(sender_cce_, state_base + offsetof(DramSenderStateBlock, fifo_wr_ptr)),
            fifo_start + sent.fifo_offset);
        const uint64_t config_address = gcb_.config_address();
        EXPECT_EQ(read_tl1_word(receiver_cce, config_address + kDramReceiverCreditLineOffset), sent.credit_units)
            << "receiver CCE " << receiver_cce << " credit line";
        EXPECT_EQ(read_tl1_word(receiver_cce, config_address + kDramReceiverPagesAckedOffset), acked_units)
            << "receiver CCE " << receiver_cce << " pages_acked";
        EXPECT_EQ(read_tl1_word(receiver_cce, config_address + kDramReceiverFifoRdPtrOffset), fifo_start + rd_offset)
            << "receiver CCE " << receiver_cce << " fifo_rd_ptr";
        EXPECT_EQ(read_cce_tl1(device_, receiver_cce, guard_->addr(), kGcbSmokeGuardBytes), guard_pattern())
            << "receiver CCE " << receiver_cce << ": a write landed past the FIFO end";
    }

private:
    experimental::GlobalCircularBuffer create_gcb(distributed::MeshDevice& mesh) const {
        const metal_SocDescriptor& soc_desc = soc_desc_of(device_);
        std::vector<CoreCoord> receivers;
        for (const uint32_t receiver_cce : receiver_cces_) {
            receivers.push_back(logical_cce_core(soc_desc, receiver_cce));
        }
        return experimental::CreateGlobalCircularBufferForTensorPrefetcher(
            mesh, BankToDramReceivers{{sender_bank_, receivers}}, fifo_bytes_);
    }

    std::vector<uint32_t> kernel_args(
        uint32_t gcb_address,
        uint64_t budget_cycles,
        uint32_t seed,
        uint32_t mode,
        uint32_t stall_pages,
        const std::vector<GcbSmokeRequest>& schedule) const {
        std::vector<uint32_t> args(gcb_abi::schedule_arg(schedule.size(), 0), 0);
        args[gcb_abi::kArgGcbAddr] = gcb_address;
        args[gcb_abi::kArgBlockAddr] = static_cast<uint32_t>(block_->addr());
        args[gcb_abi::kArgBudgetLo] = static_cast<uint32_t>(budget_cycles);
        args[gcb_abi::kArgBudgetHi] = static_cast<uint32_t>(budget_cycles >> 32);
        args[gcb_abi::kArgSeed] = seed;
        args[gcb_abi::kArgMode] = mode;
        args[gcb_abi::kArgStallPages] = stall_pages;
        args[gcb_abi::kArgNumRequests] = static_cast<uint32_t>(schedule.size());
        for (uint32_t request = 0; request < schedule.size(); ++request) {
            args[gcb_abi::schedule_arg(request, gcb_abi::kRequestTensorId)] = schedule[request].tensor_id;
            args[gcb_abi::schedule_arg(request, gcb_abi::kRequestSlab)] = schedule[request].slab;
            args[gcb_abi::schedule_arg(request, gcb_abi::kRequestPageBytes)] = schedule[request].page_bytes;
            args[gcb_abi::schedule_arg(request, gcb_abi::kRequestNumPages)] = schedule[request].num_pages;
        }
        return args;
    }

    uint32_t read_tl1_word(uint32_t cce_index, uint64_t tl1_offset) const {
        return read_cce_tl1(device_, cce_index, tl1_offset, sizeof(uint32_t)).front();
    }

    static std::vector<uint32_t> guard_pattern() {
        std::vector<uint32_t> pattern(kGcbSmokeGuardBytes / sizeof(uint32_t));
        for (uint32_t word = 0; word < pattern.size(); ++word) {
            pattern[word] = kGcbSmokeGuardPattern | word;
        }
        return pattern;
    }

    IDevice* device_;
    uint32_t sender_bank_;
    uint32_t sender_cce_;
    std::vector<uint32_t> receiver_cces_;
    uint32_t fifo_bytes_;
    experimental::GlobalCircularBuffer gcb_;
    std::shared_ptr<DriscL1Allocation> guard_;
    std::shared_ptr<DriscL1Allocation> block_;
};

CceKernelLaunch validator_launch(const GcbSmoke& smoke, uint32_t slot, const std::vector<GcbSmokeRequest>& schedule) {
    return smoke.receiver_launch(
        slot, kGcbValidatorReceiverKernel, mimir_test_seed(), 0, 0, kGcbSmokeWaitBudgetCycles, schedule);
}

void queue_tensor(distributed::MeshDevice& mesh, const GcbSmoke& smoke, const MeshTensor& tensor, uint32_t blocks) {
    experimental::QueueTensorPrefetcherRequest(
        mesh,
        smoke.gcb(),
        std::nullopt,
        {experimental::TensorPrefetcherInput{.tensor = tensor, .block_count = blocks}});
}

uint32_t gcb_smoke_local_receiver(distributed::MeshDevice& mesh) {
    const uint32_t sender_cce = gcb_layout_sender_cce(mesh.get_devices().front(), mesh, 0);
    return sibling_cce(sender_cce);
}

std::string gcb_smoke_d2d_skip_reason() {
    if (std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        return skip_reason;
    }
    if (topology_from(configured_soc_desc()).num_mimirs < 2) {
        return "Needs a receiver on another Mimir; this descriptor has one Mimir.";
    }
    return "";
}

// Drain N: the sender completes and every counter on both ends agrees.
TEST_F(MimirPrefetcher, MM7GcbSmokeDiscardDrain) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    const std::vector<GcbSmokeRequest> schedule = {{.tensor_id = 1, .slab = 0, .page_bytes = 512, .num_pages = 20}};
    const uint32_t seed = mimir_test_seed();
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0, kGcbDiscardReceiverKernel, seed, gcb_abi::kModeDrainCount, 0, kGcbSmokeWaitBudgetCycles, schedule)});

    const std::vector<uint32_t> sender_status = smoke.run_sender(kGcbSmokeWaitBudgetCycles, schedule);
    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(sender_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    EXPECT_EQ(sender_status[gcb_abi::kStatusPagesDone], gcb_smoke_total_pages(schedule));
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], gcb_smoke_total_pages(schedule));
    const GcbFifoEndState end_state = expected_gcb_fifo_end_state(schedule, kGcbSmokeFifoBytes);
    smoke.expect_fifo_state(0, end_state, end_state.credit_units, end_state.fifo_offset);
}

// Runs the smoke sender into a validator receiver on `receiver_cce` and checks both statuses and the end state. The
// validator recomputes the pages with `validator_seed`.
void run_gcb_smoke_validation(
    distributed::MeshDevice& mesh,
    uint32_t receiver_cce,
    const std::vector<GcbSmokeRequest>& schedule,
    uint32_t validator_seed) {
    GcbSmoke smoke(mesh, 0, {receiver_cce});
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0, kGcbValidatorReceiverKernel, validator_seed, 0, 0, kGcbSmokeWaitBudgetCycles, schedule)});

    const std::vector<uint32_t> sender_status = smoke.run_sender(kGcbSmokeWaitBudgetCycles, schedule);
    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(sender_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], gcb_smoke_total_pages(schedule));
    if (validator_seed == mimir_test_seed()) {
        EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    } else {
        EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeMismatch);
        EXPECT_GT(receiver_status[gcb_abi::kStatusMismatchedWords], 0u);
    }
    const GcbFifoEndState end_state = expected_gcb_fifo_end_state(schedule, kGcbSmokeFifoBytes);
    smoke.expect_fifo_state(0, end_state, end_state.credit_units, end_state.fifo_offset);
}

TEST_F(MimirPrefetcher, MM7GcbSmokeValidatorLocal) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    run_gcb_smoke_validation(
        mesh,
        gcb_smoke_local_receiver(mesh),
        {{.tensor_id = 2, .slab = 0, .page_bytes = 1024, .num_pages = 12}},
        mimir_test_seed());
}

// The validator reports the mismatch and keeps acking, so the sender still completes.
TEST_F(MimirPrefetcher, MM7GcbSmokeValidatorWrongSeed) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    run_gcb_smoke_validation(
        mesh,
        gcb_smoke_local_receiver(mesh),
        {{.tensor_id = 3, .slab = 0, .page_bytes = 256, .num_pages = 24}},
        mimir_test_seed() ^ 0x1u);
}

// Three page sizes in one GCB: 256 B pages leave wr_ptr at 1280, which 512 B pages pad to 1536; 384 B pages do not
// divide the 4 KiB FIFO, so every wrap also pads the 256 B past the page-aligned limit.
TEST_F(MimirPrefetcher, MM7GcbSmokeResize) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    run_gcb_smoke_validation(
        mesh,
        gcb_smoke_local_receiver(mesh),
        {{.tensor_id = 4, .slab = 0, .page_bytes = 256, .num_pages = 5},
         {.tensor_id = 5, .slab = 0, .page_bytes = 512, .num_pages = 6},
         {.tensor_id = 6, .slab = 0, .page_bytes = 384, .num_pages = 12}},
        mimir_test_seed());
}

TEST_F(MimirPrefetcher, MM7GcbSmokeValidatorD2d) {
    skip_unless_mimir_local_package();
    if (const std::string skip_reason = gcb_smoke_d2d_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    MIMIR_REQUIRE_REMOTE_PREFLIGHT();
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    const uint32_t sender_cce = gcb_layout_sender_cce(mesh.get_devices().front(), mesh, 0);
    const uint32_t remote_receiver_cce = next_mimir_cce(sender_cce);
    run_gcb_smoke_validation(
        mesh,
        remote_receiver_cce,
        {{.tensor_id = 7, .slab = 0, .page_bytes = 512, .num_pages = 16}},
        mimir_test_seed());
}

// The receiver drains every page the sender sends and keeps running until the host stops it.
TEST_F(MimirPrefetcher, MM7GcbSmokeDrainUntilStopped) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    const std::vector<GcbSmokeRequest> schedule = {{.tensor_id = 8, .slab = 0, .page_bytes = 512, .num_pages = 12}};
    const uint32_t seed = mimir_test_seed();
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0,
            kGcbDiscardReceiverKernel,
            seed,
            gcb_abi::kModeDrainUntilStopped,
            0,
            kGcbSmokeWaitBudgetCycles,
            schedule)});

    const std::vector<uint32_t> sender_status = smoke.run_sender(kGcbSmokeWaitBudgetCycles, schedule);
    EXPECT_EQ(sender_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    EXPECT_EQ(smoke.read_status(smoke.receiver_cce(0))[gcb_abi::kStatusMagic], gcb_abi::kStatusPoison)
        << "the receiver exited before the host stopped it";
    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeStopped);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], gcb_smoke_total_pages(schedule));
    const GcbFifoEndState end_state = expected_gcb_fifo_end_state(schedule, kGcbSmokeFifoBytes);
    smoke.expect_fifo_state(0, end_state, end_state.credit_units, end_state.fifo_offset);
}

// The receiver pops some pages, then holds a full FIFO, so the sender's free-space wait expires with CREDIT_TIMEOUT.
// Released, the receiver exits without acking the held pages and without writing rd_ptr back: rd_ptr in TL1 stays at
// the FIFO start although the kernel had moved it past the popped pages.
TEST_F(MimirPrefetcher, MM7GcbSmokeStallUntilReleased) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    constexpr uint32_t kPageBytes = 256;
    constexpr uint32_t kPoppedPages = 4;
    constexpr uint32_t kFifoPages = kGcbSmokeFifoBytes / kPageBytes;
    const std::vector<GcbSmokeRequest> popped_schedule = {
        {.tensor_id = 9, .slab = 0, .page_bytes = kPageBytes, .num_pages = kPoppedPages}};
    const std::vector<GcbSmokeRequest> credited_schedule = {
        {.tensor_id = 9, .slab = 0, .page_bytes = kPageBytes, .num_pages = kPoppedPages + kFifoPages}};
    const std::vector<GcbSmokeRequest> sender_schedule = {
        {.tensor_id = 9, .slab = 0, .page_bytes = kPageBytes, .num_pages = kPoppedPages + kFifoPages + 8}};
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0,
            kGcbDiscardReceiverKernel,
            0,
            gcb_abi::kModeStallUntilReleased,
            kFifoPages,
            kGcbSmokeWaitBudgetCycles,
            popped_schedule)});

    const GcbFifoEndState credited = expected_gcb_fifo_end_state(credited_schedule, kGcbSmokeFifoBytes);
    const GcbFifoEndState popped = expected_gcb_fifo_end_state(popped_schedule, kGcbSmokeFifoBytes);
    const std::vector<uint32_t> sender_status = smoke.run_sender(kGcbSmokeStallBudgetCycles, sender_schedule);
    EXPECT_EQ(sender_status[gcb_abi::kStatusCode], gcb_abi::kCodeCreditTimeout);
    EXPECT_EQ(sender_status[gcb_abi::kStatusPagesDone], kPoppedPages + kFifoPages);
    EXPECT_EQ(sender_status[gcb_abi::kStatusWaitReceiver], 0u);
    EXPECT_EQ(sender_status[gcb_abi::kStatusWaitPagesSent], credited.credit_units);
    EXPECT_EQ(sender_status[gcb_abi::kStatusWaitPagesAcked], popped.credit_units);
    ASSERT_TRUE(smoke.wait_for_holding(0, kFifoPages));
    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeReleased);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], kPoppedPages);
    ASSERT_NE(popped.fifo_offset, 0u);
    smoke.expect_fifo_state(0, credited, popped.credit_units, 0);
}

// The receiver holds everything the sender sent, so the sender's final barrier expires with DRAIN_TIMEOUT. Then the
// host releases the receiver.
TEST_F(MimirPrefetcher, MM7GcbSmokeDrainTimeout) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    constexpr uint32_t kPageBytes = 256;
    constexpr uint32_t kSentPages = 8;
    const std::vector<GcbSmokeRequest> sender_schedule = {
        {.tensor_id = 13, .slab = 0, .page_bytes = kPageBytes, .num_pages = kSentPages}};
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0,
            kGcbDiscardReceiverKernel,
            0,
            gcb_abi::kModeStallUntilReleased,
            kSentPages,
            kGcbSmokeWaitBudgetCycles,
            {{.tensor_id = 13, .slab = 0, .page_bytes = kPageBytes, .num_pages = 0}})});

    const std::vector<uint32_t> sender_status = smoke.run_sender(kGcbSmokeStallBudgetCycles, sender_schedule);
    EXPECT_EQ(sender_status[gcb_abi::kStatusCode], gcb_abi::kCodeDrainTimeout);
    EXPECT_EQ(sender_status[gcb_abi::kStatusPagesDone], kSentPages);
    ASSERT_TRUE(smoke.wait_for_holding(0, kSentPages));
    receivers.stop_and_wait();
    EXPECT_EQ(smoke.read_receiver_status(0)[gcb_abi::kStatusCode], gcb_abi::kCodeReleased);
    smoke.expect_fifo_state(0, expected_gcb_fifo_end_state(sender_schedule, kGcbSmokeFifoBytes), 0, 0);
}

// No sender runs. The host credits a few pages, the receiver drains them, and its next front wait expires; it reports
// CREDIT_TIMEOUT and writes rd_ptr back past the drained pages.
TEST_F(MimirPrefetcher, MM7GcbSmokeReceiverWaitTimeout) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    constexpr uint32_t kPageBytes = 256;
    constexpr uint32_t kCreditedPages = 4;
    const GcbFifoEndState credited = expected_gcb_fifo_end_state(
        {{.tensor_id = 14, .slab = 0, .page_bytes = kPageBytes, .num_pages = kCreditedPages}}, kGcbSmokeFifoBytes);
    smoke.credit_from_host(0, credited.credit_units);
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0,
            kGcbDiscardReceiverKernel,
            0,
            gcb_abi::kModeDrainCount,
            0,
            kGcbSmokeStallBudgetCycles,
            {{.tensor_id = 14, .slab = 0, .page_bytes = kPageBytes, .num_pages = 2 * kCreditedPages}})});

    receivers.wait_for_exit();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeCreditTimeout);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], kCreditedPages);
    smoke.expect_fifo_state(
        0, {.credit_units = credited.credit_units, .fifo_offset = 0}, credited.credit_units, credited.fifo_offset);
}

// Two sender and validator runs on one GCB: the second pair resumes from the wr_ptr and rd_ptr the first left in TL1,
// and its 512 B pages pad from the first run's 1280 B to 1536 B.
TEST_F(MimirPrefetcher, MM7GcbSmokeRelaunchResumes) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    const std::vector<GcbSmokeRequest> first_schedule = {
        {.tensor_id = 11, .slab = 0, .page_bytes = 256, .num_pages = 5}};
    const std::vector<GcbSmokeRequest> second_schedule = {
        {.tensor_id = 12, .slab = 0, .page_bytes = 512, .num_pages = 6}};

    CceReceiverProgram first_receivers(smoke.device(), {validator_launch(smoke, 0, first_schedule)});
    EXPECT_EQ(smoke.run_sender(kGcbSmokeWaitBudgetCycles, first_schedule)[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    first_receivers.stop_and_wait();
    EXPECT_EQ(smoke.read_receiver_status(0)[gcb_abi::kStatusCode], gcb_abi::kCodeOk);

    smoke.reset_kernel_blocks();
    CceReceiverProgram second_receivers(smoke.device(), {validator_launch(smoke, 0, second_schedule)});
    EXPECT_EQ(smoke.run_sender(kGcbSmokeWaitBudgetCycles, second_schedule)[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    second_receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], gcb_smoke_total_pages(second_schedule));

    std::vector<GcbSmokeRequest> both_schedules = first_schedule;
    both_schedules.insert(both_schedules.end(), second_schedule.begin(), second_schedule.end());
    const GcbFifoEndState end_state = expected_gcb_fifo_end_state(both_schedules, kGcbSmokeFifoBytes);
    smoke.expect_fifo_state(0, end_state, end_state.credit_units, end_state.fifo_offset);
}

// Deferred from MM7Gate: the gate checks firmware presence, not idleness, so it stays ok while a receiver Program runs
// on another bank's sender CCE.
TEST_F(MimirPrefetcher, MM7GcbSmokePreconditionsWithReceiverRunning) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    const std::optional<uint32_t> other_sender_bank = lifecycle_other_sender_bank(mesh);
    ASSERT_TRUE(other_sender_bank.has_value()) << "no DRAM view has a sender CCE other than bank 0's";
    const uint32_t other_sender_cce = gcb_layout_sender_cce(device, mesh, *other_sender_bank);

    GcbSmoke smoke(mesh, 0, {other_sender_cce});
    const std::vector<GcbSmokeRequest> schedule = {{.tensor_id = 10, .slab = 0, .page_bytes = 256, .num_pages = 0}};
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0, kGcbDiscardReceiverKernel, 0, gcb_abi::kModeDrainUntilStopped, 0, kGcbSmokeWaitBudgetCycles, schedule)});
    EXPECT_EQ(read_go_signal(device, other_sender_cce), dev_msgs::RUN_MSG_GO) << "CCE " << other_sender_cce;
    const distributed::TensorPrefetcherPreconditions preconditions =
        distributed::TensorPrefetcherManagerTestAccess::preconditions(mesh);
    EXPECT_TRUE(preconditions.ok) << preconditions.reason;

    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeStopped);
}

// A Tensor prefetcher session on the CCE senders of `sender_banks`. The default stage ring's third holds the widest
// K-row the MM7 tensors use. Declare it after any CceReceiverProgram it serves, so it stops first and Stop's DRAIN sees
// live receivers. The destructor stops through the no-throw path, which only logs a failure; call stop() to check the
// outcome.
class PrefetcherSession {
public:
    PrefetcherSession(
        distributed::MeshDevice& mesh,
        std::vector<uint32_t> sender_banks,
        uint64_t wait_budget_cycles = kGcbSmokeWaitBudgetCycles,
        uint32_t stage_ring_bytes = experimental::QuasarTensorPrefetcherConfig{}.stage_ring_bytes) :
        mesh_(mesh) {
        experimental::StartTensorPrefetcher(
            mesh_,
            experimental::TensorPrefetcherConfig{
                .quasar = experimental::QuasarTensorPrefetcherConfig{
                    .dram_sender_banks = std::move(sender_banks),
                    .wait_budget_cycles = wait_budget_cycles,
                    .stage_ring_bytes = stage_ring_bytes}});
        active_ = true;
    }
    PrefetcherSession(const PrefetcherSession&) = delete;
    PrefetcherSession& operator=(const PrefetcherSession&) = delete;

    ~PrefetcherSession() {
        if (!active_) {
            return;
        }
        mesh_.impl().tensor_prefetcher(&mesh_).stop_without_throwing();
    }

    void stop() {
        active_ = false;
        experimental::StopTensorPrefetcher(mesh_);
    }

private:
    distributed::MeshDevice& mesh_;
    bool active_ = false;
};

CceSenderStatus lifecycle_sender_status(distributed::MeshDevice& mesh, uint32_t cce_index) {
    const std::vector<uint32_t> words = read_cce_tl1(
        mesh.get_devices().front(),
        cce_index,
        distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh) + kCceSenderStatusOffset,
        sizeof(CceSenderStatus));
    CceSenderStatus status{};
    std::memcpy(&status, words.data(), sizeof(status));
    log_info(tt::LogTest, "MM7Lifecycle sender CCE {}: {}", cce_index, distributed::format_cce_sender_status(status));
    return status;
}

uint32_t lifecycle_bytes_acked(distributed::MeshDevice& mesh, uint32_t cce_index) {
    return read_cce_tl1(
               mesh.get_devices().front(),
               cce_index,
               distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh) + kCceRingMetaOffset +
                   offsetof(RingMeta, bytes_acked),
               sizeof(uint32_t))
        .front();
}

// A sender that took `requests` pages, the last a STOP, and exited cleanly.
void expect_lifecycle_sender_stopped(distributed::MeshDevice& mesh, uint32_t cce_index, uint32_t requests) {
    const CceSenderStatus status = lifecycle_sender_status(mesh, cce_index);
    EXPECT_EQ(status.state, static_cast<uint32_t>(CceSenderState::Stopped)) << "sender CCE " << cce_index;
    EXPECT_EQ(status.code, static_cast<uint32_t>(CceSenderStatusCode::Ok)) << "sender CCE " << cce_index;
    EXPECT_EQ(status.request_index, requests - 1) << "sender CCE " << cce_index;
    EXPECT_EQ(lifecycle_bytes_acked(mesh, cce_index), requests * kRequestPageBytes) << "sender CCE " << cce_index;
}

// The page the DRAIN-at-Stop case credits without data.
inline constexpr uint32_t kLifecycleDrainPageBytes = 256;
// A request-page word whose command byte names no command.
inline constexpr uint32_t kLifecycleBadCommandWord = 0xFFFF'FFFF;

// Writes a non-zero pattern over the part of `cce_index`'s control block Start zeroes, and returns it.
std::vector<uint32_t> fill_lifecycle_control_block(distributed::MeshDevice& mesh, uint32_t cce_index) {
    std::vector<uint32_t> pattern(kCceSenderZeroedBytes / sizeof(uint32_t));
    for (uint32_t word = 0; word < pattern.size(); ++word) {
        pattern[word] = kGcbLayoutFillPattern | word;
    }
    write_cce_tl1(
        mesh.get_devices().front(),
        cce_index,
        distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh),
        pattern);
    return pattern;
}

template <typename Action>
std::string lifecycle_error(Action action) {
    try {
        action();
    } catch (const std::exception& error) {
        return error.what();
    }
    return "did not throw";
}

// How long a test watches a sender held in WAIT_CQ or DRAIN. The status reads in that time advance the emulated chip
// clock by about 400,000 cycles (S5), far more than a wait already satisfied takes to pass.
inline constexpr std::chrono::seconds kSenderHoldTime{5};

// Reads the sender's status back to back, which is what advances the emulated chip clock, until `reached` holds for it
// or `limit` passes. Returns the last status read.
CceSenderStatus poll_sender_status(
    distributed::MeshDevice& mesh,
    uint32_t cce_index,
    const std::function<bool(const CceSenderStatus&)>& reached,
    std::chrono::steady_clock::duration limit) {
    IDevice* device = mesh.get_devices().front();
    const uint64_t status_address =
        distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh) + kCceSenderStatusOffset;
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + limit;
    CceSenderStatus status{};
    do {
        const std::vector<uint32_t> words = read_cce_tl1(device, cce_index, status_address, sizeof(CceSenderStatus));
        std::memcpy(&status, words.data(), sizeof(status));
    } while (!reached(status) && std::chrono::steady_clock::now() < deadline);
    log_info(tt::LogTest, "MM7 sender CCE {}: {}", cce_index, distributed::format_cce_sender_status(status));
    return status;
}

bool sender_exited(const CceSenderStatus& status) {
    return status.state == static_cast<uint32_t>(CceSenderState::Stopped) ||
           status.state == static_cast<uint32_t>(CceSenderState::Failed);
}

// Deferred from MM7Gate: IsTensorPrefetcherSupported reports the port supported, and Start and Stop work.
TEST_F(MimirPrefetcher, MM7GateStartStopWorks) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    ASSERT_TRUE(experimental::IsTensorPrefetcherSupported(mesh));
    PrefetcherSession session(mesh, {0});
    session.stop();
}

// Start with no requests: the sender runs until STOP, then reports Stopped with OK.
TEST_F(MimirPrefetcher, MM7LifecycleStartStop) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    const uint32_t sender_cce = gcb_layout_sender_cce(device, mesh, 0);
    log_info(
        tt::LogTest,
        "MM7Lifecycle sender CCE {} control block 0x{:x}",
        sender_cce,
        distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh));

    PrefetcherSession session(mesh, {0});
    EXPECT_EQ(read_go_signal(device, sender_cce), dev_msgs::RUN_MSG_GO) << "CCE " << sender_cce;
    session.stop();
    expect_lifecycle_sender_stopped(mesh, sender_cce, 1);
}

// Two sessions in one mesh. Between them the CQ slots, ring meta and status get a non-zero pattern and ring slot 0,
// which still holds the first session's STOP, gets an unknown command. The second Start zeroes the control block, so
// its sender waits for the host's STOP in slot 0 and reads like a fresh one: one request, CQ slots still zero. A Start
// that skipped the zeroing would leave bytes_sent non-zero, so the sender would read slot 0 before the host's STOP and
// fail with BadCmd.
TEST_F(MimirPrefetcher, MM7LifecycleStartStopTwice) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    const uint32_t sender_cce = gcb_layout_sender_cce(device, mesh, 0);
    const uint32_t control_base = distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh);

    PrefetcherSession first_session(mesh, {0});
    first_session.stop();
    expect_lifecycle_sender_stopped(mesh, sender_cce, 1);

    fill_lifecycle_control_block(mesh, sender_cce);
    write_cce_tl1(
        device,
        sender_cce,
        control_base + kCceRequestRingOffset,
        std::vector<uint32_t>(kRequestPageBytes / sizeof(uint32_t), kLifecycleBadCommandWord));
    PrefetcherSession second_session(mesh, {0});
    second_session.stop();
    expect_lifecycle_sender_stopped(mesh, sender_cce, 1);
    EXPECT_EQ(
        read_cce_tl1(
            device, sender_cce, control_base + kCceRingMetaOffset + offsetof(RingMeta, bytes_sent), sizeof(uint32_t))
            .front(),
        kRequestPageBytes);
    const std::vector<uint32_t> cq_slots = read_cce_tl1(
        device, sender_cce, control_base + kCceCqSignalSlotsOffset, kNumCqSignalSlots * kCceCqSignalSlotStrideBytes);
    EXPECT_EQ(cq_slots, std::vector<uint32_t>(cq_slots.size(), 0u));
}

// A receiver Program runs on another bank's sender CCE. Start over both banks fails on that CCE before it launches or
// zeroes anything: both control blocks keep the pattern written before, and bank 0's CCE stays idle.
TEST_F(MimirPrefetcher, MM7LifecycleBusySenderFailsStart) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    const std::optional<uint32_t> busy_bank = lifecycle_other_sender_bank(mesh);
    if (!busy_bank.has_value()) {
        GTEST_SKIP();
    }
    const uint32_t idle_sender_cce = gcb_layout_sender_cce(device, mesh, 0);
    const uint32_t busy_sender_cce = gcb_layout_sender_cce(device, mesh, *busy_bank);

    const uint32_t control_base = distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh);
    const std::vector<uint32_t> pattern = fill_lifecycle_control_block(mesh, idle_sender_cce);
    fill_lifecycle_control_block(mesh, busy_sender_cce);

    GcbSmoke smoke(mesh, 0, {busy_sender_cce});
    CceReceiverProgram receivers(
        smoke.device(),
        {smoke.receiver_launch(
            0,
            kGcbDiscardReceiverKernel,
            0,
            gcb_abi::kModeDrainUntilStopped,
            0,
            kGcbSmokeWaitBudgetCycles,
            {{.tensor_id = 15, .slab = 0, .page_bytes = 256, .num_pages = 0}})});

    const std::string start_error =
        lifecycle_error([&mesh, &busy_bank]() { PrefetcherSession(mesh, {0, *busy_bank}); });
    log_info(tt::LogTest, "MM7Lifecycle Start with CCE {} busy: {}", busy_sender_cce, start_error);
    EXPECT_NE(start_error.find(fmt::format("sender CCE {} ", busy_sender_cce)), std::string::npos) << start_error;
    EXPECT_FALSE(mesh.impl().tensor_prefetcher(&mesh).is_active());
    EXPECT_EQ(read_go_signal(device, idle_sender_cce), dev_msgs::RUN_MSG_DONE) << "CCE " << idle_sender_cce;
    for (const uint32_t cce_index : {idle_sender_cce, busy_sender_cce}) {
        EXPECT_EQ(read_cce_tl1(device, cce_index, control_base, kCceSenderZeroedBytes), pattern)
            << "CCE " << cce_index << " control block changed";
    }

    receivers.stop_and_wait();
    EXPECT_EQ(smoke.read_receiver_status(0)[gcb_abi::kStatusCode], gcb_abi::kCodeStopped);
}

// Bank 0's GCB delivers to the CCE that is the other provisioned bank's sender, so Queue fails before anything is
// queued. The tensor list is empty: the receiver check comes first, so no PREFETCH data is needed.
TEST_F(MimirPrefetcher, MM7LifecycleQueueRejectsReceiverOnSender) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    const std::optional<uint32_t> other_bank = lifecycle_other_sender_bank(mesh);
    if (!other_bank.has_value()) {
        GTEST_SKIP();
    }
    const uint32_t other_sender_cce = gcb_layout_sender_cce(device, mesh, *other_bank);
    GcbSmoke smoke(mesh, 0, {other_sender_cce});

    PrefetcherSession session(mesh, {0, *other_bank});
    const std::string queue_error = lifecycle_error(
        [&mesh, &smoke]() { experimental::QueueTensorPrefetcherRequest(mesh, smoke.gcb(), std::nullopt, {}); });
    log_info(tt::LogTest, "MM7Lifecycle Queue to CCE {}: {}", other_sender_cce, queue_error);
    EXPECT_NE(queue_error.find(fmt::format("receiver CCE {} ", other_sender_cce)), std::string::npos) << queue_error;
    EXPECT_NE(queue_error.find("provisioned sender CCEs"), std::string::npos) << queue_error;

    session.stop();
    expect_lifecycle_sender_stopped(mesh, gcb_layout_sender_cce(device, mesh, 0), 1);
    expect_lifecycle_sender_stopped(mesh, other_sender_cce, 1);
}

// A WAIT_CQ for CQ 0's next value, queued without its write, holds the sender. WaitForCqOnTensorPrefetcher then writes
// the value after it through the command queue's own path and queues a WAIT_CQ for that one, which releases both: the
// sender reports the STOP as its third request, and CQ 0's slot holds 2. A WAIT_CQ that did not wait would leave the
// sender Running during the hold; host and kernel disagreeing on the slot would end it in CQ_TIMEOUT.
TEST_F(MimirPrefetcher, MM7LifecycleWaitCq) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    const uint32_t sender_cce = gcb_layout_sender_cce(mesh.get_devices().front(), mesh, 0);
    distributed::MeshCommandQueue& command_queue = mesh.mesh_command_queue(0);

    // queue_cq_wait_without_signal waits on CQ 0.
    ASSERT_EQ(command_queue.id(), 0u);
    PrefetcherSession session(mesh, {0});
    const distributed::TensorPrefetcherManagerTestAccess::UnsignaledCqWait held_wait =
        distributed::TensorPrefetcherManagerTestAccess::queue_cq_wait_without_signal(
            mesh.impl().tensor_prefetcher(&mesh));
    const CceSenderStatus waiting = poll_sender_status(
        mesh,
        sender_cce,
        [](const CceSenderStatus& status) {
            return status.state != static_cast<uint32_t>(CceSenderState::NotStarted) &&
                   status.state != static_cast<uint32_t>(CceSenderState::Running);
        },
        kGcbSmokeHostWaitLimit);
    ASSERT_EQ(waiting.state, static_cast<uint32_t>(CceSenderState::WaitingCq));
    const CceSenderStatus held = poll_sender_status(
        mesh,
        sender_cce,
        [](const CceSenderStatus& status) { return status.state != static_cast<uint32_t>(CceSenderState::WaitingCq); },
        kSenderHoldTime);
    EXPECT_EQ(held.state, static_cast<uint32_t>(CceSenderState::WaitingCq));

    log_info(
        tt::LogTest, "MM7Lifecycle WAIT_CQ held for value {}; signalling through the command queue", held_wait.value);
    experimental::WaitForCqOnTensorPrefetcher(command_queue, std::nullopt);
    session.stop();
    expect_lifecycle_sender_stopped(mesh, sender_cce, 3);
    EXPECT_EQ(
        read_cce_tl1(
            mesh.get_devices().front(),
            sender_cce,
            distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh) + kCceCqSignalSlotsOffset +
                (command_queue.id() * kCceCqSignalSlotStrideBytes),
            sizeof(uint32_t))
            .front(),
        held_wait.value + 1);
}

// Stop's DRAIN waits for the receivers to ack everything the sender credited. The host credits one page with no
// receiver running, so nothing acks it: the sender holds in Draining while Stop blocks on another thread. The host then
// acks the page as a receiver would, and the sender passes DRAIN and reports the STOP as its second request.
TEST_F(MimirPrefetcher, MM7LifecycleDrainAtStop) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    const uint32_t sender_cce = gcb_layout_sender_cce(smoke.device(), mesh, 0);
    const uint32_t page_units = kLifecycleDrainPageBytes / kDramReceiverCreditUnitBytes;
    PrefetcherSession session(mesh, {0});
    distributed::TensorPrefetcherManagerTestAccess::register_drain_target(
        mesh.impl().tensor_prefetcher(&mesh), smoke.gcb());
    smoke.credit_from_host(0, page_units);

    std::string stop_error;
    std::thread stopper([&session, &stop_error]() { stop_error = lifecycle_error([&session]() { session.stop(); }); });
    // Joins on every exit path: destroying a joinable thread terminates.
    struct StopperJoiner {
        std::thread& thread;
        ~StopperJoiner() {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } stopper_joiner{stopper};
    const CceSenderStatus draining = poll_sender_status(
        mesh,
        sender_cce,
        [](const CceSenderStatus& status) {
            return status.state != static_cast<uint32_t>(CceSenderState::NotStarted) &&
                   status.state != static_cast<uint32_t>(CceSenderState::Running);
        },
        kGcbSmokeHostWaitLimit);
    EXPECT_EQ(draining.state, static_cast<uint32_t>(CceSenderState::Draining));
    const CceSenderStatus held = poll_sender_status(
        mesh,
        sender_cce,
        [](const CceSenderStatus& status) { return status.state != static_cast<uint32_t>(CceSenderState::Draining); },
        kSenderHoldTime);
    EXPECT_EQ(held.state, static_cast<uint32_t>(CceSenderState::Draining));

    smoke.ack_from_host(0, page_units);
    stopper.join();
    EXPECT_EQ(stop_error, "did not throw");
    expect_lifecycle_sender_stopped(mesh, sender_cce, 2);
}

// T1: one PREFETCH route per test. A sender CCE streams one 4 KiB page of a tensor on its own DRAM view into a
// validator receiver, then a second session in the same mesh streams another tensor's page into the same GCB.

inline constexpr uint32_t kUint32TileWords = tt::constants::TILE_HW;
inline constexpr uint32_t kUint32TilePageBytes = kUint32TileWords * sizeof(uint32_t);
inline constexpr uint32_t kT1Sessions = 2;
// One PREFETCH page, then Stop's DRAIN and STOP.
inline constexpr uint32_t kRequestsForOnePrefetch = 3;

std::optional<uint32_t> bank_with_sender_cce(distributed::MeshDevice& mesh, uint32_t cce_index) {
    IDevice* device = mesh.get_devices().front();
    for (uint32_t bank = 0; bank < soc_desc_of(device).get_num_dram_views(); ++bank) {
        if (gcb_layout_sender_cce(device, mesh, bank) == cce_index) {
            return bank;
        }
    }
    return std::nullopt;
}

// Why a test whose senders and receivers are `route_cces` cannot run on this descriptor, or empty.
std::string route_skip_reason(const std::vector<uint32_t>& route_cces) {
    if (std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        return skip_reason;
    }
    const uint32_t num_cces = topology_from(configured_soc_desc()).num_cces;
    const uint32_t highest_cce = *std::max_element(route_cces.begin(), route_cces.end());
    if (highest_cce >= num_cces) {
        return fmt::format("Descriptor exposes {} CCEs; this route needs CCE {}.", num_cces, highest_cce);
    }
    return "";
}

bool route_spans_mimirs(const std::vector<uint32_t>& route_cces) {
    for (const uint32_t cce_index : route_cces) {
        if (mimir_index_for_cce(cce_index) != mimir_index_for_cce(route_cces.front())) {
            return true;
        }
    }
    return false;
}

// Skip when route_skip_reason() is non-empty; a route that spans Mimirs then fails on a remote preflight mismatch.
// Variadic so a braced route such as {0, 1} passes through.
#define MIMIR_ROUTE_GUARD(...)                                                                \
    if (const std::string route_skip = route_skip_reason(__VA_ARGS__); !route_skip.empty()) { \
        GTEST_SKIP() << route_skip;                                                           \
    }                                                                                         \
    if (route_spans_mimirs(__VA_ARGS__)) {                                                    \
        MIMIR_REQUIRE_REMOTE_PREFLIGHT();                                                     \
    }

// A UINT32 TILE tensor one tile wide on one DRAM bank: `num_slabs` receiver slabs, each `blocks` blocks of
// `block_tiles` tiles. Slab s is shard s, so tile t of slab s is page s * slab_tiles() + t. A block is one validator
// page.
struct SlabTensorGeometry {
    uint32_t num_slabs = 1;
    uint32_t blocks = 1;
    uint32_t block_tiles = 1;

    uint32_t slab_tiles() const { return blocks * block_tiles; }
    uint32_t block_bytes() const { return block_tiles * kUint32TilePageBytes; }
};

TensorSpec single_bank_tensor_spec(uint32_t bank, const SlabTensorGeometry& geometry) {
    const CoreCoord bank_core = configured_soc_desc().get_logical_core_for_dram_view(bank);
    const CoreRangeSet bank_cores(CoreRange(bank_core, bank_core));
    const Shape shard_shape{32 * geometry.slab_tiles(), 32};
    const Shape tensor_shape{32 * geometry.slab_tiles() * geometry.num_slabs, 32};
    const MemoryConfig memory_config(
        BufferType::DRAM, NdShardSpec{shard_shape, bank_cores, ShardOrientation::ROW_MAJOR});
    return TensorSpec(tensor_shape, TensorLayout(DataType::UINT32, Layout::TILE, memory_config));
}

// A single_bank_tensor_spec tensor holding the validator's words for page p of slab s of `tensor_id` in block p of slab
// s.
MeshTensor write_single_bank_tensor(
    distributed::MeshDevice& mesh, uint32_t bank, uint32_t tensor_id, const SlabTensorGeometry& geometry) {
    const TensorSpec spec = single_bank_tensor_spec(bank, geometry);
    const uint32_t seed = mimir_test_seed();
    const uint32_t slab_words = geometry.slab_tiles() * kUint32TileWords;
    const uint32_t block_words = geometry.block_bytes() / sizeof(uint32_t);
    std::vector<uint32_t> words(geometry.num_slabs * slab_words);
    for (uint32_t slab = 0; slab < geometry.num_slabs; ++slab) {
        for (uint32_t word = 0; word < slab_words; ++word) {
            words[(slab * slab_words) + word] =
                gcb_abi::gcb_page_word(seed, tensor_id, slab, word / block_words, word % block_words);
        }
    }
    DistributedHostBuffer host_buffer = DistributedHostBuffer::create(mesh.shape());
    const distributed::MeshCoordinate first_coord = *distributed::MeshCoordinateRange(mesh.shape()).begin();
    host_buffer.emplace_shard(first_coord, [&words]() { return HostBuffer(std::vector<uint32_t>(words)); });
    const HostTensor host_tensor = host_tensor_from_buffer_with_topology(
        std::move(host_buffer), spec, TensorTopology::create_sharded_tensor_topology(mesh.shape()));
    distributed::MeshCommandQueue& command_queue = mesh.mesh_command_queue();
    MeshTensor tensor = command_queue.enqueue_write_tensor(host_tensor);
    command_queue.finish();
    log_info(
        tt::LogTest,
        "MM7 tensor {} on DRAM bank {}: {} slabs of {} blocks of {} tiles, bank-local address 0x{:x}",
        tensor_id,
        bank,
        geometry.num_slabs,
        geometry.blocks,
        geometry.block_tiles,
        tensor.mesh_buffer().address());
    return tensor;
}

// The validator schedule entry for slab 0 of a write_single_bank_tensor tensor; receiver_launch moves it to the
// receiver's slab.
GcbSmokeRequest slab_tensor_request(uint32_t tensor_id, const SlabTensorGeometry& geometry) {
    return {.tensor_id = tensor_id, .slab = 0, .page_bytes = geometry.block_bytes(), .num_pages = geometry.blocks};
}

// A validator on receiver table slot `slot` of `smoke` that consumed `schedule`: OK, every page, and both ends'
// counters and pointers where the schedule leaves them.
void expect_validated_delivery(const GcbSmoke& smoke, uint32_t slot, const std::vector<GcbSmokeRequest>& schedule) {
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(slot);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk) << "receiver CCE " << smoke.receiver_cce(slot);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], gcb_smoke_total_pages(schedule))
        << "receiver CCE " << smoke.receiver_cce(slot);
    const GcbFifoEndState end_state = expected_gcb_fifo_end_state(schedule, smoke.fifo_bytes());
    smoke.expect_fifo_state(slot, end_state, end_state.credit_units, end_state.fifo_offset);
}

// Writes one tensor per session, then runs kT1Sessions sessions of Start, one PREFETCH and Stop into one validator
// receiver Program that spans them all.
void run_t1_route(uint32_t sender_cce, uint32_t receiver_cce) {
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    if (mimir_index_for_cce(sender_cce) != mimir_index_for_cce(receiver_cce)) {
        EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    const std::optional<uint32_t> sender_bank = bank_with_sender_cce(mesh, sender_cce);
    ASSERT_TRUE(sender_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << sender_cce;
    log_info(tt::LogTest, "MM7T1 route CCE {} (bank {}) -> CCE {}", sender_cce, *sender_bank, receiver_cce);

    std::vector<MeshTensor> tensors;
    std::vector<GcbSmokeRequest> schedule;
    for (uint32_t session_index = 0; session_index < kT1Sessions; ++session_index) {
        tensors.push_back(write_single_bank_tensor(mesh, *sender_bank, session_index, {}));
        schedule.push_back({.tensor_id = session_index, .slab = 0, .page_bytes = kUint32TilePageBytes, .num_pages = 1});
    }
    GcbSmoke smoke(mesh, *sender_bank, {receiver_cce});
    CceReceiverProgram receivers(device, {validator_launch(smoke, 0, schedule)});

    for (uint32_t session_index = 0; session_index < kT1Sessions; ++session_index) {
        PrefetcherSession session(mesh, {*sender_bank});
        queue_tensor(mesh, smoke, tensors[session_index], 1);
        session.stop();
        expect_lifecycle_sender_stopped(mesh, sender_cce, kRequestsForOnePrefetch);
    }
    EXPECT_EQ(read_go_signal(device, sender_cce), dev_msgs::RUN_MSG_DONE) << "sender CCE " << sender_cce;

    receivers.stop_and_wait();
    expect_validated_delivery(smoke, 0, schedule);
}

TEST_F(MimirPrefetcher, MM7T1Local0To1) {
    MIMIR_ROUTE_GUARD({0, 1});
    run_t1_route(0, 1);
}

TEST_F(MimirPrefetcher, MM7T1Local3To2) {
    MIMIR_ROUTE_GUARD({3, 2});
    run_t1_route(3, 2);
}

TEST_F(MimirPrefetcher, MM7T1D2d0To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({0, 2});
    run_t1_route(0, 2);
}

TEST_F(MimirPrefetcher, MM7T1D2d3To1) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({3, 1});
    run_t1_route(3, 1);
}

// T2: one sender to several receivers, and two senders to one receiver CCE, on routes over CCEs 0-3. Each slab is two
// one-tile blocks against a one-tile FIFO, so the sender's second round waits for every receiver of the GCB to ack the
// first.

inline constexpr uint32_t kT2BlocksPerSlab = 2;
inline constexpr uint32_t kT2FanOutTensorId = 40;
inline constexpr uint32_t kT2FirstTensorId = 41;
inline constexpr uint32_t kT2SecondTensorId = 42;
// The hart of the second GCB's validator on a receiver CCE that serves two GCBs.
inline constexpr uint32_t kSecondGcbReceiverHart = 1;

// One GCB from `sender_cce` to every CCE of `receiver_cces`, each reading its own slab of one tensor.
void run_t2_fan_out(uint32_t sender_cce, const std::vector<uint32_t>& receiver_cces) {
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    const std::optional<uint32_t> sender_bank = bank_with_sender_cce(mesh, sender_cce);
    ASSERT_TRUE(sender_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << sender_cce;
    log_info(
        tt::LogTest,
        "MM7T2 fan-out CCE {} (bank {}) -> CCEs {}",
        sender_cce,
        *sender_bank,
        fmt::join(receiver_cces, ","));

    const SlabTensorGeometry geometry{
        .num_slabs = static_cast<uint32_t>(receiver_cces.size()), .blocks = kT2BlocksPerSlab};
    const MeshTensor tensor = write_single_bank_tensor(mesh, *sender_bank, kT2FanOutTensorId, geometry);
    const std::vector<GcbSmokeRequest> schedule = {slab_tensor_request(kT2FanOutTensorId, geometry)};
    GcbSmoke smoke(mesh, *sender_bank, receiver_cces);
    std::vector<CceKernelLaunch> launches;
    for (uint32_t slot = 0; slot < receiver_cces.size(); ++slot) {
        launches.push_back(validator_launch(smoke, slot, schedule));
    }
    CceReceiverProgram receivers(smoke.device(), launches);
    PrefetcherSession session(mesh, {*sender_bank});
    queue_tensor(mesh, smoke, tensor, geometry.blocks);
    session.stop();
    expect_lifecycle_sender_stopped(mesh, sender_cce, kRequestsForOnePrefetch);

    receivers.stop_and_wait();
    for (uint32_t slot = 0; slot < receiver_cces.size(); ++slot) {
        expect_validated_delivery(smoke, slot, schedule);
    }
}

// Two senders in one session, each with its own GCB to `receiver_cce`, deliver concurrently. The GCBs hold disjoint
// windows of the receiver zone, and their validators run on two harts of the receiver CCE in one Program.
void run_t2_two_senders(uint32_t first_sender_cce, uint32_t second_sender_cce, uint32_t receiver_cce) {
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    const std::optional<uint32_t> first_bank = bank_with_sender_cce(mesh, first_sender_cce);
    const std::optional<uint32_t> second_bank = bank_with_sender_cce(mesh, second_sender_cce);
    ASSERT_TRUE(first_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << first_sender_cce;
    ASSERT_TRUE(second_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << second_sender_cce;
    log_info(
        tt::LogTest,
        "MM7T2 CCE {} (bank {}) and CCE {} (bank {}) -> CCE {}",
        first_sender_cce,
        *first_bank,
        second_sender_cce,
        *second_bank,
        receiver_cce);

    const SlabTensorGeometry geometry{.blocks = kT2BlocksPerSlab};
    const MeshTensor first_tensor = write_single_bank_tensor(mesh, *first_bank, kT2FirstTensorId, geometry);
    const MeshTensor second_tensor = write_single_bank_tensor(mesh, *second_bank, kT2SecondTensorId, geometry);
    const std::vector<GcbSmokeRequest> first_schedule = {slab_tensor_request(kT2FirstTensorId, geometry)};
    const std::vector<GcbSmokeRequest> second_schedule = {slab_tensor_request(kT2SecondTensorId, geometry)};
    GcbSmoke first_smoke(mesh, *first_bank, {receiver_cce});
    GcbSmoke second_smoke(mesh, *second_bank, {receiver_cce});
    expect_gcb_layout_ranges_disjoint(first_smoke.gcb(), second_smoke.gcb());
    CceKernelLaunch second_launch = validator_launch(second_smoke, 0, second_schedule);
    second_launch.hart = kSecondGcbReceiverHart;
    CceReceiverProgram receivers(
        first_smoke.device(), {validator_launch(first_smoke, 0, first_schedule), second_launch});
    PrefetcherSession session(mesh, {*first_bank, *second_bank});
    queue_tensor(mesh, first_smoke, first_tensor, geometry.blocks);
    queue_tensor(mesh, second_smoke, second_tensor, geometry.blocks);
    session.stop();
    expect_lifecycle_sender_stopped(mesh, first_sender_cce, kRequestsForOnePrefetch);
    expect_lifecycle_sender_stopped(mesh, second_sender_cce, kRequestsForOnePrefetch);

    receivers.stop_and_wait();
    expect_validated_delivery(first_smoke, 0, first_schedule);
    expect_validated_delivery(second_smoke, 0, second_schedule);
}

TEST_F(MimirPrefetcher, MM7T2FanOut0To123) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({0, 1, 2, 3});
    run_t2_fan_out(0, {1, 2, 3});
}

TEST_F(MimirPrefetcher, MM7T2FanOut3To012) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({3, 0, 1, 2});
    run_t2_fan_out(3, {0, 1, 2});
}

TEST_F(MimirPrefetcher, MM7T2TwoSenders01To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({0, 1, 2});
    run_t2_two_senders(0, 1, 2);
}

TEST_F(MimirPrefetcher, MM7T2TwoSenders02To3) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({0, 2, 3});
    run_t2_two_senders(0, 2, 3);
}

// T3: what one sender delivers into one GCB, on the local route 0 -> 1 and the D2D route 0 -> 2. A tile is 4 KiB, and
// the default stage ring's third is 4 KiB. Each case writes its tensors before any receiver runs and never rewrites
// them. A repeated request gets its own tensor of the same geometry, so the validator checks bytes that request
// delivered, not ones an earlier request left in the FIFO.

inline constexpr uint32_t kT3SenderCce = 0;
inline constexpr uint32_t kT3LocalReceiverCce = 1;
inline constexpr uint32_t kT3D2dReceiverCce = 2;

struct CaseTensor {
    uint32_t tensor_id = 0;
    SlabTensorGeometry geometry;
};

struct SingleGcbCase {
    uint32_t fifo_bytes = kGcbSmokeFifoBytes;
    uint32_t stage_ring_bytes = experimental::QuasarTensorPrefetcherConfig{}.stage_ring_bytes;
    std::vector<CaseTensor> tensors;
    // One entry per Queue call: positions in `tensors`, in order.
    std::vector<std::vector<uint32_t>> queue_calls;
    // The PREFETCH pages the Queue calls make, plus Stop's DRAIN and STOP.
    uint32_t sender_requests = 0;
};

// The bank of kT3SenderCce; checks the remote remap first when `receiver_cce` is on another Mimir.
std::optional<uint32_t> t3_sender_bank(distributed::MeshDevice& mesh, uint32_t receiver_cce) {
    if (mimir_index_for_cce(kT3SenderCce) != mimir_index_for_cce(receiver_cce)) {
        EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    return bank_with_sender_cce(mesh, kT3SenderCce);
}

// Writes the case's tensors, makes each Queue call from kT3SenderCce in one session, and checks the validator on
// `receiver_cce` saw every tensor in order.
void run_single_gcb_case(uint32_t receiver_cce, const SingleGcbCase& test_case) {
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    const std::optional<uint32_t> sender_bank = t3_sender_bank(mesh, receiver_cce);
    ASSERT_TRUE(sender_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << kT3SenderCce;
    log_info(
        tt::LogTest,
        "MM7T3 route CCE {} (bank {}) -> CCE {}, FIFO {} B, stage ring {} B",
        kT3SenderCce,
        *sender_bank,
        receiver_cce,
        test_case.fifo_bytes,
        test_case.stage_ring_bytes);

    std::vector<MeshTensor> tensors;
    for (const CaseTensor& case_tensor : test_case.tensors) {
        tensors.push_back(write_single_bank_tensor(mesh, *sender_bank, case_tensor.tensor_id, case_tensor.geometry));
    }
    std::vector<GcbSmokeRequest> schedule;
    for (const std::vector<uint32_t>& queue_call : test_case.queue_calls) {
        for (const uint32_t position : queue_call) {
            const CaseTensor& case_tensor = test_case.tensors.at(position);
            schedule.push_back(slab_tensor_request(case_tensor.tensor_id, case_tensor.geometry));
        }
    }
    GcbSmoke smoke(mesh, *sender_bank, {receiver_cce}, test_case.fifo_bytes);
    CceReceiverProgram receivers(smoke.device(), {validator_launch(smoke, 0, schedule)});
    PrefetcherSession session(mesh, {*sender_bank}, kGcbSmokeWaitBudgetCycles, test_case.stage_ring_bytes);
    for (const std::vector<uint32_t>& queue_call : test_case.queue_calls) {
        std::vector<experimental::TensorPrefetcherInput> inputs;
        for (const uint32_t position : queue_call) {
            inputs.push_back(experimental::TensorPrefetcherInput{
                .tensor = tensors.at(position), .block_count = test_case.tensors.at(position).geometry.blocks});
        }
        experimental::QueueTensorPrefetcherRequest(mesh, smoke.gcb(), std::nullopt, inputs);
    }
    session.stop();
    expect_lifecycle_sender_stopped(mesh, kT3SenderCce, test_case.sender_requests);

    receivers.stop_and_wait();
    expect_validated_delivery(smoke, 0, schedule);
}

// One 16 KiB block against a 12 KiB stage third: the fit ladder takes 2-tile (8 KiB) sub-chunks, which do not divide
// the third, so each stage slot carries 8 KiB and the block passes through two slots.
SingleGcbCase t3_sub_chunk_case() {
    return {
        .fifo_bytes = 16 * kKiB,
        .stage_ring_bytes = 36 * kKiB,
        .tensors = {{.tensor_id = 50, .geometry = {.block_tiles = 4}}},
        .queue_calls = {{0}},
        .sender_requests = kRequestsForOnePrefetch};
}

// Four 4 KiB blocks against an 8 KiB stage third and a 16 KiB FIFO: one round moves all four blocks, two per stage
// slot.
SingleGcbCase t3_below_stage_third_case() {
    return {
        .fifo_bytes = 16 * kKiB,
        .stage_ring_bytes = 24 * kKiB,
        .tensors = {{.tensor_id = 51, .geometry = {.blocks = 4}}},
        .queue_calls = {{0}},
        .sender_requests = kRequestsForOnePrefetch};
}

// Three 4 KiB blocks against an 8 KiB stage third and a 12 KiB FIFO: one 12 KiB round, an 8 KiB stage chunk and then a
// 4 KiB tail chunk.
SingleGcbCase t3_partial_stage_chunk_case() {
    return {
        .fifo_bytes = 12 * kKiB,
        .stage_ring_bytes = 24 * kKiB,
        .tensors = {{.tensor_id = 66, .geometry = {.blocks = 3}}},
        .queue_calls = {{0}},
        .sender_requests = kRequestsForOnePrefetch};
}

// One 8 KiB block against the default 4 KiB stage third: one-tile sub-chunks, two stage slots per block.
SingleGcbCase t3_above_stage_third_case() {
    return {
        .fifo_bytes = 8 * kKiB,
        .tensors = {{.tensor_id = 52, .geometry = {.block_tiles = 2}}},
        .queue_calls = {{0}},
        .sender_requests = kRequestsForOnePrefetch};
}

// One Queue call with two tensors whose layouts differ (one block, then two). A page holds the 12 B header and one
// 64 B layout slot with its 8 B entry, and the second slot does not fit, so the call becomes two PREFETCH pages.
SingleGcbCase t3_page_split_case() {
    return {
        .tensors = {{.tensor_id = 53, .geometry = {}}, {.tensor_id = 54, .geometry = {.blocks = 2}}},
        .queue_calls = {{0, 1}},
        .sender_requests = 4};
}

// Tensor A (4 KiB pages), tensor B (8 KiB pages), then a second tensor of A's geometry, as three requests into a
// 16 KiB FIFO. B's resize pads wr_ptr from 4 KiB to 8 KiB on both ends, B wraps the FIFO, and the third request
// resumes at the FIFO start.
SingleGcbCase t3_aba_case() {
    return {
        .fifo_bytes = 16 * kKiB,
        .tensors =
            {{.tensor_id = 55, .geometry = {}},
             {.tensor_id = 56, .geometry = {.block_tiles = 2}},
             {.tensor_id = 62, .geometry = {}}},
        .queue_calls = {{0}, {1}, {2}},
        .sender_requests = 5};
}

// 4 KiB pages, 8 KiB pages twice, then 4 KiB pages, as four requests of four tensors into a 12 KiB FIFO that 8 KiB does
// not divide. The first 8 KiB request's resize wraps wr_ptr from 4 KiB to the start with 8 KiB of padding; every 8 KiB
// page then ends at the 8 KiB page-aligned limit and wraps with the 4 KiB gap credited; the last request resizes back
// with no padding.
SingleGcbCase t3_wrap_and_resize_case() {
    return {
        .fifo_bytes = 12 * kKiB,
        .tensors =
            {{.tensor_id = 57, .geometry = {}},
             {.tensor_id = 58, .geometry = {.block_tiles = 2}},
             {.tensor_id = 63, .geometry = {.block_tiles = 2}},
             {.tensor_id = 64, .geometry = {}}},
        .queue_calls = {{0}, {1}, {2}, {3}},
        .sender_requests = 6};
}

TEST_F(MimirPrefetcher, MM7T3SubChunkLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_single_gcb_case(kT3LocalReceiverCce, t3_sub_chunk_case());
}

TEST_F(MimirPrefetcher, MM7T3SubChunkD2d0To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3D2dReceiverCce});
    run_single_gcb_case(kT3D2dReceiverCce, t3_sub_chunk_case());
}

TEST_F(MimirPrefetcher, MM7T3BelowStageThirdLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_single_gcb_case(kT3LocalReceiverCce, t3_below_stage_third_case());
}

TEST_F(MimirPrefetcher, MM7T3BelowStageThirdD2d0To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3D2dReceiverCce});
    run_single_gcb_case(kT3D2dReceiverCce, t3_below_stage_third_case());
}

TEST_F(MimirPrefetcher, MM7T3PartialStageChunkLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_single_gcb_case(kT3LocalReceiverCce, t3_partial_stage_chunk_case());
}

TEST_F(MimirPrefetcher, MM7T3AboveStageThirdLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_single_gcb_case(kT3LocalReceiverCce, t3_above_stage_third_case());
}

TEST_F(MimirPrefetcher, MM7T3AboveStageThirdD2d0To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3D2dReceiverCce});
    run_single_gcb_case(kT3D2dReceiverCce, t3_above_stage_third_case());
}

// Local only: the split happens in host serialization, before any page reaches the route.
TEST_F(MimirPrefetcher, MM7T3PageSplitLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_single_gcb_case(kT3LocalReceiverCce, t3_page_split_case());
}

TEST_F(MimirPrefetcher, MM7T3AbaLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_single_gcb_case(kT3LocalReceiverCce, t3_aba_case());
}

TEST_F(MimirPrefetcher, MM7T3AbaD2d0To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3D2dReceiverCce});
    run_single_gcb_case(kT3D2dReceiverCce, t3_aba_case());
}

TEST_F(MimirPrefetcher, MM7T3WrapAndResizeLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_single_gcb_case(kT3LocalReceiverCce, t3_wrap_and_resize_case());
}

TEST_F(MimirPrefetcher, MM7T3WrapAndResizeD2d0To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3D2dReceiverCce});
    run_single_gcb_case(kT3D2dReceiverCce, t3_wrap_and_resize_case());
}

inline constexpr uint32_t kT3SwitchFirstTensorId = 59;
inline constexpr uint32_t kT3SwitchSecondTensorId = 60;
inline constexpr uint32_t kT3SwitchThirdTensorId = 65;
// Two pages of one tile, so GCB1's first request leaves wr_ptr mid-FIFO for its second to resume from.
inline constexpr uint32_t kT3SwitchFifoBytes = 8 * kKiB;
inline constexpr uint32_t kT3FenceTensorId = 61;

// GCB1, then GCB2, then GCB1 again, each with its own tensor, from kT3SenderCce. GCB1 delivers to `receiver_cce` and
// GCB2 to `second_receiver_cce`; on one CCE the two validators run on two of its harts. GCB1's first request leaves
// wr_ptr halfway through its FIFO, and the second resumes there from GCB1's state block. Stop's one DRAIN page names
// both GCBs: three PREFETCH pages, the DRAIN and STOP.
void run_t3_gcb_switch(uint32_t receiver_cce, uint32_t second_receiver_cce) {
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    const std::optional<uint32_t> sender_bank = t3_sender_bank(mesh, receiver_cce);
    ASSERT_TRUE(sender_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << kT3SenderCce;
    if (mimir_index_for_cce(second_receiver_cce) != mimir_index_for_cce(receiver_cce)) {
        EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    }
    const SlabTensorGeometry geometry{};
    const MeshTensor first_tensor = write_single_bank_tensor(mesh, *sender_bank, kT3SwitchFirstTensorId, geometry);
    const MeshTensor second_tensor = write_single_bank_tensor(mesh, *sender_bank, kT3SwitchSecondTensorId, geometry);
    const MeshTensor third_tensor = write_single_bank_tensor(mesh, *sender_bank, kT3SwitchThirdTensorId, geometry);
    const std::vector<GcbSmokeRequest> first_schedule = {
        slab_tensor_request(kT3SwitchFirstTensorId, geometry), slab_tensor_request(kT3SwitchThirdTensorId, geometry)};
    const std::vector<GcbSmokeRequest> second_schedule = {slab_tensor_request(kT3SwitchSecondTensorId, geometry)};
    GcbSmoke first_smoke(mesh, *sender_bank, {receiver_cce}, kT3SwitchFifoBytes);
    GcbSmoke second_smoke(mesh, *sender_bank, {second_receiver_cce}, kT3SwitchFifoBytes);
    expect_gcb_layout_ranges_disjoint(first_smoke.gcb(), second_smoke.gcb());
    CceKernelLaunch second_launch = validator_launch(second_smoke, 0, second_schedule);
    if (second_receiver_cce == receiver_cce) {
        second_launch.hart = kSecondGcbReceiverHart;
    }
    CceReceiverProgram receivers(
        first_smoke.device(), {validator_launch(first_smoke, 0, first_schedule), second_launch});
    PrefetcherSession session(mesh, {*sender_bank});
    queue_tensor(mesh, first_smoke, first_tensor, geometry.blocks);
    queue_tensor(mesh, second_smoke, second_tensor, geometry.blocks);
    queue_tensor(mesh, first_smoke, third_tensor, geometry.blocks);
    session.stop();
    expect_lifecycle_sender_stopped(mesh, kT3SenderCce, 5);

    receivers.stop_and_wait();
    expect_validated_delivery(first_smoke, 0, first_schedule);
    expect_validated_delivery(second_smoke, 0, second_schedule);
}

TEST_F(MimirPrefetcher, MM7T3GcbSwitchLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    run_t3_gcb_switch(kT3LocalReceiverCce, kT3LocalReceiverCce);
}

TEST_F(MimirPrefetcher, MM7T3GcbSwitchD2d0To2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3D2dReceiverCce});
    run_t3_gcb_switch(kT3D2dReceiverCce, kT3D2dReceiverCce);
}

// The same switch, with GCB2 on another receiver CCE across D2D: CCE 1, then CCE 2, then back to CCE 1.
TEST_F(MimirPrefetcher, MM7T3ReceiverSwitch0To1Then2) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce, kT3D2dReceiverCce});
    run_t3_gcb_switch(kT3LocalReceiverCce, kT3D2dReceiverCce);
}

// The fence: a WAIT_CQ for a value the CQ signal slot does not hold yet, queued through the manager's test hook, then a
// PREFETCH. The sender waits in WAIT_CQ and delivers nothing until the test writes the value into the slot. The session
// has the success budget, so a CQ_TIMEOUT cannot end the wait while the test watches it. Local only: the wait is on the
// sender, before anything crosses the route.
TEST_F(MimirPrefetcher, MM7T3FenceLocal0To1) {
    MIMIR_ROUTE_GUARD({kT3SenderCce, kT3LocalReceiverCce});
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    const std::optional<uint32_t> sender_bank = t3_sender_bank(mesh, kT3LocalReceiverCce);
    ASSERT_TRUE(sender_bank.has_value()) << "no DRAM view's prefetcher sender is CCE " << kT3SenderCce;
    const SlabTensorGeometry geometry{};
    const MeshTensor tensor = write_single_bank_tensor(mesh, *sender_bank, kT3FenceTensorId, geometry);
    const std::vector<GcbSmokeRequest> schedule = {slab_tensor_request(kT3FenceTensorId, geometry)};
    GcbSmoke smoke(mesh, *sender_bank, {kT3LocalReceiverCce});
    CceReceiverProgram receivers(smoke.device(), {validator_launch(smoke, 0, schedule)});
    PrefetcherSession session(mesh, {*sender_bank});

    const distributed::TensorPrefetcherManagerTestAccess::UnsignaledCqWait fence =
        distributed::TensorPrefetcherManagerTestAccess::queue_cq_wait_without_signal(
            mesh.impl().tensor_prefetcher(&mesh));
    queue_tensor(mesh, smoke, tensor, geometry.blocks);
    const CceSenderStatus waiting = poll_sender_status(
        mesh,
        kT3SenderCce,
        [](const CceSenderStatus& status) {
            return status.request_index != 0 || (status.state != static_cast<uint32_t>(CceSenderState::NotStarted) &&
                                                 status.state != static_cast<uint32_t>(CceSenderState::Running));
        },
        kGcbSmokeHostWaitLimit);
    ASSERT_EQ(waiting.state, static_cast<uint32_t>(CceSenderState::WaitingCq));
    const CceSenderStatus held = poll_sender_status(
        mesh,
        kT3SenderCce,
        [](const CceSenderStatus& status) { return status.state != static_cast<uint32_t>(CceSenderState::WaitingCq); },
        kSenderHoldTime);
    EXPECT_EQ(held.state, static_cast<uint32_t>(CceSenderState::WaitingCq));
    EXPECT_EQ(lifecycle_bytes_acked(mesh, kT3SenderCce), 0u);
    EXPECT_EQ(smoke.read_status(smoke.receiver_cce(0))[gcb_abi::kStatusMagic], gcb_abi::kStatusPoison);
    smoke.expect_fifo_state(0, {.credit_units = 0, .fifo_offset = 0}, 0, 0);

    log_info(tt::LogTest, "MM7T3 fence: writing {} to CQ signal slot 0x{:x}", fence.value, fence.slot_address);
    write_cce_tl1(smoke.device(), kT3SenderCce, fence.slot_address, {fence.value});
    session.stop();
    expect_lifecycle_sender_stopped(mesh, kT3SenderCce, 4);
    receivers.stop_and_wait();
    expect_validated_delivery(smoke, 0, schedule);
}

// T4: failures a CCE sender reports and recovers from, on D2D routes. Each failure is torn down in this order: Stop,
// release the receivers and wait for them, destroy the GCB. Then a fresh GCB, receivers and Start deliver one valid
// page. A session whose sender should time out gets the stall budget, so the expiry takes seconds of host polling.

inline constexpr uint32_t kT4FailureTensorId = 20;
inline constexpr uint32_t kT4RecoveryTensorId = 21;
inline constexpr uint32_t kT4OverflowPages = (kGcbSmokeFifoBytes / kUint32TilePageBytes) + 1;

uint32_t t4_one_page_credit_units() {
    return expected_gcb_fifo_end_state(
               {{.tensor_id = kT4FailureTensorId, .slab = 0, .page_bytes = kUint32TilePageBytes, .num_pages = 1}},
               kGcbSmokeFifoBytes)
        .credit_units;
}

CceKernelLaunch t4_stalling_receiver(const GcbSmoke& smoke) {
    return smoke.receiver_launch(
        0,
        kGcbDiscardReceiverKernel,
        0,
        gcb_abi::kModeStallUntilReleased,
        1,
        kGcbSmokeWaitBudgetCycles,
        {{.tensor_id = kT4FailureTensorId, .slab = 0, .page_bytes = kUint32TilePageBytes, .num_pages = 0}});
}

// The recovery after a failure: a fresh GCB to `receiver_cce` with a validator, a fresh Start, and `tensor`'s one page.
void expect_t4_recovery(
    distributed::MeshDevice& mesh,
    uint32_t sender_bank,
    uint32_t sender_cce,
    uint32_t receiver_cce,
    const MeshTensor& tensor) {
    const std::vector<GcbSmokeRequest> schedule = {
        {.tensor_id = kT4RecoveryTensorId, .slab = 0, .page_bytes = kUint32TilePageBytes, .num_pages = 1}};
    GcbSmoke smoke(mesh, sender_bank, {receiver_cce});
    CceReceiverProgram receivers(smoke.device(), {validator_launch(smoke, 0, schedule)});
    PrefetcherSession session(mesh, {sender_bank});
    queue_tensor(mesh, smoke, tensor, 1);
    session.stop();
    expect_lifecycle_sender_stopped(mesh, sender_cce, kRequestsForOnePrefetch);
    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], 1u);
}

// The validator recomputes the page with a wrong seed: it reports MISMATCH and keeps acking, so the sender stops OK.
// The hash is a bijection, so every word of the page differs, and the first word it read must be the tensor's: a sender
// that delivered other bytes fails these checks too, not just the recovery's.
TEST_F(MimirPrefetcher, MM7T4MismatchD2d2To0) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({2, 0});
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    const std::optional<uint32_t> sender_bank_of_cce = bank_with_sender_cce(mesh, 2);
    ASSERT_TRUE(sender_bank_of_cce.has_value()) << "no DRAM view's prefetcher sender is CCE 2";
    const uint32_t sender_bank = *sender_bank_of_cce;
    const uint32_t seed = mimir_test_seed();
    const MeshTensor failure_tensor = write_single_bank_tensor(mesh, sender_bank, kT4FailureTensorId, {});
    const MeshTensor recovery_tensor = write_single_bank_tensor(mesh, sender_bank, kT4RecoveryTensorId, {});
    {
        GcbSmoke smoke(mesh, sender_bank, {0});
        CceReceiverProgram receivers(
            smoke.device(),
            {smoke.receiver_launch(
                0,
                kGcbValidatorReceiverKernel,
                seed ^ 0x1u,
                0,
                0,
                kGcbSmokeWaitBudgetCycles,
                {{.tensor_id = kT4FailureTensorId, .slab = 0, .page_bytes = kUint32TilePageBytes, .num_pages = 1}})});
        PrefetcherSession session(mesh, {sender_bank});
        queue_tensor(mesh, smoke, failure_tensor, 1);
        session.stop();
        expect_lifecycle_sender_stopped(mesh, 2, kRequestsForOnePrefetch);
        receivers.stop_and_wait();
        const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
        EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeMismatch);
        EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], 1u);
        EXPECT_EQ(receiver_status[gcb_abi::kStatusMismatchedWords], kUint32TileWords);
        EXPECT_EQ(receiver_status[gcb_abi::kStatusFirstMismatchWord], 0u);
        EXPECT_EQ(receiver_status[gcb_abi::kStatusObserved], gcb_abi::gcb_page_word(seed, kT4FailureTensorId, 0, 0, 0));
    }
    expect_t4_recovery(mesh, sender_bank, 2, 0, recovery_tensor);
}

// A request one page larger than the FIFO into a receiver that holds the first page: the sender's free-space wait
// expires with CREDIT_TIMEOUT and the sender exits. The worker then fills its ring with one ring of further requests
// and drops the rest, Stop's DRAIN and STOP among them, because the sender has exited; Stop releases the session and
// throws, naming the failed sender and those dropped pages.
TEST_F(MimirPrefetcher, MM7T4CreditTimeoutD2d1To3) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({1, 3});
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    const std::optional<uint32_t> sender_bank_of_cce = bank_with_sender_cce(mesh, 1);
    ASSERT_TRUE(sender_bank_of_cce.has_value()) << "no DRAM view's prefetcher sender is CCE 1";
    const uint32_t sender_bank = *sender_bank_of_cce;
    const MeshTensor failure_tensor =
        write_single_bank_tensor(mesh, sender_bank, kT4FailureTensorId, {.blocks = kT4OverflowPages});
    const MeshTensor recovery_tensor = write_single_bank_tensor(mesh, sender_bank, kT4RecoveryTensorId, {});
    {
        GcbSmoke smoke(mesh, sender_bank, {3});
        CceReceiverProgram receivers(smoke.device(), {t4_stalling_receiver(smoke)});
        PrefetcherSession session(mesh, {sender_bank}, kGcbSmokeStallBudgetCycles);
        queue_tensor(mesh, smoke, failure_tensor, kT4OverflowPages);
        const CceSenderStatus failure = poll_sender_status(mesh, 1, sender_exited, kGcbSmokeHostWaitLimit);
        EXPECT_EQ(failure.state, static_cast<uint32_t>(CceSenderState::Failed));
        EXPECT_EQ(failure.code, static_cast<uint32_t>(CceSenderStatusCode::CreditTimeout));
        EXPECT_EQ(failure.request_index, 0u);
        EXPECT_EQ(failure.receiver, 0u);
        EXPECT_EQ(failure.sent, t4_one_page_credit_units());
        EXPECT_EQ(failure.acked, 0u);
        ASSERT_TRUE(smoke.wait_for_holding(0, 1));

        for (uint32_t request = 0; request <= kCceRequestRingSlots; ++request) {
            queue_tensor(mesh, smoke, recovery_tensor, 1);
        }
        const std::string stop_error = lifecycle_error([&session]() { session.stop(); });
        EXPECT_NE(stop_error.find("ended Failed"), std::string::npos) << stop_error;
        // The last further request, the DRAIN and the STOP.
        EXPECT_NE(stop_error.find("(3 more request pages never fit"), std::string::npos) << stop_error;
        EXPECT_NE(
            stop_error.find(fmt::format("{} request page(s) left unread in its ring", kCceRequestRingSlots)),
            std::string::npos)
            << stop_error;
        const uint64_t ring_meta_address =
            distributed::TensorPrefetcherManagerTestAccess::cce_sender_control_base(mesh) + kCceRingMetaOffset;
        const uint32_t bytes_sent =
            read_cce_tl1(device, 1, ring_meta_address + offsetof(RingMeta, bytes_sent), sizeof(uint32_t)).front();
        EXPECT_EQ(bytes_sent - lifecycle_bytes_acked(mesh, 1), kCceRequestRingBytes);
        receivers.stop_and_wait();
        EXPECT_EQ(smoke.read_receiver_status(0)[gcb_abi::kStatusCode], gcb_abi::kCodeReleased);
    }
    expect_t4_recovery(mesh, sender_bank, 1, 3, recovery_tensor);
}

// A request that fits into a receiver that holds it: Stop's DRAIN barrier expires with DRAIN_TIMEOUT, and Stop throws
// once the session is released.
TEST_F(MimirPrefetcher, MM7T4DrainTimeoutD2d1To3) {
    skip_unless_mimir_local_package();
    MIMIR_ROUTE_GUARD({1, 3});
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
    const std::optional<uint32_t> sender_bank_of_cce = bank_with_sender_cce(mesh, 1);
    ASSERT_TRUE(sender_bank_of_cce.has_value()) << "no DRAM view's prefetcher sender is CCE 1";
    const uint32_t sender_bank = *sender_bank_of_cce;
    const MeshTensor failure_tensor = write_single_bank_tensor(mesh, sender_bank, kT4FailureTensorId, {});
    const MeshTensor recovery_tensor = write_single_bank_tensor(mesh, sender_bank, kT4RecoveryTensorId, {});
    {
        GcbSmoke smoke(mesh, sender_bank, {3});
        CceReceiverProgram receivers(smoke.device(), {t4_stalling_receiver(smoke)});
        PrefetcherSession session(mesh, {sender_bank}, kGcbSmokeStallBudgetCycles);
        queue_tensor(mesh, smoke, failure_tensor, 1);
        const std::string stop_error = lifecycle_error([&session]() { session.stop(); });
        EXPECT_NE(stop_error.find("ended Failed"), std::string::npos) << stop_error;
        ASSERT_TRUE(smoke.wait_for_holding(0, 1));
        const CceSenderStatus failure = lifecycle_sender_status(mesh, 1);
        EXPECT_EQ(failure.state, static_cast<uint32_t>(CceSenderState::Failed));
        EXPECT_EQ(failure.code, static_cast<uint32_t>(CceSenderStatusCode::DrainTimeout));
        EXPECT_EQ(failure.request_index, 1u);
        EXPECT_EQ(failure.target_state_addr, experimental::sender_state_drisc_l1_base(smoke.gcb()));
        EXPECT_EQ(failure.receiver, 0u);
        EXPECT_EQ(failure.sent, t4_one_page_credit_units());
        EXPECT_EQ(failure.acked, 0u);
        receivers.stop_and_wait();
        EXPECT_EQ(smoke.read_receiver_status(0)[gcb_abi::kStatusCode], gcb_abi::kCodeReleased);
    }
    expect_t4_recovery(mesh, sender_bank, 1, 3, recovery_tensor);
}

// MM7Reject: Queue on Quasar refuses a request a CCE sender cannot deliver, before any page reaches the sender. Each
// case runs in a session from bank 0's sender to a local validator; after the rejection one valid page still goes
// through, and the sender takes only that PREFETCH, Stop's DRAIN and STOP.

inline constexpr uint32_t kRejectTensorId = 30;

// `queue_rejected_request` queues the case's request to the GCB and returns the error it throws.
using RejectedQueue = std::function<std::string(
    distributed::MeshDevice& mesh, const experimental::GlobalCircularBuffer& gcb, const MeshTensor& valid_tensor)>;

void run_rejected_queue_case(const RejectedQueue& queue_rejected_request, const std::string& expected_error) {
    ScopedDevice device_guard;
    distributed::MeshDevice& mesh = device_guard.mesh();
    IDevice* device = mesh.get_devices().front();
    const uint32_t sender_cce = gcb_layout_sender_cce(device, mesh, 0);
    const MeshTensor valid_tensor = write_single_bank_tensor(mesh, 0, kRejectTensorId, {});
    GcbSmoke smoke(mesh, 0, {gcb_smoke_local_receiver(mesh)});
    CceReceiverProgram receivers(
        device,
        {validator_launch(
            smoke,
            0,
            {{.tensor_id = kRejectTensorId, .slab = 0, .page_bytes = kUint32TilePageBytes, .num_pages = 1}})});
    PrefetcherSession session(mesh, {0});

    const std::string queue_error = queue_rejected_request(mesh, smoke.gcb(), valid_tensor);
    log_info(tt::LogTest, "MM7Reject Queue: {}", queue_error);
    EXPECT_NE(queue_error.find(expected_error), std::string::npos) << queue_error;

    queue_tensor(mesh, smoke, valid_tensor, 1);
    session.stop();
    expect_lifecycle_sender_stopped(mesh, sender_cce, kRequestsForOnePrefetch);
    receivers.stop_and_wait();
    const std::vector<uint32_t> receiver_status = smoke.read_receiver_status(0);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusCode], gcb_abi::kCodeOk);
    EXPECT_EQ(receiver_status[gcb_abi::kStatusPagesDone], 1u);
}

// An interleaved DRAM tensor has neither shard spec, so it resolves to the K-row-major layout.
TEST_F(MimirPrefetcher, MM7RejectKRowMajorTensor) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    run_rejected_queue_case(
        [](distributed::MeshDevice& mesh, const experimental::GlobalCircularBuffer& gcb, const MeshTensor&) {
            const MeshTensor interleaved_tensor = MeshTensor::allocate_on_device(
                mesh,
                TensorSpec(
                    Shape{32, 32}, TensorLayout(DataType::UINT32, Layout::TILE, MemoryConfig(BufferType::DRAM))));
            return lifecycle_error([&mesh, &gcb, &interleaved_tensor]() {
                experimental::QueueTensorPrefetcherRequest(
                    mesh,
                    gcb,
                    std::nullopt,
                    {experimental::TensorPrefetcherInput{.tensor = interleaved_tensor, .block_count = 1}});
            });
        },
        "resolves to the K-row-major layout");
}

TEST_F(MimirPrefetcher, MM7RejectStreamingTensor) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    run_rejected_queue_case(
        [](distributed::MeshDevice& mesh,
           const experimental::GlobalCircularBuffer& gcb,
           const MeshTensor& valid_tensor) {
            return lifecycle_error([&mesh, &gcb, &valid_tensor]() {
                experimental::QueueTensorPrefetcherRequest(
                    mesh,
                    gcb,
                    std::nullopt,
                    {experimental::TensorPrefetcherInput{.tensor = valid_tensor, .block_count = 1, .rotation = {0}}});
            });
        },
        "has a streaming rotation");
}

// The tensor sits on DRAM bank 1, so bank 0's sender would read a slab the tensor does not have there.
TEST_F(MimirPrefetcher, MM7RejectRangePastTensor) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    if (topology_from(configured_soc_desc()).num_views < 2) {
        GTEST_SKIP();
    }
    run_rejected_queue_case(
        [](distributed::MeshDevice& mesh, const experimental::GlobalCircularBuffer& gcb, const MeshTensor&) {
            const MeshTensor other_bank_tensor = MeshTensor::allocate_on_device(mesh, single_bank_tensor_spec(1, {}));
            return lifecycle_error([&mesh, &gcb, &other_bank_tensor]() {
                experimental::QueueTensorPrefetcherRequest(
                    mesh,
                    gcb,
                    std::nullopt,
                    {experimental::TensorPrefetcherInput{.tensor = other_bank_tensor, .block_count = 1}});
            });
        },
        "so the request runs past the tensor");
}

// The allocator never places a tensor past its view, so this one wraps a buffer at an explicit bank-local address half
// a page before the view's end. Nothing reads or writes it.
TEST_F(MimirPrefetcher, MM7RejectRangePastView) {
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    run_rejected_queue_case(
        [](distributed::MeshDevice& mesh, const experimental::GlobalCircularBuffer& gcb, const MeshTensor&) {
            const TensorSpec spec = single_bank_tensor_spec(0, {});
            const DeviceAddr view_bytes = soc_desc_of(mesh.get_devices().front()).dram_view_size;
            std::shared_ptr<distributed::MeshBuffer> buffer = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = spec.compute_packed_buffer_size_bytes()},
                distributed::DeviceLocalBufferConfig{
                    .page_size = spec.compute_page_size_bytes(),
                    .buffer_type = BufferType::DRAM,
                    .sharding_args = spec.compute_buffer_sharding_args()},
                &mesh,
                view_bytes - (kUint32TilePageBytes / 2));
            const MeshTensor past_view_tensor = MeshTensor::from_buffer(std::move(*buffer), spec);
            return lifecycle_error([&mesh, &gcb, &past_view_tensor]() {
                experimental::QueueTensorPrefetcherRequest(
                    mesh,
                    gcb,
                    std::nullopt,
                    {experimental::TensorPrefetcherInput{.tensor = past_view_tensor, .block_count = 1}});
            });
        },
        "B DRAM view");
}

}  // namespace tt::tt_metal::mimir_test::prefetcher
