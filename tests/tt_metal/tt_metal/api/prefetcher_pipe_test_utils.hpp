// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cross_node_dfb_test_utils.hpp"
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "hostdev/remote_dfb_config_layout.h"

namespace tt::tt_metal::prefetcher_pipe_test {

inline uint32_t persistent_config_page_l1_address(const experimental::PrefetcherPipe& pipe) {
    return pipe.config_address();
}

inline std::vector<uint8_t> read_receiver_ring_bytes(
    distributed::MeshDevice& device,
    const experimental::PrefetcherPipe& pipe,
    const CoreCoord& receiver_core,
    uint32_t num_bytes) {
    const uint32_t ring_size = pipe.ring_size();
    const uint32_t copy_size = std::min(num_bytes, ring_size);
    std::vector<uint8_t> bytes(num_bytes, 0);
    if (copy_size == 0) {
        return bytes;
    }
    slow_dispatch::ReadFromL1(
        device, receiver_core, pipe.buffer_address(), std::span<uint8_t>(bytes.data(), copy_size), CoreType::WORKER);
    return bytes;
}

inline bool verify_receiver_ring(
    distributed::MeshDevice& device,
    const experimental::PrefetcherPipe& pipe,
    const CoreCoord& receiver_core,
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t receiver_idx,
    uint32_t num_receivers,
    uint32_t counter_base = 0) {
    const auto expected = cross_node_dfb_test::expected_receiver_ring_bytes(
        data_pattern, entry_size, num_entries, receiver_idx, num_receivers, counter_base);
    const auto received = read_receiver_ring_bytes(device, pipe, receiver_core, static_cast<uint32_t>(expected.size()));
    return received == expected;
}

// Lane-0 (sent, acked) of `core`'s slot, read from TL1. SENT and ACKED live in separate blocks
// (remote_dfb_config_layout.h); the core's config page words give its slot in each.
inline std::pair<uint32_t, uint32_t> read_pipe_credits(
    distributed::MeshDevice& device, const experimental::PrefetcherPipe& pipe, const CoreCoord& core) {
    const auto& page = pipe.impl().config_page(core);
    const auto read_word = [&](uint32_t page_offset) {
        std::vector<uint8_t> bytes(sizeof(uint32_t), 0);
        slow_dispatch::ReadFromL1(
            device,
            core,
            pipe.config_address() + page_offset,
            std::span<uint8_t>(bytes.data(), bytes.size()),
            CoreType::WORKER);
        uint32_t word = 0;
        std::memcpy(&word, bytes.data(), sizeof(uint32_t));
        return word;
    };
    return {
        read_word(page[PREFETCHER_PIPE_CFG_PAGES_SENT_OFFSET]),
        read_word(page[PREFETCHER_PIPE_CFG_PAGES_ACKED_OFFSET])};
}

inline uint32_t sender_l1_staging_address(const experimental::PrefetcherPipe& pipe) {
    // The persistent arena grows bottom-up from l1_unreserved_base, so placing
    // staging below the pipe would enter reserved firmware L1. Tests use the
    // first address above this pipe's persistent ring + config allocation.
    return std::max(pipe.buffer_address() + pipe.ring_size(), pipe.config_address() + pipe.config_page_size());
}

inline void write_sender_l1_staging(
    distributed::MeshDevice& device,
    const CoreRangeSet& sender_cores,
    const experimental::PrefetcherPipe& pipe,
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_receivers,
    uint32_t counter_base = 0,
    uint32_t entry_size_resized = 0,
    uint32_t num_entries_after = 0) {
    const auto bytes = cross_node_dfb_test::build_sender_staging_bytes(
        data_pattern, entry_size, num_entries, num_receivers, counter_base, entry_size_resized, num_entries_after);
    const uint32_t staging_size_bytes = static_cast<uint32_t>(bytes.size());
    const uint32_t staging_addr = sender_l1_staging_address(pipe);
    const uint32_t aligned_words =
        (cross_node_dfb_test::align_staging_size_bytes(staging_size_bytes) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    std::vector<uint32_t> words(aligned_words, 0);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    for (const auto& core : corerange_to_cores(sender_cores)) {
        slow_dispatch::WriteToL1(device, core, staging_addr, words, CoreType::WORKER);
    }
}

}  // namespace tt::tt_metal::prefetcher_pipe_test
