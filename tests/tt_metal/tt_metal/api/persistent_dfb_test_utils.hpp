// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cross_node_dfb_test_utils.hpp"
#include "impl/dataflow_buffer/persistent_dfb.hpp"

namespace tt::tt_metal::persistent_dfb_test {

inline uint32_t persistent_config_page_l1_address(const experimental::PersistentDFB& pdfb) {
    return pdfb.config_address();
}

inline std::vector<uint8_t> read_receiver_ring_bytes(
    IDevice* device, const experimental::PersistentDFB& pdfb, const CoreCoord& receiver_core, uint32_t num_bytes) {
    IDevice* physical_device = cross_node_dfb_test::local_physical_device(device);
    const uint32_t ring_size = pdfb.ring_size();
    const uint32_t copy_size = std::min(num_bytes, ring_size);
    std::vector<uint8_t> bytes(num_bytes, 0);
    if (copy_size == 0) {
        return bytes;
    }
    detail::ReadFromDeviceL1(
        physical_device,
        receiver_core,
        pdfb.buffer_address(),
        std::span<uint8_t>(bytes.data(), copy_size),
        CoreType::WORKER);
    return bytes;
}

inline bool verify_receiver_ring(
    IDevice* device,
    const experimental::PersistentDFB& pdfb,
    const CoreCoord& receiver_core,
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t receiver_idx,
    uint32_t num_receivers,
    uint32_t counter_base = 0) {
    const auto expected = cross_node_dfb_test::expected_receiver_ring_bytes(
        data_pattern, entry_size, num_entries, receiver_idx, num_receivers, counter_base);
    const auto received = read_receiver_ring_bytes(device, pdfb, receiver_core, static_cast<uint32_t>(expected.size()));
    return received == expected;
}

inline uint32_t sender_l1_staging_address(const experimental::PersistentDFB& pdfb, uint32_t staging_size_bytes) {
    const uint32_t pdfb_bottom = std::min(pdfb.config_address(), pdfb.buffer_address());
    return pdfb_bottom - cross_node_dfb_test::align_staging_size_bytes(staging_size_bytes);
}

inline void write_sender_l1_staging(
    IDevice* device,
    const CoreRangeSet& sender_cores,
    const experimental::PersistentDFB& pdfb,
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
    const uint32_t staging_addr = sender_l1_staging_address(pdfb, staging_size_bytes);
    IDevice* physical_device = cross_node_dfb_test::local_physical_device(device);
    const uint32_t aligned_words =
        (cross_node_dfb_test::align_staging_size_bytes(staging_size_bytes) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    std::vector<uint32_t> words(aligned_words, 0);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    for (const auto& core : corerange_to_cores(sender_cores)) {
        detail::WriteToDeviceL1(physical_device, core, staging_addr, words, CoreType::WORKER);
    }
}

inline void set_sender_l1_staging_runtime_args(
    Program& program,
    KernelHandle sender_kernel,
    const CoreRangeSet& sender_cores,
    const experimental::PersistentDFB& pdfb,
    uint32_t staging_size_bytes) {
    const uint32_t l1_staging_addr = sender_l1_staging_address(pdfb, staging_size_bytes);
    for (const auto& core : corerange_to_cores(sender_cores)) {
        const CoreRangeSet single = CoreRangeSet(CoreRange(core));
        SetRuntimeArgs(program, sender_kernel, single, {l1_staging_addr});
    }
}

}  // namespace tt::tt_metal::persistent_dfb_test
