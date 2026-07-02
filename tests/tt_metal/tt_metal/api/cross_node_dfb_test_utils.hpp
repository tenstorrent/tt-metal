// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <tt-metalium/buffer.hpp>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/cross_node_dfb.hpp"

namespace tt::tt_metal::cross_node_dfb_test {

// Must match compile-time arg [5] in cross_node_dfb_sender.cpp.
enum class SenderDataPattern : uint32_t {
    MulticastCounter = 0,     // entry i: all bytes = (counter_base + i) & 0xFF
    StridedPerReceiver = 1,   // entry i, receiver r: all bytes = r & 0xFF
    PerReceiverConstant = 2,    // receiver r: all bytes = r & 0xFF (reused each entry)
};

inline constexpr uint32_t kMaxTestReceivers = 4;

// MeshDevice::id() is a mesh handle (often 1), not the physical chip id (0).
// Host-side CreateBuffer/WriteToBuffer must target the local physical device.
inline IDevice* local_physical_device(IDevice* device) {
    if (auto* mesh = dynamic_cast<distributed::MeshDevice*>(device)) {
        return mesh->get_devices().at(0);
    }
    return device;
}

inline uint32_t align_staging_size_bytes(uint32_t size_bytes) {
    const uint32_t alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    return (size_bytes + alignment - 1) & ~(alignment - 1);
}

inline uint32_t data_pattern_for_write_primitive(uint32_t write_primitive) {
    if (write_primitive == 1) {
        return static_cast<uint32_t>(SenderDataPattern::StridedPerReceiver);
    }
    if (write_primitive >= 2) {
        return static_cast<uint32_t>(SenderDataPattern::PerReceiverConstant);
    }
    return static_cast<uint32_t>(SenderDataPattern::MulticastCounter);
}

inline uint32_t sender_staging_size_bytes(
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_receivers,
    uint32_t entry_size_resized = 0,
    uint32_t num_entries_after = 0) {
    uint32_t size = 0;
    switch (data_pattern) {
        case static_cast<uint32_t>(SenderDataPattern::MulticastCounter):
            size = num_entries * entry_size;
            if (num_entries_after > 0) {
                size += num_entries_after * entry_size_resized;
            }
            break;
        case static_cast<uint32_t>(SenderDataPattern::StridedPerReceiver):
            size = num_entries * num_receivers * entry_size;
            break;
        case static_cast<uint32_t>(SenderDataPattern::PerReceiverConstant):
            size = num_receivers * entry_size;
            break;
        default: break;
    }
    return size;
}

inline std::vector<uint8_t> build_sender_staging_bytes(
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_receivers,
    uint32_t counter_base = 0,
    uint32_t entry_size_resized = 0,
    uint32_t num_entries_after = 0) {
    const uint32_t size = sender_staging_size_bytes(
        data_pattern, entry_size, num_entries, num_receivers, entry_size_resized, num_entries_after);
    std::vector<uint8_t> staging(size, 0);

    switch (data_pattern) {
        case static_cast<uint32_t>(SenderDataPattern::MulticastCounter):
            for (uint32_t i = 0; i < num_entries; ++i) {
                const uint8_t byte = static_cast<uint8_t>((counter_base + i) & 0xFF);
                std::fill(
                    staging.begin() + i * entry_size,
                    staging.begin() + (i + 1) * entry_size,
                    byte);
            }
            if (num_entries_after > 0) {
                const uint32_t base_offset = num_entries * entry_size;
                for (uint32_t i = 0; i < num_entries_after; ++i) {
                    const uint8_t byte = static_cast<uint8_t>((counter_base + num_entries + i) & 0xFF);
                    std::fill(
                        staging.begin() + base_offset + i * entry_size_resized,
                        staging.begin() + base_offset + (i + 1) * entry_size_resized,
                        byte);
                }
            }
            break;
        case static_cast<uint32_t>(SenderDataPattern::StridedPerReceiver):
            for (uint32_t i = 0; i < num_entries; ++i) {
                for (uint32_t r = 0; r < num_receivers; ++r) {
                    const uint8_t byte = static_cast<uint8_t>(r & 0xFF);
                    const uint32_t offset = i * num_receivers * entry_size + r * entry_size;
                    std::fill(
                        staging.begin() + offset,
                        staging.begin() + offset + entry_size,
                        byte);
                }
            }
            break;
        case static_cast<uint32_t>(SenderDataPattern::PerReceiverConstant):
            for (uint32_t r = 0; r < num_receivers; ++r) {
                const uint8_t byte = static_cast<uint8_t>(r & 0xFF);
                std::fill(
                    staging.begin() + r * entry_size,
                    staging.begin() + (r + 1) * entry_size,
                    byte);
            }
            break;
        default: break;
    }
    return staging;
}


// Sender-local L1 scratch placed immediately BELOW the CrossNodeDFB region.
// The L1 allocator fills top-down: ring first (highest), then config below it.
// Placing staging ABOVE ring_end would land at 0x180000 = end of WH L1 (invalid).
// Instead we subtract staging size from the bottom of the GDFB region (config_addr).
inline uint32_t sender_l1_staging_address(
    const experimental::CrossNodeDFB& gdfb, uint32_t staging_size_bytes) {
    // config is always allocated below ring in a top-down allocator
    const uint32_t gdfb_bottom = std::min(gdfb.config_address(), gdfb.buffer_address());
    return gdfb_bottom - align_staging_size_bytes(staging_size_bytes);
}

inline void assert_staging_disjoint_from_cross_node_dfb(
    uint32_t staging_addr,
    uint32_t staging_size_bytes,
    const experimental::CrossNodeDFB& gdfb) {
    const uint32_t staging_end = staging_addr + align_staging_size_bytes(staging_size_bytes);
    const uint32_t config_addr = gdfb.config_address();
    const uint32_t config_end = config_addr + static_cast<uint32_t>(gdfb.config_buffer().page_size());
    const uint32_t ring_addr = gdfb.buffer_address();
    const uint32_t ring_end = ring_addr + static_cast<uint32_t>(gdfb.dfb_buffer().page_size());
    const bool overlaps_config = staging_addr < config_end && staging_end > config_addr;
    const bool overlaps_ring = staging_addr < ring_end && staging_end > ring_addr;
    TT_FATAL(
        !overlaps_config && !overlaps_ring,
        "sender staging L1 [0x{:x}, 0x{:x}) overlaps CrossNodeDFB config [0x{:x}, 0x{:x}) or ring [0x{:x}, "
        "0x{:x})",
        staging_addr,
        staging_end,
        config_addr,
        config_end,
        ring_addr,
        ring_end);
}

// Write sender staging data directly to the sender core's L1 (bypasses DRAM).
// Must be called before run_on_mesh_device so the data is resident in L1 before the
// kernel starts reading it.  The dispatch pipeline writes the kernel config ring buffer
// at low L1 addresses (~MEM_MAP_END), which is far from the top-of-L1 staging area,
// so the dispatch does not overwrite this data.
inline void write_sender_l1_staging(
    IDevice* device,
    const CoreRangeSet& sender_cores,
    const experimental::CrossNodeDFB& gdfb,
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_receivers,
    uint32_t counter_base = 0,
    uint32_t entry_size_resized = 0,
    uint32_t num_entries_after = 0) {
    IDevice* physical_device = local_physical_device(device);
    const auto bytes = build_sender_staging_bytes(
        data_pattern, entry_size, num_entries, num_receivers, counter_base,
        entry_size_resized, num_entries_after);
    const uint32_t staging_size_bytes = static_cast<uint32_t>(bytes.size());
    const uint32_t staging_addr = sender_l1_staging_address(gdfb, staging_size_bytes);
    assert_staging_disjoint_from_cross_node_dfb(staging_addr, staging_size_bytes, gdfb);
    const uint32_t aligned_words = (align_staging_size_bytes(staging_size_bytes) + sizeof(uint32_t) - 1)
                                    / sizeof(uint32_t);
    std::vector<uint32_t> words(aligned_words, 0);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    for (const auto& core : corerange_to_cores(sender_cores)) {
        detail::WriteToDeviceL1(physical_device, core, staging_addr, words, CoreType::WORKER);
    }
}

// Set the single runtime arg [0] = l1_staging_addr for the sender kernel.
// The kernel reads staging data from this L1 address (pre-populated by write_sender_l1_staging).
inline void set_sender_l1_staging_runtime_args(
    Program& program,
    KernelHandle sender_kernel,
    const CoreRangeSet& sender_cores,
    const experimental::CrossNodeDFB& gdfb,
    uint32_t staging_size_bytes) {
    const uint32_t l1_staging_addr = sender_l1_staging_address(gdfb, staging_size_bytes);
    for (const auto& core : corerange_to_cores(sender_cores)) {
        const CoreRangeSet single = CoreRangeSet(CoreRange(core));
        SetRuntimeArgs(program, sender_kernel, single, {l1_staging_addr});
    }
}

inline std::vector<uint8_t> expected_receiver_ring_bytes(
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t receiver_idx,
    uint32_t num_receivers,
    uint32_t counter_base = 0,
    uint32_t entry_size_resized = 0,
    uint32_t num_entries_after = 0) {
    if (data_pattern == static_cast<uint32_t>(SenderDataPattern::PerReceiverConstant)) {
        std::vector<uint8_t> ring(num_entries * entry_size);
        const uint8_t byte = static_cast<uint8_t>(receiver_idx & 0xFF);
        for (uint32_t i = 0; i < num_entries; ++i) {
            std::fill(ring.begin() + i * entry_size, ring.begin() + (i + 1) * entry_size, byte);
        }
        return ring;
    }

    if (data_pattern == static_cast<uint32_t>(SenderDataPattern::StridedPerReceiver)) {
        const auto full = build_sender_staging_bytes(
            data_pattern, entry_size, num_entries, num_receivers, counter_base, entry_size_resized, num_entries_after);
        std::vector<uint8_t> ring(num_entries * entry_size);
        for (uint32_t i = 0; i < num_entries; ++i) {
            const uint32_t src_offset = i * num_receivers * entry_size + receiver_idx * entry_size;
            std::copy(
                full.begin() + src_offset,
                full.begin() + src_offset + entry_size,
                ring.begin() + i * entry_size);
        }
        return ring;
    }

    return build_sender_staging_bytes(
        data_pattern, entry_size, num_entries, num_receivers, counter_base, entry_size_resized, num_entries_after);
}

inline std::vector<uint8_t> read_receiver_ring_bytes(
    IDevice* device,
    const experimental::CrossNodeDFB& gdfb,
    const CoreCoord& receiver_core,
    uint32_t num_bytes) {
    IDevice* physical_device = local_physical_device(device);
    const uint32_t ring_size = static_cast<uint32_t>(gdfb.dfb_buffer().page_size());
    const uint32_t copy_size = std::min(num_bytes, ring_size);
    std::vector<uint8_t> bytes(num_bytes, 0);
    if (copy_size == 0) {
        return bytes;
    }
    detail::ReadFromDeviceL1(
        physical_device,
        receiver_core,
        gdfb.buffer_address(),
        std::span<uint8_t>(bytes.data(), copy_size),
        CoreType::WORKER);
    return bytes;
}

inline bool verify_receiver_ring(
    IDevice* device,
    const experimental::CrossNodeDFB& gdfb,
    const CoreCoord& receiver_core,
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t receiver_idx,
    uint32_t num_receivers,
    uint32_t counter_base = 0,
    uint32_t entry_size_resized = 0,
    uint32_t num_entries_after = 0) {
    const auto expected = expected_receiver_ring_bytes(
        data_pattern,
        entry_size,
        num_entries,
        receiver_idx,
        num_receivers,
        counter_base,
        entry_size_resized,
        num_entries_after);
    const auto received =
        read_receiver_ring_bytes(device, gdfb, receiver_core, static_cast<uint32_t>(expected.size()));
    return received == expected;
}

}  // namespace tt::tt_metal::cross_node_dfb_test
