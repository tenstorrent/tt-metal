// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/buffer.hpp>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/assert.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal::cross_node_dfb_test {

// Must match compile-time arg [5] in cross_node_dfb_sender.cpp.
enum class SenderDataPattern : uint32_t {
    MulticastCounter = 0,     // entry i: all bytes = (counter_base + i) & 0xFF
    StridedPerReceiver = 1,   // entry i, receiver r: all bytes = r & 0xFF
    PerReceiverConstant = 2,  // receiver r: all bytes = r & 0xFF (reused each entry)
};

inline uint32_t align_staging_size_bytes(uint32_t size_bytes) {
    const uint32_t alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    return (size_bytes + alignment - 1) & ~(alignment - 1);
}

// Allocate a HEIGHT_SHARDED L1 buffer matching CreateCrossNodeDFB's data-ring layout.
// Must use the same device passed to CreateCrossNodeDFB so L1 allocation is
// consistent with the runtime-allocated config buffer.
inline std::shared_ptr<Buffer> make_cross_node_data_buffer(
    distributed::MeshDevice& device,
    const CoreRangeSet& all_cores,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1) {
    const uint32_t ring_size = entry_size * num_entries;
    const uint32_t num_cores = all_cores.num_cores();
    return CreateBuffer(ShardedBufferConfig{
        .device = &device,
        .size = ring_size * num_cores,
        .page_size = ring_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1}),
    });
}

inline uint32_t data_pattern_for_write_primitive(uint32_t write_primitive) {
    if (write_primitive == 1) {
        return static_cast<uint32_t>(SenderDataPattern::StridedPerReceiver);
    }
    if (write_primitive == 2 || write_primitive == 3) {
        return static_cast<uint32_t>(SenderDataPattern::PerReceiverConstant);
    }
    // 0 (per-entry broadcast), 4 (decoupled broadcast batch) and 5 (entry-major
    // per-receiver credit) use counter staging.
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
        case static_cast<uint32_t>(SenderDataPattern::PerReceiverConstant): size = num_receivers * entry_size; break;
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
                std::fill(staging.begin() + i * entry_size, staging.begin() + (i + 1) * entry_size, byte);
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
                    std::fill(staging.begin() + offset, staging.begin() + offset + entry_size, byte);
                }
            }
            break;
        case static_cast<uint32_t>(SenderDataPattern::PerReceiverConstant):
            for (uint32_t r = 0; r < num_receivers; ++r) {
                const uint8_t byte = static_cast<uint8_t>(r & 0xFF);
                std::fill(staging.begin() + r * entry_size, staging.begin() + (r + 1) * entry_size, byte);
            }
            break;
        default: break;
    }
    return staging;
}

// Sender-local L1 scratch placed immediately below the CrossNodeDFB allocations.
// The L1 allocator fills top-down: data ring first, then the dedicated config Buffer.
inline uint32_t sender_l1_staging_address(const experimental::CrossNodeDFB& gdfb, uint32_t staging_size_bytes) {
    const uint32_t gdfb_bottom = std::min(gdfb.config_address(), gdfb.buffer_address());
    return gdfb_bottom - align_staging_size_bytes(staging_size_bytes);
}

inline void assert_staging_disjoint_from_cross_node_dfb(
    uint32_t staging_addr, uint32_t staging_size_bytes, const experimental::CrossNodeDFB& gdfb) {
    const uint32_t staging_end = staging_addr + align_staging_size_bytes(staging_size_bytes);
    const uint32_t config_addr = gdfb.config_address();
    const uint32_t config_end = config_addr + static_cast<uint32_t>(gdfb.config_buffer().page_size());
    const uint32_t ring_addr = gdfb.buffer_address();
    const uint32_t ring_end = ring_addr + gdfb.ring_size();
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
    distributed::MeshDevice& device,
    const CoreRangeSet& sender_cores,
    const experimental::CrossNodeDFB& gdfb,
    uint32_t data_pattern,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_receivers,
    uint32_t counter_base = 0,
    uint32_t entry_size_resized = 0,
    uint32_t num_entries_after = 0) {
    const auto bytes = build_sender_staging_bytes(
        data_pattern, entry_size, num_entries, num_receivers, counter_base, entry_size_resized, num_entries_after);
    const uint32_t staging_size_bytes = static_cast<uint32_t>(bytes.size());
    const uint32_t staging_addr = sender_l1_staging_address(gdfb, staging_size_bytes);
    assert_staging_disjoint_from_cross_node_dfb(staging_addr, staging_size_bytes, gdfb);
    const uint32_t aligned_words =
        (align_staging_size_bytes(staging_size_bytes) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    std::vector<uint32_t> words(aligned_words, 0);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    for (const auto& core : corerange_to_cores(sender_cores)) {
        slow_dispatch::WriteToL1(device, core, staging_addr, words, CoreType::WORKER);
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
            std::copy(full.begin() + src_offset, full.begin() + src_offset + entry_size, ring.begin() + i * entry_size);
        }
        return ring;
    }

    return build_sender_staging_bytes(
        data_pattern, entry_size, num_entries, num_receivers, counter_base, entry_size_resized, num_entries_after);
}

inline std::vector<uint8_t> read_receiver_ring_bytes(
    distributed::MeshDevice& device,
    const experimental::CrossNodeDFB& gdfb,
    const CoreCoord& receiver_core,
    uint32_t num_bytes) {
    const uint32_t ring_size = gdfb.ring_size();
    const uint32_t copy_size = std::min(num_bytes, ring_size);
    std::vector<uint8_t> bytes(num_bytes, 0);
    if (copy_size == 0) {
        return bytes;
    }
    slow_dispatch::ReadFromL1(
        device, receiver_core, gdfb.buffer_address(), std::span<uint8_t>(bytes.data(), copy_size), CoreType::WORKER);
    return bytes;
}

// CreateCrossNodeDFB does not zero the data ring; clear L1 before "untouched" checks.
inline void zero_receiver_ring(
    distributed::MeshDevice& device, const experimental::CrossNodeDFB& gdfb, const CoreCoord& receiver_core) {
    const uint32_t ring_size = gdfb.ring_size();
    const uint32_t aligned_words = (ring_size + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    std::vector<uint32_t> zeros(aligned_words, 0);
    slow_dispatch::WriteToL1(device, receiver_core, gdfb.buffer_address(), zeros, CoreType::WORKER);
}

inline bool verify_receiver_ring(
    distributed::MeshDevice& device,
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

// Read pages_sent / pages_acked from a core's sharded config page at the given slot.
inline std::pair<uint32_t, uint32_t> read_credit_pair(
    distributed::MeshDevice& device, const CoreCoord& core, uint32_t pages_sent_addr) {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    std::vector<uint8_t> bytes(2 * l1_alignment, 0);
    slow_dispatch::ReadFromL1(
        device, core, pages_sent_addr, std::span<uint8_t>(bytes.data(), bytes.size()), CoreType::WORKER);
    uint32_t sent = 0;
    uint32_t acked = 0;
    std::memcpy(&sent, bytes.data(), sizeof(uint32_t));
    std::memcpy(&acked, bytes.data() + l1_alignment, sizeof(uint32_t));
    return {sent, acked};
}

// Absolute L1 address of the dedicated, sharded CrossNode config Buffer.
inline uint32_t cross_node_config_page_l1_address(Program& program, uint8_t remote_dfb_id) {
    return program.impl().get_cross_node_dfb(remote_dfb_id).config_address();
}

// After a full drain, every receiver slot must have pages_sent == pages_acked == expected units
// on both the sender and that receiver's page in the dedicated config Buffer.
inline bool verify_credits_drained(
    distributed::MeshDevice& device,
    Program& program,
    uint8_t remote_dfb_id,
    const CoreCoord& sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t entry_size,
    uint32_t num_entries_pushed_per_receiver) {
    const experimental::CrossNodeDFB& gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    TT_FATAL(entry_size % l1_alignment == 0, "entry_size must be L1-aligned for credit accounting");
    const uint32_t expected_units = (num_entries_pushed_per_receiver * entry_size) / l1_alignment;
    const uint32_t credit_base = cross_node_config_page_l1_address(program, remote_dfb_id) + gdfb.credit_reset_offset();
    const auto receivers = corerange_to_cores(receiver_cores);

    bool ok = true;
    for (uint32_t ri = 0; ri < receivers.size(); ++ri) {
        const uint32_t slot_addr = credit_base + 2 * ri * l1_alignment;
        const auto [sent_s, acked_s] = read_credit_pair(device, sender_core, slot_addr);
        const auto [sent_r, acked_r] = read_credit_pair(device, receivers[ri], slot_addr);
        if (sent_s != expected_units || acked_s != expected_units || sent_r != expected_units ||
            acked_r != expected_units) {
            log_error(
                tt::LogTest,
                "credits drained FAIL receiver[{}] core=({},{}): expected={}, "
                "sender(sent={},acked={}), receiver(sent={},acked={})",
                ri,
                receivers[ri].x,
                receivers[ri].y,
                expected_units,
                sent_s,
                acked_s,
                sent_r,
                acked_r);
            ok = false;
        }
    }
    return ok;
}

}  // namespace tt::tt_metal::cross_node_dfb_test
