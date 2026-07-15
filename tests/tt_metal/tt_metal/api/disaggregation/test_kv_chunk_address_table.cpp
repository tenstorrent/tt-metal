// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp"
#include "experimental/fabric/fabric_types.hpp"

namespace tt::tt_metal::internal::disaggregation {
namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::MeshId;

// Helper to create a FabricNodeId.
FabricNodeId make_fnid(uint32_t mesh, uint32_t chip) { return FabricNodeId(MeshId{mesh}, chip); }

// Helper to create a KvCacheLocation with a pre-registered device group.
KvCacheLocation make_location(uint64_t noc_addr, uint32_t size_bytes, DeviceGroupIndex device_group_index) {
    return KvCacheLocation{.noc_addr = noc_addr, .size_bytes = size_bytes, .device_group_index = device_group_index};
}

// --- Construction Tests ---

TEST(KvChunkAddressTable, ConstructWithValidConfig) {
    KvChunkAddressTableConfig config{
        .num_layers = 10, .max_sequence_length = 1024, .num_slots = 4, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_EQ(table.config().num_layers, 10u);
    EXPECT_EQ(table.config().max_sequence_length, 1024u);
    EXPECT_EQ(table.config().num_slots, 4u);
    EXPECT_EQ(table.config().chunk_n_tokens, 32u);
    EXPECT_EQ(table.num_position_chunks(), 1024u / 32u);
    EXPECT_EQ(table.total_entries(), 10u * 32u * 4u);
}

TEST(KvChunkAddressTable, ConstructWithNonAlignedSequenceLength) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 100, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    // ceil(100/32) = 4 chunks
    EXPECT_EQ(table.num_position_chunks(), 4u);
    EXPECT_EQ(table.total_entries(), 1u * 4u * 1u);
}

TEST(KvChunkAddressTable, ConstructWithZeroChunkSizeThrows) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 0};
    EXPECT_ANY_THROW(KvChunkAddressTable table(config));
}

TEST(KvChunkAddressTable, ConstructEmptyTable) {
    KvChunkAddressTableConfig config{.num_layers = 0, .max_sequence_length = 0, .num_slots = 0, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);
    EXPECT_EQ(table.total_entries(), 0u);
}

// --- Device Group Tests ---

TEST(KvChunkAddressTable, AddDeviceGroupSingle) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    auto idx = table.add_device_group({make_fnid(0, 0)});
    EXPECT_EQ(idx, DeviceGroupIndex{0});
    EXPECT_EQ(table.num_device_groups(), 1u);

    const auto& group = table.get_device_group(idx);
    ASSERT_EQ(group.fabric_node_ids.size(), 1u);
    EXPECT_EQ(group.fabric_node_ids[0], make_fnid(0, 0));
}

TEST(KvChunkAddressTable, AddDeviceGroupDeduplicates) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    auto idx1 = table.add_device_group({make_fnid(0, 0), make_fnid(0, 1)});
    auto idx2 = table.add_device_group({make_fnid(0, 0), make_fnid(0, 1)});
    EXPECT_EQ(idx1, idx2);
    EXPECT_EQ(table.num_device_groups(), 1u);
}

TEST(KvChunkAddressTable, AddDeviceGroupSortsForDedup) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    // Same nodes, different insertion order — should dedup to same index.
    auto idx1 = table.add_device_group({make_fnid(0, 2), make_fnid(0, 0), make_fnid(0, 1)});
    auto idx2 = table.add_device_group({make_fnid(0, 1), make_fnid(0, 2), make_fnid(0, 0)});
    EXPECT_EQ(idx1, idx2);
    EXPECT_EQ(table.num_device_groups(), 1u);

    // Verify stored sorted.
    const auto& group = table.get_device_group(idx1);
    ASSERT_EQ(group.fabric_node_ids.size(), 3u);
    EXPECT_EQ(group.fabric_node_ids[0], make_fnid(0, 0));
    EXPECT_EQ(group.fabric_node_ids[1], make_fnid(0, 1));
    EXPECT_EQ(group.fabric_node_ids[2], make_fnid(0, 2));
}

TEST(KvChunkAddressTable, AddDeviceGroupDistinctGroups) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    auto idx1 = table.add_device_group({make_fnid(0, 0), make_fnid(0, 1)});
    auto idx2 = table.add_device_group({make_fnid(0, 2), make_fnid(0, 3)});
    EXPECT_NE(idx1, idx2);
    EXPECT_EQ(table.num_device_groups(), 2u);
}

TEST(KvChunkAddressTable, GetDeviceGroupOutOfRangeThrows) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_ANY_THROW(table.get_device_group(DeviceGroupIndex{0}));
}

// --- Set/Lookup Tests ---

TEST(KvChunkAddressTable, SetAndLookupSingleEntry) {
    KvChunkAddressTableConfig config{.num_layers = 2, .max_sequence_length = 128, .num_slots = 2, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    auto grp = table.add_device_group({make_fnid(0, 1)});
    table.set(0, 0, 0, make_location(0xDEAD'0000, 1088 * 18, grp));

    const auto& loc = table.lookup(0, 0, 0);
    EXPECT_EQ(loc.noc_addr, 0xDEAD'0000u);
    EXPECT_EQ(loc.size_bytes, 1088u * 18u);
    EXPECT_EQ(loc.device_group_index, grp);

    const auto& group = table.get_device_group(loc.device_group_index);
    ASSERT_EQ(group.fabric_node_ids.size(), 1u);
    EXPECT_EQ(group.fabric_node_ids[0], make_fnid(0, 1));
}

TEST(KvChunkAddressTable, SetAndLookupMultipleLayers) {
    KvChunkAddressTableConfig config{
        .num_layers = 80, .max_sequence_length = 8192, .num_slots = 8, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    for (uint32_t layer = 0; layer < 80; layer++) {
        uint64_t addr = static_cast<uint64_t>(layer) * 0x1000;
        table.set(layer, 0, 0, make_location(addr, 1088 * 18, grp));
    }

    for (uint32_t layer = 0; layer < 80; layer++) {
        const auto& loc = table.lookup(layer, 0, 0);
        EXPECT_EQ(loc.noc_addr, static_cast<uint64_t>(layer) * 0x1000);
    }
}

TEST(KvChunkAddressTable, SetAndLookupMultiplePositions) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 256, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    for (uint32_t pos = 0; pos < 256; pos += 32) {
        uint64_t addr = static_cast<uint64_t>(pos) * 0x100;
        table.set(0, pos, 0, make_location(addr, 1088 * 18, grp));
    }

    for (uint32_t pos = 0; pos < 256; pos += 32) {
        const auto& loc = table.lookup(0, pos, 0);
        EXPECT_EQ(loc.noc_addr, static_cast<uint64_t>(pos) * 0x100);
    }
}

TEST(KvChunkAddressTable, SetAndLookupMultipleSlots) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 4, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    constexpr size_t num_slots_to_test = 4;
    std::vector<DeviceGroupIndex> grps;
    grps.reserve(num_slots_to_test);
    for (uint32_t slot = 0; slot < num_slots_to_test; slot++) {
        grps.push_back(table.add_device_group({make_fnid(0, slot)}));
    }

    for (uint32_t slot = 0; slot < num_slots_to_test; slot++) {
        table.set(0, 0, slot, make_location(slot * 0x1000, 100, grps[slot]));
    }

    for (uint32_t slot = 0; slot < num_slots_to_test; slot++) {
        const auto& loc = table.lookup(0, 0, slot);
        EXPECT_EQ(loc.noc_addr, slot * 0x1000u);
        const auto& group = table.get_device_group(loc.device_group_index);
        ASSERT_EQ(group.fabric_node_ids.size(), 1u);
        EXPECT_EQ(group.fabric_node_ids[0], make_fnid(0, slot));
    }
}

TEST(KvChunkAddressTable, SetAndLookupMultipleSlotsAndLayersAndPositions) {
    // Asymmetric dimensions: 5 layers, 7 slots, 384 tokens (12 chunks of 32).
    constexpr uint32_t kNumLayers = 5;
    constexpr uint32_t kSeqLen = 384;
    constexpr uint32_t kNumSlots = 7;
    constexpr uint32_t kChunkSize = 32;

    KvChunkAddressTableConfig config{
        .num_layers = kNumLayers,
        .max_sequence_length = kSeqLen,
        .num_slots = kNumSlots,
        .chunk_n_tokens = kChunkSize,
    };
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    // Encode (slot, layer, position) into noc_addr so each entry is uniquely identifiable.
    // Layout: noc_addr = (slot << 40) | (layer << 24) | position
    for (uint32_t slot = 0; slot < kNumSlots; slot++) {
        for (uint32_t layer = 0; layer < kNumLayers; layer++) {
            for (uint32_t pos = 0; pos < kSeqLen; pos += kChunkSize) {
                uint64_t addr = (static_cast<uint64_t>(slot) << 40) | (static_cast<uint64_t>(layer) << 24) |
                                static_cast<uint64_t>(pos);
                uint32_t size = 1000 + (slot * 100) + (layer * 10) + (pos / kChunkSize);
                table.set(
                    layer, pos, slot, KvCacheLocation{.noc_addr = addr, .size_bytes = size, .device_group_index = grp});
            }
        }
    }

    // Verify every entry by decoding the coordinate from noc_addr.
    for (uint32_t slot = 0; slot < kNumSlots; slot++) {
        for (uint32_t layer = 0; layer < kNumLayers; layer++) {
            for (uint32_t pos = 0; pos < kSeqLen; pos += kChunkSize) {
                const auto& loc = table.lookup(layer, pos, slot);

                uint64_t expected_addr = (static_cast<uint64_t>(slot) << 40) | (static_cast<uint64_t>(layer) << 24) |
                                         static_cast<uint64_t>(pos);
                uint32_t expected_size = 1000 + (slot * 100) + (layer * 10) + (pos / kChunkSize);

                EXPECT_EQ(loc.noc_addr, expected_addr)
                    << "addr mismatch at slot=" << slot << " layer=" << layer << " pos=" << pos;
                EXPECT_EQ(loc.size_bytes, expected_size)
                    << "size mismatch at slot=" << slot << " layer=" << layer << " pos=" << pos;
                EXPECT_EQ(loc.device_group_index, grp);
            }
        }
    }

    // Also verify via lookup_range that spans are correct per (slot, layer).
    for (uint32_t slot = 0; slot < kNumSlots; slot++) {
        for (uint32_t layer = 0; layer < kNumLayers; layer++) {
            auto range = table.lookup_range(layer, 0, kSeqLen, slot);
            ASSERT_EQ(range.size(), kSeqLen / kChunkSize);
            for (uint32_t chunk = 0; chunk < range.size(); chunk++) {
                uint32_t pos = chunk * kChunkSize;
                uint64_t expected_addr = (static_cast<uint64_t>(slot) << 40) | (static_cast<uint64_t>(layer) << 24) |
                                         static_cast<uint64_t>(pos);
                EXPECT_EQ(range[chunk].noc_addr, expected_addr);
            }
        }
    }
}

TEST(KvChunkAddressTable, ReplicatedLocation) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    // Column-replicated across 4 chips.
    auto grp = table.add_device_group({make_fnid(0, 0), make_fnid(0, 1), make_fnid(0, 2), make_fnid(0, 3)});
    table.set(0, 0, 0, make_location(0xBEEF, 1088 * 18, grp));

    const auto& loc = table.lookup(0, 0, 0);
    const auto& group = table.get_device_group(loc.device_group_index);
    EXPECT_EQ(group.fabric_node_ids.size(), 4u);
}

// --- Overwrite Tests ---

TEST(KvChunkAddressTable, OverwriteEntry) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    auto grp0 = table.add_device_group({make_fnid(0, 0)});
    auto grp1 = table.add_device_group({make_fnid(0, 1)});

    table.set(0, 0, 0, make_location(0x1111, 100, grp0));
    EXPECT_EQ(table.lookup(0, 0, 0).noc_addr, 0x1111u);

    table.set(0, 0, 0, make_location(0x2222, 200, grp1));
    EXPECT_EQ(table.lookup(0, 0, 0).noc_addr, 0x2222u);
    EXPECT_EQ(table.lookup(0, 0, 0).size_bytes, 200u);
    EXPECT_EQ(table.lookup(0, 0, 0).device_group_index, grp1);
}

// --- Range Lookup Tests ---

TEST(KvChunkAddressTable, LookupRangeContiguous) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 256, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    for (uint32_t pos = 0; pos < 256; pos += 32) {
        table.set(0, pos, 0, make_location(pos, 1088 * 18, grp));
    }

    // Lookup range [64, 192) — should get chunks at positions 64, 96, 128, 160
    auto range = table.lookup_range(0, 64, 192, 0);
    EXPECT_EQ(range.size(), 4u);
    EXPECT_EQ(range[0].noc_addr, 64u);
    EXPECT_EQ(range[1].noc_addr, 96u);
    EXPECT_EQ(range[2].noc_addr, 128u);
    EXPECT_EQ(range[3].noc_addr, 160u);
}

TEST(KvChunkAddressTable, LookupRangeFullSequence) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 128, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    for (uint32_t pos = 0; pos < 128; pos += 32) {
        table.set(0, pos, 0, make_location(pos, 100, grp));
    }

    auto range = table.lookup_range(0, 0, 128, 0);
    EXPECT_EQ(range.size(), 4u);
}

TEST(KvChunkAddressTable, LookupRangeEmptyWhenStartEqualsEnd) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 128, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    auto range = table.lookup_range(0, 64, 64, 0);
    EXPECT_TRUE(range.empty());
}

TEST(KvChunkAddressTable, LookupRangeIsZeroCopy) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 128, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    table.set(0, 32, 0, make_location(0xAAAA, 100, grp));

    auto range = table.lookup_range(0, 32, 64, 0);
    ASSERT_EQ(range.size(), 1u);
    // The span should point into the same memory as a direct lookup.
    EXPECT_EQ(range.data(), &table.lookup(0, 32, 0));
}

// --- Boundary / Error Tests ---

TEST(KvChunkAddressTable, LookupOutOfRangeLayerThrows) {
    KvChunkAddressTableConfig config{.num_layers = 2, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_ANY_THROW(table.lookup(2, 0, 0));
}

TEST(KvChunkAddressTable, LookupOutOfRangePositionThrows) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_ANY_THROW(table.lookup(0, 64, 0));
}

TEST(KvChunkAddressTable, LookupOutOfRangeSlotThrows) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 2, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_ANY_THROW(table.lookup(0, 0, 2));
}

TEST(KvChunkAddressTable, SetOutOfRangeThrows) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_ANY_THROW(table.set(1, 0, 0, KvCacheLocation{}));
    EXPECT_ANY_THROW(table.set(0, 64, 0, KvCacheLocation{}));
    EXPECT_ANY_THROW(table.set(0, 0, 1, KvCacheLocation{}));
}

TEST(KvChunkAddressTable, LookupRangeEndPosBeyondMaxThrows) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 128, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_ANY_THROW(table.lookup_range(0, 0, 129, 0));
}

// --- FabricNodeId -> Host Mapping Tests ---

TEST(KvChunkAddressTable, SetAndGetHost) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    auto fnid = make_fnid(0, 0);
    table.set_fabric_node_host(fnid, "host-0");

    EXPECT_TRUE(table.has_host(fnid));
    EXPECT_EQ(table.get_host(fnid), "host-0");
}

TEST(KvChunkAddressTable, GetHostThrowsForUnknownNode) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_FALSE(table.has_host(make_fnid(0, 0)));
    EXPECT_ANY_THROW(table.get_host(make_fnid(0, 0)));
}

TEST(KvChunkAddressTable, MultipleHostMappings) {
    KvChunkAddressTableConfig config{.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    // 32 devices across 4 meshes, 8 chips each.
    for (uint32_t mesh = 0; mesh < 4; mesh++) {
        for (uint32_t chip = 0; chip < 8; chip++) {
            auto fnid = make_fnid(mesh, chip);
            std::string host = "host-" + std::to_string(mesh);
            table.set_fabric_node_host(fnid, host);
        }
    }

    for (uint32_t mesh = 0; mesh < 4; mesh++) {
        for (uint32_t chip = 0; chip < 8; chip++) {
            EXPECT_EQ(table.get_host(make_fnid(mesh, chip)), "host-" + std::to_string(mesh));
        }
    }
}

// --- Realistic Scale Test ---

TEST(KvChunkAddressTable, RealisticPrefillScale) {
    // Simulates a realistic prefill scenario:
    // 80 layers, 102400 seq_len (3200 chunks of 32), 8 slots
    constexpr uint32_t kNumLayers = 80;
    constexpr uint32_t kSeqLen = 102400;
    constexpr uint32_t kNumSlots = 8;
    constexpr uint32_t kChunkSize = 32;
    constexpr uint32_t kPageSizeBytes = 1088 * 18;

    KvChunkAddressTableConfig config{
        .num_layers = kNumLayers,
        .max_sequence_length = kSeqLen,
        .num_slots = kNumSlots,
        .chunk_n_tokens = kChunkSize,
    };
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    EXPECT_EQ(table.num_position_chunks(), kSeqLen / kChunkSize);

    // Populate a single slot/layer to verify addressing.
    uint32_t layer = 0;
    uint32_t slot = 0;
    for (uint32_t pos = 0; pos < kSeqLen; pos += kChunkSize) {
        uint64_t addr = 0x8000'0000 + (static_cast<uint64_t>(pos / kChunkSize) * kPageSizeBytes);
        table.set(layer, pos, slot, make_location(addr, kPageSizeBytes, grp));
    }

    // Verify a range lookup.
    auto range = table.lookup_range(layer, 0, kSeqLen, slot);
    EXPECT_EQ(range.size(), kSeqLen / kChunkSize);
    EXPECT_EQ(range[0].noc_addr, 0x8000'0000u);
    EXPECT_EQ(range[1].noc_addr, 0x8000'0000u + kPageSizeBytes);
}

// --- Decode Scale Test ---

TEST(KvChunkAddressTable, DecodeAccessPattern) {
    // Simulates the decode migrate() access pattern:
    // For a fixed slot, iterate layers, then iterate positions.
    constexpr uint32_t kNumLayers = 80;
    constexpr uint32_t kSeqLen = 8192;
    constexpr uint32_t kNumSlots = 4;
    constexpr uint32_t kChunkSize = 32;

    KvChunkAddressTableConfig config{
        .num_layers = kNumLayers, .max_sequence_length = kSeqLen, .num_slots = kNumSlots, .chunk_n_tokens = kChunkSize};
    KvChunkAddressTable table(config);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    // Populate all entries.
    for (uint32_t slot = 0; slot < kNumSlots; slot++) {
        for (uint32_t layer = 0; layer < kNumLayers; layer++) {
            for (uint32_t pos = 0; pos < kSeqLen; pos += kChunkSize) {
                uint64_t addr = (static_cast<uint64_t>(slot) << 40) | (static_cast<uint64_t>(layer) << 24) |
                                static_cast<uint64_t>(pos);
                table.set(layer, pos, slot, make_location(addr, 1088 * 18, grp));
            }
        }
    }

    // Simulate migrate(start_pos=1024, end_pos=4096, slot_id=2, layer=5)
    uint32_t slot = 2;
    uint32_t layer = 5;
    uint32_t start_pos = 1024;
    uint32_t end_pos = 4096;

    auto range = table.lookup_range(layer, start_pos, end_pos, slot);
    uint32_t expected_chunks = (end_pos - start_pos) / kChunkSize;
    EXPECT_EQ(range.size(), expected_chunks);

    // Verify first and last entry addresses.
    uint64_t expected_first = (static_cast<uint64_t>(slot) << 40) | (static_cast<uint64_t>(layer) << 24) | start_pos;
    uint64_t expected_last =
        (static_cast<uint64_t>(slot) << 40) | (static_cast<uint64_t>(layer) << 24) | (end_pos - kChunkSize);
    EXPECT_EQ(range.front().noc_addr, expected_first);
    EXPECT_EQ(range.back().noc_addr, expected_last);
}

// --- Multi-Config Tests ---

TEST(KvChunkAddressTable, SingleConfigConstructorHasOneConfigNamedZero) {
    KvChunkAddressTableConfig config{.num_layers = 2, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(config);

    EXPECT_EQ(table.num_configs(), 1u);
    EXPECT_EQ(table.config_name(0), "0");
    EXPECT_EQ(table.config_id_of("0"), 0u);
    // Default-config accessors match the lone config.
    EXPECT_EQ(table.config().num_layers, 2u);
    EXPECT_EQ(table.num_position_chunks(), 64u / 32u);
}

TEST(KvChunkAddressTable, SpanConstructorNamesAreStringIndices) {
    std::vector<KvChunkAddressTableConfig> configs = {
        {.num_layers = 2, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32},
        {.num_layers = 4, .max_sequence_length = 128, .num_slots = 2, .chunk_n_tokens = 32},
        {.num_layers = 1, .max_sequence_length = 256, .num_slots = 1, .chunk_n_tokens = 64},
    };
    KvChunkAddressTable table(std::span<const KvChunkAddressTableConfig>{configs});

    ASSERT_EQ(table.num_configs(), 3u);
    EXPECT_EQ(table.config_name(0), "0");
    EXPECT_EQ(table.config_name(1), "1");
    EXPECT_EQ(table.config_name(2), "2");
    EXPECT_EQ(table.config_id_of("0"), 0u);
    EXPECT_EQ(table.config_id_of("2"), 2u);

    // Each config keeps its own dims.
    EXPECT_EQ(table.config(0).num_layers, 2u);
    EXPECT_EQ(table.config(1).num_layers, 4u);
    EXPECT_EQ(table.config(2).chunk_n_tokens, 64u);
    EXPECT_EQ(table.num_position_chunks(2), 256u / 64u);
}

TEST(KvChunkAddressTable, MapConstructorAssignsIdsInSortedKeyOrder) {
    std::map<std::string, KvChunkAddressTableConfig> configs = {
        {"kv", {.num_layers = 2, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32}},
        {"index_k", {.num_layers = 4, .max_sequence_length = 128, .num_slots = 1, .chunk_n_tokens = 32}},
    };
    KvChunkAddressTable table(configs);

    ASSERT_EQ(table.num_configs(), 2u);
    // std::map sorts keys: "index_k" < "kv".
    EXPECT_EQ(table.config_name(0), "index_k");
    EXPECT_EQ(table.config_name(1), "kv");
    EXPECT_EQ(table.config_id_of("index_k"), 0u);
    EXPECT_EQ(table.config_id_of("kv"), 1u);
    EXPECT_EQ(table.config(table.config_id_of("kv")).num_layers, 2u);
    EXPECT_EQ(table.config(table.config_id_of("index_k")).num_layers, 4u);
}

TEST(KvChunkAddressTable, SetAndLookupByIdAndNameAreEquivalent) {
    std::map<std::string, KvChunkAddressTableConfig> configs = {
        {"kv", {.num_layers = 2, .max_sequence_length = 128, .num_slots = 2, .chunk_n_tokens = 32}},
        {"index_k", {.num_layers = 2, .max_sequence_length = 128, .num_slots = 2, .chunk_n_tokens = 32}},
    };
    KvChunkAddressTable table(configs);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    uint32_t kv_id = table.config_id_of("kv");
    table.set(1, 64, 1, make_location(0xAB, 100, grp), kv_id);
    // Reads back identically by id and by name.
    EXPECT_EQ(table.lookup(1, 64, 1, kv_id).noc_addr, 0xABu);
    EXPECT_EQ(table.lookup(1, 64, 1, "kv").noc_addr, 0xABu);

    // Set via name, read via id.
    table.set(0, 0, 0, make_location(0xCD, 200, grp), "index_k");
    EXPECT_EQ(table.lookup(0, 0, 0, table.config_id_of("index_k")).noc_addr, 0xCDu);

    // Configs are independent: writing "kv" does not touch "index_k".
    EXPECT_EQ(table.lookup(1, 64, 1, "index_k").noc_addr, 0u);
}

TEST(KvChunkAddressTable, ConfigsWithDifferentChunkSizesIndexIndependently) {
    std::vector<KvChunkAddressTableConfig> configs = {
        {.num_layers = 1, .max_sequence_length = 256, .num_slots = 1, .chunk_n_tokens = 32},
        {.num_layers = 1, .max_sequence_length = 256, .num_slots = 1, .chunk_n_tokens = 64},
    };
    KvChunkAddressTable table(std::span<const KvChunkAddressTableConfig>{configs});
    auto grp = table.add_device_group({make_fnid(0, 0)});

    // config 0: 32-token chunks; config 1: 64-token chunks.
    for (uint32_t pos = 0; pos < 256; pos += 32) {
        table.set(0, pos, 0, make_location(pos, 10, grp), 0u);
    }
    for (uint32_t pos = 0; pos < 256; pos += 64) {
        table.set(0, pos, 0, make_location(pos + 1, 10, grp), 1u);
    }

    auto r0 = table.lookup_range(0, 0, 256, 0, 0u);
    auto r1 = table.lookup_range(0, 0, 256, 0, 1u);
    EXPECT_EQ(r0.size(), 256u / 32u);
    EXPECT_EQ(r1.size(), 256u / 64u);
    EXPECT_EQ(r0[1].noc_addr, 32u);
    EXPECT_EQ(r1[1].noc_addr, 64u + 1u);

    // position 32 is not chunk-aligned for config 1 -> must throw.
    EXPECT_ANY_THROW(table.set(0, 32, 0, KvCacheLocation{}, 1u));
}

TEST(KvChunkAddressTable, TotalEntriesSumsAcrossConfigs) {
    std::vector<KvChunkAddressTableConfig> configs = {
        {.num_layers = 2, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32},   // 2*2*1 = 4
        {.num_layers = 1, .max_sequence_length = 128, .num_slots = 3, .chunk_n_tokens = 32},  // 1*4*3 = 12
    };
    KvChunkAddressTable table(std::span<const KvChunkAddressTableConfig>{configs});
    EXPECT_EQ(table.total_entries(), 4u + 12u);
}

TEST(KvChunkAddressTable, DeviceGroupsAreSharedAcrossConfigs) {
    std::vector<KvChunkAddressTableConfig> configs = {
        {.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32},
        {.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32},
    };
    KvChunkAddressTable table(std::span<const KvChunkAddressTableConfig>{configs});

    auto grp = table.add_device_group({make_fnid(0, 5)});
    EXPECT_EQ(table.num_device_groups(), 1u);
    // The same index resolves in either config.
    table.set(0, 0, 0, make_location(0x1, 10, grp), 0u);
    table.set(0, 0, 0, make_location(0x2, 10, grp), 1u);
    EXPECT_EQ(table.get_device_group(table.lookup(0, 0, 0, 0u).device_group_index).fabric_node_ids[0], make_fnid(0, 5));
    EXPECT_EQ(table.get_device_group(table.lookup(0, 0, 0, 1u).device_group_index).fabric_node_ids[0], make_fnid(0, 5));
}

TEST(KvChunkAddressTable, EmptyConfigSpanThrows) {
    std::vector<KvChunkAddressTableConfig> empty;
    EXPECT_ANY_THROW(KvChunkAddressTable table(std::span<const KvChunkAddressTableConfig>{empty}));

    std::map<std::string, KvChunkAddressTableConfig> empty_map;
    EXPECT_ANY_THROW(KvChunkAddressTable table(empty_map));
}

TEST(KvChunkAddressTable, OutOfRangeConfigIdThrows) {
    std::vector<KvChunkAddressTableConfig> configs = {
        {.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32},
    };
    KvChunkAddressTable table(std::span<const KvChunkAddressTableConfig>{configs});

    EXPECT_ANY_THROW(table.lookup(0, 0, 0, 1u));
    EXPECT_ANY_THROW(table.set(0, 0, 0, KvCacheLocation{}, 1u));
    EXPECT_ANY_THROW(table.config(1));
    EXPECT_ANY_THROW(table.config_name(1));
}

TEST(KvChunkAddressTable, UnknownConfigNameThrows) {
    std::vector<KvChunkAddressTableConfig> configs = {
        {.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32},
    };
    KvChunkAddressTable table(std::span<const KvChunkAddressTableConfig>{configs});

    EXPECT_ANY_THROW(table.lookup(0, 0, 0, "nope"));
    EXPECT_ANY_THROW(table.config_id_of("nope"));
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation
