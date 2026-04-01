// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "impl/experimental/disaggregation/kv_chunk_address_table_protobuf.hpp"

namespace tt::tt_metal::experimental::disaggregation {
namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::MeshId;

FabricNodeId make_fnid(uint32_t mesh, uint32_t chip) { return FabricNodeId(MeshId{mesh}, chip); }

// Simple deterministic hash to produce pseudo-random but reproducible data from indices.
uint64_t pseudo_rand(uint32_t a, uint32_t b, uint32_t c) {
    uint64_t h = static_cast<uint64_t>(a) * 2654435761ULL;
    h ^= static_cast<uint64_t>(b) * 2246822519ULL;
    h ^= static_cast<uint64_t>(c) * 3266489917ULL;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h;
}

// Builds a table with asymmetric dimensions and randomized data:
//   7 layers, 384 seq_len (12 chunks of 32), 5 slots, 6 device groups
KvChunkAddressTable make_test_table() {
    constexpr uint32_t kNumLayers = 7;
    constexpr uint32_t kSeqLen = 384;
    constexpr uint32_t kNumSlots = 5;
    constexpr uint32_t kChunkSize = 32;

    KvChunkAddressTableConfig cfg{
        .num_layers = kNumLayers,
        .max_sequence_length = kSeqLen,
        .num_slots = kNumSlots,
        .chunk_n_tokens = kChunkSize,
    };
    KvChunkAddressTable table(cfg);

    // 6 device groups with varying sizes across 3 meshes.
    auto grp0 = table.add_device_group({make_fnid(0, 0)});
    auto grp1 = table.add_device_group({make_fnid(0, 0), make_fnid(0, 1)});
    auto grp2 = table.add_device_group({make_fnid(0, 2), make_fnid(0, 3), make_fnid(0, 4)});
    auto grp3 = table.add_device_group({make_fnid(1, 0), make_fnid(1, 1), make_fnid(1, 2), make_fnid(1, 3)});
    auto grp4 = table.add_device_group({make_fnid(2, 0), make_fnid(2, 1)});
    auto grp5 =
        table.add_device_group({make_fnid(0, 0), make_fnid(1, 0), make_fnid(2, 0), make_fnid(2, 1), make_fnid(2, 2)});
    std::array<DeviceGroupIndex, 6> groups = {grp0, grp1, grp2, grp3, grp4, grp5};

    // Host mappings across 3 hosts.
    for (uint32_t chip = 0; chip < 5; chip++) {
        table.set_fabric_node_host(make_fnid(0, chip), "alpha-host");
    }
    for (uint32_t chip = 0; chip < 4; chip++) {
        table.set_fabric_node_host(make_fnid(1, chip), "beta-host");
    }
    for (uint32_t chip = 0; chip < 3; chip++) {
        table.set_fabric_node_host(make_fnid(2, chip), "gamma-host");
    }

    // Populate every entry with pseudo-random data.
    for (uint32_t slot = 0; slot < kNumSlots; slot++) {
        for (uint32_t layer = 0; layer < kNumLayers; layer++) {
            for (uint32_t pos = 0; pos < kSeqLen; pos += kChunkSize) {
                uint64_t h = pseudo_rand(slot, layer, pos);
                uint64_t addr = 0x1'0000'0000ULL + (h & 0xFFFF'FFFF'FFFF'FF00ULL);
                uint32_t size = 512 + (static_cast<uint32_t>((h >> 8) % 8) * 128);
                DeviceGroupIndex grp_idx = groups[h % groups.size()];
                table.set(
                    layer,
                    pos,
                    slot,
                    KvCacheLocation{.noc_addr = addr, .size_bytes = size, .device_group_index = grp_idx});
            }
        }
    }

    return table;
}

TEST(KvChunkAddressTableProtobuf, RoundTripViaString) {
    auto original = make_test_table();
    std::string data = export_to_protobuf(original);
    auto restored = import_from_protobuf(data);

    EXPECT_EQ(restored.config().num_layers, original.config().num_layers);
    EXPECT_EQ(restored.config().max_sequence_length, original.config().max_sequence_length);
    EXPECT_EQ(restored.config().num_slots, original.config().num_slots);
    EXPECT_EQ(restored.config().chunk_n_tokens, original.config().chunk_n_tokens);

    ASSERT_EQ(restored.num_device_groups(), original.num_device_groups());
    for (size_t i = 0; i < original.num_device_groups(); i++) {
        EXPECT_EQ(
            original.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)}),
            restored.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)}));
    }

    for (size_t i = 0; i < original.num_device_groups(); i++) {
        const auto& group = original.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)});
        for (const auto& fnid : group.fabric_node_ids) {
            ASSERT_TRUE(restored.has_host(fnid));
            EXPECT_EQ(restored.get_host(fnid), original.get_host(fnid));
        }
    }

    const auto& cfg = original.config();
    for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
        for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
            for (uint32_t pos = 0; pos < cfg.max_sequence_length; pos += cfg.chunk_n_tokens) {
                const auto& orig = original.lookup(layer, pos, slot);
                const auto& rest = restored.lookup(layer, pos, slot);
                EXPECT_EQ(rest.noc_addr, orig.noc_addr)
                    << "mismatch at slot=" << slot << " layer=" << layer << " pos=" << pos;
                EXPECT_EQ(rest.size_bytes, orig.size_bytes);
                EXPECT_EQ(rest.device_group_index, orig.device_group_index);
            }
        }
    }
}

TEST(KvChunkAddressTableProtobuf, RoundTripViaFile) {
    auto original = make_test_table();

    std::string tmp_path = std::filesystem::temp_directory_path() / "kv_chunk_address_table_test.pb";
    export_to_protobuf_file(original, tmp_path);

    auto restored = import_from_protobuf_file(tmp_path);

    EXPECT_EQ(restored.config().num_layers, original.config().num_layers);
    EXPECT_EQ(restored.config().max_sequence_length, original.config().max_sequence_length);
    EXPECT_EQ(restored.total_entries(), original.total_entries());

    // Exhaustive check through file round-trip too.
    const auto& cfg = original.config();
    for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
        for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
            for (uint32_t pos = 0; pos < cfg.max_sequence_length; pos += cfg.chunk_n_tokens) {
                const auto& orig = original.lookup(layer, pos, slot);
                const auto& rest = restored.lookup(layer, pos, slot);
                EXPECT_EQ(rest.noc_addr, orig.noc_addr)
                    << "mismatch at slot=" << slot << " layer=" << layer << " pos=" << pos;
                EXPECT_EQ(rest.size_bytes, orig.size_bytes);
                EXPECT_EQ(rest.device_group_index, orig.device_group_index);
            }
        }
    }

    std::filesystem::remove(tmp_path);
}

TEST(KvChunkAddressTableProtobuf, LargeAddressPreserved) {
    KvChunkAddressTableConfig cfg{.num_layers = 1, .max_sequence_length = 32, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(cfg);
    table.add_device_group({make_fnid(0, 0)});
    table.set(
        0,
        0,
        0,
        KvCacheLocation{
            .noc_addr = 0xDEAD'BEEF'CAFE'0000ULL, .size_bytes = 100, .device_group_index = DeviceGroupIndex{0}});

    std::string data = export_to_protobuf(table);
    auto restored = import_from_protobuf(data);

    EXPECT_EQ(restored.lookup(0, 0, 0).noc_addr, 0xDEAD'BEEF'CAFE'0000ULL);
}

TEST(KvChunkAddressTableProtobuf, EmptyTableRoundTrip) {
    KvChunkAddressTableConfig cfg{.num_layers = 2, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32};
    KvChunkAddressTable table(cfg);

    std::string data = export_to_protobuf(table);
    auto restored = import_from_protobuf(data);

    EXPECT_EQ(restored.config().num_layers, 2u);
    EXPECT_EQ(restored.config().max_sequence_length, 64u);
    EXPECT_EQ(restored.total_entries(), table.total_entries());
    EXPECT_EQ(restored.num_device_groups(), 0u);
}

TEST(KvChunkAddressTableProtobuf, SparseTableRoundTrip) {
    KvChunkAddressTableConfig cfg{.num_layers = 4, .max_sequence_length = 256, .num_slots = 2, .chunk_n_tokens = 32};
    KvChunkAddressTable table(cfg);
    auto grp = table.add_device_group({make_fnid(0, 0)});

    table.set(2, 64, 0, KvCacheLocation{.noc_addr = 0xAAAA, .size_bytes = 100, .device_group_index = grp});
    table.set(2, 128, 0, KvCacheLocation{.noc_addr = 0xBBBB, .size_bytes = 200, .device_group_index = grp});

    std::string data = export_to_protobuf(table);
    auto restored = import_from_protobuf(data);

    EXPECT_EQ(restored.lookup(2, 64, 0).noc_addr, 0xAAAAu);
    EXPECT_EQ(restored.lookup(2, 128, 0).noc_addr, 0xBBBBu);
    EXPECT_EQ(restored.lookup(0, 0, 0).noc_addr, 0u);
    EXPECT_EQ(restored.lookup(3, 0, 1).noc_addr, 0u);
}

// --- Text format round-trip (debug API) ---

TEST(KvChunkAddressTableProtobuf, TextFormatRoundTripViaString) {
    auto original = make_test_table();
    std::string text = export_to_protobuf_text(original);
    auto restored = import_from_protobuf_text(text);

    EXPECT_EQ(restored.config().num_layers, original.config().num_layers);
    EXPECT_EQ(restored.config().max_sequence_length, original.config().max_sequence_length);
    EXPECT_EQ(restored.config().num_slots, original.config().num_slots);
    EXPECT_EQ(restored.config().chunk_n_tokens, original.config().chunk_n_tokens);

    ASSERT_EQ(restored.num_device_groups(), original.num_device_groups());
    for (size_t i = 0; i < original.num_device_groups(); i++) {
        EXPECT_EQ(
            original.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)}),
            restored.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)}));
    }

    const auto& cfg = original.config();
    for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
        for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
            for (uint32_t pos = 0; pos < cfg.max_sequence_length; pos += cfg.chunk_n_tokens) {
                const auto& orig = original.lookup(layer, pos, slot);
                const auto& rest = restored.lookup(layer, pos, slot);
                EXPECT_EQ(rest.noc_addr, orig.noc_addr)
                    << "mismatch at slot=" << slot << " layer=" << layer << " pos=" << pos;
                EXPECT_EQ(rest.size_bytes, orig.size_bytes);
                EXPECT_EQ(rest.device_group_index, orig.device_group_index);
            }
        }
    }
}

TEST(KvChunkAddressTableProtobuf, TextFormatRoundTripViaFile) {
    auto original = make_test_table();

    std::string tmp_path = std::filesystem::temp_directory_path() / "kv_chunk_address_table_test.textproto";
    export_to_protobuf_text_file(original, tmp_path);

    auto restored = import_from_protobuf_text_file(tmp_path);

    const auto& cfg = original.config();
    for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
        for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
            for (uint32_t pos = 0; pos < cfg.max_sequence_length; pos += cfg.chunk_n_tokens) {
                const auto& orig = original.lookup(layer, pos, slot);
                const auto& rest = restored.lookup(layer, pos, slot);
                EXPECT_EQ(rest.noc_addr, orig.noc_addr)
                    << "mismatch at slot=" << slot << " layer=" << layer << " pos=" << pos;
                EXPECT_EQ(rest.size_bytes, orig.size_bytes);
                EXPECT_EQ(rest.device_group_index, orig.device_group_index);
            }
        }
    }

    std::filesystem::remove(tmp_path);
}

}  // namespace
}  // namespace tt::tt_metal::experimental::disaggregation
