// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "impl/internal/disaggregation/kv_chunk_address_table_protobuf.hpp"
#include "protobuf/kv_chunk_address_table.pb.h"

namespace tt::tt_metal::internal::disaggregation {
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

// --- Multi-config round-trip ---

// Builds a 3-config table (named "kv", "index_k", "v") with asymmetric per-config
// dims and pseudo-random data, sharing one device-group/host side table.
KvChunkAddressTable make_multi_config_table() {
    std::map<std::string, KvChunkAddressTableConfig> configs = {
        {"kv", {.num_layers = 3, .max_sequence_length = 256, .num_slots = 2, .chunk_n_tokens = 32}},
        {"index_k", {.num_layers = 2, .max_sequence_length = 128, .num_slots = 2, .chunk_n_tokens = 64}},
        {"v", {.num_layers = 3, .max_sequence_length = 256, .num_slots = 2, .chunk_n_tokens = 32}},
    };
    KvChunkAddressTable table(configs);

    auto grp0 = table.add_device_group({make_fnid(0, 0)});
    auto grp1 = table.add_device_group({make_fnid(0, 0), make_fnid(0, 1)});
    auto grp2 = table.add_device_group({make_fnid(1, 0), make_fnid(1, 1), make_fnid(1, 2)});
    std::array<DeviceGroupIndex, 3> groups = {grp0, grp1, grp2};
    table.set_fabric_node_host(make_fnid(0, 0), "alpha-host");
    table.set_fabric_node_host(make_fnid(0, 1), "alpha-host");
    table.set_fabric_node_host(make_fnid(1, 0), "beta-host");
    table.set_fabric_node_host(make_fnid(1, 1), "beta-host");
    table.set_fabric_node_host(make_fnid(1, 2), "beta-host");

    for (uint32_t c = 0; c < table.num_configs(); c++) {
        const auto& cfg = table.config(c);
        for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
            for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
                for (uint32_t pos = 0; pos < cfg.max_sequence_length; pos += cfg.chunk_n_tokens) {
                    uint64_t h = pseudo_rand(c * 100 + slot, layer, pos);
                    uint64_t addr = 0x1'0000'0000ULL + (h & 0xFFFF'FFFF'FFFF'FF00ULL);
                    uint32_t size = 512 + (static_cast<uint32_t>((h >> 8) % 8) * 128);
                    table.set(
                        layer,
                        pos,
                        slot,
                        KvCacheLocation{
                            .noc_addr = addr, .size_bytes = size, .device_group_index = groups[h % groups.size()]},
                        c);
                }
            }
        }
    }
    return table;
}

// Verify every (config, slot, layer, pos) entry matches between two tables.
void expect_tables_equal(const KvChunkAddressTable& a, const KvChunkAddressTable& b) {
    ASSERT_EQ(a.num_configs(), b.num_configs());
    for (uint32_t c = 0; c < a.num_configs(); c++) {
        EXPECT_EQ(a.config_name(c), b.config_name(c)) << "config name mismatch at id " << c;
        const auto& ca = a.config(c);
        const auto& cb = b.config(c);
        EXPECT_EQ(ca.num_layers, cb.num_layers);
        EXPECT_EQ(ca.max_sequence_length, cb.max_sequence_length);
        EXPECT_EQ(ca.num_slots, cb.num_slots);
        EXPECT_EQ(ca.chunk_n_tokens, cb.chunk_n_tokens);
        for (uint32_t slot = 0; slot < ca.num_slots; slot++) {
            for (uint32_t layer = 0; layer < ca.num_layers; layer++) {
                for (uint32_t pos = 0; pos < ca.max_sequence_length; pos += ca.chunk_n_tokens) {
                    const auto& la = a.lookup(layer, pos, slot, c);
                    const auto& lb = b.lookup(layer, pos, slot, c);
                    EXPECT_EQ(la.noc_addr, lb.noc_addr)
                        << "config=" << c << " slot=" << slot << " layer=" << layer << " pos=" << pos;
                    EXPECT_EQ(la.size_bytes, lb.size_bytes);
                    EXPECT_EQ(la.device_group_index, lb.device_group_index);
                }
            }
        }
    }
}

TEST(KvChunkAddressTableProtobuf, MultiConfigRoundTripViaString) {
    auto original = make_multi_config_table();
    auto restored = import_from_protobuf(export_to_protobuf(original));

    // Names round-trip (sorted-key order: "index_k" < "kv" < "v").
    ASSERT_EQ(restored.num_configs(), 3u);
    EXPECT_EQ(restored.config_name(0), "index_k");
    EXPECT_EQ(restored.config_name(1), "kv");
    EXPECT_EQ(restored.config_name(2), "v");
    expect_tables_equal(original, restored);

    // Device groups + hosts round-trip (shared side table).
    ASSERT_EQ(restored.num_device_groups(), original.num_device_groups());
    EXPECT_EQ(restored.get_host(make_fnid(1, 2)), "beta-host");
}

TEST(KvChunkAddressTableProtobuf, MultiConfigRoundTripViaFile) {
    auto original = make_multi_config_table();
    std::string tmp_path = std::filesystem::temp_directory_path() / "kv_multi_config_table_test.pb";
    export_to_protobuf_file(original, tmp_path);
    auto restored = import_from_protobuf_file(tmp_path);
    expect_tables_equal(original, restored);
    std::filesystem::remove(tmp_path);
}

TEST(KvChunkAddressTableProtobuf, MultiConfigTextFormatRoundTrip) {
    auto original = make_multi_config_table();
    auto restored = import_from_protobuf_text(export_to_protobuf_text(original));
    expect_tables_equal(original, restored);
}

TEST(KvChunkAddressTableProtobuf, SpanConstructedManyConfigsRoundTrip) {
    // >10 configs exercises that entries are placed by name, not by raw index,
    // even though span auto-names are "0".."N-1" (which do not string-sort numerically).
    std::vector<KvChunkAddressTableConfig> cfgs;
    cfgs.reserve(12);
for (uint32_t i = 0; i < 12; i++) {
        cfgs.push_back({.num_layers = 1, .max_sequence_length = 64, .num_slots = 1, .chunk_n_tokens = 32});
    }
    KvChunkAddressTable original(std::span<const KvChunkAddressTableConfig>{cfgs});
    auto grp = original.add_device_group({make_fnid(0, 0)});
    for (uint32_t i = 0; i < 12; i++) {
        original.set(0, 0, 0, KvCacheLocation{.noc_addr = 0x1000 + i, .size_bytes = 10, .device_group_index = grp}, i);
    }

    auto restored = import_from_protobuf(export_to_protobuf(original));
    ASSERT_EQ(restored.num_configs(), 12u);
    // Each entry must come back under its original name (e.g. name "11" -> 0x100B).
    for (uint32_t i = 0; i < 12; i++) {
        std::string name = std::to_string(i);
        EXPECT_EQ(restored.lookup(0, 0, 0, name).noc_addr, 0x1000u + i) << "config name " << name;
    }
}

TEST(KvChunkAddressTableProtobuf, LegacySingleConfigWireStillReads) {
    // A proto with only the legacy scalar fields (no `configs`) must import as a
    // single config named "0".
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    pb.set_num_layers(2);
    pb.set_max_sequence_length(64);
    pb.set_num_slots(1);
    pb.set_chunk_n_tokens(32);
    auto* g = pb.add_device_groups();
    auto* fnid = g->add_fabric_node_ids();
    fnid->set_mesh_id(0);
    fnid->set_chip_id(0);
    auto* e = pb.add_entries();  // config_idx defaults to 0
    e->set_layer(1);
    e->set_position(32);
    e->set_slot(0);
    e->set_noc_addr(0xABCD);
    e->set_size_bytes(100);
    e->set_device_group_index(0);

    auto restored = import_from_protobuf(pb.SerializeAsString());
    ASSERT_EQ(restored.num_configs(), 1u);
    EXPECT_EQ(restored.config_name(0), "0");
    EXPECT_EQ(restored.lookup(1, 32, 0).noc_addr, 0xABCDu);
}

// --- Strided-run compression (field 11 `runs`) ---

// Scoped override for KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES (read at each export call).
class DualWriteEnvGuard {
public:
    explicit DualWriteEnvGuard(const char* value) {
        if (const char* old = std::getenv("KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES")) {
            old_ = old;
        }
        if (value != nullptr) {
            ::setenv("KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES", value, 1);
        } else {
            ::unsetenv("KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES");
        }
    }
    ~DualWriteEnvGuard() {
        if (old_) {
            ::setenv("KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES", old_->c_str(), 1);
        } else {
            ::unsetenv("KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES");
        }
    }

private:
    std::optional<std::string> old_;
};

// 2 layers x 2 slots x 96 chunks of 32 tokens; each row is affine (chunk i at base + i*stride).
KvChunkAddressTable make_strided_table() {
    KvChunkAddressTableConfig cfg{.num_layers = 2, .max_sequence_length = 96 * 32, .num_slots = 2, .chunk_n_tokens = 32};
    KvChunkAddressTable table(cfg);
    auto grp = table.add_device_group({make_fnid(0, 0)});
    for (uint32_t slot = 0; slot < 2; slot++) {
        for (uint32_t layer = 0; layer < 2; layer++) {
            const uint64_t base = 0x1'0000'0000ULL + (slot * 2 + layer) * 0x1000'0000ULL;
            for (uint32_t i = 0; i < 96; i++) {
                table.set(
                    layer,
                    i * 32,
                    slot,
                    KvCacheLocation{
                        .noc_addr = base + i * 0x10000ULL, .size_bytes = 512, .device_group_index = grp});
            }
        }
    }
    return table;
}

TEST(KvChunkAddressTableProtobuf, StridedRunsRoundTrip) {
    DualWriteEnvGuard guard("0");  // force runs-only: prove runs alone rebuild the table
    auto original = make_strided_table();

    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    ASSERT_TRUE(pb.ParseFromString(export_to_protobuf(original)));
    EXPECT_EQ(pb.entries_size(), 0);
    EXPECT_EQ(pb.runs_size(), 4);  // one run per (slot, layer) row

    auto restored = import_from_protobuf(pb.SerializeAsString());
    expect_tables_equal(original, restored);
}

TEST(KvChunkAddressTableProtobuf, BlockCyclicRunsRoundTrip) {
    // 12-bank round-robin: chunk c at row_base + (c%12)*bank_block + (c/12)*period — the delta
    // sequence alternates, so step-1 detection fails but one affine stream per bank (step 12) fits.
    // period (0x30000) deliberately != kBanks*bank_block (0x18000): equal values would make the
    // pattern purely affine (degenerate) and it would compress at step 1 instead.
    DualWriteEnvGuard guard("0");
    constexpr uint32_t kBanks = 12;
    constexpr uint32_t kChunks = 96;  // 8 full periods
    KvChunkAddressTableConfig cfg{
        .num_layers = 2, .max_sequence_length = kChunks * 32, .num_slots = 2, .chunk_n_tokens = 32};
    KvChunkAddressTable original(cfg);
    auto grp = original.add_device_group({make_fnid(0, 0)});
    for (uint32_t slot = 0; slot < 2; slot++) {
        for (uint32_t layer = 0; layer < 2; layer++) {
            const uint64_t row_base = 0x1'0000'0000ULL + (slot * 2 + layer) * 0x1000'0000ULL;
            for (uint32_t i = 0; i < kChunks; i++) {
                original.set(
                    layer,
                    i * 32,
                    slot,
                    KvCacheLocation{
                        .noc_addr = row_base + (i % kBanks) * 0x2000ULL + (i / kBanks) * 0x30000ULL,
                        .size_bytes = 512,
                        .device_group_index = grp});
            }
        }
    }

    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    ASSERT_TRUE(pb.ParseFromString(export_to_protobuf(original)));
    EXPECT_EQ(pb.entries_size(), 0);
    EXPECT_EQ(pb.runs_size(), 4 * kBanks);  // 4 rows, one run per bank
    for (const auto& run : pb.runs()) {
        EXPECT_EQ(run.chunk_step(), kBanks);
        EXPECT_EQ(run.count(), kChunks / kBanks);
        EXPECT_EQ(run.addr_stride(), 0x30000);
    }

    auto restored = import_from_protobuf(pb.SerializeAsString());
    expect_tables_equal(original, restored);
}

TEST(KvChunkAddressTableProtobuf, MixedCompressionPerConfigRoundTrip) {
    // Compression is per-config: the affine config exports STRIDED_ROWS, the pseudo-random
    // config stays UNROLLED. The importer must honor both tags in one table.
    DualWriteEnvGuard guard("0");  // runs-only for the strided config
    std::map<std::string, KvChunkAddressTableConfig> configs = {
        {"aff", {.num_layers = 1, .max_sequence_length = 64 * 32, .num_slots = 1, .chunk_n_tokens = 32}},
        {"rnd", {.num_layers = 1, .max_sequence_length = 64 * 32, .num_slots = 1, .chunk_n_tokens = 32}},
    };
    KvChunkAddressTable original(configs);
    auto grp = original.add_device_group({make_fnid(0, 0)});
    for (uint32_t i = 0; i < 64; i++) {
        original.set(
            0,
            i * 32,
            0,
            KvCacheLocation{.noc_addr = 0x1'0000'0000ULL + i * 0x8000ULL, .size_bytes = 512, .device_group_index = grp},
            "aff");
        const uint64_t h = pseudo_rand(1, 0, i * 32);
        original.set(
            0,
            i * 32,
            0,
            KvCacheLocation{
                .noc_addr = 0x1'0000'0000ULL + (h & 0xFFFF'FFFFULL), .size_bytes = 512, .device_group_index = grp},
            "rnd");
    }

    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    ASSERT_TRUE(pb.ParseFromString(export_to_protobuf(original)));
    ASSERT_EQ(pb.configs_size(), 2);
    for (const auto& c : pb.configs()) {
        if (c.name() == "aff") {
            EXPECT_EQ(c.compression(), ::tt::disaggregation::proto::STRIDED_ROWS);
        } else {
            EXPECT_EQ(c.compression(), ::tt::disaggregation::proto::UNROLLED);
        }
    }
    EXPECT_EQ(pb.runs_size(), 1);      // the affine config's single row
    EXPECT_EQ(pb.entries_size(), 64);  // the random config's entries

    auto restored = import_from_protobuf(pb.SerializeAsString());
    EXPECT_EQ(restored.compression(restored.config_id_of("aff")), ChunkCompression::kStridedRows);
    EXPECT_EQ(restored.compression(restored.config_id_of("rnd")), ChunkCompression::kUnrolled);
    expect_tables_equal(original, restored);

    // A re-export of the mixed table must preserve the per-config forms: the strided config
    // serializes from its map (no detection), the unrolled one re-detects nothing.
    ::tt::disaggregation::proto::KvChunkAddressTable pb2;
    ASSERT_TRUE(pb2.ParseFromString(export_to_protobuf(restored)));
    EXPECT_EQ(pb2.runs_size(), 1);
    EXPECT_EQ(pb2.entries_size(), 64);
}

TEST(KvChunkAddressTableProtobuf, DualWriteSurvivesReexport) {
    // A small dual-written table imports as an in-memory StridedRowMap (runs authoritative).
    // Re-exporting THAT must keep the entries mirror for old readers downstream.
    DualWriteEnvGuard guard(nullptr);  // default threshold: dual-write
    auto original = make_strided_table();

    auto first = import_from_protobuf(export_to_protobuf(original));
    ASSERT_EQ(first.compression(0), ChunkCompression::kStridedRows);

    ::tt::disaggregation::proto::KvChunkAddressTable pb2;
    ASSERT_TRUE(pb2.ParseFromString(export_to_protobuf(first)));
    EXPECT_EQ(pb2.runs_size(), 4);
    EXPECT_EQ(pb2.entries_size(), 4 * 96);  // mirror survived the re-export

    auto second = import_from_protobuf(pb2.SerializeAsString());
    expect_tables_equal(original, second);
}

TEST(KvChunkAddressTableProtobuf, DualWriteMirrorsEntriesBelowThreshold) {
    DualWriteEnvGuard guard(nullptr);  // default threshold: a small table dual-writes
    auto original = make_strided_table();

    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    ASSERT_TRUE(pb.ParseFromString(export_to_protobuf(original)));
    EXPECT_EQ(pb.runs_size(), 4);
    EXPECT_EQ(pb.entries_size(), 4 * 96);  // every chunk mirrored

    auto restored = import_from_protobuf(pb.SerializeAsString());
    expect_tables_equal(original, restored);
}

TEST(KvChunkAddressTableProtobuf, RunValidationThrows) {
    auto make_pb = [] {
        ::tt::disaggregation::proto::KvChunkAddressTable pb;
        auto* c = pb.add_configs();
        c->set_name("0");
        c->set_num_layers(1);
        c->set_max_sequence_length(64);
        c->set_num_slots(1);
        c->set_chunk_n_tokens(32);
        auto* g = pb.add_device_groups();
        g->add_fabric_node_ids()->set_mesh_id(0);
        return pb;
    };

    {  // chunk_step 0
        auto pb = make_pb();
        auto* r = pb.add_runs();
        r->set_config_idx(0);
        r->set_chunk_step(0);
        r->set_count(1);
        EXPECT_ANY_THROW(import_from_protobuf(pb.SerializeAsString()));
    }
    {  // extends past the config's position chunks (2 exist: 64/32)
        auto pb = make_pb();
        auto* r = pb.add_runs();
        r->set_config_idx(0);
        r->set_start_chunk(1);
        r->set_chunk_step(1);
        r->set_count(2);
        EXPECT_ANY_THROW(import_from_protobuf(pb.SerializeAsString()));
    }
    {  // config_idx out of range
        auto pb = make_pb();
        auto* r = pb.add_runs();
        r->set_config_idx(7);
        r->set_chunk_step(1);
        r->set_count(1);
        EXPECT_ANY_THROW(import_from_protobuf(pb.SerializeAsString()));
    }
}

TEST(KvChunkAddressTableProtobuf, DuplicateConfigNameThrows) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    for (int i = 0; i < 2; i++) {
        auto* c = pb.add_configs();
        c->set_name("dup");
        c->set_num_layers(1);
        c->set_max_sequence_length(32);
        c->set_num_slots(1);
        c->set_chunk_n_tokens(32);
    }
    EXPECT_ANY_THROW(import_from_protobuf(pb.SerializeAsString()));
}

TEST(KvChunkAddressTableProtobuf, EntryConfigIdxOutOfRangeThrows) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    auto* c = pb.add_configs();
    c->set_name("only");
    c->set_num_layers(1);
    c->set_max_sequence_length(32);
    c->set_num_slots(1);
    c->set_chunk_n_tokens(32);
    auto* e = pb.add_entries();
    e->set_layer(0);
    e->set_position(0);
    e->set_slot(0);
    e->set_noc_addr(0x1);
    e->set_size_bytes(10);
    e->set_config_idx(5);  // only one config (idx 0) exists
    EXPECT_ANY_THROW(import_from_protobuf(pb.SerializeAsString()));
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation
