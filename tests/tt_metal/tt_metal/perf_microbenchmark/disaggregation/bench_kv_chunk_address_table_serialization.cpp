// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Serialization benchmark: Protobuf binary vs text export/import for KvChunkAddressTable.
//
// Args: [num_layers, max_seq_len, num_slots]

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

#include "impl/experimental/disaggregation/kv_chunk_address_table_protobuf.hpp"

namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::MeshId;
using tt::tt_metal::experimental::disaggregation::KvCacheLocation;
using tt::tt_metal::experimental::disaggregation::KvChunkAddressTable;
using tt::tt_metal::experimental::disaggregation::KvChunkAddressTableConfig;

constexpr uint32_t kPageSizeBytes = 1088 * 18;

KvChunkAddressTable make_table(uint32_t num_layers, uint32_t max_seq_len, uint32_t num_slots) {
    constexpr uint32_t chunk_n_tokens = 32;
    KvChunkAddressTableConfig cfg{
        .num_layers = num_layers,
        .max_sequence_length = max_seq_len,
        .num_slots = num_slots,
        .chunk_n_tokens = chunk_n_tokens,
    };
    KvChunkAddressTable table(cfg);
    auto grp = table.add_device_group({FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 1)});
    table.set_fabric_node_host(FabricNodeId(MeshId{0}, 0), "host-0");
    table.set_fabric_node_host(FabricNodeId(MeshId{0}, 1), "host-0");

    for (uint32_t slot = 0; slot < num_slots; slot++) {
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            for (uint32_t pos = 0; pos < max_seq_len; pos += chunk_n_tokens) {
                KvCacheLocation loc{
                    .noc_addr = 0x8000'0000 + (static_cast<uint64_t>(pos / chunk_n_tokens) * kPageSizeBytes),
                    .size_bytes = kPageSizeBytes,
                    .device_group_index = grp,
                };
                table.set(layer, pos, slot, loc);
            }
        }
    }
    return table;
}

uint64_t count_entries(const KvChunkAddressTable& table) {
    const auto& cfg = table.config();
    return static_cast<uint64_t>(cfg.num_layers) * (cfg.max_sequence_length / cfg.chunk_n_tokens) * cfg.num_slots;
}

// ---------------------------------------------------------------------------
// Protobuf binary benchmarks
// ---------------------------------------------------------------------------

void BM_ExportProtobuf(benchmark::State& state) {
    auto table = make_table(
        static_cast<uint32_t>(state.range(0)),
        static_cast<uint32_t>(state.range(1)),
        static_cast<uint32_t>(state.range(2)));

    size_t bytes = 0;
    for ([[maybe_unused]] auto _ : state) {
        std::string data = tt::tt_metal::experimental::disaggregation::export_to_protobuf(table);
        bytes = data.size();
        benchmark::DoNotOptimize(data);
    }

    state.counters["bytes"] = benchmark::Counter(static_cast<double>(bytes));
    state.counters["entries"] = benchmark::Counter(static_cast<double>(count_entries(table)));
}

void BM_ImportProtobuf(benchmark::State& state) {
    auto table = make_table(
        static_cast<uint32_t>(state.range(0)),
        static_cast<uint32_t>(state.range(1)),
        static_cast<uint32_t>(state.range(2)));

    std::string data = tt::tt_metal::experimental::disaggregation::export_to_protobuf(table);

    for ([[maybe_unused]] auto _ : state) {
        auto restored = tt::tt_metal::experimental::disaggregation::import_from_protobuf(data);
        benchmark::DoNotOptimize(restored.total_entries());
    }

    state.counters["bytes"] = benchmark::Counter(static_cast<double>(data.size()));
    state.counters["entries"] = benchmark::Counter(static_cast<double>(count_entries(table)));
}

// ---------------------------------------------------------------------------
// Protobuf text format benchmarks (debug path)
// ---------------------------------------------------------------------------

void BM_ExportProtobufText(benchmark::State& state) {
    auto table = make_table(
        static_cast<uint32_t>(state.range(0)),
        static_cast<uint32_t>(state.range(1)),
        static_cast<uint32_t>(state.range(2)));

    size_t bytes = 0;
    for ([[maybe_unused]] auto _ : state) {
        std::string text = tt::tt_metal::experimental::disaggregation::export_to_protobuf_text(table);
        bytes = text.size();
        benchmark::DoNotOptimize(text);
    }

    state.counters["bytes"] = benchmark::Counter(static_cast<double>(bytes));
    state.counters["entries"] = benchmark::Counter(static_cast<double>(count_entries(table)));
}

void BM_ImportProtobufText(benchmark::State& state) {
    auto table = make_table(
        static_cast<uint32_t>(state.range(0)),
        static_cast<uint32_t>(state.range(1)),
        static_cast<uint32_t>(state.range(2)));

    std::string text = tt::tt_metal::experimental::disaggregation::export_to_protobuf_text(table);

    for ([[maybe_unused]] auto _ : state) {
        auto restored = tt::tt_metal::experimental::disaggregation::import_from_protobuf_text(text);
        benchmark::DoNotOptimize(restored.total_entries());
    }

    state.counters["bytes"] = benchmark::Counter(static_cast<double>(text.size()));
    state.counters["entries"] = benchmark::Counter(static_cast<double>(count_entries(table)));
}

// ---------------------------------------------------------------------------
// Register benchmarks
// ---------------------------------------------------------------------------

//                     layers  seq_len  slots
// Small: 1280 entries
BENCHMARK(BM_ExportProtobufText)->Args({10, 1024, 4})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ImportProtobufText)->Args({10, 1024, 4})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ExportProtobuf)->Args({10, 1024, 4})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ImportProtobuf)->Args({10, 1024, 4})->Unit(benchmark::kMillisecond);

// Medium: 163840 entries
BENCHMARK(BM_ExportProtobufText)->Args({80, 8192, 8})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ImportProtobufText)->Args({80, 8192, 8})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ExportProtobuf)->Args({80, 8192, 8})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ImportProtobuf)->Args({80, 8192, 8})->Unit(benchmark::kMillisecond);

// Large: 2621440 entries
BENCHMARK(BM_ExportProtobufText)->Args({80, 131072, 8})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ImportProtobufText)->Args({80, 131072, 8})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ExportProtobuf)->Args({80, 131072, 8})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ImportProtobuf)->Args({80, 131072, 8})->Unit(benchmark::kMillisecond);

}  // namespace

BENCHMARK_MAIN();
