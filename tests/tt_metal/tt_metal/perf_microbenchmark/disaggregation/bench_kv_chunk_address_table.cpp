// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark suite for KvChunkAddressTable lookup performance.
//
// Measures the typical decode access pattern:
//   for a fixed slot_id:
//     for layer in [0, num_layers):
//       for position in range(start_pos, end_pos, chunk_n_tokens):
//         lookup(layer, position, slot_id)
//
// Also benchmarks lookup_range() which returns a contiguous span.

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>
#include <algorithm>

#include "experimental/disaggregation/kv_chunk_address_table.hpp"
#include "experimental/fabric/fabric_types.hpp"

namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::MeshId;
using tt::tt_metal::experimental::disaggregation::KvCacheLocation;
using tt::tt_metal::experimental::disaggregation::KvChunkAddressTable;
using tt::tt_metal::experimental::disaggregation::KvChunkAddressTableConfig;

constexpr uint32_t kPageSizeBytes = 1088 * 18;  // bfp8 KV chunk size

// Populate the entire table with synthetic data.
void populate_table(KvChunkAddressTable& table) {
    const auto& config = table.config();

    // Register a single device group (typical for non-replicated case).
    auto grp = table.add_device_group({FabricNodeId(MeshId{0}, 0)});

    for (uint32_t slot = 0; slot < config.num_slots; slot++) {
        for (uint32_t layer = 0; layer < config.num_layers; layer++) {
            for (uint32_t pos = 0; pos < config.max_sequence_length; pos += config.chunk_n_tokens) {
                KvCacheLocation loc{
                    .noc_addr = 0x8000'0000 + (static_cast<uint64_t>(pos / config.chunk_n_tokens) * kPageSizeBytes),
                    .size_bytes = kPageSizeBytes,
                    .device_group_index = grp,
                };
                table.set(layer, pos, slot, loc);
            }
        }
    }
}

// --- Individual lookup benchmark ---
// Simulates: for layer in layers: for pos in range(start, end, 32): lookup(layer, pos, slot)
//
// Args: [num_layers, max_seq_len, num_slots]
void BM_LookupIndividual(benchmark::State& state) {
    const uint32_t num_layers = static_cast<uint32_t>(state.range(0));
    const uint32_t max_seq_len = static_cast<uint32_t>(state.range(1));
    const uint32_t num_slots = static_cast<uint32_t>(state.range(2));
    constexpr uint32_t chunk_n_tokens = 32;

    KvChunkAddressTableConfig config{
        .num_layers = num_layers,
        .max_sequence_length = max_seq_len,
        .num_slots = num_slots,
        .chunk_n_tokens = chunk_n_tokens,
    };
    KvChunkAddressTable table(config);
    populate_table(table);

    // Fixed slot, iterate all layers and positions.
    const uint32_t slot = 0;

    for ([[maybe_unused]] auto _ : state) {
        uint64_t sink = 0;
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            for (uint32_t pos = 0; pos < max_seq_len; pos += chunk_n_tokens) {
                sink += table.lookup(layer, pos, slot).noc_addr;
            }
        }
        benchmark::DoNotOptimize(sink);
    }

    // Report throughput in lookups/second.
    uint64_t lookups_per_iter = static_cast<uint64_t>(num_layers) * (max_seq_len / chunk_n_tokens);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(lookups_per_iter));
    state.counters["lookups/iter"] = benchmark::Counter(static_cast<double>(lookups_per_iter));
}

// --- Range lookup benchmark ---
// Simulates: for layer in layers: lookup_range(layer, 0, seq_len, slot)
//
// Args: [num_layers, max_seq_len, num_slots]
void BM_LookupRange(benchmark::State& state) {
    const uint32_t num_layers = static_cast<uint32_t>(state.range(0));
    const uint32_t max_seq_len = static_cast<uint32_t>(state.range(1));
    const uint32_t num_slots = static_cast<uint32_t>(state.range(2));
    constexpr uint32_t chunk_n_tokens = 32;

    KvChunkAddressTableConfig config{
        .num_layers = num_layers,
        .max_sequence_length = max_seq_len,
        .num_slots = num_slots,
        .chunk_n_tokens = chunk_n_tokens,
    };
    KvChunkAddressTable table(config);
    populate_table(table);

    const uint32_t slot = 0;

    for ([[maybe_unused]] auto _ : state) {
        uint64_t sink = 0;
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            auto range = table.lookup_range(layer, 0, max_seq_len, slot);
            for (const auto& loc : range) {
                sink += loc.noc_addr;
            }
        }
        benchmark::DoNotOptimize(sink);
    }

    uint64_t lookups_per_iter = static_cast<uint64_t>(num_layers) * (max_seq_len / chunk_n_tokens);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(lookups_per_iter));
    state.counters["lookups/iter"] = benchmark::Counter(static_cast<double>(lookups_per_iter));
}

// --- Partial range lookup (migrate pattern) ---
// Simulates migrate(start_pos, end_pos, slot, layer) called per-layer.
//
// Args: [num_layers, max_seq_len, num_slots, range_size_tokens]
void BM_LookupMigratePattern(benchmark::State& state) {
    const uint32_t num_layers = static_cast<uint32_t>(state.range(0));
    const uint32_t max_seq_len = static_cast<uint32_t>(state.range(1));
    const uint32_t num_slots = static_cast<uint32_t>(state.range(2));
    const uint32_t range_tokens = static_cast<uint32_t>(state.range(3));
    constexpr uint32_t chunk_n_tokens = 32;

    KvChunkAddressTableConfig config{
        .num_layers = num_layers,
        .max_sequence_length = max_seq_len,
        .num_slots = num_slots,
        .chunk_n_tokens = chunk_n_tokens,
    };
    KvChunkAddressTable table(config);
    populate_table(table);

    const uint32_t slot = 0;
    const uint32_t start_pos = 0;
    const uint32_t end_pos = std::min(range_tokens, max_seq_len);

    for ([[maybe_unused]] auto _ : state) {
        uint64_t sink = 0;
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            auto range = table.lookup_range(layer, start_pos, end_pos, slot);
            for (const auto& loc : range) {
                sink += loc.noc_addr;
            }
        }
        benchmark::DoNotOptimize(sink);
    }

    uint64_t lookups_per_iter = static_cast<uint64_t>(num_layers) * ((end_pos - start_pos) / chunk_n_tokens);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(lookups_per_iter));
    state.counters["lookups/iter"] = benchmark::Counter(static_cast<double>(lookups_per_iter));
}

// --- Register benchmarks ---

// Small model: 10 layers, 1024 seq, 4 slots
//                         layers  seq_len  slots
BENCHMARK(BM_LookupIndividual)->Args({10, 1024, 4})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LookupRange)->Args({10, 1024, 4})->Unit(benchmark::kMicrosecond);

// Medium model: 80 layers, 8192 seq, 8 slots
BENCHMARK(BM_LookupIndividual)->Args({80, 8192, 8})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LookupRange)->Args({80, 8192, 8})->Unit(benchmark::kMicrosecond);

// Large model: 80 layers, 131072 seq (128K), 8 slots
BENCHMARK(BM_LookupIndividual)->Args({80, 131072, 8})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LookupRange)->Args({80, 131072, 8})->Unit(benchmark::kMillisecond);

// Very large: 100 layers, 2097152 seq (2M), 4 slots
BENCHMARK(BM_LookupRange)->Args({100, 2097152, 4})->Unit(benchmark::kMillisecond);

// Migrate pattern: partial range lookups
//                                layers  seq_len  slots  range_tokens
BENCHMARK(BM_LookupMigratePattern)->Args({80, 8192, 8, 1024})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LookupMigratePattern)->Args({80, 8192, 8, 4096})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LookupMigratePattern)->Args({80, 131072, 8, 8192})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LookupMigratePattern)->Args({100, 2097152, 4, 8192})->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
