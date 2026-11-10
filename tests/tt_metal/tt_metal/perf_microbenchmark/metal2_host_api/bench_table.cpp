// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side cost of building and reading Metal 2.0 per-node runtime args (issue #60923).
//
// Each benchmark compares three storage choices for name -> node -> value:
//   CurrentTable:  KernelRunArgs::RuntimeArgValues filled with AddRuntimeArgsForNode.
//   NoInnerSearch: the same nested small vectors, with the per-node insert as a plain append.
//                  This is the lower bound for any vector-backed Table with O(1) node lookup.
//   InnerUnorderedMap: linear search by name, then std::unordered_map<NodeCoord, ArgValue> per name.
//   UnorderedMap:  std::unordered_map<std::string, std::unordered_map<NodeCoord, ArgValue>>.
//
// Args are filled node-first, as ops do, with 7 names per node (the issue repro).

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt_stl/small_vector.hpp>

namespace {
std::size_t g_allocation_count = 0;
}  // namespace

// Counts heap allocations, so each build benchmark can report allocations per iteration.
void* operator new(std::size_t size) {
    ++g_allocation_count;
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc{};
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

namespace m2 = tt::tt_metal::experimental;
using m2::ArgValue;
using m2::NodeCoord;

using NamedValues = std::initializer_list<std::pair<std::string, ArgValue>>;

constexpr std::size_t kGridWidth = 11;  // Blackhole worker grid width, as in the issue repro.

std::vector<NodeCoord> make_nodes(std::size_t count) {
    std::vector<NodeCoord> nodes;
    nodes.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        nodes.emplace_back(i % kGridWidth, i / kGridWidth);
    }
    return nodes;
}

struct CurrentTable {
    using Map = m2::KernelRunArgs::RuntimeArgValues;
    static void add(Map& map, const NodeCoord& node, NamedValues named_values) {
        m2::AddRuntimeArgsForNode(map, node, named_values);
    }
};

struct NoInnerSearch {
    using PerNode = ttsl::SmallVector<std::pair<NodeCoord, ArgValue>, 8>;
    using Map = ttsl::SmallVector<std::pair<std::string, PerNode>, 8>;
    static void add(Map& map, const NodeCoord& node, NamedValues named_values) {
        for (const auto& [name, value] : named_values) {
            auto it = std::ranges::find(map, name, &Map::value_type::first);
            if (it == map.end()) {
                map.emplace_back(name, PerNode{});
                it = std::prev(map.end());
            }
            it->second.emplace_back(node, value);
        }
    }
};

struct InnerUnorderedMap {
    using PerNode = std::unordered_map<NodeCoord, ArgValue>;
    using Map = ttsl::SmallVector<std::pair<std::string, PerNode>, 8>;
    static void add(Map& map, const NodeCoord& node, NamedValues named_values) {
        for (const auto& [name, value] : named_values) {
            auto it = std::ranges::find(map, name, &Map::value_type::first);
            if (it == map.end()) {
                map.emplace_back(name, PerNode{});
                it = std::prev(map.end());
            }
            it->second[node] = value;
        }
    }
};

struct UnorderedMap {
    using Map = std::unordered_map<std::string, std::unordered_map<NodeCoord, ArgValue>>;
    static void add(Map& map, const NodeCoord& node, NamedValues named_values) {
        for (const auto& [name, value] : named_values) {
            map[name][node] = value;
        }
    }
};

template <typename Storage>
typename Storage::Map build_node_first(const std::vector<NodeCoord>& nodes) {
    typename Storage::Map map;
    for (const auto& node : nodes) {
        const auto v = static_cast<std::uint32_t>(node.x + node.y);
        Storage::add(map, node, {{"a", v}, {"b", v}, {"c", v}, {"d", v}, {"e", v}, {"f", v}, {"g", v}});
    }
    return map;
}

// Builds one kernel's per-node args and destroys them, as an op does on each cache hit.
template <typename Storage>
void BM_BuildPerNodeArgs(benchmark::State& state) {
    const auto nodes = make_nodes(static_cast<std::size_t>(state.range(0)));
    const std::size_t allocations_before = g_allocation_count;
    for (auto _ : state) {
        auto map = build_node_first<Storage>(nodes);
        benchmark::DoNotOptimize(map);
    }
    state.counters["allocs"] = benchmark::Counter(
        static_cast<double>(g_allocation_count - allocations_before), benchmark::Counter::kAvgIterations);
}

// Reads every value once, as validation and packing do on each dispatch.
template <typename Storage>
void BM_IteratePerNodeArgs(benchmark::State& state) {
    const auto map = build_node_first<Storage>(make_nodes(static_cast<std::size_t>(state.range(0))));
    for (auto _ : state) {
        std::uint64_t sum = 0;
        for (const auto& [name, per_node] : map) {
            for (const auto& [node, value] : per_node) {
                sum += value.to_uint32();
            }
        }
        benchmark::DoNotOptimize(sum);
    }
}

constexpr std::array<const char*, 8> kCommonArgNames = {
    "num_tiles", "start_id", "src_addr", "dst_addr", "num_blocks", "block_size", "packed_scalar", "page_size"};

// Builds a small name-keyed table (for example, common runtime args) and destroys it.
template <typename Map>
void BM_BuildSmallNamedTable(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        Map map;
        for (std::size_t i = 0; i < count; ++i) {
            map[kCommonArgNames[i]] = ArgValue{static_cast<std::uint32_t>(i)};
        }
        benchmark::DoNotOptimize(map);
    }
}

void node_counts(benchmark::internal::Benchmark* b) {
    for (const int n : {1, 8, 32, 64, 110}) {
        b->Arg(n);
    }
}

BENCHMARK_TEMPLATE(BM_BuildPerNodeArgs, CurrentTable)->Apply(node_counts);
BENCHMARK_TEMPLATE(BM_BuildPerNodeArgs, NoInnerSearch)->Apply(node_counts);
BENCHMARK_TEMPLATE(BM_BuildPerNodeArgs, InnerUnorderedMap)->Apply(node_counts);
BENCHMARK_TEMPLATE(BM_BuildPerNodeArgs, UnorderedMap)->Apply(node_counts);

BENCHMARK_TEMPLATE(BM_IteratePerNodeArgs, CurrentTable)->Apply(node_counts);
BENCHMARK_TEMPLATE(BM_IteratePerNodeArgs, NoInnerSearch)->Apply(node_counts);
BENCHMARK_TEMPLATE(BM_IteratePerNodeArgs, UnorderedMap)->Apply(node_counts);

BENCHMARK_TEMPLATE(BM_BuildSmallNamedTable, m2::Table<std::string, ArgValue>)->Arg(3)->Arg(8);
BENCHMARK_TEMPLATE(BM_BuildSmallNamedTable, std::unordered_map<std::string, ArgValue>)->Arg(3)->Arg(8);

}  // namespace

BENCHMARK_MAIN();
