// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host tests for the all_gather plan: legality and geometry, slicer, multicast routes, schedule and sync arithmetic,
// shrink order, footprints, the override grammar, the run-length model and expansion. Facts are injected, so no device
// is needed, and no test depends on a calibration threshold: expectations that involve the model ask the model.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/ccl/all_gather/device/all_gather_device_operation_types.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_plan.hpp"
#include "ttnn/operations/ccl/all_gather/device/kernels/ag_sync.hpp"
#include "ttnn/operations/ccl/all_gather/device/kernels/ag_walk.hpp"

namespace {

using namespace ttnn::operations::ccl::all_gather_plan;
using tt::tt_fabric::FabricConfig;
using tt::tt_fabric::Topology;
using tt::tt_metal::BufferType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::NdShardSpec;
using tt::tt_metal::PageConfig;
using tt::tt_metal::Shape;
using tt::tt_metal::ShardSpec;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::Tile;
using ttnn::operations::ccl::AgAlgo;
using ttnn::operations::ccl::AgChoice;
using ttnn::operations::ccl::AgFacts;
using ttnn::operations::ccl::AgForce;
using ttnn::operations::ccl::AllGatherParams;
using ttnn::operations::ccl::kAgMaxWorkers;

// ---- injected facts ----

struct ArchVals {
    tt::ARCH arch;
    uint32_t dram, l1, burst, banks_dram, banks_l1, headers, P;
};
const ArchVals kBH{tt::ARCH::BLACKHOLE, 64, 16, 16384, 8, 120, 12, 15168};
const ArchVals kWH{tt::ARCH::WORMHOLE_B0, 32, 16, 8192, 12, 64, 8, 7616};
const ArchVals kUnknown{tt::ARCH::QUASAR, 64, 16, 16384, 8, 64, 8, 15168};

CoreRangeSet grid(uint32_t w, uint32_t h) { return CoreRangeSet(CoreRange({0, 0}, {w - 1, h - 1})); }

AgFacts make_facts(
    const ArchVals& a, std::optional<uint32_t> payload = std::nullopt, CoreRangeSet cores = grid(13, 10)) {
    AgFacts f;
    f.arch = a.arch;
    f.payload = payload.value_or(a.P);
    f.header_bytes = 96;
    f.channel_bytes = f.payload + f.header_bytes;
    f.dram_align = a.dram;
    f.l1_align = a.l1;
    f.max_burst = a.burst;
    f.num_banks_dram = a.banks_dram;
    f.num_banks_l1 = a.banks_l1;
    f.l1_base = 100 * 1024;
    f.l1_small_floor = 1400 * 1024;
    f.headers_per_risc = a.headers;
    for (uint32_t n = 1; n <= kAgMaxWorkers; ++n) {
        // Control regions plus n channels of mux_buffers(n) full packets, as FabricMuxV2Config lays them out.
        f.mux_map_end[n - 1] = f.l1_base + 4096 + (n * mux_buffers(n) * f.channel_bytes);
    }
    f.worker_cores = std::move(cores);
    return f;
}

// ---- specs and params ----

constexpr bool D = true;   // DRAM
constexpr bool L = false;  // L1

Shape shape_of(const std::vector<uint32_t>& dims) { return Shape(Shape::Container(dims.begin(), dims.end())); }

MemoryConfig interleaved(BufferType bt) { return MemoryConfig(TensorMemoryLayout::INTERLEAVED, bt); }
MemoryConfig nd_sharded(BufferType bt, const Shape& shard) {
    return MemoryConfig(bt, NdShardSpec{shard, CoreRangeSet(CoreRange({0, 0}, {7, 7}))});
}
MemoryConfig legacy_sharded(TensorMemoryLayout ml, BufferType bt, uint32_t h, uint32_t w, uint32_t cores) {
    return MemoryConfig(ml, bt, ShardSpec(CoreRangeSet(CoreRange({0, 0}, {cores - 1, 0})), {h, w}));
}

TensorSpec make_spec(const Shape& shape, DataType dt, Layout layout, const MemoryConfig& mc, Tile tile = Tile()) {
    return TensorSpec(shape, TensorLayout(dt, PageConfig(layout, tile), mc));
}

enum class Fab { NeighborExchange, Line, Ring, Fabric2DLine, True2D };

// Mirrors all_gather_build_operation_args: a (1, N) mesh along axis 1, or a (2, N/2) mesh for true 2D.
AllGatherParams make_params(
    const TensorSpec& out, int32_t dim, uint32_t N, Fab fab, uint32_t links, const AgFacts& facts, AgForce force = {}) {
    const bool true_2d = fab == Fab::True2D;
    FabricConfig fc = FabricConfig::FABRIC_1D;
    if (fab == Fab::NeighborExchange) {
        fc = FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE;
    } else if (fab == Fab::Ring) {
        fc = FabricConfig::FABRIC_1D_RING;
    } else if (fab == Fab::Fabric2DLine || true_2d) {
        fc = FabricConfig::FABRIC_2D;
    }
    return AllGatherParams{
        .dim_from_end = dim,
        .output_spec = out,
        .cluster_axis = std::nullopt,
        .fabric_config = fc,
        .axis_topology = {Topology::Linear, fab == Fab::Ring ? Topology::Ring : Topology::Linear},
        .axis_is_straight = {true, true},
        .axis_num_devices = {true_2d ? 2u : 1u, true_2d ? N / 2 : N},
        .axis_num_links = {true_2d ? links : 0u, links},
        .num_devices = N,
        .subdevice_id = std::nullopt,
        .sub_core_grid = std::nullopt,
        .facts = facts,
        .force = force};
}

PlanIn plan_in_for(
    const TensorSpec& in,
    const TensorSpec& out,
    int32_t dim,
    uint32_t N,
    Fab fab,
    const AgFacts& facts,
    uint32_t links = 2,
    AgForce force = {}) {
    return make_plan_in(make_params(out, dim, N, fab, links, facts, force), in, out);
}

// Input and output specs of a gather of `out_shape` on `dim` over N devices.
std::pair<TensorSpec, TensorSpec> gather_specs(
    std::vector<uint32_t> out_shape,
    int32_t dim,
    uint32_t N,
    DataType dt = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const MemoryConfig& in_mc = interleaved(BufferType::DRAM),
    const MemoryConfig& out_mc = interleaved(BufferType::DRAM)) {
    auto in_shape = out_shape;
    in_shape[in_shape.size() + dim] /= N;
    return {make_spec(shape_of(in_shape), dt, layout, in_mc), make_spec(shape_of(out_shape), dt, layout, out_mc)};
}

// Whole-page mode: the pages do not cut on A, so each moves at its aligned pitch.
bool whole_page(const Pages& p, const Geometry& g) { return p.ci % g.A != 0 || p.co % g.A != 0; }

// "-" when illegal, else mode, A and footprints, e.g. "w64 1/1" or "s16 2/8".
std::string mode(const Pages& p, const std::optional<Geometry>& g) {
    if (!g) {
        return "-";
    }
    return fmt::format("{}{} {}/{}", whole_page(p, *g) ? 'w' : 's', g->A, g->Ui, g->Uo);
}

Pages page_geo(const ArchVals& a, uint32_t ci, uint32_t co, bool in_dram, bool out_dram) {
    const uint32_t ri = in_dram ? a.dram : a.l1;
    const uint32_t ro = out_dram ? a.dram : a.l1;
    return Pages{
        .ci = ci,
        .co = co,
        .ai = tt::align(ci, ri),
        .ao = tt::align(co, ro),
        .ri = ri,
        .ro = ro,
        .in_pages = 8,
        .pages_per_stripe_row = 1,
        .L = 8,
        .run_pages = 1u << 30};
}

// A PlanIn built straight from page numbers (1D line or ring on axis 1).
PlanIn raw_plan_in(const AgFacts& f, const Pages& pages, uint32_t N, bool ring, uint32_t links = 2) {
    PlanIn p;
    p.facts = f;
    p.pages = pages;
    p.N = N;
    p.links = links;
    p.ring = ring && N > 2;
    p.axis_devices = {1, N};
    p.axis_ring = {false, p.ring};
    return p;
}

const char* algo_str(AgAlgo a) {
    switch (a) {
        case AgAlgo::Auto: return "auto";
        case AgAlgo::Unicast: return "unicast";
        case AgAlgo::Multicast: return "multicast";
        case AgAlgo::Composite: return "composite";
    }
    return "?";
}

// ---- legality table. Unicast's A also covers its own output (relay reads); multicast's only the input ----

TEST(AllGatherPlan, Legality) {
    struct Row {
        const char* name;
        ArchVals a;
        uint32_t ci, co;
        bool in_dram, out_dram;
        const char *unicast, *multicast;
    };
    const std::vector<Row> rows = {
        {"rm 8KB rows", kBH, 8192, 8192, D, D, "s64 128/128", "s64 128/128"},
        {"rm 48B rows", kBH, 48, 48, D, D, "w64 1/1", "w64 1/1"},
        {"rm 32B rows dram", kBH, 32, 32, D, D, "w64 1/1", "w64 1/1"},
        {"rm 32B rows l1", kBH, 32, 32, L, L, "s16 2/2", "s16 2/2"},
        {"rm 16B rows dram", kBH, 16, 16, D, D, "w64 1/1", "w64 1/1"},
        {"rm 16B rows l1->dram", kBH, 16, 16, L, D, "w64 1/1", "s16 1/1"},
        {"rm 16B rows dram->l1", kBH, 16, 16, D, L, "-", "-"},
        {"bfp8 1088B tile", kBH, 1088, 1088, D, D, "s64 17/17", "s64 17/17"},
        {"bfp8 544B tile dram", kBH, 544, 544, D, D, "w64 9/9", "w64 9/9"},
        {"bfp8 544B tile dram->l1 (s > ao)", kBH, 544, 544, D, L, "-", "-"},
        {"bfp8 544B tile l1->dram", kBH, 544, 544, L, D, "w64 9/9", "s16 34/34"},
        {"bf16 tile dram->l1", kBH, 2048, 2048, D, L, "s64 32/32", "s64 32/32"},
        {"bf16 tile l1->dram", kBH, 2048, 2048, L, D, "s64 32/32", "s16 128/128"},
        {"known_bad mismatched_alignment", kBH, 32, 32, L, D, "w64 1/1", "s16 2/2"},
        {"known_bad chunk_over_packet", kBH, 8192, 8192, D, D, "s64 128/128", "s64 128/128"},
        {"20000B rows (> P)", kBH, 20000, 20000, D, D, "w64 313/313", "w64 313/313"},
        {"concat 16B->128B l1->dram", kBH, 16, 128, L, D, "-", "s16 1/8"},
        {"concat 16B->128B l1", kBH, 16, 128, L, L, "s16 1/8", "s16 1/8"},
        {"concat 16B->128B dram", kBH, 16, 128, D, D, "-", "-"},
        {"split 128B->64B l1", kBH, 128, 64, L, L, "s16 8/4", "s16 8/4"},
        {"wh rm 48B rows", kWH, 48, 48, D, D, "w32 2/2", "w32 2/2"},
        {"wh rm 16B rows", kWH, 16, 16, D, D, "w32 1/1", "w32 1/1"},
        {"wh bfp8 544B tile dram->l1", kWH, 544, 544, D, L, "s32 17/17", "s32 17/17"},
        {"wh 20000B rows", kWH, 20000, 20000, D, D, "s32 625/625", "s32 625/625"},
    };
    for (const auto& r : rows) {
        const Pages g = page_geo(r.a, r.ci, r.co, r.in_dram, r.out_dram);
        const PlanIn in = raw_plan_in(make_facts(r.a), g, 8, false);
        const auto u = make_geometry(in, true);
        const auto m = make_geometry(in, false);
        EXPECT_EQ(mode(g, u), r.unicast) << r.name;
        EXPECT_EQ(mode(g, m), r.multicast) << r.name;
        for (const auto& geo : {u, m}) {
            if (!geo) {
                continue;
            }
            EXPECT_EQ(geo->P_A % geo->A, 0u) << r.name;
            EXPECT_LE(geo->P_A, std::min(r.a.P, r.a.burst)) << r.name;
            EXPECT_EQ(geo->dst_cross, geo->Uo * geo->A == g.ao) << r.name;
            EXPECT_EQ(geo->src_cross, geo->Ui * geo->A == g.ai) << r.name;
            EXPECT_GT(geo->run_bytes, 0u) << r.name;
            EXPECT_EQ(geo->fill, std::min(geo->P_A, 4 * geo->run_bytes)) << r.name;
            if (whole_page(g, *geo)) {
                EXPECT_LE(geo->Uo * geo->A, g.ao) << r.name;  // the slot tail lands in the page's own padding
            }
        }
    }

    // Illegal for both algorithms is composite, with the rule that fired, never a TT_FATAL.
    const std::string cut = "cannot cut these pages into aligned units";
    const AgFacts bh = make_facts(kBH);
    PlanIn p = raw_plan_in(bh, page_geo(kBH, 16, 16, D, L), 8, false);
    AgPlan plan = make_plan(p);
    EXPECT_EQ(plan.choice.algo, AgAlgo::Composite);
    EXPECT_EQ(plan.composite_reason, fmt::format("unicast {0}; multicast {0}", cut));
    p.pages = page_geo(kBH, 16, 128, L, D);
    EXPECT_EQ(make_plan(p).choice.algo, AgAlgo::Multicast);
    p.true_2d = true;
    p.facts.worker_cores = CoreRangeSet();
    EXPECT_EQ(make_plan(p).composite_reason, "unicast needs a 1D axis; multicast has no free worker cores");
    // Neighbor exchange: composite works on 2 devices; past 2 it cannot route either, so the plan refuses before
    // launch instead of hanging.
    PlanIn ne = raw_plan_in(bh, page_geo(kBH, 16, 128, L, D), 2, false);
    ne.neighbor_exchange = true;
    EXPECT_EQ(
        make_plan(ne).composite_reason,
        fmt::format("unicast {}; multicast cannot route on a neighbor-exchange fabric (one hop only)", cut));
    ne = raw_plan_in(bh, page_geo(kBH, 16, 128, L, D), 8, false);
    ne.neighbor_exchange = true;
    EXPECT_ANY_THROW(make_plan(ne));
    ne.force = AgForce{.algo = AgAlgo::Composite};
    EXPECT_ANY_THROW(make_plan(ne));
    // Fewer than 2 devices, or no link: no algorithm runs; one device is a fatal, as composite cannot run either.
    p = raw_plan_in(bh, page_geo(kBH, 2048, 2048, D, D), 1, false);
    EXPECT_EQ(legality(p)[kUc].reason, "needs at least 2 devices");
    EXPECT_EQ(legality(p)[kMc].reason, "needs at least 2 devices");
    EXPECT_ANY_THROW(make_plan(p));
    p = raw_plan_in(bh, page_geo(kBH, 2048, 2048, D, D), 8, false, 0);
    EXPECT_EQ(legality(p)[kUc].reason, "has no fabric link along the gathered axis");
    EXPECT_EQ(make_plan(p).choice.algo, AgAlgo::Composite);
    // Pages that are not whole rows, and an element count past uint32.
    Pages g = page_geo(kBH, 2048, 2048, D, D);
    g.in_pages = 9;
    g.pages_per_stripe_row = 2;
    p = raw_plan_in(bh, g, 8, false);
    EXPECT_FALSE(make_geometry(p, true) || make_geometry(p, false));
    p.pages = page_geo(kBH, 128, 192, D, D);  // E = 1
    p.pages.in_pages = 3'000'000'000;
    EXPECT_FALSE(make_geometry(p, true) || make_geometry(p, false));

    // P above the NOC burst, P below the cut quantum.
    p = raw_plan_in(bh, page_geo(kBH, 2048, 2048, D, D), 8, false);
    p.facts.payload = 20000;
    auto v = legality(p);
    EXPECT_FALSE(v[kUc].ok || v[kMc].ok);
    EXPECT_EQ(v[kUc].reason, "fabric payload 20000 B exceeds the 16384 B NOC burst");
    p.facts.payload = 48;
    v = legality(p);
    EXPECT_EQ(v[kMc].reason, "fabric payload 48 B is below the 64 B cut quantum");
    EXPECT_EQ(make_plan(p).choice.algo, AgAlgo::Composite);

    // The shape rules no native algorithm handles, verbatim, from real specs.
    const auto rm = [](uint32_t h, uint32_t w) {
        return make_spec(
            shape_of({1, 1, h, w}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR,
            nd_sharded(BufferType::L1, shape_of({1, 1, 1, 64})));
    };
    // Ragged row-major width shards (96 = 64 + 32): the spec pads each row to whole shards (128). Gathering the width
    // would gather that padding (the gather-dim padding rule), so it is composite; gathering rows moves whole padded
    // pages natively.
    const PlanIn by_width = plan_in_for(rm(32, 96), rm(32, 768), -1, 8, Fab::Line, bh);
    EXPECT_EQ(
        by_width.shape_reason,
        "gather dim 3 is padded from 96 to 128; size must be a multiple of the tile/shard extent");
    EXPECT_EQ(make_plan(by_width).composite_reason, by_width.shape_reason);
    const PlanIn by_rows = plan_in_for(rm(32, 96), rm(256, 96), -2, 8, Fab::Line, bh);
    EXPECT_EQ(by_rows.pages.pages_per_stripe_row, 64u);  // 32 rows of 2 pages
    EXPECT_EQ(by_rows.pages.in_pages, 64u);
    EXPECT_NE(make_plan(by_rows).choice.algo, AgAlgo::Composite);
    // Gather-dim padding (a 40-wide tile tensor pads to 64).
    const auto [pad_in, pad_out] = gather_specs({1, 1, 32, 320}, -1, 8);
    EXPECT_EQ(
        gather_dim_padding_reason(pad_in, -1).value_or(""),
        "gather dim 3 is padded from 40 to 64; size must be a multiple of the tile/shard extent");
    EXPECT_EQ(plan_in_for(pad_in, pad_out, -1, 8, Fab::Line, bh).shape_reason, *gather_dim_padding_reason(pad_in, -1));
    // Concat with padded input rows (24 B rows pad to 32 in L1).
    const auto rm_l1 = [](uint32_t h, uint32_t w, uint32_t page_w) {
        return make_spec(
            shape_of({1, 1, h, w}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR,
            nd_sharded(BufferType::L1, shape_of({1, 1, 1, page_w})));
    };
    EXPECT_EQ(
        plan_in_for(rm_l1(4, 12, 12), rm_l1(4, 96, 24), -1, 8, Fab::Line, bh).shape_reason,
        "input rows (24 B) are padded to the 16 B memory alignment; resharding needs unpadded rows");
    // An output shard width that does not divide the gathered row pads it.
    EXPECT_EQ(
        plan_in_for(rm_l1(32, 64, 64), rm_l1(32, 512, 96), -1, 8, Fab::Line, bh)
            .shape_reason.rfind("output stored as Shape([1, 1, 32, 576]) instead of Shape([1, 1, 32, 512])", 0),
        0u);
}

// ---- geometry from real specs vs a brute-force placement model ----
// Every unit's (output page, offset) from the plan's address rule must match where its element really lands.

std::vector<uint32_t> unravel(uint64_t lin, const std::vector<uint32_t>& shape) {
    std::vector<uint32_t> idx(shape.size());
    for (size_t i = shape.size(); i-- > 0;) {
        idx[i] = lin % shape[i];
        lin /= shape[i];
    }
    return idx;
}
uint64_t ravel(const std::vector<uint32_t>& idx, const std::vector<uint32_t>& shape) {
    uint64_t lin = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        lin = (lin * shape[i]) + idx[i];
    }
    return lin;
}

// Page grid of a spec: tile -> [..., H/th, W/tw]; row-major -> [..., W/page_width].
std::vector<uint32_t> page_grid(const TensorSpec& s, uint32_t elem) {
    std::vector<uint32_t> g(s.padded_shape().cbegin(), s.padded_shape().cend());
    if (s.layout() == Layout::TILE) {
        g[g.size() - 2] /= s.tile().get_height();
        g.back() /= s.tile().get_width();
    } else {
        g.back() /= s.compute_page_size_bytes() / elem;
    }
    return g;
}

TEST(AllGatherPlan, GeometryVsPlacementModel) {
    std::mt19937 rng(1);
    auto pick = [&](const auto& v) { return v[std::uniform_int_distribution<size_t>(0, v.size() - 1)(rng)]; };
    auto rnd = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };

    std::map<std::string, int> checked;  // per kind, so a silent skip of a whole kind fails
    for (int trial = 0; trial < 1500; ++trial) {
        const ArchVals a = rnd(0, 1) ? kBH : kWH;
        const AgFacts facts = make_facts(a);
        const uint32_t rank = rnd(1, 5);
        const uint32_t N = pick(std::vector<uint32_t>{2, 3, 4, 8});
        const int32_t dim = -static_cast<int32_t>(rnd(1, rank));
        const bool tiled = rnd(0, 1);
        std::vector<uint32_t> dims(rank);
        for (auto& d : dims) {
            d = rnd(1, 3);
        }
        const BufferType in_bt = rnd(0, 1) ? BufferType::DRAM : BufferType::L1;
        const BufferType out_bt = rnd(0, 1) ? BufferType::DRAM : BufferType::L1;

        std::optional<TensorSpec> in, out;
        uint32_t elem = 0;  // row-major only
        std::string kind;
        if (tiled) {
            const auto th_tw = pick(std::vector<std::array<uint32_t, 2>>{{32, 32}, {16, 32}, {32, 16}});
            const Tile tile(th_tw);
            const DataType dt = rnd(0, 1) ? DataType::BFLOAT16 : DataType::BFLOAT8_B;
            dims.back() = th_tw[1] * rnd(1, 4);
            if (rank >= 2) {
                dims[rank - 2] = th_tw[0] * rnd(1, 3);
            }
            auto out_dims = dims;
            out_dims[rank + dim] *= N;
            in = make_spec(shape_of(dims), dt, Layout::TILE, interleaved(in_bt), tile);
            out = make_spec(shape_of(out_dims), dt, Layout::TILE, interleaved(out_bt), tile);
            kind = "tile";
        } else {
            elem = pick(std::vector<uint32_t>{1, 2, 4});
            const DataType dt = elem == 1 ? DataType::UINT8 : (elem == 2 ? DataType::BFLOAT16 : DataType::FLOAT32);
            const uint32_t W = pick(std::vector<uint32_t>{8, 16, 32, 48, 64, 96, 128});
            dims.back() = W;
            auto out_dims = dims;
            out_dims[rank + dim] *= N;
            const uint32_t outW = out_dims.back();
            std::vector<uint32_t> in_cands;
            for (uint32_t x : {W, W / 2, W / 4, 8u, 16u, 32u}) {
                if (x > 0 && W % x == 0) {
                    in_cands.push_back(x);
                }
            }
            const uint32_t in_pw = pick(in_cands);
            std::vector<uint32_t> out_cands;
            for (uint32_t x : {in_pw, in_pw * 2, in_pw * 4, in_pw / 2, std::max(1u, in_pw / 4), outW}) {
                if (x > 0 && outW % x == 0 && std::max(x, in_pw) % std::min(x, in_pw) == 0) {
                    out_cands.push_back(x);
                }
            }
            const uint32_t out_pw = pick(out_cands);
            // A page narrower than the row needs sharding; a one-row ND shard of that width does it.
            auto mc = [&](BufferType bt, uint32_t pw, uint32_t row) {
                if (pw == row) {
                    return interleaved(bt);
                }
                std::vector<uint32_t> shard(rank, 1);
                shard.back() = pw;
                return nd_sharded(bt, shape_of(shard));
            };
            try {
                in = make_spec(shape_of(dims), dt, Layout::ROW_MAJOR, mc(in_bt, in_pw, W));
                out = make_spec(shape_of(out_dims), dt, Layout::ROW_MAJOR, mc(out_bt, out_pw, outW));
            } catch (const std::exception&) {
                continue;  // a layout TensorSpec itself rejects
            }
            kind = in_pw == out_pw ? "rm matched" : (in_pw < out_pw ? "rm concat" : "rm split");
        }

        const PlanIn p = plan_in_for(*in, *out, dim, N, Fab::Line, facts);
        if (!p.shape_reason.empty()) {
            continue;  // a shape rule no native algorithm handles
        }
        const auto in_grid = page_grid(*in, elem);
        const auto out_grid = page_grid(*out, elem);
        const uint64_t in_pages = std::accumulate(in_grid.begin(), in_grid.end(), 1ull, std::multiplies<>());
        ASSERT_EQ(p.pages.in_pages, in_pages) << trial;
        const uint32_t ci = p.pages.ci;
        const uint32_t g_dim = in_grid.size() + dim;  // gather dim in the page grid (padded rank may exceed rank)

        for (bool unicast : {true, false}) {
            const auto geo = make_geometry(p, unicast);
            if (!geo) {
                continue;
            }
            const uint32_t A = geo->A;
            ASSERT_EQ(uint64_t(geo->R) * geo->S, in_pages * geo->Ui) << trial;
            ASSERT_EQ(uint64_t(geo->elements) * geo->E, uint64_t(geo->R) * geo->S) << trial;
            const uint64_t units = in_pages * geo->Ui;
            for (uint32_t d = 0; d < N; ++d) {
                for (uint64_t u = 0; u < units; ++u) {
                    const uint64_t page = u / geo->Ui;
                    const uint32_t off = (u % geo->Ui) * A;
                    if (off >= ci) {
                        continue;  // whole-page slot tail: lands in the destination page's padding
                    }
                    // Plan's address rule.
                    const uint64_t g = ((u / geo->S) * N * geo->S) + (uint64_t(d) * geo->S) + (u % geo->S);
                    const uint64_t got_page = g / geo->Uo;
                    const uint32_t got_off = (g % geo->Uo) * A;
                    // Ground truth.
                    auto pc = unravel(page, in_grid);
                    uint64_t want_page = 0;
                    uint32_t want_off = 0;
                    if (tiled) {
                        pc[g_dim] += d * in_grid[g_dim];
                        want_page = ravel(pc, out_grid);
                        want_off = off;
                    } else {
                        const uint32_t in_pw = ci / elem;
                        const uint32_t out_pw = p.pages.co / elem;
                        // Element coordinate: the row's dims, then the column.
                        std::vector<uint32_t> ec(pc.begin(), pc.end());
                        ec.back() = (pc.back() * in_pw) + (off / elem);
                        const uint32_t W = in->padded_shape()[-1];
                        if (g_dim == in_grid.size() - 1) {
                            ec.back() += d * W;
                        } else {
                            ec[g_dim] += d * in_grid[g_dim];
                        }
                        auto oc = ec;
                        oc.back() = ec.back() / out_pw;
                        want_page = ravel(oc, out_grid);
                        want_off = (ec.back() % out_pw) * elem;
                    }
                    ASSERT_EQ(got_page, want_page) << kind << " trial " << trial << " dev " << d << " unit " << u;
                    ASSERT_EQ(got_off, want_off) << kind << " trial " << trial << " dev " << d << " unit " << u;
                }
            }
            checked[kind + (whole_page(p.pages, *geo) ? " whole" : " sub")]++;
        }
    }
    for (const char* kind :
         {"tile sub", "tile whole", "rm matched sub", "rm matched whole", "rm concat sub", "rm split sub"}) {
        EXPECT_GT(checked[kind], 10) << kind;
    }
}

// ---- slicer (empty slices allowed: more workers than elements) ----

TEST(AllGatherPlan, Slicer) {
    for (uint32_t elements = 0; elements <= 300; ++elements) {
        for (uint32_t count = 1; count <= 12; ++count) {
            const auto slices = make_slices(elements, count);
            ASSERT_EQ(slices.size(), count);
            uint32_t next = 0, lo = elements, hi = 0;
            for (const auto& s : slices) {
                ASSERT_EQ(s.e0, next);
                next += s.ne;
                lo = std::min(lo, s.ne);
                hi = std::max(hi, s.ne);
            }
            ASSERT_EQ(next, elements);
            ASSERT_LE(hi - lo, 1u);
        }
    }
}

// ---- multicast route sets. Each variant's routes cover every other device of the collective exactly once ----

// Devices one connection reaches, decoded from its range bytes (identity physical slots).
std::vector<std::array<int, 2>> receivers(
    const McastConn& c,
    uint32_t range,
    std::array<int, 2> me,
    std::array<uint32_t, 2> size,
    std::array<bool, 2> wraps) {
    const int e = range & 0xFF, w = (range >> 8) & 0xFF, n = (range >> 16) & 0xFF, s = (range >> 24) & 0xFF;
    std::vector<std::array<int, 2>> out;
    auto add = [&](int dr, int dc) {
        std::array<int, 2> p{me[0] + dr, me[1] + dc};
        for (int a = 0; a < 2; ++a) {
            const int sz = static_cast<int>(size[a]);
            if (wraps[a]) {
                p[a] = ((p[a] % sz) + sz) % sz;
            }
            EXPECT_TRUE(p[a] >= 0 && p[a] < sz) << "route leaves a line";
        }
        out.push_back(p);
    };
    if (c.dir == kEast || c.dir == kWest) {
        EXPECT_EQ(n + s, 0);
        EXPECT_EQ(c.dir == kEast ? w : e, 0);
        const int hops = c.dir == kEast ? e : w;
        for (int k = 1; k <= hops; ++k) {
            add(0, c.dir == kEast ? k : -k);
        }
    } else {
        EXPECT_EQ(c.dir == kSouth ? n : s, 0);
        const int hops = c.dir == kSouth ? s : n;
        const int sign = c.dir == kSouth ? 1 : -1;
        for (int k = 1; k <= hops; ++k) {
            add(sign * k, 0);
            for (int j = 1; j <= e; ++j) {
                add(sign * k, j);
            }
            for (int j = 1; j <= w; ++j) {
                add(sign * k, -j);
            }
        }
    }
    return out;
}

void check_mcast(std::array<uint32_t, 2> mesh, std::optional<uint32_t> axis, std::array<bool, 2> ring) {
    std::array<bool, 2> active{}, wraps{};
    uint32_t N = 1;
    bool expect_alt = false;
    for (uint32_t a = 0; a < 2; ++a) {
        ring[a] = ring[a] && mesh[a] > 2;  // as the factory passes it: a 2-device ring is a line
        active[a] = mesh[a] > 1 && axis.value_or(a) == a;
        wraps[a] = active[a] && ring[a];
        expect_alt = expect_alt || (wraps[a] && mesh[a] % 2 == 0);
        N *= active[a] ? mesh[a] : 1;
    }
    const auto label =
        fmt::format("mesh {}x{} axis {} ring {}/{}", mesh[0], mesh[1], axis.value_or(9), ring[0], ring[1]);
    for (uint32_t r = 0; r < mesh[0]; ++r) {
        for (uint32_t c = 0; c < mesh[1]; ++c) {
            const auto routes = mcast_routes(N, mesh, {r, c}, axis, ring);
            ASSERT_EQ(routes.alternates, expect_alt) << label;
            ASSERT_LE(routes.conns.size(), (active[0] && active[1]) ? 4u : 2u) << label;
            for (int v = 0; v < 2; ++v) {
                std::map<std::array<int, 2>, int> hits;
                for (const auto& conn : routes.conns) {
                    const uint32_t range = v == 0 ? conn.range_v0 : conn.range_v1;
                    EXPECT_NE(range, 0u) << label;  // every (connection, variant) carries a done inc: V * (N - 1)
                    if (!routes.alternates) {
                        EXPECT_EQ(conn.range_v1, conn.range_v0) << label;
                    }
                    for (const auto& p : receivers(conn, range, {int(r), int(c)}, mesh, wraps)) {
                        hits[p]++;
                    }
                }
                ASSERT_EQ(hits.size(), N - 1) << label << " dev " << r << "," << c << " v" << v;
                for (const auto& [p, n] : hits) {
                    ASSERT_EQ(n, 1) << label;
                    ASSERT_TRUE(p[0] != int(r) || p[1] != int(c)) << label;
                    for (uint32_t a = 0; a < 2; ++a) {
                        if (!active[a]) {
                            ASSERT_EQ(p[a], int(a == 0 ? r : c)) << label << ": reached an inactive axis";
                        }
                    }
                }
            }
            // Physical slots follow phys_dir; only the byte positions move.
            const std::array<uint32_t, 4> swap{1, 0, 3, 2};
            const auto swapped = mcast_routes(N, mesh, {r, c}, axis, ring, swap);
            ASSERT_EQ(swapped.conns.size(), routes.conns.size());
            for (size_t i = 0; i < routes.conns.size(); ++i) {
                auto permute = [&](uint32_t x) {
                    uint32_t y = 0;
                    for (uint32_t k = 0; k < 4; ++k) {
                        y |= ((x >> (8 * k)) & 0xFF) << (8 * swap[k]);
                    }
                    return y;
                };
                EXPECT_EQ(swapped.conns[i].dir, routes.conns[i].dir);
                EXPECT_EQ(swapped.conns[i].range_v0, permute(routes.conns[i].range_v0));
                EXPECT_EQ(swapped.conns[i].range_v1, permute(routes.conns[i].range_v1));
            }
        }
    }
}

TEST(AllGatherPlan, MulticastRoutes) {
    for (uint32_t n = 2; n <= 16; ++n) {
        for (bool ring : {false, true}) {
            check_mcast({1, n}, std::nullopt, {false, ring});
            check_mcast({n, 1}, std::nullopt, {ring, false});
            check_mcast({4, n}, 1u, {false, ring});  // 1D collective on one axis of a 2D mesh
            check_mcast({n, 4}, 0u, {ring, true});
        }
    }
    for (std::array<uint32_t, 2> mesh : {std::array<uint32_t, 2>{2, 4}, {4, 4}, {3, 5}, {4, 8}, {2, 2}}) {
        for (bool r0 : {false, true}) {
            for (bool r1 : {false, true}) {
                check_mcast(mesh, std::nullopt, {r0, r1});
            }
        }
    }
}

// ---- schedules for synthetic plans ----

AgPlan unicast_plan(uint32_t N, bool ring, uint32_t elements, uint32_t links, uint32_t n, uint32_t E, uint32_t k = 2) {
    AgPlan p;
    p.choice = AgChoice{.algo = AgAlgo::Unicast, .n = uint8_t(n), .links = uint8_t(links), .k = uint8_t(k)};
    p.N = N;
    p.ring = ring && N > 2;  // what make_plan_in does
    p.geo.A = 64;
    p.geo.E = E;
    p.geo.P_A = 64 * 5;  // a packet holds a few elements, so iterations span several slots
    p.geo.fill = p.geo.P_A;
    p.slices = make_slices(elements, links * n);
    return p;
}

AgPlan mcast_plan(uint32_t N, uint32_t elements, uint32_t links, uint32_t variants) {
    AgPlan p;
    p.choice = AgChoice{.algo = AgAlgo::Multicast, .n = 1, .links = uint8_t(links), .k = 2};
    p.N = N;
    p.geo.A = 64;
    p.geo.E = 1;
    p.geo.P_A = 64 * 5;
    p.geo.fill = p.geo.P_A;
    p.variants = variants;
    p.slices = make_slices(elements, links);
    return p;
}

// ---- physically equal inputs give identical plans ----

TEST(AllGatherPlan, PhysicallyEqualInputs) {
    const AgFacts bh = make_facts(kBH);
    auto plan_of = [&](const std::vector<uint32_t>& out_shape, int32_t dim, DataType dt, Layout layout, Fab fab) {
        const auto [in, out] = gather_specs(out_shape, dim, 8, dt, layout);
        return make_plan(plan_in_for(in, out, dim, 8, fab, bh));
    };
    for (const Fab fab : {Fab::Line, Fab::Ring}) {
        // Tiles: dtype (2-byte types) and rank differ, pages and page grid agree.
        const AgPlan tile = plan_of({1, 1, 256, 8192}, -1, DataType::BFLOAT16, Layout::TILE, fab);
        EXPECT_NE(tile.choice.algo, AgAlgo::Composite);
        EXPECT_EQ(plan_of({256, 8192}, -1, DataType::UINT16, Layout::TILE, fab), tile);
        EXPECT_EQ(plan_of({1, 1, 1, 256, 8192}, -1, DataType::BFLOAT16, Layout::TILE, fab), tile);
        // Row-major 8 KB rows: bf16 x 4096, fp32 x 2048, uint8 x 8192, at any rank, gathered on rows.
        const AgPlan rm = plan_of({1, 1, 1024, 4096}, -2, DataType::BFLOAT16, Layout::ROW_MAJOR, fab);
        EXPECT_NE(rm.choice.algo, AgAlgo::Composite);
        EXPECT_EQ(plan_of({1024, 2048}, -2, DataType::FLOAT32, Layout::ROW_MAJOR, fab), rm);
        EXPECT_EQ(plan_of({1, 1024, 8192}, -2, DataType::UINT8, Layout::ROW_MAJOR, fab), rm);
        // Gathering an outer dim of a [R, row] block moves the same bytes as gathering the rows.
        EXPECT_EQ(plan_of({8, 128, 4096}, -3, DataType::BFLOAT16, Layout::ROW_MAJOR, fab), rm);
    }
}

// ---- a mux per active direction, channels == n, worker w on channel w; placement shape ----

TEST(AllGatherPlan, MuxPerActiveDirection) {
    const AgFacts bh = make_facts(kBH);
    for (const Fab fab : {Fab::Line, Fab::Ring}) {
        for (const uint32_t N : {2u, 3u, 4u, 8u}) {
            for (uint32_t links = 1; links <= 2; ++links) {
                for (uint32_t n = 1; n <= kAgMaxWorkers; ++n) {
                    const auto [in, out] = gather_specs({1, 1, 256, 1024 * N}, -1, N);
                    const AgForce force{.algo = AgAlgo::Unicast, .workers = uint8_t(n), .links = uint8_t(links)};
                    const AgPlan plan = make_plan(plan_in_for(in, out, -1, N, fab, bh, 2, force));
                    const auto label = fmt::format("N {} links {} n {} ring {}", N, links, n, plan.ring);
                    ASSERT_EQ(plan.choice.algo, AgAlgo::Unicast) << label;
                    ASSERT_EQ(plan.choice.n, n);
                    ASSERT_EQ(plan.choice.links, links);
                    ASSERT_EQ(plan.groups.size(), 2 * links) << label;
                    ASSERT_EQ(plan.slices.size(), links * n) << label;
                    std::set<std::pair<size_t, size_t>> used;
                    std::set<size_t> rows;
                    for (size_t g = 0; g < plan.groups.size(); ++g) {
                        const Group& grp = plan.groups[g];
                        ASSERT_EQ(grp.link, g / 2);
                        ASSERT_EQ(grp.dir, g % 2);
                        // Channels and connections both come from workers.size(): worker w is channel w.
                        ASSERT_EQ(grp.workers.size(), n) << label;
                        ASSERT_TRUE(used.insert({grp.mux.x, grp.mux.y}).second) << label;
                        for (const auto& c : grp.workers) {
                            ASSERT_TRUE(used.insert({c.x, c.y}).second) << label;
                            ASSERT_EQ(c.y, grp.mux.y) << label << ": a worker off its mux's row";
                        }
                        ASSERT_TRUE(rows.insert(grp.mux.y).second) << label << ": groups share a row";
                    }
                    for (uint32_t dev = 0; dev < N; ++dev) {
                        for (uint32_t dir = 0; dir < 2; ++dir) {
                            // A mux exists iff the direction has a neighbor; idle-direction workers still receive.
                            const bool mux = dir_active(plan, dev, dir);
                            EXPECT_EQ(mux, plan.ring || (dir == kFwd ? dev + 1 < N : dev > 0)) << label;
                            for (uint32_t l = 0; l < links; ++l) {
                                for (uint32_t w = 0; w < n; ++w) {
                                    const auto ws = schedule(plan, dev, l, dir, w);
                                    EXPECT_EQ(ws.init_send, mux) << label;
                                    EXPECT_EQ(ws.iters > 0, mux) << label;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---- sync arithmetic and coverage, per device and per worker ----

// The units each data_valid signal carries, in order, for an ideal walk (every slot full). The real walk carries no
// more per slot, so the checks below hold for it too.
std::vector<uint32_t> writer_signals(const AgPlan& plan, const WorkerSched& ws) {
    const uint32_t cap = ag::max_slot_units(plan.choice.k, plan.geo.A, plan.geo.P_A);
    ag::SignalCadence cadence(ws.cadence_slots);
    std::vector<uint32_t> signals;
    for (uint32_t i = 0; i < ws.iters; ++i) {
        uint32_t left = (i + 1 == ws.iters ? ws.last_take : ws.ne) * plan.geo.E;
        while (left > 0) {
            const uint32_t units = std::min(cap, left);
            left -= units;
            const uint32_t s =
                cadence.on_slot(ag::SlotInfo{.packets = 1, .iter_end = uint16_t(left == 0), .units = units});
            if (s != 0) {
                signals.push_back(s);
            }
        }
    }
    return signals;
}

// Simulates the relay chain of every (link, worker): every device receives every other stripe's elements exactly
// once, each relay forwards only what upstream sent in its previous iteration, init credits pair up, data_valid
// signals sum to what downstream waits for, and every relay wait target is reached.
void check_unicast_sync(const AgPlan& plan, const std::string& label) {
    const uint32_t N = plan.N;
    const uint32_t E = plan.geo.E;
    const uint32_t cap = ag::max_slot_units(plan.choice.k, plan.geo.A, plan.geo.P_A);
    // init credits per (dev, link, dir, w)
    std::map<std::array<uint32_t, 4>, uint32_t> init_got;
    uint64_t init_sends = 0, init_waits = 0;
    for (uint32_t l = 0; l < plan.choice.links; ++l) {
        for (uint32_t w = 0; w < plan.choice.n; ++w) {
            const Slice slice = plan.slices[(l * plan.choice.n) + w];
            const uint32_t ne = slice.ne;
            // got[t][s][e]: times device t received element e of stripe s
            std::vector<std::vector<std::vector<int>>> got(
                N, std::vector<std::vector<int>>(N, std::vector<int>(ne, 0)));
            for (uint32_t dev = 0; dev < N; ++dev) {
                const auto f = schedule(plan, dev, l, kFwd, w);
                const auto b = schedule(plan, dev, l, kBwd, w);
                ASSERT_EQ(int(f.local_copy) + int(b.local_copy), 1) << label << " dev " << dev;
                ASSERT_GT(f.iters + b.iters, 0u) << label;
                for (uint32_t dir = 0; dir < 2; ++dir) {
                    const auto ws = dir == kFwd ? f : b;
                    ASSERT_EQ(ws.e0, slice.e0);
                    ASSERT_EQ(ws.ne, ne);
                    ASSERT_EQ(ws.stripe0, dev);
                    ASSERT_EQ(ws.stripe_step, dir == kFwd ? N - 1 : 1) << label;
                    ASSERT_EQ(ws.iter_units, ne * E);
                    ASSERT_LE(ws.last_skip + ws.last_take, ne);
                    ASSERT_EQ(ws.iters > 0, dir_active(plan, dev, dir)) << label;
                    ASSERT_GE(ws.cadence_slots, 1u) << label;
                    if (ws.local_copy) {
                        // The copy comes from iteration 0, so that iteration must run and be whole.
                        ASSERT_TRUE(ws.iters > 1 || (ws.iters == 1 && ws.last_skip == 0 && ws.last_take == ne))
                            << label << " dev " << dev;
                    }
                    init_waits += ws.init_wait;
                    if (ws.init_send) {
                        ++init_sends;
                        const uint32_t nbr = dir == kFwd ? (dev + 1) % N : (dev + N - 1) % N;
                        init_got[{nbr, l, 1 - dir, w}]++;
                    }
                    // Signals end on iteration ends and sum to what downstream waits for.
                    const auto sig = writer_signals(plan, ws);
                    std::set<uint64_t> cumulative;
                    uint64_t sum = 0;
                    for (const uint32_t x : sig) {
                        sum += x;
                        cumulative.insert(sum);
                    }
                    uint64_t iter_end = 0;
                    for (uint32_t i = 0; i < ws.iters; ++i) {
                        iter_end += uint64_t{i + 1 == ws.iters ? ws.last_take : ws.ne} * E;
                        if (iter_end > 0) {
                            ASSERT_TRUE(cumulative.count(iter_end))
                                << label << ": no signal at the end of iteration " << i;
                        }
                    }
                    const uint32_t down = dir == kFwd ? (dev + 1) % N : (dev + N - 1) % N;
                    const auto dn = schedule(plan, down, l, dir, w);
                    const bool has_down = dir_active(plan, dev, dir);
                    ASSERT_EQ(dn.end_wait, has_down ? sum : 0) << label << " dev " << dev << " dir " << dir;
                    for (uint32_t i = 0; i < ws.iters; ++i) {
                        const bool last = i + 1 == ws.iters;
                        const uint32_t skip = last ? ws.last_skip : 0;
                        const uint32_t take = last ? ws.last_take : ne;
                        const uint32_t stripe = (dev + i * ws.stripe_step) % N;
                        for (uint32_t e = skip; e < skip + take; ++e) {
                            got[down][stripe][e]++;
                        }
                    }
                    if (!has_down) {
                        continue;
                    }
                    // The downstream relay forwards what we sent in its previous iteration, which is whole. Each of its
                    // slot waits covers the units the slot reads and is reached by what we signal.
                    for (uint32_t i = 1; i < dn.iters; ++i) {
                        ASSERT_TRUE(i - 1 < ws.iters) << label;
                        ASSERT_TRUE(i < ws.iters || (ws.last_skip == 0 && ws.last_take == ne)) << label;
                        ASSERT_EQ((down + i * dn.stripe_step) % N, (dev + (i - 1) * ws.stripe_step) % N) << label;
                        const bool last = i + 1 == dn.iters;
                        const uint32_t skip = (last ? dn.last_skip : 0) * E;
                        const uint32_t take = (last ? dn.last_take : dn.ne) * E;
                        const uint64_t base = uint64_t{i - 1} * dn.iter_units;
                        for (uint32_t c = 0; c < take;) {
                            const uint32_t r = std::min(cap, take - c);
                            const uint64_t target = base + std::min(skip + c + cap, dn.iter_units);
                            ASSERT_GE(target, base + skip + c + r) << label;
                            ASSERT_LE(target, sum) << label << ": wait target beyond upstream's final count";
                            c += r;
                        }
                    }
                }
            }
            for (uint32_t t = 0; t < N; ++t) {
                for (uint32_t st = 0; st < N; ++st) {
                    for (uint32_t e = 0; e < ne; ++e) {
                        ASSERT_EQ(got[t][st][e], st == t ? 0 : 1)
                            << label << " dev " << t << " stripe " << st << " element " << e;
                    }
                }
            }
            // Even ring: device t's antipode stripe arrives in two halves, split by walk position: forward from t - 1
            // and backward from t + 1, in their last iterations.
            if (plan.ring && N % 2 == 0) {
                for (uint32_t t = 0; t < N; ++t) {
                    const uint32_t from_f = (t + N - 1) % N;
                    const uint32_t from_b = (t + 1) % N;
                    const auto f = schedule(plan, from_f, l, kFwd, w);
                    const auto b = schedule(plan, from_b, l, kBwd, w);
                    ASSERT_EQ(
                        (from_f + (f.iters - 1) * f.stripe_step) % N, (from_b + (b.iters - 1) * b.stripe_step) % N);
                    ASSERT_EQ((from_f + (f.iters - 1) * f.stripe_step) % N, (t + N / 2) % N);
                    ASSERT_EQ(f.last_skip, 0u);
                    ASSERT_EQ(f.last_take, b.last_skip) << label;
                    ASSERT_EQ(b.last_skip + b.last_take, f.ne) << label;
                }
            }
        }
    }
    ASSERT_EQ(init_sends, init_waits) << label;
    for (uint32_t l = 0; l < plan.choice.links; ++l) {
        for (uint32_t w = 0; w < plan.choice.n; ++w) {
            for (uint32_t dir = 0; dir < 2; ++dir) {
                for (uint32_t dev = 0; dev < N; ++dev) {
                    const auto ws = schedule(plan, dev, l, dir, w);
                    const auto it = init_got.find({dev, l, dir, w});
                    ASSERT_EQ(it == init_got.end() ? 0 : it->second, ws.init_wait)
                        << label << " dev " << dev << " dir " << dir << " w " << w;
                }
            }
        }
    }
}

void check_mcast_sync(std::array<uint32_t, 2> mesh, std::array<bool, 2> ring, uint32_t links) {
    std::array<bool, 2> wraps{};
    uint32_t N = 1;
    for (uint32_t a = 0; a < 2; ++a) {
        ring[a] = ring[a] && mesh[a] > 2;
        wraps[a] = mesh[a] > 1 && ring[a];
        N *= mesh[a];
    }
    const auto label = fmt::format("mesh {}x{} ring {}/{}", mesh[0], mesh[1], ring[0], ring[1]);
    const bool alt = mcast_routes(N, mesh, {0, 0}, std::nullopt, ring).alternates;
    const AgPlan plan = mcast_plan(N, 5, links, alt ? 2 : 1);
    for (uint32_t l = 0; l < links; ++l) {
        // Init and done ride every (connection, variant) of every device's core on link l to the same core elsewhere.
        std::map<std::array<int, 2>, uint32_t> got;
        uint64_t sends = 0, waits = 0;
        for (uint32_t r = 0; r < mesh[0]; ++r) {
            for (uint32_t c = 0; c < mesh[1]; ++c) {
                const uint32_t dev = (r * mesh[1]) + c;
                const auto ws = schedule(plan, dev, l, 0, 0);
                ASSERT_TRUE(ws.init_send && ws.local_copy && ws.iters == 1) << label;
                ASSERT_EQ(ws.init_wait, ws.end_wait) << label;
                waits += ws.end_wait;
                const auto routes = mcast_routes(N, mesh, {r, c}, std::nullopt, ring);
                for (uint32_t v = 0; v < plan.variants; ++v) {
                    for (const auto& conn : routes.conns) {
                        for (const auto& p :
                             receivers(conn, v == 0 ? conn.range_v0 : conn.range_v1, {int(r), int(c)}, mesh, wraps)) {
                            got[p]++;
                            ++sends;
                        }
                    }
                }
            }
        }
        ASSERT_EQ(sends, waits) << label;
        for (uint32_t r = 0; r < mesh[0]; ++r) {
            for (uint32_t c = 0; c < mesh[1]; ++c) {
                const auto ws = schedule(plan, (r * mesh[1]) + c, l, 0, 0);
                const std::array<int, 2> me{int(r), int(c)};
                ASSERT_EQ(got[me], ws.end_wait) << label << " dev " << r << "," << c;
            }
        }
    }
}

TEST(AllGatherPlan, SyncArithmetic) {
    for (uint32_t N = 2; N <= 9; ++N) {
        for (bool ring : {false, true}) {
            for (uint32_t elements : {0u, 1u, 2u, 5u, 13u, 40u}) {
                for (uint32_t links = 1; links <= 2; ++links) {
                    for (uint32_t n = 1; n <= kAgMaxWorkers; ++n) {
                        for (uint32_t k = 1; k <= 3; ++k) {
                            AgPlan plan = unicast_plan(N, ring, elements, links, n, /*E=*/k + 1, k);
                            // cadence_slots comes from fill; vary it so cadence and iteration ends interleave
                            plan.geo.fill = plan.geo.A * (1 + (elements % 3));
                            check_unicast_sync(
                                plan,
                                fmt::format(
                                    "N {} ring {} elements {} links {} n {} k {}", N, ring, elements, links, n, k));
                        }
                    }
                }
            }
        }
    }
    for (uint32_t n = 2; n <= 8; ++n) {
        for (bool ring : {false, true}) {
            check_mcast_sync({1, n}, {false, ring}, 2);
        }
    }
    for (std::array<uint32_t, 2> mesh : {std::array<uint32_t, 2>{2, 4}, {4, 4}, {3, 5}}) {
        for (bool r0 : {false, true}) {
            for (bool r1 : {false, true}) {
                check_mcast_sync(mesh, {r0, r1}, 1);
            }
        }
    }

    // One element on a 4-device ring: the forward half of the antipode stripe is empty, and two of three workers per
    // link-direction have empty slices.
    const AgPlan plan = unicast_plan(4, true, 1, 1, 3, 1);
    EXPECT_EQ(schedule(plan, 0, 0, kFwd, 0).last_take, 0u);
    EXPECT_EQ(schedule(plan, 0, 0, kBwd, 0).last_take, 1u);
    EXPECT_EQ(schedule(plan, 0, 0, kFwd, 2).ne, 0u);
}

// ---- shrink order on synthetic worker cores: n 3 -> 2 -> 1, then links, then multicast, then composite ----

CoreRangeSet rows_of(uint32_t rows, uint32_t width) {
    std::vector<CoreRange> rs;
    for (uint32_t y = 0; y < rows; ++y) {
        rs.emplace_back(
            CoreCoord{0, 2 * y}, CoreCoord{width - 1, 2 * y});  // every other row: `rows` rows of `width` cores
    }
    return CoreRangeSet(rs);
}

TEST(AllGatherPlan, ShrinkOrder) {
    const auto [in, out] = gather_specs({1, 1, 4096, 8192}, -1, 8);  // 64 MB: the model wants its most workers
    auto plan_with = [&](const CoreRangeSet& cores, AgForce force = {}) {
        return make_plan(plan_in_for(in, out, -1, 8, Fab::Line, make_facts(kBH, std::nullopt, cores), 2, force));
    };
    const AgForce uc{.algo = AgAlgo::Unicast};
    const PlanIn ref = plan_in_for(in, out, -1, 8, Fab::Line, make_facts(kBH), 2, uc);
    const uint32_t n_want = model_workers(busiest_link_packets(*make_geometry(ref, true), 8, false, 2));
    ASSERT_GE(n_want, 2u);
    struct Case {
        CoreRangeSet cores;
        uint32_t n, links;
    };
    const std::vector<Case> cases = {
        {rows_of(4, n_want + 1), n_want, 2},        // a row per group
        {rows_of(2, 2 * (n_want + 1)), n_want, 2},  // rows doubled up
        {rows_of(4, n_want), n_want - 1, 2},        // one core short per row: n shrinks first
        {rows_of(4, 2), 1, 2},
        {rows_of(2, 4), 1, 2},  // (3, 1) would also fit: n shrinks before links
        {rows_of(2, 2), 1, 1},
        {rows_of(1, 4), 1, 1},  // both directions doubled up on one row
    };
    for (const auto& c : cases) {
        const AgPlan plan = plan_with(c.cores, uc);
        const auto label = fmt::format("{} cores", c.cores.num_cores());
        EXPECT_EQ(plan.choice.algo, AgAlgo::Unicast) << label;
        EXPECT_EQ(plan.choice.n, c.n) << label;
        EXPECT_EQ(plan.choice.links, c.links) << label;
    }
    // Unicast cannot place a mux and a worker per direction (a lone core is left for the second): multicast, then
    // composite.
    EXPECT_FALSE(
        legality(plan_in_for(in, out, -1, 8, Fab::Line, make_facts(kBH, std::nullopt, rows_of(1, 3))))[kUc].ok);
    AgPlan plan = plan_with(rows_of(1, 3));
    EXPECT_EQ(plan.choice.algo, AgAlgo::Multicast);
    EXPECT_EQ(plan.choice.links, 2u);
    plan = plan_with(rows_of(1, 1));
    EXPECT_EQ(plan.choice.algo, AgAlgo::Multicast);
    EXPECT_EQ(plan.choice.links, 1u);
    plan = plan_with(CoreRangeSet());
    EXPECT_EQ(plan.choice.algo, AgAlgo::Composite);
    EXPECT_EQ(
        plan.composite_reason,
        "unicast needs a mux and a worker on one core row for each direction; multicast has no free worker cores");
    // A forced value that does not fit is a TT_FATAL, never a silent shrink.
    EXPECT_ANY_THROW(plan_with(rows_of(4, 2), AgForce{.algo = AgAlgo::Unicast, .workers = 2}));
    EXPECT_ANY_THROW(plan_with(rows_of(1, 1), uc));
    EXPECT_ANY_THROW(plan_with(grid(13, 10), AgForce{.algo = AgAlgo::Unicast, .links = 3}));
}

// ---- injected facts: WH, BH, unknown arch at three payloads give legal, self-consistent plans ----

void check_plan(const PlanIn& in, const AgPlan& plan, const std::string& label) {
    const auto v = legality(in);
    if (plan.choice.algo == AgAlgo::Composite) {
        ASSERT_FALSE(plan.composite_reason.empty()) << label;
        return;
    }
    ASSERT_EQ(expand(in, plan.choice), plan) << label;
    const bool unicast = plan.choice.algo == AgAlgo::Unicast;
    ASSERT_TRUE(v[unicast ? kUc : kMc].ok) << label;
    ASSERT_GE(plan.choice.k, 1u) << label;
    ASSERT_LE(plan.choice.links, in.links) << label;
    ASSERT_LE(kCbSlots * plan.slot_bytes, in.facts.l1_small_floor - in.facts.l1_base) << label;
    ASSERT_EQ(
        plan.slot_bytes, ag::slot_offsets(plan.choice.k, plan.geo.A, in.facts.header_bytes, plan.geo.P_A).slot_bytes);
    ASSERT_EQ(plan.slices.size(), uint32_t{plan.choice.links} * plan.choice.n) << label;
    ASSERT_EQ(plan.slices.back().e0 + plan.slices.back().ne, plan.geo.elements) << label;
    ASSERT_GE(plan.geo.P_A, plan.geo.A) << label;
    ASSERT_LE(plan.geo.P_A, std::min(in.facts.payload, in.facts.max_burst)) << label;
    ASSERT_GT(plan.geo.fill, 0u) << label;
    if (unicast) {
        ASSERT_LE(in.facts.mux_map_end[plan.choice.n - 1], in.facts.l1_small_floor) << label;
        ASSERT_EQ(plan.groups.size(), 2u * plan.choice.links) << label;
        check_unicast_sync(plan, label);
    } else {
        ASSERT_EQ(plan.mcast_cores.size(), plan.choice.links) << label;
    }
}

TEST(AllGatherPlan, InjectedFacts) {
    const auto dram = interleaved(BufferType::DRAM);
    const auto l1 = interleaved(BufferType::L1);
    struct Shape_ {
        std::vector<uint32_t> out;
        int32_t dim;
        DataType dt;
        Layout layout;
        MemoryConfig in_mc, out_mc;
    };
    const std::vector<Shape_> shapes = {
        {{1, 1, 256, 8192}, -1, DataType::BFLOAT16, Layout::TILE, dram, dram},
        {{1, 1, 32, 256}, -1, DataType::BFLOAT16, Layout::TILE, dram, dram},
        {{1, 1, 4096, 3328}, -1, DataType::BFLOAT16, Layout::TILE, dram, dram},
        {{1, 1, 256, 256}, -1, DataType::BFLOAT8_B, Layout::TILE, dram, l1},
        {{8, 1, 256, 4096}, -4, DataType::BFLOAT16, Layout::ROW_MAJOR, dram, dram},
        {{1, 1, 4096, 24}, -2, DataType::BFLOAT16, Layout::ROW_MAJOR, dram, dram},
        {{1, 1, 2048, 64}, -1, DataType::BFLOAT16, Layout::ROW_MAJOR, l1, l1},
        {{1, 1, 512, 10000}, -2, DataType::BFLOAT16, Layout::ROW_MAJOR, dram, dram},
    };
    for (const ArchVals& a : {kWH, kBH, kUnknown}) {
        for (const uint32_t P : {4352u, 7616u, 15168u}) {
            if (P > a.burst) {
                continue;  // WH cannot configure a 15168 B payload
            }
            const AgFacts facts = make_facts(a, P);
            for (const auto& s : shapes) {
                for (const Fab fab : {Fab::Line, Fab::Ring}) {
                    for (const uint32_t N : {2u, 4u, 8u}) {
                        const auto [in, out] = gather_specs(s.out, s.dim, N, s.dt, s.layout, s.in_mc, s.out_mc);
                        const PlanIn pin = plan_in_for(in, out, s.dim, N, fab, facts);
                        const AgPlan plan = make_plan(pin);
                        const auto label = fmt::format(
                            "arch {} P {} shape {} dim {} N {} {} -> {}",
                            int(a.arch),
                            P,
                            fmt::join(s.out, "x"),
                            s.dim,
                            N,
                            fab == Fab::Ring ? "ring" : "line",
                            algo_str(plan.choice.algo));
                        // Every one of these shapes is native on every arch and payload.
                        EXPECT_NE(plan.choice.algo, AgAlgo::Composite) << label << ": " << plan.composite_reason;
                        check_plan(pin, plan, label);
                    }
                }
            }
        }
    }

    // Random plan inputs: make_plan never throws unforced (except to refuse a neighbor-exchange fabric past 2 devices
    // that no algorithm can serve), and every plan is self-consistent.
    std::mt19937 rng(7);
    auto rnd = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
    for (int trial = 0; trial < 3000; ++trial) {
        const ArchVals& a = rnd(0, 1) ? kBH : kWH;
        AgFacts f = make_facts(a, rnd(0, 1) ? a.P : 4352, rows_of(rnd(0, 6), rnd(1, 8)));
        if (rnd(0, 4) == 0) {
            f.l1_small_floor = f.mux_map_end[rnd(0, 2)];  // only some mux sizes fit
        }
        Pages g = page_geo(a, 16 * rnd(1, 140), 16, rnd(0, 1), rnd(0, 1));
        g.co = rnd(0, 1) ? g.ci : g.ci * rnd(1, 4);
        g.ao = tt::align(g.co, g.ro);
        g.pages_per_stripe_row = rnd(1, 4);
        g.in_pages = g.pages_per_stripe_row * rnd(1, 50);
        g.L = rnd(1, 12);
        g.run_pages = rnd(1, 64);
        PlanIn p = raw_plan_in(f, g, rnd(2, 16), rnd(0, 1), rnd(1, 4));
        p.true_2d = rnd(0, 5) == 0;
        p.neighbor_exchange = !p.true_2d && rnd(0, 4) == 0;
        AgPlan plan;
        try {
            plan = make_plan(p);
        } catch (const std::exception&) {
            ASSERT_TRUE(p.neighbor_exchange && p.N > 2) << "trial " << trial;
            continue;
        }
        check_plan(p, plan, fmt::format("trial {}", trial));
        if (plan.choice.algo == AgAlgo::Unicast) {
            ASSERT_FALSE(p.true_2d);
        }
        if (plan.choice.algo == AgAlgo::Multicast) {
            ASSERT_FALSE(p.neighbor_exchange);
        }
    }
}

// ---- parse_force grammar and strictness ----

TEST(AllGatherPlan, ParseForce) {
    EXPECT_EQ(parse_force("auto").algo, AgAlgo::Auto);
    EXPECT_EQ(parse_force("composite").algo, AgAlgo::Composite);
    const AgForce u = parse_force("unicast:n=2,links=1,rdelay=300");
    EXPECT_EQ(u.algo, AgAlgo::Unicast);
    EXPECT_EQ(u.workers, 2);
    EXPECT_EQ(u.links, 1);
    EXPECT_EQ(u.reader_delay, 300);
    EXPECT_EQ(u.writer_delay, 0);
    const AgForce m = parse_force("multicast:wdelay=65535,links=2");
    EXPECT_EQ(m.algo, AgAlgo::Multicast);
    EXPECT_EQ(m.writer_delay, 65535);
    EXPECT_EQ(m.links, 2);
    EXPECT_EQ(parse_force("auto:rdelay=1000").reader_delay, 1000);
    EXPECT_EQ(parse_force("unicast:n=3").workers, 3);
    const AgForce k = parse_force("unicast:n=2,k=3");
    EXPECT_EQ(k.k, 3);
    EXPECT_EQ(k.workers, 2);
    EXPECT_EQ(parse_force("multicast:k=1").k, 1);
    EXPECT_EQ(parse_force("auto:k=4").k, 4);
    // One representative per rule: range, duplicate, option on composite, unknown algorithm or key, malformed pair,
    // non-decimal value, unicast-only key.
    for (const char* bad : {
             "",
             "Unicast",
             "unicast;n=2",
             "unicast:",
             "unicast:n=",
             "unicast:n=2,",
             "unicast:n=0",
             "unicast:n=4",
             "unicast:k=5",
             "unicast:links=256",
             "unicast:rdelay=65536",
             "unicast:n=2,n=3",
             "unicast:n=+2",
             "unicast:workers=2",
             "multicast:n=1",
             "composite:links=1",
         }) {
        EXPECT_ANY_THROW(parse_force(bad)) << "'" << bad << "'";
    }
}

// ---- footprint and mux map over budget: a smaller n or the next algorithm, never a fatal ----

TEST(AllGatherPlan, FootprintAndMuxMap) {
    const auto [in, out] = gather_specs({1, 1, 4096, 8192}, -1, 8);
    const AgFacts bh = make_facts(kBH);
    const AgForce uc{.algo = AgAlgo::Unicast};
    // Only the 1-channel map fits below L1_SMALL: unicast shrinks to n = 1.
    AgFacts f = bh;
    f.mux_map_end = {f.l1_base + (64 << 10), f.l1_base + (96 << 10), f.l1_base + (128 << 10)};
    f.l1_small_floor = f.mux_map_end[0];
    AgPlan plan = make_plan(plan_in_for(in, out, -1, 8, Fab::Line, f, 2, uc));
    EXPECT_EQ(plan.choice.algo, AgAlgo::Unicast);
    EXPECT_EQ(plan.choice.n, 1u);
    // Up to 2 channels fit: n is whatever the model wants, capped at 2.
    f.l1_small_floor = f.mux_map_end[1];
    const PlanIn two = plan_in_for(in, out, -1, 8, Fab::Line, f, 2, uc);
    const uint32_t n_want = model_workers(busiest_link_packets(*make_geometry(two, true), 8, false, 2));
    EXPECT_EQ(make_plan(two).choice.n, std::min(n_want, 2u));
    // A forced n whose map does not fit is a fatal, not a shrink.
    EXPECT_ANY_THROW(
        make_plan(plan_in_for(in, out, -1, 8, Fab::Line, f, 2, AgForce{.algo = AgAlgo::Unicast, .workers = 3})));
    // No mux fits: unicast is illegal and the plan falls to multicast; forcing unicast is a fatal with the reason.
    f.l1_small_floor = f.mux_map_end[0] - 1;
    const PlanIn no_mux = plan_in_for(in, out, -1, 8, Fab::Line, f);
    EXPECT_FALSE(legality(no_mux)[kUc].ok);
    EXPECT_EQ(make_plan(no_mux).choice.algo, AgAlgo::Multicast);
    EXPECT_ANY_THROW(make_plan(plan_in_for(in, out, -1, 8, Fab::Line, f, 2, uc)));
    // The slot CB footprint shrinks k, then rules both algorithms out: composite.
    const Geometry geo = *make_geometry(plan_in_for(in, out, -1, 8, Fab::Line, bh), true);
    const uint32_t one_slot = ag::slot_offsets(1, geo.A, bh.header_bytes, geo.P_A).slot_bytes;
    f = bh;
    f.l1_small_floor = f.l1_base + (kCbSlots * one_slot);
    for (auto& end : f.mux_map_end) {
        end = std::min(end, f.l1_small_floor);
    }
    plan = make_plan(plan_in_for(in, out, -1, 8, Fab::Line, f));
    EXPECT_NE(plan.choice.algo, AgAlgo::Composite);
    EXPECT_EQ(plan.choice.k, 1u);
    // A forced k is used as is, but more packets per slot than fit in L1 is a fatal.
    EXPECT_ANY_THROW(make_plan(plan_in_for(in, out, -1, 8, Fab::Line, f, 2, AgForce{.k = 2})));
    EXPECT_EQ(make_plan(plan_in_for(in, out, -1, 8, Fab::Line, bh, 2, AgForce{.k = 3})).choice.k, 3u);
    EXPECT_EQ(
        make_plan(plan_in_for(in, out, -1, 8, Fab::Line, bh, 2, AgForce{.algo = AgAlgo::Multicast, .k = 1})).choice.k,
        1u);
    f.l1_small_floor -= 1;
    plan = make_plan(plan_in_for(in, out, -1, 8, Fab::Line, f));
    EXPECT_EQ(plan.choice.algo, AgAlgo::Composite);
    EXPECT_NE(plan.composite_reason.find("slot buffers exceed"), std::string::npos) << plan.composite_reason;
}

// ---- header pool: 2 * V * connections > headers_per_risc makes multicast illegal ----

TEST(AllGatherPlan, HeaderPool) {
    const Pages g = page_geo(kWH, 2048, 2048, D, D);
    // WH even ring: 2 variants x 2 connections x 2 headers = 8, exactly the pool.
    PlanIn p = raw_plan_in(make_facts(kWH), g, 8, true);
    EXPECT_TRUE(legality(p)[kMc].ok);
    // True 2D 2x4, ring on the 4-axis: 3 connections x 2 variants x 2 = 12 > 8 on WH, = 12 on BH.
    p.true_2d = true;
    p.ring = false;
    p.axis_devices = {2, 4};
    p.axis_ring = {false, true};
    const auto wh = legality(p);
    EXPECT_FALSE(wh[kMc].ok);
    EXPECT_EQ(wh[kMc].reason, "needs 12 packet headers per RISC, the pool holds 8");
    EXPECT_EQ(make_plan(p).choice.algo, AgAlgo::Composite);  // unicast needs a 1D axis
    p.facts = make_facts(kBH);
    p.pages = page_geo(kBH, 2048, 2048, D, D);
    EXPECT_TRUE(legality(p)[kMc].ok);
    // 2x4 torus: 4 connections x 2 variants x 2 = 16 > 12.
    p.axis_devices = {4, 4};
    p.axis_ring = {true, true};
    p.N = 16;
    EXPECT_FALSE(legality(p)[kMc].ok);
}

// ---- model: run_bytes within 5% of the walk's mean run ----

// An output accessor for the host walk: contiguity from the host BufferDistributionSpec (sharded) or the bank model
// (interleaved: pages L apart sit back to back in one bank). Addresses do not matter here.
struct HostOut {
    uint32_t L;
    std::optional<tt::tt_metal::BufferDistributionSpec> dist;
    uint32_t contiguous_page_stride() const { return L; }
    uint32_t num_contiguous_pages(uint32_t page, uint32_t end) const {
        return dist ? dist->num_contiguous_pages(page, end) : ((end - page - 1) / L) + 1;
    }
    uint64_t get_noc_addr(uint32_t, uint32_t) const { return 0; }
};

TEST(AllGatherPlan, ModelRunBytes) {
    const AgFacts bh = make_facts(kBH);
    const auto dram = interleaved(BufferType::DRAM);
    const auto ws_out = legacy_sharded(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, 1024, 256, 8);
    struct Case {
        std::vector<uint32_t> out;
        int32_t dim;
        Layout layout;
        MemoryConfig in_mc, out_mc;
        Fab fab;
    };
    const std::vector<Case> cases = {
        {{1, 1, 256, 8192}, -1, Layout::TILE, dram, dram, Fab::Line},   // 4 MB ladder
        {{1, 1, 1024, 8192}, -1, Layout::TILE, dram, dram, Fab::Ring},  // 16 MB
        {{1, 1, 1024, 2560}, -1, Layout::TILE, dram, dram, Fab::Line},
        {{1, 1, 4096, 2560}, -1, Layout::TILE, dram, dram, Fab::Line},
        {{1, 1, 8192, 8192}, -1, Layout::TILE, dram, dram, Fab::Ring},      // 128 MB: 6 slices of 42 rows
        {{1, 1, 800, 16384}, -1, Layout::TILE, dram, dram, Fab::Line},      // 26 MB
        {{1, 1, 13312, 1024}, -1, Layout::TILE, dram, dram, Fab::Ring},     // 27 MB, narrow stripe rows
        {{8, 1, 256, 4096}, -4, Layout::ROW_MAJOR, dram, dram, Fab::Line},  // 8 KB rows
        {{1, 1, 512, 1024}, -2, Layout::ROW_MAJOR, dram, dram, Fab::Line},  // 2 KB rows
        {{1, 1, 1024, 2048}, -1, Layout::TILE, dram, ws_out, Fab::Line},    // width-sharded output
    };
    for (const auto& c : cases) {
        const auto [in, out] = gather_specs(c.out, c.dim, 8, DataType::BFLOAT16, c.layout, c.in_mc, c.out_mc);
        const PlanIn pin = plan_in_for(in, out, c.dim, 8, c.fab, bh);
        ASSERT_TRUE(pin.shape_reason.empty()) << pin.shape_reason;
        const auto label = fmt::format("{} dim {}", fmt::join(c.out, "x"), c.dim);
        for (const bool unicast : {true, false}) {
            const auto geo = make_geometry(pin, unicast);
            ASSERT_TRUE(geo.has_value()) << label;
            const HostOut acc{.L = pin.pages.L, .dist = out.compute_buffer_sharding_args().buffer_distribution_spec()};
            ASSERT_EQ(acc.dist.has_value(), out.memory_config().is_sharded()) << label;
            if (acc.dist) {
                ASSERT_EQ(pin.pages.L, acc.dist->contiguous_page_stride()) << label;
                ASSERT_EQ(pin.pages.run_pages, acc.dist->num_contiguous_pages(0)) << label;
            }
            ag::Walk<HostOut> walk(
                acc,
                ag::WalkGeo{
                    .A = geo->A,
                    .Uo = geo->Uo,
                    .S = geo->S,
                    .N = 8,
                    .E = geo->E,
                    .P_A = geo->P_A,
                    .dst_cross = geo->dst_cross});
            // The model is the steady state, slices of many rows: up to 2 links x 3 workers with at least 16 rows each.
            // A slice of a few rows has shorter runs in its partial first and last rows (measured: 10-15% at 5 rows),
            // which matters only where 4 runs do not fill a packet.
            const uint32_t row_elems = geo->S / geo->E;
            const uint32_t workers = std::clamp(geo->elements / (16 * row_elems), 1u, 6u);
            // Segments as a packet carries them: a run longer than P_A spans several.
            uint64_t bytes = 0, segments = 0;
            for (const auto& s : make_slices(geo->elements, workers)) {
                walk.init(3, s.e0, s.ne, 0, s.ne);
                ag::DstRun r{};
                while (walk.next(r)) {
                    const uint64_t b = uint64_t{r.n} * geo->E * geo->A;
                    bytes += b;
                    segments += (b + geo->P_A - 1) / geo->P_A;
                }
            }
            const double mean = double(bytes) / double(segments);
            // Walk tiles (h columns x lambda lanes) are anchored at the slice's first full row; where they do not
            // divide a row they straddle row ends, and the row clip splits those columns, which the row-lane model
            // does not count (e.g. 10 tiles a row, lambda 8, h 2: 1.11 vs 1.25 pages a run).
            const bool tiles_straddle_rows = !geo->cross_row && row_elems % (geo->h * geo->lambda) != 0;
            EXPECT_NEAR(geo->run_bytes / mean, 1.0, tiles_straddle_rows ? 0.15 : 0.05)
                << label << (unicast ? " unicast" : " multicast") << ": model " << geo->run_bytes << " B, walk " << mean
                << " B";
        }
    }
}

// ---- expansion: the factories' expand(params) reproduces all_gather()'s make_plan ----

TEST(AllGatherPlan, ExpandMatchesMakePlan) {
    const auto dram = interleaved(BufferType::DRAM);
    const auto l1 = interleaved(BufferType::L1);
    struct Case {
        std::vector<uint32_t> out;
        int32_t dim;
        Layout layout;
        MemoryConfig out_mc;
        Fab fab;
        uint32_t N;
        AgForce force;
    };
    const std::vector<Case> cases = {
        {{1, 1, 256, 8192}, -1, Layout::TILE, dram, Fab::Line, 8, {}},
        {{1, 1, 256, 8192}, -1, Layout::TILE, dram, Fab::Ring, 8, {}},
        {{1, 1, 32, 256}, -1, Layout::TILE, l1, Fab::Ring, 4, {}},
        {{1, 1, 4096, 24}, -2, Layout::ROW_MAJOR, dram, Fab::Line, 8, {}},
        {{1, 1, 1024, 2560}, -1, Layout::TILE, dram, Fab::NeighborExchange, 8, {}},
        {{1, 1, 256, 256}, -1, Layout::TILE, dram, Fab::True2D, 8, {}},
        {{1, 1, 256, 8192}, -1, Layout::TILE, dram, Fab::Ring, 8, {.algo = AgAlgo::Unicast, .workers = 1}},
        {{1, 1, 256, 8192}, -1, Layout::TILE, dram, Fab::Line, 8, {.algo = AgAlgo::Multicast, .links = 1}},
        {{1, 1, 256, 8192}, -1, Layout::TILE, dram, Fab::Line, 8, {.algo = AgAlgo::Composite}},
        {{1, 1, 32, 320}, -1, Layout::TILE, dram, Fab::Line, 8, {}},  // gather-dim padded: composite
    };
    for (const auto& c : cases) {
        for (const ArchVals& a : {kBH, kWH}) {
            const auto [in, out] = gather_specs(c.out, c.dim, c.N, DataType::BFLOAT16, c.layout, dram, c.out_mc);
            // all_gather(): facts and force into the params, plan, store the choice.
            AllGatherParams params = make_params(out, c.dim, c.N, c.fab, 2, make_facts(a), c.force);
            const AgPlan planned = make_plan(make_plan_in(params, in, out));
            params.choice = planned.choice;
            if (c.force.algo != AgAlgo::Auto) {
                EXPECT_EQ(planned.choice.algo, c.force.algo);
            }
            if (planned.choice.algo == AgAlgo::Composite) {
                EXPECT_ANY_THROW(expand(params, in, out));  // composite runs no factory
                continue;
            }
            // create(): re-expand from the hashed params alone.
            EXPECT_EQ(expand(params, in, out), planned)
                << fmt::format("{}", fmt::join(c.out, "x")) << " -> " << algo_str(planned.choice.algo);
        }
    }
}

}  // namespace
