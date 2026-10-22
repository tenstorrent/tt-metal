// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host tests for the all_gather plan (G1-G7): legality, geometry, slicing, multicast route sets, the unicast schedule,
// the algorithm choice, and the pinned algorithm of every device-matrix instance. No device needed.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/ccl/all_gather/device/all_gather_device_operation_types.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_plan.hpp"

namespace {

using namespace ttnn::operations::ccl::all_gather_plan;
using tt::tt_fabric::FabricConfig;
using tt::tt_fabric::Topology;
using tt::tt_metal::BufferType;
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
using ttnn::operations::ccl::AllGatherParams;

struct ArchVals {
    tt::ARCH arch;
    uint32_t dram, l1, P;
};
const ArchVals kBH{tt::ARCH::BLACKHOLE, 64, 16, 15168};
const ArchVals kWH{tt::ARCH::WORMHOLE_B0, 32, 16, 7616};

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
AllGatherParams make_params(const TensorSpec& out, int32_t dim, uint32_t N, Fab fab, uint32_t links, uint32_t P) {
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
        .packet_size = P,
        .subdevice_id = std::nullopt,
        .sub_core_grid = std::nullopt};
}

PlanIn plan_in_for(
    const TensorSpec& in,
    const TensorSpec& out,
    int32_t dim,
    uint32_t N,
    Fab fab,
    const ArchVals& a,
    uint32_t links = 2,
    uint32_t avail_cores = 120) {
    return make_plan_in(make_params(out, dim, N, fab, links, a.P), in, out, avail_cores, a.arch, a.dram, a.l1);
}

// "-" when illegal, else mode, A and footprints, e.g. "w64 1/1" or "s16 2/8".
std::string mode(const std::optional<Geometry>& g) {
    if (!g) {
        return "-";
    }
    return fmt::format("{}{} {}/{}", g->whole_page ? 'w' : 's', g->A, g->Ui, g->Uo);
}

GeoIn page_geo(const ArchVals& a, uint32_t ci, uint32_t co, bool in_dram, bool out_dram) {
    const uint32_t ri = in_dram ? a.dram : a.l1;
    const uint32_t ro = out_dram ? a.dram : a.l1;
    return GeoIn{
        .ci = ci,
        .co = co,
        .ai = tt::align(ci, ri),
        .ao = tt::align(co, ro),
        .ri = ri,
        .ro = ro,
        .in_pages = 8,
        .pages_per_stripe_row = 1,
        .N = 8,
        .P = a.P};
}

// G1: legality table. Unicast's A also covers its own output (relay reads); multicast's only the input.
TEST(AllGatherPlan, G1Legality) {
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
        const GeoIn g = page_geo(r.a, r.ci, r.co, r.in_dram, r.out_dram);
        const auto u = make_geometry(g, true);
        const auto m = make_geometry(g, false);
        EXPECT_EQ(mode(u), r.unicast) << r.name;
        EXPECT_EQ(mode(m), r.multicast) << r.name;
        for (const auto& geo : {u, m}) {
            if (!geo) {
                continue;
            }
            EXPECT_EQ(geo->P % geo->A, 0u) << r.name;
            EXPECT_EQ(geo->dst_cross, geo->Uo * geo->A == g.ao) << r.name;
            EXPECT_EQ(geo->src_cross, geo->Ui * geo->A == g.ai) << r.name;
            if (geo->whole_page) {
                EXPECT_LE(geo->Uo * geo->A, g.ao) << r.name;  // the slot tail lands in the page's own padding
            }
        }
    }

    // Illegal for both algorithms is composite, with the rule that fired, never a TT_FATAL.
    const std::string cut = "cannot cut these pages into aligned units";
    PlanIn p{.arch = tt::ARCH::BLACKHOLE, .geo = page_geo(kBH, 16, 16, D, L), .shape_ok = true, .N = 8, .links = 2};
    p.avail_cores = 64;
    EXPECT_EQ(make_plan(p).composite_reason, fmt::format("unicast {0}; multicast {0}", cut));
    p.geo = page_geo(kBH, 16, 128, L, D);
    p.forced_unicast = true;
    EXPECT_EQ(
        make_plan(p).composite_reason, fmt::format("unicast {}; multicast cannot route on this fabric axis", cut));
    p.forced_unicast = false;
    EXPECT_EQ(make_plan(p).algo, AgAlgo::Multicast);
    p.true_2d = true;
    p.avail_cores = 0;
    EXPECT_EQ(make_plan(p).composite_reason, "unicast needs a 1D axis; multicast has no free worker cores");
    // Pages that are not whole rows, and an element count past uint32.
    GeoIn g = page_geo(kBH, 2048, 2048, D, D);
    g.in_pages = 9;
    g.pages_per_stripe_row = 2;
    EXPECT_FALSE(make_geometry(g, true) || make_geometry(g, false));
    g = page_geo(kBH, 128, 192, D, D);  // E = 1
    g.in_pages = 3'000'000'000;
    EXPECT_FALSE(make_geometry(g, true) || make_geometry(g, false));

    // Ragged row-major width shards (96 = 64 + 32): the spec pads each row to whole shards (128), as legacy
    // sharding does. Gathering the width would gather that padding, so it is composite; gathering rows moves whole
    // padded pages natively.
    const auto rm = [](uint32_t h, uint32_t w) {
        return make_spec(
            shape_of({1, 1, h, w}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR,
            nd_sharded(BufferType::L1, shape_of({1, 1, 1, 64})));
    };
    const Plan by_width = make_plan(plan_in_for(rm(32, 96), rm(32, 768), -1, 8, Fab::Line, kBH));
    EXPECT_EQ(by_width.composite_reason, "the output pages are not the input page grid scaled on the gather dim");
    const PlanIn by_rows = plan_in_for(rm(32, 96), rm(256, 96), -2, 8, Fab::Line, kBH);
    EXPECT_EQ(by_rows.geo.pages_per_stripe_row, 64u);  // 32 rows of 2 pages
    EXPECT_EQ(by_rows.geo.in_pages, 64u);
    EXPECT_NE(make_plan(by_rows).algo, AgAlgo::Composite);
}

// G2: geometry from real specs vs a brute-force placement model.
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

TEST(AllGatherPlan, G2GeometryVsModel) {
    std::mt19937 rng(1);
    auto pick = [&](const auto& v) { return v[std::uniform_int_distribution<size_t>(0, v.size() - 1)(rng)]; };
    auto rnd = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };

    std::map<std::string, int> checked;  // per kind, so a silent skip of a whole kind fails
    for (int trial = 0; trial < 1500; ++trial) {
        const ArchVals a = rnd(0, 1) ? kBH : kWH;
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

        const PlanIn p = plan_in_for(*in, *out, dim, N, Fab::Line, a);
        ASSERT_TRUE(p.shape_ok) << trial;
        const auto in_grid = page_grid(*in, elem);
        const auto out_grid = page_grid(*out, elem);
        const uint64_t in_pages = std::accumulate(in_grid.begin(), in_grid.end(), 1ull, std::multiplies<>());
        ASSERT_EQ(p.geo.in_pages, in_pages) << trial;
        const uint32_t ci = p.geo.ci;
        const uint32_t g_dim = in_grid.size() + dim;  // gather dim in the page grid (padded rank may exceed rank)

        for (bool unicast : {true, false}) {
            const auto geo = make_geometry(p.geo, unicast);
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
                        const uint32_t out_pw = p.geo.co / elem;
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
            checked[kind + (geo->whole_page ? " whole" : " sub")]++;
        }
    }
    for (const char* kind :
         {"tile sub", "tile whole", "rm matched sub", "rm matched whole", "rm concat sub", "rm split sub"}) {
        EXPECT_GT(checked[kind], 10) << kind;
    }
}

// G3: slicer.
TEST(AllGatherPlan, G3Slicer) {
    for (uint32_t elements = 1; elements <= 300; ++elements) {
        for (uint32_t links = 1; links <= std::min(elements, 8u); ++links) {
            const auto slices = make_slices(elements, links);
            ASSERT_EQ(slices.size(), links);
            uint32_t next = 0, lo = elements, hi = 0;
            for (const auto& s : slices) {
                ASSERT_EQ(s.e0, next);
                ASSERT_GT(s.ne, 0u);
                next += s.ne;
                lo = std::min(lo, s.ne);
                hi = std::max(hi, s.ne);
            }
            ASSERT_EQ(next, elements);
            ASSERT_LE(hi - lo, 1u);
        }
    }
}

// G4: multicast route sets. Each variant's routes cover every other device of the collective exactly once.
Plan mcast_plan(uint32_t N) {
    Plan p{};
    p.algo = AgAlgo::Multicast;
    p.N = N;
    return p;
}

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
    if (c.dir == 0 || c.dir == 1) {
        EXPECT_EQ(n + s, 0);
        EXPECT_EQ(c.dir == 0 ? w : e, 0);
        const int hops = c.dir == 0 ? e : w;
        for (int k = 1; k <= hops; ++k) {
            add(0, c.dir == 0 ? k : -k);
        }
    } else {
        EXPECT_EQ(c.dir == 3 ? n : s, 0);
        const int hops = c.dir == 3 ? s : n;
        const int sign = c.dir == 3 ? 1 : -1;
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
        active[a] = mesh[a] > 1 && axis.value_or(a) == a;
        wraps[a] = active[a] && ring[a] && mesh[a] > 2;
        expect_alt = expect_alt || (wraps[a] && mesh[a] % 2 == 0);
        N *= active[a] ? mesh[a] : 1;
    }
    const Plan plan = mcast_plan(N);
    const auto label =
        fmt::format("mesh {}x{} axis {} ring {}/{}", mesh[0], mesh[1], axis.value_or(9), ring[0], ring[1]);
    for (uint32_t r = 0; r < mesh[0]; ++r) {
        for (uint32_t c = 0; c < mesh[1]; ++c) {
            const auto routes = mcast_routes(plan, mesh, {r, c}, axis, ring);
            ASSERT_EQ(routes.alternates, expect_alt) << label;
            ASSERT_EQ(mcast_done_expect(plan, routes.alternates), (expect_alt ? 2 : 1) * (N - 1)) << label;
            ASSERT_LE(routes.conns.size(), (active[0] && active[1]) ? 4u : 2u) << label;
            for (int v = 0; v < 2; ++v) {
                std::map<std::array<int, 2>, int> hits;
                for (const auto& conn : routes.conns) {
                    const uint32_t range = v == 0 ? conn.range_v0 : conn.range_v1;
                    EXPECT_NE(range, 0u) << label;  // every (connection, variant) carries a done inc
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
            const auto swapped = mcast_routes(plan, mesh, {r, c}, axis, ring, swap);
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

TEST(AllGatherPlan, G4MulticastRoutes) {
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

// G5: unicast schedule. Simulates the relay chain: every device receives every other stripe's units exactly once,
// recv_units equals what upstream sends, and each relay only gates on units upstream has sent.
Plan unicast_plan(uint32_t N, bool ring, uint32_t elements, uint32_t links, uint32_t E) {
    Plan p{};
    p.algo = AgAlgo::Unicast;
    p.N = N;
    p.ring = ring && N > 2;  // what make_plan does
    p.geo.E = E;
    p.links = links;
    p.slices = make_slices(elements, links);
    return p;
}

TEST(AllGatherPlan, G5UnicastSchedule) {
    for (uint32_t N = 2; N <= 16; ++N) {
        for (bool ring : {false, true}) {
            for (uint32_t elements : {1u, 2u, 3u, 7u, 16u}) {
                for (uint32_t links = 1; links <= std::min(elements, 2u); ++links) {
                    const uint32_t E = 3;
                    const Plan plan = unicast_plan(N, ring, elements, links, E);
                    const auto label = fmt::format("N {} ring {} elements {} links {}", N, ring, elements, links);
                    for (uint32_t l = 0; l < links; ++l) {
                        const uint32_t ne = plan.slices[l].ne;
                        // got[t][s][e]: times device t received element e of stripe s
                        std::vector<std::vector<std::vector<int>>> got(
                            N, std::vector<std::vector<int>>(N, std::vector<int>(ne, 0)));
                        for (uint32_t dev = 0; dev < N; ++dev) {
                            const auto f = unicast_schedule(plan, dev, l, 0);
                            const auto b = unicast_schedule(plan, dev, l, 1);
                            ASSERT_EQ(int(f.local_copy) + int(b.local_copy), 1) << label << " dev " << dev;
                            ASSERT_GT(f.I + b.I, 0u) << label;
                            for (uint32_t dir = 0; dir < 2; ++dir) {
                                const auto u = dir == 0 ? f : b;
                                ASSERT_EQ(u.stripe_step, dir == 0 ? N - 1 : 1) << label;
                                ASSERT_EQ(u.iter_units, ne * E);
                                ASSERT_EQ(u.last_units, u.last_take * E);
                                ASSERT_LE(u.last_skip + u.last_take, ne);
                                if (u.local_copy) {
                                    // The copy comes from iteration 0, so that iteration must run and be whole.
                                    ASSERT_TRUE(u.I > 1 || (u.I == 1 && u.last_skip == 0 && u.last_take == ne))
                                        << label << " dev " << dev;
                                }
                                // Upstream (same direction) sends into this core; check the counts and the gates.
                                const bool has_up = plan.ring || (dir == 0 ? dev > 0 : dev + 1 < N);
                                const uint32_t up_dev = dir == 0 ? (dev + N - 1) % N : (dev + 1) % N;
                                const auto up = unicast_schedule(plan, up_dev, l, dir);
                                const uint32_t up_total = up.I == 0 ? 0 : ((up.I - 1) * up.iter_units) + up.last_units;
                                ASSERT_EQ(u.recv_units, has_up ? up_total : 0) << label << " dev " << dev;
                                // Iteration i sends stripe dev + i * step downstream.
                                const uint32_t down = dir == 0 ? (dev + 1) % N : (dev + N - 1) % N;
                                for (uint32_t i = 0; i < u.I; ++i) {
                                    const bool last = i == u.I - 1;
                                    const uint32_t skip = last ? u.last_skip : 0;
                                    const uint32_t take = last ? u.last_take : ne;
                                    const uint32_t stripe = (dev + i * u.stripe_step) % N;
                                    for (uint32_t e = skip; e < skip + take; ++e) {
                                        got[down][stripe][e]++;
                                    }
                                    if (i == 0) {
                                        continue;
                                    }
                                    // A relay forwards what upstream sent in iteration i - 1, which must be full,
                                    // and gates only on units upstream sends.
                                    ASSERT_TRUE(has_up && i - 1 < up.I) << label;
                                    ASSERT_TRUE(i < up.I || (up.last_skip == 0 && up.last_take == ne)) << label;
                                    ASSERT_EQ(stripe, (up_dev + (i - 1) * up.stripe_step) % N) << label;
                                    ASSERT_LE(((i - 1) * u.iter_units) + ((skip + take) * E), up_total) << label;
                                }
                            }
                        }
                        for (uint32_t t = 0; t < N; ++t) {
                            for (uint32_t s = 0; s < N; ++s) {
                                for (uint32_t e = 0; e < ne; ++e) {
                                    ASSERT_EQ(got[t][s][e], s == t ? 0 : 1)
                                        << label << " dev " << t << " stripe " << s << " element " << e;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// G6: crossover thresholds and make_plan determinism.
TEST(AllGatherPlan, G6CrossoverAndDeterminism) {
    constexpr uint64_t KB = 1024;
    const auto BH = tt::ARCH::BLACKHOLE;
    const auto WH = tt::ARCH::WORMHOLE_B0;
    // BH: ring 320*(N/2-2) KB, line 384+544*(N-4) KB, floored at 0.
    EXPECT_FALSE(prefer_unicast(BH, true, 8, 2, (640 * KB) - 1));
    EXPECT_TRUE(prefer_unicast(BH, true, 8, 2, 640 * KB));
    EXPECT_FALSE(prefer_unicast(BH, false, 8, 2, (2560 * KB) - 1));
    EXPECT_TRUE(prefer_unicast(BH, false, 8, 2, 2560 * KB));
    EXPECT_TRUE(prefer_unicast(BH, true, 4, 2, 0));
    EXPECT_TRUE(prefer_unicast(BH, false, 3, 2, 0));
    EXPECT_TRUE(prefer_unicast(BH, false, 2, 2, 0));
    EXPECT_FALSE(prefer_unicast(BH, false, 5, 2, (928 * KB) - 1));
    EXPECT_TRUE(prefer_unicast(BH, false, 5, 2, 928 * KB));
    // WH: ring multicast; 2 unicast; 3-6 multicast; 7+ by size.
    EXPECT_FALSE(prefer_unicast(WH, true, 8, 2, uint64_t(1) << 40));
    EXPECT_TRUE(prefer_unicast(WH, false, 2, 2, 0));
    EXPECT_FALSE(prefer_unicast(WH, false, 6, 2, uint64_t(1) << 40));
    EXPECT_FALSE(prefer_unicast(WH, false, 8, 2, 2'999'999));
    EXPECT_TRUE(prefer_unicast(WH, false, 8, 2, 3'000'000));

    // The busiest link carries output / effective unicast links; fewer cores cut unicast's links first.
    PlanIn p{
        .arch = BH, .geo = GeoIn{2048, 2048, 2048, 2048, 64, 64, 1024, 4, 8, 15168}, .shape_ok = true, .ring = true};
    p.N = 8;
    p.links = 2;
    p.avail_cores = 64;
    p.output_packed_bytes = 2 * 640 * KB;
    EXPECT_EQ(make_plan(p).algo, AgAlgo::Unicast);
    p.output_packed_bytes = (2 * 640 * KB) - 2;
    EXPECT_EQ(make_plan(p).algo, AgAlgo::Multicast);
    p.avail_cores = 3;  // one unicast link left: the same bytes now cross one link
    EXPECT_EQ(make_plan(p).algo, AgAlgo::Unicast);
    EXPECT_EQ(make_plan(p).links, 1u);
    p.avail_cores = 1;  // no room for unicast's two cores
    EXPECT_EQ(make_plan(p).algo, AgAlgo::Multicast);
    p.avail_cores = 0;
    EXPECT_EQ(make_plan(p).algo, AgAlgo::Composite);

    // 2-device ring is a line.
    p.avail_cores = 64;
    p.N = 2;
    EXPECT_FALSE(make_plan(p).ring);

    // Random inputs: same input, same plan; links bounded by cores and elements; one slice per link.
    std::mt19937 rng(7);
    auto rnd = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };
    for (int trial = 0; trial < 5000; ++trial) {
        PlanIn q{};
        q.arch = rnd(0, 1) ? BH : WH;
        const ArchVals& a = q.arch == BH ? kBH : kWH;
        q.shape_ok = rnd(0, 7) != 0;
        q.ring = rnd(0, 1);
        q.true_2d = rnd(0, 5) == 0;
        q.forced_unicast = !q.true_2d && rnd(0, 4) == 0;
        q.N = rnd(2, 16);
        q.links = rnd(1, 4);
        q.avail_cores = rnd(0, 10);
        const uint32_t ci = 16 * rnd(1, 140);
        const uint32_t co = rnd(0, 1) ? ci : ci * rnd(1, 4);
        q.geo = page_geo(a, ci, co, rnd(0, 1), rnd(0, 1));
        q.geo.pages_per_stripe_row = rnd(1, 4);
        q.geo.in_pages = q.geo.pages_per_stripe_row * rnd(1, 50);
        q.geo.N = q.N;
        q.output_packed_bytes = uint64_t(rnd(1, 1 << 20)) * rnd(1, 64);

        const Plan x = make_plan(q);
        const Plan y = make_plan(q);
        ASSERT_EQ(x.algo, y.algo);
        ASSERT_EQ(x.links, y.links);
        ASSERT_EQ(x.ring, y.ring);
        ASSERT_EQ(x.slices.size(), y.slices.size());
        ASSERT_EQ(x.ring, q.ring && q.N > 2);
        if (!q.shape_ok) {
            ASSERT_EQ(x.algo, AgAlgo::Composite);
        }
        if (x.algo == AgAlgo::Composite) {
            ASSERT_TRUE(x.slices.empty());
            continue;
        }
        const bool unicast = x.algo == AgAlgo::Unicast;
        ASSERT_FALSE(unicast && q.true_2d);
        ASSERT_FALSE(!unicast && q.forced_unicast);
        const auto geo = make_geometry(q.geo, unicast);
        ASSERT_TRUE(geo.has_value());
        ASSERT_EQ(x.geo.elements, geo->elements);
        ASSERT_GE(x.links, 1u);
        ASSERT_LE(x.links, std::min({q.links, q.avail_cores / (unicast ? 2 : 1), geo->elements}));
        ASSERT_EQ(x.slices.size(), x.links);
    }
}

// G7: every device-matrix instance (tests/.../test_all_gather_matrix.py) gets the algorithm it claims, so a
// recalibrated threshold that flips one fails here. BH, (1,8) mesh, 2 links unless noted. Shapes are output shapes.
// Instances dropped because the table forces unicast at every size: M on a 3-device line, a 2-device ring (a line)
// and a 4-device ring (D11, D12); G6 pins those zero thresholds.
TEST(AllGatherPlan, G7MatrixPin) {
    const auto U = AgAlgo::Unicast;
    const auto M = AgAlgo::Multicast;
    const auto C = AgAlgo::Composite;
    const auto bf16 = DataType::BFLOAT16;
    const auto bfp8 = DataType::BFLOAT8_B;
    const auto TILE = Layout::TILE;
    const auto RM = Layout::ROW_MAJOR;
    const auto dram = interleaved(BufferType::DRAM);
    const auto l1 = interleaved(BufferType::L1);
    const auto nd_dram = nd_sharded(BufferType::DRAM, shape_of({1, 1, 32, 32}));
    const auto ws = TensorMemoryLayout::WIDTH_SHARDED;
    const auto hs_l1 = legacy_sharded(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, 32, 32, 8);
    // D6: 128 B input rows split into 64 B output pages. D9: 32 B rows, L1 -> DRAM (big variant, then small).
    const auto d6_in = legacy_sharded(ws, BufferType::L1, 256, 64, 1);
    const auto d6_out = legacy_sharded(ws, BufferType::L1, 2048, 32, 2);
    const auto d9_in = legacy_sharded(ws, BufferType::L1, 20480, 16, 1);
    const auto d9_out = legacy_sharded(ws, BufferType::DRAM, 8 * 20480, 16, 1);
    const auto d9s_in = legacy_sharded(ws, BufferType::L1, 4096, 16, 1);
    const auto d9s_out = legacy_sharded(ws, BufferType::DRAM, 8 * 4096, 16, 1);

    struct Instance {
        std::string id;
        std::vector<uint32_t> out_shape;
        int32_t dim;
        DataType dt;
        Layout layout;
        MemoryConfig in_mc, out_mc;
        Fab fab;
        uint32_t N;
        AgAlgo expect;
        Tile tile = Tile();
        uint32_t avail_cores = 120;
    };
    const std::vector<Instance> instances = {
        {"D1 U-NE", {1, 1, 1024, 2560}, -1, bf16, TILE, dram, dram, Fab::NeighborExchange, 8, U},
        {"D1 U-size ring", {1, 1, 1024, 2560}, -1, bf16, TILE, dram, dram, Fab::Ring, 8, U},
        {"D1 U-size line", {1, 1, 4096, 2560}, -1, bf16, TILE, dram, dram, Fab::Line, 8, U},
        {"D1 M", {1, 1, 256, 2560}, -1, bf16, TILE, dram, dram, Fab::Line, 8, M},
        {"D2 U-NE", {1, 1, 256, 256}, -1, bfp8, TILE, dram, dram, Fab::NeighborExchange, 8, U},
        {"D2 M", {1, 1, 256, 256}, -1, bfp8, TILE, dram, dram, Fab::Line, 8, M},
        {"D3 8KB rows U-NE", {8, 1, 256, 4096}, -4, bf16, RM, dram, dram, Fab::NeighborExchange, 8, U},
        {"D3 8KB rows M", {8, 1, 16, 4096}, -4, bf16, RM, dram, dram, Fab::Line, 8, M},
        {"D3 chunk_over_packet U-NE", {1, 1, 64, 4096}, -2, bf16, RM, dram, dram, Fab::NeighborExchange, 8, U},
        {"D3 chunk_over_packet M", {1, 1, 64, 4096}, -2, bf16, RM, dram, dram, Fab::Line, 8, M},
        {"D3 20000B rows U-NE", {1, 1, 512, 10000}, -2, bf16, RM, dram, dram, Fab::NeighborExchange, 8, U},
        {"D3 20000B rows M", {1, 1, 64, 10000}, -2, bf16, RM, dram, dram, Fab::Line, 8, M},
        {"D4 48B rows U-NE", {1, 1, 4096, 24}, -2, bf16, RM, dram, dram, Fab::NeighborExchange, 8, U},
        {"D4 48B rows M", {1, 1, 4096, 24}, -2, bf16, RM, dram, dram, Fab::Line, 8, M},
        {"D5 concat U-NE", {1, 1, 2048, 64}, -1, bf16, RM, l1, l1, Fab::NeighborExchange, 8, U},
        {"D5 concat M", {1, 1, 2048, 64}, -1, bf16, RM, l1, l1, Fab::Line, 8, M},
        {"D6 split U-NE", {1, 1, 2048, 64}, -2, bf16, RM, d6_in, d6_out, Fab::NeighborExchange, 8, U},
        {"D7 hs l1 -> dram U-NE", {1, 1, 256, 256}, -1, bf16, TILE, hs_l1, dram, Fab::NeighborExchange, 8, U},
        {"D7 hs l1 -> dram M", {1, 1, 256, 256}, -1, bf16, TILE, hs_l1, dram, Fab::Line, 8, M},
        {"D7 dram -> l1 U-NE", {1, 1, 256, 256}, -1, bf16, TILE, dram, l1, Fab::NeighborExchange, 8, U},
        {"D7 dram -> l1 M", {1, 1, 256, 256}, -1, bf16, TILE, dram, l1, Fab::Line, 8, M},
        {"D8 nd dram U-NE", {1, 1, 256, 256}, -1, bf16, TILE, nd_dram, nd_dram, Fab::NeighborExchange, 8, U},
        {"D8 nd dram M", {1, 1, 256, 256}, -1, bf16, TILE, nd_dram, nd_dram, Fab::Line, 8, M},
        {"D9 mismatched_alignment U-size ring", {8, 1, 20480, 16}, -4, bf16, RM, d9_in, d9_out, Fab::Ring, 8, U},
        {"D9 mismatched_alignment M", {8, 1, 4096, 16}, -4, bf16, RM, d9s_in, d9s_out, Fab::Line, 8, M},
        {"D9b 16B rows l1 -> dram", {1, 1, 64, 64}, -1, bf16, RM, l1, dram, Fab::Line, 8, M},
        {"D9b at unicast size", {1, 1, 65536, 64}, -1, bf16, RM, l1, dram, Fab::Line, 8, M},
        {"D11 (1,5) U-NE", {1, 1, 256, 640}, -1, bf16, TILE, dram, dram, Fab::NeighborExchange, 5, U},
        {"D11 (1,5) M", {1, 1, 64, 640}, -1, bf16, TILE, dram, dram, Fab::Line, 5, M},
        {"D11 (1,3) U-NE", {1, 1, 64, 384}, -1, bf16, TILE, dram, dram, Fab::NeighborExchange, 3, U},
        {"D11 (1,2) ring U-NE", {1, 1, 64, 256}, -1, bf16, TILE, dram, dram, Fab::NeighborExchange, 2, U},
        {"D12 1 tile/dev ring", {1, 1, 32, 128}, -1, bf16, TILE, dram, dram, Fab::Ring, 4, U},
        {"D13 U-size ring", {1, 1, 1024, 2560}, -1, bf16, TILE, dram, dram, Fab::Ring, 8, U},
        {"D13 M ring", {1, 1, 64, 2560}, -1, bf16, TILE, dram, dram, Fab::Ring, 8, M},
        {"D14 gather-dim padded", {1, 1, 32, 40}, -1, bf16, TILE, dram, dram, Fab::Line, 8, C},
        {"D14 bfp8 16x32 dram -> l1", {1, 1, 64, 256}, -1, bfp8, TILE, dram, l1, Fab::Line, 8, C, Tile({16, 32})},
        {"D14 48B rows dram -> l1", {1, 1, 4096, 24}, -2, bf16, RM, dram, l1, Fab::Line, 8, C},
        {"D15 two rows U-NE", {1, 1, 1024, 2560}, -1, bf16, TILE, dram, dram, Fab::NeighborExchange, 8, U, Tile(), 26},
        {"D15 two rows M", {1, 1, 256, 2560}, -1, bf16, TILE, dram, dram, Fab::Line, 8, M, Tile(), 26},
        {"D16 2D fabric axis U-size", {1, 1, 4096, 2560}, -1, bf16, TILE, dram, dram, Fab::Fabric2DLine, 4, U},
        {"D16 2D fabric axis M", {1, 1, 32, 1024}, -1, bf16, TILE, dram, dram, Fab::Fabric2DLine, 4, M},
        {"D16 true 2D (2,4)", {1, 1, 256, 256}, -1, bf16, TILE, dram, dram, Fab::True2D, 8, M},
        {"D17 D9b on neighbor exchange", {1, 1, 64, 64}, -1, bf16, RM, l1, dram, Fab::NeighborExchange, 8, C},
        {"D18 26MB U-NE", {1, 1, 4096, 3328}, -1, bf16, TILE, dram, dram, Fab::NeighborExchange, 8, U},
        {"D18 26MB ring", {1, 1, 4096, 3328}, -1, bf16, TILE, dram, dram, Fab::Ring, 8, U},
        {"D18 26MB line", {1, 1, 4096, 3328}, -1, bf16, TILE, dram, dram, Fab::Line, 8, U},
    };

    for (const auto& inst : instances) {
        auto in_shape = inst.out_shape;
        in_shape[in_shape.size() + inst.dim] /= inst.N;
        const auto in = make_spec(shape_of(in_shape), inst.dt, inst.layout, inst.in_mc, inst.tile);
        const auto out = make_spec(shape_of(inst.out_shape), inst.dt, inst.layout, inst.out_mc, inst.tile);
        const PlanIn p = plan_in_for(in, out, inst.dim, inst.N, inst.fab, kBH, 2, inst.avail_cores);
        const Plan plan = make_plan(p);
        EXPECT_EQ(plan.algo, inst.expect) << inst.id;
        if (inst.expect == U || inst.expect == M) {
            // "U-size" / "M" instances sit 4x or more clear of the threshold, so recalibration noise can't flip them.
            if (inst.fab == Fab::Ring || inst.fab == Fab::Line || inst.fab == Fab::Fabric2DLine) {
                const uint64_t per_link = p.output_packed_bytes / 2;
                const bool ring = inst.fab == Fab::Ring && inst.N > 2;
                if (inst.id.find("U-size") != std::string::npos) {
                    EXPECT_TRUE(prefer_unicast(kBH.arch, ring, inst.N, 2, per_link / 4)) << inst.id;
                }
                if (inst.id.find(" M") != std::string::npos) {
                    EXPECT_FALSE(prefer_unicast(kBH.arch, ring, inst.N, 2, (per_link * 4) - 1)) << inst.id;
                }
            }
        }
    }

    // D12: one element, one link, and the forward half of the antipode stripe is empty.
    const auto in = make_spec(shape_of({1, 1, 32, 32}), bf16, TILE, dram);
    const auto out = make_spec(shape_of({1, 1, 32, 128}), bf16, TILE, dram);
    const Plan plan = make_plan(plan_in_for(in, out, -1, 4, Fab::Ring, kBH));
    EXPECT_EQ(plan.links, 1u);
    EXPECT_EQ(plan.slices.at(0).ne, 1u);
    EXPECT_EQ(unicast_schedule(plan, 0, 0, 0).last_take, 0u);
}

}  // namespace
