// Standalone offline unit test for the Regime-A host planner (no tt_metal / no device).
// Verifies build_plan() against the golden oracle plus the BALANCED-TAIL invariants:
//   - balanced K/M/N ownership is disjoint and covers every logical tile exactly once,
//   - no planned read lies outside logical whole-tile extents,
//   - N slicing has no gaps between Ns owners (within each bank's valid interval),
//   - ring slots cover every valid K block once (excess = local-zero),
//   - address strides come only from tensor layouts (not schedule capacities),
//   - divisible plans are "uniform" (valid == capacity, no zero-fill),
//   - rejects: Pk>Kt, Sm>Mt, empty-N, L1, core-count; grid-gap collision-free placement.
//
// Build + run:
//   g++ -std=c++17 -I ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device \
//       tools/mm_sweep/regime_a_plan_test.cpp -o /tmp/ra_plan_test && /tmp/ra_plan_test

#include "regime_a_matmul_plan.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ttnn::operations::experimental::regime_a_matmul::plan;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (cond) {                                                        \
            ++g_pass;                                                      \
        } else {                                                           \
            ++g_fail;                                                      \
            std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                                  \
    } while (0)

#define CHECK_EQ(a, b, msg)                                                                                \
    do {                                                                                                   \
        auto _a = (long long)(a);                                                                          \
        auto _b = (long long)(b);                                                                          \
        if (_a == _b) {                                                                                    \
            ++g_pass;                                                                                      \
        } else {                                                                                           \
            ++g_fail;                                                                                      \
            std::printf("  FAIL: %s  got %lld expected %lld  (%s:%d)\n", msg, _a, _b, __FILE__, __LINE__); \
        }                                                                                                  \
    } while (0)

static PlanInputs mk_inputs(uint32_t Mt, uint32_t Kt, uint32_t Nt, RegimeAConfig cfg) {
    PlanInputs in;
    in.Mt = Mt;
    in.Kt = Kt;
    in.Nt = Nt;
    in.cfg = cfg;
    in.grid_x = 11;
    in.grid_y = 10;
    for (uint32_t b = 0; b < 8; ++b) {
        in.opt0.push_back(PlanXY{b, 0});
        in.opt1.push_back(PlanXY{b, 9});
    }
    return in;
}

static bool all_distinct(const ExecutionPlan& p) {
    std::set<PlanXY> s;
    for (const auto& cp : p.cores) {
        if (!s.insert(cp.coord).second) {
            return false;
        }
    }
    return s.size() == p.cores.size();
}

struct Golden {
    const char* label;
    uint32_t Mt, Kt, Nt;
    RegimeAConfig cfg;
    uint32_t exp_cores, exp_shard_rows, exp_shard_cols, exp_l1_kb;
};

// Verify balanced ranges over `total` split into `parts` are disjoint + exactly cover [0,total).
static void check_balanced_cover(const char* what, std::vector<std::pair<uint32_t, uint32_t>> ranges, uint32_t total) {
    // sort by start
    std::sort(ranges.begin(), ranges.end());
    uint32_t cursor = 0;
    bool ok = true;
    for (auto& r : ranges) {
        if (r.first != cursor) {
            ok = false;
        }
        cursor = r.first + r.second;
    }
    if (cursor != total) {
        ok = false;
    }
    CHECK(ok, what);
}

int main() {
    std::printf("== Regime-A host planner offline tests (balanced tails) ==\n");

    // ---- Golden oracle: cores / in1 shard (Kt rows, ceil(Nt/8) cols) / L1 ----
    Golden golden[] = {
        {"Mt1 32x6144x4608", 1, 192, 144, {12, 1, 1, 2, 1}, 96, 192, 18, 60},
        {"Mt2 64x6144x4608", 2, 192, 144, {6, 1, 1, 4, 2}, 48, 192, 18, 240},
        {"Mt4 128x6144x4608", 4, 192, 144, {12, 1, 1, 2, 1}, 96, 192, 18, 192},
        {"Mt8 256x6144x4608", 8, 192, 144, {12, 1, 1, 2, 1}, 96, 192, 18, 368},
        {"Mt4 small-N 128x6144x768", 4, 192, 24, {12, 1, 1, 2, 1}, 96, 192, 3, 192},
    };
    for (const auto& go : golden) {
        std::printf("[golden] %s\n", go.label);
        auto r = build_plan(mk_inputs(go.Mt, go.Kt, go.Nt, go.cfg));
        CHECK(r.ok(), "plan built");
        if (!r.ok()) {
            std::printf("    error: %s\n", r.error.c_str());
            continue;
        }
        const auto& p = *r.plan;
        CHECK_EQ(p.geo.num_cores, go.exp_cores, "num_cores");
        CHECK_EQ(p.in1_shard_rows, go.exp_shard_rows, "in1 shard rows (Kt)");
        CHECK_EQ(p.in1_shard_cols, go.exp_shard_cols, "in1 shard cols (ceil(Nt/8))");
        CHECK_EQ(p.cb.l1_bytes / 1024u, go.exp_l1_kb, "L1 KB");
        CHECK(all_distinct(p), "collision-free placement");
        // divisible: strides from layout; zero schedule waste; valid == capacity everywhere.
        CHECK_EQ(p.geo.in0_stride_k, go.Kt, "in0_stride_k == Kt");
        CHECK_EQ(p.geo.out_stride_n, go.Nt, "out_stride_n == Nt");
        CHECK_EQ(p.geo.in1_shard_stride_n, (go.Nt + 7u) / 8u, "in1_shard_stride_n == ceil(Nt/8)");
        CHECK(std::abs(p.geo.waste_k) < 1e-9 && std::abs(p.geo.waste_n) < 1e-9, "divisible: zero waste");
        for (const auto& cp : p.cores) {
            CHECK(cp.valid_k == p.geo.K_slice_capacity, "divisible: valid_k == K_slice_capacity");
            CHECK(cp.valid_m == p.geo.M_block_capacity, "divisible: valid_m == M_block_capacity");
            CHECK(cp.valid_n == p.geo.N_slice_capacity, "divisible: valid_n == N_slice_capacity");
        }
    }

    // ---- Balanced invariants on the non-divisible shape 32x6080x4640 (Kt=190, Nt=145) ----
    {
        std::printf("[balanced] 32x6080x4640 (Kt=190 Nt=145) Pk=12\n");
        auto r = build_plan(mk_inputs(1, 190, 145, {12, 1, 1, 2, 1}));
        CHECK(r.ok(), "plan built");
        const auto& p = *r.plan;
        const auto& g = p.geo;
        // strides from layout (NOT schedule): in1 stride = ceil(145/8)=19, capacity Kt_local=16.
        CHECK_EQ(g.in0_stride_k, 190u, "in0_stride_k == Kt=190");
        CHECK_EQ(g.out_stride_n, 145u, "out_stride_n == Nt=145");
        CHECK_EQ(g.in1_shard_stride_n, 19u, "in1_shard_stride_n == 19");
        CHECK_EQ(g.K_slice_capacity, 16u, "K_slice_capacity == rup(ceil(190/12),16)=16");
        CHECK_EQ(p.in1_shard_rows, 190u, "in1 shard rows == Kt (no schedule K pad)");
        // K balanced over Pk=12: disjoint + covers [0,190); no read past Kt; valid_k<=capacity.
        std::vector<std::pair<uint32_t, uint32_t>> kr;
        for (uint32_t kk = 0; kk < 12u; ++kk) {
            const auto& cp = p.cores[kk];  // bank0, slice=kk (Ns=Sm=1)
            kr.push_back({cp.k_start, cp.valid_k});
            CHECK(cp.k_start + cp.valid_k <= 190u, "k range within Kt");
            CHECK(cp.valid_k <= g.K_slice_capacity, "valid_k <= K_slice_capacity");
        }
        check_balanced_cover("K balanced disjoint+cover [0,190)", kr, 190u);
        CHECK_EQ(p.cores[0].valid_k, 15u, "kk=0 valid_k = floor(190/12) = 15 (balanced, not 16)");
        // N across the 8 banks (Ns=1): each bank owns a contiguous global interval; union covers [0,145).
        std::vector<std::pair<uint32_t, uint32_t>> nr;
        for (uint32_t b = 0; b < 8u; ++b) {
            const auto& cp = p.cores[b * g.preaders + 0];
            nr.push_back({cp.n_start, cp.valid_n});
            CHECK(cp.n_start + cp.valid_n <= 145u, "n range within Nt");
            CHECK_EQ(cp.n_local, cp.n_start - b * g.N_band, "n_local == n_start - bank*N_band");
        }
        check_balanced_cover("N banks disjoint+cover [0,145) (no gaps)", nr, 145u);
        CHECK_EQ(p.cores[7 * g.preaders].valid_n, 12u, "bank7 valid_n = 145-133 = 12");
        // ring: valid K blocks <= G*W; capacity blocks == G*W.
        CHECK_EQ(g.K_num_blocks_eff, g.G * g.W, "K_num_blocks_eff == G*W");
        CHECK_EQ(g.K_slice_capacity, g.G * g.W * 2u, "K_slice_capacity == G*W*kb (kb=2)");
    }

    // ---- N no-gaps under Ns>1 on a bank with a tail (subdivide the VALID interval) ----
    {
        std::printf("[balanced] Ns=2 over non-divisible N (145) — no internal holes\n");
        auto r = build_plan(mk_inputs(1, 192, 145, {3, 2, 1, 2, 1}));  // Pk3 Ns2 Sm1
        CHECK(r.ok(), "plan built");
        const auto& p = *r.plan;
        const auto& g = p.geo;
        // For each bank, the Ns=2 owners must tile the bank's valid interval with no gap.
        for (uint32_t b = 0; b < 8u; ++b) {
            std::vector<std::pair<uint32_t, uint32_t>> owners;
            uint32_t b_start = b * g.N_band;
            uint32_t b_end = std::min((b + 1u) * g.N_band, 145u);
            uint32_t b_valid = (b_start < 145u) ? (b_end - b_start) : 0u;
            for (uint32_t nn = 0; nn < 2u; ++nn) {
                // core index: bank*preaders + (kk*mfac + nn*Sm + mm); pick kk=0,mm=0 => slice = nn*Sm = nn
                const auto& cp = p.cores[b * g.preaders + nn];
                owners.push_back({cp.n_start - b_start, cp.valid_n});
            }
            check_balanced_cover("Ns owners tile bank valid interval", owners, b_valid);
        }
    }

    // ---- Ring + reduction links (unchanged structure) ----
    {
        std::printf("[links] Pk3 Ns2 Sm1\n");
        auto r = build_plan(mk_inputs(1, 192, 144, {3, 2, 1, 2, 1}));
        CHECK(r.ok(), "plan built");
        const auto& p = *r.plan;
        CHECK_EQ(p.geo.num_cores, 48u, "num_cores = 48");
        CHECK_EQ(p.geo.mfac, 2u, "mfac = 2");
        for (uint32_t i = 0; i < p.cores.size(); ++i) {
            const auto& cp = p.cores[i];
            if (cp.is_bottom && p.geo.preaders > p.geo.mfac) {
                const auto& nx = p.cores[cp.red_next_idx];
                CHECK_EQ(nx.kk, cp.kk + 1u, "red_next next k-slice");
                CHECK_EQ(nx.bank, cp.bank, "red_next same bank");
            }
            if (cp.is_top) {
                CHECK_EQ(cp.red_next_idx, i, "top red_next self");
            }
        }
        for (uint32_t j = 0; j < p.geo.preaders; ++j) {
            std::set<uint32_t> pos;
            for (uint32_t b = 0; b < 8u; ++b) {
                const auto& cp = p.cores[b * p.geo.preaders + j];
                pos.insert(cp.ring_pos);
                CHECK_EQ(p.cores[cp.ring_next_idx].ring_pos, (cp.ring_pos + 1u) % 8u, "ring_next +1");
            }
            CHECK_EQ(pos.size(), (size_t)8, "ring 8 distinct positions");
        }
    }

    // ---- Pk=1: no reduction, cb7 not allocated ----
    {
        std::printf("[Pk1] no reduction\n");
        auto r = build_plan(mk_inputs(1, 192, 144, {1, 1, 1, 4, 6}));
        CHECK(r.ok(), "plan built");
        const auto& p = *r.plan;
        CHECK_EQ(p.geo.num_cores, 8u, "8 cores");
        CHECK_EQ(p.cb.cb7_tiles, 0u, "cb7 == 0 for Pk1");
        for (const auto& cp : p.cores) {
            CHECK(cp.is_bottom && cp.is_top, "Pk1 core is bottom+top");
        }
    }

    // ---- Rejections ----
    {
        std::printf("[reject] Pk>Kt / Sm>Mt / empty-N / L1 / core-count / grid-gap\n");
        auto r1 = build_plan(mk_inputs(1, 4, 144, {8, 1, 1, 1, 1}));  // Pk=8 > Kt=4
        CHECK(!r1.ok() && r1.error.find("Pk") != std::string::npos, "reject Pk>Kt");
        auto r2 = build_plan(mk_inputs(2, 192, 144, {1, 1, 4, 1, 1}));  // Sm=4 > Mt=2
        CHECK(!r2.ok() && r2.error.find("Sm") != std::string::npos, "reject Sm>Mt");
        auto r3 = build_plan(mk_inputs(1, 192, 9, {1, 1, 1, 1, 1}));  // Nt=9: N_band=2, 7*2=14>=9 empty banks
        CHECK(!r3.ok() && r3.error.find("empty banks") != std::string::npos, "reject tiny-N empty banks");
        auto r4 = build_plan(mk_inputs(8, 1920, 144, {1, 1, 1, 64, 1}));  // huge cb0
        CHECK(!r4.ok() && r4.error.find("L1") != std::string::npos, "reject L1");
        auto r5 = build_plan(mk_inputs(1, 192, 144, {4, 6, 1, 1, 1}));  // 8*24=192>110
        CHECK(!r5.ok() && r5.error.find("cores") != std::string::npos, "reject core-count");
        auto in = mk_inputs(1, 192, 144, {12, 1, 1, 2, 1});
        for (uint32_t x = 0; x < 10; ++x) {
            in.holes.insert(PlanXY{x, 5});
        }
        auto r6 = build_plan(in);
        CHECK(r6.ok(), "plan built with holes");
        if (r6.ok()) {
            bool hits = false;
            for (const auto& cp : r6.plan->cores) {
                if (in.holes.count(cp.coord)) {
                    hits = true;
                }
            }
            CHECK(!hits && all_distinct(*r6.plan), "grid-gap: collision-free, avoids holes");
        }
    }

    std::printf("\n== %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
