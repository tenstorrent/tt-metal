// Standalone offline unit test for the Regime-A host planner (no tt_metal / no device).
// Verifies build_plan() against the frozen golden oracle (GOLDEN_PARITY_SUITE.md) plus tail,
// rejection, and collision-free-placement cases the productization plan requires.
//
// Build + run:
//   g++ -std=c++17 -I ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device \
//       tools/mm_sweep/regime_a_plan_test.cpp -o /tmp/ra_plan_test && /tmp/ra_plan_test

#include "regime_a_matmul_plan.hpp"

#include <cmath>
#include <cstdio>
#include <string>

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

#define CHECK_EQ(a, b, msg)                                      \
    do {                                                         \
        auto _a = (a);                                           \
        auto _b = (b);                                           \
        if (_a == _b) {                                          \
            ++g_pass;                                            \
        } else {                                                 \
            ++g_fail;                                            \
            std::printf(                                         \
                "  FAIL: %s  got %lld expected %lld  (%s:%d)\n", \
                msg,                                             \
                (long long)_a,                                   \
                (long long)_b,                                   \
                __FILE__,                                        \
                __LINE__);                                       \
        }                                                        \
    } while (0)

// Synthetic BH-like grid: 11x10, bank anchors in row 0 (NOC0) and row 9 (NOC1).
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

// Verify every core coord is distinct (collision-free placement).
static bool all_distinct(const ExecutionPlan& p) {
    std::set<PlanXY> s;
    for (const auto& cp : p.cores) {
        if (!s.insert(cp.coord).second) {
            return false;
        }
    }
    return s.size() == p.cores.size();
}

// One golden-oracle row: shape, config, expected cores / in1 shard / L1(KB).
struct Golden {
    const char* label;
    uint32_t Mt, Kt, Nt;
    RegimeAConfig cfg;
    uint32_t exp_cores;
    uint32_t exp_shard_rows, exp_shard_cols;
    uint32_t exp_l1_kb;
};

int main() {
    std::printf("== Regime-A host planner offline tests ==\n");

    // ---- Golden parity oracle (from GOLDEN_PARITY_SUITE.md) ----
    // cfg = {k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb}
    Golden golden[] = {
        {"Mt1 base 32x6144x4608", 1, 192, 144, {12, 1, 1, 2, 1}, 96, 192, 18, 60},
        {"Mt2 base 64x6144x4608", 2, 192, 144, {6, 1, 1, 4, 2}, 48, 192, 18, 240},
        {"Mt4 base 128x6144x4608", 4, 192, 144, {12, 1, 1, 2, 1}, 96, 192, 18, 192},
        {"Mt8 base 256x6144x4608", 8, 192, 144, {12, 1, 1, 2, 1}, 96, 192, 18, 368},
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
        CHECK_EQ(p.in1_shard_rows, go.exp_shard_rows, "in1 shard rows (Kt_s)");
        CHECK_EQ(p.in1_shard_cols, go.exp_shard_cols, "in1 shard cols (Nt_s/8)");
        CHECK_EQ(p.cb.l1_bytes / 1024u, go.exp_l1_kb, "L1 KB");
        CHECK(all_distinct(p), "collision-free placement");
        // divisible shapes: no padding waste
        CHECK(std::abs(p.geo.waste_k) < 1e-9 && std::abs(p.geo.waste_n) < 1e-9, "zero waste (divisible)");
    }

    // ---- Non-divisible tail case: 32x6080x4640 -> Kt=190, Nt=145 ----
    {
        std::printf("[tail] Mt1 non-divis 32x6080x4640 (Kt=190 Nt=145)\n");
        auto r = build_plan(mk_inputs(1, 190, 145, {12, 1, 1, 2, 1}));
        CHECK(r.ok(), "plan built");
        const auto& p = *r.plan;
        // Kt_local = rup(cdiv(190,12)=16, 16) = 16 ; Kt_s = 192 -> waste ~1.05%
        CHECK_EQ(p.geo.Kt_local, 16u, "Kt_local");
        CHECK_EQ(p.geo.Kt_s, 192u, "Kt_s");
        CHECK(std::abs(p.geo.waste_k - (192.0 / 190 - 1)) < 1e-9, "waste_k ~1.05%");
        // N_band = cdiv(145,8) = 19 ; Nt_s = 152 -> waste ~4.83% ; shard cols 19
        CHECK_EQ(p.geo.N_band, 19u, "N_band");
        CHECK_EQ(p.in1_shard_cols, 19u, "in1 shard cols");
        CHECK(std::abs(p.geo.waste_n - (152.0 / 145 - 1)) < 1e-9, "waste_n ~4.83%");
        // valid extents: bank 7 owns N tiles [7*19,8*19)=[133,152); logical Nt=145 so valid_n = 145-133 = 12
        // find the core with bank==7, slice==0
        for (const auto& cp : p.cores) {
            if (cp.bank == 7 && cp.slice == 0) {
                CHECK_EQ(cp.valid_n, 12u, "bank7 valid_n clamps to logical Nt");
            }
            if (cp.bank == 0 && cp.slice == 0) {
                CHECK_EQ(cp.valid_n, 19u, "bank0 valid_n full");
                CHECK_EQ(cp.valid_m, 1u, "valid_m");
                CHECK_EQ(cp.valid_k, 16u, "bank0 slice0 valid_k (16 of 190)");
            }
        }
    }

    // ---- Ring + reduction link correctness (Pk=3, Ns=2, Sm=1 -> exercise all three factors' indexing) ----
    {
        std::printf("[links] Pk3 Ns2 Sm1 32x6144x4608\n");
        RegimeAConfig cfg{3, 2, 1, 2, 1};
        auto r = build_plan(mk_inputs(1, 192, 144, cfg));
        CHECK(r.ok(), "plan built");
        const auto& p = *r.plan;
        CHECK_EQ(p.geo.num_cores, 8u * 3u * 2u * 1u, "num_cores = 48");
        CHECK_EQ(p.geo.mfac, 2u, "mfac = Ns*Sm = 2");
        // reduction stride check: for a bottom core (kk=0), red_next must be +mfac and land on kk=1.
        for (uint32_t i = 0; i < p.cores.size(); ++i) {
            const auto& cp = p.cores[i];
            if (cp.is_bottom) {
                CHECK(!cp.is_top || p.geo.num_cores == 8u * 1u, "bottom not top when Pk>1");
                const auto& nx = p.cores[cp.red_next_idx];
                CHECK_EQ(nx.kk, cp.kk + 1u, "red_next is next k-slice");
                CHECK_EQ(nx.nn, cp.nn, "red_next same n-slice");
                CHECK_EQ(nx.bank, cp.bank, "red_next same bank");
            }
            if (cp.is_top) {
                CHECK_EQ(cp.kk, 3u - 1u, "top kk == Pk-1");
                CHECK_EQ(cp.red_next_idx, i, "top red_next == self");
            }
        }
        // ring: each ring (fixed slice j) has exactly positions 0..7 across the 8 banks, cyclic next.
        for (uint32_t j = 0; j < p.geo.preaders; ++j) {
            std::set<uint32_t> positions;
            for (uint32_t b = 0; b < 8; ++b) {
                const auto& cp = p.cores[b * p.geo.preaders + j];
                positions.insert(cp.ring_pos);
                const auto& nxt = p.cores[cp.ring_next_idx];
                CHECK_EQ(nxt.ring_pos, (cp.ring_pos + 1u) % 8u, "ring_next pos+1");
                const auto& prv = p.cores[cp.ring_prev_idx];
                CHECK_EQ(prv.ring_pos, (cp.ring_pos + 7u) % 8u, "ring_prev pos-1");
            }
            CHECK_EQ(positions.size(), (size_t)8, "ring has 8 distinct positions");
        }
    }

    // ---- Pk=1: no reduction (every core is bottom AND top), cb7 not allocated ----
    {
        std::printf("[Pk1] no-reduction ring\n");
        auto r = build_plan(mk_inputs(1, 192, 144, {1, 1, 1, 4, 6}));
        CHECK(r.ok(), "plan built");
        const auto& p = *r.plan;
        CHECK_EQ(p.geo.num_cores, 8u, "Pk1 -> 8 cores");
        CHECK_EQ(p.cb.cb7_tiles, 0u, "cb7 not allocated when Pk==1");
        for (const auto& cp : p.cores) {
            CHECK(cp.is_bottom && cp.is_top, "Pk1 core is both bottom and top");
            CHECK_EQ(cp.red_next_idx, (uint32_t)(&cp - &p.cores[0]), "Pk1 red_next self");
        }
    }

    // ---- L1 rejection: a deep-K single-slice config that blows cb0 ----
    {
        std::printf("[reject] L1 over budget\n");
        // Pk=1, huge Kt, kb large -> cb0 = M_block*Kt_local enormous.
        auto r = build_plan(mk_inputs(8, 1920, 144, {1, 1, 1, 64, 1}));
        CHECK(!r.ok(), "rejected");
        CHECK(r.error.find("L1") != std::string::npos, "L1 error message");
    }

    // ---- Core-count rejection: needs > available cores ----
    {
        std::printf("[reject] core-count over grid\n");
        // 8 * Pk * Ns * Sm with Pk=4 Ns=6 -> 8*24 = 192 > 110
        auto r = build_plan(mk_inputs(1, 192, 4608 / 32, {4, 6, 1, 1, 1}));
        CHECK(!r.ok(), "rejected");
        CHECK(r.error.find("cores") != std::string::npos, "core-count error message");
    }

    // ---- Grid-gap placement: inject holes, still collision-free and avoids holes ----
    {
        std::printf("[grid-gap] placement avoids holes\n");
        auto in = mk_inputs(1, 192, 144, {12, 1, 1, 2, 1});  // 96 cores in 110-cell grid
        // punch 10 holes (leaving exactly 100 usable >= 96)
        for (uint32_t x = 0; x < 10; ++x) {
            in.holes.insert(PlanXY{x, 5});
        }
        auto r = build_plan(in);
        CHECK(r.ok(), "plan built with holes");
        const auto& p = *r.plan;
        CHECK(all_distinct(p), "collision-free with holes");
        bool hits_hole = false;
        for (const auto& cp : p.cores) {
            if (in.holes.count(cp.coord)) {
                hits_hole = true;
            }
        }
        CHECK(!hits_hole, "no core placed on a hole");
    }

    std::printf("\n== %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
