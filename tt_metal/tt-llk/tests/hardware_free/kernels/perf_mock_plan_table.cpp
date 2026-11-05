// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side table test of the Wormhole isolate mocks' MOP stream plans (helpers/include/perf_mock_plan.h); no hardware.
#include <cstdint>
#include <cstdio>

#include "perf_mock_plan.h"

using namespace llk_perf_mock;

static int failures = 0;

#define CHECK(cond, ...)                                               \
    do                                                                 \
    {                                                                  \
        if (!(cond))                                                   \
        {                                                              \
            std::printf("FAIL %s:%d %s: ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                  \
            std::printf("\n");                                         \
            ++failures;                                                \
        }                                                              \
    } while (0)

struct SegmentRow
{
    const char* name;
    bool allow_split;
    std::uint32_t burst, run, periods;
    bool fits, whole, split;
    std::uint32_t buffered, inner, outer, mops;
};

// Expected values worked out by hand from the replay buffer's 16 handshakes, the 127 loop limit and 32 MOPs
static const SegmentRow SEGMENT_ROWS[] = {
    // name                         split burst run periods  fits  whole split buf in out mops
    {"period 16 producer", false, 1, 15, 1, true, true, false, 16, 1, 1, 1},
    {"period 16 consumer", true, 1, 15, 1, true, true, false, 16, 1, 1, 1},
    {"period 17 producer", false, 1, 16, 1, true, false, false, 1, 16, 1, 1},
    {"period 17 consumer splits", true, 1, 16, 1, true, true, true, 16, 1, 1, 1},
    {"period 18 producer", false, 1, 17, 1, true, false, false, 1, 17, 1, 1},
    {"period 18 consumer splits", true, 1, 17, 1, true, true, true, 16, 1, 1, 1},
    {"period 18 burst 2 consumer", true, 2, 16, 1, true, true, true, 16, 1, 1, 1},
    {"period 18 burst 2 producer", false, 2, 16, 1, true, false, false, 2, 16, 1, 1},
    {"period 32 burst 2 no split", true, 2, 30, 1, true, false, false, 2, 30, 1, 1},
    {"run 127", true, 1, 127, 1, true, false, false, 1, 127, 1, 1},
    {"run 128 does not fit", true, 1, 128, 1, false, false, false, 0, 0, 0, 0},
    {"run 127 many periods", false, 1, 127, 40, true, false, false, 1, 127, 40, 1},
    {"a_count 31 period 32", true, 1, 31, 64, true, false, false, 1, 31, 64, 1},
    {"a_count 32 period 33", true, 1, 32, 64, true, false, false, 1, 32, 64, 1},
    {"1024 whole periods", false, 1, 15, 1024, true, true, false, 16, 64, 16, 1},
    {"4096 whole periods", false, 1, 15, 4096, true, true, false, 16, 64, 64, 1},
    {"1024 split periods", true, 2, 15, 1024, true, true, true, 16, 64, 16, 1},
    {"32 MOPs fit", false, 1, 31, 2048, true, false, false, 1, 31, 64, 32},
    {"33 MOPs do not", false, 1, 31, 2112, false, false, false, 0, 0, 0, 0},
    {"zero periods", true, 2, 15, 0, false, false, false, 0, 0, 0, 0},
    {"zero burst", false, 0, 5, 3, false, false, false, 0, 0, 0, 0},
    {"zero run", false, 1, 0, 3, false, false, false, 0, 0, 0, 0},
    {"burst over the buffer", false, 17, 1, 3, false, false, false, 0, 0, 0, 0},
};

static void check_segment_rows()
{
    for (const SegmentRow& r : SEGMENT_ROWS)
    {
        const SegmentPlan p = plan_segments(r.allow_split, r.burst, r.run, r.periods);
        CHECK(p.fits == r.fits, "%s: fits %d", r.name, p.fits);
        if (!r.fits)
        {
            continue;
        }
        CHECK(p.whole == r.whole && p.split == r.split, "%s: whole %d split %d", r.name, p.whole, p.split);
        CHECK(p.buffered == r.buffered, "%s: buffered %u", r.name, p.buffered);
        CHECK(p.inner == r.inner && p.outer == r.outer && p.mops == r.mops, "%s: %u x %u x %u MOPs", r.name, p.inner, p.outer, p.mops);
    }
}

// Every plan that fits runs exactly the requested periods within the hardware's limits
static void check_segment_invariants()
{
    const std::uint32_t runs[]    = {1, 2, 14, 15, 16, 17, 18, 30, 31, 32, 33, 126, 127, 128, 129};
    const std::uint32_t periods[] = {0, 1, 2, 3, 15, 16, 17, 126, 127, 128, 129, 130, 255, 256, 384, 1000, 1024, 3072, 4096, 4097, 16129, 65536, 100000};
    for (bool split : {false, true})
    {
        for (std::uint32_t burst : {1u, 2u, 3u, 15u, 16u, 17u})
        {
            for (std::uint32_t run : runs)
            {
                for (std::uint32_t n : periods)
                {
                    const SegmentPlan p = plan_segments(split, burst, run, n);
                    if (!p.fits)
                    {
                        continue;
                    }
                    CHECK(n != 0 && burst <= REPLAY_HANDSHAKES, "fits with burst %u periods %u", burst, n);
                    CHECK(p.inner >= 1 && p.inner <= MOP_LOOP_MAX, "inner %u (b%u r%u n%u)", p.inner, burst, run, n);
                    CHECK(p.outer >= 1 && p.outer <= MOP_LOOP_MAX, "outer %u (b%u r%u n%u)", p.outer, burst, run, n);
                    CHECK(p.mops >= 1 && p.mops <= MAX_MOPS, "mops %u (b%u r%u n%u)", p.mops, burst, run, n);
                    // a whole plan's inner iteration is one period; else the inner loop is the run and an outer iteration one period
                    const std::uint64_t ran = p.whole ? std::uint64_t {p.inner} * p.outer * p.mops : std::uint64_t {p.outer} * p.mops;
                    CHECK(ran == n, "ran %llu of %u periods (b%u r%u)", static_cast<unsigned long long>(ran), n, burst, run);
                    CHECK(p.whole || p.inner == run, "inner %u of run %u", p.inner, run);
                    CHECK(!p.split || (split && p.whole), "split only for the consumer, and whole");
                    CHECK(p.whole == (burst + run <= REPLAY_HANDSHAKES || p.split), "whole %d at period %u", p.whole, burst + run);
                    CHECK(p.buffered == (p.whole ? (p.split ? REPLAY_HANDSHAKES : burst + run) : burst), "buffered %u", p.buffered);
                    CHECK(p.buffered <= REPLAY_HANDSHAKES, "buffered %u over the replay buffer", p.buffered);
                }
            }
        }
    }
}

static void check_mop_loops()
{
    struct
    {
        std::uint32_t n, loops;
    } rows[] = {{1, 1}, {127, 127}, {128, 64}, {129, 3}, {384, 64}, {1000, 8}, {1024, 64}, {3072, 64}, {16129, 1}};

    for (const auto& r : rows)
    {
        CHECK(mop_loops(r.n) == r.loops, "mop_loops(%u) = %u", r.n, mop_loops(r.n));
    }
    for (std::uint32_t n = 1; n <= 20000; n++)
    {
        const std::uint32_t loops = mop_loops(n);
        CHECK(loops >= 1 && loops <= MOP_LOOP_MAX && n % loops == 0, "mop_loops(%u) = %u", n, loops);
    }
}

struct MatmulRow
{
    const char* name;
    bool produce;
    std::uint32_t loop_factor, rt, kt, ct, blocks;
    Route route;
    std::uint32_t period, pattern, steps;
};

static const MatmulRow MATMUL_ROWS[] = {
    // name                          produce lf  rt kt ct blocks route            period pattern steps
    {"zero loop factor", true, 0, 1, 1, 1, 1, Route::NONE, 0, 0, 0},
    {"zero rt", false, 4, 0, 1, 1, 1, Route::NONE, 0, 0, 0},
    {"zero kt", true, 4, 1, 0, 1, 1, Route::NONE, 0, 0, 0},
    {"zero ct", true, 4, 1, 1, 0, 1, Route::NONE, 0, 0, 0},
    {"zero blocks", true, 4, 1, 1, 1, 0, Route::NONE, 0, 0, 0},
    {"1x1 producer", true, 1, 1, 1, 1, 1, Route::SEGMENTS, 2, 0x1, 1},
    {"1x1 consumer", false, 1, 1, 1, 1, 1, Route::SEGMENTS, 2, 0x2, 1},
    {"loop factor 1024, kt 4", true, 1024, 2, 4, 8, 1, Route::SEGMENTS, 10, 0, 4096},
    {"odd reuse 3 producer", true, 1, 3, 1, 3, 1, Route::STREAM, 9, 0x23, 1},
    {"odd reuse 3 consumer", false, 1, 3, 1, 3, 1, Route::STREAM, 9, 0x118, 1},
    {"odd reuse 3 long reload", true, 1, 3, 1, 20, 1, Route::PER_MOP, 43, 0, 1},
    {"reload 128 over a MOP's loop", true, 1, 2, 1, 128, 1, Route::PER_MOP, 130, 0, 1},
    {"reuse 2 reload 15 producer", true, 1, 2, 1, 15, 1, Route::SEGMENTS, 17, 0x3, 1},
};

static void check_matmul_rows()
{
    for (const MatmulRow& r : MATMUL_ROWS)
    {
        const MatmulPlan p = plan_matmul(r.produce, r.loop_factor, r.rt, r.kt, r.ct, r.blocks);
        CHECK(p.route == r.route, "%s: route %d", r.name, static_cast<int>(p.route));
        if (r.route == Route::NONE)
        {
            continue;
        }
        CHECK(p.period == r.period, "%s: period %u", r.name, p.period);
        CHECK(p.steps == r.steps, "%s: steps %u", r.name, p.steps);
        if (r.pattern != 0)
        {
            CHECK(p.pattern == r.pattern, "%s: pattern 0x%x", r.name, p.pattern);
        }
        if (r.route == Route::SEGMENTS)
        {
            CHECK(p.seg.fits, "%s: segments plan", r.name);
        }
    }
}

struct SrcbRow
{
    const char* name;
    bool produce;
    std::uint32_t a_count, periods;
    Route route;
    std::uint32_t pattern;
};

static const SrcbRow SRCB_ROWS[] = {
    // name                     produce a_count periods route            pattern
    {"a_count 31 whole run", true, 31, 5, Route::SEGMENTS, 0},
    {"a_count 32 whole run", false, 32, 5, Route::SEGMENTS, 0},
    {"a_count 31 too many MOPs producer", true, 31, 2112, Route::STREAM, 1},
    {"a_count 31 too many MOPs consumer", false, 31, 2112, Route::STREAM, 1u << 31},
    {"a_count 32 too many MOPs", true, 32, 2112, Route::PER_MOP, 0},
    {"a_count 128", false, 128, 3, Route::PER_MOP, 0},
    {"a_count 0", true, 0, 3, Route::STREAM, 1},
    {"zero periods", true, 4, 0, Route::STREAM, 1},
};

static void check_srcb_rows()
{
    for (const SrcbRow& r : SRCB_ROWS)
    {
        const SrcbReusePlan p = plan_srcb_reuse(r.produce, r.a_count, r.periods);
        CHECK(p.route == r.route, "%s: route %d", r.name, static_cast<int>(p.route));
        if (r.route == Route::STREAM)
        {
            CHECK(p.pattern == r.pattern, "%s: pattern 0x%x", r.name, p.pattern);
        }
    }
}

int main()
{
    check_segment_rows();
    check_segment_invariants();
    check_mop_loops();
    check_matmul_rows();
    check_srcb_rows();
    std::printf(failures == 0 ? "perf mock plan table: OK\n" : "perf mock plan table: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
