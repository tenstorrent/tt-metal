// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/pipeline_stage/route_forward.hpp -- ID 7, the router-side forward.
//
// ID 7 measures DECIDE: the hop decode and routing_fields update of the
// pinned first header, with no packet issued.

#ifndef X280_SUITES_PIPELINE_STAGE_ROUTE_FORWARD_HPP
#define X280_SUITES_PIPELINE_STAGE_ROUTE_FORWARD_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/regions.hpp"
#include "harness/runner.hpp"

#include "pipeline_stage_common.h"
#include "route_forward_args.h"
#include "rt/x280_pinned_input.h"
#include "suites/pipeline_stage/records.hpp"
#include "suites/pipeline_stage/suite_common.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

namespace x280::suites::pipeline_stage::route_forward {

namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hout = x280::harness::output;
namespace hreg = x280::harness::regions;
using x280::suites::pipeline_stage::common::instrumentation_name;
using x280::suites::pipeline_stage::common::kHopCount;
using x280::suites::pipeline_stage::common::state_name;

constexpr uint32_t kDefaultWarmupCount = 128u;
constexpr uint32_t kDefaultTimedCount = 16384u;

struct SweepPoint {
    const char* label;
};
using Axis = SweepPoint;

struct LimRegions {
    hreg::Region window;
    hreg::Region args;
    hreg::Region result;

    static LimRegions compute(x280::X280Tile& tile) {
        const auto avail = tile.available_lim();
        const hreg::Region w{avail.begin, avail.size()};
        hreg::Arena arena(w);
        LimRegions R{};
        R.window = w;
        R.args = arena.allocate(PIPELINE_STAGE_ROUTE_FORWARD_ARGS_BYTES);
        R.result = arena.allocate(sizeof(pipeline_stage_result_t));
        return R;
    }
};

inline uint64_t mint_run_value(uint32_t& run_nonce) {
    static uint32_t sequence = 0;
    ++sequence;
    run_nonce = 0xD00u + sequence;
    return (static_cast<uint64_t>(0xE11Cu) << 48) | (static_cast<uint64_t>(0xB3u) << 40) |
           (static_cast<uint64_t>(0x17u) << 32) | static_cast<uint64_t>(run_nonce);
}

inline void run_axis(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, const SweepPoint&) {
    constexpr uint32_t mode = PIPELINE_STAGE_INSTRUMENTATION_OFF;

    const uint32_t W = hart_::env_u32("PIPELINE_STAGE_W", kDefaultWarmupCount);
    const uint32_t N = common::timed_count_from_env(mode, kDefaultTimedCount);
    ASSERT_GT(W, 0u);
    ASSERT_GT(N, 0u);

    const LimRegions R = LimRegions::compute(*f.tile);
    hreg::zero_region(*f.tile, R.result);

    uint32_t run_nonce = 0;
    const uint64_t run_value = mint_run_value(run_nonce);

    UBENCH_SECTION_STORAGE(pipeline_stage_route_forward_args_block_t, args) store{};
    fill_section(store.args.section, W, N, mode, run_nonce, kHopCount, R.result.addr, 0ull);
    fill_layout(store.args.layout, R.window, X280_PINNED_INPUT_FIRST_HEADER_BASE);
    hreg::write_args(*f.tile, R.args, store.args);

    hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_PIPELINE_STAGE_ROUTE_FORWARD_FW_IMAGE"),
        .run_value = run_value,
        .args_addr = R.args.addr,
        .args_bytes = R.args.bytes,
        .cleanup_timeout = std::chrono::seconds{5},
        .wait_timeout = std::chrono::seconds{60},
    };
    const auto outcome = hexec::run_x280(f.handle(), launch);
    ASSERT_TRUE(outcome.enqueued()) << "enqueue rejected";
    ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    const auto r = hreg::read_struct<pipeline_stage_result_t>(*f.tile, R.result);

    EXPECT_EQ(r.backend_id, PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_FORWARD);
    ASSERT_EQ(r.terminal_state, static_cast<uint32_t>(PIPELINE_STAGE_STATE_DONE))
        << "terminal state: " << state_name(r.terminal_state) << "; refusal code 0x" << std::hex << r.verify_sentinel
        << std::dec << " (0xA0 sanity_check, 0xBxxx prepared policy)"
        << "; policy_master=" << r.workload_size << "; policy_observed=0x" << std::hex << r.secondary_batch_delta
        << std::dec;
    EXPECT_EQ(r.packet_header_bytes, FS_PACKET_HEADER_BYTES);
    EXPECT_EQ(r.instrumentation, mode);
    EXPECT_EQ(r.workload_size, kHopCount);

    /* DECIDE accumulates decoded directions.
     * The expectation mirrors seed_route() and decode_hop(). */
    {
        static_assert(kHopCount == PIPELINE_STAGE_DIR_COUNT);
        static constexpr uint32_t kSeededDirs[PIPELINE_STAGE_DIR_COUNT] = {
            PIPELINE_STAGE_DIR_SOUTH, PIPELINE_STAGE_DIR_EAST, PIPELINE_STAGE_DIR_NORTH, PIPELINE_STAGE_DIR_WEST};
        uint32_t expected_witness = 0u;
        for (uint32_t i = 0u; i < N; ++i) {
            expected_witness += kSeededDirs[i % kHopCount];
        }
        EXPECT_EQ(r.stage_witness, expected_witness) << "decide witness mismatch";
    }
    EXPECT_EQ(r.warmup_completed, W);
    EXPECT_EQ(r.timed_count, N);
    ASSERT_EQ(r.items_completed, static_cast<uint64_t>(N));
    ASSERT_GT(r.raw_batch_delta, 0ull);

    std::printf(
        "\n[pipeline_stage route_forward] packet=%u B route_capacity=%u "
        "source_placement=pinned_zero_device source_base=0x%llx "
        "source_bytes=%llu wayenable=%u pinned_way=%u "
        "waymask_bits=%llu instr=%s hops=%u "
        "boundary=decision subject=router_forward_decision "
        "W=%u N=%u  "
        "raw_batch_delta=%llu secondary_batch_delta=%llu\n",
        FS_PACKET_HEADER_BYTES,
        FS_ROUTE_BUFFER_CAPACITY,
        static_cast<unsigned long long>(X280_PINNED_INPUT_FIRST_HEADER_BASE),
        static_cast<unsigned long long>(X280_PINNED_INPUT_HEADER_BYTES),
        X280_PINNED_INPUT_WAYENABLE,
        X280_PINNED_INPUT_RESERVED_WAY,
        static_cast<unsigned long long>(X280_PINNED_INPUT_PREPARED_BITS),
        instrumentation_name(mode),
        kHopCount,
        W,
        N,
        static_cast<unsigned long long>(r.raw_batch_delta),
        static_cast<unsigned long long>(r.secondary_batch_delta));

    hout::write_jsonl(
        spec.suite_label,
        hout::record(common::x280_route_forward_record({
            .result = r,
            .run_nonce = run_nonce,
            .warmup = W,
            .timed = N,
            .instrumentation = mode,
        })));

    EXPECT_LT(r.raw_batch_delta, 5u * hmeas::kX280MtimeHz) << "batch took longer than 5 s of mtime; something stalled";
}

}  // namespace x280::suites::pipeline_stage::route_forward

#endif  // X280_SUITES_PIPELINE_STAGE_ROUTE_FORWARD_HPP
