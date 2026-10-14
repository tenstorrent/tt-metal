// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host descriptor for the fixed next-hop direction-table lookup.

#ifndef X280_SUITES_PIPELINE_STAGE_ROUTE_COMPUTE_HPP
#define X280_SUITES_PIPELINE_STAGE_ROUTE_COMPUTE_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/regions.hpp"
#include "harness/runner.hpp"

#include "fabric_shadow_header.hpp"
#include "pipeline_stage_common.h"
#include "route_compute_args.h"
#include "rt/x280_pinned_input.h"
#include "suites/pipeline_stage/records.hpp"
#include "suites/pipeline_stage/route_oracle.hpp"
#include "suites/pipeline_stage/suite_common.hpp"

#include <hostdevcommon/fabric_common.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace x280::suites::pipeline_stage::route_compute {

namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hout = x280::harness::output;
namespace hreg = x280::harness::regions;
using x280::suites::pipeline_stage::common::instrumentation_name;
using x280::suites::pipeline_stage::common::state_name;

constexpr uint32_t kDefaultWarmupCount = 128u;
constexpr uint32_t kDefaultTimedCount = 16384u;

struct SweepPoint {
    uint32_t instrumentation;
    const char* label;
};
using Axis = SweepPoint;

struct LimRegions {
    hreg::Region window;
    hreg::Region args;
    hreg::Region staged_input;
    hreg::Region result;
    hreg::Region samples;

    static LimRegions compute(x280::X280Tile& tile) {
        const auto available = tile.available_lim();
        const hreg::Region w{available.begin, available.size()};
        hreg::Arena arena(w);
        LimRegions regions{};
        regions.window = w;
        regions.args = arena.allocate(PIPELINE_STAGE_ROUTE_COMPUTE_ARGS_BYTES);
        regions.staged_input = arena.allocate(X280_PINNED_INPUT_SPAN);
        regions.result = arena.allocate(sizeof(pipeline_stage_result_t));
        regions.samples = arena.allocate(sizeof(pipeline_stage_samples_t));
        return regions;
    }
};

inline uint64_t mint_run_value(uint32_t& run_nonce) {
    static uint32_t sequence = 0u;
    ++sequence;
    run_nonce = 0xC00u + sequence;
    return (static_cast<uint64_t>(0xE11Cu) << 48u) | (static_cast<uint64_t>(0xB3u) << 40u) |
           (static_cast<uint64_t>(0x16u) << 32u) | run_nonce;
}

inline void run_axis(
    x280::harness::session::Fixture& fixture, const x280::harness::RunSpec& spec, const SweepPoint& point) {
    const uint32_t warmup = hart_::env_u32("PIPELINE_STAGE_W", kDefaultWarmupCount);
    const uint32_t timed = common::timed_count_from_env(point.instrumentation, kDefaultTimedCount);
    ASSERT_GT(warmup, 0u);
    ASSERT_GT(timed, 0u);

    const LimRegions regions = LimRegions::compute(*fixture.tile);
    hreg::zero_region(*fixture.tile, regions.result);
    if (point.instrumentation == PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE) {
        hreg::zero_region(*fixture.tile, regions.samples);
    }

    const std::vector<uint8_t> input = build_input_image();
    fixture.tile->write_available_lim(regions.staged_input.addr, input);

    uint32_t run_nonce = 0u;
    const uint64_t run_value = mint_run_value(run_nonce);

    UBENCH_SECTION_STORAGE(pipeline_stage_route_compute_args_block_t, args) store{};
    fill_section(
        store.args.section,
        warmup,
        timed,
        point.instrumentation,
        run_nonce,
        regions.staged_input.addr,
        regions.result.addr,
        point.instrumentation == PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE ? regions.samples.addr : 0u,
        PIPELINE_STAGE_LOOKUP_FIXTURE_V1);
    fill_layout(store.args.layout, regions.window, regions.staged_input.addr);
    hreg::write_args(*fixture.tile, regions.args, store.args);

    const hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_PIPELINE_STAGE_ROUTE_COMPUTE_FW_IMAGE"),
        .run_value = run_value,
        .args_addr = regions.args.addr,
        .args_bytes = regions.args.bytes,
        .cleanup_timeout = std::chrono::seconds{5},
        .wait_timeout = std::chrono::seconds{60},
    };
    const auto outcome = hexec::run_x280(fixture.handle(), launch);
    ASSERT_TRUE(outcome.enqueued()) << "enqueue rejected";
    ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    const auto result = hreg::read_struct<pipeline_stage_result_t>(*fixture.tile, regions.result);
    EXPECT_EQ(result.backend_id, PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_COMPUTE);
    ASSERT_EQ(result.terminal_state, static_cast<uint32_t>(PIPELINE_STAGE_STATE_DONE))
        << "terminal state " << state_name(result.terminal_state) << ", verification 0x" << std::hex
        << result.verify_sentinel << ", policy_master=" << std::dec << result.workload_size << ", policy_observed=0x"
        << std::hex << result.secondary_batch_delta << std::dec;
    EXPECT_EQ(pipeline_stage_outcome(static_cast<pipeline_stage_state_t>(result.terminal_state)), UBENCH_OUTCOME_OK);
    EXPECT_EQ(result.packet_header_bytes, FS_PACKET_HEADER_BYTES);
    EXPECT_EQ(result.instrumentation, point.instrumentation);
    EXPECT_EQ(result.workload_size, PIPELINE_STAGE_LOOKUP_HEADER_COUNT);
    EXPECT_EQ(result.warmup_completed, warmup);
    EXPECT_EQ(result.timed_count, timed);
    ASSERT_EQ(result.items_completed, static_cast<uint64_t>(timed));
    EXPECT_EQ(result.stage_witness, expected_witness(run_nonce, timed));
    ASSERT_EQ(result.verify_sentinel, PIPELINE_STAGE_VERIFY_MATCH);
    ASSERT_GT(result.raw_batch_delta, 0u);
    ASSERT_GT(result.secondary_batch_delta, 0u);

    pipeline_stage_samples_t samples{};
    if (point.instrumentation == PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE) {
        samples = hreg::read_struct<pipeline_stage_samples_t>(*fixture.tile, regions.samples);
        ASSERT_EQ(samples.samples_completed, timed);
    }

    std::printf(
        "\n[pipeline_stage route_compute] packet=%u B route_capacity=%u "
        "fixture=direction_table_v1 headers=%u "
        "prepared_base=0x%llx "
        "prepared_bytes=%u wayenable=%u pinned_way=%u waymask_bits=%llu "
        "instr=%s W=%u N=%u "
        "raw_batch_delta=%llu secondary_batch_delta=%llu\n",
        FS_PACKET_HEADER_BYTES,
        FS_ROUTE_BUFFER_CAPACITY,
        PIPELINE_STAGE_LOOKUP_HEADER_COUNT,
        static_cast<unsigned long long>(X280_PINNED_INPUT_BASE),
        static_cast<unsigned>(X280_PINNED_INPUT_SPAN),
        X280_PINNED_INPUT_WAYENABLE,
        X280_PINNED_INPUT_RESERVED_WAY,
        static_cast<unsigned long long>(X280_PINNED_INPUT_PREPARED_BITS),
        instrumentation_name(point.instrumentation),
        warmup,
        timed,
        static_cast<unsigned long long>(result.raw_batch_delta),
        static_cast<unsigned long long>(result.secondary_batch_delta));

    hout::write_jsonl(
        spec.suite_label,
        hout::record(common::x280_route_compute_record({
            .result = result,
            .run_nonce = run_nonce,
            .warmup = warmup,
            .timed = timed,
            .instrumentation = point.instrumentation,
            .samples = &samples,
        })));

    EXPECT_LT(result.raw_batch_delta, 5u * hmeas::kX280MtimeHz)
        << "batch took longer than 5 s of mtime; something stalled";
}

}  // namespace x280::suites::pipeline_stage::route_compute

#endif
