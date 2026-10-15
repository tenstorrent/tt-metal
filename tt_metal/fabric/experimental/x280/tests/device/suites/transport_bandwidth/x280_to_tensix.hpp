// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/transport_bandwidth/x280_to_tensix.hpp -- X280 -> Tensix member.
//
// X280 harts 0..harts-1 together store N bursts of `transfer_size` bytes,
// `store_width` bytes per store, N / harts each, each hart through its own
// paired NOC TLB slot at its own range of a passive Tensix L1 destination. The
// descriptor owns the
// LIM+L1 layouts, args pack, harvest, and records; the host test wires
// the fixture and sweep.

#ifndef X280_SUITES_TRANSPORT_BW_X280_TO_TENSIX_HPP
#define X280_SUITES_TRANSPORT_BW_X280_TO_TENSIX_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/regions.hpp"
#include "harness/regions/tensix_l1.hpp"
#include "harness/runner.hpp"
#include "tests/device/suites/transport_bandwidth/records.hpp"

#include "transport_bandwidth_common.h"
#include "x280_hart_protocol.h"
#include "x280_to_tensix.h"
#include "rt/x280_noc_map.h"

#include <gtest/gtest.h>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace x280::suites::transport_bandwidth::x280_to_tensix {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hreg = x280::harness::regions;

constexpr uint32_t kTlbIndex = X280_TLB_PAIRED_BASE;
constexpr uint32_t kDefaultWarmupCount = 128u;
constexpr uint32_t kDefaultTransferCount = 16384u;

struct SweepPoint {
    uint32_t transfer_size;
    uint32_t boundary;
    uint32_t store_width;  // bytes per store: 4 (sw), 8 (sd), 64 (vse32.v)
    const char* label;
    uint32_t harts = 1;  // participating harts, each with its own window and range
};
using Axis = SweepPoint;

// Laid out for every hart, whatever the run's hart count: one section and one
// result per hart, one shared pattern and start counter.
struct X280LimRegions {
    hreg::Region args;
    hreg::Region staging;
    hreg::Region results;
    hreg::Region barrier;

    static X280LimRegions compute(x280::X280Tile& tile) {
        const auto avail = tile.available_lim();
        const hreg::Region w{avail.begin, avail.size()};
        hreg::Arena arena(w);
        X280LimRegions R{};
        R.args = arena.allocate(X280_HART_COUNT * sizeof(transport_bw_x280_to_tensix_section_t));
        R.staging = arena.allocate(TB_X2T_PATTERN_BYTES);
        R.results = arena.allocate(X280_HART_COUNT * sizeof(transport_bw_x280_to_tensix_result_t));
        R.barrier = arena.allocate(sizeof(uint32_t));
        return R;
    }
};

// Hart h writes the h-th dst_bytes range.
struct TensixL1Layout {
    uint32_t dst_bytes;
    uint32_t block_bytes;

    static TensixL1Layout compute() {
        TensixL1Layout L{};
        L.dst_bytes = static_cast<uint32_t>(hreg::round_up(TB_X2T_DST_MAX_BYTES, hreg::kCacheLineBytes));
        L.block_bytes = X280_HART_COUNT * L.dst_bytes;
        return L;
    }
};

// Word k of the destination: the source pattern, repeated every 64 B.
inline uint32_t pattern_word(uint32_t run_nonce, uint32_t k) {
    return run_nonce ^ ((k % (TB_X2T_PATTERN_BYTES / sizeof(uint32_t))) * 0x9E3779B9u);
}

inline uint64_t mint_run_value(uint32_t& run_nonce) {
    static uint32_t sequence = 0;
    ++sequence;
    run_nonce = 0xA00u + sequence;
    return (static_cast<uint64_t>(0xE11Cu) << 48) | (static_cast<uint64_t>(0xB3u) << 40) |
           (static_cast<uint64_t>(0x11u) << 32) | static_cast<uint64_t>(run_nonce);
}

inline void run_axis(
    x280::harness::session::Fixture& f, const x280::harness::RunSpec& /*spec*/, const SweepPoint& point) {
    const uint32_t transfer_size = point.transfer_size;
    const uint32_t boundary = point.boundary;
    const uint32_t store_width = point.store_width;
    const uint32_t harts = point.harts;
    ASSERT_GE(harts, 1u);
    ASSERT_LE(harts, X280_HART_COUNT);
    ASSERT_LE(harts, X280_TLB_PAIRED_COUNT) << "each hart needs its own paired TLB slot";

    // TRANSPORT_BW_N is the run's total; each hart issues its share.
    const uint32_t W = hart_::env_u32("TRANSPORT_BW_W", kDefaultWarmupCount);
    const uint32_t total = hart_::env_u32("TRANSPORT_BW_N", kDefaultTransferCount);
    ASSERT_GT(W, 0u);
    ASSERT_GT(total, 0u);
    ASSERT_EQ(total % harts, 0u) << "TRANSPORT_BW_N=" << total << " does not split evenly over " << harts << " harts";
    const uint32_t N = total / harts;
    const uint64_t logical_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(transfer_size);

    const TensixL1Layout TL = TensixL1Layout::compute();
    const ttm::CoreCoord worker{0, 0};
    const ttm::CoreRangeSet worker_set(ttm::CoreRange(worker, worker));
    auto l1 = hreg::allocate_tile_l1_buffer(f.mesh.get(), worker_set, TL.block_bytes);
    const uint32_t l1_base = l1.l1_tile_base;

    const auto worker_virtual =
        x280::X280Runtime::virtual_core_from_logical_core(*f.mesh, worker, tt::CoreType::TENSIX, ttm::NOC::NOC_0);
    auto dst_addr = [&](uint32_t h) { return l1_base + h * TL.dst_bytes; };
    for (uint32_t h = 0; h < harts; ++h) {
        ASSERT_LE(
            static_cast<uint64_t>(dst_addr(h) + TL.dst_bytes), (static_cast<uint64_t>(dst_addr(h)) | 0x1FFFFFu) + 1ull)
            << "hart " << h << "'s destination range must lie in one 2 MiB window";
    }

    const X280LimRegions XR = X280LimRegions::compute(*f.tile);
    hreg::zero_region(*f.tile, XR.args);
    hreg::zero_region(*f.tile, XR.staging);
    hreg::zero_region(*f.tile, XR.results);
    hreg::zero_region(*f.tile, XR.barrier);
    auto result_addr = [&](uint32_t h) { return XR.results.addr + h * sizeof(transport_bw_x280_to_tensix_result_t); };

    uint32_t run_nonce = 0;
    const uint64_t run_value = mint_run_value(run_nonce);
    std::vector<uint8_t> staging_bytes(XR.staging.bytes, 0);
    for (uint32_t k = 0; k < TB_X2T_PATTERN_BYTES / sizeof(uint32_t); ++k) {
        const uint32_t v = pattern_word(run_nonce, k);
        std::memcpy(staging_bytes.data() + k * sizeof(uint32_t), &v, sizeof(v));
    }
    f.tile->write_available_lim(XR.staging.addr, staging_bytes);

    std::vector<transport_bw_x280_to_tensix_section_t> block(harts);
    for (uint32_t h = 0; h < harts; ++h) {
        auto& section = block[h];
        section = {};
        section.cookie_common = transport_bw_cookie_for(UBENCH_SECTION_COMMON);
        section.common.transfer_size = transfer_size;
        section.common.warmup_count = W;
        section.common.transfer_count = N;
        section.common.boundary = boundary;
        section.common.run_nonce = run_nonce;
        section.cookie_backend = transport_bw_cookie_for(UBENCH_SECTION_BACKEND);
        section.backend.peer_dst_addr = static_cast<uint64_t>(dst_addr(h));
        section.backend.own_staging_addr = XR.staging.addr;
        section.backend.own_result_addr = result_addr(h);
        section.backend.own_barrier_addr = XR.barrier.addr;
        section.backend.peer_noc = worker_virtual.noc();
        section.backend.peer_tlb_index = kTlbIndex + h;
        section.backend.store_width = store_width;
        section.backend.harts = harts;
    }
    // Enqueue takes whole cache lines; the padding holds less than one section.
    static_assert(sizeof(transport_bw_x280_to_tensix_section_t) >= hreg::kCacheLineBytes);
    const uint64_t block_bytes =
        hreg::round_up(harts * sizeof(transport_bw_x280_to_tensix_section_t), hreg::kCacheLineBytes);
    std::vector<uint8_t> block_raw(block_bytes, 0);
    std::memcpy(block_raw.data(), block.data(), harts * sizeof(transport_bw_x280_to_tensix_section_t));
    f.tile->write_available_lim(XR.args.addr, block_raw);

    hexec::X280Launch launch{
        .image_path = hart_::image_path(
            "X280_TRANSPORT_BW_X280_TO_TENSIX_FW_IMAGE", "firmware-transport-bandwidth-x280-to-tensix.bin"),
        .run_value = run_value,
        .args_addr = XR.args.addr,
        .args_bytes = block_bytes,
        .cleanup_timeout = std::chrono::seconds{5},
        .wait_timeout = std::chrono::seconds{60},
    };
    const auto outcome = hexec::run_x280(f.handle(), launch);
    ASSERT_TRUE(outcome.enqueued()) << "enqueue rejected";
    ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    std::vector<transport_bw_x280_to_tensix_result_t> results;
    for (uint32_t h = 0; h < harts; ++h) {
        hexec::expect_window_cleared(*f.tile, kTlbIndex + h);
        results.push_back(hreg::read_struct<transport_bw_x280_to_tensix_result_t>(
            *f.tile, hreg::Region{result_addr(h), sizeof(transport_bw_x280_to_tensix_result_t)}));
        const auto& hr = results.back();
        const auto& result = hr.common;
        SCOPED_TRACE("hart " + std::to_string(h));
        EXPECT_EQ(result.backend_id, TRANSPORT_BW_BACKEND_ID_X280_TO_TENSIX);
        EXPECT_EQ(result.boundary, boundary);
        ASSERT_EQ(result.terminal_state, static_cast<uint32_t>(TRANSPORT_BW_STATE_DONE))
            << "terminal state: " << state_name(result.terminal_state) << " (path_sentinel=0x" << std::hex
            << result.path_sentinel << std::dec << ")";
        EXPECT_EQ(result.transfer_size, transfer_size);
        EXPECT_EQ(result.transfer_count, N);
        EXPECT_EQ(result.warmup_completed, W);
        EXPECT_EQ(result.logical_bytes, logical_bytes);
        ASSERT_GT(result.raw_batch_delta, 0ull);
        ASSERT_GT(result.secondary_batch_delta, 0ull);
        ASSERT_NE(result.raw_batch_delta, UINT64_MAX) << "mtime interval refused on device";
        ASSERT_NE(result.secondary_batch_delta, UINT64_MAX) << "rdcycle interval refused on device";
        ASSERT_EQ(hr.wall_end - hr.wall_start, result.raw_batch_delta) << "mtime endpoints disagree with the delta";
    }

    // Only each hart's first transfer_size bytes are checked, so only those are read.
    const tt_cxy_pair worker_at(f.mesh->get_device(0, 0)->id(), worker_virtual.x(), worker_virtual.y());
    std::vector<uint32_t> range(transfer_size / sizeof(uint32_t));
    for (uint32_t h = 0; h < harts; ++h) {
        x280::harness::cluster().read_core(range.data(), transfer_size, worker_at, dst_addr(h));
        uint32_t mismatches = 0;
        uint32_t first_mismatch = 0;
        for (uint32_t k = 0; k < transfer_size / sizeof(uint32_t); ++k) {
            if (range[k] != pattern_word(run_nonce, k) && mismatches++ == 0) {
                first_mismatch = k;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "hart " << h
                                  << "'s destination words did not receive the staging pattern, first at word "
                                  << first_mismatch << " (read 0x" << std::hex << range[first_mismatch]
                                  << ", expected 0x" << pattern_word(run_nonce, first_mismatch) << std::dec
                                  << "); did the transport drop writes?";
    }

    const auto body = x280::harness::output::record(x280_to_tensix_record(
        results, {.transfer_size = transfer_size, .boundary = boundary, .warmup = W, .samples = N}, store_width));
    x280::harness::output::write_jsonl("transport_bandwidth_x280_to_tensix", body);
    std::printf("%s\n", body.c_str());
}

}  // namespace x280::suites::transport_bandwidth::x280_to_tensix

#endif  // X280_SUITES_TRANSPORT_BW_X280_TO_TENSIX_HPP
