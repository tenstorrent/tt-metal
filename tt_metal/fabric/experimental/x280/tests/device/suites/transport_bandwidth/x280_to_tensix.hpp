// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/transport_bandwidth/x280_to_tensix.hpp -- X280 -> Tensix member.
//
// X280 hart 0 stores N bursts of `transfer_size` bytes, `store_width` bytes
// per store, through one paired NOC TLB slot at a passive Tensix L1
// destination. The descriptor owns the
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
};
using Axis = SweepPoint;

struct X280LimRegions {
    hreg::Region args;
    hreg::Region staging;
    hreg::Region result;

    static X280LimRegions compute(x280::X280Tile& tile) {
        const auto avail = tile.available_lim();
        const hreg::Region w{avail.begin, avail.size()};
        hreg::Arena arena(w);
        X280LimRegions R{};
        R.args = arena.allocate(sizeof(transport_bw_x280_to_tensix_args_block_t));
        R.staging = arena.allocate(TB_X2T_PATTERN_BYTES);
        R.result = arena.allocate(sizeof(transport_bw_result_t));
        return R;
    }
};

struct TensixL1Layout {
    uint32_t dst_offset;
    uint32_t dst_bytes;
    uint32_t block_bytes;

    static TensixL1Layout compute() {
        TensixL1Layout L{};
        L.dst_offset = 0;
        L.dst_bytes = static_cast<uint32_t>(hreg::round_up(TB_X2T_DST_MAX_BYTES, hreg::kCacheLineBytes));
        L.block_bytes = L.dst_bytes;
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

    const uint32_t W = hart_::env_u32("TRANSPORT_BW_W", kDefaultWarmupCount);
    const uint32_t N = hart_::env_u32("TRANSPORT_BW_N", kDefaultTransferCount);
    ASSERT_GT(W, 0u);
    ASSERT_GT(N, 0u);
    const uint64_t logical_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(transfer_size);

    const TensixL1Layout TL = TensixL1Layout::compute();
    const ttm::CoreCoord worker{0, 0};
    const ttm::CoreRangeSet worker_set(ttm::CoreRange(worker, worker));
    auto l1 = hreg::allocate_tile_l1_buffer(f.mesh.get(), worker_set, TL.block_bytes);
    const uint32_t l1_base = l1.l1_tile_base;

    const auto worker_virtual =
        x280::X280Runtime::virtual_core_from_logical_core(*f.mesh, worker, tt::CoreType::TENSIX, ttm::NOC::NOC_0);
    const uint32_t dst_addr = l1_base + TL.dst_offset;
    ASSERT_LE(static_cast<uint64_t>(dst_addr + TL.dst_bytes), (static_cast<uint64_t>(dst_addr) | 0x1FFFFFu) + 1ull)
        << "destination region must lie in one 2 MiB window";

    const X280LimRegions XR = X280LimRegions::compute(*f.tile);
    hreg::zero_region(*f.tile, XR.args);
    hreg::zero_region(*f.tile, XR.staging);
    hreg::zero_region(*f.tile, XR.result);

    uint32_t run_nonce = 0;
    const uint64_t run_value = mint_run_value(run_nonce);
    std::vector<uint8_t> staging_bytes(XR.staging.bytes, 0);
    for (uint32_t k = 0; k < TB_X2T_PATTERN_BYTES / sizeof(uint32_t); ++k) {
        const uint32_t v = pattern_word(run_nonce, k);
        std::memcpy(staging_bytes.data() + k * sizeof(uint32_t), &v, sizeof(v));
    }
    f.tile->write_available_lim(XR.staging.addr, staging_bytes);

    transport_bw_x280_to_tensix_args_block_t block{};
    block.cookie_common = transport_bw_cookie_for(UBENCH_SECTION_COMMON);
    block.pad0 = 0u;
    block.common.transfer_size = transfer_size;
    block.common.warmup_count = W;
    block.common.transfer_count = N;
    block.common.boundary = boundary;
    block.common.run_nonce = run_nonce;
    block.common.reserved = 0u;
    block.cookie_backend = transport_bw_cookie_for(UBENCH_SECTION_BACKEND);
    block.pad1 = 0u;
    block.backend.peer_dst_addr = static_cast<uint64_t>(dst_addr);
    block.backend.own_staging_addr = XR.staging.addr;
    block.backend.own_result_addr = XR.result.addr;
    block.backend.peer_noc = worker_virtual.noc();
    block.backend.peer_tlb_index = kTlbIndex;
    block.backend.store_width = store_width;
    hreg::write_args(*f.tile, XR.args, block);

    hexec::X280Launch launch{
        .image_path = hart_::image_path(
            "X280_TRANSPORT_BW_X280_TO_TENSIX_FW_IMAGE", "firmware-transport-bandwidth-x280-to-tensix.bin"),
        .run_value = run_value,
        .args_addr = XR.args.addr,
        .args_bytes = XR.args.bytes,
        .cleanup_timeout = std::chrono::seconds{5},
        .wait_timeout = std::chrono::seconds{60},
    };
    const auto outcome = hexec::run_x280(f.handle(), launch);
    ASSERT_TRUE(outcome.enqueued()) << "enqueue rejected";
    ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);
    hexec::expect_window_cleared(*f.tile, kTlbIndex);

    const auto result = hreg::read_struct<transport_bw_result_t>(*f.tile, XR.result);

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

    std::vector<uint32_t> l1_readback;
    ttmd::EnqueueReadMeshBuffer(f.mesh->mesh_command_queue(), l1_readback, l1.buffer, /*blocking=*/true);
    ASSERT_GE(l1_readback.size(), transfer_size / sizeof(uint32_t));
    uint32_t mismatches = 0;
    uint32_t first_mismatch = 0;
    for (uint32_t k = 0; k < transfer_size / sizeof(uint32_t); ++k) {
        if (l1_readback[k] != pattern_word(run_nonce, k) && mismatches++ == 0) {
            first_mismatch = k;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "destination words did not receive the staging pattern, first at word "
                              << first_mismatch << " (read 0x" << std::hex << l1_readback[first_mismatch]
                              << ", expected 0x" << pattern_word(run_nonce, first_mismatch) << std::dec
                              << "); did the transport drop writes?";

    const auto body = x280::harness::output::record(x280_to_tensix_record(
        result, {.transfer_size = transfer_size, .boundary = boundary, .warmup = W, .samples = N}, store_width));
    x280::harness::output::write_jsonl("transport_bandwidth_x280_to_tensix", body);
    std::printf("%s\n", body.c_str());
}

}  // namespace x280::suites::transport_bandwidth::x280_to_tensix

#endif  // X280_SUITES_TRANSPORT_BW_X280_TO_TENSIX_HPP
