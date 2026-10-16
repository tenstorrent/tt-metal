// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host tests of the all_gather host layer (ttnn/.../all_gather/device/):
//   - WalkGeometry::make, the pure form the factories call through: k, slot, stride, xfer, run sources and CB
//     entry for each copy mode, the padded slot, the run-cap budget, the packet-load entry's L1 clamp, and its
//     TT_FATALs (an output buffer whose aligned page is not the spec's; a k the policy cannot ask for);
//   - WalkGeometry::relay_ok: the relay rule on the chunk the unicast walk actually moves (k > 1, padded slot);
//   - sharded_contiguous_page_stride, the host mirror of TensorAccessor::contiguous_page_stride(), against real
//     sharded TensorAccessor instances;
//   - the per-arch policy (all_gather_policy.cpp): routing heuristic rows, the unicast / multicast tiers, and
//     the invariants every policy keeps.
//
// No device: GatherGeometry is built field by field (GatherGeometry::make needs a device tensor), so these pin the
// walk's arithmetic, not the tensor-to-geometry step (covered on device by the nightly harness).

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "ttnn/operations/ccl/all_gather/device/all_gather_geometry.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_policy.hpp"

#include "host_kernel_shim.hpp"  // TensorAccessor<DSpec> on the host

namespace {

using namespace ttnn::operations::ccl;

constexpr uint32_t kBhPacket = 15232;
constexpr uint32_t kBhBurst = 16384;
constexpr uint32_t kBhDram = 64, kL1 = 16;

// A geometry as GatherGeometry::make would derive it from the page sizes, alignments and stripe.
GatherGeometry geometry(
    uint32_t in_page,
    uint32_t in_align,
    uint32_t out_page,
    uint32_t out_align,
    uint32_t stripe_input_pages,
    uint32_t num_input_pages) {
    GatherGeometry g;
    g.in_page = in_page;
    g.in_read_alignment = in_align;
    g.in_aligned_page = (in_page + in_align - 1) / in_align * in_align;
    g.out_page = out_page;
    g.out_read_alignment = out_align;
    g.out_aligned_page = (out_page + out_align - 1) / out_align * out_align;
    const bool split = in_page > out_page;
    g.walk_page_bytes = split ? out_page : g.in_aligned_page;
    g.out_walk_pages_per_page = split ? 1 : out_page / in_page;
    g.in_walk_pages_per_page = split ? in_page / out_page : 1;
    g.mode = split ? CopyMode::Split : (g.out_walk_pages_per_page > 1 ? CopyMode::Concat : CopyMode::Matched);
    g.relay_ok = relay_read_ok(g.walk_page_bytes, g.out_walk_pages_per_page, g.out_aligned_page, out_align);
    g.relay_needs_padded_slot =
        relay_needs_padded_slot(g.walk_page_bytes, g.out_walk_pages_per_page, g.out_aligned_page);
    g.num_input_pages = num_input_pages;
    g.num_walk_pages = num_input_pages * g.in_walk_pages_per_page;
    g.walk_pages_per_stripe = stripe_input_pages * g.in_walk_pages_per_page;
    return g;
}

WalkInputs bh_inputs(const GatherGeometry& g, uint32_t page_stride = 8) {
    WalkInputs in;
    in.in_page_stride = page_stride;
    in.out_page_stride = page_stride;
    in.out_buffer_aligned_page = g.out_aligned_page;
    in.packet_size = kBhPacket;
    in.burst = kBhBurst;
    in.max_l1_space = 1u << 20;
    return in;
}

// Unicast's walk (BH): k = auto, one-packet entries, padded slot. The other: packet-load entries at k = 1.
WalkPolicy unicast_walk() { return {.k = WalkPolicy::kAutoK, .one_packet_short_stripe = true, .padded_slot = true}; }
WalkPolicy multicast_walk() { return {.k = 1, .one_packet_short_stripe = false, .padded_slot = false}; }

}  // namespace

// ---------------------------------------------------------------------------------------------------
// WalkGeometry::make
// ---------------------------------------------------------------------------------------------------

// bf16 tile, DRAM -> DRAM, 8 banks, stripe 4: k stays 1 (2 KB fills 7 of 15232 B), one-packet entry of
// 4 scatter segments of 1-page lane runs; the packet-load entry is one 7-chunk packet.
TEST(AllGatherGeometry, WalkTileMatched) {
    const auto g = geometry(2048, kBhDram, 2048, kBhDram, 4, 64);
    const auto w = WalkGeometry::make(g, unicast_walk(), bh_inputs(g));
    EXPECT_EQ(w.k, 1u);
    EXPECT_EQ(w.chunk_size, 2048u);
    EXPECT_EQ(w.slot_bytes, 2048u);
    EXPECT_EQ(w.stride, 8u);
    EXPECT_EQ(w.xfer, 7u);
    EXPECT_TRUE(w.out_runs.cross_page && w.out_runs.whole_page);
    EXPECT_TRUE(w.in_runs.cross_page && w.in_runs.whole_page);
    EXPECT_EQ(w.chunks_per_entry, 4u);  // min(xfer 7, 4 segments x a 1-page lane run)
    EXPECT_EQ(w.cb_entry_bytes, 4u * 2048u);
    EXPECT_EQ(w.cps, 4u);
    EXPECT_TRUE(w.relay_ok(g));

    const auto legacy = WalkGeometry::make(g, multicast_walk(), bh_inputs(g));
    EXPECT_EQ(legacy.chunks_per_entry, 7u);
    // A stripe longer than the stride (8) keeps packet-load entries: one 7-chunk packet, or two in the 9-15 band.
    const auto g10 = geometry(2048, kBhDram, 2048, kBhDram, 10, 80);
    auto two = bh_inputs(g10);
    two.packets_per_cb_entry = 2;
    EXPECT_EQ(WalkGeometry::make(g10, unicast_walk(), two).chunks_per_entry, 14u);
    const auto g32 = geometry(2048, kBhDram, 2048, kBhDram, 32, 64);
    EXPECT_EQ(WalkGeometry::make(g32, unicast_walk(), bh_inputs(g32)).chunks_per_entry, 7u);
    // At the stride itself (stripe 8, 8 banks) the entry is one packet of 4 one-page lane runs; the packet-load
    // count does not apply there.
    const auto g8 = geometry(2048, kBhDram, 2048, kBhDram, 8, 64);
    auto ignored = bh_inputs(g8);
    ignored.packets_per_cb_entry = 2;
    EXPECT_EQ(WalkGeometry::make(g8, unicast_walk(), ignored).chunks_per_entry, 4u);
    // On an L1-interleaved output (stride = 120 banks, say) a stripe of 10 is short.
    EXPECT_EQ(WalkGeometry::make(g10, unicast_walk(), bh_inputs(g10, 120)).chunks_per_entry, 4u);
}

// Row-major 8 KB pages: k = auto cuts them in 4 (2 KB chunks fill 7 of a packet); a run cap of 8 KB is the budget,
// and a whole page already fills it, so k stays 1 there.
TEST(AllGatherGeometry, WalkAutoK) {
    const auto g = geometry(8192, kBhDram, 8192, kBhDram, 1, 256);
    const auto w = WalkGeometry::make(g, unicast_walk(), bh_inputs(g));
    EXPECT_EQ(w.k, 4u);
    EXPECT_EQ(w.chunk_size, 2048u);
    EXPECT_EQ(w.in_cpp, 4u);
    EXPECT_EQ(w.out_cpp, 4u);
    EXPECT_EQ(w.cps, 4u);
    EXPECT_EQ(w.stride, 8u);  // out_cpp == k: still the output's page stride
    EXPECT_EQ(w.xfer, 7u);
    EXPECT_TRUE(w.relay_ok(g));

    auto capped = bh_inputs(g);
    capped.run_cap_bytes = 8192;
    const auto wc = WalkGeometry::make(g, unicast_walk(), capped);
    EXPECT_EQ(wc.k, 1u);
    EXPECT_EQ(wc.xfer, 1u);

    // k = 1 policy leaves it alone; 4 KB pages get k = 2.
    EXPECT_EQ(WalkGeometry::make(g, multicast_walk(), bh_inputs(g)).k, 1u);
    const auto g4 = geometry(4096, kBhDram, 4096, kBhDram, 1, 256);
    EXPECT_EQ(WalkGeometry::make(g4, unicast_walk(), bh_inputs(g4)).k, 2u);
    // A padded page is never cut.
    const auto gp = geometry(8192 + 32, kBhDram, 8192 + 32, kBhDram, 1, 256);
    EXPECT_EQ(WalkGeometry::make(gp, unicast_walk(), bh_inputs(gp)).k, 1u);
}

// 32 B rows, L1 -> DRAM: the output page pads to 64 B. Unicast's padded slot steps L1 like the output (cross-page
// output runs, xfer of 64 B slots, relay legal); the input then reads one chunk at a time. Multicast keeps
// slot == chunk, so the unpacked output walks xfer 1 and could not be relayed.
TEST(AllGatherGeometry, WalkPaddedSlot) {
    const auto g = geometry(32, kL1, 32, kBhDram, 1, 4096);
    ASSERT_TRUE(g.relay_needs_padded_slot);
    const auto w = WalkGeometry::make(g, unicast_walk(), bh_inputs(g));
    EXPECT_EQ(w.k, 1u);
    EXPECT_EQ(w.slot_bytes, 64u);
    EXPECT_EQ(w.chunk_size, 32u);
    EXPECT_TRUE(w.out_runs.cross_page);
    EXPECT_FALSE(w.in_runs.cross_page);
    EXPECT_FALSE(w.in_runs.whole_page);
    EXPECT_EQ(w.xfer, kBhPacket / 64);
    EXPECT_TRUE(w.relay_ok(g));

    const auto m = WalkGeometry::make(g, multicast_walk(), bh_inputs(g));
    EXPECT_EQ(m.slot_bytes, 32u);
    EXPECT_EQ(m.xfer, 1u);
    EXPECT_FALSE(m.out_runs.cross_page);
    EXPECT_FALSE(m.relay_ok(g)) << "a 32 B slot over a 64 B output page cannot be relayed";
}

// Concat: 4 x 32 B into a 128 B DRAM page. Stride 1 (neighbours share the page); in-page runs only (the
// output's page stride is 8); BH's 64 B DRAM alignment makes the relay illegal, as routing knows.
TEST(AllGatherGeometry, WalkConcat) {
    const auto g = geometry(32, kL1, 128, kBhDram, 1, 4096);
    ASSERT_EQ(g.mode, CopyMode::Concat);
    const auto w = WalkGeometry::make(g, unicast_walk(), bh_inputs(g));
    EXPECT_EQ(w.k, 1u);
    EXPECT_EQ(w.out_cpp, 4u);
    EXPECT_EQ(w.stride, 1u);
    EXPECT_TRUE(w.out_runs.whole_page);
    EXPECT_FALSE(w.out_runs.cross_page);
    EXPECT_FALSE(g.relay_ok);
    EXPECT_FALSE(w.relay_ok(g));
}

// Split: 128 B input rows into 32 B DRAM-sharded output rows (64 B aligned): a padded slot, one read per chunk.
TEST(AllGatherGeometry, WalkSplit) {
    const auto g = geometry(128, kL1, 32, kBhDram, 1, 1024);
    ASSERT_EQ(g.mode, CopyMode::Split);
    const auto w = WalkGeometry::make(g, unicast_walk(), bh_inputs(g));
    EXPECT_EQ(w.in_cpp, 4u);
    EXPECT_EQ(w.slot_bytes, 64u);
    EXPECT_FALSE(w.in_runs.whole_page || w.in_runs.cross_page);
    EXPECT_TRUE(w.relay_ok(g));
}

// The packet-load entry's L1 clamp: packets_per_cb_entry 4 at depth 3 shrinks to what fits.
TEST(AllGatherGeometry, WalkPacketLoadEntryClamp) {
    const auto g = geometry(2048, kBhDram, 2048, kBhDram, 64, 1024);
    auto in = bh_inputs(g);
    in.packets_per_cb_entry = 4;
    in.cb_depth = 3;
    EXPECT_EQ(WalkGeometry::make(g, multicast_walk(), in).chunks_per_entry, 28u);
    in.max_l1_space = 3 * 2 * 7 * 2048;  // room for two packets per entry
    EXPECT_EQ(WalkGeometry::make(g, multicast_walk(), in).chunks_per_entry, 14u);
    in.max_l1_space = 0;  // never below one packet
    EXPECT_EQ(WalkGeometry::make(g, multicast_walk(), in).chunks_per_entry, 7u);
}

// Every walk keeps the kernels' CT invariants (check_walk_ct) and the CB entry is whole slots.
TEST(AllGatherGeometry, WalkInvariants) {
    for (uint32_t in_page : {32u, 64u, 128u, 1088u, 2048u, 4096u, 8192u, 32768u}) {
        for (uint32_t ratio : {1u, 2u, 4u}) {
            for (bool split : {false, true}) {
                const uint32_t out_page = split ? in_page / ratio : in_page * ratio;
                if (out_page % 16 != 0 || out_page == 0) {
                    continue;
                }
                for (uint32_t out_align : {kL1, kBhDram}) {
                    const auto g = geometry(in_page, kL1, out_page, out_align, 3, 96);
                    for (const auto& p : {unicast_walk(), multicast_walk()}) {
                        const auto w = WalkGeometry::make(g, p, bh_inputs(g));
                        SCOPED_TRACE(
                            "in " + std::to_string(in_page) + " out " + std::to_string(out_page) + " align " +
                            std::to_string(out_align));
                        EXPECT_TRUE(slot_is_valid(w.chunk_size, w.slot_bytes, w.out_cpp, w.k));
                        EXPECT_EQ(w.chunk_size % gather_noc_write_alignment, 0u);
                        EXPECT_EQ(w.cb_entry_bytes % w.slot_bytes, 0u);
                        EXPECT_TRUE(w.xfer == 1 || w.xfer * w.slot_bytes <= kBhPacket);
                        EXPECT_EQ(w.in_cpp % w.k, 0u);
                        EXPECT_EQ(w.out_cpp % w.k, 0u);
                        EXPECT_GE(w.chunks_per_entry, 1u);
                    }
                }
            }
        }
    }
}

// The output buffer's aligned page must be the spec's (the kernels step by the buffer's); a policy may only ask
// for k = 1 or auto.
TEST(AllGatherGeometry, WalkFatals) {
    const auto g = geometry(2048, kBhDram, 2048, kBhDram, 4, 64);
    auto in = bh_inputs(g);
    in.out_buffer_aligned_page = 2048 + 64;
    EXPECT_ANY_THROW(WalkGeometry::make(g, unicast_walk(), in));
    WalkPolicy forced = unicast_walk();
    forced.k = 2;
    EXPECT_ANY_THROW(WalkGeometry::make(g, forced, bh_inputs(g)));
}

// ---------------------------------------------------------------------------------------------------
// sharded_contiguous_page_stride vs TensorAccessor<DSpec>::contiguous_page_stride()
// ---------------------------------------------------------------------------------------------------
namespace {
template <size_t... Dims>
using U32 = tensor_accessor::ArrayStaticWrapperU32<Dims...>;
template <size_t... Dims>
using U16 = tensor_accessor::ArrayStaticWrapperU16<Dims...>;

template <typename DSpec, size_t R>
void check_stride(const char* name, const std::array<uint32_t, R>& tensor, const std::array<uint32_t, R>& shard) {
    const uint32_t kernel = TensorAccessor<DSpec>(0, 32).contiguous_page_stride();
    const uint32_t host = sharded_contiguous_page_stride(
        tt::tt_metal::Shape(std::vector<uint32_t>(tensor.begin(), tensor.end())),
        tt::tt_metal::Shape(std::vector<uint32_t>(shard.begin(), shard.end())));
    EXPECT_EQ(host, kernel) << name;
}
}  // namespace

TEST(AllGatherGeometry, ShardedPageStrideMatchesAccessor) {
    using namespace tensor_accessor;
    check_stride<DistributionSpec<2, 3, U32<3, 5>, U32<2, 2>, U16<0, 1, 2>>, 2>("padded 2D", {3, 5}, {2, 2});
    check_stride<DistributionSpec<3, 3, U32<4, 3, 4>, U32<1, 3, 4>, U16<0, 1, 2>>, 3>("rank 3", {4, 3, 4}, {1, 3, 4});
    check_stride<DistributionSpec<3, 1, U32<2, 3, 4>, U32<2, 3, 4>, U16<0>>, 3>("one shard", {2, 3, 4}, {2, 3, 4});
    check_stride<DistributionSpec<2, 6, U32<4, 6>, U32<4, 1>, U16<0, 1, 2, 3, 4, 5>>, 2>(
        "one page wide", {4, 6}, {4, 1});
    check_stride<DistributionSpec<4, 4, U32<4, 3, 2, 2>, U32<2, 1, 1, 1>, U16<0, 1, 2, 3>>, 4>(
        "rank 4", {4, 3, 2, 2}, {2, 1, 1, 1});
    check_stride<DistributionSpec<1, 2, U32<10>, U32<5>, U16<0, 1>>, 1>("rank 1", {10}, {5});
    check_stride<DistributionSpec<2, 4, U32<8, 4>, U32<2, 4>, U16<0, 1, 2, 3>>, 2>("height sharded", {8, 4}, {2, 4});
    check_stride<DistributionSpec<2, 4, U32<4, 6>, U32<2, 3>, U16<0, 1, 2, 3>>, 2>("block sharded", {4, 6}, {2, 3});
    check_stride<DistributionSpec<2, 3, U32<4, 6>, U32<4, 2>, U16<0, 1, 2>>, 2>("width sharded", {4, 6}, {4, 2});
    check_stride<DistributionSpec<3, 4, U32<3, 5, 3>, U32<2, 2, 2>, U16<0, 1, 2, 3>>, 3>(
        "padded rank 3", {3, 5, 3}, {2, 2, 2});
    check_stride<DistributionSpec<2, 4, U32<5, 4>, U32<2, 1>, U16<0, 1, 2, 3>>, 2>(
        "padded one page wide", {5, 4}, {2, 1});
    check_stride<DistributionSpec<3, 2, U32<2, 8, 2>, U32<1, 2, 1>, U16<0, 1>>, 3>(
        "nd shard, unit last dim", {2, 8, 2}, {1, 2, 1});
}

// ---------------------------------------------------------------------------------------------------
// Policy (all_gather_policy.cpp)
// ---------------------------------------------------------------------------------------------------
namespace {
PolicyInputs inputs(tt::ARCH arch, bool ring, uint32_t n, uint32_t links, uint64_t per_link, uint32_t stripe) {
    PolicyInputs in;
    in.arch = arch;
    in.is_ring = ring;
    in.num_devices = n;
    in.num_links = links;
    in.per_link_bytes = per_link;
    in.stripe_pages = stripe;
    return in;
}
constexpr uint64_t KB = 1024, MB = 1024 * 1024;
constexpr auto BH = tt::ARCH::BLACKHOLE;
constexpr auto WH = tt::ARCH::WORMHOLE_B0;
}  // namespace

TEST(AllGatherPolicy, PrefersUnicast) {
    // BH: ring threshold 320 * (N / 2 - 2) KB, line 384 + 544 * (N - 4) KB, floored at 0.
    EXPECT_FALSE(prefers_unicast(inputs(BH, true, 8, 2, 639 * KB, 4)));
    EXPECT_TRUE(prefers_unicast(inputs(BH, true, 8, 2, 640 * KB, 4)));
    EXPECT_TRUE(prefers_unicast(inputs(BH, true, 4, 2, 1, 4)));
    EXPECT_FALSE(prefers_unicast(inputs(BH, false, 8, 2, 2559 * KB, 4)));
    EXPECT_TRUE(prefers_unicast(inputs(BH, false, 8, 2, 2560 * KB, 4)));
    EXPECT_TRUE(prefers_unicast(inputs(BH, false, 4, 2, 384 * KB, 4)));
    EXPECT_TRUE(prefers_unicast(inputs(BH, false, 2, 2, 1, 4)));
    // WH: ring never; 2 devices always; 3-6 never; 7+ from 3 MB per link.
    EXPECT_FALSE(prefers_unicast(inputs(WH, true, 8, 1, 100 * MB, 4)));
    EXPECT_TRUE(prefers_unicast(inputs(WH, false, 2, 1, 1, 4)));
    EXPECT_FALSE(prefers_unicast(inputs(WH, false, 4, 1, 100 * MB, 4)));
    EXPECT_FALSE(prefers_unicast(inputs(WH, false, 8, 1, 2'999'999, 4)));
    EXPECT_TRUE(prefers_unicast(inputs(WH, false, 8, 1, 3'000'000, 4)));
    EXPECT_FALSE(prefers_unicast(inputs(tt::ARCH::QUASAR, false, 2, 1, 1, 4)));
}

TEST(AllGatherPolicy, UnicastBlackhole) {
    const auto p = tune_unicast(inputs(BH, true, 8, 2, 2 * MB, 8));
    EXPECT_EQ(p.shared.walk.k, WalkPolicy::kAutoK);
    EXPECT_TRUE(p.shared.walk.one_packet_short_stripe);
    EXPECT_TRUE(p.shared.walk.padded_slot);
    EXPECT_EQ(p.shared.packets_per_cb_entry, 1u);
    // The measured two-packet band, stripes 9-15, for the packet-load entries of longer stripes.
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 2 * MB, 9)).shared.packets_per_cb_entry, 2u);
    EXPECT_EQ(tune_unicast(inputs(BH, false, 8, 2, 2 * MB, 15)).shared.packets_per_cb_entry, 2u);
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 2 * MB, 16)).shared.packets_per_cb_entry, 1u);
    EXPECT_EQ(p.shared.cb_depth, 2u);
    EXPECT_EQ(p.signals_per_stripe, 2u);
    EXPECT_EQ(p.mux_slots_per_channel, 2u);
    // Ring workers by total output bytes (per link x links), capped at 2 below a long stripe.
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 31 * KB, 8)).workers_per_dir, 1u);
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 32 * KB, 8)).workers_per_dir, 2u);
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 768 * KB, 8)).workers_per_dir, 3u);
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 8 * MB, 8)).workers_per_dir, 4u);
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 8 * MB, 7)).workers_per_dir, 2u);
    // Line: 2 workers from 640 KB / N per link.
    EXPECT_EQ(tune_unicast(inputs(BH, false, 8, 2, 79 * KB, 8)).workers_per_dir, 1u);
    EXPECT_EQ(tune_unicast(inputs(BH, false, 8, 2, 80 * KB, 8)).workers_per_dir, 2u);
    // Run cap only from 4 links and 8 MB total.
    EXPECT_EQ(tune_unicast(inputs(BH, true, 8, 2, 50 * MB, 32)).shared.run_cap_bytes, 0u);
    EXPECT_EQ(tune_unicast(inputs(BH, true, 4, 4, 2 * MB, 32)).shared.run_cap_bytes, 4096u);
    EXPECT_EQ(tune_unicast(inputs(BH, true, 4, 4, 2 * MB - 1, 32)).shared.run_cap_bytes, 0u);
}

TEST(AllGatherPolicy, UnicastWormhole) {
    const auto p = tune_unicast(inputs(WH, false, 8, 1, 4 * MB, 64));
    EXPECT_EQ(p.shared.walk.k, 1u);
    EXPECT_FALSE(p.shared.walk.one_packet_short_stripe);
    EXPECT_TRUE(p.shared.walk.padded_slot);
    EXPECT_EQ(p.shared.packets_per_cb_entry, 3u);
    EXPECT_EQ(p.shared.run_cap_bytes, 0u);
    // Small per-device share -> 1 worker; mid share on a line with a short stripe -> 2.
    EXPECT_EQ(tune_unicast(inputs(WH, false, 8, 1, 512 * KB, 8)).workers_per_dir, 1u);
    EXPECT_EQ(tune_unicast(inputs(WH, false, 8, 1, 8 * MB, 8)).workers_per_dir, 2u);
    EXPECT_EQ(tune_unicast(inputs(WH, true, 8, 1, 2 * MB, 8)).mux_slots_per_channel, 1u);
    EXPECT_EQ(tune_unicast(inputs(WH, false, 8, 1, 2 * MB, 8)).mux_slots_per_channel, 2u);
}

TEST(AllGatherPolicy, Multicast) {
    const auto bh = tune_multicast(inputs(BH, true, 8, 2, 4 * MB, 32));
    EXPECT_EQ(bh.shared.cb_depth, 3u);
    EXPECT_EQ(bh.shared.walk.k, WalkPolicy::kAutoK);
    EXPECT_FALSE(bh.shared.walk.one_packet_short_stripe);
    EXPECT_FALSE(bh.shared.walk.padded_slot);
    EXPECT_EQ(bh.shared.packets_per_cb_entry, 4u);
    EXPECT_EQ(bh.shared.run_cap_bytes, 0u);
    EXPECT_EQ(tune_multicast(inputs(BH, false, 8, 2, 4 * MB, 32)).shared.packets_per_cb_entry, 1u);
    EXPECT_EQ(tune_multicast(inputs(BH, false, 8, 4, 4 * MB, 32)).shared.packets_per_cb_entry, 4u);
    EXPECT_EQ(tune_multicast(inputs(BH, false, 8, 2, 4 * MB, 31)).shared.packets_per_cb_entry, 4u);
    const auto wh = tune_multicast(inputs(WH, true, 8, 1, 4 * MB, 32));
    EXPECT_EQ(wh.shared.cb_depth, 2u);
    EXPECT_EQ(wh.shared.walk.k, 1u);
    EXPECT_EQ(wh.shared.packets_per_cb_entry, 1u);
    EXPECT_EQ(wh.shared.run_cap_bytes, 0u);
}

// Whatever the tier, every policy keeps the invariants the walk and kernels rely on.
TEST(AllGatherPolicy, Invariants) {
    for (auto arch : {BH, WH, tt::ARCH::QUASAR}) {
        for (bool ring : {false, true}) {
            for (uint32_t n : {2u, 4u, 8u, 16u}) {
                for (uint32_t links : {1u, 2u, 4u}) {
                    for (uint64_t bytes : {1 * KB, 100 * KB, 1 * MB, 10 * MB, 100 * MB}) {
                        for (uint32_t stripe : {1u, 2u, 8u, 10u, 32u, 64u}) {
                            const auto in = inputs(arch, ring, n, links, bytes, stripe);
                            const auto u = tune_unicast(in);
                            const auto m = tune_multicast(in);
                            for (const auto* s : {&u.shared, &m.shared}) {
                                EXPECT_TRUE(s->walk.k == 1 || s->walk.k == WalkPolicy::kAutoK);
                                EXPECT_GE(s->cb_depth, 2u);
                                EXPECT_GE(s->packets_per_cb_entry, 1u);
                            }
                            EXPECT_TRUE(u.shared.walk.padded_slot);
                            EXPECT_GE(u.workers_per_dir, 1u);
                            EXPECT_LE(u.workers_per_dir, 4u);
                            EXPECT_GE(u.signals_per_stripe, 1u);
                            EXPECT_TRUE(u.mux_slots_per_channel == 1 || u.mux_slots_per_channel == 2);
                        }
                    }
                }
            }
        }
    }
}
