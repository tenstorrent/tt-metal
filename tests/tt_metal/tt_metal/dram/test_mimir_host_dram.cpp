// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/host_api.hpp>
#include "umd/device/utils/error_detail.hpp"
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test::host_dram {

inline constexpr ChipId kChipId = 0;
inline constexpr uint32_t kTl1CceIndex = 0;
inline constexpr uint32_t kPersistView = 0;
inline constexpr uint32_t kRegionBytes = 1024;
inline constexpr uint32_t kBoundaryTransferBytes = 128;
inline constexpr uint32_t kFullPatternSweepMaxBytes = 4 * 1024;
inline constexpr uint32_t kL1PoisonTag = 0xE00;
inline constexpr uint32_t kL1PatternTag = 0xE01;
inline constexpr uint32_t kRejectSentinelTag = 0xE02;
inline constexpr uint32_t kStrideEndSentinelTag = 0xE03;
inline constexpr uint32_t kTl1EndSentinelTag = 0xE04;
inline constexpr uint32_t kTl1AliasGddrTag = 0xE05;
inline constexpr uint32_t kAliasWalkTagBase = 0xA00;

// Sizes for PatternReadback. The full 7-pair sweep runs up to kFullPatternSweepMaxBytes; larger sizes run one pair.
inline constexpr std::array<uint32_t, 7> kHostDramSizesBytes = {4, 16, 64, 512, 4096, 32768, 131072};
// Centres of the 128 B straddles inside the "start" anchor for BoundaryAndAlias.
inline constexpr std::array<uint64_t, 3> kStartAnchorStraddleCentres = {0x80, 0x200, 0x8000};

struct PatternPair {
    MimirPattern first;
    MimirPattern second;
};

struct HostDramPoint {
    std::string name;
    uint64_t view_offset = 0;
};

struct GuardSpot {
    uint64_t view_offset;
    uint32_t tag;
};

std::vector<PatternPair> pattern_pairs_for_size(uint32_t size_bytes) {
    if (size_bytes > kFullPatternSweepMaxBytes) {
        return std::vector<PatternPair>{PatternPair{MimirPattern::Prng, MimirPattern::AllOnes}};
    }
    std::vector<PatternPair> pairs;
    for (size_t pair_index = 0; pair_index < kAllMimirPatterns.size(); ++pair_index) {
        pairs.push_back(
            {kAllMimirPatterns[pair_index], kAllMimirPatterns[(pair_index + 1) % kAllMimirPatterns.size()]});
    }
    return pairs;
}

// Offsets for PatternReadback. Each payload sits kGuardBytes inside its region so a guard fits on both sides.
std::vector<HostDramPoint> host_dram_offsets(uint64_t view_size, uint64_t unreserved, uint32_t size_bytes) {
    return {
        {"start", kGuardBytes},
        {"unreserved", unreserved + kGuardBytes},
        {"view_mid", view_size / 2 + kGuardBytes},
        {"view_end", view_size - kGuardBytes - size_bytes},
    };
}

std::string view_param_name(const ::testing::TestParamInfo<uint32_t>& info) {
    return fmt::format("View{}", info.param);
}

::testing::AssertionResult read_and_compare(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    uint32_t view,
    uint64_t view_offset,
    const std::vector<uint32_t>& expected,
    const std::string& label) {
    std::vector<uint32_t> actual;
    mimir_host_dram_read(
        cluster, soc, kChipId, view, view_offset, static_cast<uint32_t>(expected.size() * sizeof(uint32_t)), actual);
    return compare_words(label, expected, actual);
}

::testing::AssertionResult write_then_read_back(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    uint32_t view,
    uint64_t view_offset,
    const std::vector<uint32_t>& expected,
    const std::string& label) {
    mimir_host_dram_write(cluster, soc, kChipId, view, view_offset, expected);
    return read_and_compare(cluster, soc, view, view_offset, expected, label);
}

// Writes a 128 B Prng pattern centred on `boundary` and reads it back through the same view.
::testing::AssertionResult write_straddle_and_read_back(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    uint32_t view,
    uint64_t boundary,
    const std::string& boundary_name) {
    const uint64_t straddle_offset = boundary - kBoundaryTransferBytes / 2;
    const MimirDramAddressInfo info = mimir_dram_address(soc, kChipId, view, straddle_offset, kBoundaryTransferBytes);
    const std::vector<uint32_t> pattern = make_pattern(
        MimirPattern::Prng, kBoundaryTransferBytes / sizeof(uint32_t), info.flat_address, mimir_test_seed());
    return write_then_read_back(
        cluster,
        soc,
        view,
        straddle_offset,
        pattern,
        fmt::format("{} straddles {} 0x{:x}", info.host_string(), boundary_name, boundary));
}

struct PersistedDramRegion {
    uint32_t view;
    std::string name;
    uint64_t view_offset;
    std::vector<uint32_t> expected;
};

struct PersistedTl1Region {
    uint32_t cce_index;
    uint64_t tl1_offset;
    std::vector<uint32_t> expected;
};

class MimirHostDramView : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MimirHostDramView, PatternReadback) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    const uint32_t view = GetParam();
    if (view >= topology.num_views) {
        GTEST_SKIP() << "Descriptor exposes " << topology.num_views << " DRAM views.";
    }
    MIMIR_REQUIRE_IDENTITY();
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(kChipId);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, kChipId, false);

    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    uint32_t tag = 0;
    for (uint32_t size_bytes : kHostDramSizesBytes) {
        const uint32_t word_count = size_bytes / sizeof(uint32_t);
        const std::vector<PatternPair> pattern_pairs = pattern_pairs_for_size(size_bytes);
        for (const HostDramPoint& offset_case : host_dram_offsets(topology.view_size, unreserved, size_bytes)) {
            const std::vector<uint64_t> guard_offsets = {
                offset_case.view_offset - kGuardBytes, offset_case.view_offset + size_bytes};
            for (const PatternPair& pair : pattern_pairs) {
                const MimirDramAddressInfo info =
                    mimir_dram_address(soc, kChipId, view, offset_case.view_offset, size_bytes);
                const std::string label = fmt::format(
                    "{} {} {}->{} tag {}",
                    info.host_string(),
                    offset_case.name,
                    pattern_name(pair.first),
                    pattern_name(pair.second),
                    tag);
                std::vector<GuardSpot> guards;
                for (uint64_t guard_offset : guard_offsets) {
                    guards.push_back({guard_offset, tag});
                    ++tag;
                }
                for (const GuardSpot& guard : guards) {
                    mimir_host_dram_write(cluster, soc, kChipId, view, guard.view_offset, make_guard(guard.tag));
                }
                EXPECT_TRUE(write_then_read_back(
                    cluster,
                    soc,
                    view,
                    offset_case.view_offset,
                    make_pattern(pair.first, word_count, info.flat_address, mimir_test_seed()),
                    label + " pattern A"));
                EXPECT_TRUE(write_then_read_back(
                    cluster,
                    soc,
                    view,
                    offset_case.view_offset,
                    make_pattern(pair.second, word_count, info.flat_address, mimir_test_seed()),
                    label + " pattern B"));
                for (const GuardSpot& guard : guards) {
                    EXPECT_TRUE(read_and_compare(
                        cluster,
                        soc,
                        view,
                        guard.view_offset,
                        make_guard(guard.tag),
                        fmt::format("{} guard at 0x{:x} tag {}", label, guard.view_offset, guard.tag)));
                }
            }
        }
    }
}

TEST_P(MimirHostDramView, BoundaryAndAlias) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const MimirTopology topology = topology_from(configured_soc_desc());
    const uint32_t view = GetParam();
    if (view >= topology.num_views) {
        GTEST_SKIP() << "Descriptor exposes " << topology.num_views << " DRAM views.";
    }
    MIMIR_REQUIRE_IDENTITY();
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(kChipId);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, kChipId, false);

    const size_t boundary_words = kBoundaryTransferBytes / sizeof(uint32_t);
    const uint64_t unique_offset = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    for (uint32_t other_view = 0; other_view < topology.num_views; ++other_view) {
        mimir_host_dram_write(cluster, soc, kChipId, other_view, unique_offset, make_guard(other_view, boundary_words));
    }
    for (uint32_t other_view = 0; other_view < topology.num_views; ++other_view) {
        EXPECT_TRUE(read_and_compare(
            cluster,
            soc,
            other_view,
            unique_offset,
            make_guard(other_view, boundary_words),
            fmt::format("view {} unique pattern at offset 0x{:x}", other_view, unique_offset)));
    }

    std::vector<uint64_t> alias_walk_offsets = {0};
    for (uint64_t offset = sizeof(uint32_t); offset < topology.view_size; offset <<= 1) {
        alias_walk_offsets.push_back(offset);
    }
    // All words are written before any is read, so a stuck or swapped address bit shows as an overwritten word.
    for (size_t index = 0; index < alias_walk_offsets.size(); ++index) {
        mimir_host_dram_write(
            cluster,
            soc,
            kChipId,
            view,
            alias_walk_offsets[index],
            make_guard(kAliasWalkTagBase + static_cast<uint32_t>(index), 1));
    }
    for (size_t index = 0; index < alias_walk_offsets.size(); ++index) {
        EXPECT_TRUE(read_and_compare(
            cluster,
            soc,
            view,
            alias_walk_offsets[index],
            make_guard(kAliasWalkTagBase + static_cast<uint32_t>(index), 1),
            fmt::format("view {} alias walk offset 0x{:x}", view, alias_walk_offsets[index])));
    }

    // The 128 KiB anchor edge is not straddled separately: every GiB edge below is also a 128 KiB edge.
    for (uint64_t centre : kStartAnchorStraddleCentres) {
        EXPECT_TRUE(write_straddle_and_read_back(cluster, soc, view, centre, "start anchor offset"));
    }
    for (uint64_t boundary = kGiB; boundary < topology.view_size; boundary += kGiB) {
        EXPECT_TRUE(write_straddle_and_read_back(cluster, soc, view, boundary, "GiB boundary"));
    }

    // When the next view on the same Mimir starts where this one ends, a write that straddles the end of this view
    // reads back through the next view at offset 0. The GDDR path resolves on the channel only, so this is
    // deterministic. Views that leave a gap between them skip this: the straddle would land outside both.
    const bool next_view_continues_this_one =
        view + 1 < topology.num_views && topology.view_mimir[view] == topology.view_mimir[view + 1] &&
        topology.view_offset[view] + topology.view_size == topology.view_offset[view + 1];
    if (next_view_continues_this_one) {
        const uint32_t next_view = view + 1;
        const uint64_t straddle_offset = topology.view_size - kBoundaryTransferBytes / 2;
        const MimirDramAddressInfo info =
            mimir_dram_address(soc, kChipId, view, straddle_offset, kBoundaryTransferBytes);
        const std::vector<uint32_t> pattern =
            make_pattern(MimirPattern::Prng, boundary_words, info.flat_address, mimir_test_seed());
        const std::string label = fmt::format("{} straddles the end of view {}", info.host_string(), view);
        EXPECT_TRUE(write_then_read_back(cluster, soc, view, straddle_offset, pattern, label));

        const std::vector<uint32_t> upper_half(pattern.begin() + boundary_words / 2, pattern.end());
        std::vector<uint32_t> next_view_words;
        mimir_host_dram_read(cluster, soc, kChipId, next_view, 0, kBoundaryTransferBytes / 2, next_view_words);
        EXPECT_TRUE(compare_words(
            fmt::format("view {} offset 0 returns the bytes written past the end of view {}", next_view, view),
            upper_half,
            next_view_words));
    }

    log_info(tt::LogTest, "Legal unaligned host DRAM access: N/A, host DRAM access is 4 B aligned only.");
}

INSTANTIATE_TEST_SUITE_P(
    MM0, MimirHostDramView, ::testing::ValuesIn(mimir_contract::dram_view_indices()), view_param_name);

TEST(MimirHostDram, ReconnectPersists) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    MIMIR_REQUIRE_IDENTITY();
    const Cluster& metal_cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& metal_soc = metal_cluster.get_soc_desc(kChipId);
    MIMIR_HOST_ONLY_GUARD(metal_cluster, metal_soc, kChipId, true);

    const MimirTopology topology = topology_from(metal_soc);
    const uint64_t unreserved = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    std::vector<MimirAnchor> persisted_anchors;
    std::vector<uint32_t> persisted_cces = {kTl1CceIndex};
    if (topology.num_mimirs > 1) {
        const std::vector<uint32_t> mimir_one_views = dram_views_for_channel(metal_soc, 1);
        ASSERT_FALSE(mimir_one_views.empty()) << "no DRAM view on Mimir 1";
        const uint32_t mimir_one_view = mimir_one_views.front();
        const std::vector<MimirAnchor> mimir_one_anchors =
            mimir_dram_anchors(metal_soc, mimir_one_view, unreserved, true);
        persisted_anchors.push_back(mimir_one_anchors.front());
        const uint32_t mimir_one_cce = cce_index_for(1, 0);
        ASSERT_LT(mimir_one_cce, topology.num_cces) << "no CCE on Mimir 1";
        persisted_cces.push_back(mimir_one_cce);
    }
    for (const MimirAnchor& anchor : mimir_dram_anchors(metal_soc, kPersistView, unreserved, true)) {
        persisted_anchors.push_back(anchor);
    }

    std::vector<PersistedDramRegion> dram_regions;
    std::vector<PersistedTl1Region> tl1_regions;
    {
        llrt::RunTimeOptions writer_options;
        Cluster writer(writer_options);
        const metal_SocDescriptor& writer_soc = writer.get_soc_desc(kChipId);
        for (const MimirAnchor& anchor : persisted_anchors) {
            const uint32_t view = anchor.view;
            const MimirDramAddressInfo info =
                mimir_dram_address(writer_soc, kChipId, view, anchor.view_offset, kRegionBytes);
            PersistedDramRegion region{
                view,
                anchor.name,
                anchor.view_offset,
                make_pattern(
                    MimirPattern::Prng, kRegionBytes / sizeof(uint32_t), info.flat_address, mimir_test_seed())};
            mimir_host_dram_write(writer, writer_soc, kChipId, view, region.view_offset, region.expected);
            dram_regions.push_back(region);
        }
        for (uint32_t cce_index : persisted_cces) {
            const MimirCceL1AddressInfo info =
                mimir_cce_l1_address(writer_soc, kChipId, cce_index, tl1_data(0), kRegionBytes);
            const std::vector<uint32_t> expected = make_pattern(
                MimirPattern::Prng, kRegionBytes / sizeof(uint32_t), info.host_axi_address, mimir_test_seed());
            write_cce_tl1(writer, kChipId, cce_index, tl1_data(0), expected, Tl1Range::WholeTl1);
            tl1_regions.push_back({cce_index, tl1_data(0), expected});
        }
    }

    llrt::RunTimeOptions reader_options;
    Cluster reader(reader_options);
    const metal_SocDescriptor& reader_soc = reader.get_soc_desc(kChipId);
    for (const PersistedDramRegion& region : dram_regions) {
        EXPECT_TRUE(read_and_compare(
            reader,
            reader_soc,
            region.view,
            region.view_offset,
            region.expected,
            fmt::format(
                "DRAM {} view {} offset 0x{:x} after reconnect", region.name, region.view, region.view_offset)));
    }
    for (const PersistedTl1Region& region : tl1_regions) {
        const std::vector<uint32_t> tl1_actual =
            read_cce_tl1(reader, kChipId, region.cce_index, region.tl1_offset, kRegionBytes, Tl1Range::WholeTl1);
        EXPECT_TRUE(compare_words(
            fmt::format("CCE {} TL1 data after reconnect", region.cce_index), region.expected, tl1_actual));
    }
}

TEST(MimirHostDram, HostSideRejections) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    MIMIR_REQUIRE_IDENTITY();
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(kChipId);
    MIMIR_HOST_ONLY_GUARD(cluster, soc, kChipId, true);
    const uint64_t mimir_window_bytes = tt::umd::mimir_local_address_windows(soc).dram_stride;
    const uint64_t l1_noc_offset = MetalContext::instance().hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);

    constexpr uint32_t kView = 0;
    constexpr uint32_t kRejectedBytes = 8;
    std::vector<uint32_t> words(2, 0xA5A5A5A5u);
    using UmdError = tt::umd::error::UmdBaseException;

    // write_dram_vec sends to the view's preferred core, so that core is the one to check.
    const CoreCoord preferred_dram_core = soc.get_preferred_worker_core_for_dram_view(kView, NOC::NOC_0);
    uint32_t tl1_cce_index = 0;
    bool found_tl1_cce = false;
    for (uint32_t cce_index = 0; cce_index < num_cces_from_soc(soc) && !found_tl1_cce; ++cce_index) {
        const CoreCoord translated = cce_virtual_core(cluster, kChipId, cce_index);
        if (translated.x == preferred_dram_core.x && translated.y == preferred_dram_core.y) {
            tl1_cce_index = cce_index;
            found_tl1_cce = true;
        }
    }
    ASSERT_TRUE(found_tl1_cce) << "view " << kView << " preferred core has no CCE";

    const Hal& hal = MetalContext::instance().hal();
    const uint64_t last_word_of_stride = mimir_window_bytes - sizeof(uint32_t);
    const uint64_t tl1_last_word_offset = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE) +
                                          hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE) -
                                          sizeof(uint32_t);
    mimir_host_dram_write(cluster, soc, kChipId, kView, 0, make_guard(kRejectSentinelTag, 2));
    mimir_host_dram_write(cluster, soc, kChipId, kView, last_word_of_stride, make_guard(kStrideEndSentinelTag, 1));
    write_cce_tl1(
        cluster, kChipId, tl1_cce_index, tl1_last_word_offset, make_guard(kTl1EndSentinelTag, 1), Tl1Range::WholeTl1);

    // Rejected before any word is sent.
    for (uint64_t offset : {mimir_window_bytes, 64 * kGiB}) {
        EXPECT_THROW(cluster.write_dram_vec(words.data(), kRejectedBytes, kChipId, kView, offset), UmdError);
        EXPECT_THROW(cluster.read_dram_vec(words.data(), kRejectedBytes, kChipId, kView, offset), UmdError);
    }
    EXPECT_THROW(cluster.write_dram_vec(words.data(), kRejectedBytes, kChipId, kView, last_word_of_stride), UmdError);
    EXPECT_THROW(cluster.read_dram_vec(words.data(), kRejectedBytes, kChipId, kView, last_word_of_stride), UmdError);
    const uint64_t past_tl1_window = l1_noc_offset +
                                     hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE) +
                                     hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE);
    EXPECT_THROW(cluster.write_dram_vec(words.data(), sizeof(uint32_t), kChipId, kView, past_tl1_window), UmdError);
    EXPECT_THROW(cluster.read_dram_vec(words.data(), sizeof(uint32_t), kChipId, kView, past_tl1_window), UmdError);
    EXPECT_THROW(cluster.write_dram_vec(words.data(), sizeof(uint32_t), kChipId, kView, 2), std::invalid_argument);
    EXPECT_THROW(cluster.read_dram_vec(words.data(), sizeof(uint32_t), kChipId, kView, 2), std::invalid_argument);

    EXPECT_TRUE(read_and_compare(
        cluster,
        soc,
        kView,
        0,
        make_guard(kRejectSentinelTag, 2),
        "DRAM offset 0 untouched by the misaligned and out-of-window offset rejections"));
    EXPECT_TRUE(read_and_compare(
        cluster,
        soc,
        kView,
        last_word_of_stride,
        make_guard(kStrideEndSentinelTag, 1),
        "DRAM last word of the stride untouched by the 8-byte overflow rejection"));
    EXPECT_TRUE(compare_words(
        "TL1 last word untouched by the past-TL1 rejection",
        make_guard(kTl1EndSentinelTag, 1),
        read_cce_tl1(cluster, kChipId, tl1_cce_index, tl1_last_word_offset, sizeof(uint32_t), Tl1Range::WholeTl1)));

    // A DRAM-API access at the L1 tag lands in that CCE's TL1. It touches no GDDR page.
    const uint64_t tl1_offset = tl1_data(0);
    const uint32_t tl1_bytes = 4 * sizeof(uint32_t);
    const std::vector<uint32_t> poison = make_guard(kL1PoisonTag, 4);
    write_cce_tl1(cluster, kChipId, tl1_cce_index, tl1_offset, poison, Tl1Range::WholeTl1);
    ASSERT_TRUE(compare_words(
        "TL1 poison before the DRAM API write",
        poison,
        read_cce_tl1(cluster, kChipId, tl1_cce_index, tl1_offset, tl1_bytes, Tl1Range::WholeTl1)));

    const std::vector<uint32_t> tl1_words = make_guard(kL1PatternTag, 4);
    mimir_host_dram_write(cluster, soc, kChipId, kView, tl1_offset, make_guard(kTl1AliasGddrTag, 4));
    cluster.write_dram_vec(tl1_words.data(), tl1_bytes, kChipId, kView, l1_noc_offset + tl1_offset);
    EXPECT_TRUE(compare_words(
        "DRAM API write at the L1 tag, read through CCE TL1",
        tl1_words,
        read_cce_tl1(cluster, kChipId, tl1_cce_index, tl1_offset, tl1_bytes, Tl1Range::WholeTl1)));
    EXPECT_TRUE(read_and_compare(
        cluster,
        soc,
        kView,
        tl1_offset,
        make_guard(kTl1AliasGddrTag, 4),
        "GDDR at the TL1 offset untouched by the DRAM API write at the L1 tag"));

    // A write of 6 bytes to an aligned address does not throw. The last word takes the 2 written bytes and keeps
    // its other 2 bytes.
    const uint64_t partial_offset = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t prefill_bytes = 3 * sizeof(uint32_t);
    const MimirDramAddressInfo prefill_info = mimir_dram_address(soc, kChipId, kView, partial_offset, prefill_bytes);
    ASSERT_TRUE(write_then_read_back(
        cluster,
        soc,
        kView,
        partial_offset,
        make_pattern(MimirPattern::AllOnes, 3, prefill_info.flat_address, mimir_test_seed()),
        "Prefill before the 6-byte write"));

    const std::vector<uint8_t> partial_bytes = {0x11, 0x12, 0x13, 0x14, 0x15, 0x16};
    const uint32_t partial_size = static_cast<uint32_t>(partial_bytes.size());
    cluster.write_dram_vec(partial_bytes.data(), partial_size, kChipId, kView, partial_offset);

    std::vector<uint32_t> readback(3, 0);
    cluster.read_dram_vec(readback.data(), prefill_bytes, kChipId, kView, partial_offset);
    const std::vector<uint32_t> expected_readback = {0x14131211u, 0xFFFF1615u, 0xFFFFFFFFu};
    EXPECT_TRUE(
        compare_words("6-byte write at an aligned DRAM address, readback of 12 bytes", expected_readback, readback));
}

}  // namespace tt::tt_metal::mimir_test::host_dram
