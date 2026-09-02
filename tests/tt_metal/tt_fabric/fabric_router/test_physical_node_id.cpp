// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Offline unit tests for PhysicalNodeId: the packed (host_id, tray, loc) address the topology
// solver keys on. No device, no descriptors -- just the encoding and its canonicalization.

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/physical_node_id.hpp>

namespace tt::tt_metal {
namespace {

constexpr std::string_view kHost = "bh-glx-110-c01u02";

PhysicalNodeId node(std::string_view host_id, uint32_t tray, uint32_t loc, bool hosts_unique = true) {
    return make_physical_node_id(host_id, TrayID{tray}, ASICLocation{loc}, hosts_unique);
}

TEST(PhysicalNodeIdTest, SameAddressGivesSameId) { EXPECT_EQ(node(kHost, 1, 2), node(kHost, 1, 2)); }

// An FQDN, a short name and a differently-cased name are the same machine, and different sides of a
// join report different ones.
TEST(PhysicalNodeIdTest, CanonicalizationCollapsesFqdnAndCase) {
    const PhysicalNodeId expected = node(kHost, 1, 2);

    EXPECT_EQ(node("bh-glx-110-c01u02.tenstorrent.com", 1, 2), expected);
    EXPECT_EQ(node("BH-GLX-110-C01U02", 1, 2), expected);
    EXPECT_EQ(node("BH-GLX-110-C01U02.Tenstorrent.COM", 1, 2), expected);
}

TEST(PhysicalNodeIdTest, RankSuffixStrippedOnlyWhenHostsAreNotUnique) {
    // The suffix is what discovery appends as hostname + "_" + rank when two ranks collide.
    EXPECT_EQ(node("bh-glx-110-c01u02_3", 1, 2, /*hosts_unique=*/false), node(kHost, 1, 2));

    // With unique hosts there is no suffix to strip, so the id keeps the trailing text.
    EXPECT_NE(node("bh-glx-110-c01u02_3", 1, 2, /*hosts_unique=*/true), node(kHost, 1, 2));
}

// Only a run of digits is a rank suffix; a host id that genuinely ends in "_word" is left alone.
TEST(PhysicalNodeIdTest, NonNumericTrailingSegmentIsNotARankSuffix) {
    EXPECT_EQ(
        decode_physical_node_id(node("metal-wh-09_spare", 1, 2, /*hosts_unique=*/false)).host_id, "metal-wh-09_spare");
    EXPECT_EQ(decode_physical_node_id(node("host_", 1, 2, /*hosts_unique=*/false)).host_id, "host_");
}

TEST(PhysicalNodeIdTest, DifferentComponentGivesDifferentId) {
    const PhysicalNodeId base = node(kHost, 1, 2);

    EXPECT_NE(base, node(kHost, 1, 3));                // loc
    EXPECT_NE(base, node(kHost, 2, 2));                // tray
    EXPECT_NE(base, node("bh-glx-110-c01u08", 1, 2));  // host id
    EXPECT_NE(base, node(kHost, 2, 1));                // tray and loc swapped -- not symmetric
}

TEST(PhysicalNodeIdTest, DecodeRestoresTheAddress) {
    const PhysicalNodeFields fields = decode_physical_node_id(node("BH-GLX-110-C01U02.local", 3, 7));

    EXPECT_EQ(fields.host_id, kHost);  // canonical, not what was passed in
    EXPECT_EQ(fields.tray, TrayID{3});
    EXPECT_EQ(fields.loc, ASICLocation{7});
}

TEST(PhysicalNodeIdTest, UnsetIdIsAllZeroesAndMakeNeverReturnsIt) {
    const PhysicalNodeId unset{};

    EXPECT_TRUE(is_unset(unset));
    EXPECT_TRUE(host_id_view(unset).empty());
    EXPECT_EQ(unset.tray, TrayID{0});
    EXPECT_EQ(unset.loc, ASICLocation{0});

    // Tray and loc of 0 are legal values; what makes an id unset is the empty host id, and that
    // fatals rather than being packed.
    EXPECT_FALSE(is_unset(node(kHost, 0, 0)));
}

// TT_FATAL throws std::runtime_error (unless TT_ASSERT_ABORT is set in the environment).
TEST(PhysicalNodeIdTest, IllegalAddressIsFatal) {
    EXPECT_THROW(node("", 1, 2), std::runtime_error);
    EXPECT_THROW(node(".", 1, 2), std::runtime_error);                 // first label is empty
    EXPECT_THROW(node(".tenstorrent.com", 1, 2), std::runtime_error);  // ditto
    // Nothing left once the rank suffix comes off.
    EXPECT_THROW(node("_7", 1, 2, /*hosts_unique=*/false), std::runtime_error);

    // Not truncated: a truncated host id would silently collide with its neighbours.
    EXPECT_THROW(node(std::string(kPhysicalHostNameLen, 'a'), 1, 2), std::runtime_error);
    EXPECT_THROW(node(std::string(kPhysicalHostNameLen + 1, 'a'), 1, 2), std::runtime_error);
    EXPECT_NO_THROW(node(std::string(kPhysicalHostNameLen - 1, 'a'), 1, 2));

    EXPECT_THROW(node(kHost, 0x10000, 2), std::runtime_error);
    EXPECT_THROW(node(kHost, 1, 0x10000), std::runtime_error);
    EXPECT_NO_THROW(node(kHost, 0xffff, 0xffff));
}

// Pins the exact bytes, so a change to the packing shows up here rather than as a topology that
// silently stops matching the FSD.
TEST(PhysicalNodeIdTest, GoldenVector) {
    const PhysicalNodeId id = node(kHost, 4, 6);

    ASSERT_EQ(sizeof(id.host_id), kPhysicalHostNameLen);
    EXPECT_EQ(std::string(id.host_id, kHost.size()), kHost);
    for (std::size_t i = kHost.size(); i < kPhysicalHostNameLen; ++i) {
        EXPECT_EQ(id.host_id[i], '\0') << "byte " << i << " is not NUL padding";
    }
    EXPECT_EQ(id.tray, TrayID{4});
    EXPECT_EQ(id.loc, ASICLocation{6});
}

// std::map iteration order is what the solver erases to dense indices, so the order has to be
// host_id bytes, then tray, then loc.
TEST(PhysicalNodeIdTest, SortOrderIsHostThenTrayThenLoc) {
    const PhysicalNodeId a = node("host-a", 1, 1);
    const PhysicalNodeId b = node("host-b", 1, 1);
    const PhysicalNodeId a_tray2 = node("host-a", 2, 1);
    const PhysicalNodeId a_loc2 = node("host-a", 1, 2);

    EXPECT_LT(a, b);
    EXPECT_LT(a, a_tray2);
    EXPECT_LT(a, a_loc2);
    EXPECT_LT(a_loc2, a_tray2);  // tray outranks loc
    EXPECT_LT(a_tray2, b);       // host id outranks both
}

// Two maps built from the same addresses in different insertion orders have to compare equal --
// that is what makes an FSD-built graph and a live-discovered graph the same graph.
TEST(PhysicalNodeIdTest, MapOrderIsIndependentOfInsertionOrder) {
    const std::vector<PhysicalNodeId> ids = {
        node("bh-glx-110-c01u02", 1, 0),
        node("bh-glx-110-c01u02", 2, 0),
        node("bh-glx-110-c01u08", 1, 0),
        node("bh-glx-110-c01u08.tenstorrent.com", 1, 1),
    };

    std::map<PhysicalNodeId, int> forward;
    std::map<PhysicalNodeId, int> reverse;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        forward[ids[i]] = static_cast<int>(i);
        reverse[ids[ids.size() - 1 - i]] = static_cast<int>(ids.size() - 1 - i);
    }

    ASSERT_EQ(forward.size(), ids.size());
    EXPECT_EQ(forward, reverse);
    EXPECT_TRUE(std::equal(forward.begin(), forward.end(), reverse.begin(), [](const auto& l, const auto& r) {
        return l.first == r.first;
    }));
}

TEST(PhysicalNodeIdTest, UsableAsUnorderedContainerKey) {
    std::unordered_map<PhysicalNodeId, int> by_id;
    by_id[node(kHost, 1, 2)] = 7;

    // A differently-spelled but equal address has to hit the same bucket.
    EXPECT_EQ(by_id.at(node("BH-GLX-110-C01U02.tenstorrent.com", 1, 2)), 7);
    EXPECT_EQ(by_id.count(node(kHost, 1, 3)), 0u);

    std::unordered_set<PhysicalNodeId> ids;
    ids.insert(node(kHost, 1, 2));
    ids.insert(node(kHost, 1, 2));
    EXPECT_EQ(ids.size(), 1u);

    // Equal ids must hash equally (the padding-is-zero property the static_asserts protect).
    EXPECT_EQ(
        std::hash<PhysicalNodeId>{}(node(kHost, 1, 2)),
        std::hash<PhysicalNodeId>{}(node("bh-glx-110-c01u02.tenstorrent.com", 1, 2)));
}

TEST(PhysicalNodeIdTest, CanonicalHostIsExposedForTheFsdHostFilter) {
    // The FSD host filter joins on the same canonical string, so it is public and has to agree with
    // what make_physical_node_id packs.
    EXPECT_EQ(canonical_host_for_node_id("BH-GLX-110-C01U02.tenstorrent.com"), kHost);
    EXPECT_EQ(canonical_host_for_node_id("bh-glx-110-c01u02_3", /*hosts_unique=*/false), kHost);
    EXPECT_EQ(canonical_host_for_node_id("bh-glx-110-c01u02_3", /*hosts_unique=*/true), "bh-glx-110-c01u02_3");
    EXPECT_EQ(canonical_host_for_node_id(""), "");

    EXPECT_EQ(
        decode_physical_node_id(node("BH-GLX-110-C01U02.tenstorrent.com", 1, 2)).host_id,
        canonical_host_for_node_id("BH-GLX-110-C01U02.tenstorrent.com"));
}

// Mock + FSD is the case the whole encoding exists for: a mock descriptor carrying the UMD host_id
// has to pack the same id as the FSD builder, and the YAML basename must not.
TEST(PhysicalNodeIdTest, MockWithHostIdMatchesFsdButBasenameDoesNot) {
    const PhysicalNodeId from_fsd = node(kHost, 1, 2);
    const PhysicalNodeId from_umd_host_id = node(kHost, 1, 2);
    const PhysicalNodeId from_basename = node("SC20_32x4_revAB_aisleC_cluster_desc_bh-glx-c01u02_rank_0.yaml", 1, 2);

    EXPECT_EQ(from_umd_host_id, from_fsd);
    EXPECT_NE(from_basename, from_fsd);

    // The SC36 case: the filename token disagrees with the FSD on the hall, and the id follows the
    // FSD because that is what the descriptor's host_id now carries.
    EXPECT_EQ(node("bh-glx-110-d10u20", 1, 2), node("bh-glx-110-d10u20", 1, 2));
    EXPECT_NE(node("bh-glx-120-d10u20", 1, 2), node("bh-glx-110-d10u20", 1, 2));
}

TEST(PhysicalNodeIdTest, FormattingShowsTheWholeAddress) {
    const std::string formatted = fmt::format("{}", node(kHost, 4, 6));

    EXPECT_NE(formatted.find(kHost), std::string::npos);
    EXPECT_NE(formatted.find('4'), std::string::npos);
    EXPECT_NE(formatted.find('6'), std::string::npos);
}

}  // namespace
}  // namespace tt::tt_metal
