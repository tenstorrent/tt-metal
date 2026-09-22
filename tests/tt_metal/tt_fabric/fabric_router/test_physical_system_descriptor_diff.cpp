// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Offline tests for diff_physical_system_descriptors. Everything here builds descriptors in memory:
// the diff joins on physical position, so what matters is which addresses and channels each side
// declares, not where the descriptor came from.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/physical_node_id.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "llrt/tt_target_device.hpp"

namespace tt::tt_metal {
namespace {

// An ASIC to declare: its address, the label this descriptor happens to give it, and its board.
struct AsicSpec {
    std::string host;
    uint32_t tray = 1;
    uint32_t loc = 0;
    uint64_t label = 0;
    BoardType board = BoardType::N300;
};

// A cable to declare, by the labels of the ASICs it joins.
struct CableSpec {
    uint64_t src_label = 0;
    uint8_t src_chan = 0;
    uint64_t dst_label = 0;
    uint8_t dst_chan = 0;
    PortType port_type = PortType::TRACE;
    bool is_local = true;
};

// Builds a descriptor from those declarations. Each cable is recorded from both ends, which is what
// discovery does -- a cross-host cable appears in each host's own topology.
PhysicalSystemDescriptor make_descriptor(
    const std::vector<AsicSpec>& asics, const std::vector<CableSpec>& cables, bool mirror_cables = true) {
    PhysicalSystemDescriptor descriptor(tt::TargetDevice::Silicon);
    auto& graph = descriptor.get_system_graph().asic_connectivity_graph;

    for (const auto& asic : asics) {
        const AsicID id{asic.label};
        descriptor.get_asic_descriptors()[id] = ASICDescriptor{
            TrayID{asic.tray}, ASICLocation{asic.loc}, asic.board, id, static_cast<ChipId>(asic.label), asic.host};
        graph[asic.host][id];  // an ASIC with no cables still exists
    }

    auto host_of = [&asics](uint64_t label) {
        for (const auto& asic : asics) {
            if (asic.label == label) {
                return asic.host;
            }
        }
        return std::string{};
    };
    auto record = [&graph](const std::string& host, uint64_t src, uint64_t dst, const EthConnection& connection) {
        graph[host][AsicID{src}].emplace_back(AsicID{dst}, std::vector<EthConnection>{connection});
    };

    for (const auto& cable : cables) {
        record(
            host_of(cable.src_label),
            cable.src_label,
            cable.dst_label,
            EthConnection{cable.src_chan, cable.dst_chan, cable.is_local, cable.port_type});
        if (mirror_cables) {
            record(
                host_of(cable.dst_label),
                cable.dst_label,
                cable.src_label,
                EthConnection{cable.dst_chan, cable.src_chan, cable.is_local, cable.port_type});
        }
    }
    return descriptor;
}

// Two chips on one host, one chip on another, cabled in a line: a->b locally, b->c across hosts.
std::vector<AsicSpec> three_asics(uint64_t base) {
    return {
        AsicSpec{"host-a", 1, 0, base + 0},
        AsicSpec{"host-a", 1, 1, base + 1},
        AsicSpec{"host-b", 1, 0, base + 2},
    };
}

std::vector<CableSpec> three_asic_cables(uint64_t base) {
    return {
        CableSpec{base + 0, 0, base + 1, 1, PortType::TRACE, true},
        CableSpec{base + 1, 2, base + 2, 3, PortType::QSFP_DD, false},
    };
}

// Directed records in a link map, as (src, dst, src_chan) triples, for compact assertions.
std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> directed_records(const AsicTopology& links) {
    std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> records;
    for (const auto& [src, edges] : links) {
        for (const auto& [dst, connections] : edges) {
            for (const auto& connection : connections) {
                records.emplace_back(*src, *dst, connection.src_chan);
            }
        }
    }
    std::sort(records.begin(), records.end());
    return records;
}

std::size_t count_directed(const AsicTopology& links) { return directed_records(links).size(); }

TEST(PhysicalSystemDescriptorDiff, IdenticalDescriptorsMatch) {
    const auto golden = make_descriptor(three_asics(1), three_asic_cables(1));
    const auto candidate = make_descriptor(three_asics(1), three_asic_cables(1));

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_TRUE(delta.matches());
}

// The reason this function joins on position. A descriptor built from a factory system descriptor
// labels its ASICs 1..N in file order; a discovered one labels them with UMD chip unique ids. The
// two label spaces share nothing, and the descriptors still describe the same hardware.
TEST(PhysicalSystemDescriptorDiff, DisjointAsicIdSpacesWithTheSameAddressesMatch) {
    const auto golden = make_descriptor(three_asics(1), three_asic_cables(1));
    const auto candidate = make_descriptor(three_asics(0x9a3f'0000), three_asic_cables(0x9a3f'0000));

    // The labels really are disjoint, or this test proves nothing.
    for (const auto& [golden_id, _] : golden.get_asic_descriptors()) {
        EXPECT_FALSE(candidate.get_asic_descriptors().contains(golden_id));
    }

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_TRUE(delta.matches());
}

// Hosts may be spelled differently on either side; the shared canonicalization collapses them.
TEST(PhysicalSystemDescriptorDiff, HostSpellingDoesNotSplitTheJoin) {
    const auto golden = make_descriptor({AsicSpec{"host-a", 1, 0, 1}}, {});
    const auto candidate = make_descriptor({AsicSpec{"HOST-A.local.example.com", 1, 0, 77}}, {});

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_TRUE(delta.matches());
}

TEST(PhysicalSystemDescriptorDiff, MissingCableIsReportedFromBothEnds) {
    const auto golden = make_descriptor(three_asics(1), three_asic_cables(1));
    // Drop the cross-host cable (2 chan 2 <-> 3 chan 3).
    const auto candidate = make_descriptor(three_asics(1), {three_asic_cables(1).front()});

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_TRUE(delta.extra_links.empty());
    EXPECT_TRUE(delta.mismatched_links.empty());
    EXPECT_TRUE(delta.missing_asics.empty());
    // One cable, both directed halves, so a caller keying on the source end sees it either way.
    EXPECT_EQ(
        directed_records(delta.missing_links),
        (std::vector<std::tuple<uint64_t, uint64_t, uint8_t>>{{2, 3, 2}, {3, 2, 3}}));
}

TEST(PhysicalSystemDescriptorDiff, SwappingTheArgumentsTurnsMissingIntoExtra) {
    const auto full = make_descriptor(three_asics(1), three_asic_cables(1));
    const auto reduced = make_descriptor(three_asics(1), {three_asic_cables(1).front()});

    const auto missing = diff_physical_system_descriptors(full, reduced);
    const auto extra = diff_physical_system_descriptors(reduced, full);

    EXPECT_EQ(directed_records(missing.missing_links), directed_records(extra.extra_links));
    EXPECT_TRUE(missing.extra_links.empty());
    EXPECT_TRUE(extra.missing_links.empty());
}

// FSD is golden: a cable live has and the factory descriptor does not is not a downed link.
TEST(PhysicalSystemDescriptorDiff, ExtraCableIsNotMissing) {
    auto cables = three_asic_cables(1);
    cables.push_back(CableSpec{1, 8, 3, 9, PortType::QSFP_DD, false});

    const auto golden = make_descriptor(three_asics(1), three_asic_cables(1));
    const auto candidate = make_descriptor(three_asics(1), cables);

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_TRUE(delta.missing_links.empty());
    EXPECT_EQ(count_directed(delta.extra_links), 2u);
    EXPECT_FALSE(delta.matches());
}

TEST(PhysicalSystemDescriptorDiff, DroppedAndAddedCableAreBothReported) {
    const auto golden = make_descriptor(three_asics(1), three_asic_cables(1));
    const auto candidate = make_descriptor(
        three_asics(1),
        {
            three_asic_cables(1).front(),
            CableSpec{2, 20, 3, 21, PortType::QSFP_DD, false},  // a different pair of channels
        });

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_EQ(count_directed(delta.missing_links), 2u);
    EXPECT_EQ(count_directed(delta.extra_links), 2u);
}

TEST(PhysicalSystemDescriptorDiff, MissingAndExtraAsicsAreReported) {
    const auto golden = make_descriptor(three_asics(1), {});
    const auto candidate = make_descriptor(
        {
            AsicSpec{"host-a", 1, 0, 1},
            AsicSpec{"host-a", 1, 1, 2},
            AsicSpec{"host-b", 1, 5, 9},  // a different location than golden's third chip
        },
        {});

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_EQ(delta.missing_asics, std::vector<AsicID>{AsicID{3}});
    EXPECT_EQ(delta.extra_asics, std::vector<AsicID>{AsicID{9}});
    EXPECT_TRUE(delta.mismatched_asics.empty());
}

// Tray is part of the address, so a chip that moved is not one chip described differently -- it is
// one address that lost its chip and another that gained one.
TEST(PhysicalSystemDescriptorDiff, MovedChipReadsAsMissingPlusExtraNotMismatched) {
    const auto golden = make_descriptor({AsicSpec{"host-a", 1, 0, 1}}, {});
    const auto candidate = make_descriptor({AsicSpec{"host-a", 2, 0, 1}}, {});

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_EQ(delta.missing_asics, std::vector<AsicID>{AsicID{1}});
    EXPECT_EQ(delta.extra_asics, std::vector<AsicID>{AsicID{1}});
    EXPECT_TRUE(delta.mismatched_asics.empty());
}

TEST(PhysicalSystemDescriptorDiff, BoardTypeMismatchAtTheSameAddressIsReported) {
    const auto golden = make_descriptor({AsicSpec{"host-a", 1, 0, 1, BoardType::N300}}, {});
    const auto candidate = make_descriptor({AsicSpec{"host-a", 1, 0, 1, BoardType::P150}}, {});

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_EQ(delta.mismatched_asics, std::vector<AsicID>{AsicID{1}});
    EXPECT_TRUE(delta.missing_asics.empty());
    EXPECT_TRUE(delta.extra_asics.empty());
}

// A re-typed cable is still plugged in. Reporting it as missing-plus-extra would make it look like
// one cable was pulled and another added.
TEST(PhysicalSystemDescriptorDiff, RetypedCableIsMismatchedNotMissingAndExtra) {
    auto cables = three_asic_cables(1);
    const auto golden = make_descriptor(three_asics(1), cables);
    cables.back().port_type = PortType::WARP400;
    const auto candidate = make_descriptor(three_asics(1), cables);

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_TRUE(delta.missing_links.empty());
    EXPECT_TRUE(delta.extra_links.empty());
    EXPECT_EQ(count_directed(delta.mismatched_links), 2u);
    EXPECT_FALSE(delta.matches());
}

TEST(PhysicalSystemDescriptorDiff, CableThatChangedLocalityIsMismatched) {
    auto cables = three_asic_cables(1);
    const auto golden = make_descriptor(three_asics(1), cables);
    cables.front().is_local = false;
    const auto candidate = make_descriptor(three_asics(1), cables);

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_EQ(count_directed(delta.mismatched_links), 2u);
    EXPECT_TRUE(delta.missing_links.empty());
    EXPECT_TRUE(delta.extra_links.empty());
}

// The descriptor stores a cross-host cable twice, once under each host. That is one cable, so a
// descriptor compared against itself must not report anything, and a missing one must be counted
// once rather than once per stored copy.
TEST(PhysicalSystemDescriptorDiff, OneCableIsOneComparisonNotTwo) {
    const auto mirrored = make_descriptor(three_asics(1), three_asic_cables(1), /*mirror_cables=*/true);
    const auto one_sided = make_descriptor(three_asics(1), three_asic_cables(1), /*mirror_cables=*/false);

    // Whether the descriptor happens to store both halves does not change the comparison.
    EXPECT_TRUE(diff_physical_system_descriptors(mirrored, one_sided).matches());
    EXPECT_TRUE(diff_physical_system_descriptors(one_sided, mirrored).matches());
}

// AsicTopology lets the same destination appear in several entries for one source, which is what
// the descriptor does when it discovers channels one at a time. A diff that compared entries rather
// than cables would report the extra entries as missing.
TEST(PhysicalSystemDescriptorDiff, RepeatedDestinationEntriesAreMerged) {
    const std::vector<AsicSpec> asics{AsicSpec{"host-a", 1, 0, 1}, AsicSpec{"host-a", 1, 1, 2}};
    // Two cables between the same pair of chips, recorded as two separate entries.
    const std::vector<CableSpec> cables{
        CableSpec{1, 0, 2, 0, PortType::TRACE, true},
        CableSpec{1, 1, 2, 1, PortType::TRACE, true},
    };

    const auto golden = make_descriptor(asics, cables);
    const auto candidate = make_descriptor(asics, cables);
    EXPECT_TRUE(diff_physical_system_descriptors(golden, candidate).matches());

    // Dropping one of the two must report exactly that one.
    const auto reduced = make_descriptor(asics, {cables.front()});
    const auto delta = diff_physical_system_descriptors(golden, reduced);
    EXPECT_EQ(
        directed_records(delta.missing_links),
        (std::vector<std::tuple<uint64_t, uint64_t, uint8_t>>{{1, 2, 1}, {2, 1, 1}}));
    // Both cables share a (src, dst) pair, so the surviving one must not be split into two entries.
    EXPECT_EQ(delta.missing_links.at(AsicID{1}).size(), 1u);
}

TEST(PhysicalSystemDescriptorDiff, OutputIsSorted) {
    const std::vector<AsicSpec> asics{
        AsicSpec{"host-a", 1, 0, 30},
        AsicSpec{"host-a", 1, 1, 10},
        AsicSpec{"host-a", 1, 2, 20},
    };
    const std::vector<CableSpec> cables{
        CableSpec{30, 7, 10, 1, PortType::TRACE, true},
        CableSpec{30, 3, 20, 2, PortType::TRACE, true},
        CableSpec{30, 5, 10, 4, PortType::TRACE, true},
    };

    const auto golden = make_descriptor(asics, cables);
    const auto delta = diff_physical_system_descriptors(golden, make_descriptor(asics, {}));

    EXPECT_EQ(delta.missing_asics, std::vector<AsicID>{});
    const auto& edges = delta.missing_links.at(AsicID{30});
    ASSERT_EQ(edges.size(), 2u);
    EXPECT_LT(*edges[0].first, *edges[1].first);  // destinations ascending
    for (const auto& [dst, connections] : edges) {
        EXPECT_TRUE(std::is_sorted(connections.begin(), connections.end()));
    }
}

TEST(PhysicalSystemDescriptorDiff, MissingAsicTakesItsCablesWithIt) {
    const auto golden = make_descriptor(three_asics(1), three_asic_cables(1));
    // host-b's chip is gone, so the cross-host cable has no candidate endpoint at all.
    const auto candidate =
        make_descriptor({AsicSpec{"host-a", 1, 0, 1}, AsicSpec{"host-a", 1, 1, 2}}, {three_asic_cables(1).front()});

    const auto delta = diff_physical_system_descriptors(golden, candidate);

    EXPECT_EQ(delta.missing_asics, std::vector<AsicID>{AsicID{3}});
    EXPECT_EQ(count_directed(delta.missing_links), 2u);
    EXPECT_TRUE(delta.extra_links.empty());
}

// The positional join needs each address to name one chip. Two chips at one address would make the
// join ambiguous, so it is refused rather than silently resolved.
TEST(PhysicalSystemDescriptorDiff, DuplicateAddressIsFatal) {
    const auto duplicate = make_descriptor(
        {
            AsicSpec{"host-a", 1, 0, 1}, AsicSpec{"host-a", 1, 0, 2},  // same address, different label
        },
        {});
    const auto clean = make_descriptor({AsicSpec{"host-a", 1, 0, 1}}, {});

    EXPECT_ANY_THROW(diff_physical_system_descriptors(duplicate, clean));
    EXPECT_ANY_THROW(diff_physical_system_descriptors(clean, duplicate));
}

}  // namespace
}  // namespace tt::tt_metal
