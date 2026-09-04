// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Deriving the factory-descriptor host filter from a live descriptor, and the checks that stop a bad
// pairing before any rank commits to it. All offline: descriptors are built in memory or from a temp file.
//
// The multi-rank half of agree_or_throw_fsd_host_filter -- ranks that disagree without any of them failing
// locally -- needs two ranks with different inputs and lives in the driver, not here.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>
#include <tt-metalium/experimental/fabric/physical_node_id.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "protobuf/factory_system_descriptor.pb.h"
#include "protobuf/physical_system_descriptor.pb.h"
#include "tt_metal/fabric/fsd_host_filter.hpp"

namespace tt::tt_metal::experimental::tt_fabric {
namespace {

using FSD = ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor;

// One tray of N300 per host, a chain of cross-host cables, and one intra-host cable per host. Enough
// structure that filtering out a host has to drop cables as well as ASICs.
FSD make_fsd(const std::vector<std::string>& hostnames) {
    FSD fsd;
    for (const auto& hostname : hostnames) {
        auto* host = fsd.add_hosts();
        host->set_hostname(hostname);
        host->set_motherboard("mobo");
    }
    auto* board_types = fsd.mutable_board_types();
    for (uint32_t host_id = 0; host_id < hostnames.size(); ++host_id) {
        auto* location = board_types->add_board_locations();
        location->set_host_id(host_id);
        location->set_tray_id(0);
        location->set_board_type("N300");
    }

    auto add_conn = [&fsd](uint32_t ha, uint32_t aa, uint32_t ca, uint32_t hb, uint32_t ab, uint32_t cb) {
        auto* connection = fsd.mutable_eth_connections()->add_connection();
        auto* a = connection->mutable_endpoint_a();
        a->set_host_id(ha);
        a->set_tray_id(0);
        a->set_asic_location(aa);
        a->set_chan_id(ca);
        auto* b = connection->mutable_endpoint_b();
        b->set_host_id(hb);
        b->set_tray_id(0);
        b->set_asic_location(ab);
        b->set_chan_id(cb);
    };
    for (uint32_t host_id = 0; host_id < hostnames.size(); ++host_id) {
        add_conn(host_id, 0, 0, host_id, 1, 0);  // intra-host
        if (host_id + 1 < hostnames.size()) {
            add_conn(host_id, 1, 1, host_id + 1, 0, 1);  // to the next host
        }
    }
    return fsd;
}

::tt::tt_metal::PhysicalSystemDescriptor descriptor_for(const std::vector<std::string>& hostnames) {
    return ::tt::tt_metal::PhysicalSystemDescriptor(build_physical_descriptor(make_fsd(hostnames)));
}

// A temp file holding an FSD textproto, so the path-taking API can be exercised.
class FsdFile {
public:
    explicit FsdFile(const FSD& fsd, const std::string& name) : path_(std::filesystem::temp_directory_path() / name) {
        std::ofstream out(path_);
        out << fsd.DebugString();
    }
    ~FsdFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    FsdFile(const FsdFile&) = delete;
    FsdFile& operator=(const FsdFile&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

// The filter is the live descriptor's host set. There is no other source for it, on purpose.
TEST(FsdHostFilter, FilterIsTheLiveHostSet) {
    const FsdFile file(make_fsd({"hosta", "hostb", "hostc"}), "fhf_live_host_set.textproto");
    const auto live = descriptor_for({"hosta", "hostb", "hostc"});

    const auto hosts = fsd_host_filter_from_live(file.path(), live);

    EXPECT_EQ(std::set<std::string>(hosts.begin(), hosts.end()), (std::set<std::string>{"hosta", "hostb", "hostc"}));
}

// The point of the filter: a job on part of a superpod ingests only its part.
TEST(FsdHostFilter, ALiveSubsetSelectsOnlyThatSubset) {
    const FsdFile file(make_fsd({"hosta", "hostb", "hostc"}), "fhf_subset.textproto");
    const auto live = descriptor_for({"hosta", "hostb"});

    const auto hosts = fsd_host_filter_from_live(file.path(), live);
    ASSERT_EQ(hosts.size(), 2u);

    FilterReport report;
    const auto filtered = build_physical_descriptor_from_file(file.path(), hosts, &report);

    EXPECT_EQ(filtered.get_all_hostnames().size(), 2u);
    EXPECT_EQ(report.fsd_host_count, 3u);
    EXPECT_EQ(report.retained_host_count, 2u);
    // The hostb<->hostc cable had an endpoint on a dropped host.
    EXPECT_GT(report.dropped_connection_count, 0u);
}

// The live descriptor's host keys and the descriptor author's spelling need not agree. They must still
// join, or the filter silently retains nothing.
TEST(FsdHostFilter, FqdnAndShortSpellingsOfOneHostJoin) {
    const FsdFile file(make_fsd({"hosta", "hostb"}), "fhf_fqdn.textproto");
    const auto live = descriptor_for({"hosta.dc.example.com", "hostb.dc.example.com"});

    const auto hosts = fsd_host_filter_from_live(file.path(), live);

    EXPECT_EQ(std::set<std::string>(hosts.begin(), hosts.end()), (std::set<std::string>{"hosta", "hostb"}));
    EXPECT_EQ(build_physical_descriptor_from_file(file.path(), hosts).get_all_hostnames().size(), 2u);
}

TEST(FsdHostFilter, CaseDoesNotSplitTheJoin) {
    const FsdFile file(make_fsd({"hosta", "hostb"}), "fhf_case.textproto");
    const auto live = descriptor_for({"HostA", "HOSTB"});

    EXPECT_EQ(fsd_host_filter_from_live(file.path(), live).size(), 2u);
}

// A different operator problem from a partial mismatch -- the wrong descriptor, not a stale one -- so it
// gets its own message rather than a list of every live host being "missing".
TEST(FsdHostFilter, ZeroOverlapIsItsOwnError) {
    const FsdFile file(make_fsd({"hosta", "hostb"}), "fhf_zero_overlap.textproto");
    const auto live = descriptor_for({"other1", "other2"});

    try {
        fsd_host_filter_from_live(file.path(), live);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("does not describe this system"), std::string::npos) << e.what();
    }
}

TEST(FsdHostFilter, ALiveHostAbsentFromTheDescriptorIsRejected) {
    const FsdFile file(make_fsd({"hosta", "hostb"}), "fhf_absent.textproto");
    // Overlaps on hosta, so this is a stale descriptor rather than the wrong one.
    const auto live = descriptor_for({"hosta", "hostz"});

    try {
        fsd_host_filter_from_live(file.path(), live);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("not present"), std::string::npos) << what;
        EXPECT_NE(what.find("hostz"), std::string::npos) << what;
    }
}

// Every absent host at once. One wrong allocation is several hosts, and a per-host abort makes the
// operator re-run init once per host to discover the next one.
TEST(FsdHostFilter, AllAbsentHostsAreReportedTogether) {
    const FsdFile file(make_fsd({"hosta", "hostb"}), "fhf_absent_all.textproto");
    const auto live = descriptor_for({"hosta", "hostx", "hosty"});

    try {
        fsd_host_filter_from_live(file.path(), live);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("hostx"), std::string::npos) << what;
        EXPECT_NE(what.find("hosty"), std::string::npos) << what;
    }
}

// Two live hosts under one canonical name make the address join ambiguous, which would attach one
// machine's cables to the other.
TEST(FsdHostFilter, LiveHostsThatCanonicalizeAlikeAreRejected) {
    const FsdFile file(make_fsd({"hosta"}), "fhf_ambiguous_live.textproto");
    const auto live = descriptor_for({"hosta.dc1.example.com", "hosta.dc2.example.com"});

    try {
        fsd_host_filter_from_live(file.path(), live);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("not distinct after canonicalization"), std::string::npos) << e.what();
    }
}

// The same collision on the descriptor side: one requested name would retain two machines and double
// their chips in the mesh.
TEST(FsdHostFilter, DescriptorHostsThatCanonicalizeAlikeAreRejected) {
    const FsdFile file(make_fsd({"hosta.dc1.example.com", "hosta.dc2.example.com"}), "fhf_ambiguous_fsd.textproto");
    const auto live = descriptor_for({"hosta"});

    try {
        fsd_host_filter_from_live(file.path(), live);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("same host after canonicalization"), std::string::npos) << e.what();
    }
}

TEST(FsdHostFilter, AnUnreadableDescriptorIsRejected) {
    const auto live = descriptor_for({"hosta"});
    EXPECT_THROW(fsd_host_filter_from_live("/nonexistent/path/fsd.textproto", live), std::runtime_error);
}

// Every ingest-time failure has to be reachable from the derivation, because that is what runs before the
// ranks agree. A check that only fires during the build would abort some ranks mid-ingest.
TEST(FsdHostFilter, DerivationRejectsWhateverTheIngestWould) {
    const FsdFile file(make_fsd({"hosta", "hostb"}), "fhf_dry_run.textproto");
    const auto live = descriptor_for({"hosta", "hostz"});

    EXPECT_THROW(fsd_host_filter_from_live(file.path(), live), std::runtime_error);
    // The same pairing, ingested directly, fails the same way.
    EXPECT_THROW(build_physical_descriptor_from_file(file.path(), {"hosta", "hostz"}), std::runtime_error);
}

// A filtered-out neighbour leaves the edge host's outward channels unconnected in the factory descriptor
// while live still shows them cabled. That has to read as an extra cable, not a downed one: the other
// reading reports the entire allocation boundary as broken on every healthy job.
TEST(FsdHostFilter, CablesLeavingTheAllocationAreNotDowned) {
    const FsdFile file(make_fsd({"hosta", "hostb", "hostc"}), "fhf_boundary.textproto");
    const auto live = descriptor_for({"hosta", "hostb"});

    const auto hosts = fsd_host_filter_from_live(file.path(), live);
    const auto expected = build_physical_descriptor_from_file(file.path(), hosts);

    // Live still has the hostb<->hostc cable; the filtered factory descriptor does not.
    const auto live_with_boundary = descriptor_for({"hosta", "hostb", "hostc"});
    const auto delta = ::tt::tt_metal::diff_physical_system_descriptors(expected, live_with_boundary);

    EXPECT_TRUE(delta.missing_links.empty());
    EXPECT_TRUE(delta.missing_asics.empty());
    EXPECT_FALSE(delta.extra_links.empty());
}

TEST(FsdHostFilter, ChecksumIgnoresOrderAndSeparatesNeighbours) {
    EXPECT_EQ(checksum_sorted_host_list({"a", "b", "c"}), checksum_sorted_host_list({"c", "a", "b"}));
    EXPECT_NE(checksum_sorted_host_list({"a", "b"}), checksum_sorted_host_list({"a", "b", "c"}));
    // Without a separator these two lists would hash alike, and two ranks holding different allocations
    // would agree.
    EXPECT_NE(checksum_sorted_host_list({"ab", "c"}), checksum_sorted_host_list({"a", "bc"}));
}

// Catches what the host checksum cannot: ranks agreeing on which hosts to use while reading different
// files for them, which is what a half-rolled-out asset update looks like.
TEST(FsdHostFilter, FingerprintTracksFileContents) {
    const FsdFile one(make_fsd({"hosta", "hostb"}), "fhf_fp_one.textproto");
    const FsdFile same(make_fsd({"hosta", "hostb"}), "fhf_fp_same.textproto");
    const FsdFile other(make_fsd({"hosta", "hostb", "hostc"}), "fhf_fp_other.textproto");

    EXPECT_EQ(fsd_fingerprint(one.path()), fsd_fingerprint(same.path()));
    EXPECT_NE(fsd_fingerprint(one.path()), fsd_fingerprint(other.path()));
    EXPECT_NE(fsd_fingerprint(one.path()), 0u);
    // Unreadable is 0 rather than a throw: the caller is already reporting that through local_ok.
    EXPECT_EQ(fsd_fingerprint("/nonexistent/path/fsd.textproto"), 0u);
}

// A local failure has to come back as a collective throw even when there is only one rank, because that
// is the path a real failure takes on every rank at once.
TEST(FsdHostFilter, ALocalFailureThrowsThroughTheAgreement) {
    const auto& world = ::tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

    EXPECT_NO_THROW(agree_or_throw_fsd_host_filter(*world, 0x1234, 0x5678, true));

    try {
        agree_or_throw_fsd_host_filter(*world, 0x1234, 0x5678, false);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("not identical on every rank"), std::string::npos) << what;
        EXPECT_NE(what.find("local_ok_min=0"), std::string::npos) << what;
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental::tt_fabric
