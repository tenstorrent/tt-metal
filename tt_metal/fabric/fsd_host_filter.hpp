// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Which slice of a factory system descriptor a job should ingest, and how every rank agrees on it.
//
// An FSD in the wild describes a whole superpod or an aggregated datacenter, while a job runs on a subset.
// Ingesting the whole thing breaks the feature twice over: the mapper is handed every chip in the
// datacenter and places the mesh on hardware the job does not own, and every cable on every non-allocated
// host is factory-expected and missing from live, so the downed-link set becomes the rest of the
// datacenter.
//
// Also holds the other decision made about a factory descriptor before the mapper runs: whether the chips
// it declares are actually there.
//
// Internal header: ControlPlane is the only consumer, and nothing here needs to be installed.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

namespace tt::tt_metal::experimental::tt_fabric {

// The host names to restrict a factory descriptor to, derived from the live descriptor.
//
// Discovery ran across exactly the ranks the job owns, so the live descriptor's host set *is* the
// allocation and there is nothing else to ask. Deliberately no `--fsd-hosts` override: a second source of
// truth for "which hosts" is a bug generator, because the two lists can disagree and nothing would notice.
//
// Returns canonical names, which is what makes them comparable to the mapper's addresses.
//
// Throws, without side effects, for every reason an ingest of this pair would fail:
//   - the descriptor cannot be read or parsed;
//   - two live hosts canonicalize to one name, so the join would attach one machine's cables to another;
//   - no live host appears in the descriptor at all, which means the wrong descriptor;
//   - some live host is absent from it, which means a stale one;
//   - two of the requested descriptor hosts are the same host after canonicalization.
//
// That completeness is the point. On a multi-rank job the caller must catch this, agree on it across ranks
// via agree_or_throw_fsd_host_filter, and only then ingest -- so anything that could fail at ingest has to
// be reachable here, before any rank has committed.
std::vector<std::string> fsd_host_filter_from_live(
    const std::string& fsd_path, const ::tt::tt_metal::PhysicalSystemDescriptor& live);

// A value that is equal on two ranks exactly when they derived the same host list.
//
// Sorted before hashing, so two ranks that enumerated the same hosts in different orders still agree.
std::uint64_t checksum_sorted_host_list(const std::vector<std::string>& hosts);

// A value that is equal on two ranks exactly when they read the same descriptor file.
//
// Catches the case the host checksum cannot: every rank agreeing on which hosts to use while reading
// different files for them, which is what a partially rolled-out asset update looks like. Returns 0 for an
// unreadable path rather than throwing, since the caller is already reporting that failure through
// local_ok.
std::uint64_t fsd_fingerprint(const std::string& fsd_path);

// Return on every rank, or throw on every rank. Call before ingesting.
//
// A rank that throws locally on a bad filter leaves every other rank hanging at the next collective, so
// the decision to abort has to be collective too. Ranks can also disagree without any of them failing
// locally -- different files, or a live descriptor that is not actually global -- and that is worse than a
// clean abort: one rank maps on the factory descriptor while another maps on live, their downed-link sets
// disagree, and any collective gated on that deadlocks.
//
// The message is byte-identical on every rank, built only from the reduced scalars, so operators see one
// error rather than a different one per rank.
void agree_or_throw_fsd_host_filter(
    const ::tt::tt_metal::distributed::multihost::DistributedContext& ctx,
    std::uint64_t host_checksum,
    std::uint64_t descriptor_fingerprint,
    bool local_ok);

// Throw if the factory descriptor declares a whole ASIC that the live cluster does not have.
//
// A missing *cable* is the downed-link feature. A missing *chip* is a wrong allocation, and the two must
// not be conflated: one pulled board is a pile of factory-expected cables that all read as downed, which
// describes the failure in the least useful possible terms. So this runs at ControlPlane init, before the
// mapper is built and before any link health exists, and it is fatal.
//
// Reports every absent chip rather than the first: one pulled board is several ASICs, and a per-chip abort
// makes the operator re-run init once per chip to find the next one. The message is sorted, so ranks fed
// the same descriptors throw identical text and there is no partial-abort deadlock.
//
// Only ASICs on hosts the live descriptor enumerates are considered. Callers are expected to have filtered
// the factory descriptor to the live host set already, which makes that a no-op; it matters when discovery
// was local-only, where every remote chip would otherwise look absent and every rank would fatal on a
// healthy system.
//
// Deliberately not a check on ASIC ids: the two descriptors label chips from unrelated spaces (1..N in file
// order versus UMD chip unique ids), so this joins on address like everything else on this path.
void throw_on_fsd_chips_absent_from_live(
    const ::tt::tt_metal::PhysicalSystemDescriptor& fsd, const ::tt::tt_metal::PhysicalSystemDescriptor& live);

}  // namespace tt::tt_metal::experimental::tt_fabric
