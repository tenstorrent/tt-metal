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
#include <set>
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

// Give a factory-built descriptor the runtime facts only discovery knows, so it can be mapped on.
//
// A factory descriptor knows the shape of the machine but not this process's place in it: which host is
// running this rank, and which MPI rank each host is. Both are needed by anything that asks "is this chip
// mine", and the builder cannot supply either.
//
// Host names are matched canonically, but the descriptor's own spelling is left alone -- the local host is
// recorded under the name the descriptor uses for it, so every lookup inside the descriptor stays
// self-consistent and only the join with live has to canonicalize.
//
// Ranks are copied from live, not kept from the file. The builder numbers hosts by file order and the
// filter renumbers densely, so a factory descriptor's ranks are unrelated to the MPI ranks the job is
// actually running on.
void align_factory_descriptor_with_live(
    ::tt::tt_metal::PhysicalSystemDescriptor& fsd, const ::tt::tt_metal::PhysicalSystemDescriptor& live);

// Whether this job owns a chip subset of its hosts: TT_VISIBLE_DEVICES (which UMD honours) or
// TT_METAL_VISIBLE_DEVICES is set. Under such an allocation the live descriptor's *chip* set is the
// allocation, the way its host set is otherwise, and a factory chip absent from live on a live host is
// outside the job rather than missing -- UMD refuses to start when a device it was told to open is not
// there, so a chip that was meant to be opened cannot be silently absent.
//
// Agreed across ranks (any rank restricted => every rank prunes): one rank mapping on the pruned
// descriptor and another on the full one would disagree on the placement. Collective; call on every rank.
bool agree_job_runs_on_device_subset(const ::tt::tt_metal::distributed::multihost::DistributedContext& ctx);

// The live chip set as the addresses the builder's allocation filter takes.
std::set<::tt::tt_metal::experimental::PhysicalNodeId> live_allocation(
    const ::tt::tt_metal::PhysicalSystemDescriptor& live);

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
// healthy system. Likewise a job on a device subset filters the descriptor to live_allocation() first
// (see agree_job_runs_on_device_subset), so the chips it does not own never reach this check.
//
// Deliberately not a check on ASIC ids: the two descriptors label chips from unrelated spaces (1..N in file
// order versus UMD chip unique ids), so this joins on address like everything else on this path.
void throw_on_fsd_chips_absent_from_live(
    const ::tt::tt_metal::PhysicalSystemDescriptor& fsd, const ::tt::tt_metal::PhysicalSystemDescriptor& live);

}  // namespace tt::tt_metal::experimental::tt_fabric
