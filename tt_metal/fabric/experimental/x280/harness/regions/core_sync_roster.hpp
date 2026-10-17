// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/regions/core_sync_roster.hpp -- Host-side participant roster setup
// for core_sync_barrier.

#ifndef X280_HARNESS_REGIONS_CORE_SYNC_ROSTER_HPP
#define X280_HARNESS_REGIONS_CORE_SYNC_ROSTER_HPP

#include "harness/metal_context.hpp"
#include "core_sync_arrival.hpp"
#include "core_sync_layout.h"
#include "core_sync_transport_metal.hpp"
#include "core_sync_transport_x280.hpp"
#include "regions.hpp"
#include "release_selector.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>

#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace x280::harness::regions {

/* Per-worker barrier configuration and assignment. logical_core identifies the
 * participant within the grid. endpoint describes this participant as a release
 * destination whether or not the director writes it: under a multicast release
 * a Tensix participant is reached by the group instead and does not appear in
 * worker_endpoints. metal_launched is false for a participant Metal does not
 * launch -- an L2CPU hart. Such a participant has no logical core, so
 * logical_core repeats its NoC coordinates; it takes no share of the program's
 * semaphores and is never inside the multicast group. */
struct CoreSyncWorkerInfo {
    tt::tt_metal::CoreCoord logical_core;
    core_sync::MetalEndpoint endpoint{};
    bool metal_launched = true;
};

/* Host-side coordinator roster for core_sync_barrier. Contains director NoC
 * coordinates, ready-table base in director L1, the release and arrival
 * semaphore IDs, per-worker assignments, the flat array of endpoints passed to
 * notify_group, and the word-arrival table passed to wait_for_arrivals.
 *
 * counter_arrivals is how many participants increment the arrival counter. The
 * counter's director-local address is not resolved here: a semaphore ID becomes
 * an address only on the core that owns it, so the director kernel resolves
 * arrival_semaphore_id itself and fills counter_addr in the roster it builds. */
struct CoreSyncRoster {
    tt::tt_metal::CoreCoord director_logical;
    TranslatedCoord director_noc;
    uint32_t release_semaphore_id = 0;
    uint32_t arrival_semaphore_id = 0;
    uint32_t counter_arrivals = 0;
    std::vector<CoreSyncWorkerInfo> workers{};
    std::vector<core_sync::MetalEndpoint> worker_endpoints{};
    std::vector<core_sync::WordArrival> word_arrivals{};

    /* The selector's group, in translated coordinates, with num_dests the
     * logical count of the Tensix workers it releases (never the director).
     * Empty when the selector chose endpoints, where every worker is in
     * worker_endpoints instead. */
    core_sync::MetalGroup grid{};

    /* The release roster and the word-arrival table, as the director transport
     * takes them. counter_addr is left zero for the director kernel to fill;
     * the returned view borrows this object's vectors and does not outlive it. */
    core_sync::MetalRoster roster() const {
        return core_sync::MetalRoster{
            worker_endpoints.data(),
            static_cast<uint32_t>(worker_endpoints.size()),
            /*counter_addr=*/0u,
            counter_arrivals,
            word_arrivals.data(),
            static_cast<uint32_t>(word_arrivals.size()),
            grid,
        };
    }
};

struct CoreSyncMemoryRequirements {
    /* Raw bytes required in the director's local ready table. */
    uint64_t director_ready_bytes = 0;
    /* True when any participant arrives through the Metal counter. */
    bool director_counter_required = false;
    /* One raw release region per participant without a Metal semaphore. */
    uint32_t raw_release_count = 0;
};

struct CoreSyncMemoryBindings {
    /* Caller-owned regions that must outlive barrier execution. */
    Region director_ready{};
    std::vector<Region> raw_releases{};
};

namespace detail {

/* The core family a participant transport is written for, as the SoC descriptor
 * names it. Undefined for a transport that names no family: a roster group is
 * checked against the coordinates it is given, and a transport that cannot say
 * what it runs on cannot be checked. */
template <typename Transport>
struct ParticipantCoreFamily;

template <ProgrammableCoreType CoreTypeV, core_sync::ArrivalKind Arrival>
struct ParticipantCoreFamily<core_sync::MetalTransport<CoreTypeV, Arrival>> {
    static constexpr tt::CoreType value = CoreTypeV == ProgrammableCoreType::DRAM         ? tt::CoreType::DRAM
                                          : CoreTypeV == ProgrammableCoreType::ACTIVE_ETH ? tt::CoreType::ACTIVE_ETH
                                          : CoreTypeV == ProgrammableCoreType::IDLE_ETH   ? tt::CoreType::IDLE_ETH
                                                                                          : tt::CoreType::TENSIX;
};

template <uint32_t TlbSlot>
struct ParticipantCoreFamily<core_sync::X280Transport<TlbSlot>> {
    static constexpr tt::CoreType value = tt::CoreType::L2CPU;
};

/* Whether the core type a SoC descriptor reports answers to the family a
 * transport claims. A descriptor names a Tensix core TENSIX or WORKER and an
 * ethernet core ETH, ACTIVE_ETH or IDLE_ETH, so those stand for one another;
 * every other family must match exactly. */
inline bool core_family_matches(tt::CoreType found, tt::CoreType claimed) {
    if (found == claimed) {
        return true;
    }
    switch (claimed) {
        case tt::CoreType::TENSIX: return found == tt::CoreType::WORKER;
        case tt::CoreType::ACTIVE_ETH:
        case tt::CoreType::IDLE_ETH:
            return found == tt::CoreType::ETH || found == tt::CoreType::ACTIVE_ETH || found == tt::CoreType::IDLE_ETH;
        default: return false;
    }
}

/* Validates one device-local byte region against a required extent. */
inline void validate_core_sync_region(const Region& region, uint64_t required_bytes, const std::string& what) {
    if (required_bytes == 0u) {
        if (region.addr != 0u || region.bytes != 0u) {
            throw std::invalid_argument(
                "CoreSyncRosterBuilder::build: " + what + " must be empty when no storage is required");
        }
        return;
    }
    if (region.addr == 0u) {
        throw std::invalid_argument("CoreSyncRosterBuilder::build: " + what + " base must be nonzero");
    }
    if ((region.addr % CORE_SYNC_SLOT_BYTES) != 0u) {
        throw std::invalid_argument(
            "CoreSyncRosterBuilder::build: " + what + " base " + std::to_string(region.addr) + " is not aligned to " +
            std::to_string(CORE_SYNC_SLOT_BYTES) + " bytes");
    }
    if ((region.bytes % CORE_SYNC_SLOT_BYTES) != 0u) {
        throw std::invalid_argument(
            "CoreSyncRosterBuilder::build: " + what + " size " + std::to_string(region.bytes) + " is not aligned to " +
            std::to_string(CORE_SYNC_SLOT_BYTES) + " bytes");
    }
    if (region.bytes < required_bytes) {
        throw std::invalid_argument(
            "CoreSyncRosterBuilder::build: " + what + " size " + std::to_string(region.bytes) +
            " is smaller than required size " + std::to_string(required_bytes));
    }
    if (region.addr > UINT32_MAX || region.bytes > UINT32_MAX || region.addr + region.bytes < region.addr ||
        region.addr + region.bytes > static_cast<uint64_t>(UINT32_MAX) + 1u) {
        throw std::invalid_argument(
            "CoreSyncRosterBuilder::build: " + what + " does not fit in the 32-bit device-local address space");
    }
}

/* Rejects raw memory bindings that cannot hold the computed topology. */
inline void validate_core_sync_memory_bindings(
    const CoreSyncMemoryRequirements& requirements, const CoreSyncMemoryBindings& memory) {
    validate_core_sync_region(memory.director_ready, requirements.director_ready_bytes, "director ready region");
    if (memory.raw_releases.size() != requirements.raw_release_count) {
        throw std::invalid_argument(
            "CoreSyncRosterBuilder::build: raw release binding count " + std::to_string(memory.raw_releases.size()) +
            " does not match required count " + std::to_string(requirements.raw_release_count));
    }
    for (size_t i = 0; i < memory.raw_releases.size(); ++i) {
        validate_core_sync_region(
            memory.raw_releases[i], CORE_SYNC_SLOT_BYTES, "raw release region " + std::to_string(i));
    }
}

/* Refuses a participant coordinate that is not of the core family its transport
 * is written for. noc is a translated coordinate, the space
 * SocDescriptor::get_coord_at resolves against CoordSystem::TRANSLATED, and
 * `what` names the participant in the message.
 *
 * A harvested core is refused by the same path: it is a real entry of the
 * descriptor and matches no transport's family, so it earns its own message
 * rather than a family mismatch. Nothing runs there, so a roster naming one
 * waits for an arrival that never comes.
 *
 * Throws std::invalid_argument naming the coordinate, the type found and the
 * type expected; also when the coordinate resolves to no core at all. */
inline void require_core_family(
    const metal_SocDescriptor& soc, const TranslatedCoord& noc, tt::CoreType claimed, const std::string& what) {
    const std::string where =
        what + " at translated core (" + std::to_string(noc.x()) + ", " + std::to_string(noc.y()) + ")";
    tt::umd::CoreCoord found;
    try {
        found = soc.get_coord_at(tt_xy_pair(noc.x(), noc.y()), tt::CoordSystem::TRANSLATED);
    } catch (const std::exception& e) {
        throw std::invalid_argument(
            "core_sync roster: " + where + " resolves to no core on this device (expected " + tt::to_str(claimed) +
            "): " + e.what());
    }
    if (found.core_type == tt::CoreType::HARVESTED) {
        throw std::invalid_argument(
            "core_sync roster: " + where + " is " + tt::to_str(found.core_type) + ", not the " + tt::to_str(claimed) +
            " its transport is written for; nothing runs on a harvested core, so it "
            "never arrives");
    }
    if (!core_family_matches(found.core_type, claimed)) {
        throw std::invalid_argument(
            "core_sync roster: " + where + " is " + tt::to_str(found.core_type) + ", not the " + tt::to_str(claimed) +
            " its transport is written for");
    }
}

/* Whether Metal launches the core a transport is written for. A Metal-launched
 * transport names the programmable core type Metal resolves its kernel and its
 * semaphores against, kCoreType; a transport for a core outside Metal's launch
 * path -- an L2CPU hart -- declares none. Such a participant holds no
 * semaphore and is released at a raw L1 address. */
template <typename Transport, typename = void>
struct IsMetalLaunched : std::false_type {};

template <typename Transport>
struct IsMetalLaunched<Transport, std::void_t<decltype(Transport::kCoreType)>> : std::true_type {};

/* The core type Metal's semaphore machinery names a participant family by.
 * The HAL registers Tensix as WORKER and both ethernet families as ETH
 * (tt-metal tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp:153,
 * bh_hal_active_eth.cpp:219), and both the per-core id allocation
 * (program.cpp:1717) and the sizing and initialization of the semaphore region
 * (dispatch.cpp:286-294,1133-1157) match a semaphore's core type against that
 * registration exactly. A semaphore allocated under another spelling of the
 * same family is therefore never sized and never written, and shares no id
 * space with the one the kernel resolves.
 *
 * Throws std::invalid_argument naming the family for one Metal does not
 * launch, which holds no Metal semaphore at all. */
inline tt::CoreType semaphore_core_type(tt::CoreType family) {
    switch (family) {
        case tt::CoreType::TENSIX:
        case tt::CoreType::WORKER: return tt::CoreType::WORKER;
        case tt::CoreType::ACTIVE_ETH:
        case tt::CoreType::IDLE_ETH:
        case tt::CoreType::ETH: return tt::CoreType::ETH;
        case tt::CoreType::DRAM:
            /* Metal allocates a DRAM semaphore id but never writes its initial
             * value: dispatch.cpp:1133-1157 emits a transfer for a WORKER
             * semaphore and for an ETH one, and has no branch for any other
             * core type. An uninitialized release semaphore may already carry
             * the expected token. Such a group needs an explicit zeroing path
             * before it can share the program's semaphores. */
            throw std::invalid_argument(
                "core_sync roster: a DRAM participant cannot take a share of the "
                "program's semaphores, because Metal allocates a DRAM semaphore id "
                "without ever writing its initial value, and an unzeroed release "
                "semaphore releases its worker before the director runs");
        default:
            throw std::invalid_argument(
                "core_sync roster: a participant of core family " + tt::to_str(family) +
                " takes no share of the program's semaphores, because Metal does not "
                "launch that family; such a participant is released at a raw L1 address "
                "instead");
    }
}

/* The core type the release selector resolves a family's logical coordinate
 * against: Metal resolves both ethernet families as ETH. */
inline tt::CoreType resolver_core_type(tt::CoreType family) {
    return family == tt::CoreType::ACTIVE_ETH || family == tt::CoreType::IDLE_ETH ? tt::CoreType::ETH : family;
}

}  // namespace detail

/* Builds a CoreSyncRoster for a barrier with the given director, one worker
 * group at a time. Each group names the transport its cores run; its arrival
 * kind follows from that transport.
 *
 * DirectorTransport names the transport the director core will run, and decides
 * whether it may hold an arrival counter at all.
 *
 * mode chooses the release through the shared release selector: under Auto the
 * Tensix workers take one multicast group when their logical coordinates fill a
 * logical rectangle -- the director may fill one cell, since Metal's
 * non-loopback multicast does not write to self (dataflow_api.h:904-907) -- and
 * every other participant takes an Endpoint. UnicastOnly gives every worker an
 * Endpoint; RequireMulticast refuses a roster the group cannot release.
 *
 * The builder holds program and mesh by reference and must not outlive either.
 * mesh must be a live, local unit mesh. */
template <typename DirectorTransport = core_sync::MetalTransport<ProgrammableCoreType::TENSIX>>
class CoreSyncRosterBuilder {
public:
    /* director_logical is a Metal logical core, so the director must be one
     * Metal launches. A director outside Metal's launch path -- an L2CPU hart
     * -- has no logical core to name, and its roster is built by hand from
     * translated coordinates rather than here. */
    static_assert(
        detail::IsMetalLaunched<DirectorTransport>::value,
        "CoreSyncRosterBuilder identifies the director by a Metal logical core, "
        "which a director Metal does not launch does not have");

    /* director_noc is the NoC the director kernel issues its release on; every
     * coordinate is resolved for it, and only NoC 0 multicasts. */
    CoreSyncRosterBuilder(
        tt::tt_metal::Program& program,
        const tt::tt_metal::distributed::MeshDevice& mesh,
        const tt::tt_metal::CoreCoord& director_logical,
        tt::tt_metal::NOC director_noc,
        ReleaseMode mode = ReleaseMode::Auto) :
        program_(program), mesh_(mesh), director_logical_(director_logical), director_noc_(director_noc), mode_(mode) {}

    /* Adds every core of worker_core_set as a participant running
     * WorkerTransport, which fixes the group's arrival kind, core family, and
     * release-storage policy.
     * Counter-kind participants contribute to counter_arrivals and get no ready
     * word; word-kind participants take dense worker indices from zero, in the
     * order groups are added and cores within a group are enumerated.
     *
     * Throws std::invalid_argument, naming the offending value, if the group
     * arrives by counter and the director's transport cannot host one, or if
     * director_logical is in worker_core_set. A core not of the family
     * WorkerTransport is written for is refused by build().
     *
     * Returns *this so groups chain. */
    template <typename WorkerTransport>
    CoreSyncRosterBuilder& add_workers(const tt::tt_metal::CoreRangeSet& worker_core_set) {
        constexpr core_sync::ArrivalKind kind = WorkerTransport::kArrivalKind;
        constexpr ProgrammableCoreType core_type = WorkerTransport::kCoreType;
        constexpr bool semaphore_backed = WorkerTransport::kUsesSemaphoreRelease;
        static_assert(
            core_type != ProgrammableCoreType::DRAM || kind == core_sync::ArrivalKind::Word,
            "a DRISC worker must announce arrival through a private word");
        const std::vector<tt::tt_metal::CoreCoord> worker_coords = tt::tt_metal::corerange_to_cores(worker_core_set);
        refuse_counter_group<kind>(worker_coords.size());

        for (const auto& w_logical : worker_coords) {
            if (w_logical == director_logical_) {
                throw std::invalid_argument(
                    "CoreSyncRosterBuilder::add_workers: director core (" + std::to_string(director_logical_.x) + ", " +
                    std::to_string(director_logical_.y) + ") must not be in worker set");
            }
            pending_.push_back(PendingWorker{
                .family = detail::ParticipantCoreFamily<WorkerTransport>::value,
                .where = w_logical,
                .kind = kind,
                .core_type = core_type,
                .metal_launched = true,
                .semaphore_backed = semaphore_backed,
            });
        }
        if constexpr (semaphore_backed) {
            tt::tt_metal::CoreRangeSet& family_cores =
                metal_cores_[detail::semaphore_core_type(detail::ParticipantCoreFamily<WorkerTransport>::value)];
            family_cores = family_cores.merge(worker_core_set);
        }
        return *this;
    }

    /* Adds one participant Metal does not launch -- an L2CPU hart -- by its
     * translated NoC coordinate, resolved for the director's NoC. It takes no
     * share of the program's semaphores and is never inside the multicast
     * group. Its release region is bound by external-participant order in
     * build().
     *
     * Throws std::invalid_argument if the group arrives by counter and the
     * director's transport cannot host one.
     *
     * Returns *this so groups chain. */
    template <typename WorkerTransport>
    CoreSyncRosterBuilder& add_external_worker(const TranslatedCoord& noc) {
        constexpr core_sync::ArrivalKind kind = WorkerTransport::kArrivalKind;
        refuse_counter_group<kind>(1u);
        pending_.push_back(PendingWorker{
            .family = detail::ParticipantCoreFamily<WorkerTransport>::value,
            .where = noc,
            .kind = kind,
            .core_type = ProgrammableCoreType::TENSIX,
            .metal_launched = false,
            .semaphore_backed = false,
        });
        return *this;
    }

    /* Returns the raw storage required by the participants added so far. */
    CoreSyncMemoryRequirements memory_requirements() const {
        CoreSyncMemoryRequirements requirements;
        for (const auto& pending : pending_) {
            if (pending.kind == core_sync::ArrivalKind::Word) {
                requirements.director_ready_bytes += CORE_SYNC_SLOT_BYTES;
            } else {
                requirements.director_counter_required = true;
            }
            if (!pending.semaphore_backed) {
                ++requirements.raw_release_count;
            }
        }
        return requirements;
    }

    /* Resolves the groups added so far into a roster through select_release.
     *
     * Allocates two semaphores on program across every Metal-launched
     * participant and the director via
     * CreateSemaphore, both at initial value 0:
     * the release semaphore the director writes, and the arrival counter the
     * counter-kind workers increment. One pair per core type the participants
     * span, because a semaphore id belongs to a (logical core, core type) pair
     * and the families' logical spaces overlap; every pair must land on the same
     * two ids, since the roster carries one of each for all participants. Metal
     * allows eight per core
     * (tt-metal/tt_metal/api/tt-metalium/host_api.hpp:301); this takes the
     * second. Zeroing the counter per run is what makes a counter arrival fresh,
     * which is why a counter arrival writes no token.
     *
     * memory supplies the director's ready table and each raw release region.
     * The referenced storage must outlive barrier execution.
     * Invalid bindings are rejected before program semaphores are allocated.
     * A participant or director not of its transport's core family -- a
     * harvested core included -- other invalid topology, and the selector's
     * refusals (an unsatisfiable RequireMulticast among them) throw
     * std::invalid_argument; resolution failures propagate as the resolver
     * throws them. */
    CoreSyncRoster build(const CoreSyncMemoryBindings& memory = {}) {
        const CoreSyncMemoryRequirements requirements = memory_requirements();
        detail::validate_core_sync_memory_bindings(requirements, memory);

        std::vector<ReleaseParticipant> participants;
        participants.reserve(pending_.size());
        for (const auto& pending : pending_) {
            participants.push_back({detail::resolver_core_type(pending.family), pending.where});
        }
        constexpr tt::CoreType director_family = detail::ParticipantCoreFamily<DirectorTransport>::value;
        ReleaseSender sender{.noc = director_noc_, .can_multicast = DirectorTransport::kHasMulticast};
        if constexpr (director_family == tt::CoreType::TENSIX) {
            sender.non_loopback_tensix = director_logical_;
        }
        const ReleasePlan plan =
            select_release(mesh_, participants, sender, mode_, static_cast<uint32_t>(participants.size()));

        const metal_SocDescriptor& soc = soc_desc();
        for (size_t i = 0; i < pending_.size(); ++i) {
            detail::require_core_family(soc, plan.translated[i], pending_[i].family, describe(pending_[i]));
        }
        const TranslatedCoord director_noc = X280Runtime::virtual_core_from_logical_core(
            mesh_, director_logical_, detail::resolver_core_type(director_family), director_noc_);
        detail::require_core_family(
            soc,
            director_noc,
            director_family,
            "director at logical core (" + std::to_string(director_logical_.x) + ", " +
                std::to_string(director_logical_.y) + ")");

        std::map<tt::CoreType, tt::tt_metal::CoreRangeSet> sem_cores = metal_cores_;
        const tt::tt_metal::CoreRangeSet director_set(tt::tt_metal::CoreRange(director_logical_, director_logical_));
        tt::tt_metal::CoreRangeSet& director_cores = sem_cores[detail::semaphore_core_type(director_family)];
        director_cores = director_cores.merge(director_set);

        /* The director's own core type always has a non-empty set, so the loop
         * below allocates at least once and these are always overwritten. */
        uint32_t release_sem = 0;
        uint32_t arrival_sem = 0;
        bool allocated = false;
        tt::CoreType allocated_type = tt::CoreType::WORKER;
        for (const auto& [core_type, cores] : sem_cores) {
            if (cores.ranges().empty()) {
                continue;
            }
            const uint32_t release = tt::tt_metal::CreateSemaphore(program_, cores, 0, core_type);
            const uint32_t arrival = tt::tt_metal::CreateSemaphore(program_, cores, 0, core_type);
            if (!allocated) {
                release_sem = release;
                arrival_sem = arrival;
                allocated_type = core_type;
                allocated = true;
            } else if (release != release_sem || arrival != arrival_sem) {
                throw std::invalid_argument(
                    "CoreSyncRosterBuilder::build: the " + tt::to_str(core_type) +
                    " participants were allocated release semaphore " + std::to_string(release) +
                    " and arrival semaphore " + std::to_string(arrival) + ", but the " + tt::to_str(allocated_type) +
                    " participants were allocated " + std::to_string(release_sem) + " and " +
                    std::to_string(arrival_sem) +
                    "; every participant shares one release id and one arrival id, and they "
                    "diverge because the program already had semaphores allocated on one "
                    "core type's cores and not the other's before the roster was built");
            }
        }

        CoreSyncRoster roster{
            .director_logical = director_logical_,
            .director_noc = director_noc,
            .release_semaphore_id = release_sem,
            .arrival_semaphore_id = arrival_sem,
        };
        const uint32_t director_ready_base = static_cast<uint32_t>(memory.director_ready.addr);

        roster.workers.reserve(pending_.size());
        roster.worker_endpoints.reserve(plan.endpoints.size());

        uint32_t next_word_index = 0;
        uint32_t next_raw_release = 0;
        for (size_t i = 0; i < pending_.size(); ++i) {
            const PendingWorker& pending = pending_[i];
            const TranslatedCoord& noc = plan.translated[i];
            const uint32_t release_addr =
                pending.semaphore_backed ? 0u : static_cast<uint32_t>(memory.raw_releases[next_raw_release++].addr);
            const core_sync::MetalEndpoint ep =
                pending.semaphore_backed
                    ? core_sync::MetalEndpoint(noc.noc(), release_sem, pending.core_type, /*is_sem=*/true)
                    : core_sync::MetalEndpoint::raw_l1(noc.noc(), release_addr, pending.core_type);

            if (pending.kind == core_sync::ArrivalKind::Counter) {
                ++roster.counter_arrivals;
            } else {
                const uint32_t word_index = next_word_index++;
                const uint32_t slot_addr = core_sync_slot_addr(director_ready_base, word_index);
                roster.word_arrivals.push_back(core_sync::WordArrival{slot_addr, word_index});
            }

            roster.workers.push_back(CoreSyncWorkerInfo{
                .logical_core = pending.metal_launched ? std::get<tt::tt_metal::CoreCoord>(pending.where)
                                                       : tt::tt_metal::CoreCoord{noc.x(), noc.y()},
                .endpoint = ep,
                .metal_launched = pending.metal_launched,
            });
        }
        for (const uint32_t i : plan.endpoints) {
            roster.worker_endpoints.push_back(roster.workers[i].endpoint);
        }
        if (plan.group) {
            roster.grid = core_sync::MetalGroup{plan.group->rect(), release_sem, plan.group->num_dests};
        }

        if constexpr (!DirectorTransport::kMayHostCounter) {
            if (roster.counter_arrivals > 0u) {
                throw std::invalid_argument(
                    "CoreSyncRosterBuilder::build: " + std::to_string(roster.counter_arrivals) +
                    " counter arrivals were asked of a director whose transport cannot "
                    "host an arrival counter; those participants must arrive by word");
            }
        }

        return roster;
    }

private:
    struct PendingWorker {
        tt::CoreType family = tt::CoreType::TENSIX;
        /* The logical core of a Metal-launched participant, or the translated
         * coordinate of one Metal does not launch. */
        std::variant<tt::tt_metal::CoreCoord, TranslatedCoord> where;
        core_sync::ArrivalKind kind = core_sync::ArrivalKind::Counter;

        /* The family the release endpoint resolves its semaphore id against.
         * Read only for a Metal-launched participant; one Metal does not
         * launch takes a raw L1 address and names no family. */
        ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX;
        bool metal_launched = true;
        bool semaphore_backed = true;
    };

    static std::string describe(const PendingWorker& pending) {
        if (const auto* logical = std::get_if<tt::tt_metal::CoreCoord>(&pending.where)) {
            return "worker at logical core (" + std::to_string(logical->x) + ", " + std::to_string(logical->y) + ")";
        }
        return "external worker";
    }

    const metal_SocDescriptor& soc_desc() const { return x280::harness::cluster().get_soc_desc(mesh_.build_id()); }

    /* Refuses a counter-arriving group when the director cannot host one. */
    template <core_sync::ArrivalKind Kind>
    void refuse_counter_group(size_t group_size) const {
        if constexpr (Kind == core_sync::ArrivalKind::Counter && !DirectorTransport::kMayHostCounter) {
            throw std::invalid_argument(
                "CoreSyncRosterBuilder: a group of " + std::to_string(group_size) +
                " workers whose transport arrives by counter was added to a director at "
                "logical core (" +
                std::to_string(director_logical_.x) + ", " + std::to_string(director_logical_.y) +
                ") whose transport cannot host an arrival counter; that group must run a "
                "transport that arrives by word");
        } else {
            (void)group_size;
        }
    }

    tt::tt_metal::Program& program_;
    const tt::tt_metal::distributed::MeshDevice& mesh_;
    tt::tt_metal::CoreCoord director_logical_;
    tt::tt_metal::NOC director_noc_;
    ReleaseMode mode_;
    std::vector<PendingWorker> pending_{};

    /* Metal-launched logical cores keyed by the core type their semaphores use.
     * A semaphore id belongs to a (logical core, core type) pair. */
    std::map<tt::CoreType, tt::tt_metal::CoreRangeSet> metal_cores_{};
};

/* Builds a CoreSyncRoster whose workers are one homogeneous group: every core
 * of worker_core_set runs WorkerTransport, and arrives by
 * WorkerTransport::kArrivalKind. A roster of several groups is built with
 * CoreSyncRosterBuilder directly.
 *
 * Throws std::invalid_argument on the grounds CoreSyncRosterBuilder's
 * add_workers and build state. */
template <
    typename DirectorTransport = core_sync::MetalTransport<ProgrammableCoreType::TENSIX>,
    typename WorkerTransport = core_sync::MetalTransport<ProgrammableCoreType::TENSIX>>
inline CoreSyncRoster make_core_sync_roster(
    tt::tt_metal::Program& program,
    const tt::tt_metal::distributed::MeshDevice& mesh,
    const tt::tt_metal::CoreRangeSet& worker_core_set,
    const tt::tt_metal::CoreCoord& director_logical,
    tt::tt_metal::NOC director_noc,
    const CoreSyncMemoryBindings& memory,
    ReleaseMode mode = ReleaseMode::Auto) {
    CoreSyncRosterBuilder<DirectorTransport> builder(program, mesh, director_logical, director_noc, mode);
    builder.template add_workers<WorkerTransport>(worker_core_set);
    return builder.build(memory);
}

}  // namespace x280::harness::regions

#endif  // X280_HARNESS_REGIONS_CORE_SYNC_ROSTER_HPP
