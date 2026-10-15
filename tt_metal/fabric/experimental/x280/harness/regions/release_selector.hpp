// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_HARNESS_REGIONS_RELEASE_SELECTOR_HPP
#define X280_HARNESS_REGIONS_RELEASE_SELECTOR_HPP

#include "harness/regions/release_group.hpp"
#include "x280/device/x280_runtime.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace x280::harness::regions {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;

/** How a barrier release is issued. Resolved on the host; the device executes
 * the roster the mode produced and never learns the mode. */
enum class ReleaseMode {
    /** One multicast group for a Tensix selection that exactly fills a logical
     * rectangle on a multicast-capable NoC 0 sender; endpoints otherwise. */
    Auto,
    /** Every participant takes an endpoint. */
    UnicastOnly,
    /** As Auto, but throws before launch unless the Tensix participants form
     * the group, so a test cannot silently fall back to endpoints. */
    RequireMulticast,
};

/** One release participant: a logical core of `type`, or the already-resolved
 * coordinate of one with no logical core (an L2CPU tile identity, from
 * X280Runtime::virtual_core_from_tile). A resolved participant is never
 * Tensix, because Tensix density is judged in logical space. */
struct ReleaseParticipant {
    tt::CoreType type;
    std::variant<ttm::CoreCoord, TranslatedCoord> where;
};

/** What the releasing transport can issue. */
struct ReleaseSender {
    /** The NoC the release is issued on, and the NoC every coordinate is
     * resolved for. Multicast is selected only on NoC 0, the configuration
     * the mapping contract covers. */
    ttm::NOC noc;
    bool can_multicast = false;
    /** The sender's own logical Tensix core when its multicast does not write
     * to itself (Metal's). The group may then cover that core, which neither
     * counts toward num_dests nor may be a participant. */
    std::optional<ttm::CoreCoord> non_loopback_tensix{};
};

/** An X280 releasing through its L2CPU outbound TLB window, which multicasts on
 * NoC 0. */
inline constexpr ReleaseSender kX280ReleaseSender{.noc = ttm::NOC::NOC_0, .can_multicast = true};

/** A release roster, in the shapes the wire rosters carry. Addresses stay with
 * the caller: a group's common address and each endpoint's are the caller's
 * to attach. */
struct ReleasePlan {
    /** Every participant's coordinate, in input order: the arrival table. */
    std::vector<TranslatedCoord> translated;
    std::optional<ReleaseGroup> group;
    /** Input indices of the participants released one write each, ascending. */
    std::vector<uint32_t> endpoints;
};

namespace detail {

inline bool is_tensix(tt::CoreType type) { return type == tt::CoreType::TENSIX || type == tt::CoreType::WORKER; }

/** Why the Tensix participants cannot form one group, or nullopt when they
 * can. `tensix` is nonempty and duplicate-free. */
inline std::optional<std::string> multicast_refusal(
    const std::set<ttm::CoreCoord>& tensix, const ReleaseSender& sender, ttm::CoreCoord* lo, ttm::CoreCoord* hi) {
    if (!sender.can_multicast) {
        return std::string("the sender's transport cannot multicast");
    }
    if (sender.noc != ttm::NOC::NOC_0) {
        return std::string("multicast is selected only on NoC 0");
    }
    *lo = *tensix.begin();
    *hi = *lo;
    for (const auto& c : tensix) {
        lo->x = std::min(lo->x, c.x);
        lo->y = std::min(lo->y, c.y);
        hi->x = std::max(hi->x, c.x);
        hi->y = std::max(hi->y, c.y);
    }
    uint64_t covered = tensix.size();
    if (const auto& s = sender.non_loopback_tensix;
        s && s->x >= lo->x && s->x <= hi->x && s->y >= lo->y && s->y <= hi->y) {
        ++covered;
    }
    const uint64_t area = static_cast<uint64_t>(hi->x - lo->x + 1u) * (hi->y - lo->y + 1u);
    if (covered != area) {
        return "the " + std::to_string(tensix.size()) + " Tensix participants do not fill their logical rectangle (" +
               lo->str() + ")-(" + hi->str() + ")";
    }
    return std::nullopt;
}

}  // namespace detail

/**
 * The one host release selector for packet injection and both core-sync
 * director paths. Resolves every logical participant through
 * X280Runtime::virtual_core_from_logical_core on `mesh` for `sender.noc`, takes
 * an already-resolved participant as given, and chooses at most one multicast
 * group plus endpoints under `mode`:
 *
 * - Non-Tensix participants are endpoints in every mode: firmware opts their
 *   rows and columns out of broadcasts.
 * - The Tensix participants form the group when they exactly fill one logical
 *   rectangle (a non-loopback sender may fill one cell) and the sender can
 *   multicast on NoC 0. Translation is order-preserving and its virtual gaps
 *   are opted-out columns, so the group delivers exactly the selection, even
 *   across translated columns 8-9 (docs/harness.md
 *   "Host-access gate").
 *   Otherwise each is an endpoint; a selection is never split into groups.
 *
 * `mesh` must be a live, local unit mesh, not closed or reshaped during the
 * call; no device I/O. Throws std::invalid_argument for an empty selection, a
 * duplicate Tensix participant, a resolved Tensix participant, a non-loopback
 * sender listed as a participant, more than `max_endpoints` endpoints, or an
 * unsatisfiable RequireMulticast; resolution failures propagate as the
 * resolver throws them.
 */
inline ReleasePlan select_release(
    const ttmd::MeshDevice& mesh,
    const std::vector<ReleaseParticipant>& participants,
    const ReleaseSender& sender,
    ReleaseMode mode,
    uint32_t max_endpoints) {
    if (!mesh.is_initialized()) {
        throw std::invalid_argument("release selection requires a live mesh");
    }
    if (mesh.num_devices() != 1u || mesh.is_remote_only()) {
        throw std::invalid_argument("release selection requires a local unit mesh");
    }
    if (participants.empty()) {
        throw std::invalid_argument("release selection requires at least one participant");
    }

    ReleasePlan plan;
    plan.translated.reserve(participants.size());
    std::set<ttm::CoreCoord> tensix;
    for (const auto& p : participants) {
        if (const auto* logical = std::get_if<ttm::CoreCoord>(&p.where)) {
            if (detail::is_tensix(p.type)) {
                if (!tensix.insert(*logical).second) {
                    throw std::invalid_argument("release selection has duplicate Tensix participant " + logical->str());
                }
                if (sender.non_loopback_tensix == *logical) {
                    throw std::invalid_argument(
                        "release selection lists the non-loopback sender " + logical->str() +
                        " as a participant; its own multicast cannot release it");
                }
            }
            plan.translated.push_back(X280Runtime::virtual_core_from_logical_core(mesh, *logical, p.type, sender.noc));
        } else {
            if (detail::is_tensix(p.type)) {
                throw std::invalid_argument("release selection takes Tensix participants by logical core");
            }
            plan.translated.push_back(std::get<TranslatedCoord>(p.where));
        }
    }

    std::optional<std::string> refusal;
    ttm::CoreCoord lo{};
    ttm::CoreCoord hi{};
    if (tensix.empty()) {
        refusal = "the selection has no Tensix participant";
    } else {
        refusal = detail::multicast_refusal(tensix, sender, &lo, &hi);
    }
    if (mode == ReleaseMode::RequireMulticast && refusal) {
        throw std::invalid_argument("multicast required, but " + *refusal);
    }
    if (mode != ReleaseMode::UnicastOnly && !refusal) {
        plan.group = ReleaseGroup{
            X280Runtime::virtual_core_from_logical_core(mesh, lo, tt::CoreType::TENSIX, sender.noc),
            X280Runtime::virtual_core_from_logical_core(mesh, hi, tt::CoreType::TENSIX, sender.noc),
            static_cast<uint32_t>(tensix.size())};
    }

    for (uint32_t i = 0; i < participants.size(); ++i) {
        if (!plan.group || !detail::is_tensix(participants[i].type)) {
            plan.endpoints.push_back(i);
        }
    }
    if (plan.endpoints.size() > max_endpoints) {
        throw std::invalid_argument(
            "release selection needs " + std::to_string(plan.endpoints.size()) + " endpoints, more than the roster's " +
            std::to_string(max_endpoints));
    }
    return plan;
}

}  // namespace x280::harness::regions

#endif
