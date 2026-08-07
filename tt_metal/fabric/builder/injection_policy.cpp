// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/injection_policy.hpp"

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

namespace tt::tt_fabric {

NonExpressInjectionPolicy::NonExpressInjectionPolicy(Topology topology, eth_chan_directions facing) :
    topology_(topology), facing_(facing) {
    const bool is_ew = builder::is_east_or_west(facing);
    const bool is_ns = builder::is_north_or_south(facing);
    const bool is_z = facing == eth_chan_directions::Z;
    TT_FATAL(
        is_ew + is_ns + is_z == 1,
        "In NonExpressInjectionPolicy, exactly one of east/west, north/south, and Z must be true for facing {}",
        enchantum::to_string(facing));
}

bool NonExpressInjectionPolicy::participates(uint32_t vc) const {
    // VC1 carries intermesh traffic with no bubble flow control; Linear/Mesh mark no injection
    // channels at all.
    return vc != 1 && topology_ != Topology::Linear && topology_ != Topology::Mesh;
}

bool NonExpressInjectionPolicy::worker_is_injection(uint32_t /*vc*/) const {
    // The worker channel is always an injection channel on any VC this policy participates in --
    // including VC2, which is the visible difference from the express policy's gate.
    return true;
}

bool NonExpressInjectionPolicy::producer_is_injection(uint32_t /*vc*/, eth_chan_directions producer) const {
    // Turn channels exist on Torus only; every other topology leaves producers unguarded.
    if (topology_ != Topology::Torus) {
        return false;
    }
    // A turn channel: the producer's axis differs from this router's. On a Z-facing router both
    // axis predicates are false, so nothing is a turn -- preserved, not a bug.
    const bool producer_is_ew = builder::is_east_or_west(producer);
    const bool producer_is_ns = builder::is_north_or_south(producer);
    return (builder::is_east_or_west(facing_) && !producer_is_ew) ||
           (builder::is_north_or_south(facing_) && !producer_is_ns);
}

ExpressInjectionPolicy::ExpressInjectionPolicy(
    const ProtectedRingQueries& queries,
    const PerDirectionCapabilities& capabilities,
    ZPortRole chip_z_role,
    RoutingDirection egress,
    EdgeCapability egress_capability) :
    queries_(queries),
    capabilities_(capabilities),
    chip_z_role_(chip_z_role),
    egress_(egress),
    egress_capability_(egress_capability) {}

bool ExpressInjectionPolicy::participates(uint32_t vc) const {
    // VC2 stays out of the derivation: the optional existing VC2 behavior is separate from the
    // express design, and its sender is not a fixed-direction producer slot that the slot mapping
    // can name. It keeps the ordinary non-injection guard until its express-mesh role is defined.
    return vc < 2;
}

bool ExpressInjectionPolicy::worker_is_injection(uint32_t /*vc*/) const {
    // Worker source injection has no ingress direction; it is a first acquisition whenever its
    // egress is protected.
    return is_injection_effect(classify_worker_effect(queries_, egress_));
}

bool ExpressInjectionPolicy::producer_is_injection(uint32_t vc, eth_chan_directions producer) const {
    const auto ingress = builder::eth_direction_to_routing_direction(producer);
    const auto& ingress_capability = capabilities_.at(ingress);
    if (!ingress_capability.has_value()) {
        return false;  // no neighbour that way, so nothing is wired into this slot
    }
    // The wiring gate stays ahead of classification: under express wiring some slots carry no
    // producer (an intramesh X ingress is dimension-order-unwired from every intramesh Y egress,
    // and nothing is wired into a Z egress on a chip with no Z port), so a DOR-forbidden turn that
    // still reaches classify_producer_effect genuinely signals map/derivation disagreement, not a
    // correctly unwired slot.
    if (!wires_into(ingress, *ingress_capability, egress_, chip_z_role_, /*express_routing_enabled=*/true, vc)) {
        return false;
    }
    return is_injection_effect(
        classify_producer_effect(queries_, ingress, *ingress_capability, egress_, egress_capability_));
}

std::vector<bool> compute_sender_channel_injection_flags(
    const builder::RouterProducerSlots& slots, uint32_t vc, const InjectionPolicy& policy) {
    std::vector<bool> flags(slots.sender_count(vc), false);
    if (!policy.participates(vc)) {
        return flags;
    }
    if (const auto worker = slots.worker_channel(vc)) {
        flags.at(*worker) = policy.worker_is_injection(vc);
    }
    for (const auto [channel, producer] : slots.producer_slots(vc)) {
        flags.at(channel) = policy.producer_is_injection(vc, producer);
    }
    return flags;
}

}  // namespace tt::tt_fabric
