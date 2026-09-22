// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>  // Topology

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"

namespace tt::tt_fabric {

// ============ Injection-flag policies ============
//
// The per-slot decision that selects a sender's flow-control guard. What differs between families
// is the decision, not the walk: one slot walk serves both. The VC gate also exposes the open VC2
// question -- non-express excludes VC1 but not VC2, express excludes both.
class InjectionPolicy {
public:
    virtual ~InjectionPolicy() = default;
    // Whether this VC's senders carry any guard at all (whole-VC / whole-router gate).
    virtual bool participates(uint32_t vc) const = 0;
    // The worker slot's guard (consulted only where the slot mapping reports a worker channel).
    virtual bool worker_is_injection(uint32_t vc) const = 0;
    // One producer's guard on one VC.
    virtual bool producer_is_injection(uint32_t vc, eth_chan_directions producer) const = 0;
};

// The frozen axis-turn heuristic, named so it can be driven from a host test: the worker channel
// is always an injection channel, and on Torus a "turn channel" (a producer whose axis differs
// from this router's) is one too; Linear/Mesh and VC1 carry no guard at all. Byte-identical
// preservation of the standing decision is the whole point of this class.
class NonExpressInjectionPolicy : public InjectionPolicy {
public:
    NonExpressInjectionPolicy(Topology topology, eth_chan_directions facing);

    bool participates(uint32_t vc) const override;
    bool worker_is_injection(uint32_t vc) const override;
    bool producer_is_injection(uint32_t vc, eth_chan_directions producer) const override;

private:
    Topology topology_;
    eth_chan_directions facing_;
};

// The express derivation, which the axis-turn heuristic cannot represent: at an express node the
// same Z output is same-ring transit when fed by the ring and a ring acquisition when fed by a
// leaf attachment, and both producers share one axis pair, so each producer's total effect is
// derived from the protected-ring facts and only an acquisition becomes an injection channel.
// Every fact arrives bound, so the policy is drivable from a host-side ring model without a
// ControlPlane. VC-independent: the wiring primitive's one VC-sensitive case (a boundary
// producer) is answered from the VC argument, so one policy serves every VC of the router.
class ExpressInjectionPolicy : public InjectionPolicy {
public:
    ExpressInjectionPolicy(
        const ProtectedRingQueries& queries,
        const PerDirectionCapabilities& capabilities,
        ZPortRole chip_z_role,
        RoutingDirection egress,
        EdgeCapability egress_capability);

    bool participates(uint32_t vc) const override;
    bool worker_is_injection(uint32_t vc) const override;
    bool producer_is_injection(uint32_t vc, eth_chan_directions producer) const override;

private:
    const ProtectedRingQueries& queries_;
    const PerDirectionCapabilities& capabilities_;
    ZPortRole chip_z_role_;
    RoutingDirection egress_;
    EdgeCapability egress_capability_;
};

// One slot walk for every family: size from the mapping, worker from the policy, producers from
// the mapping filtered by the policy. The mapping's bijection makes one-producer-per-channel
// structural, so an ENTER/REMAIN alias on one concrete sender cannot arise.
std::vector<bool> compute_sender_channel_injection_flags(
    const builder::RouterProducerSlots& slots, uint32_t vc, const InjectionPolicy& policy);

}  // namespace tt::tt_fabric
