// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/small_vector.hpp>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

namespace tt::tt_fabric {

// Which VCs carry receiver-to-sender ack and completion credits in L1 counters rather than stream
// registers.
//
// One of the two axes this file separates. The plan says which transport each VC uses; the assignment
// below says which register serves which role once that is decided. Keeping them apart means changing
// a VC's transport is a single decision rather than an edit threaded through the map.
struct CreditTransportPlan {
    bool vc0_uses_counters = false;
    bool vc1_uses_counters = false;
    // VC2 is a carrier VC in its own right (one sender at flat position 9, one receiver), so it
    // needs its own transport decision rather than inheriting a neighbour's. Its sender sits past
    // the completed table's declared extent (positions 0..8), so counters are the only transport
    // that can carry its credits today; stream_requirements enforces that.
    bool vc2_uses_counters = false;

    bool any_vc_uses_counters() const { return vc0_uses_counters || vc1_uses_counters || vc2_uses_counters; }
};

struct StreamRegAssignments {
    // An ethernet core has 32 NOC overlay stream registers. That is a hardware limit rather than a
    // convention -- ETH_NOC_NUM_STREAMS in noc_overlay_parameters.h, and 32 on Wormhole, Blackhole,
    // and Quasar alike, while Tensix cores get 64. Mirrored here because that header is
    // architecture-private and not visible to host code.
    //
    // Ids 0-29 are computed by the requirement-driven assignment below (group definitions plus name
    // patterns), so the only register constants that survive are the pinned region {30, 31}.
    static constexpr uint32_t num_eth_stream_registers = 32;

    // The pinned inc-on-write ids, deliberately multiplexed among mutually exclusive consumers.
    struct IncrementOnWrite {
        // Local tensix relay free slots stream ID (UDM mode only)
        // Dual-use: also used as scratch for eth_retrain (see Scratch::eth_retrain_link_sync_stream_id)
        static constexpr uint32_t tensix_relay_local_free_slots_stream_id = 30;
        // VC2 sender flow control: free-slots from worker (dual-use with tensix_relay at ID 30; VC2 and
        // UDM/mux mutually exclusive)
        static constexpr uint32_t vc2_sender_free_slots_stream_id = 30;
        // VC2 receiver flow control: free-slots from sender (non-Z routers only; dual-use with scratch
        // multi_risc_teardown at ID 31)
        static constexpr uint32_t vc2_receiver_free_slots_stream_id = 31;
    };

    // Stream registers used as scratch/overlay storage. Writing overwrites the register value.
    struct Scratch {
        // Eth retrain synchronization stream ID
        // Dual-use: also used as inc-on-write for tensix_relay (see
        // IncrementOnWrite::tensix_relay_local_free_slots_stream_id)
        static constexpr uint32_t eth_retrain_link_sync_stream_id = 30;
        // Multi-RISC teardown synchronization stream ID
        static constexpr uint32_t multi_risc_teardown_sync_stream_id = 31;
    };
};

// The declared name-set extents: how many names of each pattern the router kernel declares
// unconditionally (fabric_erisc_router_ct_args.hpp). These extents are ABI and cannot be verified
// host-side -- they must move with the kernel header, and num_stream_register_names is only as
// good as this mirror. Keep the two sides in step; the kernel header names this file as its
// counterpart.
constexpr uint32_t num_receiver_pkts_sent_names = 3;            // TO_RECEIVER_{0,1,2}_PKTS_SENT_ID
constexpr uint32_t num_pkts_acked_names = 5;                    // TO_SENDER_{0..4}_PKTS_ACKED_ID
constexpr uint32_t num_pkts_completed_names = 9;                // TO_SENDER_{0..8}_PKTS_COMPLETED_ID
constexpr uint32_t num_downstream_free_slots_names_per_vc = 4;  // VC{0,1}_..._EDGE_{1..4}
// Sender free slots: builder_config::num_max_sender_channels is already the kernel's extent.

// The total declared set size: receivers + acked + completed + both VCs' downstream + sender free
// + the two pinned entries + the two scratch entries.
constexpr uint32_t num_stream_register_names = num_receiver_pkts_sent_names + num_pkts_acked_names +
                                               num_pkts_completed_names + 2 * num_downstream_free_slots_names_per_vc +
                                               builder_config::num_max_sender_channels + 2 + 2;

// ============================================================================
// Requirement-driven stream-register assignment
// ============================================================================
//
// The roles a stream register can play for one router. Requirements are grouped by (role, vc); the
// index inside a group means what the role says -- sender channel for the sender roles, downstream
// compact index for DOWNSTREAM_FREE_SLOTS, nothing for the per-VC receiver role. Deliberately not
// one index space.
enum class StreamRole : uint8_t {
    RECEIVER_PKTS_SENT,       // per receiver CHANNEL (not VC -- VC2's receiver densifies), need-driven
    SENDER_PKTS_ACKED,        // per sender channel; VC0 only, and only while VC0 is on registers
    SENDER_PKTS_COMPLETED,    // per sender channel, per VC still on registers
    DOWNSTREAM_FREE_SLOTS,    // per downstream compact index (span 4, direction-keyed), per VC
    SENDER_FREE_SLOTS,        // per sender channel
    VC2_SENDER_FREE_SLOTS,    // pinned to id 30 -- dual-use with tensix relay; see the aliasing note
    VC2_RECEIVER_FREE_SLOTS,  // pinned to id 31
    TENSIX_RELAY_FREE_SLOTS,  // pinned to id 30 -- dual-use with VC2 sender; see the aliasing note
};

// Placement inputs, fabric-scoped by type: the kernel resolves a downstream router's register id
// through its own table, which is only correct if the flat-channel -> id map is identical on every
// router that can be another's downstream. Deriving placement from the fabric's family maxima
// makes a per-router divergence unrepresentable; taking a router's shape here would invite the
// bug, so the struct exists to make the parameter unpassable.
struct StreamPlacementInputs {
    std::array<uint32_t, builder_config::MAX_NUM_VCS> max_sender_counts;    // FabricBuilderContext
    std::array<uint32_t, builder_config::MAX_NUM_VCS> max_receiver_counts;  // FabricBuilderContext
    bool vc2_present;           // IntermeshVCConfig::requires_vc2 -- config, not per router
    bool tensix_relay_present;  // UDM mode -- config, not per router
};

// What one fabric configuration needs, derived from the placement inputs (the fabric's family
// maxima, never a router's own counts or the trimmed actuals -- one stream layout per fabric, so
// neither trimming nor per-router narrowing can perturb the shared map). The RouterTurnSet is
// deliberately NOT an input: the downstream span stays direction-keyed at four per VC until the
// kernel indexes densely, so the turn set would contribute nothing to sizing -- and a turn-set
// bound check here would duplicate check_vc0_downstream_capacity.
struct StreamRequirements {
    struct Group {
        StreamRole role;
        uint32_t vc;
        uint32_t count;
    };
    ttsl::SmallVector<Group, builder_config::MAX_NUM_VCS * 4 + 1> groups;
    void add(StreamRole role, uint32_t vc, uint32_t count);  // no-ops on count == 0

    // Carried from the shape so the assignment can answer by flat sender index.
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_flat_base{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_counts{};

    // Pinned consumers that are live in this configuration, declared from the build site's facts
    // (requires_vc2, UDM mode) -- the allocator asserts its own bookkeeping on them, it does not
    // re-derive the upstream exclusivity.
    bool vc2_present = false;
    bool tensix_relay_present = false;

    // The credit plan this need was derived from, carried so the assignment stays one object.
    CreditTransportPlan plan;
};

StreamRequirements stream_requirements(const StreamPlacementInputs& placement, const CreditTransportPlan& plan);

// The assignment: which register serves each (role, vc, index), plus the ONE place the
// role -> CT-arg-name mapping lives. Register numbers are not ABI -- the kernel reads each through
// NAMED_CT_ARG by name -- but the name set is: emission materialises the full declared set,
// allocated entries filled and the out-of-range sentinel (k_unused_stream_id) for inactive ones,
// because the kernel declares every name unconditionally and an omitted name is an unset CT arg.
class StreamAssignment {
public:
    uint32_t id(StreamRole role, uint32_t vc, uint32_t index) const;  // fatals if not allocated
    bool has(StreamRole role, uint32_t vc, uint32_t index) const;

    // The full declared set as one ordered vector of (name, value) -- the host mirror of
    // fabric_erisc_router_ct_args.hpp's stream-register reads. Allocated entries carry their id;
    // everything else carries the out-of-range sentinel. Completeness is checked as SET equality
    // against the declared set: a duplicate paired with an omission has the right count and must
    // not pass.
    std::vector<std::pair<std::string, uint32_t>> named_args() const;

    // The credit plan this assignment was derived for, carried so consumers need one object.
    const CreditTransportPlan& plan() const { return plan_; }

    // The flat base of a VC's sender channels. Fabric-scoped: prefix sums of the fabric's family
    // maxima, identical on every router in the fabric.
    uint32_t sender_flat_base(uint32_t vc) const { return sender_flat_base_[vc]; }

private:
    struct Entry {
        StreamRole role;
        uint32_t vc;
        uint32_t first_id;
        uint32_t count;
    };
    ttsl::SmallVector<Entry, 12> entries_;
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_flat_base_{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_counts_{};
    bool vc2_present_ = false;
    bool tensix_relay_present_ = false;
    CreditTransportPlan plan_;

    // One group entry's count (0 when the group is absent): an entry read, not a scan.
    uint32_t group_count(StreamRole role, uint32_t vc) const;
    // One flat-indexed name table, emitted as four monotone segments (group, gap, group, tail)
    // with the extent guard between them.
    void emit_flat_table(
        StreamRole role,
        const char* pattern,
        uint32_t extent,
        std::vector<std::pair<std::string, uint32_t>>& out) const;

    friend StreamAssignment make_stream_assignment(const StreamRequirements& need);
};

// Derive the assignment from the requirements: groups are placed in the order stream_requirements
// adds them (that order IS the placement policy -- stable and documented), allocated ids stay
// below 30, and the pinned region {30, 31} is reserved for the pinned/scratch consumers. Fatals up
// front with a legible message when the requirements overrun, rather than a late "no register".
StreamAssignment make_stream_assignment(const StreamRequirements& need);

}  // namespace tt::tt_fabric
