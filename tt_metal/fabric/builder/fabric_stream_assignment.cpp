// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"

#include <enchantum/enchantum.hpp>
#include <hostdevcommon/fabric_common.h>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include <set>

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"

namespace tt::tt_fabric {

// ============================================================================
// Requirement-driven assignment
// ============================================================================

void StreamRequirements::add(StreamRole role, uint32_t vc, uint32_t count) {
    if (count == 0) {
        return;
    }
    groups.push_back(Group{role, vc, count});
}

StreamRequirements stream_requirements(const StreamPlacementInputs& placement, const CreditTransportPlan& plan) {
    StreamRequirements need;
    need.sender_counts = placement.max_sender_counts;
    // The flat space is defined by the fabric maxima, so the bases are their prefix sums --
    // computed here, once, from the placement inputs.
    uint32_t base = 0;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        need.sender_flat_base[vc] = base;
        base += placement.max_sender_counts[vc];
    }

    // The downstream span is four per VC with a receiver, direction-keyed (not densified).
    constexpr uint32_t downstream_span_per_vc = 4;

    // Need-driven receivers, keyed by receiver CHANNEL rather than by VC, because that is how the
    // kernel indexes to_receiver_packets_sent_streams. VC2's receiver densifies onto channel 1 when
    // VC1 has no receiver and channel 2 when it does, so keying by VC left channel 1 unassigned in
    // exactly the VC2-without-VC1 case: arrivals incremented the out-of-range sentinel and the
    // receiver polled it forever, stalling the link with the packets already delivered.
    const bool vc1_has_receiver = placement.max_receiver_counts[1] > 0;
    const bool vc2_has_receiver = placement.max_receiver_counts[2] > 0;
    need.add(StreamRole::RECEIVER_PKTS_SENT, 0, placement.max_receiver_counts[0] > 0 ? 1 : 0);
    need.add(StreamRole::RECEIVER_PKTS_SENT, 1, (vc1_has_receiver || vc2_has_receiver) ? 1 : 0);
    need.add(StreamRole::RECEIVER_PKTS_SENT, 2, (vc1_has_receiver && vc2_has_receiver) ? 1 : 0);

    // A VC still on registers needs one ack (VC0 only) and one completion register per sender.
    if (!plan.vc0_uses_counters) {
        need.add(StreamRole::SENDER_PKTS_ACKED, 0, placement.max_sender_counts[0]);
        need.add(StreamRole::SENDER_PKTS_COMPLETED, 0, placement.max_sender_counts[0]);
    }
    if (!plan.vc1_uses_counters) {
        // The family's full VC1 width is the need. The completed table's declared extent covers
        // every position VC0 and VC1 can produce (0..8), and the register budget is
        // make_stream_assignment's check -- no extent arithmetic belongs here.
        need.add(StreamRole::SENDER_PKTS_COMPLETED, 1, placement.max_sender_counts[1]);
    }

    // VC2 takes no completed group: its sender is flat position 9, one past the table's declared
    // extent, so no register can name it. That makes counters its only workable transport -- and a
    // plan that says otherwise leaves its sender polling a sentinel while the receiver acks into
    // the other mechanism, which wedges the link with no error. Widening the table to ten names
    // (kernel header included) is what it would take to put VC2 back on registers.
    TT_FATAL(
        !placement.vc2_present || plan.vc2_uses_counters,
        "VC2 is present but its credit plan says stream registers. Its sender is flat position {}, past "
        "the completed table's declared extent of {} positions, so it has no completion register.",
        need.sender_flat_base[2],
        num_pkts_completed_names);

    need.add(StreamRole::DOWNSTREAM_FREE_SLOTS, 0, placement.max_receiver_counts[0] > 0 ? downstream_span_per_vc : 0);
    need.add(StreamRole::DOWNSTREAM_FREE_SLOTS, 1, placement.max_receiver_counts[1] > 0 ? downstream_span_per_vc : 0);

    // VC2's sender and receiver ride pinned registers, so VC2 takes no group here.
    need.add(StreamRole::SENDER_FREE_SLOTS, 0, placement.max_sender_counts[0]);
    need.add(StreamRole::SENDER_FREE_SLOTS, 1, placement.max_sender_counts[1]);

    need.vc2_present = placement.vc2_present;
    need.tensix_relay_present = placement.tensix_relay_present;
    need.plan = plan;
    return need;
}

namespace {

constexpr uint32_t k_first_pinned_id = 30;  // the pinned/scratch region {30, 31}

// The worker-facing pin, read from the header worker-space and ControlPlane already consume -- one
// constant, one authority. With the inactive sentinel out of register range (k_unused_stream_id),
// id 0 is an ordinary register, so the pin can sit at the bottom of the file and every other group
// simply packs above it in one contiguous run.
static constexpr uint32_t k_worker_free_slots_pin = connection_interface::sender_channel_0_free_slots_stream_id;

std::string describe(const StreamRequirements& need) {
    std::string out;
    for (const auto& g : need.groups) {
        out += fmt::format("  {} vc{} x{}\n", enchantum::to_string(g.role), g.vc, g.count);
    }
    return out;
}

// The kernel's declared stream-register name set, generated from the same extents the emission
// walks (fabric_erisc_router_ct_args.hpp's unconditional declarations). The mirror's two sides:
// this function and named_args() must both move with the kernel header.
std::set<std::string> expected_stream_name_set() {
    std::set<std::string> names;
    for (uint32_t vc = 0; vc < num_receiver_pkts_sent_names; ++vc) {
        names.insert(fmt::format("TO_RECEIVER_{}_PKTS_SENT_ID", vc));
    }
    for (uint32_t flat = 0; flat < num_pkts_acked_names; ++flat) {
        names.insert(fmt::format("TO_SENDER_{}_PKTS_ACKED_ID", flat));
    }
    for (uint32_t flat = 0; flat < num_pkts_completed_names; ++flat) {
        names.insert(fmt::format("TO_SENDER_{}_PKTS_COMPLETED_ID", flat));
    }
    for (uint32_t vc = 0; vc <= 1; ++vc) {
        for (uint32_t compact = 1; compact <= num_downstream_free_slots_names_per_vc; ++compact) {
            names.insert(fmt::format("VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID", vc, compact));
        }
    }
    for (uint32_t flat = 0; flat < builder_config::num_max_sender_channels; ++flat) {
        names.insert(fmt::format("SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", flat));
    }
    names.insert("VC2_RECEIVER_FREE_SLOTS_STREAM_ID");
    names.insert("TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID");
    names.insert("MULTI_RISC_TEARDOWN_SYNC_STREAM_ID");
    names.insert("ETH_RETRAIN_LINK_SYNC_STREAM_ID");
    return names;
}

}  // namespace

uint32_t StreamAssignment::id(StreamRole role, uint32_t vc, uint32_t index) const {
    for (const auto& entry : entries_) {
        if (entry.role == role && entry.vc == vc && index < entry.count) {
            return entry.first_id + index;
        }
    }
    TT_FATAL(false, "No stream register allocated for role {} vc {} index {}", enchantum::to_string(role), vc, index);
    return 0;
}

bool StreamAssignment::has(StreamRole role, uint32_t vc, uint32_t index) const {
    for (const auto& entry : entries_) {
        if (entry.role == role && entry.vc == vc && index < entry.count) {
            return true;
        }
    }
    return false;
}

uint32_t StreamAssignment::group_count(StreamRole role, uint32_t vc) const {
    for (const auto& entry : entries_) {
        if (entry.role == role && entry.vc == vc) {
            return entry.count;
        }
    }
    return 0;
}

void StreamAssignment::emit_flat_table(
    StreamRole role, const char* pattern, uint32_t extent, std::vector<std::pair<std::string, uint32_t>>& out) const {
    // One flat-indexed table, emitted as monotone segments: the vc0 group, the gap after it, the
    // vc1 group, then the tail. The flat index is generated, never resolved per position.
    uint32_t flat = 0;
    for (uint32_t vc = 0; vc <= 1; ++vc) {
        const uint32_t base = sender_flat_base_[vc];
        const uint32_t count = group_count(role, vc);
        for (; flat < base; ++flat) {
            out.emplace_back(fmt::format(fmt::runtime(pattern), flat), k_unused_stream_id);
        }
        for (uint32_t ch = 0; ch < count; ++ch, ++flat) {
            out.emplace_back(fmt::format(fmt::runtime(pattern), flat), id(role, vc, ch));
        }
    }
    // The walk may not overrun the declared table: if it does, the group's flat span exceeds the
    // kernel's declared extent, and that failure belongs to this table, not the aggregate count.
    TT_FATAL(flat <= extent, "Table {} walked {} flat positions, the kernel declares {}", pattern, flat, extent);
    for (; flat < extent; ++flat) {
        // The VC2 sender's flat slot rides its pinned register, not the sentinel.
        const bool is_vc2_slot = role == StreamRole::SENDER_FREE_SLOTS && vc2_present_ && flat == sender_flat_base_[2];
        out.emplace_back(
            fmt::format(fmt::runtime(pattern), flat),
            is_vc2_slot ? StreamRegAssignments::IncrementOnWrite::vc2_sender_free_slots_stream_id : k_unused_stream_id);
    }
}

std::vector<std::pair<std::string, uint32_t>> StreamAssignment::named_args() const {
    std::vector<std::pair<std::string, uint32_t>> out;
    out.reserve(num_stream_register_names);

    // Receiver pkts-sent counters, named per receiver channel (not per VC, and not sender-flat).
    for (uint32_t channel = 0; channel < num_receiver_pkts_sent_names; ++channel) {
        out.emplace_back(
            fmt::format("TO_RECEIVER_{}_PKTS_SENT_ID", channel),
            has(StreamRole::RECEIVER_PKTS_SENT, channel, 0) ? id(StreamRole::RECEIVER_PKTS_SENT, channel, 0)
                                                            : k_unused_stream_id);
    }

    emit_flat_table(StreamRole::SENDER_PKTS_ACKED, "TO_SENDER_{}_PKTS_ACKED_ID", num_pkts_acked_names, out);
    emit_flat_table(StreamRole::SENDER_PKTS_COMPLETED, "TO_SENDER_{}_PKTS_COMPLETED_ID", num_pkts_completed_names, out);

    // Downstream free slots: direction-keyed, per VC.
    for (uint32_t vc = 0; vc <= 1; ++vc) {
        for (uint32_t compact = 0; compact < num_downstream_free_slots_names_per_vc; ++compact) {
            out.emplace_back(
                fmt::format("VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID", vc, compact + 1),
                has(StreamRole::DOWNSTREAM_FREE_SLOTS, vc, compact) ? id(StreamRole::DOWNSTREAM_FREE_SLOTS, vc, compact)
                                                                    : k_unused_stream_id);
        }
    }

    emit_flat_table(
        StreamRole::SENDER_FREE_SLOTS,
        "SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID",
        builder_config::num_max_sender_channels,
        out);

    // Pinned entries: the pinned id when the consumer is live, else the sentinel.
    out.emplace_back(
        "VC2_RECEIVER_FREE_SLOTS_STREAM_ID",
        vc2_present_ ? StreamRegAssignments::IncrementOnWrite::vc2_receiver_free_slots_stream_id : k_unused_stream_id);
    out.emplace_back(
        "TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID",
        tensix_relay_present_ ? StreamRegAssignments::IncrementOnWrite::tensix_relay_local_free_slots_stream_id
                              : k_unused_stream_id);
    // Scratch entries: kernel lifecycle workspace, always emitted with the pinned ids.
    out.emplace_back(
        "MULTI_RISC_TEARDOWN_SYNC_STREAM_ID", StreamRegAssignments::Scratch::multi_risc_teardown_sync_stream_id);
    out.emplace_back("ETH_RETRAIN_LINK_SYNC_STREAM_ID", StreamRegAssignments::Scratch::eth_retrain_link_sync_stream_id);

    // Completeness as SET equality, not count: a duplicate paired with an omission has the right
    // count, and a map insertion would then swallow the duplicate and lose the omitted name as an
    // unset CT arg.
    std::set<std::string> emitted;
    for (const auto& [name, value] : out) {
        TT_FATAL(emitted.insert(name).second, "Stream-register name {} emitted twice", name);
    }
    TT_FATAL(
        emitted == expected_stream_name_set(),
        "Emitted stream-register names do not match the kernel's declared set (mirror of "
        "fabric_erisc_router_ct_args.hpp)");
    return out;
}

StreamAssignment make_stream_assignment(const StreamRequirements& need) {
    uint32_t total = 0;
    for (const auto& g : need.groups) {
        total += g.count;
    }
    TT_FATAL(
        total <= k_first_pinned_id,
        "This router needs {} stream registers, over the {} available below the aliased pair:\n{}",
        total,
        k_first_pinned_id,
        describe(need));
    // The pinned id 30 is dual-use: the VC2 sender and the tensix relay may not both be live. The
    // upstream derivation (requires_vc2 = !udm && !mux) already makes that unreachable; this is
    // the allocator's own bookkeeping check from the same inputs, not a second copy of it.
    TT_FATAL(
        !(need.vc2_present && need.tensix_relay_present),
        "Stream id {} has two live consumers: the VC2 sender and the tensix relay",
        k_first_pinned_id);

    StreamAssignment a;
    a.sender_flat_base_ = need.sender_flat_base;
    a.sender_counts_ = need.sender_counts;
    a.vc2_present_ = need.vc2_present;
    a.tensix_relay_present_ = need.tensix_relay_present;
    a.plan_ = need.plan;

    // Placement: the pinned vc0 sender-free-slots base at the worker-facing constant, then every
    // other group packed upward from it in one contiguous run, ending below the aliased pair.
    uint32_t run_start = k_worker_free_slots_pin;
    for (const auto& g : need.groups) {
        if (g.role == StreamRole::SENDER_FREE_SLOTS && g.vc == 0) {
            a.entries_.push_back(StreamAssignment::Entry{g.role, g.vc, k_worker_free_slots_pin, g.count});
            run_start = k_worker_free_slots_pin + g.count;
        }
    }
    uint32_t next = run_start;
    for (const auto& g : need.groups) {
        if (g.role == StreamRole::SENDER_FREE_SLOTS && g.vc == 0) {
            continue;
        }
        a.entries_.push_back(StreamAssignment::Entry{g.role, g.vc, next, g.count});
        next += g.count;
    }
    TT_FATAL(
        next <= k_first_pinned_id,
        "The allocated run ends at {}, over the {} start of the aliased pair, for:\n{}",
        next,
        k_first_pinned_id,
        describe(need));
    return a;
}

}  // namespace tt::tt_fabric
