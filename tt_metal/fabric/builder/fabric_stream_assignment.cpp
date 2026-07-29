// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/erisc_datamover_builder.hpp"

namespace tt::tt_fabric {

namespace {

using Inc = StreamRegAssignments::IncrementOnWrite;

// Baseline: the assignment every configuration starts from, which is the map as it stands with no VC
// on counters. Entries beyond what the baseline covers are left unused.
StreamAssignment baseline_assignment() {
    StreamAssignment a;

    a.to_sender_acked[0] = Inc::to_sender_0_pkts_acked_id;
    a.to_sender_acked[1] = Inc::to_sender_1_pkts_acked_id;
    a.to_sender_acked[2] = Inc::to_sender_2_pkts_acked_id;
    a.to_sender_acked[3] = Inc::to_sender_3_pkts_acked_id;
    // Flat sender 4 onward have no ack register in the baseline map. VC1 and VC2 never need one --
    // they run without first-level acks -- but a five-wide VC0 does, and that is what the release
    // below pays for.

    a.to_sender_completed[0] = Inc::to_sender_0_pkts_completed_id;
    a.to_sender_completed[1] = Inc::to_sender_1_pkts_completed_id;
    a.to_sender_completed[2] = Inc::to_sender_2_pkts_completed_id;
    a.to_sender_completed[3] = Inc::to_sender_3_pkts_completed_id;
    a.to_sender_completed[4] = Inc::to_sender_4_pkts_completed_id;
    a.to_sender_completed[5] = Inc::to_sender_5_pkts_completed_id;
    a.to_sender_completed[6] = Inc::to_sender_6_pkts_completed_id;
    a.to_sender_completed[7] = Inc::to_sender_7_pkts_completed_id;

    a.sender_free_slots[0] = Inc::sender_channel_0_free_slots_stream_id;
    a.sender_free_slots[1] = Inc::sender_channel_1_free_slots_stream_id;
    a.sender_free_slots[2] = Inc::sender_channel_2_free_slots_stream_id;
    a.sender_free_slots[3] = Inc::sender_channel_3_free_slots_stream_id;
    a.sender_free_slots[4] = Inc::sender_channel_4_free_slots_stream_id;
    a.sender_free_slots[5] = Inc::sender_channel_5_free_slots_stream_id;
    a.sender_free_slots[6] = Inc::sender_channel_6_free_slots_stream_id;
    a.sender_free_slots[7] = Inc::sender_channel_7_free_slots_stream_id;

    return a;
}

// Release the ack and completion registers of a flat sender range, because a VC on counters reads
// neither. Released ids become available for reassignment.
void release_credit_registers(StreamAssignment& a, uint32_t first, uint32_t last_exclusive) {
    for (uint32_t ch = first; ch < last_exclusive && ch < a.to_sender_acked.size(); ++ch) {
        for (auto* slot : {&a.to_sender_acked[ch], &a.to_sender_completed[ch]}) {
            if (*slot != StreamAssignment::k_unused) {
                a.spare[a.num_spare++] = *slot;
                *slot = StreamAssignment::k_unused;
            }
        }
    }
}

uint32_t take_spare(StreamAssignment& a, const char* purpose) {
    TT_FATAL(
        a.num_spare > 0,
        "No ethernet stream register is available for {}. All {} are allocated at baseline, so one has to be released "
        "by putting a VC's credits on L1 counters before it can be reassigned.",
        purpose,
        StreamRegAssignments::num_eth_stream_registers);
    return a.spare[--a.num_spare];
}

void validate(const StreamAssignment& a) {
    // Every live id must address a real register, and may not be the reserved zero.
    const std::array<const std::array<uint32_t, builder_config::num_max_sender_channels>*, 3> tables = {
        &a.to_sender_acked, &a.to_sender_completed, &a.sender_free_slots};

    for (const auto* table : tables) {
        for (const uint32_t id : *table) {
            TT_FATAL(
                id < StreamRegAssignments::num_eth_stream_registers,
                "Stream id {} is outside the {}-register file",
                id,
                StreamRegAssignments::num_eth_stream_registers);
        }
    }

    // No two live consumers may hold the same id. These three tables are all sender-indexed and none
    // of their roles can be shared, so any duplicate here is a mistake regardless of the declared
    // shared ids elsewhere in the map.
    for (size_t i = 0; i < tables.size(); ++i) {
        for (size_t j = i; j < tables.size(); ++j) {
            for (size_t x = 0; x < tables[i]->size(); ++x) {
                for (size_t y = (i == j ? x + 1 : 0); y < tables[j]->size(); ++y) {
                    const uint32_t lhs = (*tables[i])[x];
                    const uint32_t rhs = (*tables[j])[y];
                    if (lhs == StreamAssignment::k_unused || rhs == StreamAssignment::k_unused) {
                        continue;
                    }
                    TT_FATAL(lhs != rhs, "Stream id {} is claimed by two live sender-side consumers", lhs);
                }
            }
        }
    }
}

}  // namespace

StreamAssignment make_stream_assignment(const CreditTransportPlan& plan, const StreamAssignmentInputs& inputs) {
    StreamAssignment a = baseline_assignment();

    const uint32_t vc1_first = inputs.num_vc0_senders;
    const uint32_t vc2_first = vc1_first + inputs.num_vc1_senders;
    const uint32_t num_flat_senders = vc2_first + inputs.num_vc2_senders;

    if (plan.vc0_uses_counters) {
        release_credit_registers(a, 0, vc1_first);
    }
    if (plan.vc1_uses_counters) {
        release_credit_registers(a, vc1_first, vc2_first);
    }

    // A VC still on stream registers needs one ack register per sender. The baseline covers four, so a
    // wider VC0 needs the difference from what was released.
    if (!plan.vc0_uses_counters) {
        for (uint32_t ch = 0; ch < inputs.num_vc0_senders; ++ch) {
            if (a.to_sender_acked[ch] == StreamAssignment::k_unused) {
                a.to_sender_acked[ch] = take_spare(a, "a VC0 sender's first-level ack");
            }
        }
    }

    // Likewise every serviced sender needs somewhere to read its own free-slot count. The baseline
    // covers eight; VC2's sender has its own dedicated register.
    for (uint32_t ch = 0; ch < num_flat_senders && ch < a.sender_free_slots.size(); ++ch) {
        if (a.sender_free_slots[ch] != StreamAssignment::k_unused) {
            continue;
        }
        if (inputs.num_vc2_senders > 0 && ch >= vc2_first) {
            a.sender_free_slots[ch] = Inc::vc2_sender_free_slots_stream_id;
        } else {
            a.sender_free_slots[ch] = take_spare(a, "a sender's free-slot counter");
        }
    }

    validate(a);
    return a;
}

}  // namespace tt::tt_fabric
