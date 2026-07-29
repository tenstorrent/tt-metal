// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

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

    bool any_vc_uses_counters() const { return vc0_uses_counters || vc1_uses_counters; }
};

// Per-router facts the assignment depends on.
struct StreamAssignmentInputs {
    uint32_t num_vc0_senders = 0;
    uint32_t num_vc1_senders = 0;
    uint32_t num_vc2_senders = 0;
};

// A complete ethernet stream-register assignment for one router configuration.
//
// This type exists because all 32 registers are allocated at baseline. Any new consumer has to take
// one that some configuration genuinely released, and without a single owner that reuse ends up as a
// conditional expression with a bare register number in it -- correct, perhaps, but no longer
// readable. Here the release and the reassignment are named operations, and validate() checks the
// result rather than trusting it.
struct StreamAssignment {
    // A stream id of zero means the consumer is inactive for this configuration. Register 0 is a real
    // register (the VC0 receiver's packets-sent counter), so a live consumer may never hold 0, and the
    // device must skip initialising an entry that is zero.
    static constexpr uint32_t k_unused = 0;

    // Indexed by flat sender channel.
    std::array<uint32_t, builder_config::num_max_sender_channels> to_sender_acked{};
    std::array<uint32_t, builder_config::num_max_sender_channels> to_sender_completed{};
    std::array<uint32_t, builder_config::num_max_sender_channels> sender_free_slots{};

    // Registers this configuration released and did not reassign. Recorded rather than discarded so a
    // future consumer can see what is genuinely available.
    std::array<uint32_t, builder_config::num_max_sender_channels> spare{};
    uint32_t num_spare = 0;
};

// Build the assignment for one configuration, then validate it.
//
// Throws if the result is unusable: an id outside the register file, a live consumer holding the
// reserved zero, two live consumers sharing an id that was not declared shareable, or a consumer that
// needs a register when none was released.
StreamAssignment make_stream_assignment(const CreditTransportPlan& plan, const StreamAssignmentInputs& inputs);

}  // namespace tt::tt_fabric
