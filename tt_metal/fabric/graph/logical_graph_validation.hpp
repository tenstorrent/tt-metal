// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "fabric_graph_types.hpp"
#include "fabric_wiring_rules.hpp"

namespace tt::tt_fabric::graph {

struct ValidationLimits {
    uint32_t max_sender_channels_per_vc = 0;  // 0 = no limit
    uint32_t max_receiver_channels_per_vc = 0;
};

// Every problem found: missing or wrong eth edges, unfed forwarding senders, noc edges that leave
// the chip or that the rules don't wire, and channel counts over the limits.
std::vector<std::string> find_problems(
    const LogicalGraph& graph, const WiringRules& rules, const ValidationLimits& limits);

void validate(const LogicalGraph& graph, const WiringRules& rules, const ValidationLimits& limits);

}  // namespace tt::tt_fabric::graph
