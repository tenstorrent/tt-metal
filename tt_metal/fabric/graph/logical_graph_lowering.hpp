// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "fabric_graph_types.hpp"
#include "fabric_wiring_rules.hpp"

namespace tt::tt_fabric::graph {

// One router per port. Each router gets a receiver (and maybe a worker) per VC it carries, and one
// forwarding sender per ingress direction the rules wire into it.
LogicalGraph lower_to_logical(const std::vector<Port>& ports, const WiringRules& rules);

// A turn the 1D corner fold adds: VC0 between the only two routers of a chip and plane, not opposite.
bool is_folded_corner_turn(const LogicalGraph& graph, const WiringRules& rules, RouterIndex up, RouterIndex down, uint32_t vc);

}  // namespace tt::tt_fabric::graph
