// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// routing_l1_info_t as JSON: one writer per scope its members have. routing_l1_info_t is the interpreted value, so
// nothing here unpacks it into a copy. Each member is read through the kernel's own readers and named through the
// kernel's own types.

#pragma once

#include <cstdint>

#include <hostdevcommon/fabric_common.h>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"

namespace tt::tt_fabric::decode {

// Every member of the routing table but state_manager, which is the router's own. Which of the 1D and 2D members it
// writes comes from context.fabric (the others are null).
json make_chip_routing_table_json(const routing_l1_info_t& table, const InterpretContext& context);

// Every name make_chip_routing_table_json writes, for analysis.vocabulary: a direction-table entry's (the directions'
// names, then the two eth_chan_magic_values'), a 2D action-vector or multicast-tree code's ("stop", then the
// directions'), and a 1D hop action's.
json make_direction_table_entry_vocabulary_json();
json make_2d_axis_code_vocabulary_json();
json make_1d_hop_action_vocabulary_json();

namespace detail {

// One direction-table entry.
json make_direction_table_entry_json(uint8_t original_direction);

// A 1D hop action's name.
json make_1d_hop_action_json(RoutingFieldsConstants::LowLatency::HopAction action);

// A 2-bit action-vector or multicast-tree code on `axis`.
json make_2d_axis_code_json(Fabric2DAxis axis, uint8_t code);

}  // namespace detail

}  // namespace tt::tt_fabric::decode
