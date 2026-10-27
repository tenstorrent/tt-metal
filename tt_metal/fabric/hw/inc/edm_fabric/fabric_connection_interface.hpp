// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_fabric::connection_interface {

inline constexpr uint8_t edm_fabric_write_noc_index = 0;

// Constants that define different connection states for a connection to a fabric router
static constexpr uint32_t unused_connection_value = 0;
static constexpr uint32_t open_connection_value = 1;
static constexpr uint32_t close_connection_request_value = 2;

// Stream ID for the worker connection credits: the local worker writes to the auto-inc register of
// this stream ID to notify the fabric router of new packets available.
//
// This is the single authority for the worker-facing id: the router's stream assignment pins the
// VC0 sender-free-slots base to exactly this value, and worker-space (adapters) and ControlPlane
// (populate_fabric_connection_info) read the constant rather than the number -- so all sides agree
// by construction. Id 0 is an ordinary register here: the inactive sentinel is out of range (see
// k_unused_stream_id), so the pinned base can sit at the bottom of the file.
static constexpr uint32_t sender_channel_0_free_slots_stream_id = 0;

// VC2 worker sender flow control stream ID (dual-use with tensix_relay; VC2 and UDM/mux are mutually exclusive)
static constexpr uint32_t vc2_sender_free_slots_stream_id = 30;

};  // namespace tt::tt_fabric::connection_interface
