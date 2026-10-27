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

// How a producer returns sender-channel credits to its receiver: by subtracting from the channel's
// free-slots stream register (Wormhole, Blackhole), or, on hardware without stream registers, by
// writing its free-running write count into the channel's SenderChannelProducerCursor. Every fabric
// kernel, producer or receiver, is built with FABRIC_STREAM_REGISTERS_AVAILABLE set to its router's
// STREAM_REGISTERS_AVAILABLE; it defaults to 1.
#ifndef FABRIC_STREAM_REGISTERS_AVAILABLE
#define FABRIC_STREAM_REGISTERS_AVAILABLE 1
#endif
inline constexpr bool kernel_stream_registers_available = FABRIC_STREAM_REGISTERS_AVAILABLE != 0;

// EDMChannelWorkerLocationInfo::credit_mode values. Neither is 0, so a receiver that never wrote the
// field fails the producer's check in both modes. kernel_credit_mode is this kernel's own: a receiver
// (router, mux) writes it into each channel's connection info, a producer compares against it.
inline constexpr uint32_t credit_mode_stream_registers = 0xC2ED0001;
inline constexpr uint32_t credit_mode_software_counters = 0xC2ED0002;
inline constexpr uint32_t kernel_credit_mode =
    kernel_stream_registers_available ? credit_mode_stream_registers : credit_mode_software_counters;

};  // namespace tt::tt_fabric::connection_interface
