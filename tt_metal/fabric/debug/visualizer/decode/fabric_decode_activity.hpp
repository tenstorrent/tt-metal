// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Activity: whether a router was moving during the capture, and if its loop ran, whether it made progress. It's judged
// from the regions the manifest marks multi_sampled, by their role, across every read of them: each round's, then the
// full read's. The router's analysis.activity lists every read's value, then the verdict.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// What a router was doing during the capture. The decoded file writes it by lower_enum_name.
enum class RouterActivity {
    // The loop heartbeat and every telemetry heartbeat the router writes moved: moving packets, or idle. Its L1 may
    // be torn, since it changed while capture read it.
    RUNNING,
    // The loop heartbeat moved, but at least one telemetry heartbeat the router writes didn't: a side has work it
    // can't do. The hang case; its L1 is consistent.
    STUCK,
    // The loop heartbeat moved, and no telemetry heartbeat says more: running and stuck can't be told apart.
    LOOPING,
    // The loop heartbeat didn't move, or no read of it was the router's: the router never started, is stuck before
    // or outside its main loop, or exited.
    NOT_LOOPING,
    // Too few of the router's reads of the loop heartbeat to tell.
    UNKNOWN,
};

// Which side of an ERISC a telemetry heartbeat is. The decoded file writes it by lower_enum_name.
enum class HeartbeatSide {
    TX,
    RX,
};

// One read of a heartbeat: which pass made it (null for the full read), when, and what it held.
struct HeartbeatRead {
    std::optional<uint32_t> round;
    uint64_t t_ns = 0;
    uint64_t value = 0;
};

// The region the manifest marks LOOP_HEARTBEAT, across its reads.
struct LoopHeartbeat {
    // The region's path below the router's node, such as "l1_areas/heartbeat". Empty when the manifest marks none.
    std::string path;
    // Every read of it, the full read last.
    std::vector<HeartbeatRead> reads;
    // Across the reads with the router's magic: whether any two counts differ. Null with fewer than two such reads.
    std::optional<bool> moved;
};

// One side of one ERISC's heartbeats in the region the manifest marks FABRIC_TELEMETRY.
struct TelemetryHeartbeat {
    uint32_t erisc = 0;
    HeartbeatSide side = HeartbeatSide::TX;
    // Whether the ERISC writes it: it runs the side (services a sender channel for TX, a receiver channel for RX) and
    // its telemetry fields enable the statistic (writes_telemetry_statistic). False also when a read's FabricTelemetry
    // is another version than this build's, since its layout is then another.
    bool written = false;
    // Every read of it, the full read last. Empty when it isn't written.
    std::vector<HeartbeatRead> reads;
    // Whether any two reads differ. Null when it isn't written, or with fewer than two reads.
    std::optional<bool> moved;
};

struct RouterActivityVerdict {
    LoopHeartbeat loop_heartbeat;
    // One per side per ERISC of the router.
    std::vector<TelemetryHeartbeat> telemetry_heartbeats;
    RouterActivity verdict = RouterActivity::UNKNOWN;
    // The written telemetry heartbeats that didn't move, when the verdict is STUCK.
    std::vector<std::pair<uint32_t, HeartbeatSide>> stopped;
};

// Whether a loop heartbeat word is the router's: its bits under magic_mask are the router's magic. Base firmware
// writes the same word with its own magic, which decode needn't know.
bool is_router_heartbeat(uint32_t word, const manifest::HeartbeatProtocol& protocol);

// The iteration count in a router's loop heartbeat word: its bits outside magic_mask.
uint32_t get_heartbeat_count(uint32_t word, const manifest::HeartbeatProtocol& protocol);

// The router's activity, from its regions the manifest marks multi_sampled, found by role (LOOP_HEARTBEAT,
// FABRIC_TELEMETRY), so a heartbeat the manifest marks later on the router counts with no change here. Fails when the
// manifest marks more than one region with either role, which this reading doesn't combine.
RouterActivityVerdict judge_router_activity(const RouterValues& values, const manifest::Arch& arch);

// The router's analysis.activity: each heartbeat's reads and whether it moved, then the verdict.
json make_activity_json(const RouterActivityVerdict& activity, const manifest::HeartbeatProtocol& protocol);

}  // namespace tt::tt_fabric::decode
