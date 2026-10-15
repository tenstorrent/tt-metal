// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Returns fabric's completion credits from the host for a bridged sender channel. Without it the
// sender halts at credit_ceiling_frames(): no far receiver exists to complete what the bridge took.
#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/device.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "tt_metal/distributed/erisc_bridge_doorbell.hpp"

namespace tt::tt_fabric::erisc_bridge {

// One bridged sender channel's return path. Counters take the running TOTAL and are idempotent;
// stream registers take the DELTA, so a lost or repeated write corrupts the count with no signal.
struct E2HCreditReturn {
    // Completions restore the sender's outbound slots; ACKS are what release the upstream worker
    // once first level ack is on -- and bubble flow control requires it, so both must be returned.
    struct Line {
        std::uint32_t addr = 0;    // counter form: base + 4 * sender channel
        std::uint32_t stream = 0;  // register form
        // SEEDED FROM THE LIVE COUNTER, never 0: the router keeps its processed total across
        // benchmark cases, so restarting at 0 underflows its difference and pins free slots negative.
        std::uint32_t total = 0;
        std::uint32_t injected = 0;
        std::uint32_t seeded = 0;
    };

    bool active = false;
    bool use_streams = false;
    tt::tt_metal::MetalEnvImpl* env = nullptr;
    tt::tt_metal::IDevice* dev = nullptr;
    const tt::tt_metal::Hal* hal = nullptr;
    tt::tt_metal::CoreCoord eth;
    Line acks;
    Line completions;

    // ACKS FIRST: the router reads completions then acks and relies on never seeing more
    // completions than acks, which is also what a real receiver produces -- it acks on arrival.
    bool give(std::uint32_t frames, std::string& err) {
        if (!active || frames == 0) {
            return true;
        }
        return write(acks, frames, "ack", err) && write(completions, frames, "completion", err);
    }

private:
    // Advances the totals only on success, so a failed write is retried rather than skipped.
    bool write(Line& l, std::uint32_t frames, const char* what, std::string& err) {
        const std::uint32_t want = l.total + frames;
        if (!use_streams) {
            if (!write_bytes_to_l1(*env, dev, eth, l.addr, &want, sizeof(want), tt::CoreType::ETH)) {
                err = std::string("credit return: ") + what + " counter write failed";
                return false;
            }
            l.total = want;
            return true;
        }
        if (!reg_write(
                *env,
                dev,
                eth,
                tt::CoreType::ETH,
                doorbell_addr(*hal, l.stream),
                doorbell_value(static_cast<std::int32_t>(want - l.injected), kWordsFreeIncShiftBlackhole))) {
            err = std::string("credit return: ") + what + " stream write failed";
            return false;
        }
        l.total = want;
        l.injected = want;
        return true;
    }
};

}  // namespace tt::tt_fabric::erisc_bridge
