// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Buffer slots for a fabric router's channels.
//
// allocate_slots() is a pure function of what the host knows about a router: the channel buffer bytes, the slot size,
// the VCs and their channels, and whether the router is 2D or 1D. It returns a slot count per channel and the stage
// that set it.
//
// Each channel kind has a full-rate depth: the slots it needs to run at full rate when it alone carries its link. A
// slot holds one packet whatever its size, so depths are in slots.
//
// The stages:
//   0. N = buffer bytes / slot size. Every channel gets at least min_slots; minimums that do not fit are an error.
//   1. Receivers first, at their depth x f, f = min(1, N / every channel's depth), rounded to the nearest power of 2.
//      They go first because the router derives rules from the receiver count (index wrap, completion groups,
//      transaction IDs, bubble flow control).
//   2. Senders split the B slots left at their depth x f', f' = min(1, B / the senders' depths), by largest remainder.
//      A VC's forwarding channels are one unit with one count; each worker is a unit of its own.
//   3. Leftover slots go to the groups of the leftover order in turn, in whole steps: receivers to their next power of
//      2 (all of a group's or none, up to max_receiver_slots_from_leftovers), senders +1 a unit at a time. A step that
//      does not fit passes the rest on.
//
// Contract with the router: it keeps one downstream buffer count per VC (router_downstream.hpp,
// DownstreamTraits::num_slots), so every forwarding channel of a VC has the same count.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tt::tt_fabric::slot_allocator {

inline constexpr std::size_t NUM_VCS = 3;

// What a channel does.
enum class ChannelKind : std::size_t {
    Worker = 0,      // a sender channel fed by local workers
    Forwarding = 1,  // a sender channel fed by this router's receivers (traffic passing through)
    Receiver = 2,    // a receiver channel, fed by the far router's senders
};

// The full-rate depths of one VC's channel kinds, in slots.
struct FullRateDepths {
    std::size_t worker = 0;
    std::size_t forwarding = 0;
    std::size_t receiver = 0;
};

// The depths are measured, not chosen: each is the knee of a slot-depth sweep on Blackhole, the smallest depth within
// 1% of the best rate that channel kind reached, with small packets (router-bound, so they need the most slots). VC2's
// are a starting floor, meant to grow by override.
//
// 2D routers (mesh, torus), per VC:
//   VC0: a worker's or forwarding channel's slot frees when the far receiver's credit returns; a receiver's, when its
//     local write and forward have left and its credit has been sent.
//   VC1: inter-mesh forwarding only (a VC1 worker, if built, counts as forwarding).
//   VC2: neighbor exchange into a delivery-only receiver (a VC2 forwarding channel, if built, counts as its worker).
inline constexpr std::array<FullRateDepths, NUM_VCS> full_rate_depths_2d = {{
    {.worker = 18, .forwarding = 16, .receiver = 16},
    {.worker = 10, .forwarding = 10, .receiver = 8},
    {.worker = 4, .forwarding = 4, .receiver = 4},
}};
// 1D routers (line, ring) build VC0 only. They serve fewer channels per pass, so the same credit loop holds more packets.
inline constexpr FullRateDepths vc0_full_rate_depths_1d = {.worker = 24, .forwarding = 24, .receiver = 32};

// The full-rate depth of a channel kind on a VC (0 past the last VC).
std::size_t full_rate_depth(std::size_t vc, ChannelKind kind, bool is_2d);

// What the stages read about one router.
struct RouterShape {
    std::size_t buffer_bytes = 0;  // bytes for channel buffers
    std::size_t slot_bytes = 0;    // one slot: the largest packet, header included

    // Channels of each kind on each VC; a VC with no channel is not built.
    struct Vc {
        std::size_t workers = 0;
        std::size_t forwarding = 0;
        std::size_t receivers = 0;
    };
    std::array<Vc, NUM_VCS> vcs{};

    bool is_2d = true;  // a 2D router (mesh, torus); otherwise 1D (line, ring)
};

// One channel kind on one VC.
using ChannelKindOnVc = std::pair<std::size_t, ChannelKind>;
// Groups of channel kinds, in the order stage 3 gives them leftover slots (data, so an override can replace it).
using LeftoverOrder = std::vector<std::vector<ChannelKindOnVc>>;

// The default leftover order. Both start with the VC0 receiver: a deeper receiver lets the far sender run further
// ahead.
//   2D: then the VC0 forwarding channels (traffic that turns or crosses several hops can use more than its depth),
//       then the VC0 worker, which takes the rest.
//   1D: then the VC0 worker and forwarding channel together, +1 on each in turn (same full-rate depth).
// A router without forwarding channels gives the rest to its worker.
LeftoverOrder default_leftover_order(bool is_2d);

struct ChannelSlots {
    std::size_t vc = 0;
    ChannelKind kind = ChannelKind::Worker;
    std::size_t index = 0;  // among the channels of this kind on this VC
    std::size_t slots = 0;
    std::size_t stage = 0;  // the last stage that set the count
};

struct Allocation {
    std::size_t total_slots = 0;         // N
    std::size_t unused_slots = 0;        // left over in no whole step
    std::vector<ChannelSlots> channels;  // receivers first, then senders, each in VC order
    // Stage 1's squeeze is total_slots / all_depths, stage 2's sender_slots / sender_depths (each capped at 1).
    std::size_t all_depths = 0;
    std::size_t sender_slots = 0;
    std::size_t sender_depths = 0;
    std::optional<std::string> error;

    // The slots of the index-th channel of a kind on a VC (0 if there is none).
    std::size_t slots(std::size_t vc, ChannelKind kind, std::size_t index = 0) const;
};

struct Limits {
    // Two slots: one filling while the other is sent; bubble flow control also needs two free at a receiver on a ring
    // or torus.
    std::size_t min_slots = 2;
    std::size_t max_slots = 127;  // the most slots the router takes per channel
    // Stage 3 deepens a receiver no further than this: the deepest receiver measured with the rules that read the
    // receiver count.
    std::size_t max_receiver_slots_from_leftovers = 32;
};

Allocation allocate_slots(const RouterShape& shape, const LeftoverOrder& leftover_order, const Limits& limits = {});

// The same, with default_leftover_order(shape.is_2d).
Allocation allocate_slots(const RouterShape& shape, const Limits& limits = {});

}  // namespace tt::tt_fabric::slot_allocator
