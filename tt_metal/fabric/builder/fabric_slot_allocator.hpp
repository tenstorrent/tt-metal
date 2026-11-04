// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Slot allocation for a fabric router's channel buffers.
//
// allocate_slots() is a pure function: the router's shape in, a slot count per channel out, each with the stage and
// the reason that set it. It reads only facts the host knows without measuring anything: the channel buffer bytes,
// the slot size, the VCs built and their channels, and whether the router is 1D or 2D. Standard library only.
//
// Each channel kind has a full-rate depth: the slots the channel needs to run at full rate when it alone carries its
// link. A slot holds one packet whatever its size, so depths are in slots, not bytes. Small packets are router-bound
// and set the depth; larger ones take longer each and need fewer.
//
// The stages:
//   0. N = channel buffer bytes / slot size. Every channel gets at least min_slots and at most max_slots; minimums
//      that do not fit in N are an error.
//   1. Receivers first: each gets its full-rate depth scaled by f = min(1, N / sum of every channel's depth), rounded
//      to the nearest power of 2 (ties up). They go first because rules downstream read the final receiver count
//      (the ring index wrap, the completion group size, the transaction IDs, the bubble flow control threshold).
//   2. Senders: the slots left, B, split by f' = min(1, B / sum of the senders' depths), by largest remainder. A
//      VC's forwarding channels are one unit with one count: the unit takes the floor of f' x depth on every channel
//      and grows by a slot on each of them at a time. Each worker is a unit of its own. A unit below the minimum is
//      raised, with slots from the unit most above its own.
//   3. What is left (the slots past every full-rate depth, or the rounding a squeeze's units could not take) goes to
//      the groups of the leftover order in turn, each in whole steps. A group's receivers take one step: each to its
//      next power of 2, all of them or none, so their counts stay powers of 2, and none past
//      max_receiver_slots_from_leftovers. A group's senders take +1 a unit at a time, unit after unit in the group's
//      order and round after round, up to max_slots. A step that does not fit whole, or a group with no channels,
//      passes what is left on to the next group. The default order ends with the VC0 worker, which takes the rest.
//
// The router contract stages 2 and 3 keep: the router has one downstream forwarding ring size per VC
// (router_downstream.hpp, DownstreamTraits::num_slots), so every forwarding channel of a VC has the same count.

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

const char* to_string(ChannelKind kind);

// Full-rate depths, in slots. 2D: mesh and torus routers; 1D: line and ring routers.
//
// The VC0 worker's slot frees only after the router has sent the packet and the far receiver's credit has come back
// over Ethernet; the depth covers that loop at the router's per-packet rate. A 1D router has fewer channels to serve
// per pass, so its per-packet time is shorter and the same loop takes more packets. The 2D depth sits in the measured
// knee of 17-20 slots.
inline constexpr std::size_t vc0_worker_full_rate_depth_2d = 18;
inline constexpr std::size_t vc0_worker_full_rate_depth_1d = 24;
// A forwarding channel's loop is the worker's one hop on: the packet arrives from the router's own receiver, is sent,
// and the slot frees when the next router's credit returns.
inline constexpr std::size_t vc0_forwarding_full_rate_depth_2d = 16;
inline constexpr std::size_t vc0_forwarding_full_rate_depth_1d = 24;
// A receiver slot frees only when its local write and its forward have left the router and its credit has been sent
// back to the far sender.
inline constexpr std::size_t vc0_receiver_full_rate_depth_2d = 16;
inline constexpr std::size_t vc0_receiver_full_rate_depth_1d = 32;
// VC1 carries inter-mesh traffic through forwarding channels only. Its slots free by the same loops as VC0's
// forwarding and receiver channels; the depths are those loops with VC1 alone on its link.
inline constexpr std::size_t vc1_forwarding_full_rate_depth_2d = 10;
inline constexpr std::size_t vc1_receiver_full_rate_depth_2d = 8;
// VC2 carries one-hop neighbor exchange to a delivery-only receiver; what it carries depends on the workload, so its
// depths start near the minimum and are meant to grow by override.
inline constexpr std::size_t vc2_worker_full_rate_depth_2d = 4;
inline constexpr std::size_t vc2_receiver_full_rate_depth_2d = 4;

// The full-rate depth of a channel kind on a VC. VC1 and VC2 are built on 2D fabrics, so they have 2D depths only.
std::size_t full_rate_depth(std::size_t vc, ChannelKind kind, bool one_d);

// The facts the stages read about one router.
struct RouterShape {
    std::size_t buffer_bytes = 0;  // bytes available for channel buffers
    std::size_t slot_bytes = 0;    // one slot: the largest packet, header included

    // Channels of each kind on each VC; a VC with no channel is not built.
    struct Vc {
        std::size_t workers = 0;
        std::size_t forwarding = 0;
        std::size_t receivers = 0;
    };
    std::array<Vc, NUM_VCS> vcs{};

    bool one_d = false;  // a 1D router (line, ring); otherwise 2D (mesh, torus)
};

// One channel kind on one VC, as the leftover order names it.
using ChannelKindOnVc = std::pair<std::size_t, ChannelKind>;
// Groups of channel kinds, in the order stage 3 gives them leftover slots. Data, not branches: an override can replace
// the list.
using LeftoverOrder = std::vector<std::vector<ChannelKindOnVc>>;

// The default leftover order. Both start with the VC0 receiver (to its next power of 2, when that whole step fits): a
// deeper receiver ring lets the far sender run further ahead.
//   2D: then the VC0 forwarding channels, then the VC0 worker channel. A worker past its full-rate depth already runs
//       at full rate; traffic that turns or crosses several hops through a forwarding channel can use more than the
//       depth set for straight-through traffic. The worker takes what no whole forwarding round fits.
//   1D: then the VC0 worker and forwarding channel in one group, +1 on each in turn, worker first. Both have the same
//       full-rate depth on a 1D router, so the rest is spread over them evenly.
// On a router with no forwarding channels (one that only exchanges with its neighbor) the worker takes the rest.
LeftoverOrder default_leftover_order(bool one_d);

struct ChannelSlots {
    std::size_t vc = 0;
    ChannelKind kind = ChannelKind::Worker;
    std::size_t index = 0;  // among the channels of this kind on this VC
    std::size_t slots = 0;
    std::size_t stage = 0;  // the last stage that set the count
    std::string reason;
};

struct Allocation {
    std::size_t total_slots = 0;         // N
    std::size_t unused_slots = 0;        // slots no group of the leftover order takes in whole steps
    std::vector<ChannelSlots> channels;  // receivers first, then senders, each in VC order
    std::vector<std::string> notes;      // the squeeze factors
    std::optional<std::string> error;

    // The slots of the index-th channel of a kind on a VC (0 if there is none).
    std::size_t slots(std::size_t vc, ChannelKind kind, std::size_t index = 0) const;
};

struct Limits {
    // A channel needs two slots to fill one while the other is sent; bubble flow control also needs two free slots at
    // a ring's receiver.
    std::size_t min_slots = 2;
    std::size_t max_slots = 127;  // the largest slot count the router takes per channel
    // Stage 3 raises a receiver no deeper than this: the deepest receiver measured. Rules that read the receiver count
    // (the bubble flow control threshold, the transaction IDs per slot) were checked up to it.
    std::size_t max_receiver_slots_from_leftovers = 32;
};

Allocation allocate_slots(const RouterShape& shape, const LeftoverOrder& leftover_order, const Limits& limits = {});

// The same, with default_leftover_order(shape.one_d).
Allocation allocate_slots(const RouterShape& shape, const Limits& limits = {});

}  // namespace tt::tt_fabric::slot_allocator
