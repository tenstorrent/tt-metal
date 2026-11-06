// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dynamic slot allocation for fabric router channels (study prototype, fabric_exp/consolidation/SLOT_ALLOCATOR.md).
//
// A pure function: the router's shape in, slots per channel out, each count with the reason it got it. It reads only
// facts the host knows without measuring anything (the channel buffer bytes, the slot size, which VCs exist and their
// channel kinds, the mesh dims); the policy is a few integer ratios fitted to the slot study's measured rows. Standard
// library only, so fabric_exp's offline tool compiles this same file.
//
// Step 1 (now): the host fills one shape for every router of the fabric. Step 2 (later): one shape per router, from
// that router's own facts (its axis, its mesh edge, its trimmed channels); the rules do not change.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tt::tt_fabric::slot_allocator {

inline constexpr std::size_t NUM_VCS = 3;

// What a channel does, which is what its weight follows.
enum class ChannelKind : std::size_t {
    Worker = 0,      // a sender channel fed by local workers (injection)
    Forwarding = 1,  // a sender channel fed by this router's receivers (traffic passing through)
    Receiver = 2,    // a receiver channel, fed by the far router's senders
};
inline constexpr std::size_t NUM_CHANNEL_KINDS = 3;

const char* to_string(ChannelKind kind);

// The facts the rules read about one router.
struct RouterShape {
    std::size_t budget_bytes = 0;  // L1 bytes for channel buffers (the arch's layout); only used for N
    std::size_t slot_bytes = 0;    // one packet slot: the largest payload plus the packet header

    // Channels per kind of each VC. A VC is present when it has any channel.
    struct Vc {
        std::size_t workers = 0;
        std::size_t forwarding = 0;
        std::size_t receivers = 0;
        bool present() const { return workers + forwarding + receivers > 0; }
    };
    std::array<Vc, NUM_VCS> vcs{};

    // Chips along each mesh dim (Y, X); step 1: the largest of the fabric's meshes.
    std::array<std::size_t, 2> mesh_dims{};

    // A 2D router (mesh or torus): the VC0 receiver floor applies.
    bool two_d = false;
};

// How a VC0 receiver's share is rounded: the nearest power of 2 by ratio (E1's ring index is a mask), or the nearest
// multiple of 4 (whole completion groups).
enum class ReceiverRound { Pow2, Mult4 };

// The integer ratios and options. Defaults: SLOT_ALLOCATOR.md, where each one comes from.
struct Policy {
    // Each VC's share of the slots, for the VCs built (VC order). Empty: the default for that set of VCs
    // (VC0 alone 1; VC0+VC1 3:1; VC0+VC2 12:1; VC0+VC1+VC2 12:4:1). "VC1 first" is 2:1.
    std::vector<unsigned> vc_ratio;
    // Kind weights inside a VC. Forwarding counts long_dim_bump times when a mesh dim has more than 2 chips (traffic
    // can pass straight through a router only there).
    unsigned worker_weight = 2;
    unsigned forwarding_weight = 1;
    unsigned long_dim_bump = 2;
    // A receiver weighs receiver_num / receiver_den of its VC's heaviest sender kind.
    unsigned receiver_num = 4;
    unsigned receiver_den = 5;
    ReceiverRound receiver_round = ReceiverRound::Pow2;
    // On a 2D router the VC0 receiver's share is at least floor_num / floor_den of N, before the rounding, as far as
    // VC0's senders keep their minimum (floor_num 0: off). 1/5 is 16 of 81 at 4352 B, the smallest power of 2 that
    // binds no 2D row.
    unsigned vc0_receiver_floor_num = 1;
    unsigned vc0_receiver_floor_den = 5;
    // The VC0 worker's cap in slots (0: none); the leftover past it goes to VC0 forwarding.
    std::size_t worker_cap_slots = 0;
    std::size_t min_slots = 2;    // every channel (a ring VC0 receiver: the bubble rule's 2)
    std::size_t max_slots = 127;  // per channel (MAX_CHANNEL_BUFFER_SLOTS on the device)
};

// The default VC ratio for the VCs present (VC order), as Policy::vc_ratio documents.
std::vector<unsigned> default_vc_ratio(const RouterShape& shape);

// Applies "key=value,key=value" to a policy (keys: SLOT_ALLOCATOR.md). Returns an error message, or nothing.
std::optional<std::string> parse_policy(const std::string& text, Policy& policy);

struct Count {
    std::size_t channels = 0;  // channels of this kind in the VC (0: none)
    std::size_t slots = 0;     // slots per channel
    std::string reason;
};

struct Allocation {
    std::size_t total_slots = 0;   // N = budget / slot size
    std::size_t unused_slots = 0;  // left after every rule (a remainder VC0 forwarding cannot split)
    std::array<std::array<Count, NUM_CHANNEL_KINDS>, NUM_VCS> counts{};  // [vc][kind]
    std::vector<std::string> notes;                                      // the shape and the VC shares
    std::optional<std::string> error;                                    // the shape does not fit the budget

    const Count& at(std::size_t vc, ChannelKind kind) const { return counts[vc][static_cast<std::size_t>(kind)]; }
};

Allocation allocate_slots(const RouterShape& shape, const Policy& policy = {});

}  // namespace tt::tt_fabric::slot_allocator
