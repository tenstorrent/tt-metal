// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router's buffer slot allocator (fabric_slot_allocator.hpp) and transaction ID allocator
// (fabric_trid_allocator.hpp): pure functions, tested without a device. The checks are the rules each allocator
// promises, over a range of router shapes and buffer sizes, not exact slot counts.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "tt_metal/fabric/builder/fabric_slot_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_trid_allocator.hpp"

namespace tt::tt_fabric {
namespace {

namespace sa = slot_allocator;
namespace trids = trid_allocator;

bool is_power_of_2(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

struct NamedShape {
    std::string name;
    sa::RouterShape shape;  // buffer_bytes left 0: each test sets the budget
};

// Router shapes the fabric builds: VC0 with its worker and forwarding channels, plus VC1 (forwarding only) and VC2 (a
// worker into a delivery-only receiver); 1D lines and rings; a neighbor-exchange router (no forwarding).
std::vector<NamedShape> router_shapes() {
    const auto shape = [](bool is_2d, sa::RouterShape::Vc vc0, sa::RouterShape::Vc vc1, sa::RouterShape::Vc vc2) {
        sa::RouterShape s;
        s.slot_bytes = is_2d ? 4448 : 4400;  // 4352 B payload plus the 2D / 1D header
        s.is_2d = is_2d;
        s.vcs = {vc0, vc1, vc2};
        return s;
    };
    return {
        {"2D VC0", shape(true, {1, 3, 1}, {}, {})},
        {"2D VC0 + VC1", shape(true, {1, 3, 1}, {0, 3, 1}, {})},
        {"2D VC0 + VC2", shape(true, {1, 3, 1}, {}, {1, 0, 1})},
        {"2D VC0 + VC1 + VC2", shape(true, {1, 3, 1}, {0, 3, 1}, {1, 0, 1})},
        {"1D", shape(false, {1, 1, 1}, {}, {})},
        {"neighbor exchange", shape(false, {1, 0, 1}, {}, {})},
    };
}

std::size_t num_channels(const sa::RouterShape& shape) {
    std::size_t n = 0;
    for (const auto& vc : shape.vcs) {
        n += vc.workers + vc.forwarding + vc.receivers;
    }
    return n;
}

// The rules every successful allocation keeps.
void expect_rules_hold(const std::string& name, const sa::RouterShape& shape, const sa::Allocation& a) {
    SCOPED_TRACE(name + ", " + std::to_string(shape.buffer_bytes / shape.slot_bytes) + " slots");
    const sa::Limits limits;
    ASSERT_FALSE(a.error.has_value()) << a.error.value_or("");
    EXPECT_EQ(a.total_slots, shape.buffer_bytes / shape.slot_bytes);
    ASSERT_EQ(a.channels.size(), num_channels(shape));

    // Every slot is given out or counted unused, and no channel is outside its limits.
    std::size_t given = 0;
    for (const auto& c : a.channels) {
        given += c.slots;
        EXPECT_GE(c.slots, limits.min_slots);
        EXPECT_LE(c.slots, limits.max_slots);
        EXPECT_EQ(a.slots(c.vc, c.kind, c.index), c.slots);
        if (c.kind == sa::ChannelKind::Receiver) {
            // The router derives its index wrap, completion groups and transaction IDs from a power of 2.
            EXPECT_TRUE(is_power_of_2(c.slots)) << "VC" << c.vc << " receiver: " << c.slots;
            EXPECT_LE(c.slots, limits.max_receiver_slots_from_leftovers);
        }
    }
    EXPECT_EQ(given + a.unused_slots, a.total_slots);

    // The router keeps one downstream buffer count per VC: a VC's forwarding channels share one count.
    for (std::size_t vc = 0; vc < sa::NUM_VCS; ++vc) {
        for (std::size_t i = 1; i < shape.vcs[vc].forwarding; ++i) {
            EXPECT_EQ(a.slots(vc, sa::ChannelKind::Forwarding, i), a.slots(vc, sa::ChannelKind::Forwarding, 0))
                << "VC" << vc << " forwarding channel " << i;
        }
    }

    // Where the buffer covers every full-rate depth, every channel gets at least its depth.
    if (a.total_slots >= a.all_depths) {
        for (const auto& c : a.channels) {
            EXPECT_GE(c.slots, std::min(sa::full_rate_depth(c.vc, c.kind, shape.is_2d), limits.max_slots))
                << "VC" << c.vc << " kind " << static_cast<std::size_t>(c.kind) << " index " << c.index;
        }
    }
}

TEST(FabricSlotAllocatorTest, RulesHoldFromTheMinimumBudgetUp) {
    const sa::Limits limits;
    for (const auto& [name, base] : router_shapes()) {
        const std::size_t minimum = num_channels(base) * limits.min_slots;
        for (std::size_t n = minimum; n <= 300; ++n) {
            sa::RouterShape shape = base;
            shape.buffer_bytes = n * shape.slot_bytes + shape.slot_bytes / 2;  // a partial slot is not a slot
            expect_rules_hold(name, shape, sa::allocate_slots(shape));
        }
    }
}

TEST(FabricSlotAllocatorTest, SameShapeSameAllocation) {
    // Every router of a fabric must decide alike: an upstream router's view of a downstream's depths depends on it.
    for (auto [name, shape] : router_shapes()) {
        shape.buffer_bytes = 81 * shape.slot_bytes;
        const sa::Allocation a = sa::allocate_slots(shape);
        const sa::Allocation b = sa::allocate_slots(shape);
        ASSERT_EQ(a.channels.size(), b.channels.size()) << name;
        for (std::size_t i = 0; i < a.channels.size(); ++i) {
            EXPECT_EQ(a.channels[i].slots, b.channels[i].slots) << name << ", channel " << i;
        }
    }
}

TEST(FabricSlotAllocatorTest, LeftoversFollowTheLeftoverOrder) {
    // A generous budget: the 2D order deepens the VC0 receiver first, then forwarding, and the worker takes the rest.
    sa::RouterShape shape = router_shapes()[0].shape;
    shape.buffer_bytes = 200 * shape.slot_bytes;
    const sa::Allocation a = sa::allocate_slots(shape);
    ASSERT_FALSE(a.error.has_value());
    const sa::Limits limits;
    EXPECT_EQ(a.slots(0, sa::ChannelKind::Receiver), limits.max_receiver_slots_from_leftovers);
    EXPECT_GT(a.slots(0, sa::ChannelKind::Forwarding), sa::full_rate_depth(0, sa::ChannelKind::Forwarding, true));
    EXPECT_EQ(a.unused_slots, 0u);
}

TEST(FabricSlotAllocatorTest, TooSmallABufferIsAnError) {
    const sa::Limits limits;
    for (auto [name, shape] : router_shapes()) {
        shape.buffer_bytes = (num_channels(shape) * limits.min_slots - 1) * shape.slot_bytes;
        EXPECT_TRUE(sa::allocate_slots(shape).error.has_value()) << name;
    }
    sa::RouterShape no_slot_size = router_shapes()[0].shape;
    no_slot_size.buffer_bytes = 81 * 4448;
    no_slot_size.slot_bytes = 0;
    EXPECT_TRUE(sa::allocate_slots(no_slot_size).error.has_value());
}

TEST(FabricTridAllocatorTest, GroupSizeFollowsSlots) {
    EXPECT_EQ(trids::completion_group_size({.slots = 4}), 1u);
    EXPECT_EQ(trids::completion_group_size({.slots = 8}), 2u);
    EXPECT_EQ(trids::completion_group_size({.slots = 15}), 2u);
    EXPECT_EQ(trids::completion_group_size({.slots = 16}), 4u);
    EXPECT_EQ(trids::completion_group_size({.slots = 32}), 4u);
    // A channel credited slot by slot.
    EXPECT_EQ(trids::completion_group_size({.slots = 32, .credit_each_slot = true}), 1u);
}

// The rules every successful assignment keeps.
void expect_ids_hold(const std::vector<trids::ReceiverChannel>& channels, const trids::Assignment& ids) {
    ASSERT_FALSE(ids.error.has_value()) << ids.error.value_or("");
    ASSERT_EQ(ids.channels.size(), channels.size());
    std::size_t next = 0;
    for (std::size_t i = 0; i < channels.size(); ++i) {
        const trids::ChannelIds& c = ids.channels[i];
        EXPECT_EQ(c.group_size, trids::completion_group_size(channels[i]));
        EXPECT_EQ(c.groups, (channels[i].slots + c.group_size - 1) / c.group_size);
        EXPECT_GE(c.num_ids, 1u);
        EXPECT_LE(c.num_ids, std::max<std::size_t>(c.groups, 1));
        if (c.num_ids < c.groups) {
            EXPECT_TRUE(is_power_of_2(c.num_ids)) << "channel " << i << " shares " << c.num_ids << " IDs by mask";
        }
        EXPECT_EQ(c.start, next) << "channel " << i;  // consecutive ranges, in channel order
        next += c.num_ids;
    }
    EXPECT_EQ(ids.used_ids, next);
    EXPECT_LE(ids.used_ids, trids::noc_transaction_ids);
}

TEST(FabricTridAllocatorTest, EachGroupGetsAnIdWhenTheyFit) {
    const std::vector<trids::ReceiverChannel> channels = {{.slots = 16}, {.slots = 8}};  // 4 + 4 groups
    const trids::Assignment ids = trids::assign_transaction_ids(channels);
    expect_ids_hold(channels, ids);
    EXPECT_EQ(ids.channels[0].num_ids, ids.channels[0].groups);
    EXPECT_EQ(ids.channels[1].num_ids, ids.channels[1].groups);
}

TEST(FabricTridAllocatorTest, TheLargestCountShrinksUntilTheyFit) {
    // 32 + 16 groups do not fit in 16 IDs: the largest halves, then the next, until they do.
    const std::vector<trids::ReceiverChannel> channels = {
        {.slots = 32, .credit_each_slot = true}, {.slots = 16, .credit_each_slot = true}};
    const trids::Assignment ids = trids::assign_transaction_ids(channels);
    expect_ids_hold(channels, ids);
    EXPECT_EQ(ids.used_ids, trids::noc_transaction_ids);
}

TEST(FabricTridAllocatorTest, EverySlotAllocationGetsIds) {
    // The receivers the slot allocator produces always fit in the NoC's IDs, credited per group or slot by slot.
    const sa::Limits limits;
    for (const auto& [name, base] : router_shapes()) {
        const std::size_t minimum = num_channels(base) * limits.min_slots;
        for (std::size_t n : {minimum, std::size_t{40}, std::size_t{81}, std::size_t{200}}) {
            sa::RouterShape shape = base;
            shape.buffer_bytes = n * shape.slot_bytes;
            const sa::Allocation a = sa::allocate_slots(shape);
            ASSERT_FALSE(a.error.has_value()) << name;
            for (const bool each_slot : {false, true}) {
                std::vector<trids::ReceiverChannel> receivers;
                for (const auto& c : a.channels) {
                    if (c.kind == sa::ChannelKind::Receiver) {
                        receivers.push_back({.slots = c.slots, .credit_each_slot = each_slot});
                    }
                }
                SCOPED_TRACE(name + ", " + std::to_string(n) + " slots" + (each_slot ? ", slot by slot" : ""));
                expect_ids_hold(receivers, trids::assign_transaction_ids(receivers));
            }
        }
    }
}

TEST(FabricTridAllocatorTest, MoreChannelsThanIdsIsAnError) {
    const std::vector<trids::ReceiverChannel> channels(trids::noc_transaction_ids + 1, {.slots = 2});
    EXPECT_TRUE(trids::assign_transaction_ids(channels).error.has_value());
}

}  // namespace
}  // namespace tt::tt_fabric
