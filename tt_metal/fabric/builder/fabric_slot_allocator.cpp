// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_slot_allocator.hpp"

#include <algorithm>
#include <numeric>
#include <string>

#include <fmt/format.h>

namespace tt::tt_fabric::slot_allocator {

std::size_t full_rate_depth(std::size_t vc, ChannelKind kind, bool is_2d) {
    if (vc >= NUM_VCS) {
        return 0;
    }
    // VC1 and VC2 are built on 2D fabrics only.
    const FullRateDepths& depths = (is_2d || vc != 0) ? full_rate_depths_2d[vc] : vc0_full_rate_depths_1d;
    switch (kind) {
        case ChannelKind::Worker: return depths.worker;
        case ChannelKind::Forwarding: return depths.forwarding;
        case ChannelKind::Receiver: return depths.receiver;
    }
    return 0;
}

LeftoverOrder default_leftover_order(bool is_2d) {
    if (is_2d) {
        return {{{0, ChannelKind::Receiver}}, {{0, ChannelKind::Forwarding}}, {{0, ChannelKind::Worker}}};
    }
    return {{{0, ChannelKind::Receiver}}, {{0, ChannelKind::Worker}, {0, ChannelKind::Forwarding}}};
}

std::size_t Allocation::slots(std::size_t vc, ChannelKind kind, std::size_t index) const {
    for (const auto& c : channels) {
        if (c.vc == vc && c.kind == kind && c.index == index) {
            return c.slots;
        }
    }
    return 0;
}

namespace {

// The power of 2 nearest to num/den (ties up), at least min_slots and at most max_slots.
std::size_t nearest_power_of_2(std::size_t num, std::size_t den, const Limits& limits) {
    std::size_t lo = 1;
    while (lo * 2 * den <= num) {
        lo *= 2;
    }
    const std::size_t hi = lo * 2;
    std::size_t p = (hi * den - num <= num - std::min(num, lo * den)) ? hi : lo;
    while (p > limits.max_slots) {
        p /= 2;
    }
    return std::max(p, limits.min_slots);
}

}  // namespace

Allocation allocate_slots(const RouterShape& shape, const LeftoverOrder& leftover_order, const Limits& limits) {
    Allocation a;
    if (shape.slot_bytes == 0) {
        a.error = "the slot size is 0";
        return a;
    }

    // Stage 0: the budget in slots, every channel at its minimum.
    const std::size_t n = shape.buffer_bytes / shape.slot_bytes;
    a.total_slots = n;
    std::vector<ChannelSlots> rx;
    std::vector<ChannelSlots> tx;  // channel order: per VC, workers, then forwarding
    for (std::size_t vc = 0; vc < NUM_VCS; ++vc) {
        const auto& v = shape.vcs[vc];
        for (std::size_t i = 0; i < v.receivers; ++i) {
            rx.push_back({vc, ChannelKind::Receiver, i, limits.min_slots, 0});
        }
        for (std::size_t i = 0; i < v.workers; ++i) {
            tx.push_back({vc, ChannelKind::Worker, i, limits.min_slots, 0});
        }
        for (std::size_t i = 0; i < v.forwarding; ++i) {
            tx.push_back({vc, ChannelKind::Forwarding, i, limits.min_slots, 0});
        }
    }
    const std::size_t num_channels = rx.size() + tx.size();
    if (num_channels * limits.min_slots > n) {
        a.error = fmt::format(
            "{} channels at {} slots each do not fit in {} slots of {} B",
            num_channels,
            limits.min_slots,
            n,
            shape.slot_bytes);
        return a;
    }
    const auto depth = [&](const ChannelSlots& c) { return full_rate_depth(c.vc, c.kind, shape.is_2d); };
    std::size_t all_depths = 0;
    for (const auto& c : rx) {
        all_depths += depth(c);
    }
    std::size_t sender_depths = 0;
    for (const auto& c : tx) {
        sender_depths += depth(c);
    }
    all_depths += sender_depths;
    a.all_depths = all_depths;
    a.sender_depths = sender_depths;

    // Stage 1: receivers at f = min(1, N / every channel's depth), nearest power of 2.
    const bool squeezed = n < all_depths;
    for (auto& c : rx) {
        c.slots = squeezed ? nearest_power_of_2(depth(c) * n, all_depths, limits)
                           : nearest_power_of_2(depth(c), 1, limits);
        c.stage = 1;
    }
    std::size_t rx_slots = 0;
    for (const auto& c : rx) {
        rx_slots += c.slots;
    }
    // A power of 2 rounded up can leave the senders less than their minimums: halve the largest receiver until they
    // fit.
    while (n - std::min(n, rx_slots) < tx.size() * limits.min_slots) {
        auto it =
            std::max_element(rx.begin(), rx.end(), [](const auto& x, const auto& y) { return x.slots < y.slots; });
        if (it == rx.end() || it->slots / 2 < limits.min_slots) {
            a.error = "the receivers' minimums leave the senders less than theirs";
            return a;
        }
        rx_slots -= it->slots / 2;
        it->slots /= 2;
    }

    // Stage 2: the senders at f' = min(1, B / the senders' depths). A VC's forwarding channels form one unit with one
    // count (the router keeps one downstream forwarding ring size per VC); each worker is a unit of its own.
    std::vector<std::vector<std::size_t>> units;  // indices into tx; a unit's channels share one count
    for (std::size_t i = 0; i < tx.size(); ++i) {
        auto same = units.end();
        if (tx[i].kind == ChannelKind::Forwarding) {
            same = std::find_if(units.begin(), units.end(), [&](const auto& u) {
                return tx[u.front()].kind == ChannelKind::Forwarding && tx[u.front()].vc == tx[i].vc;
            });
        }
        if (same == units.end()) {
            units.push_back({i});
        } else {
            same->push_back(i);
        }
    }
    const auto set_unit = [&](const std::vector<std::size_t>& u, std::size_t slots) {
        for (std::size_t i : u) {
            tx[i].slots = slots;
            tx[i].stage = 2;
        }
    };
    const std::size_t b = n - rx_slots;
    a.sender_slots = b;
    if (b >= sender_depths) {
        for (const auto& u : units) {
            set_unit(u, std::min(depth(tx[u.front()]), limits.max_slots));
        }
    } else {
        // Every channel of a unit gets the floor of its share; a unit grows a slot on each of its channels at a time.
        std::vector<std::size_t> floor_of(units.size());
        std::vector<std::size_t> rem_of(units.size());
        std::vector<std::size_t> count(units.size());
        std::size_t given = 0;
        for (std::size_t u = 0; u < units.size(); ++u) {
            const std::size_t d = depth(tx[units[u].front()]);
            floor_of[u] = d * b / sender_depths;
            rem_of[u] = d * b % sender_depths;
            count[u] = std::max(floor_of[u], limits.min_slots);
            given += count[u] * units[u].size();
        }
        // The largest remainders get what the floors leave, a whole unit at a time. Ties: lower VC first, forwarding
        // before the worker, lower index first (a fixed order, so every router of a fabric decides alike).
        std::vector<std::size_t> order(units.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
            if (rem_of[x] != rem_of[y]) {
                return rem_of[x] > rem_of[y];
            }
            const auto& cx = tx[units[x].front()];
            const auto& cy = tx[units[y].front()];
            if (cx.vc != cy.vc) {
                return cx.vc < cy.vc;
            }
            if (cx.kind != cy.kind) {
                return cx.kind == ChannelKind::Forwarding;
            }
            return cx.index < cy.index;
        });
        for (std::size_t u : order) {
            if (count[u] == floor_of[u] && floor_of[u] >= limits.min_slots && given + units[u].size() <= b) {
                count[u]++;
                given += units[u].size();
            }
        }
        // Minimums raised above the share: each step takes a slot from every channel of the unit most above its
        // minimum. What a step frees beyond the overshoot is left to stage 3.
        while (given > b) {
            std::size_t pick = units.size();
            for (std::size_t u = 0; u < units.size(); ++u) {
                if (count[u] > limits.min_slots &&
                    (pick == units.size() || count[u] - limits.min_slots > count[pick] - limits.min_slots)) {
                    pick = u;
                }
            }
            if (pick == units.size()) {
                a.error = "the senders' minimums do not fit";
                return a;
            }
            count[pick]--;
            given -= units[pick].size();
        }
        for (std::size_t u = 0; u < units.size(); ++u) {
            set_unit(units[u], count[u]);
        }
    }

    // Stage 3: what is left (the slots past every full-rate depth, or the rounding a squeeze's units could not take)
    // by the leftover order, in whole steps. A group's receivers each go to their next power of 2 (no deeper than
    // max_receiver_slots_from_leftovers), all or none; its senders take +1 a unit at a time (stage 2's units, so a
    // VC's forwarding channels keep one count), unit after unit in the group's order and round after round. A step
    // that does not fit whole, or a group with no channels, passes what is left on.
    std::size_t used = 0;
    for (const auto& c : rx) {
        used += c.slots;
    }
    for (const auto& c : tx) {
        used += c.slots;
    }
    std::size_t left = n - used;
    const auto in_group = [](const auto& group, const ChannelSlots& c) {
        return std::find(group.begin(), group.end(), ChannelKindOnVc{c.vc, c.kind}) != group.end();
    };
    for (const auto& group : leftover_order) {
        std::vector<ChannelSlots*> receivers;
        for (auto& c : rx) {
            if (in_group(group, c)) {
                receivers.push_back(&c);
            }
        }
        std::size_t step = 0;
        bool fits = !receivers.empty();
        for (const auto* c : receivers) {
            step += c->slots;  // a power of 2 doubles to the next one
            fits = fits && 2 * c->slots <= std::min(limits.max_slots, limits.max_receiver_slots_from_leftovers);
        }
        if (fits && step <= left) {
            for (auto* c : receivers) {
                c->slots *= 2;
                c->stage = 3;
            }
            left -= step;
        }
        std::vector<const std::vector<std::size_t>*> sender_units;  // in the group's order
        for (const auto& kind_on_vc : group) {
            for (const auto& u : units) {
                if (ChannelKindOnVc{tx[u.front()].vc, tx[u.front()].kind} == kind_on_vc) {
                    sender_units.push_back(&u);
                }
            }
        }
        for (bool grew = !sender_units.empty(); grew;) {
            grew = false;
            for (const auto* unit : sender_units) {
                const auto& u = *unit;
                if (u.size() > left || tx[u.front()].slots >= limits.max_slots) {
                    continue;
                }
                for (std::size_t i : u) {
                    tx[i].slots += 1;
                    tx[i].stage = 3;
                }
                left -= u.size();
                grew = true;
            }
        }
    }
    a.unused_slots = left;
    a.channels = std::move(rx);
    a.channels.insert(a.channels.end(), tx.begin(), tx.end());
    return a;
}

Allocation allocate_slots(const RouterShape& shape, const Limits& limits) {
    return allocate_slots(shape, default_leftover_order(shape.is_2d), limits);
}

}  // namespace tt::tt_fabric::slot_allocator
