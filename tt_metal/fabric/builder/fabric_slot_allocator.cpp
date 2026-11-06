// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_slot_allocator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tt::tt_fabric::slot_allocator {

const char* to_string(ChannelKind kind) {
    switch (kind) {
        case ChannelKind::Worker: return "worker";
        case ChannelKind::Forwarding: return "forwarding";
        case ChannelKind::Receiver: return "receiver";
    }
    return "?";
}

// Named, not anonymous: the host builds this file in a unity batch with its neighbours.
namespace allocator_detail {

std::string number(double value, int decimals) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    return text;
}

std::string ratio_text(const std::vector<unsigned>& ratio) {
    std::string text;
    for (unsigned r : ratio) {
        text += (text.empty() ? "" : ":") + std::to_string(r);
    }
    return text;
}

// "a:b[:c...]" of positive integers.
bool parse_ratio(const std::string& text, std::vector<unsigned>& out) {
    out.clear();
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t end = text.find(':', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string item = text.substr(start, end - start);
        char* parse_end = nullptr;
        const unsigned long value = std::strtoul(item.c_str(), &parse_end, 10);
        if (item.empty() || parse_end == nullptr || *parse_end != '\0' || value == 0) {
            return false;
        }
        out.push_back(static_cast<unsigned>(value));
        start = end + 1;
    }
    return !out.empty();
}

std::size_t floor_power_of_2(std::size_t n) {
    std::size_t p = 1;
    while (p * 2 <= n) {
        p *= 2;
    }
    return p;
}

// A VC0 receiver's count from its share.
std::size_t round_receiver(double share, ReceiverRound mode) {
    if (mode == ReceiverRound::Mult4) {
        return std::max<std::size_t>(4, 4 * static_cast<std::size_t>(std::lround(share / 4)));
    }
    return share < 1 ? 1 : std::size_t{1} << static_cast<unsigned>(std::lround(std::log2(share)));
}

}  // namespace allocator_detail

std::vector<unsigned> default_vc_ratio(const RouterShape& shape) {
    const bool vc0 = shape.vcs[0].present(), vc1 = shape.vcs[1].present(), vc2 = shape.vcs[2].present();
    if (vc0 && vc1 && vc2) {
        return {12, 4, 1};
    }
    if (vc0 && vc1 && !vc2) {
        return {3, 1};
    }
    if (vc0 && vc2 && !vc1) {
        return {12, 1};
    }
    std::vector<unsigned> equal;
    for (const auto& v : shape.vcs) {
        if (v.present()) {
            equal.push_back(1);
        }
    }
    return equal;
}

std::optional<std::string> parse_policy(const std::string& text, Policy& policy) {
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string item = text.substr(start, end - start);
        start = end + 1;
        if (item.empty()) {
            continue;
        }
        const std::size_t eq = item.find('=');
        if (eq == std::string::npos) {
            return "policy item '" + item + "': expected key=value";
        }
        const std::string key = item.substr(0, eq);
        const std::string value = item.substr(eq + 1);
        if (key == "vc_ratio") {
            if (!allocator_detail::parse_ratio(value, policy.vc_ratio)) {
                return "policy item '" + item + "': expected positive integers a:b[:c], one per VC built";
            }
            continue;
        }
        if (key == "receiver") {
            std::vector<unsigned> r;
            if (!allocator_detail::parse_ratio(value, r) || r.size() != 2) {
                return "policy item '" + item + "': expected receiver=num:den (of the VC's heaviest sender)";
            }
            policy.receiver_num = r[0];
            policy.receiver_den = r[1];
            continue;
        }
        if (key == "vc0_receiver_floor") {
            std::vector<unsigned> r;
            if (value == "off") {
                policy.vc0_receiver_floor_num = 0;
                policy.vc0_receiver_floor_den = 1;
            } else if (allocator_detail::parse_ratio(value, r) && r.size() == 2) {
                policy.vc0_receiver_floor_num = r[0];
                policy.vc0_receiver_floor_den = r[1];
            } else {
                return "policy item '" + item + "': expected vc0_receiver_floor=num:den (of N) or off";
            }
            continue;
        }
        if (key == "receiver_round") {
            if (value == "pow2") {
                policy.receiver_round = ReceiverRound::Pow2;
            } else if (value == "mult4") {
                policy.receiver_round = ReceiverRound::Mult4;
            } else {
                return "policy item '" + item + "': expected receiver_round=pow2 or mult4";
            }
            continue;
        }
        char* parse_end = nullptr;
        const unsigned long number = std::strtoul(value.c_str(), &parse_end, 10);
        if (value.empty() || parse_end == nullptr || *parse_end != '\0') {
            return "policy item '" + item + "': expected an integer";
        }
        if (key == "worker") {
            policy.worker_weight = static_cast<unsigned>(number);
        } else if (key == "forwarding") {
            policy.forwarding_weight = static_cast<unsigned>(number);
        } else if (key == "long_dim_bump") {
            policy.long_dim_bump = static_cast<unsigned>(number);
        } else if (key == "worker_cap_slots") {
            policy.worker_cap_slots = number;
        } else if (key == "min") {
            policy.min_slots = number;
        } else {
            return "policy key '" + key +
                   "': expected vc_ratio, worker, forwarding, long_dim_bump, receiver, receiver_round, "
                   "vc0_receiver_floor, worker_cap_slots or min";
        }
    }
    if (policy.worker_weight == 0 || policy.forwarding_weight == 0 || policy.long_dim_bump == 0 ||
        policy.receiver_den == 0 || policy.vc0_receiver_floor_den == 0 || policy.min_slots == 0) {
        return "policy: weights, long_dim_bump, receiver and min must be > 0";
    }
    return std::nullopt;
}

Allocation allocate_slots(const RouterShape& shape, const Policy& policy) {
    using allocator_detail::number;
    Allocation result;
    if (shape.slot_bytes == 0) {
        result.error = "slot size is 0";
        return result;
    }
    const std::size_t total = shape.budget_bytes / shape.slot_bytes;
    result.total_slots = total;
    const bool long_dims = shape.mesh_dims[0] > 2 || shape.mesh_dims[1] > 2;
    result.notes.push_back(
        "N = " + std::to_string(total) + " slots of " + std::to_string(shape.slot_bytes) + " B (" +
        std::to_string(shape.budget_bytes) + " B of channel buffers)");
    result.notes.push_back(
        "mesh " + std::to_string(shape.mesh_dims[0]) + "x" + std::to_string(shape.mesh_dims[1]) +
        (long_dims ? ": a dim longer than 2 chips, forwarding weight x" + std::to_string(policy.long_dim_bump)
                   : ": no dim longer than 2 chips, forwarding carries only turns"));

    // The VCs built and each one's share of N: the ratio, rounded by largest remainder (ties to the lower VC).
    std::vector<std::size_t> vcs;
    for (std::size_t vc = 0; vc < NUM_VCS; vc++) {
        if (shape.vcs[vc].present()) {
            vcs.push_back(vc);
        }
    }
    const std::vector<unsigned> ratio = policy.vc_ratio.empty() ? default_vc_ratio(shape) : policy.vc_ratio;
    if (vcs.empty() || ratio.size() != vcs.size()) {
        result.error = "vc_ratio " + allocator_detail::ratio_text(ratio) + " needs one entry per VC built (" +
                       std::to_string(vcs.size()) + ")";
        return result;
    }
    unsigned ratio_total = 0;
    for (unsigned r : ratio) {
        ratio_total += r;
    }
    std::array<std::size_t, NUM_VCS> share{};
    std::vector<std::pair<std::size_t, std::size_t>> remainders;  // (remainder, position)
    std::size_t shared = 0;
    for (std::size_t i = 0; i < vcs.size(); i++) {
        share[vcs[i]] = total * ratio[i] / ratio_total;
        shared += share[vcs[i]];
        remainders.emplace_back(total * ratio[i] % ratio_total, i);
    }
    std::stable_sort(
        remainders.begin(), remainders.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t i = 0; shared < total && i < remainders.size(); i++, shared++) {
        share[vcs[remainders[i].second]]++;
    }
    std::string shares = "VC shares (ratio " + allocator_detail::ratio_text(ratio) +
                         (policy.vc_ratio.empty() ? ", the default" : ", vc_ratio") + ", largest remainder):";
    for (std::size_t vc : vcs) {
        shares += " VC" + std::to_string(vc) + " " + std::to_string(share[vc]);
    }

    // Every channel's minimum comes first: a VC whose share is below its channels' minimums is raised to them, and
    // the slots come from the VC with the most above its own minimums (one at a time).
    std::array<std::size_t, NUM_VCS> minimum{};
    std::size_t minimum_total = 0;
    for (std::size_t vc : vcs) {
        const RouterShape::Vc& v = shape.vcs[vc];
        minimum[vc] = policy.min_slots * (v.workers + v.forwarding + v.receivers);
        minimum_total += minimum[vc];
    }
    if (minimum_total > total) {
        result.error = "every channel's minimum of " + std::to_string(policy.min_slots) + " needs " +
                       std::to_string(minimum_total) + " slots, N is " + std::to_string(total);
        return result;
    }
    bool raised = false;
    for (std::size_t vc : vcs) {
        while (share[vc] < minimum[vc]) {
            std::size_t donor = vc;
            for (std::size_t other : vcs) {
                if (share[other] > minimum[other] &&
                    (donor == vc || share[other] - minimum[other] > share[donor] - minimum[donor])) {
                    donor = other;
                }
            }
            share[donor]--;
            share[vc]++;
            raised = true;
        }
    }
    if (raised) {
        shares += "; raised to every channel's minimum of " + std::to_string(policy.min_slots) + ":";
        for (std::size_t vc : vcs) {
            shares += " VC" + std::to_string(vc) + " " + std::to_string(share[vc]);
        }
    }
    result.notes.push_back(shares);

    // Per VC: (a) receivers take their share first, (b) senders split the rest by weight, rounding down, and any
    // channel below the minimum is raised, (c) the VC0 worker's cap. Weights are kept as integers scaled by
    // receiver_den, so the receiver's num/den of the heaviest sender is whole too.
    const std::size_t den = policy.receiver_den;
    std::size_t used = 0;
    std::size_t vc0_worker_cap = policy.max_slots;
    for (std::size_t vc : vcs) {
        const RouterShape::Vc& v = shape.vcs[vc];
        const std::size_t s = share[vc];
        const std::size_t w_worker = policy.worker_weight * den;
        const std::size_t w_forwarding = policy.forwarding_weight * (long_dims ? policy.long_dim_bump : 1u) * den;
        std::size_t heaviest = 0;
        if (v.workers > 0) {
            heaviest = std::max(heaviest, w_worker);
        }
        if (v.forwarding > 0) {
            heaviest = std::max(heaviest, w_forwarding);
        }
        if (heaviest == 0) {
            heaviest = w_forwarding;
        }
        const std::size_t w_receiver = heaviest / den * policy.receiver_num;
        const std::size_t weight_total = v.workers * w_worker + v.forwarding * w_forwarding + v.receivers * w_receiver;
        const double scale = static_cast<double>(den);
        const std::string vc_name = "VC" + std::to_string(vc);
        auto& counts = result.counts[vc];

        // (a) Receivers.
        std::size_t receiver = 0;
        if (v.receivers > 0) {
            const double x = static_cast<double>(s * w_receiver) / static_cast<double>(weight_total);
            std::string why = "weight " + number(w_receiver / scale, 2) + " (" + std::to_string(policy.receiver_num) +
                              "/" + std::to_string(policy.receiver_den) + " of the heaviest sender " +
                              number(heaviest / scale, 2) + ") of " + number(weight_total / scale, 2) + ": share " +
                              number(x, 1) + " of " + vc_name + "'s " + std::to_string(s);
            if (vc == 0) {
                // The 2D floor: the share is at least floor_num / floor_den of N, as far as VC0's senders keep their
                // minimum; it never lowers the weighted count.
                const std::size_t unfloored = allocator_detail::round_receiver(x, policy.receiver_round);
                const double floor_share =
                    static_cast<double>(total * policy.vc0_receiver_floor_num) / policy.vc0_receiver_floor_den;
                const bool floored = shape.two_d && policy.vc0_receiver_floor_num > 0 && floor_share > x;
                if (floored) {
                    why += ", raised to the 2D floor " + std::to_string(policy.vc0_receiver_floor_num) + "/" +
                           std::to_string(policy.vc0_receiver_floor_den) + " of N = " + number(floor_share, 1);
                }
                receiver = allocator_detail::round_receiver(floored ? floor_share : x, policy.receiver_round);
                why += policy.receiver_round == ReceiverRound::Pow2 ? " -> nearest power of 2"
                                                                    : " -> nearest multiple of 4";
                const std::size_t senders_minimum = policy.min_slots * (v.workers + v.forwarding);
                const std::size_t room = s > senders_minimum ? (s - senders_minimum) / v.receivers : 0;
                if (floored && receiver > unfloored && receiver > room) {
                    // Keep the rounding's kind: the largest power of 2 (multiple of 4) that fits.
                    const std::size_t fits = policy.receiver_round == ReceiverRound::Pow2
                                                 ? (room == 0 ? 0 : allocator_detail::floor_power_of_2(room))
                                                 : room / 4 * 4;
                    why += " " + std::to_string(receiver) + "; FLOOR MISSED: " + std::to_string(room) +
                           " fit after the senders' minimum of " + std::to_string(policy.min_slots) +
                           ", rounded down to " +
                           (policy.receiver_round == ReceiverRound::Pow2 ? "a power of 2: " : "a multiple of 4: ") +
                           std::to_string(fits);
                    receiver = std::max(unfloored, fits);
                    if (receiver > fits) {
                        why += "; keeps the weighted " + std::to_string(receiver);
                    }
                }
            } else {
                receiver = s * w_receiver / weight_total;
                why += " -> rounded down";
            }
            if (receiver < policy.min_slots) {
                receiver = policy.min_slots;
                why += ", raised to the minimum " + std::to_string(policy.min_slots);
            }
            if (receiver > policy.max_slots) {
                receiver = policy.max_slots;
                why += ", held at " + std::to_string(policy.max_slots);
            }
            counts[static_cast<std::size_t>(ChannelKind::Receiver)] = {v.receivers, receiver, why};
        }

        // (b) Senders.
        const std::size_t receivers_total = v.receivers * receiver;
        const std::size_t left = s > receivers_total ? s - receivers_total : 0;
        const std::size_t sender_weight = v.workers * w_worker + v.forwarding * w_forwarding;
        const auto sender = [&](ChannelKind kind, std::size_t channels, std::size_t weight) {
            if (channels == 0) {
                return;
            }
            std::size_t slots = left * weight / sender_weight;
            std::string why = "weight " + number(weight / scale, 2) + " of the senders' " +
                              number(sender_weight / scale, 2) + ": " + std::to_string(left) + " left -> " +
                              std::to_string(slots) + " (rounded down)";
            if (slots < policy.min_slots) {
                slots = policy.min_slots;
                why += ", raised to the minimum " + std::to_string(policy.min_slots);
            }
            if (slots > policy.max_slots) {
                slots = policy.max_slots;
                why += ", held at " + std::to_string(policy.max_slots);
            }
            counts[static_cast<std::size_t>(kind)] = {channels, slots, why};
        };
        sender(ChannelKind::Worker, v.workers, w_worker);
        sender(ChannelKind::Forwarding, v.forwarding, w_forwarding);
        // The minimum comes first: if raising a channel to it took the VC past its share, the slots come back from the
        // sender kind with the most above the minimum (one slot per channel at a time).
        std::size_t vc_used = 0;
        for (const Count& c : counts) {
            vc_used += c.channels * c.slots;
        }
        const std::size_t worker_before = counts[static_cast<std::size_t>(ChannelKind::Worker)].slots;
        const std::size_t forwarding_before = counts[static_cast<std::size_t>(ChannelKind::Forwarding)].slots;
        while (vc_used > s) {
            Count* donor = nullptr;
            for (ChannelKind kind : {ChannelKind::Worker, ChannelKind::Forwarding}) {
                Count& c = counts[static_cast<std::size_t>(kind)];
                if (c.channels > 0 && c.slots > policy.min_slots && (donor == nullptr || c.slots > donor->slots)) {
                    donor = &c;
                }
            }
            if (donor == nullptr) {
                break;
            }
            donor->slots--;
            vc_used -= donor->channels;
        }
        for (auto [kind, before] :
             {std::pair{ChannelKind::Worker, worker_before}, std::pair{ChannelKind::Forwarding, forwarding_before}}) {
            Count& c = counts[static_cast<std::size_t>(kind)];
            if (c.slots < before) {
                c.reason += ", -" + std::to_string(before - c.slots) + " to the senders raised to the minimum";
            }
        }

        // (c) The VC0 worker's cap.
        Count& worker = counts[static_cast<std::size_t>(ChannelKind::Worker)];
        if (vc == 0 && policy.worker_cap_slots > 0) {
            vc0_worker_cap = std::min(policy.max_slots, policy.worker_cap_slots);
            if (worker.slots > vc0_worker_cap) {
                worker.reason += ", capped at " + std::to_string(vc0_worker_cap) + " (worker_cap_slots)";
                worker.slots = vc0_worker_cap;
            }
        }
        for (const Count& c : counts) {
            used += c.channels * c.slots;
        }
    }
    if (used > total) {
        result.error = "the counts need " + std::to_string(used) + " slots, N is " + std::to_string(total);
        return result;
    }

    // (d) The leftover to the VC0 worker, up to its cap; past it, to VC0 forwarding (one slot per channel at a time).
    std::size_t leftover = total - used;
    Count& worker = result.counts[0][static_cast<std::size_t>(ChannelKind::Worker)];
    Count& forwarding = result.counts[0][static_cast<std::size_t>(ChannelKind::Forwarding)];
    if (worker.channels > 0 && leftover > 0) {
        const std::size_t room = worker.slots < vc0_worker_cap ? vc0_worker_cap - worker.slots : 0;
        const std::size_t add = std::min(leftover / worker.channels, room);
        if (add > 0) {
            worker.slots += add;
            leftover -= add * worker.channels;
            worker.reason += "; + " + std::to_string(add) + " leftover";
        }
    }
    if (forwarding.channels > 0 && leftover >= forwarding.channels) {
        const std::size_t room = forwarding.slots < policy.max_slots ? policy.max_slots - forwarding.slots : 0;
        const std::size_t add = std::min(leftover / forwarding.channels, room);
        if (add > 0) {
            forwarding.slots += add;
            leftover -= add * forwarding.channels;
            forwarding.reason += "; + " + std::to_string(add) + " leftover past the worker's cap";
        }
    }
    result.unused_slots = leftover;
    return result;
}

}  // namespace tt::tt_fabric::slot_allocator
