// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <cstdio>
#include <string>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>

#include "impl/streaming_profiler/spsc_packet.h"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"
#include "impl/streaming_profiler/streaming_profiler_placement_map.hpp"

namespace tt::tt_metal::streaming_profiler {

void SyncEngine::on_attach(const CaptureContext& ctx) {
    ctx_ = ctx;
    local_.clear();
    links_.reset(ctx_);
    series_.reset();
    to_root_gen_ = ~0ull;
    // A chip the capture cannot place -- no eth tracker, or no link path to the root -- is finished at once, so no
    // consumer waits for it.
    std::vector<bool> reach(ctx.devices.size(), false);
    const uint32_t root = ctx.root_dev;
    if (root < ctx.devices.size() && ctx.devices[root].has_eth_tracker) {
        reach[root] = true;
        for (bool progress = true; progress;) {
            progress = false;
            for (const CaptureContext::Link& L : ctx.links) {
                if (L.dev_a < reach.size() && L.dev_b < reach.size() && reach[L.dev_a] != reach[L.dev_b]) {
                    reach[L.dev_a] = reach[L.dev_b] = true;
                    progress = true;
                }
            }
        }
    }
    for (size_t dev = 0; dev < ctx.devices.size(); dev++) {
        const CaptureContext::Device& d = ctx.devices[dev];
        map_.clear(d.chip_id);
        if (!reach[dev] || !d.has_eth_tracker) {
            map_.finish(d.chip_id);
        }
    }
}

void SyncEngine::on_clock(const ClockSample& s) {
    if (s.kind != PP_CLOCK_LOCAL_REFCLK) {
        links_.on_stamp(s);
        return;
    }
    LocalClockModel& l = local_[s.dev];
    if (s.role == PP_CLOCK_LOCAL_RAW) {
        l.add_raw();
        return;
    }
    l.add_point(s.value, s.ts, s.round & 0xFFu, s.round >> 8, s.role == PP_CLOCK_LOCAL_CLOSE);
    if (publish_dev(s.dev)) {
        service().wake_consumers();
    }
}

void LinkSolver::reset(const CaptureContext& ctx) {
    ctx_ = &ctx;
    rounds_.assign(ctx.links.size(), LinkRounds{});
    side_of_.clear();
    for (size_t li = 0; li < ctx.links.size(); li++) {
        const CaptureContext::Link& L = ctx.links[li];
        side_of_[{L.dev_a, L.core_a}] = {li, true};
        side_of_[{L.dev_b, L.core_b}] = {li, false};
    }
    solved_.assign(ctx.links.size(), LinkSolution{});
    gen_ = 0;
    dropped_ = 0;
}

void LinkSolver::on_stamp(const ClockSample& s) {
    const auto side = side_of_.find({s.dev, s.lane / profiler::kSpscNRiscDecode});
    if (s.kind != PP_CLOCK_LINK_PTP || side == side_of_.end()) {
        dropped_++;
        return;
    }
    const auto [li, sender] = side->second;
    Stamp Round::* slot = nullptr;
    if (sender) {
        slot = s.role == PP_CLOCK_ROLE_T0 ? &Round::t0 : s.role == PP_CLOCK_ROLE_T2 ? &Round::t2 : nullptr;
    } else {
        slot = s.role == PP_CLOCK_ROLE_T1 ? &Round::t1 : s.role == PP_CLOCK_ROLE_T1B ? &Round::t1b : nullptr;
    }
    if (slot == nullptr) {
        dropped_++;
        return;
    }
    LinkRounds& lr = rounds_[li];
    Round& r = lr.pending[s.round];
    r.id = s.round;
    r.*slot = Stamp{.units = s.value, .have = true};
    if (r.complete()) {
        lr.rounds.push_back(r);
        lr.pending.erase(s.round);
        try_solve_links(/*final=*/false);
    }
    while (lr.pending.size() > kPendingMax) {
        lr.pending.erase(lr.pending.begin());
    }
}

void LinkSolver::solve_final() { try_solve_links(/*final=*/true); }

void LinkSolver::try_solve_links(bool final) {
    for (size_t li = 0; li < ctx_->links.size(); li++) {
        LinkSolution& out = solved_[li];
        const CaptureContext::Link& L = ctx_->links[li];
        const std::vector<Round>& rounds = rounds_[li].rounds;
        const size_t n = rounds.size();
        if (final) {
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link {} (dev {} eth({},{}) core {}) -> (dev {} eth({},{}) core {}): "
                "{} rounds complete, {} pending",
                li,
                L.dev_a,
                L.eth_a.x,
                L.eth_a.y,
                L.core_a,
                L.dev_b,
                L.eth_b.x,
                L.eth_b.y,
                L.core_b,
                rounds.size(),
                rounds_[li].pending.size());
        }
        if (n == 0) {
            continue;
        }
        const auto pos = [](const Round& r) { return mid_a_refclk(r); };
        const double newest = pos(rounds[n - 1]);
        // Live: the first solution once kFirstSolveTicks of rounds are in, so held records are released, then one
        // every half window. Final: whatever the window holds, if it is enough for a fit at all.
        if (!final && (out.ok ? newest < out.solved_at_refclk + kLinkWindowTicks / 2
                              : newest - pos(rounds[0]) < kFirstSolveTicks)) {
            continue;
        }
        size_t begin = n;
        while (begin > 0 && pos(rounds[begin - 1]) > newest - kLinkWindowTicks) {
            begin--;
        }
        const size_t w = n - begin;
        if (w < kMinSolveRounds) {
            continue;
        }
        out.solved_at_refclk = newest;
        std::vector<RoundPoint> pts;
        pts.reserve(w);
        const double path_med = path_median(rounds, begin, n);
        out.path_dropped = 0;
        for (size_t i = begin; i < n; i++) {
            const Round& r = rounds[i];
            if (std::abs(path_ns(r) - path_med) > kPathDevNs) {
                out.path_dropped++;
                continue;
            }
            const double mid = mid_a_refclk(r);
            pts.push_back(RoundPoint{mid, mid_b_refclk(r) - mid});
        }
        if (solve_link(L, std::move(pts), out)) {
            out.rounds = w;
            gen_++;
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} -> chip {}: solved at round {} over {} ({} kept): offset "
                "{:.1f} ns, rate {:.3f} ppm, residual {:.1f} ns{}",
                L.chip_a,
                L.chip_b,
                n,
                w,
                out.kept,
                out.offset_refclk * kNsPerRefclk,
                out.rate * 1e6,
                out.residual_rms_ns,
                out.path_dropped != 0 ? fmt::format(", {} rounds off the stamp path band", out.path_dropped) : "");
        }
    }
}

double LinkSolver::mid_a_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(r.t0.units) + static_cast<double>(r.t2.units)) * kRefclkPerStampUnit;
}

double LinkSolver::mid_b_refclk(const Round& r) {
    return 0.5 * (static_cast<double>(r.t1.units) + static_cast<double>(r.t1b.units)) * kRefclkPerStampUnit;
}

double LinkSolver::path_median(const std::vector<Round>& rounds, size_t begin, size_t n) {
    if (n <= begin) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> v;
    v.reserve(n - begin);
    for (size_t i = begin; i < n; i++) {
        v.push_back(path_ns(rounds[i]));
    }
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// Compose each device's refclk onto the root's along the solved-link tree, so a chip with no DIRECT link to the
// root still lands on the fleet timeline through its neighbours. A link solves receiver = sender*(1+rate) +
// (offset - rate*mid), an affine in the sender's refclk, and an affine's inverse and composition are affine, so
// each reachable device carries one { scale, shift } with root_refclk = scale * dev_refclk + shift. A breadth
// relaxation over the links (few devices, so O(links^2) is nothing) fills them from the root outward; a device
// no path reaches keeps its own anchor and the local term alone. `used`, when given, marks the links the tree
// took; the others close loops and their disagreement with the tree is path asymmetry (see log_summary).
std::vector<LinkSolver::LinkSolution> LinkSolver::pair_solutions(std::vector<std::vector<size_t>>* members) const {
    std::vector<LinkSolution> out;
    std::vector<std::vector<size_t>> groups;
    for (size_t li = 0; li < solved_.size(); li++) {
        if (!solved_[li].ok) {
            continue;
        }
        size_t g = 0;
        while (g < groups.size() && (solved_[groups[g][0]].dev_snd != solved_[li].dev_snd ||
                                     solved_[groups[g][0]].dev_rcv != solved_[li].dev_rcv)) {
            g++;
        }
        if (g == groups.size()) {
            groups.emplace_back();
        }
        groups[g].push_back(li);
    }
    for (const std::vector<size_t>& g : groups) {
        LinkSolution c = solved_[g[0]];
        if (g.size() > 1) {
            const auto weight = [&](size_t li) {
                const double p = std::max(solved_[li].precision_ns, 1e-3);
                return 1.0 / (p * p);
            };
            double wsum = 0, mid = 0;
            for (size_t li : g) {
                wsum += weight(li);
                mid += weight(li) * solved_[li].mid_refclk;
            }
            mid /= wsum;
            double rate = 0, offset = 0, rr = 0;
            c.rounds = c.kept = c.path_dropped = 0;
            c.solved_at_refclk = 0.0;
            for (size_t li : g) {
                const LinkSolution& s = solved_[li];
                const double w = weight(li) / wsum;
                rate += w * s.rate;
                offset += w * (s.offset_refclk + s.rate * (mid - s.mid_refclk));
                rr += w * s.residual_rms_ns * s.residual_rms_ns;
                c.rounds += s.rounds;
                c.kept += s.kept;
                c.path_dropped += s.path_dropped;
                c.solved_at_refclk = std::max(c.solved_at_refclk, s.solved_at_refclk);
            }
            c.mid_refclk = mid;
            c.rate = rate;
            c.offset_refclk = offset;
            c.residual_rms_ns = std::sqrt(rr);
            c.precision_ns = 1.0 / std::sqrt(wsum);
        }
        out.push_back(c);
    }
    if (members != nullptr) {
        *members = std::move(groups);
    }
    return out;
}

size_t LinkSolver::pair_size(size_t li) const {
    size_t n = 0;
    for (const LinkSolution& s : solved_) {
        n += s.ok && s.dev_snd == solved_[li].dev_snd && s.dev_rcv == solved_[li].dev_rcv;
    }
    return n;
}

std::map<uint32_t, RootXf> LinkSolver::root_transforms(uint32_t root, std::vector<bool>* used) const {
    std::map<uint32_t, RootXf> to_root;
    to_root[root] = RootXf{1.0, 0.0, true};
    if (used != nullptr) {
        used->assign(solved_.size(), false);
    }
    std::vector<std::vector<size_t>> members;
    const std::vector<LinkSolution> pairs = pair_solutions(&members);
    std::vector<bool> taken(pairs.size(), false);
    for (bool progress = true; progress;) {
        progress = false;
        for (size_t pi = 0; pi < pairs.size(); pi++) {
            const LinkSolution& s = pairs[pi];
            if (taken[pi]) {
                continue;
            }
            const double m = 1.0 + s.rate;  // receiver = m * sender + o
            const double o = s.offset_refclk - s.rate * s.mid_refclk;
            const auto rs = to_root.find(s.dev_rcv);
            const auto ss = to_root.find(s.dev_snd);
            const bool r_ok = rs != to_root.end() && rs->second.ok;
            const bool s_ok = ss != to_root.end() && ss->second.ok;
            if (s_ok && !r_ok) {
                // root = A_s * sender + B_s, and sender = (receiver - o) / m.
                const RootXf& S = ss->second;
                to_root[s.dev_rcv] = RootXf{S.scale / m, S.shift - S.scale * o / m, true};
                progress = true;
            } else if (r_ok && !s_ok) {
                // root = A_r * receiver + B_r, and receiver = m * sender + o.
                const RootXf& R = rs->second;
                to_root[s.dev_snd] = RootXf{R.scale * m, R.scale * o + R.shift, true};
                progress = true;
            } else {
                continue;
            }
            taken[pi] = true;
            if (used != nullptr && members[pi].size() == 1) {
                (*used)[members[pi][0]] = true;
            }
        }
    }
    return to_root;
}

SeriesPublisher::Fresh SeriesPublisher::fresh_nodes(
    const Series& s, const LocalClockModel& fit, const RootXf& xf) const {
    const std::vector<LocalClockModel::Run>& runs = fit.runs;
    // A node places one instant of a run on the root: its eth wall tick (the key every record of the chip is looked
    // up by, worker lanes through their tile offset) and the root's refclk at that instant, via the chip's refclk
    // and the solved links. Along a run the map is exactly linear.
    const auto node_at = [&](const LocalClockModel::Run& run, double r) {
        const double T = run.wall_of_refclk(r);
        const double root = xf.scale * r + xf.shift;
        const double tangent = xf.scale / run.slope();
        if (!(tangent > 0.0 && tangent < 1.0)) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync: a node at refclk {:.0f} is not a rate; its segment [{:.0f}, {:.0f}] "
                "has "
                "{} samples, k8 {:.0f}; root scale {:.9f}",
                r,
                run.r_first,
                run.r_last,
                run.n,
                run.k8,
                xf.scale);
        }
        return Node{T, root, r, tangent};
    };
    // Nodes only where the map bends: at the first sample, at each run boundary (the transition, placed where the
    // two exact lines meet, or two nodes a stride apart bridging the intercept step when the slopes agree), and the
    // open run's frontier. Nothing is placed on a run whose line has not settled (a young run's line would freeze a
    // misplaced node).
    Fresh out;
    out.knots_after = s.knots;
    double last_r = s.last_r;
    if (s.nodes.empty() && last_r < 0.0) {
        if (!runs.front().settled() || runs.front().slope() <= 0.0) {
            return out;
        }
        out.knots.push_back(node_at(runs.front(), runs.front().r_first));
        last_r = runs.front().r_first;
    }
    for (size_t i = s.knots; i + 1 < runs.size(); i++) {
        const LocalClockModel::Run& a = runs[i];
        const LocalClockModel::Run& b = runs[i + 1];
        if (i + 2 == runs.size() && !b.settled()) {
            break;
        }
        // Two lines with different slopes meet at the transition; two with the same slope (a run cut by length, or a
        // spurious split) are bridged by a node on each side of the seam. Past the knot the records lie on b, so the
        // knot leaves on b's tangent. The a-side node of a bridge is dropped when the frontier already passed it: a's
        // newest sample was handed to b after the tangent was frozen on it, and the frozen tangent's end is that node.
        if (const auto r_x = LocalClockModel::knot(a, b)) {
            if (*r_x < s.cover_r) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] d2d sync: runs meet at refclk {:.0f}, {:.1f} us behind the frozen frontier "
                    "{:.0f}; records between them were placed on the earlier run",
                    *r_x,
                    (s.cover_r - *r_x) / 50.0,
                    s.cover_r);
            }
            Node k = node_at(a, *r_x);
            k.tangent = node_at(b, *r_x).tangent;
            out.knots.push_back(k);
            last_r = *r_x;
        } else {
            if (a.r_last >= s.cover_r) {
                out.knots.push_back(node_at(a, a.r_last));
            }
            out.knots.push_back(node_at(b, b.r_first));
            last_r = b.r_first;
        }
        out.knots_after = i + 1;
    }
    // The open segment's line reaches to its newest point, which the pusher places behind its newest sample by more
    // than the time it takes to confirm a step, so no node freezes past a transition. A closed segment gets no
    // frontier: its close is the last sample still within the step threshold of its line, a few microseconds past
    // the crossing, and the knot with the next segment is what ends it.
    const LocalClockModel::Run& open = runs.back();
    if (!open.closed && open.settled() && open.slope() > 0.0 && open.r_last > last_r &&
        out.knots_after + 1 == runs.size()) {
        out.frontier = node_at(open, open.r_last);
    }
    return out;
}

void SeriesPublisher::push_node(Series& s, uint32_t chip, const Node& n) {
    const int64_t wall = static_cast<int64_t>(std::llround(n.H));
    if (!s.nodes.empty() && wall <= static_cast<int64_t>(std::llround(s.nodes.back().H))) {
        return;  // within the tick of the last node: the placement cannot differ measurably there
    }
    s.nodes.push_back(n);
    s.cover_H = n.H;
    s.cover_r = n.r;
    s.last_r = std::max(s.last_r, n.r);
    map_.append(chip, SyncNode{.at = wall, .value = n.root, .tangent = n.tangent});
}

// Frozen nodes never move (consumers have placed records against them), so a publish can only add beyond them, at
// the newest estimate's values; a join carries whatever the estimate moved by since the tangent was frozen (kFreezeNs
// at most on a frontier, a few ns at a knot). Shifting fresh nodes to meet the frozen tail and fading that shift over
// a quarter second is worse: every discrepancy at a join becomes a level the map carries for 250 ms, 30-60 ns during
// DVFS dithering at 1 ms.
void SeriesPublisher::freeze_append(Series& s, uint32_t chip, const Node& n) {
    const double frontier_H = std::max(s.nodes.empty() ? -1.0 : s.nodes.back().H, s.cover_H);
    if (n.H >= frontier_H && n.H < frontier_H + 1.0) {
        return;  // the series' end re-derived, or a knot within the tick of it: the same node
    }
    if (n.H < frontier_H) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync: placement node at wall {:.0f} lies {:.0f} wall ticks behind the frozen "
            "series' end; the frozen node stands",
            n.H,
            frontier_H - n.H);
        s.dropped++;
        return;
    }
    // The tangent's confirmed stretch becomes a node first, so nothing placed on it changes.
    if (!s.nodes.empty() && s.cover_H > s.nodes.back().H + 0.5) {
        const Node& last = s.nodes.back();
        push_node(s, chip, Node{s.cover_H, last.root + last.tangent * (s.cover_H - last.H), s.cover_r, last.tangent});
    }
    if (!std::isfinite(n.H) || !std::isfinite(n.root) || !std::isfinite(n.tangent) ||
        !(n.tangent > 0.0 && n.tangent < 1.0)) {
        s.dropped++;
        return;
    }
    push_node(s, chip, n);
}

bool SeriesPublisher::advance(Series& s, uint32_t chip, Fresh fresh) {
    const double cover_before = s.cover_H;
    for (const Node& k : fresh.knots) {
        freeze_append(s, chip, k);
        s.last_r = std::max(s.last_r, k.r);
    }
    s.knots = fresh.knots_after;
    if (fresh.frontier) {
        const Node& f = *fresh.frontier;
        const Node* last = s.nodes.empty() ? nullptr : &s.nodes.back();
        const double freeze_ticks = kFreezeNs * LocalClockModel::kRefclkHz * 1e-9;
        if (last != nullptr && f.H > s.cover_H &&
            std::abs(f.root - (last->root + last->tangent * (f.H - last->H))) <= freeze_ticks) {
            s.cover_H = f.H;
            s.cover_r = f.r;
            s.last_r = std::max(s.last_r, f.r);
            s.extended++;
            map_.extend(chip, static_cast<int64_t>(std::llround(f.H)));
        } else {
            freeze_append(s, chip, f);
        }
    }
    return s.cover_H > cover_before;
}

bool SeriesPublisher::publish(uint32_t dev, uint32_t chip, const LocalClockModel& fit, const RootXf& xf) {
    Series& s = series_[dev];
    return advance(s, chip, fresh_nodes(s, fit, xf));
}

bool SyncEngine::publish_dev(uint32_t dev) {
    const auto st = local_.find(dev);
    if (st == local_.end() || st->second.runs.empty() || dev >= ctx_.devices.size()) {
        return false;
    }
    // Nothing is placed before the host series exists: a record converted then would land nowhere.
    if (map_.host_published() == 0) {
        return false;
    }
    if (to_root_gen_ != links_.generation()) {
        to_root_ = links_.root_transforms(root_dev(), nullptr);
        to_root_gen_ = links_.generation();
    }
    // The series starts only once the chip is on the root's tree (the root is there from the start): its first node
    // fixes the placement every later node joins.
    const auto xf = to_root_.find(dev);
    if (xf == to_root_.end() || !xf->second.ok) {
        return false;
    }
    return series_.publish(dev, ctx_.devices[dev].chip_id, st->second, xf->second);
}

void SyncEngine::publish_all() {
    for (const auto& kv : local_) {
        publish_dev(kv.first);
    }
    for (const CaptureContext::Device& d : ctx_.devices) {
        map_.finish(d.chip_id);
    }
}

void SyncEngine::log_summary() const {
    for (const auto& [dev, l] : local_) {
        size_t nb = 0;
        double smin = std::numeric_limits<double>::max(), smax = 0.0;
        long double ssum = 0.0, wsum = 0.0;
        for (const LocalClockModel::Run& b : l.runs) {
            if (b.n == 0 || b.slope() <= 0.0) {
                continue;
            }
            const double s = b.slope();
            nb++;
            ssum += static_cast<long double>(s) * static_cast<long double>(b.n);
            wsum += static_cast<long double>(b.n);
            smin = std::min(smin, s);
            smax = std::max(smax, s);
        }
        const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
        if (nb == 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync chip {}: {} clock records but no segment of the local clock model",
                chip,
                l.n_total);
            continue;
        }
        const double mean = static_cast<double>(ssum / wsum);
        const double to_ghz = LocalClockModel::kRefclkHz * 1e-9;
        const double anchor_ghz = dev < ctx_.devices.size() ? ctx_.devices[dev].clock.frequency_ghz : 0.0;
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync chip {}: local clock {} points in {} segments ({} steps), {} raw transition "
            "samples; applied AICLK mean {:.5f} GHz (segment min {:.5f}, max {:.5f}; boot anchor {:.5f}), spread "
            "{:.1f} "
            "ppm; {} correction nodes, the tangent extended {} times",
            chip,
            l.points,
            nb,
            l.transitions,
            l.raws,
            mean * to_ghz,
            smin * to_ghz,
            smax * to_ghz,
            anchor_ghz,
            mean > 0.0 ? (smax - smin) / mean * 1e6 : 0.0,
            series_.series(dev) != nullptr ? series_.series(dev)->nodes.size() : 0,
            series_.series(dev) != nullptr ? series_.series(dev)->extended : 0);
        if (const auto* ps = series_.series(dev); ps != nullptr && ps->dropped != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync chip {}: {} correction nodes refused (non-finite, or moving faster "
                "than host time)",
                chip,
                ps->dropped);
        }
    }
    const std::vector<LinkSolver::LinkSolution>& solved = links_.solutions();
    for (size_t li = 0; li < solved.size() && li < ctx_.links.size(); li++) {
        const LinkSolver::LinkSolution& s = solved[li];
        const CaptureContext::Link& L = ctx_.links[li];
        if (!s.ok) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): not solved (no or too "
                "few link stamps drained)",
                L.chip_a,
                L.eth_a.x,
                L.eth_a.y,
                L.chip_b,
                L.eth_b.x,
                L.eth_b.y);
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {} eth({},{}): {} rounds ({} kept); refclk "
            "domain offset {:.1f} ns, rate {:.3f} ppm, residual rms {:.1f} ns, estimate precision {:.2f} ns",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            s.rounds,
            s.kept,
            s.offset_refclk * kNsPerRefclk,
            s.rate * 1e6,
            s.residual_rms_ns,
            s.precision_ns);
    }
    // Loop closure. The tree composes every chip onto the root through some of the links; a solved link the tree
    // did not take predicts the same receiver refclk a second way, and the two ways can differ only by path
    // asymmetry (true clock offsets cancel around a loop) plus the solutions' own precision. This is the only
    // handle on asymmetry without an external reference.
    if (!local_.empty()) {
        std::vector<bool> used;
        const std::map<uint32_t, RootXf> to_root = links_.root_transforms(root_dev(), &used);
        for (size_t li = 0; li < solved.size() && li < ctx_.links.size(); li++) {
            const LinkSolver::LinkSolution& s = solved[li];
            if (!s.ok || used[li]) {
                continue;
            }
            const auto S = to_root.find(s.dev_snd);
            const auto R = to_root.find(s.dev_rcv);
            if (S == to_root.end() || R == to_root.end() || !S->second.ok || !R->second.ok) {
                continue;
            }
            const double direct = (1.0 + s.rate) * s.mid_refclk + (s.offset_refclk - s.rate * s.mid_refclk);
            const double via_tree =
                (S->second.scale * s.mid_refclk + S->second.shift - R->second.shift) / R->second.scale;
            if (links_.pair_size(li) > 1) {
                log_info(
                    tt::LogMetal,
                    "[streaming profiler] d2d sync link chip {} eth({},{}) -> chip {}: {:+.1f} ns off its pair's mean "
                    "(the parallel links' path-asymmetry difference, shared out)",
                    ctx_.links[li].chip_a,
                    ctx_.links[li].eth_a.x,
                    ctx_.links[li].eth_a.y,
                    ctx_.links[li].chip_b,
                    (via_tree - direct) * kNsPerRefclk);
            } else {
                log_info(
                    tt::LogMetal,
                    "[streaming profiler] d2d sync loop through link chip {} -> chip {}: closes to {:+.1f} ns (path "
                    "asymmetry around the loop, solutions good to ~{:.1f} ns each)",
                    ctx_.links[li].chip_a,
                    ctx_.links[li].chip_b,
                    (via_tree - direct) * kNsPerRefclk,
                    s.precision_ns);
            }
        }
    }
    if (links_.dropped() != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync: {} PP_CLOCK samples ignored (unknown kind, a core on no link, or a role "
            "that end does not stamp)",
            links_.dropped());
    }
}

// The link solve: receiver refclk = sender refclk + offset + rate * (sender refclk - mean midpoint), a straight line
// through the rounds' (midpoint, receiver minus midpoint) points with two passes of 3-sigma trimming. A solution
// that is not finite, beyond 100 ppm or a millisecond of residual is refused and the previous one stands.
bool LinkSolver::solve_link(const CaptureContext::Link& L, std::vector<RoundPoint> pts, LinkSolution& out) const {
    if (pts.size() < 4) {
        return false;
    }
    double mid0 = pts.front().mid_refclk;
    long double mid_acc = 0;
    for (const RoundPoint& p : pts) {
        mid0 = std::min(mid0, p.mid_refclk);
        mid_acc += p.mid_refclk;
    }
    std::vector<char> keep(pts.size(), 1);
    double inter = 0.0, slope = 0.0, rms = 0.0;
    size_t nk = 0;
    for (int pass = 0; pass < 3; pass++) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        nk = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (!keep[i]) {
                continue;
            }
            const double x = pts[i].mid_refclk - mid0, y = pts[i].off_refclk;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            nk++;
        }
        if (nk < 4) {
            return false;
        }
        const double nn = static_cast<double>(nk);
        const double den = nn * sxx - sx * sx;
        slope = std::abs(den) > 1e-9 ? (nn * sxy - sx * sy) / den : 0.0;
        inter = (sy - slope * sx) / nn;
        double ss = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (keep[i]) {
                const double res = pts[i].off_refclk - (inter + slope * (pts[i].mid_refclk - mid0));
                ss += res * res;
            }
        }
        rms = std::sqrt(ss / nn);
        if (pass == 2) {
            break;
        }
        const double cut = 3.0 * std::max(rms, 0.5);
        for (size_t i = 0; i < pts.size(); i++) {
            if (keep[i] && std::abs(pts[i].off_refclk - (inter + slope * (pts[i].mid_refclk - mid0))) > cut) {
                keep[i] = 0;
            }
        }
    }
    if (!std::isfinite(inter) || !std::isfinite(slope) || std::abs(slope) > 1e-4 || rms * kNsPerRefclk > 1e6) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} -> chip {}: link solution refused (offset {:.1f} ns, rate "
            "{:.3f} ppm, "
            "residual {:.1f} ns); keeping the previous one",
            L.chip_a,
            L.chip_b,
            inter * kNsPerRefclk,
            slope * 1e6,
            rms * kNsPerRefclk);
        return false;
    }
    out.ok = true;
    out.dev_snd = L.dev_a;
    out.dev_rcv = L.dev_b;
    out.rate = slope;
    out.mid_refclk = static_cast<double>(mid_acc / static_cast<long double>(pts.size()));
    // The fit's intercept sits at mid0, the window's first round; the solution is read as offset + rate * (mid -
    // out.mid_refclk), so move it to the mean or every consumer carries rate * half a window (0.45 ppm * 0.5 s = 225
    // ns).
    out.offset_refclk = inter + slope * (out.mid_refclk - mid0);
    out.residual_rms_ns = rms * kNsPerRefclk;
    out.precision_ns = rms * kNsPerRefclk / std::sqrt(static_cast<double>(nk));
    out.rounds = pts.size();
    out.kept = nk;
    return true;
}

bool SyncEngine::round_error(
    const CaptureContext::Link& L, const Round& r, int64_t& tsc_a, double& err, RoundTerms* terms) const {
    if (L.dev_a >= ctx_.devices.size() || L.dev_b >= ctx_.devices.size()) {
        return false;
    }
    const auto la = local_.find(L.dev_a);
    const auto lb = local_.find(L.dev_b);
    if (la == local_.cend() || lb == local_.cend()) {
        return false;
    }
    if (!series_.has_nodes(L.dev_a) || !series_.has_nodes(L.dev_b)) {
        return false;
    }
    const double wa = la->second.wall_at(LinkSolver::mid_a_refclk(r));
    const double wb = lb->second.wall_at(LinkSolver::mid_b_refclk(r));
    if (wa <= 0.0 || wb <= 0.0) {
        return false;
    }
    RoundTerms t;
    t.wall_a = wa;
    t.wall_b = wb;
    t.root_a = map_.lookup_root(ctx_.devices[L.dev_a].chip_id, std::llround(wa));
    t.root_b = map_.lookup_root(ctx_.devices[L.dev_b].chip_id, std::llround(wb));
    tsc_a = std::llround(map_.host_tsc(t.root_a));
    err = (t.root_b - t.root_a) * kNsPerRefclk;
    if (terms != nullptr) {
        *terms = t;
    }
    return true;
}

// Per link and round: the receiver's stamp and the sender's round midpoint (one instant, under the symmetric path)
// placed on the host timeline exactly as the sink places a record from each chip's eth core, against the final map.
// The worst rounds are logged with their distance to the nearest correction node, since the map's residual lives at
// the transitions.
void SyncEngine::publish_error_plots() {
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        const std::vector<Round>& rounds = links_.rounds(li);
        if (rounds.empty()) {
            continue;
        }
        const size_t n = rounds.size();
        // Per round, next to the placement error: the stamps' own residual against a line through the neighbouring
        // rounds' raw offsets (the ruler's noise -- a stamp glitch shows here, a map error does not; the link rate
        // wanders ~0.1 ppm over a run, so a single run-wide line would not do) and the sender's round trip (a glitch
        // on either sender stamp shows here, one on the receiver does not).
        std::vector<PlotPoint> pts;
        std::vector<double> resid, rtt, raw_x, raw_y, turn, path;
        const double path_med = LinkSolver::path_median(rounds, 0, n);
        size_t off_path = 0;
        std::vector<RoundTerms> terms;
        pts.reserve(n);
        rtt.reserve(n);
        terms.reserve(n);
        long double se = 0, ss = 0;
        for (const Round& r : rounds) {
            if (std::abs(LinkSolver::path_ns(r) - path_med) > LinkSolver::kPathDevNs) {
                off_path++;
                continue;
            }
            int64_t H = 0;
            double e = 0.0;
            RoundTerms t;
            if (!round_error(L, r, H, e, &t)) {
                continue;
            }
            pts.push_back(PlotPoint{H, e});
            terms.push_back(t);
            se += e;
            ss += static_cast<long double>(e) * e;
            raw_x.push_back(LinkSolver::mid_a_refclk(r));
            raw_y.push_back(LinkSolver::mid_b_refclk(r) - raw_x.back());
            rtt.push_back(LinkSolver::rtt_ns(r));
            path.push_back(LinkSolver::path_ns(r));
            turn.push_back(rtt.back() - 2.0 * path.back());
        }
        if (pts.empty()) {
            continue;
        }
        resid.assign(pts.size(), 0.0);
        long double srr = 0;
        if (pts.size() >= 8) {
            constexpr size_t kHalf = 12;
            for (size_t i = 0; i < pts.size(); i++) {
                const size_t lo = i > kHalf ? i - kHalf : 0, hi = std::min(pts.size(), i + kHalf + 1);
                long double sx = 0, sy = 0, sxx = 0, sxy = 0;
                size_t m = 0;
                for (size_t j = lo; j < hi; j++) {
                    if (j == i) {
                        continue;
                    }
                    const long double x = raw_x[j] - raw_x[i], y = raw_y[j];
                    sx += x, sy += y, sxx += x * x, sxy += x * y, m++;
                }
                const long double den = static_cast<long double>(m) * sxx - sx * sx;
                const long double b = den > 0 ? (static_cast<long double>(m) * sxy - sx * sy) / den : 0;
                const long double a = (sy - b * sx) / static_cast<long double>(m);
                resid[i] = static_cast<double>((raw_y[i] - a) * kNsPerRefclk);
                srr += static_cast<long double>(resid[i]) * resid[i];
            }
        }
        double rtt_median = 0.0;
        {
            std::vector<double> tmp = rtt;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            rtt_median = tmp[tmp.size() / 2];
        }
        const double nn = static_cast<double>(pts.size());
        {
            const auto pct = [](std::vector<double> v, double q) {
                std::erase_if(v, [](double x) { return std::isnan(x); });
                if (v.empty()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const size_t k = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())));
                std::nth_element(v.begin(), v.begin() + k, v.end());
                return v[k];
            };
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} -> chip {}: one way inside the stamps {:.1f} "
                "ns "
                "(p10 {:.1f}, p90 {:.1f}); receiver's stamped turnaround {:.1f} ns (p10 {:.1f}, p90 {:.1f}); "
                "sender's "
                "round trip {:.1f} ns (p10 {:.1f}, p90 {:.1f}); {} rounds, {} off the path band dropped",
                L.chip_a,
                L.chip_b,
                pct(path, 0.5),
                pct(path, 0.1),
                pct(path, 0.9),
                pct(turn, 0.5),
                pct(turn, 0.1),
                pct(turn, 0.9),
                pct(rtt, 0.5),
                pct(rtt, 0.1),
                pct(rtt, 0.9),
                n,
                off_path);
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync error chip {} vs chip {}: {} rounds, mean {:+.1f} ns, rms {:.1f} ns{}",
            L.chip_b,
            L.chip_a,
            pts.size(),
            static_cast<double>(se) / nn,
            std::sqrt(static_cast<double>(ss) / nn),
            pts.size() >= 8 ? fmt::format(
                                  "; the stamps' own noise (residual to the neighbouring rounds): rms {:.1f} ns",
                                  std::sqrt(static_cast<double>(srr / static_cast<long double>(pts.size()))))
                            : "");
        // The five worst rounds, how far each sits from a correction node of either chip, and its two glitch
        // indicators: a map error has a small residual; a stamp glitch has a large one and often a shifted round
        // trip.
        std::vector<double> node_a_us(pts.size(), -1.0), node_b_us(pts.size(), -1.0);
        for (const auto& [dev, out] : {std::pair{L.dev_a, &node_a_us}, std::pair{L.dev_b, &node_b_us}}) {
            const SeriesPublisher::Series* ps = series_.series(dev);
            if (ps == nullptr) {
                continue;
            }
            const std::vector<SeriesPublisher::Node>& nodes = ps->nodes;
            const double ghz = std::max(ctx_.devices[dev].clock.frequency_ghz, 0.1);
            for (size_t i = 0; i < pts.size(); i++) {
                const double wall = dev == L.dev_a ? terms[i].wall_a : terms[i].wall_b;
                const auto up =
                    std::lower_bound(nodes.begin(), nodes.end(), wall, [](const SeriesPublisher::Node& nd, double h) {
                        return nd.H < h;
                    });
                for (const auto* nd :
                     {up != nodes.end() ? &*up : nullptr, up != nodes.begin() ? &*(up - 1) : nullptr}) {
                    if (nd == nullptr) {
                        continue;
                    }
                    const double d = std::abs(nd->H - wall) / ghz / 1e3;
                    if ((*out)[i] < 0.0 || d < (*out)[i]) {
                        (*out)[i] = d;
                    }
                }
            }
        }
        {
            std::vector<size_t> order(pts.size());
            for (size_t i = 0; i < order.size(); i++) {
                order[i] = i;
            }
            std::partial_sort(
                order.begin(), order.begin() + std::min<size_t>(5, order.size()), order.end(), [&](size_t a, size_t b) {
                    return std::abs(pts[a].value) > std::abs(pts[b].value);
                });
            std::string worst;
            for (size_t k = 0; k < std::min<size_t>(5, order.size()); k++) {
                const size_t i = order[k];
                const PlotPoint& p = pts[i];
                worst += fmt::format(
                    " {:+.0f} ns at {:.3f} s (nearest node: chip {} {:.0f} us, chip {} {:.0f} us; stamp resid "
                    "{:+.0f} "
                    "ns, path {:+.1f} ns vs median);",
                    p.value,
                    static_cast<double>(SteadyView::mono_ns(p.tsc) - SteadyView::mono_ns(pts.front().tsc)) / 1e9,
                    L.chip_a,
                    node_a_us[i],
                    L.chip_b,
                    node_b_us[i],
                    resid[i],
                    path[i] - path_med);
            }
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync error chip {} vs chip {}: worst rounds{}",
                L.chip_b,
                L.chip_a,
                worst);
        }
        if (const std::string& csv = ctx_.d2d_csv_path; !csv.empty()) {
            if (std::FILE* ef = std::fopen(
                    fmt::format("{}.err_{}_{}_eth{}_{}.csv", csv, L.chip_b, L.chip_a, L.eth_a.x, L.eth_a.y).c_str(),
                    "w");
                ef != nullptr) {
                std::fprintf(
                    ef,
                    "host_ns,err_ns,stamp_resid_ns,rtt_dev_ns,node_a_us,node_b_us,mid_a_refclk,r1_b_refclk,wall_a,"
                    "wall_b,root_a,root_b,rtt_ns,turn_ns,path_ns\n");
                for (size_t i = 0; i < pts.size(); i++) {
                    const RoundTerms& t = terms[i];
                    std::fprintf(
                        ef,
                        "%lld,%.2f,%.2f,%.1f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f\n",
                        static_cast<long long>(SteadyView::mono_ns(pts[i].tsc)),
                        pts[i].value,
                        resid[i],
                        rtt[i] - rtt_median,
                        node_a_us[i],
                        node_b_us[i],
                        raw_x[i],
                        raw_x[i] + raw_y[i],
                        t.wall_a,
                        t.wall_b,
                        t.root_a,
                        t.root_b,
                        rtt[i],
                        turn[i],
                        path[i]);
                }
                std::fclose(ef);
            }
        }
        plot(fmt::format("d2d sync error chip{} vs chip{} (ns)", L.chip_b, L.chip_a), pts);
    }
}

void SyncEngine::publish_clock_plots() {
    for (const auto& [dev, fit] : local_) {
        if (!series_.has_nodes(dev) || dev >= ctx_.devices.size()) {
            continue;
        }
        const uint32_t chip = ctx_.devices[dev].chip_id;
        std::vector<PlotPoint> pts;
        for (const LocalClockModel::Run& run : fit.runs) {
            if (run.n == 0 || run.slope() <= 0.0) {
                continue;
            }
            const double ghz = run.slope() * LocalClockModel::kRefclkHz * 1e-9;
            for (const double r : {run.r_first, run.r_last}) {
                const double root = map_.lookup_root(chip, std::llround(run.wall_of_refclk(r)));
                pts.push_back(PlotPoint{std::llround(map_.host_tsc(root)), ghz});
            }
        }
        if (!pts.empty()) {
            plot(fmt::format("AICLK chip{} (GHz)", chip), pts);
        }
    }
}

void SyncEngine::plot([[maybe_unused]] const std::string& name, [[maybe_unused]] const std::vector<PlotPoint>& points) {
#if defined(TRACY_ENABLE)
    if (!plots_to_tracy_) {
        return;
    }
    const char* nm = plot_names_.insert(name).first->c_str();
    for (const PlotPoint& p : points) {
        if (p.tsc > 0) {
            tracy::Profiler::PlotDataAt(nm, p.value, p.tsc);
        }
    }
#endif
}

void SyncEngine::on_capture_end(const CaptureContext& ctx) {
    (void)ctx;
    links_.solve_final();
    publish_all();
    publish_error_plots();
    publish_clock_plots();
    log_summary();
    // The published corrections stay for the sinks that write at process end; the next attach starts fresh.
    local_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
