// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <cstdio>
#include <string>

#include <fmt/format.h>

#include "impl/streaming_profiler/spsc_packet.h"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"

namespace tt::tt_metal::streaming_profiler {

void D2dSyncConsumer::on_attach(const CaptureContext& ctx) {
    ctx_ = ctx;
    local_.clear();
    links_.assign(ctx.links.size(), LinkStreams{});
    side_of_.clear();
    for (size_t li = 0; li < ctx.links.size(); li++) {
        const CaptureContext::Link& L = ctx.links[li];
        if (const int64_t c = core_index(L.dev_a, L.eth_a); c >= 0) {
            side_of_[{L.dev_a, static_cast<uint32_t>(c)}] = {li, true};
        }
        if (const int64_t c = core_index(L.dev_b, L.eth_b); c >= 0) {
            side_of_[{L.dev_b, static_cast<uint32_t>(c)}] = {li, false};
        }
    }
    solved_.assign(ctx.links.size(), LinkSolution{});
    live_err_.assign(ctx.links.size(), {});
    live_done_.assign(ctx.links.size(), 0);
    dropped_kind_ = 0;
    published_.clear();
    to_root_gen_ = ~0ull;
    solve_gen_ = 0;
    // A new capture: its placements start from nothing. A chip the capture cannot place -- no eth tracker, or no
    // link path to the root -- is finished at once, so no consumer waits for it.
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
        SyncCorrections::clear(d.chip_id);
        if (!reach[dev] || !d.has_eth_tracker) {
            SyncCorrections::finish(d.chip_id);
        }
    }
    SyncPlots::expect();
}

void D2dSyncConsumer::on_clock(const ClockSample& s) {
    if (s.kind == PP_CLOCK_LOCAL_REFCLK) {
        LocalState& l = local_[s.dev];
        l.fit.add(s.value, s.ts);
        if (csv_path_ != nullptr) {
            l.samples.emplace_back(s.value, s.ts);
        }
        if (publish_dev(s.dev)) {
            service().wake_consumers();
        }
        return;
    }
    const auto side = side_of_.find({s.dev, s.lane / profiler::kSpscNRiscDecode});
    if ((s.kind != PP_CLOCK_LINK_REFCLK && s.kind != PP_CLOCK_LINK_PTP) || side == side_of_.end()) {
        dropped_kind_++;
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
        dropped_kind_++;
        return;
    }
    const bool hw = s.kind == PP_CLOCK_LINK_PTP;
    LinkRounds& lr = hw ? links_[li].hw : links_[li].sw;
    Round& r = lr.pending[s.round];
    r.id = s.round;
    r.*slot = Stamp{.value = s.value, .wall = s.ts, .have = true};
    if (r.complete()) {
        lr.rounds.push_back(r);
        lr.pending.erase(s.round);
        try_solve_links(/*final=*/false);
    }
    while (lr.pending.size() > kPendingMax) {
        lr.pending.erase(lr.pending.begin());
    }
    // As-delivered errors: a round is evaluated once both chips' covers reach its stamps, the moment a consumer
    // waiting on them converts its records.
    const CaptureContext::Link& L = ctx_.links[li];
    const std::vector<Round>& rounds = links_[li].primary().rounds;
    const bool phw = links_[li].have_hw();
    for (; live_done_[li] < rounds.size(); live_done_[li]++) {
        int64_t H = 0;
        double e = 0.0;
        RoundTerms t;
        if (!round_error(L, rounds[live_done_[li]], phw, H, e, &t)) {
            continue;
        }
        if (t.wall_a > static_cast<double>(SyncCorrections::cover_ticks(L.chip_a)) ||
            t.wall_b > static_cast<double>(SyncCorrections::cover_ticks(L.chip_b))) {
            break;
        }
        live_err_[li].push_back(SyncPlotPoint{H, e});
    }
}

// Only the trailing eth cores are searched: eth and worker logical coordinates overlap ((0,7) is both a worker and
// an eth core), and the decoder indexes a pushed eth frame by its core's own position in the roster.
int64_t D2dSyncConsumer::core_index(uint32_t dev, const CoreCoord& eth) const {
    if (dev >= ctx_.devices.size()) {
        return -1;
    }
    const CaptureContext::Device& d = ctx_.devices[dev];
    const size_t n_cores = d.lanes.size() / profiler::kSpscNRiscDecode;
    for (size_t ci = n_cores >= d.n_eth_cores ? n_cores - d.n_eth_cores : 0; ci < n_cores; ci++) {
        if (d.lanes[ci * profiler::kSpscNRiscDecode].logical == eth) {
            return static_cast<int64_t>(ci);
        }
    }
    return -1;
}

// Both stamp kinds solve in the refclk domain over the latest window of rounds with one regression. Hardware
// stamps are refclk ticks already. Software wall stamps are put there through the tracker's run map (one line per
// constant-rate run, so a DVFS excursion inside the window no longer bends them as one ratio per window did) and
// keep their fastest quartile by round trip, since an ERISC stalled inside a round shows in its trip time.
void D2dSyncConsumer::try_solve_links(bool final) {
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        LinkSolution& out = solved_[li];
        const CaptureContext::Link& L = ctx_.links[li];
        const int64_t ca = core_index(L.dev_a, L.eth_a);
        const int64_t cb = core_index(L.dev_b, L.eth_b);
        if (ca < 0 || cb < 0) {
            continue;
        }
        const LinkStreams& ls = links_[li];
        const bool hw = ls.have_hw();
        const std::vector<Round>& rounds = ls.primary().rounds;
        const size_t n = rounds.size();
        if (final) {
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link {} (dev {} eth({},{}) core {}) -> (dev {} eth({},{}) core {}): "
                "software rounds {} complete, {} pending; hardware rounds {} complete, {} pending",
                li,
                L.dev_a,
                L.eth_a.x,
                L.eth_a.y,
                ca,
                L.dev_b,
                L.eth_b.x,
                L.eth_b.y,
                cb,
                ls.sw.rounds.size(),
                ls.sw.pending.size(),
                ls.hw.rounds.size(),
                ls.hw.pending.size());
        }
        const auto la = local_.find(L.dev_a);
        const auto lb = local_.find(L.dev_b);
        if (n == 0 || (!hw && (la == local_.end() || lb == local_.end() || la->second.fit.runs.empty() ||
                               lb->second.fit.runs.empty()))) {
            continue;
        }
        const auto to_refclk = [](const LocalClockFit& fit, uint64_t wall) {
            const double wd = static_cast<double>(wall);
            return fit.run_at_wall(wd).refclk_of_wall(wd);
        };
        const auto pos = [&](const Round& r) { return hw ? mid_a(r, true) : to_refclk(la->second.fit, r.t0.wall); };
        const double newest = pos(rounds[n - 1]);
        // Live: the first solution once kFirstSolveTicks of rounds are in, so held records are released, then one
        // every half window. Final: whatever the window holds, if it is enough for a fit at all.
        if (!final && (out.ok ? newest < out.solved_at + kLinkWindowTicks / 2
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
        out.solved_at = newest;
        std::vector<RoundPoint> pts;
        pts.reserve(w);
        if (hw) {
            const double path_med = path_median(rounds, begin, n);
            out.path_dropped = 0;
            for (size_t i = begin; i < n; i++) {
                const Round& r = rounds[i];
                if (std::abs(path_ns(r) - path_med) > kPathDevNs) {
                    out.path_dropped++;
                    continue;
                }
                const double mid = mid_a(r, true);
                pts.push_back(RoundPoint{mid, mid_b(r, true) - mid, 0.0});
            }
        } else {
            for (size_t i = begin; i < n; i++) {
                const Round& r = rounds[i];
                if (r.t0.wall == 0 || r.t2.wall == 0 || r.t1.wall == 0 || r.t1b.wall == 0 || r.t2.wall < r.t0.wall) {
                    continue;
                }
                const double mid = 0.5 * (to_refclk(la->second.fit, r.t0.wall) + to_refclk(la->second.fit, r.t2.wall));
                const double mid_rcv =
                    0.5 * (to_refclk(lb->second.fit, r.t1.wall) + to_refclk(lb->second.fit, r.t1b.wall));
                pts.push_back(RoundPoint{mid, mid_rcv - mid, static_cast<double>(r.t2.wall - r.t0.wall)});
            }
            std::sort(pts.begin(), pts.end(), [](const RoundPoint& a, const RoundPoint& b) { return a.rtt < b.rtt; });
            pts.resize(std::clamp<size_t>(pts.size() / 4, std::min<size_t>(8, pts.size()), pts.size()));
        }
        if (solve_link(L, std::move(pts), hw, out)) {
            out.rounds = w;
            solve_gen_++;
            SyncCorrections::set_asymmetry_ns(max_closure_ns());
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync link chip {} -> chip {}: solved at round {} over {} ({} kept): offset "
                "{:.1f} ns, rate {:.3f} ppm, residual {:.1f} ns{}{}",
                L.chip_a,
                L.chip_b,
                n,
                w,
                out.kept,
                out.offset_ns,
                out.rate_ppm,
                out.residual_rms_ns,
                hw ? " [hw]" : " [sw]",
                out.path_dropped != 0 ? fmt::format(", {} rounds off the stamp path band", out.path_dropped) : "");
        }
    }
}

double D2dSyncConsumer::mid_a(const Round& r, bool hw) {
    return 0.5 * (static_cast<double>(r.t0.value) + static_cast<double>(r.t2.value)) * (hw ? kHwUnitTicks : 1.0);
}

double D2dSyncConsumer::mid_b(const Round& r, bool hw) {
    return 0.5 * (static_cast<double>(r.t1.value) + static_cast<double>(r.t1b.value)) * (hw ? kHwUnitTicks : 1.0);
}

double D2dSyncConsumer::path_median(const std::vector<Round>& rounds, size_t begin, size_t n) {
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
std::map<uint32_t, D2dSyncConsumer::RootXf> D2dSyncConsumer::root_transforms(
    uint32_t root, std::vector<bool>* used) const {
    std::map<uint32_t, RootXf> to_root;
    to_root[root] = RootXf{1.0, 0.0, 0.0, true};
    if (used != nullptr) {
        used->assign(solved_.size(), false);
    }
    for (bool progress = true; progress;) {
        progress = false;
        for (size_t li = 0; li < solved_.size(); li++) {
            const LinkSolution& s = solved_[li];
            if (!s.ok) {
                continue;
            }
            const double m = 1.0 + s.rate;  // receiver = m * sender + o
            const double o = s.offset_ticks - s.rate * s.mid;
            const auto rs = to_root.find(s.dev_rcv);
            const auto ss = to_root.find(s.dev_snd);
            const bool r_ok = rs != to_root.end() && rs->second.ok;
            const bool s_ok = ss != to_root.end() && ss->second.ok;
            const auto along = [&](const RootXf& from) {
                return std::sqrt(from.prec_ns * from.prec_ns + s.precision_ns * s.precision_ns);
            };
            if (s_ok && !r_ok) {
                // root = A_s * sender + B_s, and sender = (receiver - o) / m.
                const RootXf& S = ss->second;
                to_root[s.dev_rcv] = RootXf{S.scale / m, S.shift - S.scale * o / m, along(S), true};
                progress = true;
            } else if (r_ok && !s_ok) {
                // root = A_r * receiver + B_r, and receiver = m * sender + o.
                const RootXf& R = rs->second;
                to_root[s.dev_snd] = RootXf{R.scale * m, R.scale * o + R.shift, along(R), true};
                progress = true;
            } else {
                continue;
            }
            if (used != nullptr) {
                (*used)[li] = true;
            }
        }
    }
    return to_root;
}

double D2dSyncConsumer::max_closure_ns() const {
    if (local_.empty()) {
        return 0.0;
    }
    std::vector<bool> used;
    const std::map<uint32_t, RootXf> to_root = root_transforms(root_dev(), &used);
    double worst = 0.0;
    for (size_t li = 0; li < solved_.size(); li++) {
        const LinkSolution& s = solved_[li];
        if (!s.ok || used[li]) {
            continue;
        }
        const auto S = to_root.find(s.dev_snd);
        const auto R = to_root.find(s.dev_rcv);
        if (S == to_root.end() || R == to_root.end() || !S->second.ok || !R->second.ok) {
            continue;
        }
        const double direct = (1.0 + s.rate) * s.mid + (s.offset_ticks - s.rate * s.mid);
        const double via_tree = (S->second.scale * s.mid + S->second.shift - R->second.shift) / R->second.scale;
        worst = std::max(worst, std::abs(via_tree - direct) * 20.0);
    }
    return worst;
}

D2dSyncConsumer::Fresh D2dSyncConsumer::fresh_nodes(const Series& s, const LocalClockFit& fit, const RootXf& xf) const {
    const std::vector<LocalClockFit::Run>& runs = fit.runs;
    // A node places one instant of a run on the root: its eth wall tick (the key every record of the chip is looked
    // up by, worker lanes through their tile offset) and the root's refclk at that instant, via the chip's refclk
    // and the solved links. Along a run the map is exactly linear.
    const auto node_at = [&](const LocalClockFit::Run& run, double r) {
        const double T = run.wall_of_refclk(r);
        const double root = xf.scale * r + xf.shift;
        const double tangent = xf.scale / run.slope();
        const double wall_ghz = run.slope() * LocalClockFit::kRefclkHz * 1e-9;
        const double se_ns = wall_ghz > 0.0 ? run.se_ticks(r) / wall_ghz : 0.0;
        const double sigma = std::sqrt(se_ns * se_ns + xf.prec_ns * xf.prec_ns);
        if (!(tangent > 0.0 && tangent < 1.0)) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync: a node at refclk {:.0f} is not a rate; its run [{:.0f}, {:.0f}] has {} "
                "samples, fitted slope {:.6f}, ratio {:.4f}, slope {:.6f}; root scale {:.9f}",
                r,
                run.r_first,
                run.r_last,
                run.n,
                run.fitted_slope(),
                run.ratio(),
                run.slope(),
                xf.scale);
        }
        return Node{T, root, r, tangent, sigma};
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
        const LocalClockFit::Run& a = runs[i];
        const LocalClockFit::Run& b = runs[i + 1];
        if (i + 2 == runs.size() && !b.settled()) {
            break;
        }
        // Two lines with different slopes meet at the transition; two with the same slope (a run cut by length, or a
        // spurious split) are bridged by a node on each side of the seam. Past the knot the records lie on b, so the
        // knot leaves on b's tangent. The a-side node of a bridge is dropped when the frontier already passed it: a's
        // newest sample was handed to b after the tangent was frozen on it, and the frozen tangent's end is that node.
        if (const auto r_x = LocalClockFit::knot(a, b)) {
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
    // The open run's line reaches to now, but no node is frozen inside its last kProvisionalTicks: a small DVFS step
    // hides below the tracker's threshold for ~10 us, so the newest samples may yet prove to belong to the next run,
    // and a node frozen past a transition cannot be taken back -- it once froze 9.6 us past one, 45 ns off, and the
    // knot then computed earlier in time was dropped behind it.
    const LocalClockFit::Run& open = runs.back();
    const double r_end = open.r_last - kProvisionalTicks;
    if (open.settled() && open.slope() > 0.0 && r_end > last_r && out.knots_after + 1 == runs.size()) {
        out.frontier = node_at(open, r_end);
    }
    return out;
}

void D2dSyncConsumer::push_node(Series& s, uint32_t chip, const Node& n) {
    const int64_t wall = static_cast<int64_t>(std::llround(n.H));
    if (!s.nodes.empty() && wall <= static_cast<int64_t>(std::llround(s.nodes.back().H))) {
        return;  // within the tick of the last node: the placement cannot differ measurably there
    }
    s.nodes.push_back(n);
    s.cover_H = n.H;
    s.cover_r = n.r;
    s.last_r = std::max(s.last_r, n.r);
    SyncCorrections::append(
        chip, SyncNode{.at = wall, .value = n.root, .tangent = n.tangent, .sigma_ns = static_cast<float>(n.sigma)});
}

// Frozen nodes never move (consumers have placed records against them), so a publish can only add beyond them, at
// the newest estimate's values; a join carries whatever the estimate moved by since the tangent was frozen (kFreezeNs
// at most on a frontier, a few ns at a knot). Shifting fresh nodes to meet the frozen tail, and fading that shift
// over a quarter second, was tried first: it turned every discrepancy at a join into a level the map carried for
// 250 ms, 30-60 ns during DVFS dithering at 1 ms.
void D2dSyncConsumer::freeze_append(Series& s, uint32_t chip, const Node& n) {
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
        push_node(
            s,
            chip,
            Node{s.cover_H, last.root + last.tangent * (s.cover_H - last.H), s.cover_r, last.tangent, last.sigma});
    }
    if (!std::isfinite(n.H) || !std::isfinite(n.root) || !std::isfinite(n.tangent) ||
        !(n.tangent > 0.0 && n.tangent < 1.0)) {
        s.dropped++;
        return;
    }
    push_node(s, chip, n);
}

bool D2dSyncConsumer::advance(Series& s, uint32_t chip, Fresh fresh) {
    const double cover_before = s.cover_H;
    for (const Node& k : fresh.knots) {
        freeze_append(s, chip, k);
        s.last_r = std::max(s.last_r, k.r);
    }
    s.knots = fresh.knots_after;
    if (fresh.frontier) {
        const Node& f = *fresh.frontier;
        const Node* last = s.nodes.empty() ? nullptr : &s.nodes.back();
        const double freeze_ticks = kFreezeNs * LocalClockFit::kRefclkHz * 1e-9;
        if (last != nullptr && f.H > s.cover_H &&
            std::abs(f.root - (last->root + last->tangent * (f.H - last->H))) <= freeze_ticks) {
            s.cover_H = f.H;
            s.cover_r = f.r;
            s.last_r = std::max(s.last_r, f.r);
            s.extended++;
            SyncCorrections::extend(chip, static_cast<int64_t>(std::llround(f.H)));
        } else {
            freeze_append(s, chip, f);
        }
    }
    return s.cover_H > cover_before;
}

bool D2dSyncConsumer::publish_dev(uint32_t dev) {
    const auto st = local_.find(dev);
    if (st == local_.end() || st->second.fit.runs.empty() || dev >= ctx_.devices.size()) {
        return false;
    }
    // Nothing is placed before the host series exists: a record converted then would land nowhere.
    if (SyncCorrections::host_published() == 0) {
        return false;
    }
    if (to_root_gen_ != solve_gen_) {
        to_root_ = root_transforms(root_dev(), nullptr);
        to_root_gen_ = solve_gen_;
    }
    // The series starts only once the chip is on the root's tree (the root is there from the start): its first node
    // fixes the placement every later node joins.
    const auto xf = to_root_.find(dev);
    if (xf == to_root_.end() || !xf->second.ok) {
        return false;
    }
    Published& pub = published_[dev];
    return advance(pub.linked, ctx_.devices[dev].chip_id, fresh_nodes(pub.linked, st->second.fit, xf->second));
}

// Capture end: every chip's series from the final fits and solutions, then closed, so the consumers' remaining
// records convert on the newest tangents and nothing waits.
void D2dSyncConsumer::publish_all() {
    for (const auto& kv : local_) {
        publish_dev(kv.first);
    }
    for (const CaptureContext::Device& d : ctx_.devices) {
        SyncCorrections::finish(d.chip_id);
    }
}

void D2dSyncConsumer::log_summary() const {
    for (const auto& [dev, l] : local_) {
        size_t nb = 0;
        double smin = std::numeric_limits<double>::max(), smax = 0.0;
        long double ssum = 0.0, wsum = 0.0;
        for (const LocalClockFit::Run& b : l.fit.runs) {
            if (b.n < 2) {
                continue;
            }
            const double s = b.slope();
            if (s <= 0.0) {
                continue;
            }
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
                "[streaming profiler] d2d sync chip {}: {} local clock samples but no fittable constant-rate run",
                chip,
                l.fit.n_total);
            continue;
        }
        const double mean = static_cast<double>(ssum / wsum);
        const LocalClockFit::Run* longest = nullptr;
        for (const LocalClockFit::Run& b : l.fit.runs) {
            if (b.n >= 2 && (longest == nullptr || b.n > longest->n)) {
                longest = &b;
            }
        }
        const double off_ppm =
            (longest != nullptr && longest->ratio() > 0.0) ? (longest->fitted_slope() / longest->ratio() - 1.0) * 1e6 : 0.0;
        const double to_ghz = LocalClockFit::kRefclkHz * 1e-9;
        const double anchor_ghz = dev < ctx_.devices.size() ? ctx_.devices[dev].clock.frequency_ghz : 0.0;
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync chip {}: local clock {} samples in {} constant-rate runs ({} transitions, "
            "{} runs merged back, {} glitch samples dropped, {} tail samples handed over; longest run {:+.3f} ppm off "
            "its PLL multiple); applied AICLK mean {:.5f} GHz "
            "(run min {:.5f}, max {:.5f}; boot anchor {:.5f}), spread {:.1f} ppm; {} correction nodes, the tangent "
            "extended {} times",
            chip,
            l.fit.n_total,
            nb,
            l.fit.transitions,
            l.fit.merged,
            l.fit.glitches,
            l.fit.handed_over,
            off_ppm,
            mean * to_ghz,
            smin * to_ghz,
            smax * to_ghz,
            anchor_ghz,
            mean > 0.0 ? (smax - smin) / mean * 1e6 : 0.0,
            SyncCorrections::published(chip),
            published_.contains(dev) ? published_.at(dev).linked.extended : 0);
        if (const auto pit = published_.find(dev); pit != published_.end() && pit->second.linked.dropped != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] d2d sync chip {}: {} correction nodes refused (non-finite, or moving faster "
                "than host time)",
                chip,
                pit->second.linked.dropped);
        }
    }
    for (size_t li = 0; li < solved_.size() && li < ctx_.links.size(); li++) {
        const LinkSolution& s = solved_[li];
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
            "domain offset {:.1f} ns, rate {:.3f} ppm, residual rms {:.1f} ns, estimate precision {:.2f} ns{}",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            s.rounds,
            s.kept,
            s.offset_ns,
            s.rate_ppm,
            s.residual_rms_ns,
            s.precision_ns,
            s.hw ? " [1588 hardware stamps]" : "");
    }
    // Loop closure. The tree composes every chip onto the root through some of the links; a solved link the tree
    // did not take predicts the same receiver refclk a second way, and the two ways can differ only by path
    // asymmetry (true clock offsets cancel around a loop) plus the solutions' own precision. This is the only
    // handle on asymmetry without an external reference.
    if (!local_.empty()) {
        std::vector<bool> used;
        const std::map<uint32_t, RootXf> to_root = root_transforms(root_dev(), &used);
        for (size_t li = 0; li < solved_.size() && li < ctx_.links.size(); li++) {
            const LinkSolution& s = solved_[li];
            if (!s.ok || used[li]) {
                continue;
            }
            const auto S = to_root.find(s.dev_snd);
            const auto R = to_root.find(s.dev_rcv);
            if (S == to_root.end() || R == to_root.end() || !S->second.ok || !R->second.ok) {
                continue;
            }
            const double direct = (1.0 + s.rate) * s.mid + (s.offset_ticks - s.rate * s.mid);
            const double via_tree = (S->second.scale * s.mid + S->second.shift - R->second.shift) / R->second.scale;
            log_info(
                tt::LogMetal,
                "[streaming profiler] d2d sync loop through link chip {} -> chip {}: closes to {:+.1f} ns (path asymmetry "
                "around the loop, solutions good to ~{:.1f} ns each)",
                ctx_.links[li].chip_a,
                ctx_.links[li].chip_b,
                (via_tree - direct) * 20.0,
                s.precision_ns);
        }
    }
    if (dropped_kind_ != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync: {} PP_CLOCK samples ignored (unknown kind, a core on no link, or a role "
            "that end does not stamp)",
            dropped_kind_);
    }
}

// The link solve: receiver refclk = sender refclk + offset + rate * (sender refclk - mean midpoint), a straight line
// through the rounds' (midpoint, receiver minus midpoint) points with two passes of 3-sigma trimming. A solution
// that is not finite, beyond 100 ppm or a millisecond of residual is refused and the previous one stands.
bool D2dSyncConsumer::solve_link(
    const CaptureContext::Link& L, std::vector<RoundPoint> pts, bool hw, LinkSolution& out) const {
    if (pts.size() < 4) {
        return false;
    }
    double mid0 = pts.front().mid;
    long double mid_acc = 0;
    for (const RoundPoint& p : pts) {
        mid0 = std::min(mid0, p.mid);
        mid_acc += p.mid;
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
            const double x = pts[i].mid - mid0, y = pts[i].off;
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
                const double res = pts[i].off - (inter + slope * (pts[i].mid - mid0));
                ss += res * res;
            }
        }
        rms = std::sqrt(ss / nn);
        if (pass == 2) {
            break;
        }
        const double cut = 3.0 * std::max(rms, 0.5);
        for (size_t i = 0; i < pts.size(); i++) {
            if (keep[i] && std::abs(pts[i].off - (inter + slope * (pts[i].mid - mid0))) > cut) {
                keep[i] = 0;
            }
        }
    }
    if (!std::isfinite(inter) || !std::isfinite(slope) || std::abs(slope) > 1e-4 || rms * 20.0 > 1e6) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] d2d sync link chip {} -> chip {}: link solution refused (offset {:.1f} ns, rate {:.3f} ppm, "
            "residual {:.1f} ns); keeping the previous one",
            L.chip_a,
            L.chip_b,
            inter * 20.0,
            slope * 1e6,
            rms * 20.0);
        return false;
    }
    out.ok = true;
    out.hw = hw;
    out.dev_snd = L.dev_a;
    out.dev_rcv = L.dev_b;
    out.rate = slope;
    out.mid = static_cast<double>(mid_acc / static_cast<long double>(pts.size()));
    // The fit's intercept sits at mid0, the window's first round; the solution is read as offset + rate * (mid -
    // out.mid), so move it to the mean or every consumer carries rate * half a window (0.45 ppm * 0.5 s = 225 ns).
    out.offset_ticks = inter + slope * (out.mid - mid0);
    out.offset_ns = out.offset_ticks * 20.0;
    out.rate_ppm = slope * 1e6;
    out.residual_rms_ns = rms * 20.0;
    out.precision_ns = rms * 20.0 / std::sqrt(static_cast<double>(nk));
    out.rounds = pts.size();
    out.kept = nk;
    return true;
}

// Nodes -> segments: a constant hold before the first node (so the correction applies from the capture's first
// record instead of stepping in at the first node), linear interpolation between nodes, and the lookup's own hold
// past the last one. Every segment starts where the previous ended, so the series is continuous and its slope,
// the difference of neighbouring corrections over a millisecond, is far below 1: it cannot reorder records.
// The hedge next to the Tracy plots: a CSV of the local and linked corrections per chip over time and the
// local-vs-linked error, so the accuracy numbers exist even if the Tracy capture is fiddly. One row per run start and per ms
// bucket per chip. Gated on TT_METAL_STREAMING_PROFILER_D2D_CSV=<path>.
void D2dSyncConsumer::dump_csv() const {
    const char* path = std::getenv("TT_METAL_STREAMING_PROFILER_D2D_CSV");
    if (path == nullptr || *path == 0) {
        return;
    }
    // Each chip's placement, one row per run start and per ms: its eth wall tick, the root refclk, host TSC and
    // steady_clock ns it lands on, and the bound.
    if (std::FILE* f = std::fopen(path, "w"); f != nullptr) {
        std::fprintf(f, "chip,wall_tick,root_refclk,tsc,steady_ns,error_ns\n");
        for (const auto& kv : local_) {
            const uint32_t dev = kv.first;
            const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
            for (const LocalClockFit::Run& b : kv.second.fit.runs) {
                if (b.n < 2 || b.slope() <= 0.0) {
                    continue;
                }
                for (double r = b.r_first; r <= b.r_last; r += 50000.0) {
                    const int64_t T = static_cast<int64_t>(b.wall_of_refclk(r));
                    const int64_t tsc = SyncCorrections::lookup_tsc(chip, T);
                    std::fprintf(
                        f,
                        "%u,%lld,%.3f,%lld,%lld,%lld\n",
                        chip,
                        static_cast<long long>(T),
                        SyncCorrections::lookup_root(chip, T),
                        static_cast<long long>(tsc),
                        static_cast<long long>(SyncCorrections::tsc_to_mono_ns(tsc)),
                        static_cast<long long>(SyncCorrections::lookup_error_ns(chip, T)));
                }
            }
        }
        std::fclose(f);
    } else {
        log_warning(tt::LogMetal, "[streaming profiler] d2d sync: cannot open CSV {}", path);
        return;
    }
    if (std::FILE* nf = std::fopen((std::string(path) + ".nodes.csv").c_str(), "w"); nf != nullptr) {
        std::fprintf(nf, "chip,wall_tick,refclk,root_refclk,sigma_ns\n");
        for (const auto& [dev, pub] : published_) {
            const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
            for (const Node& nd : pub.linked.nodes) {
                std::fprintf(nf, "%u,%.0f,%.0f,%.3f,%.2f\n", chip, nd.H, nd.r, nd.root, nd.sigma);
            }
        }
        std::fclose(nf);
    }
    if (std::FILE* hf = std::fopen((std::string(path) + ".host.csv").c_str(), "w"); hf != nullptr) {
        std::fprintf(hf, "root_refclk,tsc,tangent,sigma_ns\n");
        for (const HostNode& nd : SyncCorrections::host_nodes()) {
            std::fprintf(hf, "%.3f,%.0f,%.9f,%.2f\n", nd.at, nd.value, nd.tangent, nd.sigma_ns);
        }
        std::fclose(hf);
    }
    // The runs themselves, one row each: where the local map bends and by how much.
    if (std::FILE* rf = std::fopen((std::string(path) + ".runs.csv").c_str(), "w"); rf != nullptr) {
        std::fprintf(rf, "chip,wall_first,wall_last,n,slope,ratio\n");
        for (const auto& kv : local_) {
            const uint32_t dev = kv.first;
            const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
            for (const LocalClockFit::Run& b : kv.second.fit.runs) {
                if (b.n < 2 || b.slope() <= 0.0) {
                    continue;
                }
                std::fprintf(
                    rf,
                    "%u,%.0f,%.0f,%llu,%.9f,%.4f\n",
                    chip,
                    b.wall_of_refclk(b.r_first),
                    b.wall_of_refclk(b.r_last),
                    static_cast<unsigned long long>(b.n),
                    b.fitted_slope(),
                    b.ratio());
            }
        }
        std::fclose(rf);
    }
    if (std::FILE* sf = std::fopen((std::string(path) + ".samples.csv").c_str(), "w"); sf != nullptr) {
        std::fprintf(sf, "chip,refclk,wall\n");
        for (const auto& kv : local_) {
            const uint32_t dev = kv.first;
            const uint32_t chip = dev < ctx_.devices.size() ? ctx_.devices[dev].chip_id : dev;
            for (const auto& [r, w] : kv.second.samples) {
                std::fprintf(sf, "%u,%llu,%llu\n", chip, static_cast<unsigned long long>(r), static_cast<unsigned long long>(w));
            }
        }
        std::fclose(sf);
    }
}

// The cross-chip refclk SCALE (chip b's refclk rate over chip a's -- the crystal ratio) as a RUNNING linear
// regression over the link rounds, for the Tracy sink: each point is the regression over every round up to its time,
// so the curve shows the estimate converging. Each round's wall stamps are converted to refclk through the chip's constant-rate
// LocalClockFit run (the 3 us tracker), NOT the solver's single linear ratio per end: over a long baseline DVFS
// moves that ratio by percent and the solver's conversion error leaks relative DVFS into the rate (measured: +/-1000
// ppm over 4 s), while the constant-rate runs track it and stay at the ~ppm crystal ratio. Same rounds and the solver's
// shortest-25%-round-trip keep rule, regressing (receiver refclk - sender midpoint refclk) on the sender midpoint.
void D2dSyncConsumer::publish_rate_plots() {
    for (size_t li = 0; li < ctx_.links.size(); li++) {
        const CaptureContext::Link& L = ctx_.links[li];
        if (L.dev_a >= ctx_.devices.size()) {
            continue;
        }
        const std::vector<Round>& prim = links_[li].primary().rounds;
        const auto la = local_.find(L.dev_a);
        const auto lb = local_.find(L.dev_b);
        if (la == local_.end() || lb == local_.end()) {
            continue;
        }
        if (SyncCorrections::published(L.chip_a) == 0) {
            continue;
        }
        const size_t n = prim.size();
        if (n < 16) {
            continue;
        }
        const LocalClockFit& fa = la->second.fit;
        const LocalClockFit& fb = lb->second.fit;
        if (fa.runs.empty() || fb.runs.empty()) {
            continue;
        }
        const auto refclk_at = [](const LocalClockFit& fit, double w) {
            return fit.run_at_wall(w).refclk_of_wall(w);
        };
        struct Rnd {
            uint64_t t0, t2, t1, t1b;
        };
        std::vector<Rnd> rounds(n);
        for (size_t i = 0; i < n; i++) {
            rounds[i] = Rnd{prim[i].t0.wall, prim[i].t2.wall, prim[i].t1.wall, prim[i].t1b.wall};
        }
        const std::string name = fmt::format("d2d refclk scale chip{}/chip{}", L.chip_b, L.chip_a);
        std::vector<SyncPlotPoint> pts;
        const size_t stride = std::max<size_t>(8, n / 250);
        for (size_t m = std::max<size_t>(16, stride); m <= n; m += stride) {
            // This point's instant: the sender's eth wall at the last round's end, placed as its records are.
            const int64_t H = SyncCorrections::lookup_tsc(L.chip_a, static_cast<int64_t>(rounds[m - 1].t2));
            struct RT {
                double off, mid;
                uint64_t rtt;
            };
            std::vector<RT> rts;
            rts.reserve(m);
            for (size_t i = 0; i < m; i++) {
                const Rnd& r = rounds[i];
                if (r.t0 == 0 || r.t1 == 0 || r.t1b == 0 || r.t2 == 0 || r.t2 < r.t0) {
                    continue;
                }
                const double r0 = refclk_at(fa, static_cast<double>(r.t0));
                const double r2 = refclk_at(fa, static_cast<double>(r.t2));
                const double r1 =
                    0.5 * (refclk_at(fb, static_cast<double>(r.t1)) + refclk_at(fb, static_cast<double>(r.t1b)));
                const double mid = 0.5 * (r0 + r2);
                rts.push_back(RT{r1 - mid, mid, r.t2 - r.t0});
            }
            if (rts.size() < 4) {
                continue;
            }
            std::sort(rts.begin(), rts.end(), [](const RT& a, const RT& b) { return a.rtt < b.rtt; });
            const size_t keep =
                std::clamp<size_t>(static_cast<size_t>(static_cast<double>(rts.size()) * 0.25), size_t{4}, rts.size());
            rts.resize(keep);
            double mid0 = rts[0].mid;
            const double off0 = rts[0].off;
            for (const auto& r : rts) {
                mid0 = std::min(mid0, r.mid);
            }
            long double sx = 0, sy = 0, sxx = 0, sxy = 0;
            const long double nn = static_cast<long double>(keep);
            for (const auto& r : rts) {
                const long double x = r.mid - mid0, y = r.off - off0;
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            const long double den = nn * sxx - sx * sx;
            if (std::abs(static_cast<double>(den)) < 1e-9) {
                continue;
            }
            const double slope = static_cast<double>((nn * sxy - sx * sy) / den);
            pts.push_back(SyncPlotPoint{H, 1.0 + slope});
        }
        if (pts.empty()) {
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] d2d sync {}: {} points over {} rounds; final scale {:.9f} ({:+.3f} ppm)",
            name,
            pts.size(),
            n,
            pts.back().value,
            (pts.back().value - 1.0) * 1e6);
        SyncPlots::publish(name, std::move(pts));
    }
}

bool D2dSyncConsumer::round_error(
    const CaptureContext::Link& L, const Round& r, bool hw, int64_t& tsc_a, double& err, RoundTerms* terms) const {
    if (L.dev_a >= ctx_.devices.size() || L.dev_b >= ctx_.devices.size()) {
        return false;
    }
    const auto la = local_.find(L.dev_a);
    const auto lb = local_.find(L.dev_b);
    if (la == local_.cend() || lb == local_.cend()) {
        return false;
    }
    if (SyncCorrections::published(L.chip_a) == 0 || SyncCorrections::published(L.chip_b) == 0) {
        return false;
    }
    double wa = 0.0, wb = 0.0;
    if (hw) {
        wa = la->second.fit.wall_at(mid_a(r, true));
        wb = lb->second.fit.wall_at(mid_b(r, true));
    } else if (r.t0.wall != 0 && r.t2.wall != 0 && r.t1.wall != 0 && r.t1b.wall != 0 && r.t2.wall >= r.t0.wall) {
        wa = 0.5 * (static_cast<double>(r.t0.wall) + static_cast<double>(r.t2.wall));
        wb = 0.5 * (static_cast<double>(r.t1.wall) + static_cast<double>(r.t1b.wall));
    }
    if (wa <= 0.0 || wb <= 0.0) {
        return false;
    }
    RoundTerms t;
    t.wall_a = wa;
    t.wall_b = wb;
    t.root_a = SyncCorrections::lookup_root(L.chip_a, std::llround(wa));
    t.root_b = SyncCorrections::lookup_root(L.chip_b, std::llround(wb));
    tsc_a = SyncCorrections::lookup_tsc(L.chip_a, std::llround(wa));
    err = (t.root_b - t.root_a) * (1e9 / LocalClockFit::kRefclkHz);
    if (terms != nullptr) {
        *terms = t;
    }
    return true;
}

// Per link and round: the receiver's stamp and the sender's round midpoint (one instant, under the symmetric path)
// placed on the host timeline exactly as the sink places a record from each chip's eth core -- against the FINAL
// map here, and, in a second series, against the map as it stood when the round arrived (what a sink converting
// on arrival applied). The worst rounds of the final series are logged with their distance to the nearest
// correction node, since the map's residual lives at the transitions.
void D2dSyncConsumer::publish_error_plots() const {
    for (size_t li = 0; li < ctx_.links.size(); li++) {
      const CaptureContext::Link& L = ctx_.links[li];
      const LinkStreams& ls = links_[li];
      for (const bool hw : {true, false}) {
          const std::vector<Round>& rounds = (hw ? ls.hw : ls.sw).rounds;
          if (rounds.empty()) {
              continue;
          }
          const bool is_primary = hw == ls.have_hw();
          const char* tag = hw ? " [hw stamps]" : " [sw stamps]";
          const size_t n = rounds.size();
          // Software rounds carry the ERISC's jitter (~15-20 ns rms over all rounds); as the solver does, the check
          // keeps only the fastest quartile by round trip, where the stamps are good to ~2 ns.
          uint64_t rtt_cut = std::numeric_limits<uint64_t>::max();
          if (!hw) {
              std::vector<uint64_t> rtts;
              rtts.reserve(n);
              for (const Round& r : rounds) {
                  if (r.t2.wall >= r.t0.wall) {
                      rtts.push_back(r.t2.wall - r.t0.wall);
                  }
              }
              if (rtts.size() >= 8) {
                  std::nth_element(rtts.begin(), rtts.begin() + rtts.size() / 4, rtts.end());
                  rtt_cut = rtts[rtts.size() / 4];
              }
          }
          // Per round, next to the placement error: the stamps' own residual against a line through the neighbouring
          // rounds' raw offsets (the ruler's noise -- a stamp glitch shows here, a map error does not; the link rate
          // wanders ~0.1 ppm over a run, so a single run-wide line would not do) and the sender's round trip (a glitch
          // on either sender stamp shows here, one on the receiver does not).
          std::vector<SyncPlotPoint> pts;
          std::vector<double> resid, rtt, raw_x, raw_y, turn, path;
          const double path_med = hw ? path_median(rounds, 0, n) : std::numeric_limits<double>::quiet_NaN();
          size_t off_path = 0;
          std::vector<RoundTerms> terms;
          pts.reserve(n);
          rtt.reserve(n);
          terms.reserve(n);
          long double se = 0, ss = 0;
          const double ghz_a = std::max(ctx_.devices[L.dev_a].clock.frequency_ghz, 0.1);
          for (const Round& r : rounds) {
              if (!hw && r.t2.wall - r.t0.wall > rtt_cut) {
                  continue;
              }
              if (hw && std::abs(path_ns(r) - path_med) > kPathDevNs) {
                  off_path++;
                  continue;
              }
              int64_t H = 0;
              double e = 0.0;
              RoundTerms t;
              if (!round_error(L, r, hw, H, e, &t)) {
                  continue;
              }
              pts.push_back(SyncPlotPoint{H, e});
              terms.push_back(t);
              se += e;
              ss += static_cast<long double>(e) * e;
              if (hw) {
                  raw_x.push_back(mid_a(r, true));
                  raw_y.push_back(mid_b(r, true) - raw_x.back());
              }
              rtt.push_back(hw ? rtt_ns(r) : (static_cast<double>(r.t2.wall) - static_cast<double>(r.t0.wall)) / ghz_a);
              path.push_back(hw ? path_ns(r) : std::numeric_limits<double>::quiet_NaN());
              turn.push_back(rtt.back() - 2.0 * path.back());
          }
          if (pts.empty()) {
              continue;
          }
          resid.assign(pts.size(), 0.0);
          long double srr = 0;
          if (hw && pts.size() >= 8) {
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
                  resid[i] = static_cast<double>((raw_y[i] - a) * 20.0);
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
          if (hw) {
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
                  "[streaming profiler] d2d sync link chip {} -> chip {} [hw stamps]: one way inside the stamps {:.1f} "
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
              "[streaming profiler] d2d sync error chip {} vs chip {}{}: {} rounds, mean {:+.1f} ns, rms {:.1f} ns{}{}",
              L.chip_b,
              L.chip_a,
              tag,
              pts.size(),
              static_cast<double>(se) / nn,
              std::sqrt(static_cast<double>(ss) / nn),
              is_primary ? " (the solving stamps)" : " (checking the map solved from the other stamps)",
              hw && pts.size() >= 8
                  ? fmt::format(
                        "; the stamps' own noise (residual to the neighbouring rounds): rms {:.1f} ns",
                        std::sqrt(static_cast<double>(srr / static_cast<long double>(pts.size()))))
                  : "");
          // The five worst rounds, how far each sits from a correction node of either chip, and its two glitch
          // indicators: a map error has a small residual; a stamp glitch has a large one and often a shifted round
          // trip.
          std::vector<double> node_a_us(pts.size(), -1.0), node_b_us(pts.size(), -1.0);
          for (const auto& [dev, out] : {std::pair{L.dev_a, &node_a_us}, std::pair{L.dev_b, &node_b_us}}) {
              const auto pb = published_.find(dev);
              if (pb == published_.end()) {
                  continue;
              }
              const std::vector<Node>& nodes = pb->second.linked.nodes;
              const double ghz = std::max(ctx_.devices[dev].clock.frequency_ghz, 0.1);
              for (size_t i = 0; i < pts.size(); i++) {
                  const double wall = dev == L.dev_a ? terms[i].wall_a : terms[i].wall_b;
                  const auto up = std::lower_bound(
                      nodes.begin(), nodes.end(), wall, [](const Node& nd, double h) { return nd.H < h; });
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
                  order.begin(),
                  order.begin() + std::min<size_t>(5, order.size()),
                  order.end(),
                  [&](size_t a, size_t b) { return std::abs(pts[a].value) > std::abs(pts[b].value); });
              std::string worst;
              for (size_t k = 0; k < std::min<size_t>(5, order.size()); k++) {
                  const size_t i = order[k];
                  const SyncPlotPoint& p = pts[i];
                  worst += fmt::format(
                      " {:+.0f} ns at {:.3f} s (nearest node: chip {} {:.0f} us, chip {} {:.0f} us; stamp resid "
                      "{:+.0f} "
                      "ns, {} {:+.1f} ns vs median);",
                      p.value,
                      static_cast<double>(
                          SyncCorrections::tsc_to_mono_ns(p.tsc) - SyncCorrections::tsc_to_mono_ns(pts.front().tsc)) /
                          1e9,
                      L.chip_a,
                      node_a_us[i],
                      L.chip_b,
                      node_b_us[i],
                      resid[i],
                      hw && !std::isnan(path[i]) ? "path" : "rtt",
                      hw && !std::isnan(path[i]) ? path[i] - path_med : rtt[i] - rtt_median);
              }
              log_info(
                  tt::LogMetal,
                  "[streaming profiler] d2d sync error chip {} vs chip {}{}: worst rounds{}",
                  L.chip_b,
                  L.chip_a,
                  tag,
                  worst);
          }
          if (const char* csv = std::getenv("TT_METAL_STREAMING_PROFILER_D2D_CSV"); csv != nullptr && *csv != 0) {
              if (std::FILE* ef = std::fopen(
                      fmt::format("{}.err_{}_{}{}.csv", csv, L.chip_b, L.chip_a, hw ? "_hw" : "_sw").c_str(), "w");
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
                          static_cast<long long>(SyncCorrections::tsc_to_mono_ns(pts[i].tsc)),
                          pts[i].value,
                          resid[i],
                          rtt[i] - rtt_median,
                          node_a_us[i],
                          node_b_us[i],
                          hw ? raw_x[i] : 0.0,
                          hw ? raw_x[i] + raw_y[i] : 0.0,
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
          SyncPlots::publish(
              fmt::format("d2d sync error chip{} vs chip{} (ns){}", L.chip_b, L.chip_a, tag), std::move(pts));
          if (!is_primary) {
              continue;
          }
          // The live series: what a sink converting on arrival applied. Rounds before the chip's first linked publish
          // carry the whole uncorrected anchor difference; they are counted apart so the rest can be read.
          if (li < live_err_.size() && !live_err_[li].empty()) {
              const std::vector<SyncPlotPoint>& lv = live_err_[li];
              size_t big = 0;
              long double ls = 0, lss = 0;
              double lmax = 0.0;
              for (const SyncPlotPoint& p : lv) {
                  if (std::abs(p.value) > 1e6) {
                      big++;
                      continue;
                  }
                  ls += p.value;
                  lss += static_cast<long double>(p.value) * p.value;
                  lmax = std::max(lmax, std::abs(p.value));
              }
              const size_t nl = lv.size() - big;
              log_info(
                  tt::LogMetal,
                  "[streaming profiler] d2d sync error chip {} vs chip {} AS DELIVERED (corrections as the consumers' "
                  "batches were released): {} rounds beyond 1 ms (before the first linked publish), the other {}: mean "
                  "{:+.1f} ns, rms {:.1f} ns, worst {:.0f} ns",
                  L.chip_b,
                  L.chip_a,
                  big,
                  nl,
                  nl ? static_cast<double>(ls) / static_cast<double>(nl) : 0.0,
                  nl ? std::sqrt(static_cast<double>(lss) / static_cast<double>(nl)) : 0.0,
                  lmax);
              SyncPlots::publish(
                  fmt::format("d2d sync error (as delivered) chip{} vs chip{} (ns)", L.chip_b, L.chip_a), lv);
          }
      }
      }
}

void D2dSyncConsumer::on_capture_end(const CaptureContext& ctx) {
    (void)ctx;
    try_solve_links(/*final=*/true);
    publish_all();
    publish_rate_plots();
    // Rounds still ahead of the watermark at capture end are what the final flush releases: evaluate them against the
    // final map so the live series is complete.
    for (size_t li = 0; li < ctx_.links.size() && li < live_done_.size(); li++) {
        const std::vector<Round>& rounds = links_[li].primary().rounds;
        const bool hw = links_[li].have_hw();
        for (; live_done_[li] < rounds.size(); live_done_[li]++) {
            int64_t H = 0;
            double e = 0.0;
            if (round_error(ctx_.links[li], rounds[live_done_[li]], hw, H, e)) {
                live_err_[li].push_back(SyncPlotPoint{H, e});
            }
        }
    }
    publish_error_plots();
    SyncPlots::complete();
    log_summary();
    dump_csv();
    // The published corrections stay for the sinks that write at process end; the next attach starts fresh.
    local_.clear();
    links_.clear();
    dropped_kind_ = 0;
}

}  // namespace tt::tt_metal::streaming_profiler
