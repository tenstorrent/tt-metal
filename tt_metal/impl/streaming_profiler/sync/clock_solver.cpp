// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/clock_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <ranges>
#include <span>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include <fmt/format.h>
#include <client/TracyProfiler.hpp>

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/check.hpp"

namespace tt::tt_metal::streaming_profiler {

// Every link gets the same weight, because a link's error is dominated by the fixed difference between its two
// directions' delays, set each time the link trains, which its stamp scatter doesn't show.
std::vector<std::optional<RootTransform>> compose_on_root(
    const CaptureContext& ctx, std::span<const LineFit* const> lines) {
    const size_t devices = ctx.devices.size();
    std::vector<std::optional<RootTransform>> to_root(devices);
    to_root[CaptureContext::kRootDevice] = RootTransform{.scale = 1.0, .shift = 0.0};
    const std::vector<bool> reached =
        reached_from_root(ctx.links, devices, [&](size_t link_index) { return lines[link_index] != nullptr; });
    std::vector<std::optional<size_t>> unknown_of(devices);
    size_t unknowns = 0;
    for (size_t dev = 0; dev < devices; dev++) {
        if (dev != CaptureContext::kRootDevice && reached[dev]) {
            unknown_of[dev] = unknowns++;
        }
    }
    struct Edge {
        std::optional<size_t> transmitter, receiver;
        const LineFit* line;
    };
    std::vector<Edge> edges;
    for (size_t link_index = 0; link_index < lines.size(); link_index++) {
        const CaptureContext::Link& link = ctx.links[link_index];
        const LineFit* line = lines[link_index];
        if (line != nullptr && reached[link.dev_a]) {
            edges.push_back({.transmitter = unknown_of[link.dev_a], .receiver = unknown_of[link.dev_b], .line = line});
        }
    }
    // scale_transmitter = scale_receiver * (1 + slope)
    const auto log_scale_ratio = [](const Edge& edge) { return std::log1p(edge.line->slope); };
    const std::vector<double> log_scale =
        solve_potential(edges, &Edge::transmitter, &Edge::receiver, log_scale_ratio, unknowns);
    const auto scale_of = [&](std::optional<size_t> unknown) { return unknown ? std::exp(log_scale[*unknown]) : 1.0; };
    // At the centre of the link's line the transmitter reads x_mean and the receiver reads x_mean + y_mean. Both are
    // the same instant on the root, so scale_transmitter * x_mean + shift_transmitter = scale_receiver * (x_mean +
    // y_mean) + shift_receiver.
    const auto shift_difference = [&](const Edge& edge) {
        return scale_of(edge.receiver) * (edge.line->x_mean + edge.line->y_mean) -
               scale_of(edge.transmitter) * edge.line->x_mean;
    };
    const std::vector<double> shift =
        solve_potential(edges, &Edge::transmitter, &Edge::receiver, shift_difference, unknowns);
    for (size_t dev = 0; dev < devices; dev++) {
        if (unknown_of[dev]) {
            to_root[dev] = RootTransform{.scale = scale_of(unknown_of[dev]), .shift = shift[*unknown_of[dev]]};
        }
    }
    return to_root;
}

namespace {

// A link's line is fitted to its rounds in this many refclk ticks before its newest round. Over 250 ms, the offset
// between two chips' refclks stays within about 0.4 ns of a straight line, and averaging the 25 or so rounds in that
// window brings a round's 0.6 ns of stamp noise down to about 0.12 ns.
constexpr double kLinkWindowTicks = 250 * kRefclkTicksPerMs;
// At most this many rounds wait for their other port's record, which is about 41 s of rounds at one round per 10 ms, or
// 4 s at the sync check's one round per millisecond.
constexpr size_t kPendingMax = 4096;

int64_t latch_base(std::optional<int64_t>& base, int64_t first) {
    if (!base) {
        base = first;
    }
    return *base;
}

int64_t rebase(std::optional<int64_t>& base, int64_t value) { return value - latch_base(base, value); }

// Widens a chip's refclk low word `lo` to the full count nearest the chip's previous refclk `last`, stores it in `last`
// and returns it. Chips report only each refclk's low word, and the nearest value is exact because consecutive refclks
// are far less than 2^31 ticks apart.
int64_t widen_refclk(std::optional<int64_t>& last, uint32_t lo) {
    last = last ? widen(*last, lo) : lo;
    return *last;
}

constexpr size_t kWindowBursts = std::chrono::seconds(1) / kRefclkBurstPeriod;
// The spacing of the host nodes, in root refclk ticks. Each refclk burst adds one node, on a fixed grid one burst
// period apart.
constexpr double kNodeTicks = kernel_profiler::kEthRefclkHz * std::chrono::duration<double>(kRefclkBurstPeriod).count();
constexpr size_t kMinLineBursts = 3;

}  // namespace

ClockSolver::ClockSolver(const CaptureContext& ctx, const ClockBases& bases) :
    bases_(bases),
    map_(ctx.devices.size(), bases),
    reader_(map_.reader()),
    ctx_(ctx),
    chips_(ctx.devices.size()),
    links_(ctx.links.size()) {
    chips_[CaptureContext::kRootDevice].refclk_base = bases_.root_refclk;
    chips_[CaptureContext::kRootDevice].last_refclk = bases_.root_refclk;
    for (size_t link_index = 0; link_index < ctx.links.size(); link_index++) {
        const CaptureContext::Link& link = ctx.links[link_index];
        link_of_[{link.dev_a, link.core_a}] = link_index;
        link_of_[{link.dev_b, link.core_b}] = link_index;
    }
    aiclk_mean_plot_ = service().plot_name("AICLK mean (GHz)");
    if (ctx.sync_check) {
        for (size_t dev = 0; dev < chips_.size(); dev++) {
            chips_[dev].aiclk_plot = service().plot_name(fmt::format("AICLK chip{} (GHz)", ctx.devices[dev].chip_id));
        }
        check_ = std::make_unique<SyncCheck>(ctx, map_);
    }
}

ClockSolver::~ClockSolver() = default;

bool ClockSolver::on_refclk_burst(const RefclkBurst& burst) {
    std::array<int64_t, kRefclkBurstReads> rtts;
    std::ranges::transform(burst, rtts.begin(), &RefclkRead::rtt);
    std::ranges::nth_element(rtts, rtts.begin() + rtts.size() / 2);
    const int64_t median_rtt = rtts[rtts.size() / 2];
    const RefclkRead& origin = burst.front();
    double sum_tsc = 0.0, sum_refclk = 0.0;
    uint32_t kept = 0;
    for (const RefclkRead& read : burst) {
        if (read.rtt <= median_rtt) {
            sum_tsc += static_cast<double>(read.mid - origin.mid);
            sum_refclk += static_cast<double>(read.refclk - origin.refclk);
            kept++;
        }
    }
    // A read returns the count of the register's last update, on average half an update before the read.
    burst_points_.push_back(BurstPoint{
        .tsc = static_cast<double>(origin.mid - bases_.tsc) + sum_tsc / kept,
        .refclk = static_cast<double>(static_cast<int64_t>(origin.refclk) - bases_.root_refclk) + sum_refclk / kept +
                  kernel_profiler::kRefclkTicksPerUpdate / 2.0});
    if (burst_points_.size() > kWindowBursts) {
        burst_points_.pop_front();
    }
    if (burst_points_.size() < kMinLineBursts) {
        return false;
    }
    const LineFit line = fit_line(burst_points_, &BurstPoint::refclk, &BurstPoint::tsc);
    // Each node starts where the previous node's tangent reaches at this node, so host times never jump backwards at a
    // node. Its tangent aims at the line just fitted, one grid step later.
    const double grid_at = std::ceil(burst_points_.back().refclk / kNodeTicks) * kNodeTicks;
    const double node_at = host_node_ ? std::max(grid_at, host_node_->at + kNodeTicks) : grid_at;
    const double value =
        host_node_ ? host_node_->value + host_node_->tangent * (node_at - host_node_->at) : line.at(node_at);
    host_node_ =
        HostNode{.at = node_at, .value = value, .tangent = (line.at(node_at + kNodeTicks) - value) / kNodeTicks};
    map_.append_host(*host_node_, node_at + kNodeTicks);
    return true;
}

void ClockSolver::on_record(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& record) {
    const kernel_profiler::SyncMeta meta = record.header.meta;
    if (meta.kind == kernel_profiler::SyncKind::Link) {
        on_stamp(dev, core, record.link);
        return;
    }
    const kernel_profiler::SyncWallClockRecord& clock = record.wall_clock;
    Chip& chip = chips_[dev];
    const auto first_wall_eighths =
        static_cast<int64_t>(kernel_profiler::join_words(clock.first_wall_eighths_hi, clock.points[0].wall_eighths_lo));
    for (uint32_t i = 0; i < meta.count; i++) {
        const int64_t refclk = rebase(chip.refclk_base, widen_refclk(chip.last_refclk, clock.points[i].refclk_lo));
        const int64_t wall_eighths = widen(first_wall_eighths, clock.points[i].wall_eighths_lo);
        if (meta.kind == kernel_profiler::SyncKind::Check) {
            check_->batch().chips[dev].readings.push_back(
                {.refclk = static_cast<double>(refclk),
                 .wall_eighths = wall_eighths + kWallEighths * ctx_.devices[dev].check_offset,
                 .weight = meta.dense ? 1.0f : kernel_profiler::kSyncCheckKeepEvery});
        } else {
            const Instant instant{
                .refclk = refclk,
                .wall_eighths = wall_eighths - kWallEighths * latch_base(chip.wall_base, whole_ticks(wall_eighths)),
                .wall_per_refclk_eighths = clock.wall_per_refclk_eighths[i]};
            if (chip.instants.empty() || (instant.refclk > chip.instants.back().refclk &&
                                          instant.wall_eighths > chip.instants.back().wall_eighths)) {
                chip.instants.push_back(instant);
            }
        }
    }
}

bool ClockSolver::on_batch_end() {
    bool moved = false;
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        moved |= publish_dev(dev);
    }
    plot_aiclk_mean(std::ranges::min(chips_ | std::views::transform(&Chip::aiclk_through)));
    if (check_) {
        check_->submit();
    }
    return moved;
}

void ClockSolver::on_stamp(uint32_t dev, uint32_t core, const kernel_profiler::SyncLinkRecord& record) {
    using enum kernel_profiler::SyncRole;
    const kernel_profiler::SyncRole role = record.meta.role;
    const auto found = link_of_.find({dev, core});
    TT_FATAL(found != link_of_.end(), "streaming profiler: device {} core {} sent a link stamp for no link", dev, core);
    const size_t link_index = found->second;
    const CaptureContext::Link& link = ctx_.links[link_index];
    const uint32_t refclk_dev = role == ForwardEgress || role == ReturnIngress ? link.dev_a : link.dev_b;
    Chip& chip = chips_[refclk_dev];
    const int64_t first_ticks =
        rebase(chip.refclk_base, widen_refclk(chip.last_refclk, static_cast<uint32_t>(record.first_ns / kNsPerRefclk)));
    LinkState& state = links_[link_index];
    Round& round = state.pending[record.round];
    round[role] =
        static_cast<double>(first_ticks * kNsPerRefclk + static_cast<int64_t>(record.first_ns % kNsPerRefclk)) +
        static_cast<double>(record.sum_from_first_ns) / static_cast<double>(record.count);
    if (round.complete()) {
        // The two ports may average different frames. The midpoints stay unbiased because the refclks are linearly
        // related.
        const double mid_a = 0.5 * (*round[ForwardEgress] + *round[ReturnIngress]) / kNsPerRefclk;
        const double mid_b = 0.5 * (*round[ForwardIngress] + *round[ReturnEgress]) / kNsPerRefclk;
        const RoundPoint point{.mid = mid_a, .offset = mid_b - mid_a};
        state.pending.erase(record.round);
        if (check_ && record.round % kernel_profiler::kLinkSyncCheckSolveEvery != 0) {
            check_->batch().rounds.push_back({link_index, point});
        } else {
            state.rounds.push_back(point);
            solve(state, Window::Full);
        }
    }
    while (state.pending.size() > kPendingMax) {
        state.pending.erase(state.pending.begin());
    }
}

void ClockSolver::solve(LinkState& state, Window window) {
    const std::vector<RoundPoint>& rounds = state.rounds;
    const double newest = rounds.back().mid;
    if (newest <= state.solved_at) {
        return;
    }
    const double start = newest - kLinkWindowTicks;
    size_t begin = rounds.size() - 1;
    while (begin > 0 && rounds[begin].mid > start) {
        begin--;
    }
    if (window == Window::Full && rounds[begin].mid > start) {
        return;
    }
    state.solved_at = newest;
    const std::span<const RoundPoint> in_window = std::span(rounds).subspan(begin);
    state.line = fit_line(in_window, &RoundPoint::mid, &RoundPoint::offset);
    if (window == Window::Full && in_window.size() > 2) {
        double sum_squares = 0.0;
        for (const RoundPoint& round : in_window) {
            const double residual = round.offset - state.line->at(round.mid);
            sum_squares += residual * residual;
        }
        const double scatter = std::sqrt(sum_squares / static_cast<double>(in_window.size() - 2)) * kNsPerRefclk;
        state.scatter_ns.add(scatter, 1.0);
        state.worst_centre_ns =
            std::max(state.worst_centre_ns, scatter / std::sqrt(static_cast<double>(in_window.size())));
    }
    state.rounds.erase(state.rounds.begin(), state.rounds.begin() + static_cast<std::ptrdiff_t>(begin));
    links_moved_ = true;
}

void ClockSolver::report_precision() const {
    double worst_scatter = 0.0, worst_centre = 0.0;
    size_t measured = 0;
    for (const LinkState& state : links_) {
        if (state.scatter_ns.weight_sum == 0.0) {
            continue;
        }
        measured++;
        worst_scatter = std::max(worst_scatter, state.scatter_ns.abs_quantile(0.5));
        worst_centre = std::max(worst_centre, state.worst_centre_ns);
    }
    if (measured == 0) {
        return;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] link sync precision over {} of {} links: round scatter {:.3f} ns, worst window fit "
        "error {:.3f} ns",
        measured,
        links_.size(),
        worst_scatter,
        worst_centre);
}

void ClockSolver::report_parallel_links() const {
    std::map<std::pair<uint32_t, uint32_t>, std::vector<size_t>> by_pair;
    for (size_t link_index = 0; link_index < links_.size(); link_index++) {
        const CaptureContext::Link& link = ctx_.links[link_index];
        by_pair[{link.dev_a, link.dev_b}].push_back(link_index);
    }
    double sum_squares = 0.0, worst = 0.0;
    size_t compared = 0;
    std::pair<uint32_t, uint32_t> worst_chips;
    for (const std::vector<size_t>& parallel : std::views::values(by_pair)) {
        const CaptureContext::Link& first = ctx_.links[parallel[0]];
        const LineFit& first_line = *links_[parallel[0]].line;
        for (size_t k = 1; k < parallel.size(); k++) {
            const double offset = links_[parallel[k]].line->at(first_line.x_mean);
            const double apart_ns = std::abs(offset - first_line.y_mean) * kNsPerRefclk;
            sum_squares += apart_ns * apart_ns;
            compared++;
            if (apart_ns > worst) {
                worst = apart_ns;
                worst_chips = {ctx_.devices[first.dev_a].chip_id, ctx_.devices[first.dev_b].chip_id};
            }
        }
    }
    if (compared == 0) {
        return;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] parallel link agreement over {} link pairs: rms {:.3f} ns, worst {:.3f} ns (chip {} - "
        "chip {})",
        compared,
        std::sqrt(sum_squares / static_cast<double>(compared)),
        worst,
        worst_chips.first,
        worst_chips.second);
}

const std::vector<std::optional<RootTransform>>& ClockSolver::transforms() {
    if (links_moved_) {
        std::vector<const LineFit*> lines(links_.size());
        std::ranges::transform(
            links_, lines.begin(), [](const LinkState& state) { return state.line ? &*state.line : nullptr; });
        to_root_ = compose_on_root(ctx_, lines);
        links_moved_ = false;
    }
    return to_root_;
}

// Appends the chip's new instants to its series in the clock map, and returns whether there were any. Nodes already
// appended never change. Each new node uses the current transform, even when that makes the map jump at the node,
// because blending from the old transform to the new one would spread the jump's error over the blend.
bool ClockSolver::publish_dev(uint32_t dev) {
    Chip& chip = chips_[dev];
    // Nothing is published before the host series has a node, because the plotted AICLK points need a host time.
    if (chip.published == chip.instants.size() || !map_.has_host_nodes()) {
        return false;
    }
    const std::optional<RootTransform>& to_root = transforms()[dev];
    if (!to_root) {
        return false;
    }
    const std::deque<Instant>& instants = chip.instants;
    const size_t before = chip.published;
    aiclk_.clear();
    aiclk_estimates_.clear();
    const auto root_of = [&](const Instant& instant) { return (*to_root)(static_cast<double>(instant.refclk)); };
    for (; chip.published < instants.size(); chip.published++) {
        const Instant& instant = instants[chip.published];
        const double root = root_of(instant);
        const bool has_rate = instant.wall_per_refclk_eighths != 0;
        double tangent = 0.0;
        if (has_rate) {
            tangent = to_root->scale / instant.wall_per_refclk();
        } else if (instants.size() > 1) {
            const Instant& neighbour = instants[chip.published == 0 ? 1 : chip.published - 1];
            tangent = (root_of(neighbour) - root) / (neighbour.wall() - instant.wall());
        } else {
            break;
        }
        const int64_t tick = instant.wall_tick();
        map_.append(
            dev,
            SyncNode{
                .at = *chip.wall_base + tick,
                .value = root + tangent * (static_cast<double>(tick) - instant.wall()),
                .tangent = tangent});
        if (has_rate) {
            aiclk_.push_back({.root = root, .value = instant.wall_per_refclk() * kernel_profiler::kEthRefclkHz * 1e-9});
            aiclk_estimates_.push_back(aiclk_.back());
        } else if (chip.published > 0) {
            size_t from = chip.published - 1;
            while (from > 0 && instant.refclk - instants[from].refclk < kAiclkSpanTicks &&
                   instants[from].wall_per_refclk_eighths == 0) {
                from--;
            }
            const Instant& start = instants[from];
            if (instant.refclk - start.refclk >= kAiclkSpanTicks) {
                aiclk_estimates_.push_back(
                    {.root = (root_of(start) + root) / 2,
                     .value = (instant.wall() - start.wall()) / static_cast<double>(instant.refclk - start.refclk) *
                              kernel_profiler::kEthRefclkHz * 1e-9});
            }
        }
    }
    if (chip.published == before) {
        return false;
    }
    if (check_) {
        const Instant& last = instants[chip.published - 1];
        SyncCheck::ChipBatch& batch = check_->batch().chips[dev];
        batch.root_minus_refclk = root_of(last) - static_cast<double>(last.refclk);
        batch.aiclk.insert(batch.aiclk.end(), aiclk_.begin(), aiclk_.end());
        plot(chip.aiclk_plot, aiclk_);
    }
    chip.aiclk_pending.insert(chip.aiclk_pending.end(), aiclk_estimates_.begin(), aiclk_estimates_.end());
    if (!aiclk_estimates_.empty()) {
        chip.aiclk_through = aiclk_estimates_.back().root;
    }
    // The newest instants are kept for the next batch, whose first instant may take its tangent and AICLK from them.
    const size_t kept = std::min(chip.published, kAiclkKeptInstants);
    chip.instants.erase(
        chip.instants.begin(), chip.instants.begin() + static_cast<std::ptrdiff_t>(chip.published - kept));
    chip.published = kept;
    return true;
}

void ClockSolver::plot([[maybe_unused]] const char* name, [[maybe_unused]] std::span<const PlotPoint> series) {
#if defined(TRACY_ENABLE)
    for (const PlotPoint& point : series) {
        if (const std::optional<int64_t> tsc = map_.place_tsc(reader_, point.root)) {
            tracy::Profiler::PlotDataAt(name, point.value, *tsc);
        }
    }
#endif
}

void ClockSolver::plot_aiclk_mean(double until) {
    std::vector<PlotPoint> mean;
    while (true) {
        Chip* next = nullptr;
        for (Chip& chip : chips_) {
            if (!chip.aiclk_pending.empty() && chip.aiclk_pending.front().root <= until &&
                (next == nullptr || chip.aiclk_pending.front().root < next->aiclk_pending.front().root)) {
                next = &chip;
            }
        }
        if (next == nullptr) {
            break;
        }
        const PlotPoint point = next->aiclk_pending.front();
        next->aiclk_pending.pop_front();
        aiclk_known_ += !next->aiclk.has_value();
        aiclk_sum_ += point.value - next->aiclk.value_or(0.0);
        next->aiclk = point.value;
        if (aiclk_known_ == chips_.size()) {
            mean.push_back({.root = point.root, .value = aiclk_sum_ / static_cast<double>(chips_.size())});
        }
    }
    plot(aiclk_mean_plot_, mean);
}

void ClockSolver::on_capture_end() {
    for (size_t link_index = 0; link_index < links_.size(); link_index++) {
        LinkState& state = links_[link_index];
        if (!state.line && state.rounds.size() >= 2) {
            solve(state, Window::Partial);
        }
        const CaptureContext::Link& link = ctx_.links[link_index];
        TT_FATAL(
            state.line,
            "streaming profiler: link chip {} eth({},{}) -> chip {} eth({},{}) not solved: {} complete "
            "rounds and {} waiting for their other port",
            ctx_.devices[link.dev_a].chip_id,
            link.eth_a.x,
            link.eth_a.y,
            ctx_.devices[link.dev_b].chip_id,
            link.eth_b.x,
            link.eth_b.y,
            state.rounds.size(),
            state.pending.size());
    }
    report_precision();
    report_parallel_links();
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        publish_dev(dev);
    }
    plot_aiclk_mean(std::numeric_limits<double>::infinity());
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        map_.finish(dev);
    }
    if (check_) {
        check_->submit();
        check_->finish();
        for (const auto& [name, points] : check_->plots()) {
            plot(service().plot_name(name), points);
        }
        check_.reset();
    }
}

}  // namespace tt::tt_metal::streaming_profiler
