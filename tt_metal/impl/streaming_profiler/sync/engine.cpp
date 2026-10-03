// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <ranges>
#include <span>
#include <utility>

#include <tt_stl/assert.hpp>

#include <fmt/format.h>
#include <client/TracyProfiler.hpp>

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/check.hpp"

namespace tt::tt_metal::streaming_profiler {

void ErrorStats::add(double ns, double root, double weight) {
    weight_sum += weight;
    sum += weight * ns;
    if (std::abs(ns) > worst) {
        worst = std::abs(ns);
        worst_root = root;
    }
}

// Every link weighs the same: its error is the fixed asymmetry each link training draws, which stamp precision doesn't
// reflect.
std::vector<std::optional<RootTransform>> compose_on_root(
    const CaptureContext& ctx, std::span<const LineFit* const> lines) {
    const size_t devices = ctx.devices.size();
    std::vector<std::optional<RootTransform>> to_root(devices);
    to_root[CaptureContext::kRootDevice] = RootTransform{.scale = 1.0, .shift = 0.0};
    const std::vector<bool> reached =
        reached_from_root(ctx.links, devices, [&](size_t link_index) { return lines[link_index] != nullptr; });
    std::vector<std::optional<size_t>> unknown_of(devices);
    size_t unknowns = 0;
    for (size_t dev = 1; dev < devices; dev++) {
        if (reached[dev]) {
            unknown_of[dev] = unknowns++;
        }
    }
    struct Edge {
        std::optional<size_t> sender, receiver;
        const LineFit* line;
    };
    std::vector<Edge> edges;
    for (size_t link_index = 0; link_index < lines.size(); link_index++) {
        const CaptureContext::Link& link = ctx.links[link_index];
        const LineFit* line = lines[link_index];
        if (line != nullptr && reached[link.dev_a] && reached[link.dev_b]) {
            edges.push_back({.sender = unknown_of[link.dev_a], .receiver = unknown_of[link.dev_b], .line = line});
        }
    }
    // scale_snd = scale_rcv * (1 + slope)
    const auto log_scale_ratio = [](const Edge& edge) { return std::log1p(edge.line->slope); };
    const std::vector<double> log_scale =
        solve_potential(edges, &Edge::sender, &Edge::receiver, log_scale_ratio, unknowns).x;
    const auto scale_of = [&](std::optional<size_t> unknown) { return unknown ? std::exp(log_scale[*unknown]) : 1.0; };
    // At the link's midpoint the sender reads mid and the receiver reads mid + offset, and both are the same instant on
    // the root:
    // scale_snd * mid + shift_snd = scale_rcv * (mid + offset) + shift_rcv.
    const auto shift_difference = [&](const Edge& edge) {
        return scale_of(edge.receiver) * (edge.line->x_mean + edge.line->y_mean) -
               scale_of(edge.sender) * edge.line->x_mean;
    };
    const std::vector<double> shift =
        solve_potential(edges, &Edge::sender, &Edge::receiver, shift_difference, unknowns).x;
    for (size_t dev = 1; dev < devices; dev++) {
        if (unknown_of[dev]) {
            to_root[dev] = RootTransform{.scale = scale_of(unknown_of[dev]), .shift = shift[*unknown_of[dev]]};
        }
    }
    return to_root;
}

namespace {

struct LocalPoint {
    uint64_t refclk, wall_eighths;
    uint32_t wall_per_refclk_eighths;
};
struct LocalPoints {
    std::array<LocalPoint, kernel_profiler::kSyncLocalPoints> points{};
    uint32_t count = 0;
    auto begin() const { return points.begin(); }
    auto end() const { return points.begin() + count; }
};

LocalPoints unpack_local(const kernel_profiler::SyncLocalRecord& rec) {
    const auto rates = kernel_profiler::word_as<kernel_profiler::SyncLocalRates>(rec.rates);
    LocalPoints out{.count = kernel_profiler::word_as<kernel_profiler::SyncMeta>(rec.meta).count};
    for (uint32_t i = 0; i < out.count; i++) {
        uint64_t refclk = rec.first_refclk, wall_eighths = rec.first_wall8;
        if (i != 0) {
            const auto step = kernel_profiler::word_as<kernel_profiler::SyncLocalStep>(rec.from_first[i - 1]);
            refclk += step.refclk_from_first;
            wall_eighths += uint64_t{rates.base_wall_per_refclk_eighths} * step.refclk_from_first +
                            static_cast<uint64_t>(static_cast<int64_t>(step.wall_off));
        }
        out.points[i] = {refclk, wall_eighths, rates.wall_per_refclk_eighths[i]};
    }
    return out;
}

// Over 250 ms two chips' crystals stay on a line to about 0.4 ns, and averaging the ~25 rounds in that window brings a
// round's ~0.6 ns of stamp noise down to about 0.12 ns.
constexpr double kLinkWindowTicks = 250 * kRefclkTicksPerMs;
// Rounds waiting for the other end's record: about 41 s of them at one round per 10 ms, or 4 s under the sync check.
constexpr size_t kPendingMax = 4096;

// The first value to arrive sets the base.
int64_t latch_base(std::optional<int64_t>& base, int64_t first) {
    if (!base) {
        base = first;
    }
    return *base;
}

int64_t rebase(std::optional<int64_t>& base, int64_t value) { return value - latch_base(base, value); }

}  // namespace

SyncEngine::SyncEngine(const CaptureContext& ctx, ClockMap& map) :
    map_(map), reader_(map.reader()), ctx_(ctx), chips_(ctx.devices.size()), links_(ctx.links.size()) {
    chips_[CaptureContext::kRootDevice].refclk_base = map_.root_base();
    for (size_t dev = 0; dev < chips_.size(); dev++) {
        chips_[dev].aiclk_plot = service().plot_name(fmt::format("AICLK chip{} (GHz)", ctx.devices[dev].chip_id));
    }
    for (size_t link_index = 0; link_index < ctx.links.size(); link_index++) {
        const CaptureContext::Link& link = ctx.links[link_index];
        link_of_[{link.dev_a, link.core_a}] = link_index;
        link_of_[{link.dev_b, link.core_b}] = link_index;
    }
    if (ctx.sync_check) {
        check_ = std::make_unique<SyncCheck>(ctx, map_);
    }
}

SyncEngine::~SyncEngine() = default;

void SyncEngine::on_record(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& rec) {
    const auto meta = kernel_profiler::word_as<kernel_profiler::SyncMeta>(rec.header.meta);
    if (meta.kind == kernel_profiler::SyncKind::Link) {
        on_stamp(dev, core, rec.link);
        return;
    }
    const LocalPoints points = unpack_local(rec.local);
    Chip& chip = chips_[dev];
    if (meta.kind == kernel_profiler::SyncKind::Ruler) {
        const float weight = meta.dense ? 1.0f : kernel_profiler::kSyncRulerKeepEvery;
        const int64_t offset_eighths = kWallEighths * ctx_.devices[dev].ruler_offset;
        for (const LocalPoint& point : points) {
            check_->batch().chips[dev].readings.push_back(
                {.refclk = static_cast<double>(rebase(chip.refclk_base, static_cast<int64_t>(point.refclk))),
                 .wall_eighths = static_cast<int64_t>(point.wall_eighths) + offset_eighths,
                 .weight = weight});
        }
    } else {
        for (const LocalPoint& point : points) {
            const auto wall_eighths = static_cast<int64_t>(point.wall_eighths);
            const Instant instant{
                .refclk = rebase(chip.refclk_base, static_cast<int64_t>(point.refclk)),
                .wall_eighths = wall_eighths - kWallEighths * latch_base(chip.wall_base, whole_ticks(wall_eighths)),
                .wall_per_refclk_eighths = point.wall_per_refclk_eighths};
            if (chip.instants.empty() || (instant.refclk > chip.instants.back().refclk &&
                                          instant.wall_eighths > chip.instants.back().wall_eighths)) {
                chip.instants.push_back(instant);
            }
        }
    }
}

bool SyncEngine::on_batch_end() {
    bool moved = false;
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        moved |= publish_dev(dev);
    }
    if (check_) {
        check_->submit();
    }
    return moved;
}

void SyncEngine::on_stamp(uint32_t dev, uint32_t core, const kernel_profiler::SyncLinkRecord& record) {
    const kernel_profiler::SyncRole role = kernel_profiler::word_as<kernel_profiler::SyncMeta>(record.meta).role;
    const auto found = link_of_.find({dev, core});
    TT_FATAL(found != link_of_.end(), "streaming profiler: device {} core {} sent a link stamp for no link", dev, core);
    const size_t link_index = found->second;
    const uint32_t count = record.count;
    const CaptureContext::Link& ends = ctx_.links[link_index];
    const uint32_t refclk_dev =
        role == kernel_profiler::SyncRole::ForwardEgress || role == kernel_profiler::SyncRole::ReturnIngress
            ? ends.dev_a
            : ends.dev_b;
    const uint64_t sum_from_first_ns = record.sum_from_first_ns;
    const int64_t average_ns = static_cast<int64_t>(record.first) + static_cast<int64_t>(sum_from_first_ns / count);
    const int64_t base_ns = kNsPerRefclk * latch_base(chips_[refclk_dev].refclk_base, average_ns / kNsPerRefclk);
    Link& link = links_[link_index];
    Round& round = link.pending[record.round];
    round[role] = static_cast<double>(average_ns - base_ns) + static_cast<double>(sum_from_first_ns % count) / count;
    if (round.complete()) {
        // The ends may average different frames. The midpoints stay unbiased because the refclk relation is a line.
        const double mid_a =
            0.5 *
            (*round[kernel_profiler::SyncRole::ForwardEgress] + *round[kernel_profiler::SyncRole::ReturnIngress]) /
            kNsPerRefclk;
        const double mid_b =
            0.5 *
            (*round[kernel_profiler::SyncRole::ForwardIngress] + *round[kernel_profiler::SyncRole::ReturnEgress]) /
            kNsPerRefclk;
        const RoundPoint point{.mid = mid_a, .offset = mid_b - mid_a};
        link.pending.erase(record.round);
        if (check_ && record.round % kernel_profiler::kLinkSyncCheckSolveEvery != 0) {
            check_->batch().rounds.push_back({link_index, point});
        } else {
            link.rounds.push_back(point);
            solve(link, Window::Full);
        }
    }
    while (link.pending.size() > kPendingMax) {
        link.pending.erase(link.pending.begin());
    }
}

void SyncEngine::solve(Link& link, Window window) {
    const std::vector<RoundPoint>& rounds = link.rounds;
    const double newest = rounds.back().mid;
    if (newest <= link.solved_at) {
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
    link.solved_at = newest;
    const std::span<const RoundPoint> in_window = std::span(rounds).subspan(begin);
    link.line = fit_line(in_window, &RoundPoint::mid, &RoundPoint::offset);
    if (window == Window::Full && in_window.size() > 2) {
        double sum_squares = 0.0;
        for (const RoundPoint& round : in_window) {
            const double residual = round.offset - link.line->at(round.mid);
            sum_squares += residual * residual;
        }
        const double scatter = std::sqrt(sum_squares / static_cast<double>(in_window.size() - 2)) * kNsPerRefclk;
        link.scatter_ns.add(scatter, 1.0);
        link.centre_ns.add(scatter / std::sqrt(static_cast<double>(in_window.size())), 1.0);
    }
    link.rounds.erase(link.rounds.begin(), link.rounds.begin() + static_cast<std::ptrdiff_t>(begin));
    links_moved_ = true;
}

void SyncEngine::report_precision() const {
    double worst_link_scatter = 0.0, scatter_sum = 0.0, worst_link_centre = 0.0, worst_window_centre = 0.0;
    size_t measured = 0, worst_centre_link = 0;
    for (size_t link_index = 0; link_index < links_.size(); link_index++) {
        const Link& link = links_[link_index];
        if (link.centre_ns.weight_sum == 0.0) {
            continue;
        }
        measured++;
        const double link_scatter = link.scatter_ns.abs_quantile(0.5);
        worst_link_scatter = std::max(worst_link_scatter, link_scatter);
        scatter_sum += link_scatter;
        if (const double link_centre = link.centre_ns.abs_quantile(0.5); link_centre > worst_link_centre) {
            worst_link_centre = link_centre;
            worst_centre_link = link_index;
        }
        worst_window_centre = std::max(worst_window_centre, link.centre_ns.worst);
    }
    if (measured == 0) {
        return;
    }
    const CaptureContext::Link& ends = ctx_.links[worst_centre_link];
    log_info(
        tt::LogMetal,
        "[streaming profiler] link sync precision over {} of {} links: round scatter {:.3f} ns (links' mean "
        "{:.3f} ns), fit at the window centre {:.3f} ns (chip {} -> chip {}), worst window {:.3f} ns",
        measured,
        links_.size(),
        worst_link_scatter,
        scatter_sum / static_cast<double>(measured),
        worst_link_centre,
        ctx_.devices[ends.dev_a].chip_id,
        ctx_.devices[ends.dev_b].chip_id,
        worst_window_centre);
}

void SyncEngine::report_parallel_links() const {
    std::map<std::pair<uint32_t, uint32_t>, std::vector<size_t>> by_pair;
    for (size_t link_index = 0; link_index < links_.size(); link_index++) {
        const CaptureContext::Link& ends = ctx_.links[link_index];
        by_pair[std::minmax(ends.dev_a, ends.dev_b)].push_back(link_index);
    }
    double sum_squares = 0.0, worst = 0.0;
    size_t compared = 0;
    std::pair<uint32_t, uint32_t> worst_chips;
    for (const std::vector<size_t>& parallel : std::views::values(by_pair)) {
        const CaptureContext::Link& first = ctx_.links[parallel[0]];
        const LineFit& first_line = *links_[parallel[0]].line;
        for (size_t k = 1; k < parallel.size(); k++) {
            // A link whose sender is the other chip measures the opposite offset, on that chip's refclk.
            const LineFit& line = *links_[parallel[k]].line;
            const double offset = ctx_.links[parallel[k]].dev_a == first.dev_a
                                      ? line.at(first_line.x_mean)
                                      : -line.at(first_line.x_mean + first_line.y_mean);
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

const std::vector<std::optional<RootTransform>>& SyncEngine::transforms() {
    if (links_moved_) {
        std::vector<const LineFit*> lines(links_.size());
        std::ranges::transform(
            links_, lines.begin(), [](const Link& link) { return link.line ? &*link.line : nullptr; });
        to_root_ = compose_on_root(ctx_, lines);
        links_moved_ = false;
    }
    return to_root_;
}

// Frozen nodes never move, since records were placed against them. Shifting fresh nodes to meet them and fading the
// shift out is worse: every mismatch at a join becomes an offset carried for the whole fade.
bool SyncEngine::publish_dev(uint32_t dev) {
    Chip& chip = chips_[dev];
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
    const auto root_of = [&](const Instant& instant) { return (*to_root)(static_cast<double>(instant.refclk)); };
    for (; chip.published < instants.size(); chip.published++) {
        const Instant& instant = instants[chip.published];
        const double root = root_of(instant);
        double tangent = 0.0;
        if (instant.wall_per_refclk_eighths != 0) {
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
        if (instant.wall_per_refclk_eighths != 0) {
            aiclk_.push_back({.root = root, .value = instant.wall_per_refclk() * kernel_profiler::kEthRefclkHz * 1e-9});
        }
    }
    if (chip.published == before) {
        return false;
    }
    if (check_) {
        const Instant& last = instants[chip.published - 1];
        SyncCheck::ChipBatch& batch = check_->batch().chips[dev];
        batch.offset = root_of(last) - static_cast<double>(last.refclk);
        batch.aiclk.insert(batch.aiclk.end(), aiclk_.begin(), aiclk_.end());
    }
    plot(chip.aiclk_plot, aiclk_);
    // The newest published instant stays for the next batch's first instant to take its secant to.
    chip.instants.erase(chip.instants.begin(), chip.instants.begin() + static_cast<std::ptrdiff_t>(chip.published - 1));
    chip.published = 1;
    return true;
}

void SyncEngine::plot([[maybe_unused]] const char* name, [[maybe_unused]] std::span<const PlotPoint> series) {
#if defined(TRACY_ENABLE)
    for (const PlotPoint& point : series) {
        if (const std::optional<int64_t> tsc = map_.place_tsc(reader_, point.root)) {
            tracy::Profiler::PlotDataAt(name, point.value, *tsc);
        }
    }
#endif
}

void SyncEngine::on_capture_end() {
    for (size_t link_index = 0; link_index < links_.size(); link_index++) {
        Link& link = links_[link_index];
        if (!link.line && link.rounds.size() >= 2) {
            solve(link, Window::Partial);
        }
        const CaptureContext::Link& ends = ctx_.links[link_index];
        TT_FATAL(
            link.line,
            "streaming profiler: link chip {} eth({},{}) -> chip {} eth({},{}) not solved: {} complete "
            "rounds and {} waiting for their other end",
            ctx_.devices[ends.dev_a].chip_id,
            ends.eth_a.x,
            ends.eth_a.y,
            ctx_.devices[ends.dev_b].chip_id,
            ends.eth_b.x,
            ends.eth_b.y,
            link.rounds.size(),
            link.pending.size());
    }
    report_precision();
    report_parallel_links();
    for (uint32_t dev = 0; dev < chips_.size(); dev++) {
        publish_dev(dev);
    }
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
